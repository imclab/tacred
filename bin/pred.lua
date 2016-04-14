#!/usr/bin/env th
local lapp = require 'pl.lapp'
local opt = lapp [[
Predicts using a model
  --n_batch   (default 128)   Batch size
  --save    (default saves/best) Which model to use
  --vocab (default dataset/sent/vocab.t7)   Vocab to use
  --gpu     (default 0)       Whether to use the gpu
  --typecheck  (default TACRED/typecheck.json)
]]

local tl = require 'torchlib'

--print('arguments:')
--print(opt)

-- optional gpu
if opt.gpu > -1 then
  require 'cutorch'
  require 'cunn'
  --print('using GPU ' .. opt.gpu)
  local id = opt.gpu + 1
  local free, total = cutorch.getMemoryUsage(id)
  --print('free mem '..free..' total mem '..total)
  cutorch.setDevice(id)
else
  opt.gpu = nil
  require 'torch'
  require 'nn'
end

require 'dpnn'
require 'rnn'
local json = require 'cjson'

local vocab = torch.load(opt.vocab)
local typecheck = json.decode(file.read(opt.typecheck))

local config = path.join(opt.save, 'opt.json')
local model_opt = pretty.read(file.read(config))
model_opt.gpu = opt.gpu
--print('model config')
--print(model_opt)

local model = require(model_opt.model).new(model_opt)
model.params:copy(torch.load(path.join(opt.save, 'params.t7')))
if opt.gpu then model:cuda() end

local parse_line = function(row, lower)
  if lower == nil then lower = true end
  local words, subj_id, obj_id, subj_ner, obj_ner, subj_start, subj_end, obj_start, obj_end = table.unpack(row)
  table.insert(words, '***END***')
  words = tl.util.map(words, string.lower)
  local sequence = {}
  for i, w in ipairs(words) do
    if i == subj_start then table.insert(sequence, '<subj>') end
    if i == obj_start then table.insert(sequence, '<obj>') end
    table.insert(sequence, w)
    if i == subj_end then table.insert(sequence, '</subj>') end
    if i == obj_end then table.insert(sequence, '</obj>') end
  end
  local types = typecheck[subj_ner .. ' ' .. obj_ner]
  return sequence, types, subj_id, obj_id
end

-- start reading from stdin
local X, T, I = {}, {}, {}
local words, types
local one_hot = nn.OneHot(model_opt.n_class)
local typecheck = torch.Tensor()
while true do
  -- reset
  for i = 1, #X do
    X[i] = nil
  end

  -- read line
  local line = io.read()

  -- check for ending
  if not line then break end

  -- parse data
  local data = json.decode(line)
  for i = 1, #data do
    words, types, subj_id, obj_id = parse_line(data[i])
    if types then
      -- do not process types that we didn't see in training
      table.insert(X, torch.Tensor(vocab.word:indicesOf(words, false)))
      table.insert(T, torch.Tensor(vocab.label:indicesOf(types, false)))
      table.insert(I, {subj_id, obj_id})
    end
  end

  -- predict using model
  model:set_mode('evaluate')
  local x, t = tl.Dataset.pad(X, model_opt.pad_index), T
  typecheck:resize(x:size(1), model_opt.n_class)
  for i = 1, x:size(1) do
    typecheck[i] = one_hot:forward(t[i]):sum(1)
    typecheck[i][1] = 1  -- no_relation is always a valid option
  end
  if model_opt.gpu > -1 then
    x = x:cuda()
    typecheck = typecheck:cuda()
  end
  local scores = model.net:forward(x)
  local typechecked = model.typecheck:forward{scores, typecheck}
  local max_scores, preds = model.softmax:forward(typechecked):max(2)
  max_scores = max_scores:squeeze(2)
  preds = preds:squeeze(2)

  for i = 1, #I do
    local subj_id, obj_id = table.unpack(I[i])
    local r, s = preds[i], max_scores[i]
    print(subj_id, r, obj_id, s)
  end

  collectgarbage()
end
