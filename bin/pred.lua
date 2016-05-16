#!/usr/bin/env th
local lapp = require 'pl.lapp'
local path = require 'pl.path'
local opt = lapp [[
Predicts using a model
  --n_batch   (default 128)   Batch size
  --save    (default saves/best) Which model to use
  --gpu     (default 0)       Whether to use the gpu
  --typecheck  (default TACRED/typecheck.json)
]]
opt.vocab = path.join(opt.save, 'dataset/vocab.t7')
local pretty = require 'pl.pretty'
local stringx = require 'pl.stringx'

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

local cluster_map
local get_cluster_unks = function(words)
  if not cluster_map then
    local map_file = 'egw4-reut.512.clusters'
    assert(path.exists(map_file), map_file..' does not exist!')
    cluster_map = {}
    for line in io.lines(map_file) do
      local tokens = stringx.split(stringx.rstrip(line, '\n'), '\t')
      local word = tokens[1]
      local index = tonumber(tokens[2])
      cluster_map[assert(word, 'could not retrieve word')] = assert(index, 'could not retrieve index')
    end
  end
  local unks = {}
  for i, word in ipairs(words) do
    local cluster = cluster_map[word] or 0
    unks[i] = '***UNK-cluster'..cluster..'***'
  end
  return unks
end

local get_ner_unks = function(words, ner)
  if #words ~= #ner then
    for i = 1, math.max(#words, #ner) do
      print(words[i], ner[i])
    end
    error('received '..#words..' words and '..#ner..' ner')
  end
  local unks = {}
  for i, w in ipairs(words) do
    table.insert(unks, '***UNK-'..ner[i]..'***')
  end
  return unks
end

local get_cluster_ner_unks = function(words, ner)
  local cluster_unks = get_cluster_unks(words)
  local ner_unks = get_ner_unks(words, ner)
  for i, cunk in ipairs(cluster_unks) do
    if cunk == '***UNK-cluster0***' then
      cluster_unks[i] = ner_unks[i]
    end
  end
  return cluster_unks
end

local parse_line = function(row, lower)
  if lower == nil then lower = true end
  local words, ners, subj_id, obj_id, subj_ner, obj_ner, subj_start, subj_end, obj_start, obj_end = table.unpack(row)
  -- 0 indexed
  subj_start = subj_start + 1
  obj_start = obj_start + 1
  local clusters = get_cluster_unks(words)
  local sequence_words = {}
  local sequence_unks = {}
  local unks = get_cluster_ner_unks(words, ners)

  local in_subj, in_obj

  for i, w in ipairs(words) do
    if i == subj_start then
      table.insert(sequence_words, '<subj>')
      table.insert(sequence_unks, '<subj>')
      in_subj = true
    end
    if i == obj_start then
      table.insert(sequence_words, '<obj>')
      table.insert(sequence_unks, '<obj>')
      in_obj = true
    end

    if in_subj then
      unks[i] = 'SUBJ-'..ners[i]
      table.insert(sequence_words, unks[i])
      table.insert(sequence_unks, unks[i])
    elseif in_obj then
      unks[i] = 'OBJ-'..ners[i]
      table.insert(sequence_words, unks[i])
      table.insert(sequence_unks, unks[i])
    else
      if vocab.word:contains(w) then
        table.insert(sequence_words, w)
        table.insert(sequence_unks, unks[i])
      else
        table.insert(sequence_words, unks[i])
        table.insert(sequence_unks, unks[i])
      end
    end

    if i == subj_end then
      table.insert(sequence_words, '</subj>')
      table.insert(sequence_unks, '</subj>')
      in_subj = false
    end
    if i == obj_end then
      table.insert(sequence_words, '</obj>')
      table.insert(sequence_unks, '</obj>')
      in_obj = false
    end
  end
  table.insert(sequence_words, '***END***')
  table.insert(sequence_unks, '***END***')
  local types = typecheck[subj_ner .. ' ' .. obj_ner]
  return sequence_words, sequence_unks, types, subj_id, obj_id
end

-- start reading from stdin
local X, T, I = {}, {}, {}
local words, types, subj_id, obj_id
local one_hot = nn.OneHot(model_opt.n_class)
local typecheck = torch.Tensor()
while true do
  -- reset
  for i = 1, #X do
    X[i] = nil
    T[i] = nil
    I[i] = nil
  end

  -- read line
  local line = io.read()

  -- check for ending
  if not line then break end

  -- parse data
  local data = json.decode(line)
  for i = 1, #data do
    words, unks, types, subj_id, obj_id = parse_line(data[i])

    -- debug:
    --io.stderr:write(stringx.join(' ', words) .. '\n')

    if types then
      -- do not process types that we didn't see in training
      for i, w in ipairs(words) do
        if not vocab.word:contains(w) then words[i] = unks[i] end
      end
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
    r = vocab.label:wordAt(r)
    if r ~= 'no_relation' then
      print(subj_id..'\t'..r..'\t'..obj_id..'\t'..s)
    end
  end

  collectgarbage()
end
