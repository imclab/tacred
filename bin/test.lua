#!/usr/bin/env th
local lapp = require 'pl.lapp'
local opt = lapp [[
Predicts using a model
  --n_batch   (default 128)   Batch size
  --save    (default saves/best) Which model to use
  --vocab (default dataset/sent/vocab.t7)   Vocab to use
  --dataset (default dataset/sent/dataset.t7)   Dataset to use
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
local dataset = torch.load(opt.dataset)

local config = path.join(opt.save, 'opt.json')
local model_opt = pretty.read(file.read(config))
model_opt.gpu = opt.gpu
--print('model config')
--print(model_opt)

local model = require(model_opt.model).new(model_opt)
model.params:copy(torch.load(path.join(opt.save, 'params.t7')))
if opt.gpu then model:cuda() end

local compute_stats = function(pred, targ)
  local stats = {}
  local scorer = tl.Scorer()
  for i = 1, pred:size(1) do
    scorer:add_pred(vocab.label:wordAt(targ[i]), vocab.label:wordAt(pred[i]), i)
  end
  stats.micro, stats.macro = scorer:precision_recall_f1(vocab.label.unk)
  stats.acc = pred:eq(targ):sum()/pred:numel()
  stats.cf = optim.ConfusionMatrix(vocab.label.index2word)
  stats.cf:zero()
  stats.cf:batchAdd(pred, targ)
  return stats
end

local debug = function(X, Y, P)
  for try = 1, opt.n_debug do
    local i = torch.random(#X)
    local x, y, p = X[i], Y[i], P[i]
    print('example', i)
    print('sent', stringx.join(' ', vocab.word:tensorWordsAt(x)))
    print('gold', vocab.label:wordAt(y))
    print('pred', vocab.label:wordAt(p))
    print('\n')
  end
end

local printable = function(stats)
  local p = tl.util.tableCopy(stats)
  for _, t in ipairs{'train', 'dev', 'test'} do
    if p[t] then p[t].cf = nil end
  end
  return p
end

local stats = {}
for split, d in pairs(data) do
  print('evaluation ' .. split)
  local loss, pred, targ = model:eval(d)
  stats[split] = compute_stats(pred, targ)
  stats[split].loss = loss
  print('examples')
  debug(d.X, targ, pred)
end

print('scores')
print(printable(stats))
