#!/usr/bin/env th
local lapp = require 'pl.lapp'
local args = lapp [[
Trains a model
  --n_batch   (default 64)   Batch size
  --n_epoch  (default 100)    Number of epochs
  --patience (default 10)     Number of epochs
  --n_emb   (default 50)      Word embedding size
  --n_hid   (default 200)     Hidden layer size
  --lr      (default 0.001)   Learning rate
  --dropout_emb (default 0.6)     Dropout rate
  --dropout_feat (default 0.2)    Dropout rate
  --dropout_softmax (default 0.2)     Dropout rate
  --negative_weight (default 0.7) Weight for the negative class
  --sequencer (default BiSequencer)
  --rnn       (default FastLSTM)
  --p_corrupt (default 0.3)   Corruption rate
  --model   (default model/ConvMaxPool) Which model to use
  --dataset (default dataset/sent) Directory where dataset and vocab are stored
  --seed    (default 42)      Seed for RNG
  --clamp   (default 5)       Gradient clamp
  --gpu     (default 0)       Whether to use the gpu
  --save    (default saves)      Save directory
  --n_debug (default 5)
]]
local tl = require 'torchlib'
local stringx = require 'pl.stringx'
local optim = require 'optim'


print('arguments:')
print(args)

if args.save == '' then args.save = nil end
-- optional gpu
if args.gpu > -1 then
  require 'cutorch'
  require 'cunn'
  print('using GPU ' .. args.gpu)
  local id = args.gpu + 1
  local free, total = cutorch.getMemoryUsage(id)
  print('free mem '..free..' total mem '..total)
  cutorch.setDevice(id)
else
  require 'torch'
  require 'nn'
end

-- fix seed
torch.manualSeed(args.seed)
math.randomseed(args.seed)
if args.gpu then
  cutorch.manualSeed(args.seed)
end

local dataset = torch.load(path.join(args.dataset, 'dataset.t7'))
local vocab = torch.load(path.join(args.dataset, 'vocab.t7'))
args.n_vocab = vocab.word:size()
args.n_class = vocab.label:size()
args.pad_index = vocab.word:indexOf('***PAD***')

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
  for try = 1, args.n_debug do
    local i = torch.random(#X)
    local x, y, p = X[i], Y[i], P[i]
    print('example', i)
    print('sent', stringx.join(' ', vocab.word:tensorWordsAt(x)))
    print('gold', vocab.label:wordAt(y))
    print('pred', vocab.label:wordAt(p))
    print('\n')
  end
end

local logger = {
  acc = optim.Logger(path.join(args.save, 'acc.log')),
  f1 = optim.Logger(path.join(args.save, 'f1.log')),
}
for name, l in pairs(logger) do
  l.name = name
  l.showPlot = false
  l:setNames{'train', 'dev'}
end
logger.all = optim.Logger(path.join(args.save, 'all.log'))

local log = function(stats)
  logger.acc:add{stats.train.acc, stats.dev.acc}
  logger.acc:style{'-', '-'}
  logger.acc:plot()
  logger.f1:add{stats.train.micro.f1, stats.dev.micro.f1}
  logger.f1:style{'-', '-'}
  logger.f1:plot()
  logger.all:add(tl.util.tableFlatten(stats))
end

local model = require(args.model).new(args)
model:fit(dataset, {stats=compute_stats, debug=debug, log=log})

-- symlink the dataset that was used with this run
local lfs = require 'lfs'
lfs.link(path.abspath(opt.dataset), path.join(path.abspath(opt.save, 'dataset')), true)
