#!/usr/bin/env th
local lapp = require 'pl.lapp'
local args = lapp [[
Tunes a model
  --n_batch   (default 64)   Batch size
  --n_epoch  (default 100)    Number of epochs
  --patience (default 10)     Number of epochs
  --model   (default model/play) Which model to use
  --dataset (default dataset/sent/dataset.t7) Dataset to use
  --vocab   (default dataset/sent/vocab.t7)   Vocab to use
  --gpu     (default 0)       Whether to use the gpu
  --tune_save (default saves/tune)   Save directory for tuning
  --name    (default play_kbp)  Default name for tuning
  --trials  (default 10)      Number of tuning trials
  --n_debug (default 5)  How many examples to show during debugging
  --save_tuning_jobs    Whether to save tuning jobs
]]
local tl = require 'torchlib'
local stringx = require 'pl.stringx'
local optim = require 'optim'
local hypero = require 'hypero'
local pretty = require 'pl.pretty'


print('arguments:')
print(args)

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
  args.gpu = nil
  require 'torch'
  require 'nn'
end

-- random seed
args.seed = torch.random()
torch.manualSeed(args.seed)
math.randomseed(args.seed)
print('setting random seed: '..args.seed)
if args.gpu then
  cutorch.manualSeed(args.seed)
end

-- change me for different sets of tuning runs
local tune_opt = {
  name = args.name,
  dataset = "KBP final splits",
  version = 1.0,
  description = "train and validate using split",
  meta = {hostname=os.getenv("HOST")},
}

if not path.isdir(args.tune_save) then
  print('making directory '..args.tune_save)
  dir.makepath(args.tune_save)
end
torch.save(path.join(args.tune_save, tune_opt.name .. '.t7'), tune_opt)


local dataset = torch.load(args.dataset)
local vocab = torch.load(args.vocab)
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

-- sample hyperparameters
local sampler = hypero.Sampler()
for trial = 1, args.trials do
  collectgarbage()
  local params = tablex.copy(args)
  --n_emb   (default 50)      Word embedding size
  params.n_emb = 50  -- sampler:randint(300)
  --n_hid   (default 256)     Hidden layer size
  params.n_hid = sampler:randint(100, 300)
  --lr      (default 0.001)   Learning rate
  params.lr = sampler:logUniform(math.log(1e-4), math.log(1e-3))
  --dropout_emb (default 0.5)     Dropout rate
  params.dropout_emb = sampler:uniform(0.0, 0.65)
  --dropout_feat (default 0.3)    Dropout rate
  params.dropout_feat = sampler:uniform(0.0, 0.65)
  --dropout_softmax (default 0.3)     Dropout rate
  params.dropout_softmax = sampler:uniform(0.0, 0.5)
  params.dropout_rnn = sampler:uniform(0.0, 0.3)
  --sequencer (default Sequencer)
  params.sequencer = 'BiSequencer'  --sampler:categorical({1/2, 1/2}, {'Sequencer', 'BiSequencer'})
  --rnn       (default FastLSTM)
  params.rnn = sampler:categorical({1/2, 1/2}, {'LSTM', 'GRU'})
  --clamp   (default 5)       Gradient clamp
  params.clamp = 10  -- sampler:uniform(5, 10)
  --p_corrupt (default -1)   Corruption rate
  params.p_corrupt = 0.3 -- sampler:uniform(0.0, 0.3)

  local log
  if args.save_tuning_jobs then
    params.save = path.join(args.tune_save, args.name, tostring(args.seed))
    if not path.isdir(params.save) then
      print('making directory '..params.save)
      dir.makepath(params.save)
    end
    local logger = {
      acc = optim.Logger(path.join(params.save, 'acc.log')),
      f1 = optim.Logger(path.join(params.save, 'f1.log')),
    }
    for name, l in pairs(logger) do
      l.name = name
      l.showPlot = false
      l:setNames{'train', 'dev'}
    end
    logger.all = optim.Logger(path.join(params.save, 'all.log'))
    
    log = function(stats)
      logger.acc:add{stats.train.acc, stats.dev.acc}
      logger.acc:style{'-', '-'}
      logger.acc:plot()
      logger.f1:add{stats.train.micro.f1, stats.dev.micro.f1}
      logger.f1:style{'-', '-'}
      logger.f1:plot()
      logger.all:add(tl.util.tableFlatten(stats))
    end
    print('saving directory', params.save)
  else
    params.save = nil
  end

  print('starting experiment '..trial..' out of '..params.trials)
  print(pretty.write(params))

  local model = require(params.model).new(params)
  local best = model:fit(dataset, {stats=compute_stats, debug=debug, log=log})
  print(best)

  -- submit results
  local conn = hypero.connect()
  local battery = conn:battery(tune_opt.name, tune_opt.description)
  local exp = battery:experiment()
  exp:setParam(params)
  exp:setMeta(tune_opt.meta)
  exp:setResult(tl.util.tableFlatten(best))
  print('submitted experiment')
end

