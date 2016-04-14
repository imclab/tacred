#!/usr/bin/env th
local tl = require 'torchlib'
local lapp = require 'pl.lapp'
local pretty = require 'pl.pretty'
local tablex = require 'pl.tablex'
local args = lapp [[
Converts the data to numerical torch objects
  -i, --input (default dataset/sent)  Input directory
  -o, --output (default dataset/sent) Output directory
  -l, --lower  Lowercase words
  -c, --cutoff (default 3)       Words occuring less than this number of times will be replace with UNK
  --embeddings (default random)  Which embeddings to use
]]

if not path.exists(args.output) then
  print('making directory at '..args.output)
  path.mkdir(args.output)
end

local dataset = {}
for _, split in ipairs{'train', 'dev', 'test'} do
  print('loading '..split..'...')
  dataset[split] = torch.load(path.join(args.input, split..'.t7'))
  print('  loaded '..dataset[split]:size()..' examples')
end

local convert = function(split, vocab, train)
  local fields = {X={}, Y={}, typecheck={}}
  for i = 1, split:size() do
    local x = tablex.deepcopy(split.X[i])
    table.insert(x, '***END***')
    if args.lower then x = tl.util.map(x, string.lower) end
    table.insert(fields.X, torch.Tensor(vocab.word:indicesOf(x, train)))
    table.insert(fields.Y, vocab.label:indexOf(split.Y[i], true))
    table.insert(fields.typecheck, vocab.label:indicesOf(split.candidates[i], true))
  end
  return tl.Dataset(fields)
end

local word_vocab_map = {
  random = tl.Vocab('***UNK***'),
  glove = tl.GloveVocab(),
}

local vocab = {word=assert(word_vocab_map[args.embeddings]), label=tl.Vocab('no_relation')}
local stats = {pad_index = vocab.word:add('***PAD***', 100)}

print('converting train: '..dataset.train:tostring())
convert(dataset.train, vocab, true)
vocab.word = vocab.word:copyAndPruneRares(args.cutoff)

for name, v in pairs(vocab) do
  stats[name..'_size'] = v:size()
end
torch.save(path.join(args.output, 'vocab.t7'), vocab)

for _, split in ipairs{'train', 'dev', 'test'} do
  print('converting '..split)
  dataset[split] = convert(dataset[split], vocab, false)
  stats[split..'_size'] = dataset[split]:size()
end

torch.save(path.join(args.output, 'dataset.t7'), dataset)

pretty.dump(stats, path.join(args.output, 'stats.json'))
