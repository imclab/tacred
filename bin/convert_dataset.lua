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
  --unk        (default rare)    How to encode rare words. Can be {rare, ent, pos}
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

local get_rare_unks = function(words)
  local unks = {}
  for i, w in ipairs(words) do
    table.insert(unks, '***UNK***')
  end
  return unks
end

local get_pos_unks = function(words, pos)
  if #words ~= #pos then
    for i = 1, math.max(#words, #pos) do
      print(words[i], pos[i])
    end
    error('received '..#words..' words and '..#pos..' pos')
  end
  local unks = {}
  for i, w in ipairs(words) do
    table.insert(unks, '***UNK-'..pos[i]..'***')
  end
  return unks
end

local get_ner_unks = function(words, pos, ner)
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

local get_ent_unks = function(words)
  local in_subj, in_obj = false, false
  local unks = {}
  for i, w in ipairs(words) do
    if w == '<subj>' then in_subj = true end
    if w == '<obj>' then in_obj = true end
    if w == '</subj>' then in_subj = false end
    if w == '</obj>' then in_obj = false end
    if in_subj then
      table.insert(unks, '***UNK-subj***')
    elseif in_obj then
      table.insert(unks, '***UNK-obj***')
    else
      table.insert(unks, '***UNK***')
    end
  end
  return unks
end

local convert = function(split, vocab, train)
  local fields = {X={}, Y={}, typecheck={}}
  if not train then fields.unks = {} end
  for i = 1, split:size() do
    local x = tablex.deepcopy(split.X[i])
    if args.lower then x = tl.util.map(x, string.lower) end
    if train then
      table.insert(fields.X, torch.Tensor(vocab.word:indicesOf(x, true)))
    else
      local get_unks_map = {
        rare = get_rare_unks,
        ent = get_ent_unks,
        pos = get_pos_unks,
        ner = get_ner_unks,
      }
      local get_unks = assert(get_unks_map[args.unk], 'unsupported unk mode: '..args.unk)
      local unks = get_unks(x, split.pos[i], split.ner[i])
      local words = {}
      for i, w in ipairs(x) do
        if vocab.word:contains(w) then
          table.insert(words, w)
        else
          table.insert(words, unks[i])
        end
      end
      table.insert(words, '***END***')
      table.insert(unks, '***END***')
      table.insert(fields.X, torch.Tensor(vocab.word:indicesOf(words, true)))
      table.insert(fields.unks, torch.Tensor(vocab.word:indicesOf(unks, true)))
    end
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

print('converting train: '..tostring(dataset.train))
convert(dataset.train, vocab, true)
vocab.word = vocab.word:copyAndPruneRares(args.cutoff)

for name, v in pairs(vocab) do
  stats[name..'_size'] = v:size()
end

for _, split in ipairs{'train', 'dev', 'test'} do
  print('converting '..split)
  dataset[split] = convert(dataset[split], vocab, false)
  stats[split..'_size'] = dataset[split]:size()
end

torch.save(path.join(args.output, 'vocab.t7'), vocab)
torch.save(path.join(args.output, 'dataset.t7'), dataset)

pretty.dump(stats, path.join(args.output, 'stats.json'))
