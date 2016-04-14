#!/usr/bin/env th
local tl = require 'torchlib'
local lapp = require 'pl.lapp'
local file = require 'pl.file'
local json = require 'json'
local args = lapp [[
Converts the data to torch objects
  -i, --input  (default TACRED)  Input directory
  -o, --output (default dataset/sent) Output directory
  -m, --mode   (default sent)    How to tokenize the sequence
]]

if not path.exists(args.output) then
  print('making directory at '..args.output)
  path.mkdir(args.output)
end

local typecheck = json.decode(file.read(path.join(args.input, 'typecheck.json')))

local to_sent = function(raw, i)
  local sequence = {}
  local words, subj, obj = raw.word[i], raw.subj[i], raw.obj[i]
  local subj_ner, obj_ner
  local in_subj, in_obj = false
  for j = 1, #words do
    local w, s, o = words[j], subj[j], obj[j]
    if not in_subj and s == 'SUBJECT' then
      table.insert(sequence, '<subj>')
      in_subj = true
      subj_ner = raw.subj_ner[i][j]
    end
    if not in_obj and o == 'OBJECT' then
      table.insert(sequence, '<obj>')
      obj_ner = raw.obj_ner[i][j]
      in_obj = true
    end
    if in_subj and s ~= 'SUBJECT' then
      table.insert(sequence, '</subj>')
      in_subj = false
    end
    if in_obj and o ~= 'OBJECT' then
      table.insert(sequence, '</obj>')
      in_obj = false
    end
    table.insert(sequence, w)
  end
  if in_subj then table.insert(sequence, '</subj>') end
  if in_obj then table.insert(sequence, '</obj>') end
  return sequence, assert(typecheck[subj_ner..' '..obj_ner])
end

local to_sequence_map = {sent=to_sent}
local to_sequence = to_sequence_map[args.mode]
assert(to_sequence)

for _, split in ipairs{'train', 'dev', 'test'} do
  print('loading '..split..'...')
  local raw = tl.Dataset.from_conll(path.join(args.input, split .. '.conll'))
  print('  loaded '..raw:size()..' examples')

  local fields = {X={}, Y={}, candidates={}}
  for i = 1, raw:size() do
    local sequence, candidates = to_sequence(raw, i)
    table.insert(fields.X, sequence)
    table.insert(fields.Y, raw.label[i])
    table.insert(fields.candidates, candidates)
  end

  torch.save(path.join(args.output, split..'.t7'), tl.Dataset(fields))
  print('  saved to '..args.output)
end
