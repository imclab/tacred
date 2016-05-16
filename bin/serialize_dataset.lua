#!/usr/bin/env th
local tl = require 'torchlib'
local lapp = require 'pl.lapp'
local file = require 'pl.file'
local json = require 'json'
local stringx = require 'pl.stringx'
local path = require 'pl.path'
local corenlp = require 'corenlp'
local args = lapp [[
Converts the data to torch objects
  -i, --input  (default TACRED)  Input directory
  --train      (default train)   Train file to use
  -o, --output (default dataset/sent) Output directory
  -m, --mode   (default sent)    How to tokenize the sequence
]]

if not path.exists(args.output) then
  print('making directory at '..args.output)
  path.mkdir(args.output)
end

local typecheck = json.decode(file.read(path.join(args.input, 'typecheck.json')))

local c
local to_sent = function(raw, i)
  if not c then c = corenlp.Client('http://localhost:9001') end
  local sequence = {}
  local pos_sequence = {}
  local ner_sequence = {}
  local words, subj, obj = raw.word[i], raw.subj[i], raw.obj[i]
  local sent = stringx.replace(stringx.join(' ', words), '\\', '/')
  local parse = c:parse(sent, {["ssplit.isOneSentence"]=true, ["tokenize.whitespace"]=true, annotators="tokenize,ssplit,pos,ner"})
  local tokens = parse.sentences[1].tokens
  local pos, ner = {}, {}
  if #tokens ~= #words then
    print(tokens)
    print(words)
  end
  assert(#tokens == #words, 'There are '..#tokens..' tokens but '..#words..' words')

  local subj_ner, obj_ner
  local in_subj, in_obj = false
  for j = 1, #words do
    local w, s, o, p, n = words[j], subj[j], obj[j], pos[j], ner[j]
    if not in_subj and s == 'SUBJECT' then
      table.insert(sequence, '<subj>')
      table.insert(pos_sequence, '<subj>')
      table.insert(ner_sequence, '<subj>')
      in_subj = true
      subj_ner = raw.subj_ner[i][j]
    end
    if not in_obj and o == 'OBJECT' then
      table.insert(sequence, '<obj>')
      table.insert(pos_sequence, '<obj>')
      table.insert(ner_sequence, '<obj>')
      obj_ner = raw.obj_ner[i][j]
      in_obj = true
    end
    if in_subj and s ~= 'SUBJECT' then
      table.insert(sequence, '</subj>')
      table.insert(pos_sequence, '</subj>')
      table.insert(ner_sequence, '</subj>')
      in_subj = false
    end
    if in_obj and o ~= 'OBJECT' then
      table.insert(sequence, '</obj>')
      table.insert(pos_sequence, '</obj>')
      table.insert(ner_sequence, '</obj>')
      in_obj = false
    end
    table.insert(sequence, w)
    table.insert(pos_sequence, tokens[j].pos)
    if in_subj then
      table.insert(ner_sequence, subj_ner)
    elseif in_obj then
      table.insert(ner_sequence, obj_ner)
    else
      table.insert(ner_sequence, tokens[j].ner)
    end
  end
  if in_subj then
    table.insert(sequence, '</subj>')
    table.insert(pos_sequence, '</subj>')
    table.insert(ner_sequence, '</subj>')
  end
  if in_obj then
    table.insert(sequence, '</obj>')
    table.insert(pos_sequence, '</obj>')
    table.insert(ner_sequence, '</obj>')
  end
  assert(#sequence == #pos_sequence)
  assert(#sequence == #ner_sequence)
  if not (subj_ner and obj_ner) then return nil end
  return sequence, assert(typecheck[subj_ner..' '..obj_ner]), pos_sequence, ner_sequence
end

local to_sequence_map = {sent=to_sent}
local to_sequence = to_sequence_map[args.mode]
assert(to_sequence)

for _, split in ipairs{args.train, 'dev', 'test'} do
  print('loading '..split..'...')
  local raw = tl.Dataset.from_conll(path.join(args.input, split .. '.conll'))
  print('  loaded '..raw:size()..' examples')

  local fields = {X={}, Y={}, pos={}, ner={}, candidates={}}
  for i = 1, raw:size() do
    local sequence, candidates, pos, ner = to_sequence(raw, i)
    if sequence then
      table.insert(fields.X, sequence)
      table.insert(fields.Y, raw.label[i])
      table.insert(fields.pos, pos)
      table.insert(fields.ner, ner)
      table.insert(fields.candidates, candidates)
    end
  end

  torch.save(path.join(args.output, split..'.t7'), tl.Dataset(fields))
  print('  saved to '..args.output)
end
