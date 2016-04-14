#!/usr/bin/env th
local lapp = require 'pl.lapp'
local args = lapp [[
Retrieves tuning results
  <save>    (string) Save  file from the tuning run
  --out     (default '') Output csv file
  --order   (default dev__micro__f1)   How to order the results (in descending order)
  --params  (default *) What hyperparameter columns to report (comma delimited)
  --result  (default *) What result columns to report (comma delimited)
]]

require 'hypero'
require 'paths'

local opt = torch.load(args.save)
opt.out = args.out or args.save .. '.csv'

print('Options')
print(opt)

local conn = hypero.connect()
local bat = conn:battery(opt.name, opt.desciption, true, true)

local data, header = bat:exportTable{
  minVer = opt.version,
  paramNames = args.params,
  metaNames = '*',
  resultNames = args.result,
  orderBy = args.order,
  asc = true
}

local printrow = function(row)
  print(table.unpack(row))
end

if opt.out ~= '' then
  hypero.writecsv(opt.out, header, data)
else
  printrow(header)
  for i, row in ipairs(data) do
    printrow(row)
  end
end


