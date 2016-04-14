require 'rnn'
require 'model/Model'

local Model = torch.class('ConvModel', 'Model')
nn.FastLSTM.usenngraph = true

function Model:add_padding(net, dim, step)
  dim = dim or 2
  step = step or 1
  net:add(nn.Padding(dim, step))
  net:add(nn.Padding(dim, -step))
  return net
end

function Model:add_conv(net, n_in, n_out, opt)
  opt.conv = opt.conv or 3
	local pad = opt.pad or (opt.conv - 1) / 2
	assert(math.floor(pad) == pad, 'pad is not an odd number! your conv dimension probably didnt work out.')
  self:add_padding(net, 2, pad)
  net:add(nn.TemporalConvolution(n_in, n_out, opt.conv, 1))
  net:add(nn.Dropout(opt.dropout_feat))
  net:add(nn.ReLU())
	-- dims are (batch, time, feat)
  return net
end


function Model:get_net(opt)
  local RNN = nn[opt.rnn]
  local Sequencer = nn[opt.sequencer]
  local net = nn.Sequential()
                :add(nn.LookupTable(opt.n_vocab, opt.n_emb))
                :add(nn.Dropout(opt.dropout_emb))
	              -- dims are (batch, time, feat)
  self:add_conv(net, opt.n_emb, 64, {conv=3})
  self:add_conv(net, 64, 128, {conv=3})
  self:add_conv(net, 128, 256, {conv=3})
  self:add_padding(net)
  net:add(nn.TemporalMaxPooling(2, 1, 1, 1, 1, 0))
  self:add_conv(net, 256, 32, {conv=1})
  self:add_conv(net, 32, 128, {conv=3})
	-- dims are (batch, time, feat)
  local rnn = nn.Sequential()
                :add(RNN(128, opt.n_hid, opt.bptt))
                :add(nn.Unsqueeze(1))
  net:add(nn.SplitTable(2))
                :add(Sequencer(rnn))
                :add(nn.JoinTable(1))
                :add(nn.Max(1))
                :add(nn.Dropout(opt.dropout_softmax))
                :add(nn.Linear(opt.n_hid, opt.n_class))
  return net
end

return Model
