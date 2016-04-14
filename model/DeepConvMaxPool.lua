require 'rnn'
require 'model/Model'
require 'dpnn'

local Model = torch.class('ConvMaxPoolModel', 'Model')
nn.FastLSTM.usenngraph = true

function Model:get_net(opt)
  assert(opt.rnn)
  assert(opt.sequencer)
  assert(opt.n_hid)
  assert(opt.n_vocab)
  assert(opt.n_emb)
  assert(opt.dropout_emb)
  assert(opt.dropout_feat)
  assert(opt.dropout_softmax)
  assert(opt.n_class)
  local RNN = nn[opt.rnn]
  local Sequencer = nn[opt.sequencer]
  local rnn = nn.Sequential()
                :add(RNN(128, opt.n_hid, opt.bptt))
                :add(nn.Unsqueeze(1))
  local net = nn.Sequential()
                :add(nn.LookupTable(opt.n_vocab, opt.n_emb))
                :add(nn.Dropout(opt.dropout_emb))
                :add(nn.Unsqueeze(4))
                :add(nn.Transpose({2, 3}))
                :add(nn.SpatialConvolution(opt.n_emb, 64, 3, 1, 1, 1, 1, 0))
                :add(nn.SpatialDropout(opt.dropout_feat))
                :add(nn.SpatialConvolution(64, 128, 3, 1, 1, 1, 1, 0))
                :add(nn.SpatialDropout(opt.dropout_feat))
                :add(nn.SpatialConvolution(128, 256, 3, 1, 1, 1, 1, 0))
                :add(nn.SpatialMaxPooling(2, 1, 1, 1, 1, 0))
                :add(nn.SpatialConvolution(256, 32, 1, 1, 1, 1, 0, 0))
                :add(nn.SpatialConvolution(32, 128, 3, 1, 1, 1, 1, 0))
                :add(nn.ReLU())
                :add(nn.Transpose({2, 3}))
                :add(nn.Max(4))
                :add(nn.SplitTable(2))
                :add(Sequencer(rnn))
                :add(nn.JoinTable(1))
                :add(nn.Max(1))
                :add(nn.Dropout(opt.dropout_softmax))
                :add(nn.Linear(opt.n_hid, opt.n_class))
  return net
end

return Model
