require 'rnn'
require 'model/Model'

local Model = torch.class('ConvMaxPoolModel', 'Model')
nn.FastLSTM.usenngraph = true

function Model:get_net(opt)
  local RNN = nn[opt.rnn]
  local Sequencer = nn[opt.sequencer]
  local rnn = nn.Sequential()
                :add(RNN(opt.n_emb, opt.n_hid, opt.bptt, opt.dropout_rnn))
                :add(nn.Unsqueeze(1))
  local net = nn.Sequential()
                :add(nn.LookupTable(opt.n_vocab, opt.n_emb))
                :add(nn.Dropout(opt.dropout_emb))
                :add(nn.Unsqueeze(4))
                :add(nn.Transpose({2, 3}))
                :add(nn.SpatialConvolution(opt.n_emb, opt.n_emb, 3, 1, 1, 1, 1, 0))
                :add(nn.ReLU())
                :add(nn.Transpose({2, 3}))
                :add(nn.Squeeze(4))
                :add(nn.SplitTable(2))
                :add(Sequencer(rnn))
                :add(nn.JoinTable(1))
                :add(nn.Max(1))
                :add(nn.Dropout(opt.dropout_softmax))
                :add(nn.Linear(opt.n_hid, opt.n_class))
  return net
end

return Model
