require 'rnn'
require 'model/Model'

local Model = torch.class('MaxPoolModel', 'Model')
nn.FastLSTM.usenngraph = true

function Model:get_net(opt)
  local RNN = nn[opt.rnn]
  local Sequencer = nn[opt.sequencer]
  local rnn = nn.Sequential()
                :add(RNN(opt.n_emb, opt.n_hid, opt.bptt))
                :add(nn.Unsqueeze(1))
  local net = nn.Sequential()
                :add(nn.LookupTable(opt.n_vocab, opt.n_emb))
                :add(nn.Dropout(opt.dropout_emb))
                :add(nn.SplitTable(2))
                :add(Sequencer(rnn))
                :add(nn.JoinTable(1))
                :add(nn.Max(1))
                :add(nn.Dropout(opt.dropout_softmax))
                :add(nn.Linear(opt.n_hid, opt.n_class))
  return net
end

return Model
