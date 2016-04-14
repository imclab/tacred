local optim = require 'optim'
require 'model/Typecheck'

local Model = torch.class('Model')
function Model:__init(opt)
  self.opt = opt
  self.net = self:get_net(opt)
  self.softmax = nn.SoftMax()
  self.typecheck = nn.Typecheck()
  self.criterion = self:get_criterion(opt)
  if opt.gpu then
    --print('moving model to gpu')
    self:cuda()
  end
  self.params, self.dparams = self.net:getParameters()
  self.params:uniform(-0.08, 0.08)
  self.dparams:zero()
  self.set_batch, self.ftrain, self.predict = self:get_functions()
end

function Model:cuda()
  self.net:cuda()
  self.softmax:cuda()
  self.criterion:cuda()
  self.typecheck:cuda()
end

function Model:get_criterion(opt)
  -- assume first class is no_relation
  local weights = torch.Tensor(opt.n_class):fill(1)
  weights[1] = opt.negative_weight
  return nn.CrossEntropyCriterion(weights)
end

function Model:set_mode(mode)
  if mode == 'train' then
    self.net:training()
  elseif mode == 'evaluate' then
    self.net:evaluate()
  else
    error('unsupported mode '..mode)
  end
end

function Model:get_net(opt)
  error('Not implemented')
end

function Model:get_functions()
  local x, y
  local typecheck = torch.Tensor()
  local one_hot = nn.OneHot(self.opt.n_class)
  local set_batch = function(new_x, new_y, new_typecheck)
    x, y = new_x, new_y
    -- set typecheck mask
    typecheck:resize(x:size(1), self.opt.n_class)
    for i = 1, x:size(1) do
      typecheck[i] = one_hot:forward(torch.Tensor(new_typecheck[i])):sum(1)
      typecheck[i][1] = 1
      assert(typecheck[i][y[i]] == 1, 'rel: '..y[i]..' typecheck: '..tostring(typecheck[i]))
    end
    if self.opt.gpu > -1 then
      x = x:cuda()
      y = y:cuda()
      typecheck = typecheck:cuda()
    end
  end
  local ftrain = function(new_params)
    if self.params ~= new_params then
      self.params:copy(new_params)
    end
    self.dparams:zero()

    local scores = self.net:forward(x)
    local typechecked = self.typecheck:forward{scores, typecheck}

    local loss = self.criterion:forward(typechecked, y)
    local dtypechecked = self.criterion:backward(typechecked, y)
    local dscores = self.typecheck:backward({scores, typecheck}, dtypechecked)
    self.net:backward(x, dscores)

    self.dparams:clamp(-self.opt.clamp, self.opt.clamp)
    return loss, self.dparams
  end
  local predict = function()
    local scores = self.net:forward(x)
    local typechecked = self.typecheck:forward{scores, typecheck}
    local loss = self.criterion:forward(typechecked, y)
    local max_scores, preds = self.softmax:forward(typechecked):max(2)
    return loss, preds:squeeze(2)
  end
  return set_batch, ftrain, predict
end

function Model:gradnorm()
  return self.dparams[self.dparams:ne(0)]:norm(2)
end

function Model:train(data)
  local optimize = optim.adam
  local opt = self.opt
  local optim_opt = {learningRate = opt.lr or 1e-3}
  local p_corrupt = opt.p_corrupt or 0.1
  self:set_mode('train')
  local loss = 0
  for batch, batch_end in data:batches(opt.n_batch, opt) do
    local x, y, t = tl.Dataset.pad(batch.X, opt.pad_index), torch.Tensor(batch.Y), batch.typecheck
    -- random corruption of words to UNK (index 1)
    if opt.p_corrupt ~= -1 then
      local corrupt = x:clone():bernoulli(p_corrupt):cmul(x):gt(0) -- cmul again to avoid corrupting 0 indices aka PADs
      x:maskedFill(corrupt, 1)
    end
    self.set_batch(x, y, t)
    local _, optim_ret = optimize(self.ftrain, self.params, optim_opt)
    loss = loss + optim_ret[1]
    if not opt.silent then
      xlua.progress(batch_end, data:size())
    end
  end
  return loss/data:size()
end

function Model:eval(data)
  local opt = self.opt
  local loss, pred, targ = 0, {}, {}
  self:set_mode('evaluate')
  for batch, batch_end in data:batches(opt.n_batch, opt) do
    local x, y, t = tl.Dataset.pad(batch.X, opt.pad_index), torch.Tensor(batch.Y), batch.typecheck
    self.set_batch(x, y, t)
    local loss_, pred_ = self.predict()
    loss = loss + loss_
    tl.util.extend(pred, pred_:totable())
    tl.util.extend(targ, y:totable())
    if not opt.silent then
      xlua.progress(batch_end, data:size())
    end
  end
  pred = torch.Tensor(pred)
  targ = torch.Tensor(targ)
  return loss/data:size(), pred, targ
end

function Model:fit(dataset, callbacks)
  local opt = self.opt
  callbacks = callbacks or {}

  local printable = function(stats)
    local p = tl.util.tableCopy(stats)
    for _, t in ipairs{'train', 'dev', 'test'} do
      if p[t] then p[t].cf = nil end
    end
    return p
  end

  local patience = opt.patience
  local best, loss, pred, targ
  local hist = {}
  local best_params = self.params:clone()

  local save = function(epoch)
    epoch = epoch or ''
    if opt.save then
      print('saving to '..opt.save)
      if not path.isdir(opt.save) then
        print('making directory at '..opt.save)
        dir.makepath(opt.save)
      end
      pretty.dump(opt, path.join(opt.save, 'opt.json'))
      pretty.dump(hist, path.join(opt.save, 'hist.json'))
      pretty.dump(printable(best), path.join(opt.save, 'best.json'))
      for k, v in pairs(best) do
        if type(v) == 'table' and v.cf then
          torch.save(path.join(opt.save, k..'.cf.t7'), v.cf)
        end
      end
      torch.save(path.join(opt.save, 'model'..epoch..'.t7'), self)
      torch.save(path.join(opt.save, 'params'..epoch..'.t7'), self.params:float())
    end
  end


  for epoch = 1, opt.n_epoch do
    stats = {epoch=epoch, train={}, dev={}}
    collectgarbage()
    dataset.train:shuffle()
    print('epoch '..epoch)
    self:train(dataset.train)
    loss, pred, targ = self:eval(dataset.train)
    if callbacks.stats then stats.train = callbacks.stats(pred, targ) else stats.train = {} end
    stats.train.dparam_norm, stats.train.loss = self:gradnorm(), loss

    loss, pred, targ = self:eval(dataset.dev)
    if callbacks.stats then stats.dev = callbacks.stats(pred, targ) else stats.dev = {} end
    stats.dev.loss = loss

    if callbacks.debug then
      callbacks.debug(dataset.dev.X, targ, pred)
    end

    if callbacks.log then
      callbacks.log(printable(stats))
    end

    if not best or stats.dev.micro.f1 > best.dev.micro.f1 then
      print('found new best!')
      best = tablex.deepcopy(stats)
      patience = opt.patience
      best_params:copy(self.params)
      save()
    else
      patience = patience - 1
    end
    stats.patience = patience
    print(pretty.write(printable(stats)))
    hist[epoch] = stats
    if opt.patience > 0 and patience == 0 then
      print('early stopping triggered!')
      break
    end
  end

  self.params:copy(best_params)

  if dataset.test then
    loss, pred, targ = self:eval(dataset.test)
    if callbacks.stats then best.test = callbacks.stats(pred, targ) else best.test = {} end
    best.test.loss = loss
  end
  print('best score')
  print(pretty.write(printable(best)))
  save()

  return printable(best)
end

return Model
