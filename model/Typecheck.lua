local Typecheck, Parent = torch.class('nn.Typecheck', 'nn.Module')

function Typecheck:__init(invalid_type_scale)
  self.invalid_type_scale = invalid_type_scale or 10
  Parent.__init(self)
  self.invalid_types = torch.Tensor()
end

function Typecheck:updateOutput(input)
  local scores, typecheck = table.unpack(input)
  if self.invalid_types:type() ~= scores:type() then
    self.invalid_types = self.invalid_types:type(scores:type())
  end
  self.invalid_types:resizeAs(scores):fill(1):add(-typecheck)
  self.output:resizeAs(scores):copy(scores)
  self.output:maskedFill(self.invalid_types, math.min(0, scores:min() * self.invalid_type_scale))
  return self.output
end

function Typecheck:updateGradInput(input, gradOutput)
  local scores, typecheck = table.unpack(input)
  self.gradInput:resizeAs(gradOutput):copy(gradOutput)
  self.gradInput:maskedFill(self.invalid_types, 0)
  return self.gradInput
end
