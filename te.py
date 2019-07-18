from torch import  nn
import torch
m = nn.LogSoftmax(dim=1)
loss = nn.NLLLoss()
# input is of size N x C = 3 x 5
input = torch.randn(3, 5, requires_grad=True)
res = m(input)
print(res)
# each element in target has to have 0 <= value < C
target = torch.tensor([1, 0, 4])
print(target)
print(target.size(), res.size())
output = loss(res, target)
print(output)
output.backward()