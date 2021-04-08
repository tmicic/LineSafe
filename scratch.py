import torch

x = torch.zeros((3,512,512))

print(x.shape)

y = x.permute(1,2,0)

print(y.shape)
