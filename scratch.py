import torch
import timeit

x = torch.randn((1,256,256))



starttime = timeit.default_timer()
for d in range(160):
    y = x.rot90(k=3, dims=(1,2))
endtime = timeit.default_timer()


print(endtime-starttime)




