import torch
import common
import models

#filename = r'basic_resnet_self_sup_rotation.model'
filename = r'unet_256x256_sing_ch.model'

x = torch.load(filename)

print(x['inc.double_conv.0.weight'].shape)





