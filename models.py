import torch
import torch.nn as nn
from torchvision.models import resnet18
import torch.nn.functional as F

class Resnet(nn.Module):

    def __init__(self, out_features=2):
        super().__init__()

        self.out_features = out_features

        self.resnet_component = resnet18(pretrained=True)

        self.resnet_component.fc = nn.Linear(self.resnet_component.fc.in_features, self.out_features)


    def forward(self, X):
        
        X = self.resnet_component(X)
        X = F.log_softmax(X, dim=1)

        return X

if __name__ == '__main__':
    
    model = Resnet()


