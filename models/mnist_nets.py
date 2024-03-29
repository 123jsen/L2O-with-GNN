import torch
from torch.nn import functional as F
from torch import nn as nn

'''Simple classification network for MNIST dataset'''
class class_net(nn.Module):
    def __init__(self, inputNum, outputNum):
        super(class_net, self).__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(inputNum, 32)
        self.layer2 = nn.Linear(32, outputNum)

    def forward(self, x):
        x = self.flatten(x)

        x = self.layer1(x)
        x = F.relu(x)

        x = self.layer2(x)
        return x

'''Deeper classification network for MNIST dataset'''
class deep_net(nn.Module):
    def __init__(self, inputNum, outputNum):
        super(deep_net, self).__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(inputNum, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, outputNum)

    def forward(self, x):
        x = self.flatten(x)
         
        x = self.layer1(x)
        x = F.relu(x)

        x = self.layer2(x)
        x = F.relu(x)

        x = self.layer3(x)
        return x