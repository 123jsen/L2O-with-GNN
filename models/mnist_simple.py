import torch
from torch.nn import functional as F
from torch import nn as nn

class simple_training_NN(nn.Module):
    def __init__(self):
        super(simple_training_NN, self).__init__()
        self.flatten = nn.Flatten()
        self.hidden_layer = nn.Linear(28 * 28, 32)
        self.output_layer = nn.Linear(32, 10)

    def forward(self, x):
        x = self.flatten(x)

        x = self.hidden_layer(x)
        x = F.relu(x)

        x = self.output_layer(x)
        return x

class simple_testing_NN(nn.Module):
    def __init__(self):
        super(simple_testing_NN, self).__init__()
        self.flatten = nn.Flatten()
        self.hidden_layer_1 = nn.Linear(28 * 28, 64)
        self.hidden_layer_2 = nn.Linear(64, 64)
        self.output_layer = nn.Linear(64, 10)

    def forward(self, x):
        x = self.flatten(x)

        x = self.hidden_layer_1(x)
        x = F.relu(x)

        x = self.hidden_layer_2(x)
        x = F.relu(x)

        x = self.output_layer(x)
        return x
