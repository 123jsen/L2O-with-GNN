import torch.nn as nn
import torch.nn.functional as F

class l2o_lstm_model(nn.Module):
    def __init__(self):
        super(l2o_lstm_model, self).__init__()
        self.LSTM = nn.LSTM(1, 24, num_layers=2)
        self.linear = nn.Linear(24, 1)
    
    def forward(self, x):
        out, _ = self.LSTM(x)
        out = out[-1]       # Only the last value to do prediction
        out = self.linear(out)
        return out