import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

'''Simple learnable model by variable gradient descent step size'''
class gd_l2o_weight(nn.Module):
    def __init__(self, start_lr):
        super(gd_l2o_weight, self).__init__()
        self.w = nn.Parameter(torch.tensor(start_lr))

    def forward(self, x):
        out = self.w * x
        return out

'''l2o model based on lstm, refer to Androchowicz et al.'''
class lstm_l2o_optimizer(nn.Module):
    def __init__(self):
        super(lstm_l2o_optimizer, self).__init__()
        self.LSTM = nn.LSTM(1, 24, num_layers=2)    # gradient is the sole input
        self.linear = nn.Linear(24, 1)
    
    def forward(self, x, h_in):
        out, h_out = self.LSTM(x, h_in)
        out = self.linear(out)
        return x * out, h_out

'''l2o Recurrent GNN model. The diffused output state is not kept'''
class gnn_l2o_optimizer(nn.Module):
    def __init__(self):
        super(gnn_l2o_optimizer, self).__init__()
        self.LSTM = nn.LSTM(1, 24)    # gradient is the sole input
                                        # For LSTMs, h_out and out are equal
        self.graph_conv = GCNConv(24, 24)
        self.linear = nn.Linear(24, 1)
    
    # h is short term hidden state, c is long term cell state
    def forward(self, x, h, edge_index):
        out = x.reshape(1, -1, 1)
        out, h = self.LSTM(out, h)

        out = out.reshape(-1, 24)
        out = self.graph_conv(out, edge_index)

        out = self.linear(out)

        out = out.reshape(-1)
        return x * out, h