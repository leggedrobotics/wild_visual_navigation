import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data


class SimpleGCN(torch.nn.Module):
    def __init__(self, input_size: int, reconstruction: bool, hidden_sizes=[64, 32, 1]):
        super(SimpleGCN, self).__init__()

        self.layers = []
        inp = input_size
        for j, h in enumerate(hidden_sizes):
            if reconstruction and j == len(hidden_sizes) - 1:
                h += input_size

            self.layers.append(GCNConv(inp, h))
            inp = h

        self.layers = torch.nn.ModuleList(self.layers)

    def forward(self, data: Data) -> torch.tensor:
        x, edge_index = data.x, data.edge_index
        for j, layer in enumerate(self.layers):
            if j != len(self.layers) - 1:
                x = F.relu(layer(x, edge_index))
            else:
                x = layer(x, edge_index)

        # x = F.dropout(x, training=self.training)
        x[:, 0] = torch.sigmoid(x[:, 0])
        return x
