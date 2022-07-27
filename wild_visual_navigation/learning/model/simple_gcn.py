import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data


class SimpleGCN(torch.nn.Module):
    def __init__(self, num_node_features: int, num_classes: int, reconstruction: bool):
        super(SimpleGCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        if reconstruction:
            num_classes += num_node_features

        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data: Data) -> torch.tensor:
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x[:, 0] = F.sigmoid(x[:, 0])
        return x
