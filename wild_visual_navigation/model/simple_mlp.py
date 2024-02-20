#
# Copyright (c) 2022-2024, ETH Zurich, Jonas Frey, Matias Mattamala.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
from wild_visual_navigation.utils import Data
import torch


class SimpleMLP(torch.nn.Module):
    def __init__(
        self,
        input_size: int = 64,
        hidden_sizes: [int] = [255],
        reconstruction: bool = False,
    ):
        super(SimpleMLP, self).__init__()
        layers = []
        self.nr_sigmoid_layers = hidden_sizes[-1]

        if reconstruction:
            hidden_sizes[-1] = hidden_sizes[-1] + input_size

        for hs in hidden_sizes[:-1]:
            layers.append(torch.nn.Linear(input_size, hs))
            layers.append(torch.nn.ReLU())
            input_size = hs
        layers.append(torch.nn.Linear(input_size, hidden_sizes[-1]))

        self.layers = torch.nn.Sequential(*layers)
        self.output_features = hidden_sizes[-1]

    def forward(self, data: Data) -> torch.Tensor:
        x = data.x
        # Checked data is correctly memory aligned and can be reshaped
        # If you change something in the dataloader make sure this is still working
        x = self.layers(x)
        x[:, : self.nr_sigmoid_layers] = torch.sigmoid(x[:, : self.nr_sigmoid_layers])
        return x


class DoubleMLP(torch.nn.Module):
    def __init__(self, input_size: int = 64, hidden_sizes: [int] = [255]):
        super(DoubleMLP, self).__init__()
        self.nr_sigmoid_layers = hidden_sizes[-1]
        self.networks = []
        for network_last_layer in [hidden_sizes[-1], input_size]:
            layers = []
            inter_size = input_size
            for hs in hidden_sizes[:-1]:
                layers.append(torch.nn.Linear(inter_size, hs))
                layers.append(torch.nn.ReLU())
                inter_size = hs

            layers.append(torch.nn.Linear(inter_size, network_last_layer))

            self.networks.append(torch.nn.Sequential(*layers))
        self.networks = torch.nn.ModuleList(self.networks)
        self.output_features = hidden_sizes[-1] + input_size

    def forward(self, data: Data) -> torch.Tensor:
        x = data.x
        # Checked data is correctly memory aligned and can be reshaped
        # If you change something in the dataloader make sure this is still working
        x1 = torch.sigmoid(self.networks[0](x))
        x2 = self.networks[1](x)
        return torch.cat([x1, x2], dim=1)
