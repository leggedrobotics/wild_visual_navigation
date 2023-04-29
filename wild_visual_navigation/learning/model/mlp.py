import torch


class BinaryClassification(torch.nn.Module):
    def __init__(self, feat_size, batch_size):
        super(BinaryClassification, self).__init__()

        self.relu = torch.nn.ReLU()
        self.layer_1 = torch.nn.Linear(feat_size, batch_size)
        self.batchnorm1 = torch.nn.BatchNorm1d(batch_size)
        self.layer_2 = torch.nn.Linear(batch_size, 64)
        self.batchnorm2 = torch.nn.BatchNorm1d(64)
        self.dropout = torch.nn.Dropout(p=0.2)
        self.layer_out = torch.nn.Linear(64, 1)

    def forward(self, inputs):
        x = self.layer_1(inputs)
        x = self.relu(x)
        x = self.batchnorm1(x)
        x = self.layer_2(x)
        x = self.relu(x)
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)

        return x