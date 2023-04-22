import torch.nn as nn

class Normalize(nn.Module):
    def __init__(self, factor):
        super(Normalize, self).__init__()
        self.factor = factor

    def forward(self, x):
        return x / self.factor

class Denormalize(nn.Module):
    def __init__(self, factor):
        super(Denormalize, self).__init__()
        self.factor = factor

    def forward(self, x):
        return x * self.factor
