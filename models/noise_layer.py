import torch
import torch.nn as nn
import numpy as np


class NoiseLayer(nn.Module):
    def __init__(self, theta, k):
        super(NoiseLayer, self).__init__()
        self.theta = nn.Linear(k, k, bias=False)
        self.theta.weight.data = nn.Parameter(theta)
        self.eye = torch.Tensor(np.eye(k))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        theta = self.eye.to(x.device).detach()
        theta = self.theta(theta)
        theta = self.softmax(theta)
        out = torch.matmul(x, theta)
        return out