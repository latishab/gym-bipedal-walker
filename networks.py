import torch
from torch import nn
import numpy as np

# Orthogonal Initialization
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class FeedForwardNN(nn.Module):
    def __init__(self, in_dim, out_dim, std=0.01):
        super(FeedForwardNN, self).__init__()

        self.network = nn.Sequential(
            layer_init(nn.Linear(in_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, out_dim), std=std)
        )

    def forward(self, obs):
        obs = torch.tensor(obs, dtype=torch.float) if isinstance(obs, np.ndarray) else obs
        return self.network(obs)
