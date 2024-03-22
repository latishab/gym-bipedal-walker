import torch
from torch import nn
import numpy as np
from torch.distributions import Normal

# Orthogonal Initialization
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class MLPCritic(nn.Module):
    def __init__(self, in_dim, out_dim, std=0.01):
        super(MLPCritic, self).__init__()

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

class MLPActor(nn.Module):
    """Include state-independet log standard deviation"""
    def __init__(self, in_dim, out_dim, std=0.01):
        super(MLPActor, self).__init__()

        self.network = nn.Sequential(
            layer_init(nn.Linear(in_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, out_dim), std=std)
        )

        # Log standard deviation parameter, initialized to 0, implying std = 1.
        self.log_std = nn.Parameter(torch.zeros(1, out_dim))

    def forward(self, obs):
        obs = torch.tensor(obs, dtype=torch.float) if isinstance(obs, np.ndarray) else obs
        mean = self.network(obs) # calculate mean action values
        std = torch.exp(self.log_std) # standard deviation is derived from log std parameter
        return mean, std
