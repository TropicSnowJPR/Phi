import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import sys
import os
import warnings

class Policy(nn.Module):
    def __init__(self, input_size=256, action_size=6):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, action_size)
        )

    def forward(self, x):
        return self.net(x)