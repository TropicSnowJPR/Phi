import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

class Memory(nn.Module):
    def __init__(self, input_size=3136, hidden_size=256):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        return out, hidden