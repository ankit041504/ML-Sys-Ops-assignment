import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self, input_size=784, hidden=128, output=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output)
        )

    def forward(self, x):
        return self.net(x)
