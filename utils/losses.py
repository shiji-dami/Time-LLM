# utils/losses.py

import torch.nn as nn

class MSE(nn.Module):
    def forward(self, prediction, target):
        return torch.mean((prediction - target)**2)

class MAE(nn.Module):
    def forward(self, prediction, target):
        return torch.mean(torch.abs(prediction - target))
