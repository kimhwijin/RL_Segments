import torch
import torch.nn as nn
import torch.nn.functional as F
import timesynth as ts

class ZeroMask(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, mask):
        return x * mask
    

class SeqCombMask(nn.Module):
    def __init__(self):
        super().__init__()
        noise = ts.noise.GaussianNoise(std=0.01)
        x = ts.signals.NARMA(order=2)
        self.x_ts = ts.TimeSeries(x, noise_generator=noise)
    def forward(self, x, mask):
        B, T, D = x.shape
        x_star, _, _ = self.x_ts.sample(torch.arange(T).repeat(B*D))
        x_star = x_star.reshape(B, T, D)
        x_star = torch.tensor(x_star, dtype=x.dtype, device=x.device)
        return x * mask + x_star * torch.logical_not(mask)
    