import sys
sys.path.append("/home/hjkim/RL_TimeSegment")

import torch
import torch.nn as nn
import torch.nn.functional as F
# 
from backbones import get_default_backbone


class PredictorNetwork(nn.Module):
    def __init__(self, d_in, d_model, d_out, seq_len, backbone):
        super().__init__()
        self.backbone = get_default_backbone(d_in, d_model, seq_len, backbone)
        self.proj = nn.Linear(d_model, d_out)

    def forward(self, x):   
        # B x T x D
        z = self.backbone(x)
        z = F.gelu(z)
        out = self.proj(z)
        return out


