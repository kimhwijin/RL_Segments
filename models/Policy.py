import sys
sys.path.append("/home/hjkim/RL_TimeSegment")

import torch
import torch.nn as nn
import torch.nn.functional as F
# 
from backbones import get_default_backbone

class PolicyNetwork(nn.Module):
    def __init__(self, d_in, d_model, d_start, d_end, seq_len, backbone):
        super().__init__()
        
        self.backbone = get_default_backbone(d_in, d_model, seq_len, backbone)
        self.start_proj = nn.Linear(d_model, d_start)
        self.end_proj = nn.Linear(d_model, d_end)

    def forward(self, x, curr_mask):        
        inputs = torch.concat([x, curr_mask], dim=-1)
        # B x T x D
        z = self.backbone(inputs)
        z = z.reshape(z.shape[0], -1)
        z = F.tanh(z)
        start_params = self.start_proj(z)
        end_params = self.end_proj(z)

        return start_params, end_params


