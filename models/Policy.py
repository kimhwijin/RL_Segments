import sys
sys.path.append("/home/hjkim/RL_TimeSegment")

import torch
import torch.nn as nn
import torch.nn.functional as F

from tensordict import TensorDict
from tensordict.nn import TensorDictModule, InteractionType

from torchrl.modules import ProbabilisticActor
from torchrl.data import BoundedTensorSpec

# 
from backbones import get_default_backbone
from distributions import CategoricalToNegativeBinomial, CategoricalToCategorical, NegativeBinomialToNegativeBinomial


def init_policy_module(
    d_in,
    d_model,
    seq_len,
    backbone,
    seg_dist
):
    if seg_dist == 'cat_nb':
        d_start, d_end = 100, 2
        SegmentDistribution = CategoricalToNegativeBinomial
    elif seg_dist == 'cat_cat':
        d_start, d_end = 100, 100
        SegmentDistribution = CategoricalToCategorical
    elif seg_dist == 'nb_nb':
        d_start, d_end = 2, 2
        SegmentDistribution = NegativeBinomialToNegativeBinomial

    policy_net = PolicyNetwork(
        d_in = d_in+1,
        d_model = d_model,
        d_start = d_start,
        d_end = d_end,
        seq_len = seq_len,
        backbone = backbone
    )

    policy_module = TensorDictModule(
        policy_net, 
        in_keys=['x', 'curr_mask'], 
        out_keys=['start_logits', 'end_logits']
    )

    policy_module = ProbabilisticActor(
        module=policy_module,
        spec=BoundedTensorSpec(low=0, high=seq_len-1, dtype=int, shape=(2,)),
        in_keys=["start_logits", "end_logits"],
        distribution_class=SegmentDistribution,
        distribution_kwargs={"seq_len": seq_len},
        return_log_prob=True,
        default_interaction_type=InteractionType.RANDOM
    )

    return policy_module

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


