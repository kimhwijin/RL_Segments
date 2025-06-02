import sys
sys.path.append("/home/hjkim/RL_TimeSegment")

import torch
import torch.nn.functional as F

from utils import masking

@torch.no_grad()
def compose_reward(reward_fns, weights, **kwargs):
    rewards = torch.stack([reward_fn(**kwargs) for reward_fn in reward_fns], dim=1)

    weights = torch.tensor([weights], dtype=rewards.dtype, device=rewards.device)
    reward = (rewards * weights).sum(-1)
    return reward

@torch.no_grad()
def length_reward(x, y, curr_mask, new_mask):
    seq_len = x.shape[1]
    next_mask = torch.logical_or(curr_mask, new_mask)

    reward = next_mask.sum([1, 2], dtype=x.dtype) / seq_len
    return 1. - reward.detach()


@torch.no_grad()
def minus_cross_entropy_reward(x, y, curr_mask, new_mask, predictor, mask_fn):
    curr_x = mask_fn(x, curr_mask)
    next_x = mask_fn(x, torch.logical_or(curr_mask, new_mask))
    logits = predictor(next_x)
    
    reward = -F.cross_entropy(logits, y, reduction='none')
    return reward.detach()

@torch.no_grad()
def exp_minus_cross_entropy_reward(x, y, curr_mask, new_mask, predictor, mask_fn):
    curr_x = mask_fn(x, curr_mask)
    next_x = mask_fn(x, torch.logical_or(curr_mask, new_mask))
    logits = predictor(next_x)
    
    reward = torch.exp(-F.cross_entropy(logits, y, reduction='none'))
    return reward.detach()

@torch.no_grad()
def inverse_cross_entropy_reward(x, y, curr_mask, new_mask, predictor, mask_fn):
    curr_x = mask_fn(x, curr_mask)
    next_x = mask_fn(x, torch.logical_or(curr_mask, new_mask))
    logits = predictor(next_x)
    
    reward = 1 / (F.cross_entropy(logits, y, reduction='none') + 1e-10)
    return reward.detach()


    