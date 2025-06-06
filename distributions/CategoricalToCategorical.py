import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

import torchrl

class CategoricalToCategorical(D.Distribution):
    def __init__(self, start_logits, end_logits, seq_len=None):
        self.cat_start = D.Categorical(logits=start_logits)
        self.end_logits = end_logits
        self.seq_len = seq_len

    def get_end_dist(self, start):
        arange = torch.arange(self.seq_len, dtype=start.dtype, device=start.device).unsqueeze(0)

        mask = start <= arange
        end_logits = mask * self.end_logits + torch.logical_not(mask) * -1e10
        return D.Categorical(logits=end_logits)

    def rsample(self, sample_shape):
        start = self.cat_start.sample(sample_shape).unsqueeze(-1)
        self.cat_end = self.get_end_dist(start)

        end = self.cat_end.sample(sample_shape).unsqueeze(-1)

        return torch.concat([start, end], dim=1).int()

    def log_prob(self, value):
        start = value[:, 0]
        end = value[:, 1]

        self.cat_end = self.get_end_dist(start.unsqueeze(-1))

        start_log_probs = self.cat_start.log_prob(start)
        end_log_probs = self.cat_end.log_prob(end)

        return start_log_probs + end_log_probs
    
    @property
    def deterministic_sample(self):
        start = self.cat_start.mode.unsqueeze(-1)
        self.cat_end = self.get_end_dist(start)

        end = self.cat_end.mode.unsqueeze(-1)
        return torch.concat([start, end], dim=1).int()
    
    def entropy(self):
        return self.cat_start.entropy()

    @torch.no_grad()
    def calculate_marginal_mask(self):
        start_probs = self.cat_start.probs
        exp_scores = self.end_logits.exp()
        cum_rev = torch.cumsum(exp_scores.flip(-1), dim=-1).flip(-1)  # (B, T)
        denom   = cum_rev + 1e-7

        ratio = cum_rev[:, :, None] / denom[:, None, :]
        tri_mask = torch.tril(torch.ones(
            self.seq_len, self.seq_len, device=ratio.device, dtype=ratio.dtype)
        )                                                            # (T, T)
        ratio *= tri_mask                                            # broadcast

        # ----- 4) Σ_s  P(S=s) · P(E ≥ t | S=s)  -----------------------
        mask_probs = torch.sum(ratio * start_probs[:, None, :], dim=-1)  # (B, T)
        return mask_probs.clamp_(0.0, 1.0)   # 수치 오차 방지

