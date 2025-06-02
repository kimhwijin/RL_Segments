import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

import torchrl

class NegativeBinomialToNegativeBinomial(D.Distribution):

    def __init__(self, start_logits, end_logits, seq_len=None):
        self.nb_start = D.NegativeBinomial(
            total_count=torchrl.modules.utils.biased_softplus(1.)(start_logits[:, 0]),
            logits=start_logits[:, 1]
        )
        self.nb_end = D.NegativeBinomial(
            total_count=torchrl.modules.utils.biased_softplus(1.)(end_logits[:, 0]),
            logits=end_logits[:, 1]
        )
        self.seq_len = seq_len

    def rsample(self, sample_shape):
        start = self.nb_start.sample(sample_shape).unsqueeze(-1)
        start = torch.clamp(start, 0, self.seq_len-1)

        delta = self.nb_end.sample(sample_shape).unsqueeze(-1)
        end = torch.clamp(start + delta, 0., self.seq_len-1)

        return torch.concat([start, end], dim=1).int()

    def log_prob(self, value):
        start = value[:, 0]
        end = value[:, 1] - start

        start_log_probs = self.nb_start.log_prob(start)
        end_log_probs = self.nb_end.log_prob(end)
        return start_log_probs + end_log_probs
    
    @property
    def deterministic_sample(self):
        start = self.nb_start.mode.unsqueeze(-1)
        start = torch.clamp(start, 0, self.seq_len-1)

        delta = self.nb_end.mode.unsqueeze(-1)
        end = torch.clamp(start + delta, 0., self.seq_len-1)
        
        return torch.concat([start, end], dim=1).int()
    
    def entropy(self):
        return self.nb_start.entropy() # + self.nb.entropy()