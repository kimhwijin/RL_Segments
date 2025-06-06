import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

import torchrl
import scipy.special as S

class CategoricalToNegativeBinomial(D.Distribution):
    def __init__(self, start_logits, end_logits, seq_len=None):
        self.cat = D.Categorical(logits=start_logits)
        self.nb = D.NegativeBinomial(
            total_count=torchrl.modules.utils.biased_softplus(1.)(end_logits[:, 0]),
            logits=end_logits[:, 1]
        )
        self.seq_len = seq_len

    def rsample(self, sample_shape):
        start = self.cat.sample(sample_shape).unsqueeze(-1)
        delta = self.nb.sample(sample_shape).unsqueeze(-1)
        end = torch.clamp(start + delta, 0., self.seq_len-1)

        return torch.concat([start, end], dim=1).int()

    def log_prob(self, value):
        start = value[:, 0]
        end = value[:, 1] - start
        start_log_probs = self.cat.log_prob(start)
        end_log_probs = self.nb.log_prob(end)
        return start_log_probs + end_log_probs
    
    @property
    def deterministic_sample(self):
        start = self.cat.mode.unsqueeze(-1)
        delta = self.nb.mode.unsqueeze(-1)
        end = torch.clamp(start + delta, 0., self.seq_len-1)
        
        return torch.concat([start, end], dim=1).int()
    
    def entropy(self):
        return self.cat.entropy() # + self.nb.entropy()

    @torch.no_grad()
    def calculate_marginal_mask(self):
        
        start_p = self.cat.probs.detach().cpu()                      # (B, T)
        B, T    = start_p.shape
        device  = start_p.device
        dtype   = start_p.dtype
        
        r = self.nb.total_count.to(dtype).detach().cpu()             # (B,) or scalar
        p = self.nb.probs.to(dtype).detach().cpu()                   # (B,)

        # --- NB CDF: F(k; r, p) = I_p(r, k+1) ------------------------
        k_vals  = torch.arange(T, dtype=dtype)   # 0…T-1
        # shape → (B, T) by unsqueeze & broadcast
        F_k     = S.betainc(r.unsqueeze(1), k_vals.unsqueeze(0) + 1, p.unsqueeze(1))
        # survival for offset d = P(Δ ≥ d) = 1 - F(d-1),  with F(-1)=0
        surv = torch.ones_like(F_k)                   # (B, T)  d=0 → 1
        surv[:, 1:] = 1.0 - F_k[:, :-1]               # d≥1

        # --- convolution‐style accumulation -------------------------
        # mask_probs[b, t] = Σ_{s≤t} start_p[b, s] * surv[b, t-s]
        mask_probs = torch.zeros_like(start_p)

        for s in range(T):                            # loop ≤100 → 빠름
            mask_probs[:, s:] += start_p[:, s:s+1] * surv[:, :T-s]
        return mask_probs.clamp_(0.0, 1.0)

