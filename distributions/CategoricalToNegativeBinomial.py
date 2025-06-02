import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

import torchrl

class CategoricalToNegativeBinomial(D.Distribution):
    has_rsample = True           # we keep rsample()

    def __init__(self, start_logits, end_logits, seq_len):
        super().__init__()
        self.cat = D.Categorical(logits=start_logits)          # [B,T]
        # NB params : r>0, p∈(0,1)
        r = torchrl.modules.utils.biased_softplus(1.)(end_logits[:, 0])
        p = torch.sigmoid(end_logits[:, 1])
        self.r, self.p = r, p
        self.seq_len = seq_len

    # ---------- helper : Z_s (normalizer) ------------------
    def _Z(self, L_max):
        """F_raw(L_max)  (broadcast w/ r,p)."""
        # enumerate 0..L_max  (max  T≈100  → OK)
        k = torch.arange(self.seq_len, device=L_max.device)                     # [T]
        pmf = nb_raw_pmf(k, self.r.unsqueeze(-1), self.p.unsqueeze(-1))   # [B,T]
        cdf = torch.cumsum(pmf, -1)                                       # [B,T]
        return cdf.gather(-1, L_max.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12)

    # ---------- rsample ------------------------------------
    def rsample(self, sample_shape=torch.Size()):
        # sample start
        start = self.cat.sample(sample_shape).unsqueeze(-1)               # [S,B,1]
        L_max = self.seq_len - 1 - start                                   # ...
        # enumerate logits 0..L_max
        k = torch.arange(self.seq_len, device=start.device)               # [T]
        pmf = nb_raw_pmf(k, self.r.unsqueeze(-1), self.p.unsqueeze(-1))   # [B,T]
        pmf = pmf / self._Z(L_max)                                        # truncate+norm
        pmf = pmf * (k<=L_max)                                            # mask tall stack
        # categorical draw for length
        cat_len = D.Categorical(probs=pmf)
        length = cat_len.sample(sample_shape).unsqueeze(-1)
        end = start + length
        return torch.cat([start, end], -1).int()

    # ---------- log_prob -----------------------------------
    def log_prob(self, value):
        start, end = value[..., 0], value[..., 1]
        length = end - start
        L_max = self.seq_len - 1 - start
        lp_start = self.cat.log_prob(start)

        lp_len_raw = torch.log(
            nb_raw_pmf(length, self.r, self.p).clamp_min(1e-30)
        )
        lp_len = lp_len_raw - torch.log(self._Z(L_max))
        return lp_start + lp_len

    # ---------- deterministic (mode) -----------------------
    @property
    def deterministic_sample(self):
        start = self.cat.mode.unsqueeze(-1)                     # [B,1]
        L_max = self.seq_len - 1 - start
        # raw NB mode then clamp to L_max
        raw_mode = torch.clamp(((self.p*(self.r-1))/(1-self.p)).floor(), min=0)
        length = torch.minimum(raw_mode.int(), L_max)
        end = start + length
        return torch.cat([start, end], -1).int()

    def entropy(self):
        # closed-form for truncated NB 가 없으므로 근사(샘플링)하거나 enumerate
        return None


# class CategoricalToNegativeBinomial(D.Distribution):
#     def __init__(self, start_logits, end_logits, seq_len=None):
#         self.cat = D.Categorical(logits=start_logits)
#         self.nb = D.NegativeBinomial(
#             total_count=torchrl.modules.utils.biased_softplus(1.)(end_logits[:, 0]),
#             logits=end_logits[:, 1]
#         )
#         self.seq_len = seq_len

#     def rsample(self, sample_shape):
#         start = self.cat.sample(sample_shape).unsqueeze(-1)
#         delta = self.nb.sample(sample_shape).unsqueeze(-1)
#         end = torch.clamp(start + delta, 0., self.seq_len-1)

#         return torch.concat([start, end], dim=1).int()

#     def log_prob(self, value):
#         start = value[:, 0]
#         end = value[:, 1] - start
#         start_log_probs = self.cat.log_prob(start)
#         end_log_probs = self.nb.log_prob(end)
#         return start_log_probs + end_log_probs
    
#     @property
#     def deterministic_sample(self):
#         start = self.cat.mode.unsqueeze(-1)
#         delta = self.nb.mode.unsqueeze(-1)
#         end = torch.clamp(start + delta, 0., self.seq_len-1)
        
#         return torch.concat([start, end], dim=1).int()
    
#     def entropy(self):
#         return self.cat.entropy() # + self.nb.entropy()