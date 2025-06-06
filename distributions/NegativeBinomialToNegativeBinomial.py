import scipy.special as S

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

    @torch.no_grad()
    def calculate_marginal_mask(self):
        """
        Returns
        -------
        mask_probs : torch.Tensor  shape = (B, seq_len)
            mask_probs[b, t] = P(t ∈ [start, end])  ∈ [0, 1]
            (start, Δ are NB; end = clamp(start+Δ, seq_len-1))
        """
        if self.seq_len is None:
            raise ValueError("seq_len must be set.")

        T        = self.seq_len
        device   = self.nb_start.probs.device
        dtype    = self.nb_start.probs.dtype
        B        = self.nb_start.total_count.shape[0]

        # ── 1) 시작점 분포 (NB + 상한 클램프) ─────────────────────────
        k_vals   = torch.arange(T - 1, device=device).unsqueeze(0)         # 0 … T-2
        pmf_part = self.nb_start.log_prob(k_vals).exp().cpu()        # (B, T-1)
        tail_prob = 1.0 - pmf_part.sum(dim=1, keepdim=True)    # P(start ≥ T-1)
        start_p  = torch.cat([pmf_part, tail_prob], dim=1)     # (B, T)

        # ── 2) Δ 생존함수  S(d)=P(Δ ≥ d)  for d=0…T-1 ───────────────
        r        = self.nb_end.total_count.to(dtype).detach().cpu()           # (B,)
        p        = self.nb_end.probs.to(dtype).detach().cpu()                 # (B,)

        d_vals   = torch.arange(T, dtype=dtype)  # 0…T-1
        # NB CDF  F(k;r,p)=I_p(r,k+1)
        F_d      = S.betainc(r.unsqueeze(1), d_vals.unsqueeze(0) + 1,
                             p.unsqueeze(1))                   # (B, T)
        surv     = torch.empty_like(F_d)                       # (B, T)
        surv[:, 0] = 1.0                                       # d = 0
        if T > 1:
            surv[:, 1:] = 1.0 - F_d[:, :-1]                    # d ≥ 1

        # ── 3) 컨볼루션  P_t = Σ_s P(S=s)·S(t−s) (t ≥ s) ────────────
        mask_probs = torch.zeros(B, T, dtype=dtype)

        for s in range(T):                                     # T ≤ 100 OK
            mask_probs[:, s:] += start_p[:, s:s+1] * surv[:, :T - s]

        return mask_probs.clamp_(0.0, 1.0)     # 수치 오차 안전