import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self,
                 enc_in: int = 1,
                 d_model: int = 128,
                 e_layers: int = 3,
                 bidirectional: bool = True,
                 dropout: float = 0.2,
                 pooling: str = "max"):          # "avg", "last", "max", "attn"
        super().__init__()

        self.bidirectional = bidirectional
        self.num_dirs = 2 if bidirectional else 1
        self.pooling = pooling.lower()

        hidden = d_model // self.num_dirs      # 각 방향 hidden 크기
        self.lstm = nn.LSTM(
            enc_in, hidden,
            num_layers=e_layers,
            batch_first=True,
            dropout=dropout if e_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        if self.pooling == "attn":
            # 학습용 attention 점수 계층
            self.attn = nn.Linear(hidden * self.num_dirs, 1, bias=False)

        # 최종 출력 차원
        self.output_dim = hidden * self.num_dirs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        y, (h_n, _) = self.lstm(x)             # y: (B, T, H*num_dirs)

        if self.pooling == "avg":              # 기존과 동일
            out = y.mean(dim=1)

        elif self.pooling == "max":
            out, _ = y.max(dim=1)              # (B, H*num_dirs)

        elif self.pooling == "last":
            out = y[:, -1]                     # (B, H*num_dirs)

        elif self.pooling == "attn":
            # 점수: (B,T,1) → softmax → 가중합
            scores = self.attn(y)              # (B, T, 1)
            α = F.softmax(scores, dim=1)       # (B, T, 1)
            out = (α * y).sum(dim=1)           # (B, H*num_dirs)

        else:
            raise ValueError(f"Unknown pooling '{self.pooling}'")

        return out