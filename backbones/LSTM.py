import torch
import torch.nn as nn

class Model(nn.Module):
    """Stacked LSTM encoder with optional sequence output aggregation."""
    def __init__(self,
                 enc_in: int = 1,
                 d_model: int = 128,
                 e_layers: int = 3,
                 bidirectional: bool = True,
                 dropout: float = 0.2):
        super().__init__()
        d_model = d_model // 2
        
        self.bidirectional = bidirectional
        self.num_dirs = 2 if bidirectional else 1
        self.lstm = nn.LSTM(
            enc_in, d_model,
            num_layers=e_layers,
            batch_first=True,
            dropout=dropout if e_layers>1 else 0.0,
            bidirectional=bidirectional
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.output_dim = d_model*self.num_dirs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        y, _ = self.lstm(x)          # (B, T, d_model*num_dirs)
        return self.pool(y.transpose(1,2)).squeeze(-1)