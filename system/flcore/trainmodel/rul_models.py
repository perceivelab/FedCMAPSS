from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# 1) Zhu et al. (2025): AFT-Conv2D regressor
#    Collaborative Prognostics of Lithium-Ion Batteries Using Federated Learning With Dynamic Weighting and Attention Mechanism
# -----------------------------
class AFTBlock(nn.Module):
    def __init__(self, input_size=25, window_size=30):
        super().__init__()
        self.D, self.T = input_size, window_size
        self.q = nn.Linear(input_size, input_size)
        self.k = nn.Linear(input_size, input_size)
        self.v = nn.Linear(input_size, input_size)
        self.pos = nn.Parameter(torch.zeros(window_size, window_size))  # positional bias

    def forward(self, x):
        # x: [B,T,D]
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)
        rQ = torch.sigmoid(Q)

        # w[b,t,t',d] = exp(K[b,t',d] + pos[t,t'])
        w = torch.exp(K.unsqueeze(1) + self.pos.unsqueeze(0).unsqueeze(-1))  # [B,T,T,D]
        context = (w * V.unsqueeze(1)).sum(dim=2) / (w.sum(dim=2) + 1e-8)     # [B,T,D]

        return rQ * context  # [B,T,D]


class AFTConv2D(nn.Module):
    def __init__(self, window_size=30, input_size=25):
        super().__init__()
        self.aft = AFTBlock(window_size=window_size, input_size=input_size)

        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Sequential(
            nn.Linear(32 * 4 * 4, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.aft(x)
        x = x.transpose(1, 2).unsqueeze(1)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).flatten(1)
        return self.fc(x).squeeze(-1)
    

# -----------------------------
# 2) Chen et al. (2023): Chen_CNN_RUL
#    Federated Learning with Network Pruning and Rebirth for Remaining Useful Life Prediction of Engineering Systems
# -----------------------------
class Chen_CNN_RUL(nn.Module):
    """
    Implemantion details within the paper:
      Conv1..Conv5: kernel 10x1
      Conv6: kernel 3x1
      tanh activation
      AdaptiveAvgPool2d
      FC: 350 -> 120 -> 1 (dropout in FC)

    NOTE: conv_channels is NOT specified
    """
    def __init__(self, input_size, window_size, conv_channels, fc_dropout=0.0):
        super().__init__()
        assert len(conv_channels) == 6

        kernels = [(10, 1)] * 5 + [(3, 1)]
        self.convs = nn.ModuleList()
        prev = 1
        for out_ch, k in zip(conv_channels, kernels):
            self.convs.append(nn.Conv2d(prev, out_ch, kernel_size=k, stride=1, padding=0))
            prev = out_ch

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        with torch.no_grad():
            dummy = torch.zeros(1, 1, window_size, input_size)
            z = dummy
            for conv in self.convs:
                z = self._conv_time_same(z, conv)
            flat_dim = self.pool(z).flatten(1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(flat_dim, 350), nn.Tanh(), nn.Dropout(fc_dropout),
            nn.Linear(350, 120), nn.Tanh(), nn.Dropout(fc_dropout),
            nn.Linear(120, 1),
        )

    @staticmethod
    def _conv_time_same(x: torch.Tensor, conv: nn.Conv2d) -> torch.Tensor:
        kH, kW = conv.kernel_size  # kW è 1
        pad_total_h = kH - 1
        pad_top = pad_total_h // 2
        pad_bottom = pad_total_h - pad_top
        
        x = F.pad(x, (0, 0, pad_top, pad_bottom))
        return torch.tanh(conv(x))

    def forward(self, x):
        # x: [B, window_size, input_size]
        x = x.unsqueeze(1)  # [B,1,H,W]
        for conv in self.convs:
            x = self._conv_time_same(x, conv)
        x = self.pool(x).flatten(1)
        return self.fc(x).squeeze(-1)
    

# -----------------------------
# 3) Qin et al. (2023): AttBiGRU
#    Dynamic weighted federated remaining useful life prediction approach for rotating machinery
# -----------------------------
class AttBiGRU(nn.Module):
    """
    Att-BiGRU:
      input window -> BiGRU -> MultiHeadSelfAttention -> flatten -> FC -> RUL

    Paper hints:
      - BiGRU hidden = 32
      - regressor input size = 32*2*Ws (= 64*Ws), hidden=32, output=1
      - dropout1=0.3, dropout2=0.1
      - Ws=10 (sliding window)
    """
    def __init__(
        self,
        input_size: int = 64,
        ws: int = 10,
        gru_hidden: int = 32,
        attn_heads: int = 4,
        dropout1: float = 0.3,
        dropout2: float = 0.1,
    ):
        super().__init__()
        self.ws = ws
        self.feature_dim = input_size
        self.gru_hidden = gru_hidden
        self.embed_dim = 2 * gru_hidden  # bidirectional

        self.bigru = nn.GRU(
            input_size=input_size,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=attn_heads,
            dropout=0.0,
            batch_first=True,
        )

        self.fc1 = nn.Linear(self.embed_dim * ws, 32)
        self.act1 = nn.PReLU()
        self.drop1 = nn.Dropout(dropout1)

        self.fc2 = nn.Linear(32, 1)
        self.drop2 = nn.Dropout(dropout2)

    def forward(self, x_win):
        """
        x_win: (B, Ws, feature_dim)
        """
        H, _ = self.bigru(x_win)      # [B, Ws, 2*gru_hidden] = [B, Ws, 64]
        A, _ = self.attn(H, H, H)
        Fflat = A.reshape(A.size(0), -1)

        y = self.fc1(Fflat)
        y = self.act1(y)
        y = self.drop1(y)

        y = self.fc2(y)
        y = self.drop2(y)

        return y.squeeze(-1)
