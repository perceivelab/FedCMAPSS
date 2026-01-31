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


# -----------------------------
# 4) Barbosa et al. (2025): LSTM_v2_RUL
#    Using Federated Machine Learning in Predictive Maintenance of Jet Engines
# -----------------------------
class GaussianNoise(nn.Module):
    def __init__(self, std: float = 0.01):
        super().__init__()
        self.std = std

    def forward(self, x):
        if self.training and self.std > 0:
            return x + torch.randn_like(x) * self.std
        return x

class VariationalDropout(nn.Module):
    """
    Dropout with a time-locked (variational) mask.
    The same dropout mask is reused across all time steps.
    Used here to approximate "recurrent dropout" in an LSTMCell loop.
    """
    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def forward(self, h):
        if (not self.training) or self.p <= 0:
            return h
        keep = 1.0 - self.p
        mask = torch.empty_like(h).bernoulli_(keep) / keep
        return h * mask

class StackedLSTMCell(nn.Module):
    """
    Implements a multi-layer LSTM using LSTMCell, with recurrent dropout applied to h_t.
    Input:  x of shape (B, T, D)
    Output: last_h of shape (B, H), i.e., the hidden state of the last layer at the final time step.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 4,
                 layer_dropout: float = 0.1, recurrent_dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.cells.append(nn.LSTMCell(in_dim, hidden_dim))

        self.inter_layer_dropout = nn.Dropout(layer_dropout)
        self.rec_dropout = VariationalDropout(recurrent_dropout)

    def forward(self, x):
        B, T, D = x.shape

        h = [x.new_zeros(B, self.hidden_dim) for _ in range(self.num_layers)]
        c = [x.new_zeros(B, self.hidden_dim) for _ in range(self.num_layers)]

        for t in range(T):
            inp = x[:, t, :]
            for l, cell in enumerate(self.cells):
                h[l], c[l] = cell(inp, (h[l], c[l]))
                h[l] = self.rec_dropout(h[l])
                inp = h[l] if l == self.num_layers - 1 else self.inter_layer_dropout(h[l])

        return h[-1]  # (B, H)

class LSTM_v2_RUL(nn.Module):
    """
    RUL model: Gaussian noise + stacked LSTM + dense stack + output.
    Predicts a single RUL value per input window (many-to-one).
    """
    def __init__(
        self,
        input_size: int,
        lstm_layers: int = 4,
        dense_layers: int = 4,
        units: int = 64,
        layer_dropout: float = 0.1,
        recurrent_dropout: float = 0.2,
        gaussian_noise: float = 0.01,
    ):
        super().__init__()
        self.noise = GaussianNoise(std=gaussian_noise)

        self.rnn = StackedLSTMCell(
            input_dim=input_size,
            hidden_dim=units,
            num_layers=lstm_layers,
            layer_dropout=layer_dropout,
            recurrent_dropout=recurrent_dropout,
        )

        mlp = []
        in_dim = units
        for _ in range(dense_layers):
            mlp += [
                nn.Linear(in_dim, units),
                nn.ReLU(),
                nn.Dropout(layer_dropout),
            ]
            in_dim = units
        mlp += [nn.Linear(in_dim, 1)]
        self.mlp = nn.Sequential(*mlp)

    def forward(self, x):
        """
        x: (B, T, D)  with T = ws/sequence length
        """
        x = self.noise(x)
        h_last = self.rnn(x)
        y = self.mlp(h_last).squeeze(-1)
        return y


# -----------------------------
# 5) Chen et al. (2023): RNN_RUL
#    A remaining useful life estimation method based on long short-term memory
#    and federated learning for electric vehicles in smart cities
# -----------------------------
class RNN_RUL(nn.Module):
    """
    Chen et al. (2023) RNN model:
      SimpleRNN layers with units: 64 -> 32 -> 16 -> 8 -> 4 (ReLU),
      then 2 fully-connected layers -> output 1 (RUL).

    Input:  x (B, T, 16)  where T=100 in the paper setup (sliding window size).
    Output: y (B, 1)
    """
    def __init__(self, input_size: int = 16, fc_hidden: int = 40):
        super().__init__()

        self.rnn1 = nn.RNN(input_size, 64, nonlinearity="relu", batch_first=True)
        self.rnn2 = nn.RNN(64, 32, nonlinearity="relu", batch_first=True)
        self.rnn3 = nn.RNN(32, 16, nonlinearity="relu", batch_first=True)
        self.rnn4 = nn.RNN(16, 8, nonlinearity="relu", batch_first=True)
        self.rnn5 = nn.RNN(8, 4, nonlinearity="relu", batch_first=True)

        self.fc1 = nn.Linear(4, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, 1)

    def forward(self, x):
        x, _ = self.rnn1(x)
        x, _ = self.rnn2(x)
        x, _ = self.rnn3(x)
        x, _ = self.rnn4(x)
        x, _ = self.rnn5(x)

        h_last = x[:, -1, :]

        y = F.relu(self.fc1(h_last))
        y = self.fc2(y)
        return y