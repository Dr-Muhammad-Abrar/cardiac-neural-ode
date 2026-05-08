# =============================================================================
# src/model.py  —  Cardiac Latent Neural ODE + BiLSTM
# =============================================================================

import torch
import torch.nn as nn
from torchdiffeq import odeint

from config import LATENT_DIM, HIDDEN_DIM, LSTM_H, N_CLASSES


class ODEFunc(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim,     hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim,     latent_dim),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, t, z):
        t_vec = t.expand(z.shape[0], 1) if z.dim() > 1 else t.unsqueeze(0)
        return self.net(torch.cat([z, t_vec], dim=-1))


class CardiacEncoder(nn.Module):
    def __init__(self, input_dim=3, latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.rnn    = nn.GRU(input_dim, hidden_dim, batch_first=True,
                             bidirectional=True, num_layers=2, dropout=0.2)
        self.mu     = nn.Linear(hidden_dim * 2, latent_dim)
        self.logvar = nn.Linear(hidden_dim * 2, latent_dim)

    def forward(self, x):
        _, h = self.rnn(x)
        h    = torch.cat([h[-2], h[-1]], dim=-1)
        return self.mu(h), self.logvar(h)

    def reparam(self, mu, lv):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * lv)


class CardiacNeuralODE(nn.Module):
    def __init__(self, input_dim=3, latent_dim=LATENT_DIM,
                 hidden_dim=HIDDEN_DIM, n_classes=N_CLASSES, lstm_h=LSTM_H):
        super().__init__()
        self.encoder  = CardiacEncoder(input_dim, latent_dim, hidden_dim)
        self.ode_func = ODEFunc(latent_dim, hidden_dim)
        self.lstm     = nn.LSTM(latent_dim, lstm_h, batch_first=True,
                                bidirectional=True, num_layers=2, dropout=0.2)
        self.head     = nn.Sequential(
            nn.LayerNorm(lstm_h * 2),
            nn.Linear(lstm_h * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x, t_span):
        mu, lv      = self.encoder(x)
        z0          = self.encoder.reparam(mu, lv)
        z_traj      = odeint(self.ode_func, z0, t_span,
                             method='euler', options={'step_size': 0.05})
        z_seq       = z_traj.permute(1, 0, 2)
        _, (h, _)   = self.lstm(z_seq)
        h           = torch.cat([h[-2], h[-1]], dim=-1)
        return self.head(h), mu, lv

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
