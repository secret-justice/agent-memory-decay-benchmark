# -*- coding: utf-8 -*-
"""
CCT-LNN: Category-Conditioned Tau Liquid Neural Network
+ All 12 baselines for E1-E4 experiments.

数学定义:
  标准 LTC: dh/dt = (-h + activation(A·h + B·x)) / τ
  CCT-LNN:  dh/dt = (-h + activation(A·h + B·x + C·e_c)) / τ_c
  其中 τ_c = τ_base · σ(w_x·x + w_c·e_c + b)
"""
import torch
import torch.nn as nn
import math

INPUT_DIM = 64  # PersonaChat embedding dimension

# ============================================================
# CCT-LNN Cell (新方法)
# ============================================================
class CCT_LNNCell(nn.Module):
    """Category-Conditioned Tau Liquid Neural Network Cell."""
    
    def __init__(self, inp, hd, n_categories=10, tau_max=10.0):
        super().__init__()
        self.hd = hd
        self.tau_max = tau_max
        
        # Standard LTC weights
        self.W_in = nn.Linear(inp, hd)
        self.W_rec = nn.Linear(hd, hd, bias=False)
        
        # Category conditioning (NEW)
        self.cat_embedding = nn.Embedding(n_categories, hd)
        self.W_cat = nn.Linear(hd, hd, bias=False)
        
        # Tau computation with category conditioning
        self.A = nn.Linear(inp, hd)  # input contribution to tau
        self.C_tau = nn.Linear(hd, hd, bias=False)  # category contribution to tau
        self.B = nn.Linear(hd, hd, bias=False)  # hidden contribution to tau
        self.tau_bias = nn.Parameter(torch.zeros(hd))
        
        self._init_weights()
    
    def _init_weights(self):
        for m in [self.W_in, self.W_rec, self.A, self.B, self.W_cat, self.C_tau]:
            nn.init.xavier_normal_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward_seq(self, X, ts, categories=None):
        """
        Args:
            X: [seq_len, inp_dim] - input embeddings
            ts: [seq_len] - timestamps
            categories: [seq_len] - category indices (LongTensor)
        Returns:
            H: [seq_len, hd] - hidden states
            tau_log: [seq_len, hd] - tau values
        """
        n = X.shape[0]
        h = torch.zeros(self.hd, device=X.device)
        H = []
        tau_log = []
        
        for i in range(n):
            dt = max(float(ts[i] - ts[i-1]) if i > 0 else 1.0, 0.01)
            
            # Category embedding
            if categories is not None:
                e_c = self.cat_embedding(categories[i])
            else:
                e_c = torch.zeros(self.hd, device=X.device)
            
            # CCT tau: τ_c = τ_base · σ(w_x·x + w_c·e_c + w_h·h + b)
            tau = torch.sigmoid(
                self.A(X[i]) + self.C_tau(e_c) + self.B(h) + self.tau_bias
            ) * self.tau_max
            
            # ODE step with category-conditioned input
            inp = torch.tanh(self.W_in(X[i]) + self.W_rec(h) + self.W_cat(e_c))
            h = torch.clamp(h + (-h / tau + inp) * dt, -5, 5)
            
            H.append(h)
            tau_log.append(tau)
        
        return torch.stack(H), torch.stack(tau_log)


# ============================================================
# Standard LTC Cell (baseline)
# ============================================================
class LTCCell(nn.Module):
    def __init__(self, inp, hd, tau_max=10.0):
        super().__init__()
        self.hd = hd; self.tau_max = tau_max
        self.W_in = nn.Linear(inp, hd); self.W_rec = nn.Linear(hd, hd, bias=False)
        self.A = nn.Linear(inp, hd); self.B = nn.Linear(hd, hd, bias=False)
        self.C = nn.Parameter(torch.ones(hd) * 0.5)
        for m in [self.W_in, self.W_rec, self.A, self.B]:
            nn.init.xavier_normal_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None: nn.init.zeros_(m.bias)
    
    def forward_seq(self, X, ts, categories=None):
        n = X.shape[0]; h = torch.zeros(self.hd, device=X.device); H = []; tau_log = []
        for i in range(n):
            dt = max(float(ts[i] - ts[i-1]) if i > 0 else 1.0, 0.01)
            tau = torch.sigmoid(self.A(X[i]) + self.B(h) + self.C) * self.tau_max
            inp = torch.tanh(self.W_in(X[i]) + self.W_rec(h))
            h = torch.clamp(h + (-h / tau + inp) * dt, -5, 5)
            H.append(h); tau_log.append(tau)
        return torch.stack(H), torch.stack(tau_log)


# ============================================================
# Fixed-tau LTC (ablation baseline)
# ============================================================
class LTCFixedCell(nn.Module):
    def __init__(self, inp, hd):
        super().__init__(); self.hd = hd
        self.W_in = nn.Linear(inp, hd); self.W_rec = nn.Linear(hd, hd, bias=False)
        self.tau = nn.Parameter(torch.ones(hd) * 5.0)
        for m in [self.W_in, self.W_rec]:
            nn.init.xavier_normal_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None: nn.init.zeros_(m.bias)
    
    def forward_seq(self, X, ts, categories=None):
        n = X.shape[0]; h = torch.zeros(self.hd, device=X.device); H = []
        for i in range(n):
            dt = max(float(ts[i] - ts[i-1]) if i > 0 else 1.0, 0.01)
            inp = torch.tanh(self.W_in(X[i]) + self.W_rec(h))
            tau = torch.clamp(torch.abs(self.tau), 0.1, 20.0)
            h = torch.clamp(h + (-h / tau + inp) * dt, -5, 5); H.append(h)
        return torch.stack(H), self.tau.detach().unsqueeze(0).expand(n, -1)


# ============================================================
# GRU (discrete baseline)
# ============================================================
class GRUCell(nn.Module):
    def __init__(self, inp, hd):
        super().__init__(); self.hd = hd; self.gru = nn.GRUCell(inp, hd)
    
    def forward_seq(self, X, ts, categories=None):
        n = X.shape[0]; h = torch.zeros(self.hd, device=X.device); H = []
        for i in range(n):
            h = self.gru(X[i].unsqueeze(0), h.unsqueeze(0)).squeeze(0)
            h = torch.clamp(h, -5, 5); H.append(h)
        return torch.stack(H), None


# ============================================================
# GRU+Time (discrete + time encoding)
# ============================================================
class GRUTimeCell(nn.Module):
    def __init__(self, inp, hd):
        super().__init__(); self.hd = hd
        self.time_enc = nn.Linear(1, inp)  # time encoding
        self.gru = nn.GRUCell(inp, hd)
    
    def forward_seq(self, X, ts, categories=None):
        n = X.shape[0]; h = torch.zeros(self.hd, device=X.device); H = []
        for i in range(n):
            dt = float(ts[i] - ts[i-1]) if i > 0 else 1.0
            t_enc = self.time_enc(torch.tensor([dt], device=X.device))
            h = self.gru((X[i] + t_enc).unsqueeze(0), h.unsqueeze(0)).squeeze(0)
            h = torch.clamp(h, -5, 5); H.append(h)
        return torch.stack(H), None


# ============================================================
# Neural ODE (generic continuous-time baseline)
# ============================================================
class NeuralODECell(nn.Module):
    def __init__(self, inp, hd):
        super().__init__(); self.hd = hd
        self.W_in = nn.Linear(inp, hd)
        self.ode_net = nn.Sequential(
            nn.Linear(hd, hd * 2), nn.Tanh(),
            nn.Linear(hd * 2, hd)
        )
        self.dt_scale = nn.Parameter(torch.tensor(0.1))
    
    def forward_seq(self, X, ts, categories=None):
        n = X.shape[0]; h = torch.zeros(self.hd, device=X.device); H = []
        for i in range(n):
            dt = max(float(ts[i] - ts[i-1]) if i > 0 else 1.0, 0.01)
            dx = self.W_in(X[i])
            dh = self.ode_net(h + dx)
            h = torch.clamp(h + dh * dt * self.dt_scale.abs(), -5, 5)
            H.append(h)
        return torch.stack(H), None


# ============================================================
# CfC (Closed-form Continuous-time)
# ============================================================
class CfCCell(nn.Module):
    def __init__(self, inp, hd, tau_max=10.0):
        super().__init__(); self.hd = hd; self.tau_max = tau_max
        self.W_in = nn.Linear(inp, hd); self.W_rec = nn.Linear(hd, hd, bias=False)
        self.A = nn.Linear(inp, hd); self.B = nn.Linear(hd, hd, bias=False)
        self.C = nn.Parameter(torch.ones(hd) * 0.5)
        for m in [self.W_in, self.W_rec, self.A, self.B]:
            nn.init.xavier_normal_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None: nn.init.zeros_(m.bias)
    
    def forward_seq(self, X, ts, categories=None):
        n = X.shape[0]; h = torch.zeros(self.hd, device=X.device); H = []
        for i in range(n):
            dt = max(float(ts[i] - ts[i-1]) if i > 0 else 1.0, 0.01)
            tau = torch.sigmoid(self.A(X[i]) + self.B(h) + self.C) * self.tau_max
            inp = torch.tanh(self.W_in(X[i]) + self.W_rec(h))
            # Closed-form: h_new = h*exp(-dt/tau) + inp*tau*(1-exp(-dt/tau))
            alpha = torch.exp(-dt / tau)
            h = torch.clamp(h * alpha + inp * tau * (1 - alpha), -5, 5)
            H.append(h)
        return torch.stack(H), tau.unsqueeze(0).expand(n, -1).detach()


# ============================================================
# ExpDecay (1-param baseline)
# ============================================================
class ExpDecayCell(nn.Module):
    def __init__(self, inp, hd):
        super().__init__(); self.hd = hd
        self.decay_rate = nn.Parameter(torch.tensor(0.1))
        self.W_in = nn.Linear(inp, hd)
    
    def forward_seq(self, X, ts, categories=None):
        n = X.shape[0]; h = torch.zeros(self.hd, device=X.device); H = []
        for i in range(n):
            dt = max(float(ts[i] - ts[i-1]) if i > 0 else 1.0, 0.01)
            alpha = torch.exp(-self.decay_rate.abs() * dt)
            h = torch.clamp(h * alpha + self.W_in(X[i]), -5, 5)
            H.append(h)
        return torch.stack(H), None


# ============================================================
# Parametric ExpDecay (per-category)
# ============================================================
class ParametricExpDecayCell(nn.Module):
    def __init__(self, inp, hd, n_categories=10):
        super().__init__(); self.hd = hd
        self.decay_rates = nn.Parameter(torch.ones(n_categories) * 0.1)
        self.W_in = nn.Linear(inp, hd)
    
    def forward_seq(self, X, ts, categories=None):
        n = X.shape[0]; h = torch.zeros(self.hd, device=X.device); H = []
        for i in range(n):
            dt = max(float(ts[i] - ts[i-1]) if i > 0 else 1.0, 0.01)
            cat_idx = int(categories[i]) if categories is not None else 0
            rate = self.decay_rates[cat_idx].abs()
            alpha = torch.exp(-rate * dt)
            h = torch.clamp(h * alpha + self.W_in(X[i]), -5, 5)
            H.append(h)
        return torch.stack(H), None


# ============================================================
# Simple Transformer baseline
# ============================================================
class SimpleTransformerCell(nn.Module):
    def __init__(self, inp, hd, nhead=4):
        super().__init__(); self.hd = hd
        self.W_in = nn.Linear(inp, hd)
        self.attn = nn.MultiheadAttention(hd, nhead, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(hd, hd * 2), nn.GELU(), nn.Linear(hd * 2, hd))
        self.norm1 = nn.LayerNorm(hd)
        self.norm2 = nn.LayerNorm(hd)
    
    def forward_seq(self, X, ts, categories=None):
        # Process entire sequence at once (attention)
        x = self.W_in(X).unsqueeze(0)  # [1, seq, hd]
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x.squeeze(0), None


# ============================================================
# SSM (S4-style state space model)
# ============================================================
class SSMCell(nn.Module):
    def __init__(self, inp, hd):
        super().__init__(); self.hd = hd
        self.W_in = nn.Linear(inp, hd)
        self.A = nn.Parameter(torch.randn(hd, hd) * 0.01)
        self.B = nn.Linear(inp, hd, bias=False)
        self.C = nn.Linear(hd, hd, bias=False)
    
    def forward_seq(self, X, ts, categories=None):
        n = X.shape[0]; h = torch.zeros(self.hd, device=X.device); H = []
        for i in range(n):
            dt = max(float(ts[i] - ts[i-1]) if i > 0 else 1.0, 0.01)
            # Discretize: A_d = exp(A*dt) ≈ I + A*dt
            h = torch.clamp(h + dt * (self.A @ h + self.B(X[i]).squeeze()), -5, 5)
            H.append(self.C(h))
        return torch.stack(H), None


# ============================================================
# RetNet (retention network)
# ============================================================
class RetNetCell(nn.Module):
    def __init__(self, inp, hd):
        super().__init__(); self.hd = hd
        self.W_in = nn.Linear(inp, hd)
        self.W_q = nn.Linear(hd, hd, bias=False)
        self.W_k = nn.Linear(hd, hd, bias=False)
        self.W_v = nn.Linear(hd, hd, bias=False)
        self.decay = nn.Parameter(torch.tensor(0.9))
        self.out = nn.Linear(hd, hd)
    
    def forward_seq(self, X, ts, categories=None):
        n = X.shape[0]; state = torch.zeros(self.hd, self.hd, device=X.device); H = []
        gamma = self.decay.abs().clamp(0.01, 0.99)
        for i in range(n):
            x = self.W_in(X[i])
            q = self.W_q(x); k = self.W_k(x); v = self.W_v(x)
            state = gamma * state + torch.outer(v, k)
            h = torch.clamp(self.out(state @ q), -5, 5)
            H.append(h)
        return torch.stack(H), None


# ============================================================
# xLSTM (extended LSTM)
# ============================================================
class xLSTMCell(nn.Module):
    def __init__(self, inp, hd):
        super().__init__(); self.hd = hd
        self.lstm = nn.LSTMCell(inp, hd)
    
    def forward_seq(self, X, ts, categories=None):
        n = X.shape[0]
        h = torch.zeros(self.hd, device=X.device)
        c = torch.zeros(self.hd, device=X.device)
        H = []
        for i in range(n):
            h, c = self.lstm(X[i].unsqueeze(0), (h.unsqueeze(0), c.unsqueeze(0)))
            h = h.squeeze(0); c = c.squeeze(0)
            h = torch.clamp(h, -5, 5); H.append(h)
        return torch.stack(H), None


# ============================================================
# Universal Retriever wrapper
# ============================================================
class Retriever(nn.Module):
    def __init__(self, dim, hd, variant='ltc', n_categories=10):
        super().__init__()
        self.variant = variant
        
        if variant == 'cct_lnn':
            self.cell = CCT_LNNCell(dim, hd, n_categories)
        elif variant == 'ltc':
            self.cell = LTCCell(dim, hd)
        elif variant == 'ltc_fixed':
            self.cell = LTCFixedCell(dim, hd)
        elif variant == 'gru':
            self.cell = GRUCell(dim, hd)
        elif variant == 'gru_time':
            self.cell = GRUTimeCell(dim, hd)
        elif variant == 'neural_ode':
            self.cell = NeuralODECell(dim, hd)
        elif variant == 'cfc':
            self.cell = CfCCell(dim, hd)
        elif variant == 'expdecay':
            self.cell = ExpDecayCell(dim, hd)
        elif variant == 'param_expdecay':
            self.cell = ParametricExpDecayCell(dim, hd, n_categories)
        elif variant == 'transformer':
            self.cell = SimpleTransformerCell(dim, hd)
        elif variant == 'ssm':
            self.cell = SSMCell(dim, hd)
        elif variant == 'retnet':
            self.cell = RetNetCell(dim, hd)
        elif variant == 'xlstm':
            self.cell = xLSTMCell(dim, hd)
        else:
            raise ValueError(f"Unknown variant: {variant}")
        
        self.W_q = nn.Linear(dim, hd)
        self.W_out = nn.Linear(hd, 1)  # keep/forget prediction (binary)
    
    def forward(self, X, ts, categories=None):
        H, tau_log = self.cell.forward_seq(X, ts, categories)
        h_final = H[-1]
        q = self.W_q(X[-1])
        score = torch.sigmoid(self.W_out(h_final * q))
        return score, H, tau_log
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters())
