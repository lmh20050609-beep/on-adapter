"""
hetero_moe_adapter.py
=====================
Truly-heterogeneous MoE adapter for the iAdapter z-space (B, N, D_z).

Drop-in replacement for `OffsetAwareHeteroMoEAdapter` in
iadapter_multimodal_traffic.py.  Same constructor kwargs and the SAME forward
signature, so the online loops (`_compute_text_gradient`, `online_multimodal`)
do not need to change:

    z = adapter.mm_adp(feature, his_grad, text_emb,
                       his_text_grad=..., offset_feat=...)   # -> (B, N, D_z)

WHY THIS FIXES "multimodal doesn't beat unimodal":
--------------------------------------------------
The original experts were all Linear+ReLU MLPs differing only by which inputs
they concatenate.  MLPs cover each other's function space, so gradient descent
collapses them to near-identical solutions and the gate degenerates to a uniform
average (== one bigger MLP, harder to train).  Dynamic-TMoE's experts are
heterogeneous at the *inductive-bias* level (Trend = low-rank/smooth,
Seasonality = Fourier-mode truncation, Fluctuation = high-pass, Identity =
no-op).  Here we reproduce that heterogeneity along the per-variate embedding
axis D_z, so each expert can only fit a restricted family and the gate has a
real division of labour to learn.

Extra robustness knobs vs. the original:
  * per-VARIATE gate (B, N, E)     — finer than one weight per whole sample
  * top-k sparse routing            — forces specialisation, not soft-averaging
  * load-balance aux loss           — keeps experts from dying (self.aux_loss)
  * offset-feature LayerNorm        — the 8 offset dims no longer dominate/vanish
  * shared trunk + thin expert heads — far fewer params than 4 full MLPs
                                       (matters a lot in the online small-batch
                                        regime, which is where the old version
                                        under-fit)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────
# Heterogeneous experts over the per-variate embedding axis D_z
# Each expert takes a shared hidden representation h : (B, N, hidden)
# plus whatever raw signal it specialises on, and returns a z
# correction of shape (B, N, D_z).
# ──────────────────────────────────────────────────────────────

class IdentityExpert(nn.Module):
    """
    'No-correction' / residual stabiliser (mirrors Dynamic-TMoE IdentityExpert).
    Output is initialised near zero so the model can choose to leave z untouched.
    """
    def __init__(self, hidden, d_z):
        super().__init__()
        self.out = nn.Linear(hidden, d_z)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, h, feature, g_h, txt_mod):
        return self.out(h)                                     # (B, N, D_z)


class TrendExpert(nn.Module):
    """
    Smooth / low-DOF correction (mirrors TrendExpert).
    A rank-r bottleneck over D_z keeps the correction low-rank => globally smooth
    across the embedding, the analogue of a polynomial trend.
    """
    def __init__(self, hidden, d_z, rank=2):
        super().__init__()
        self.pre  = nn.Linear(hidden, d_z)
        self.down = nn.Linear(d_z, rank, bias=False)           # squeeze
        self.up   = nn.Linear(rank, d_z, bias=False)           # expand

    def forward(self, h, feature, g_h, txt_mod):
        x = self.pre(h)
        return self.up(self.down(x))                           # rank-r, smooth


class SeasonalityExpert(nn.Module):
    """
    Fourier-mode-truncated correction (mirrors SeasonalityExpert +
    num_fourier_modes).  We rFFT the feature along D_z, keep only the K lowest
    modes, learn a complex gain per kept mode, and iFFT back.  This restricts the
    expert to smooth periodic structure in the embedding axis.
    """
    def __init__(self, hidden, d_z, num_modes=8):
        super().__init__()
        self.d_z = d_z
        self.n_freq = d_z // 2 + 1
        self.num_modes = min(num_modes, self.n_freq)
        # learnable complex gain (real, imag) for the kept low modes
        self.gain = nn.Parameter(torch.zeros(self.num_modes, 2))
        self.mix  = nn.Linear(hidden, d_z)                     # blends in trunk info

    def forward(self, h, feature, g_h, txt_mod):
        # spectral branch on the raw feature (B, N, D_z)
        Xf = torch.fft.rfft(feature, dim=-1)                   # (B, N, n_freq) complex
        gain = torch.view_as_complex(self.gain.contiguous())  # (num_modes,)
        mask = torch.zeros(self.n_freq, dtype=Xf.dtype, device=Xf.device)
        mask[: self.num_modes] = 1.0 + gain                    # keep low modes only
        Xf = Xf * mask                                         # broadcast over B,N
        seasonal = torch.fft.irfft(Xf, n=self.d_z, dim=-1)     # (B, N, D_z)
        return seasonal + self.mix(h)


class FluctuationExpert(nn.Module):
    """
    High-frequency / error-driven correction (mirrors FluctuationExpert).
    Combines a high-pass of the feature (feature minus its embedding mean) with
    the FRESH z-gradient branch g_h, which carries the latest prediction error.
    """
    def __init__(self, hidden, d_z):
        super().__init__()
        self.hp   = nn.Linear(d_z, hidden)
        self.body = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.GELU(),
            nn.Linear(hidden, d_z),
        )

    def forward(self, h, feature, g_h, txt_mod):
        highpass = feature - feature.mean(dim=-1, keepdim=True)   # remove DC
        hp = torch.relu(self.hp(highpass))
        return self.body(torch.cat([hp, g_h], dim=-1))


class TextFiLMExpert(nn.Module):
    """
    Cross-modal expert: text produces a FiLM (gamma, beta) that modulates the
    feature trunk.  This is the ONLY expert with a text pathway, so if text is
    uninformative the gate can simply route away from it (and you'll SEE that in
    self.last_gate) instead of the whole adapter silently ignoring text.
    """
    def __init__(self, hidden, d_z, text_dim):
        super().__init__()
        self.film = nn.Linear(text_dim, hidden * 2)
        self.out  = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(),
                                  nn.Linear(hidden, d_z))

    def forward(self, h, feature, g_h, txt_mod, f_h):
        # f_h : (B, N, hidden) feature-only trunk
        gamma, beta = self.film(txt_mod).unsqueeze(1).chunk(2, dim=-1)  # (B,1,H)
        gamma = torch.tanh(gamma)
        return self.out(f_h * (1.0 + gamma) + beta)


# ──────────────────────────────────────────────────────────────
# The MoE adapter itself
# ──────────────────────────────────────────────────────────────

class HeteroMoEAdapter(nn.Module):
    """
    Heterogeneous MoE fusion adapter.

    Constructor is signature-compatible with OffsetAwareHeteroMoEAdapter:
        HeteroMoEAdapter(z_shape, d_model, text_dim=64, hidden=64,
                         offset_dim=8, num_experts=5, temperature=1.0, top_k=2,
                         num_modes=8, aux_weight=0.01)

    forward(feature, his_grad, text_emb, his_text_grad=None, offset_feat=None)
        -> z : (B, N, D_z)
    Also sets:
        self.last_gate : (B, N, E) detached  — for gate_history logging
        self.aux_loss  : scalar tensor       — add to your training loss (optional)
    """
    def __init__(self, z_shape, d_model, text_dim=64, hidden=64,
                 offset_dim=8, num_experts=5, temperature=1.0,
                 top_k=2, num_modes=8, aux_weight=0.0):
        super().__init__()
        D_z = z_shape[-1]
        self.D_z         = D_z
        self.d_model     = d_model
        self.text_dim    = text_dim
        self.hidden      = hidden
        self.offset_dim  = offset_dim
        self.temperature = temperature
        self.aux_weight  = aux_weight

        # ADAPT-Z style learnable gradient scales (kept identical to original,
        # so nothing upstream that touches .lr / .txt_lr breaks)
        self.lr     = nn.Parameter(torch.ones(z_shape[1:]))        # (N, d_model)
        self.txt_lr = nn.Parameter(torch.ones(text_dim) * 1.0)     # (text_dim,)

        # shared trunk projections
        self.f_proj = nn.Linear(D_z, hidden)
        self.g_proj = nn.Linear(d_model, hidden)
        self.t_proj = nn.Linear(text_dim, hidden)
        self.trunk  = nn.Linear(hidden * 3, hidden)   # fused shared representation

        # heterogeneous experts (Dynamic-TMoE style)
        self.experts = nn.ModuleList([
            IdentityExpert(hidden, D_z),                         # 0
            TrendExpert(hidden, D_z, rank=2),                    # 1
            SeasonalityExpert(hidden, D_z, num_modes=num_modes), # 2
            FluctuationExpert(hidden, D_z),                      # 3
            TextFiLMExpert(hidden, D_z, text_dim),               # 4 (cross-modal)
        ])
        self.num_experts = len(self.experts)
        self.top_k = min(top_k, self.num_experts)

        # ── gate ──  per-variate, offset-aware
        self.offset_norm = nn.LayerNorm(offset_dim)   # <- fixes offset scale issue
        self.gate_f = nn.Linear(D_z, hidden)
        self.gate_g = nn.Linear(d_model, hidden)
        self.gate_t = nn.Linear(text_dim, hidden)
        self.gate_o = nn.Linear(offset_dim, hidden)
        self.gate_net = nn.Sequential(
            nn.Linear(hidden * 4, hidden), nn.ReLU(),
            nn.Linear(hidden, self.num_experts),
        )

        self.last_gate = None
        self.aux_loss  = torch.zeros(())

    def forward(self, feature, his_grad, text_emb,
                his_text_grad=None, offset_feat=None):
        # feature : (B, N, D_z) ; his_grad : (N, d_model) ; text_emb : (B, text_dim)
        B, N, _ = feature.shape
        device  = feature.device

        if offset_feat is None:
            offset_feat = torch.zeros(B, self.offset_dim, device=device)

        # ── gradient branch (ADAPT-Z scaling) ──
        g = (his_grad * self.lr).unsqueeze(0).expand(B, -1, -1)   # (B, N, d_model)
        f_h = torch.relu(self.f_proj(feature))                    # (B, N, hidden)
        g_h = torch.relu(self.g_proj(g))                          # (B, N, hidden)

        # ── text branch (gradient-modulated), used by gate + FiLM expert ──
        if his_text_grad is not None:
            # use the DIRECTION of the text gradient, not raw magnitude, so the
            # tiny txt_lr no longer makes scale ~= 1 (the old no-op problem)
            direction = his_text_grad / (his_text_grad.norm() + 1e-6)
            txt_mod = text_emb * (1.0 + self.txt_lr * direction).unsqueeze(0)
        else:
            txt_mod = text_emb
        txt_exp = txt_mod.unsqueeze(1).expand(-1, N, -1)          # (B, N, text_dim)
        t_h = torch.relu(self.t_proj(txt_exp))                    # (B, N, hidden)

        # ── shared trunk ──
        h = torch.relu(self.trunk(torch.cat([f_h, g_h, t_h], dim=-1)))  # (B,N,hid)

        # ── expert outputs ──  each (B, N, D_z)
        outs = [
            self.experts[0](h, feature, g_h, txt_mod),
            self.experts[1](h, feature, g_h, txt_mod),
            self.experts[2](h, feature, g_h, txt_mod),
            self.experts[3](h, feature, g_h, txt_mod),
            self.experts[4](h, feature, g_h, txt_mod, f_h),   # text FiLM
        ]
        experts = torch.stack(outs, dim=2)                       # (B, N, E, D_z)

        # ── per-variate gate ──
        o_norm = self.offset_norm(offset_feat)                   # (B, offset_dim)
        o_exp  = o_norm.unsqueeze(1).expand(-1, N, -1)           # (B, N, offset_dim)
        gate_h = torch.cat([
            torch.relu(self.gate_f(feature)),
            torch.relu(self.gate_g(g)),
            torch.relu(self.gate_t(txt_exp)),
            torch.relu(self.gate_o(o_exp)),
        ], dim=-1)                                               # (B, N, 4*hidden)
        logits = self.gate_net(gate_h) / max(self.temperature, 1e-6)  # (B, N, E)

        # top-k sparse routing
        if self.top_k < self.num_experts:
            topv, topi = logits.topk(self.top_k, dim=-1)
            mask = torch.full_like(logits, float("-inf"))
            mask.scatter_(-1, topi, topv)
            logits = mask
        weights = torch.softmax(logits, dim=-1)                  # (B, N, E)
        self.last_gate = weights.detach()

        # load-balancing aux loss (Switch-Transformer style):
        # importance = avg gate prob per expert; encourage uniform usage.
        importance = weights.mean(dim=(0, 1))  # (E,)
        # TextFiLMExpert 是 index 4，不参与均衡
        non_text = torch.cat([importance[:4]])
        self.aux_loss = self.aux_weight * 4 * (non_text ** 2).sum()

        # ── combine ──
        z = (experts * weights.unsqueeze(-1)).sum(dim=2)         # (B, N, D_z)
        return z


# Backward-compatible alias so you can literally swap the class name.
OffsetAwareHeteroMoEAdapter = HeteroMoEAdapter
