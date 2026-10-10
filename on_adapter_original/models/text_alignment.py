"""
TS–text alignment module — extracted from adapt_z_multimodal.py (SECTION 6b).

  · VICRegAlign              : single pooled alignment (the original file already described itself as a "simplified VoT
                                Endogenous Text Alignment": no trend/seasonal
                                decomposition contrast; a single VICReg alignment directly on the pooled TS representation
                                and the text representation). Used by iAdapter/GPT4MTS_z.
  · MultiComponentVICRegAlign : the "full" Endogenous Text Alignment of VoT_z —
                                one VICRegAlign each between the trend-component representation and the text representation and between
                                the seasonal-component representation and the text representation, summed into one total alignment loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _dominant_period(x):
    """x: (B,L,N) standardized input window -> dominant period length of each sample (amplitude spectrum averaged across channels;
    after removing DC, take the frequency component with the largest amplitude and convert it to a period length L/k)."""
    xf = torch.fft.rfft(x - x.mean(dim=1, keepdim=True), dim=1)
    mag = xf.abs().mean(dim=-1)                  # (B, L//2+1)
    mag = mag.clone()
    mag[:, 0] = 0.0                               # remove DC
    k = mag.argmax(dim=1).clamp(min=1)
    return x.size(1) / k.float()


def _jump_score(x):
    """x: (B,L,N) -> maximum absolute first difference / standard deviation, per channel, then averaged.
    The larger the value, the more the window looks like it contains abrupt changes/jumps rather than smooth periodic fluctuation."""
    d = x.diff(dim=1)
    return (d.abs().amax(dim=1) / (x.std(dim=1) + 1e-6)).mean(dim=-1)


class VICRegAlign(nn.Module):
    """VICReg-style (Bardes et al. 2022) TS–text alignment module, replacing the original CLIPAlign
    (CLIP-style symmetric InfoNCE).

    When all negatives are similar (frozen LLM pooled vectors are generally highly clustered), the InfoNCE gradient
    tends to vanish, and the alignment exists in name only. VICReg does not rely on contrast with negatives; three loss terms directly
    prevent representation collapse:
      · invariance: MSE(ts_e, text_e) pulls paired samples together;
      · variance:   forces std > gamma per dimension, directly preventing collapse;
      · covariance: decorrelation, preventing all dimensions from carrying the same information.
    The projection head uses BatchNorm1d (normalization across the batch) instead of LayerNorm (per-sample normalization),
    actively pushing the clustered LLM pooled vectors apart along the batch dimension.
    """

    def __init__(self, d_z, text_dim, d_clip=64,
                 lam=25.0, mu=25.0, nu=1.0, gamma=1.0):
        super().__init__()
        self.lam, self.mu, self.nu, self.gamma = lam, mu, nu, gamma

        # Deeper projection head + BN instead of LayerNorm
        # BN normalizes across the batch → prevents all vectors from collapsing to one point
        self.ts_proj = nn.Sequential(
            nn.Linear(d_z, d_clip * 2),
            nn.BatchNorm1d(d_clip * 2),
            nn.ReLU(),
            nn.Linear(d_clip * 2, d_clip),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, d_clip * 2),
            nn.BatchNorm1d(d_clip * 2),
            nn.ReLU(),
            nn.Linear(d_clip * 2, d_clip),
        )

    def embed(self, feature, text_emb):
        """feature: (B,N,D_z) TS representation; text_emb: (B,text_dim) text representation
        (not multiplied by confidence) -> (ts_e, text_e), both (B,d_clip)."""
        ts_e = self.ts_proj(feature.mean(dim=1))     # (B, d_clip)
        text_e = self.text_proj(text_emb)            # (B, d_clip)
        return ts_e, text_e

    @staticmethod
    def _off_diag(m):
        n = m.size(0)
        return m.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def _var_loss(self, z):
        std = z.std(dim=0)
        return F.relu(self.gamma - std).mean()

    def _cov_loss(self, z):
        n, d = z.shape
        z = z - z.mean(dim=0)
        cov = (z.T @ z) / max(n - 1, 1)
        return self._off_diag(cov).pow(2).sum() / d

    def forward(self, feature, text_emb, conf=None, dup_mask=None):
        """conf: (B,1) text confidence; 0 marks an uninformative placeholder text -> per-sample down-weighting,
        so "fake text" is not forcibly aligned. The dup_mask argument is kept for compatibility with the original call signature
        (VICReg does no contrast with negatives, so it is not needed to remove duplicate positives). When B<2 the variance/
        covariance terms are meaningless, and 0 is returned.
        """
        B = feature.size(0)
        if B < 2:
            return feature.new_zeros(())
        ts_e, text_e = self.embed(feature, text_emb)

        # conf weighting: low-confidence samples are not forcibly aligned
        w = (conf.squeeze(-1) if conf is not None
             else torch.ones(B, device=ts_e.device))
        wsum = w.sum().clamp_min(1e-6)

        inv = ((ts_e - text_e).pow(2).mean(-1) * w).sum() / wsum
        var = self._var_loss(ts_e) + self._var_loss(text_e)
        cov = self._cov_loss(ts_e) + self._cov_loss(text_e)

        return self.lam * inv + self.mu * var + self.nu * cov


class MultiComponentVICRegAlign(nn.Module):
    """The "full" Endogenous Text Alignment used by VoT_z: one VICRegAlign each between the trend-component representation
    and the text representation and between the seasonal-component representation and the text representation, instead of a single
    alignment on one pooled representation — so the text information calibrates both subspaces, "slowly varying trend semantics" and "periodic fluctuation semantics",
    at the same time; a direct extension of VICRegAlign (the simplified version).

    trend_weight/seasonal_weight set the relative weights of the two components' alignment losses; default
    equal weights (1.0, 1.0).
    """

    def __init__(self, d_z, text_dim, d_clip=64,
                 lam=25.0, mu=25.0, nu=1.0, gamma=1.0,
                 trend_weight=1.0, seasonal_weight=1.0):
        super().__init__()
        self.trend_align = VICRegAlign(d_z, text_dim, d_clip, lam, mu, nu, gamma)
        self.seasonal_align = VICRegAlign(d_z, text_dim, d_clip, lam, mu, nu, gamma)
        self.trend_weight = trend_weight
        self.seasonal_weight = seasonal_weight

    def forward(self, trend_feature, seasonal_feature, text_emb, conf=None):
        """trend_feature/seasonal_feature: (B,N,D_z); text_emb: (B,text_dim).
        Both components share the same text representation (VoT paper semantics: endogenous text constrains both the
        trend and periodic subspaces at the representation level); each goes through its own projection head and computes the three VICReg loss terms once."""
        l_trend = self.trend_align(trend_feature, text_emb, conf)
        l_seasonal = self.seasonal_align(seasonal_feature, text_emb, conf)
        return self.trend_weight * l_trend + self.seasonal_weight * l_seasonal
