"""Definitions extracted from outer source; no original module imports or top-level experiment calls.
Source: /home/wuwa/projects/research/adaptz_paper_work/ADAPT-Z-main/adapt_z_multimodal.py
"""
import re
import torch
from torch import nn

_DATE_SEG = re.compile(r"(\d{4}-\d{2}-\d{2})\s*:\s*")

_DROP_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r"^\s*(?:na|n/?a)\b",
    r"\bno (?:objective facts?|relevant|direct|specific|useful) ",
    r"\bthere (?:is|are|was|were) no\b",
    r"\bno (?:predictions?|information|basis|data)\b",
    r"\bit is impossible to make\b",
    r"\bunable to (?:extract|make|provide|glean)\b",
    r"\bcould(?: not|n't) (?:provide|extract|be gleaned)\b",
    r"\bsearch results (?:do not|did not|don't|didn't|seem to be|appear to be|are (?:mostly )?unrelated|primarily focused|were not relevant)\b",
    r"\bresults (?:are|appear) (?:mostly )?(?:unrelated|irrelevant)\b",
    r"\bunfortunately\b",
    r"\bno useful information\b",
    r"\bnot relevant to (?:making )?predict",
    r"^\s*(?:long|short)[- ]?term predictions?\s*:?\s*$",
    r"^\s*objective facts about [^:]{0,120}:?\s*$",
    r"^\s*predictions you can make for the (?:long|short)[- ]?term future[^:]*:?\s*$",
]]

_STRIP_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r"here is (?:the |my )?[^:;]{0,140}:",
    r"note\s*:[^;]*",
    r"after reviewing the search results[^:;,.]*[,:.]?",
    r"based on the (?:provided )?search results[^:;,.]*[,:.]?",
    r"since (?:the search results|there is no relevant information)[^;]*",
    r"the (?:other|majority of the|provided|first) [^;]{0,60}(?:results?|title)[^;]{0,120}(?:relevant|related|irrelevant|discarded|summary)[^;]*",
    r"therefore,? (?:the predictions and analysis|my output will be|no useful information)[^;]*",
    r"as a result,? i must conclude[^;]*",
    r"i (?:was able to|have) (?:extract|found)[^;]{0,160}(?::|\.)",
    r"predictions you can make for the (?:long|short)[- ]?term future\s*(?:\([^)]*\))?\s*:\s*",
    r"^\s*(?:my )?(?:long|short)[- ]?term predictions?\s*(?:are)?\s*:\s*",
    r"^\s*the objective facts about [^:]{0,120}(?:are|situation)\s*:\s*",
    r"\(no relevant information found\)",
    r"\(next (?:few months|1-4 weeks)\)",
]]

_WS = re.compile(r"\s+")

def _clean_clause(s: str) -> str:
    """清洗单个子句; 无信息量返回 ''。"""
    if not s:
        return ""
    s = re.sub(r"\bNA(?=[A-Z(])", " ", s)          # 修复 "NAHere is" 拼接错位
    for pat in _DROP_PATTERNS:
        if pat.search(s):
            return ""
    for pat in _STRIP_PATTERNS:
        s = pat.sub(" ", s)
    s = _WS.sub(" ", s).strip(" ;,.-")
    if not s:
        return ""
    for pat in _DROP_PATTERNS:                     # 剥离后复检
        if pat.search(s):
            return ""
    return s

def split_date_segments(text: str):
    """'d1: seg1; d2: seg2 ...' -> [(date|None, body), ...] (按出现顺序)。"""
    if not isinstance(text, str) or not text.strip():
        return []
    parts = _DATE_SEG.split(text)
    out = []
    if parts[0].strip():
        out.append((None, parts[0]))
    for i in range(1, len(parts) - 1, 2):
        out.append((parts[i], parts[i + 1]))
    return out

def clean_timemmd_text(raw, max_clauses: int = 6, min_clause_chars: int = 20):
    """
    清洗一个 fact/preds 单元格 -> (clean_text, n_kept, n_total)。
    保留 "YYYY-MM-DD:" 日期锚点; 信息量优先保留 **最新** 日期段
    (对在线预测最相关), 输出按时间正序。
    """
    segs = split_date_segments(raw)
    n_total, kept = 0, []
    for pos, (d, body) in enumerate(segs):
        for clause in body.split(";"):
            n_total += 1
            c = _clean_clause(clause)
            if c and len(c) >= min_clause_chars:
                kept.append((pos, d, c))
    kept = kept[-max_clauses:] if max_clauses > 0 else kept
    pieces, last_d = [], None
    for _pos, d, c in kept:
        if d and d != last_d:
            pieces.append(f"{d}: {c}")
            last_d = d
        else:
            pieces.append(c)
    return " ".join(pieces), len(kept), n_total

def compute_offset_features(x):
    """
    分布偏移统计特征 (B, 8): 供 MoE 门控感知 concept drift。
    x 已被训练集 StandardScaler 归一化, 故训练分布 ≈ N(0,1)。
    """
    B, L, N = x.shape
    flat = x.reshape(B, -1)
    mean_all = flat.mean(dim=1)
    std_all = flat.std(dim=1, unbiased=False)
    q90 = torch.quantile(flat, 0.90, dim=1)
    q10 = torch.quantile(flat, 0.10, dim=1)
    q = max(1, L // 4)
    trend = x[:, -q:, :].mean(dim=(1, 2)) - x[:, :q, :].mean(dim=(1, 2))
    diff_std = (x[:, 1:, :] - x[:, :-1, :]).std(dim=(1, 2), unbiased=False)
    last_level = x[:, -1, :].mean(dim=1)
    return torch.stack([
        mean_all, mean_all.abs(), std_all, (std_all - 1.0).abs(),
        q90 - q10, trend, trend.abs(), diff_std + last_level.abs() * 0.1,
    ], dim=-1)

class AddFusionAdapter(nn.Module):
    """加性融合适配器: h = h_feat + h_grad (+ h_text)。"""
    def __init__(self, d_model, z_shape, text_dim=64, hidden=64, dropout=0.0,
                 use_text=True, use_text_grad=True, device="cpu"):
        super().__init__()
        self.use_text, self.use_text_grad = use_text, use_text_grad
        D_z = z_shape[-1]
        self.linear = nn.Linear(D_z, hidden)
        self.linear_ = nn.Linear(d_model, hidden)
        # [O9] 直接按 (N, D) 构造, 语义同原 torch.ones(z_shape)[0]
        self.lr = nn.Parameter(torch.ones(z_shape[1:], dtype=torch.float32))
        self.drop = nn.Dropout(dropout)
        if use_text:
            self.text_proj = nn.Linear(text_dim, hidden)
            self.txt_lr = nn.Parameter(torch.ones(text_dim) * 0.3)
        self.linear2 = nn.Linear(hidden, hidden)
        self.linear3 = nn.Linear(hidden, D_z)
        self.last_gate = None
        self.aux_loss = torch.zeros(())

    def forward(self, x, his_grad, text_emb=None, his_text_grad=None,
                offset_feat=None):
        B = x.size(0)
        g = (his_grad * self.lr).unsqueeze(0).expand(B, -1, -1)
        h_g = self.linear_(g)
        h_f = self.linear(x)
        if self.use_text and text_emb is not None:
            N = x.size(1)
            if self.use_text_grad and his_text_grad is not None:
                scale = 1.0 + self.txt_lr * his_text_grad
                text_emb = text_emb * scale.unsqueeze(0)
            h_t = self.text_proj(text_emb.unsqueeze(1).expand(B, N, -1))
            h = h_f + h_g + h_t
        else:
            h = h_f + h_g
        h = self.drop(torch.relu(h))
        h = self.drop(torch.relu(self.linear2(h)))
        return self.linear3(h)

class IdentityExpert(nn.Module):
    """零初始化残差专家: 允许门控选择「不做修正」。"""
    def __init__(self, hidden, d_z):
        super().__init__()
        self.out = nn.Linear(hidden, d_z)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, h, feature, g_h, txt_mod, f_h):
        return self.out(h)

class TrendExpert(nn.Module):
    """低秩瓶颈 → 平滑趋势型修正。"""
    def __init__(self, hidden, d_z, rank=2):
        super().__init__()
        self.pre = nn.Linear(hidden, d_z)
        self.down = nn.Linear(d_z, rank, bias=False)
        self.up = nn.Linear(rank, d_z, bias=False)

    def forward(self, h, feature, g_h, txt_mod, f_h):
        return self.up(self.down(self.pre(h)))

class SeasonalityExpert(nn.Module):
    """Fourier 低频模截断 → 周期型修正。"""
    def __init__(self, hidden, d_z, num_modes=8):
        super().__init__()
        self.d_z = d_z
        self.n_freq = d_z // 2 + 1
        self.num_modes = min(num_modes, self.n_freq)
        self.gain = nn.Parameter(torch.zeros(self.num_modes, 2))
        self.mix = nn.Linear(hidden, d_z)

    def forward(self, h, feature, g_h, txt_mod, f_h):
        Xf = torch.fft.rfft(feature, dim=-1)
        gain = torch.view_as_complex(self.gain.contiguous())
        mask = torch.zeros(self.n_freq, dtype=Xf.dtype, device=Xf.device)
        mask[: self.num_modes] = 1.0 + gain
        seasonal = torch.fft.irfft(Xf * mask, n=self.d_z, dim=-1)
        return seasonal + self.mix(h)

class FluctuationExpert(nn.Module):
    """高通 + 最新 Z 梯度 → 误差驱动高频修正。"""
    def __init__(self, hidden, d_z):
        super().__init__()
        self.hp = nn.Linear(d_z, hidden)
        self.body = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(),
                                  nn.Linear(hidden, d_z))

    def forward(self, h, feature, g_h, txt_mod, f_h):
        highpass = feature - feature.mean(dim=-1, keepdim=True)
        return self.body(torch.cat([torch.relu(self.hp(highpass)), g_h],
                                   dim=-1))

class TextFiLMExpert(nn.Module):
    """唯一持有文本通道的专家: 文本 → FiLM(γ, β) 调制特征。"""
    def __init__(self, hidden, d_z, text_dim):
        super().__init__()
        self.film = nn.Linear(text_dim, hidden * 2)
        self.out = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, d_z))

    def forward(self, h, feature, g_h, txt_mod, f_h):
        gamma, beta = self.film(txt_mod).unsqueeze(1).chunk(2, dim=-1)
        gamma = torch.tanh(gamma)
        return self.out(f_h * (1.0 + gamma) + beta)

class _DataEmbed_inv(nn.Module):
    def __init__(self, seq_len, d_model, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(seq_len, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.proj(x.permute(0, 2, 1)))

class _EncLayer(nn.Module):
    def __init__(self, d_model, nhead=4, d_ff=256, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                          batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(d_ff, d_model),
                                nn.Dropout(dropout))
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)

    def forward(self, x):
        y, _ = self.attn(x, x, x)
        x = self.norm1(x + y)
        return self.norm2(x + self.ff(x))

class ITransformerZ(nn.Module):
    """Minimal iTransformer with additive Z injection (checkpoint-compatible)."""
    def __init__(self, seq_len, pred_len, enc_in,
                 d_model=64, e_layers=2, nhead=4, d_ff=128, dropout=0.1):
        super().__init__()
        self.seq_len, self.pred_len, self.d_model = seq_len, pred_len, d_model
        self.embedding = _DataEmbed_inv(seq_len, d_model, dropout)
        self.layers = nn.ModuleList([_EncLayer(d_model, nhead, d_ff, dropout)
                                     for _ in range(e_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.projector = nn.Linear(d_model, pred_len)

    def forward(self, x, z=None, z_loc=None):
        B, L, N = x.shape
        mu = x.mean(1, keepdim=True).detach()
        sigma = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + 1e-5
                           ).detach()
        x = (x - mu) / sigma
        h, feature = self.embedding(x), None
        for idx, layer in enumerate(self.layers):
            h = layer(h)
            if z is not None and z_loc is not None and z_loc == idx + 1:
                feature = h.clone()
                h = h + z[:B]
        if feature is None:
            feature = h.clone()
        pred = self.projector(self.norm(h)).permute(0, 2, 1)
        pred = pred * sigma[:, 0, :].unsqueeze(1) + mu[:, 0, :].unsqueeze(1)
        return {"pred": pred, "feature": feature}
