"""
TS-文本对齐模块 —— 从 adapt_z_multimodal.py (SECTION 6b) 抽出。

  · VICRegAlign              : 单次池化对齐 (原文件已自注释为"简化版 VoT
                                Endogenous Text Alignment": 不做趋势/季节
                                分解式对比, 直接对池化 TS 表征与文本表征做一次
                                VICReg 对齐)。iAdapter/GPT4MTS_z 使用。
  · MultiComponentVICRegAlign : VoT_z 的"完整版" Endogenous Text Alignment ——
                                对趋势分量表征、季节分量表征分别与文本表征各做
                                一次 VICRegAlign, 求和成一个总对齐损失。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _dominant_period(x):
    """x: (B,L,N) 标准化输入窗口 -> 每个样本的主周期长度 (跨通道平均幅谱,
    去 DC 后取幅值最大的频率分量, 转换为周期长度 L/k)。"""
    xf = torch.fft.rfft(x - x.mean(dim=1, keepdim=True), dim=1)
    mag = xf.abs().mean(dim=-1)                  # (B, L//2+1)
    mag = mag.clone()
    mag[:, 0] = 0.0                               # 去 DC
    k = mag.argmax(dim=1).clamp(min=1)
    return x.size(1) / k.float()


def _jump_score(x):
    """x: (B,L,N) -> 一阶差分最大绝对值 / 标准差, 逐通道后取均值。
    数值越大, 窗口内越像存在突变/跳变而非平滑周期波动。"""
    d = x.diff(dim=1)
    return (d.abs().amax(dim=1) / (x.std(dim=1) + 1e-6)).mean(dim=-1)


class VICRegAlign(nn.Module):
    """VICReg (Bardes et al. 2022) 式 TS-文本对齐模块, 替换原 CLIPAlign
    (CLIP 式对称 InfoNCE)。

    InfoNCE 在负样本全部相似 (冻结 LLM 池化向量普遍高度聚集) 时梯度
    趋于消失, 对齐名存实亡。VICReg 不依赖负样本对比, 用三项损失直接
    防止表征坍塌:
      · 不变性 (invariance): MSE(ts_e, text_e) 拉近配对样本;
      · 方差   (variance):   逐维强制 std > gamma, 直接阻止坍塌;
      · 协方差 (covariance): 去相关, 防止所有维度携带同一信息。
    投影头用 BatchNorm1d (跨批次归一化) 取代 LayerNorm (逐样本归一化),
    主动把聚集的 LLM 池化向量在批次维度上推开。
    """

    def __init__(self, d_z, text_dim, d_clip=64,
                 lam=25.0, mu=25.0, nu=1.0, gamma=1.0):
        super().__init__()
        self.lam, self.mu, self.nu, self.gamma = lam, mu, nu, gamma

        # 更深的投影头 + BN 取代 LayerNorm
        # BN 跨批次归一化 → 阻止所有向量坍塌到同一点
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
        """feature: (B,N,D_z) TS 表征; text_emb: (B,text_dim) 文本表征
        (未乘置信度) -> (ts_e, text_e), 均为 (B,d_clip)。"""
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
        """conf: (B,1) 文本置信度, 0 表示无信息占位文本 -> 逐样本降权,
        不强行对齐「假文本」。dup_mask 保留形参以兼容原调用签名
        (VICReg 不做负样本对比, 无需用它剔除重复正样本)。B<2 时方差/
        协方差项无意义, 返回 0。
        """
        B = feature.size(0)
        if B < 2:
            return feature.new_zeros(())
        ts_e, text_e = self.embed(feature, text_emb)

        # conf 加权：低置信度样本不强行对齐
        w = (conf.squeeze(-1) if conf is not None
             else torch.ones(B, device=ts_e.device))
        wsum = w.sum().clamp_min(1e-6)

        inv = ((ts_e - text_e).pow(2).mean(-1) * w).sum() / wsum
        var = self._var_loss(ts_e) + self._var_loss(text_e)
        cov = self._cov_loss(ts_e) + self._cov_loss(text_e)

        return self.lam * inv + self.mu * var + self.nu * cov


class MultiComponentVICRegAlign(nn.Module):
    """VoT_z 用的"完整版" Endogenous Text Alignment: 分别对趋势分量表征、
    季节分量表征各自与文本表征做一次 VICRegAlign, 而不是只对一个池化表征做
    一次对齐 —— 让文本信息同时校准"慢变化的趋势语义"与"周期性波动语义"两个
    子空间, 是 VICRegAlign(简化版) 的直接扩展。

    trend_weight/seasonal_weight 控制两个分量各自对齐损失的相对权重, 默认
    等权 (1.0, 1.0)。
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
        """trend_feature/seasonal_feature: (B,N,D_z); text_emb: (B,text_dim)。
        两路分量共享同一份文本表征 (VoT 论文语义: 内生文本在表征层面同时约束
        趋势与周期两个子空间), 各自过独立投影头再各算一次 VICReg 三项损失。"""
        l_trend = self.trend_align(trend_feature, text_emb, conf)
        l_seasonal = self.seasonal_align(seasonal_feature, text_emb, conf)
        return self.trend_weight * l_trend + self.seasonal_weight * l_seasonal
