"""
频率分解 + 自适应频率融合工具 —— VoT_z 专用。

`models/blocks/trendDecomposition.py::FourierDecomp` 是一个未完成的空壳
(只有 `x_ft = torch.fft.rfft(x, dim=-1)`, 无返回值), 不可复用, 因此本文件
从零实现两个独立的频域组件:

  · split_low_high      : 输入窗口的低/高频分解 (供编码阶段构造额外的
                           "频率分支"特征, 与 series_decomp 的时域移动平均
                           趋势/季节分解是互补视角)。
  · AdaptiveFrequencyFusion : VoT 论文的 Adaptive Frequency Fusion —— 在
                           预测阶段, 把 Event-driven 预测与数值预测各自的
                           频谱按 (可学习 + 文本置信度调制的) 逐频段门控
                           混合, 而不是在时域简单加权平均, 使两路预测各自
                           擅长的频段被保留。
"""
import torch
import torch.nn as nn


def split_low_high(x, low_ratio=0.25):
    """x: (B, L, N) -> (low, high) 时域分量。

    在频域按最低 low_ratio 比例的频段做矩形 mask 分离: 低频分量对应缓变的
    趋势/背景, 高频分量 (= x - low) 对应快速波动/噪声/突变。与
    `series_decomp` 的滑动平均分解是互补视角 (频域截断 vs 时域滑窗)。
    """
    B, L, N = x.shape
    Xf = torch.fft.rfft(x, dim=1)                      # (B, n_freq, N)
    n_freq = Xf.size(1)
    cut = max(1, int(round(n_freq * low_ratio)))
    mask = torch.zeros(n_freq, dtype=Xf.dtype, device=x.device)
    mask[:cut] = 1.0
    low = torch.fft.irfft(Xf * mask.view(1, -1, 1), n=L, dim=1)
    high = x - low
    return low, high


class AdaptiveFrequencyFusion(nn.Module):
    """把 event-driven 预测与数值预测的频谱按逐频段门控混合。

    gate = sigmoid(base_gate[freq, channel] + conf_mlp(text_conf)[freq])，
    fused_spectrum = gate * FFT(pred_event) + (1-gate) * FFT(pred_numeric)，
    最终结果 = IFFT(fused_spectrum)。

    base_gate 初始化为 0 (即初始 50/50 等权，训练中学习每个频段该更信任哪一
    路)；text_conf 越高 (文本置信度越强)，conf_mlp 越倾向把 gate 推向偏好
    event-driven 分支的方向 —— 但具体推向高频还是低频完全由训练学出，不做
    人为先验假设。
    """

    def __init__(self, pred_len, enc_in, conf_dim=1, hidden=16):
        super().__init__()
        self.pred_len = pred_len
        self.n_freq = pred_len // 2 + 1
        self.base_gate = nn.Parameter(torch.zeros(self.n_freq, enc_in))
        self.conf_mlp = nn.Sequential(
            nn.Linear(conf_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, self.n_freq))

    def forward(self, pred_event, pred_numeric, conf=None):
        """pred_event/pred_numeric: (B, pred_len, enc_in); conf: (B, conf_dim)
        或 None (视为全 0，即不额外调制，退化为纯可学习门控) ->
        融合后的 (B, pred_len, enc_in)。"""
        B, L, N = pred_numeric.shape
        Xe = torch.fft.rfft(pred_event, dim=1)          # (B, n_freq, N)
        Xn = torch.fft.rfft(pred_numeric, dim=1)
        if conf is None:
            conf = pred_numeric.new_zeros(B, 1)
        gate_shift = self.conf_mlp(conf)                 # (B, n_freq)
        gate = torch.sigmoid(self.base_gate.unsqueeze(0)
                             + gate_shift.unsqueeze(-1))  # (B, n_freq, N)
        fused_spec = gate * Xe + (1.0 - gate) * Xn
        return torch.fft.irfft(fused_spec, n=L, dim=1)
