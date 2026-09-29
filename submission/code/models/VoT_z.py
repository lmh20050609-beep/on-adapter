"""
VoT_z —— 在 GPT4MTS_z 基础上叠加 VoT (View-of-Time 风格) 论文的三个要点,
按"表征代理"简化 (不调用生成式 LLM, 全部换成轻量可训练模块), 以便接入本仓库
的在线自适应管线做公平对比:

  1. 趋势/季节分解 + 频率分解 (Multi-level Alignment 的输入侧):
     · `series_decomp` (复用 `models/blocks/trendDecomposition.py`) 把输入窗口
       拆成 trend/seasonal 两支, 分别过 patch embedding, 再相加送入冻结 GPT2
       (与 GPT4MTS_z 的 `use_decomp=True` 分支同构), 同时各自的 patch 表征
       (池化后) 单独暴露给下面的 Endogenous Text Alignment 使用。
     · `split_low_high` (复用 `models/freq_decomp.py`) 把归一化后的输入窗口
       拆成低/高频分量, 送入一个小型 `freq_encoder` MLP 产出一个残差修正,
       叠加到 GPT2 数值预测上 (用一个初始化很小的可学习标量 `freq_scale`
       控制强度, 避免训练初期扰乱已预训练/冻结的 GPT2 数值主干)。

  2. Event-driven Reasoning (表征代理, 已与用户确认的简化): 不调用生成式
     LLM 做逐步推理, 而是用冻结文本池化向量 (`text_emb`, 由 OnlineMM 的
     `RealTextProvider` + 文本编码器算好传入) 过一个轻量可训练 `EventHead`
     (MLP: text_emb -> (pred_len, enc_in)) 直接产出"事件驱动预测"。

     Historical In-Context Learning 用一个 FIFO 记忆缓冲区
     (`register_buffer`, 非训练参数) 代理: 缓存最近 K 条 `(text_emb, y)`,
     推理时按当前 batch 的 text_emb 与缓冲区逐条做余弦相似度加权检索历史
     y, 与 `EventHead` 的输出按一个可学习标量门控 `hist_gate` 混合, 得到
     最终的"事件驱动预测" `pred_event`。缓冲区需要外部在每步训练后调用
     `update_memory(text_emb, y)` 写入 (不在 forward 内部自动写, 因为
     forward 在验证/梯度探测阶段也会被调用, 不应污染缓冲区) —— 由
     `OnlineMM` 检测 `hasattr(backbone, "update_memory")` 后调用。

  3. Multi-level Alignment:
     · Endogenous Text Alignment (完整版): `MultiComponentVICRegAlign`
       (`models/text_alignment.py`) 分别对 trend 分量表征、seasonal 分量
       表征各自与 text_emb 做一次 VICReg 对齐, 返回值累加进 `aux_loss`。
     · Adaptive Frequency Fusion: `AdaptiveFrequencyFusion`
       (`models/freq_decomp.py`) 把 `pred_event` 与 `pred_numeric` (GPT2 数值
       预测 + 频率残差) 沿 pred_len 轴做 rfft, 用一个以文本置信度为条件的
       可学习逐频段 sigmoid 门控混合两个频谱, irfft 回时域得到最终融合预测
       —— 真正意义上的"频率级互补", 而不是时域简单加权平均。

z 注入位置: 支持 4 个 shape 兼容 ((B, patch_num, 768)) 的候选点 (`POS_TREND`/
`POS_SEASONAL`/`POS_PRE_GPT2`/`POS_POST_GPT2`, 见下)，`z` 为普通张量时按
`z_loc` 命中其中一个点注入 (未命中/`z_loc=None` 时不注入)；`z` 为
`{pos: tensor}` 的 dict 时 (来自 `PositionAdaptiveMoEFusionAdapter`,
`SUPPORTS_POS_GATE=True`) 同时对命中的每个点分别注入，供 iAdapter+PosGate
消融对比"固定位置 vs 自适应位置"。默认单张量场景下 `z_loc=3`
(`POS_PRE_GPT2`，patch embedding 之后/冻结 GPT2 之前) 与旧版单点固定注入
行为一致 (由 `adapt_z_multimodal.py::_backbone_dims` 的 VoT_z 分支设置)。
`USES_TEXT = True`, 直接消费 OnlineMM 传入的 (text_emb, text_conf)。
"""
import torch
import torch.nn as nn
from einops import rearrange

from models.text_latent import load_pretrained_hf, GPT2_HF_NAME
from models.blocks.trendDecomposition import series_decomp
from models.freq_decomp import split_low_high, AdaptiveFrequencyFusion
from models.text_alignment import MultiComponentVICRegAlign
from models.GPT4MTS_z import infer_patch_num


POS_TREND, POS_SEASONAL, POS_PRE_GPT2, POS_POST_GPT2 = 1, 2, 3, 4


class Model(nn.Module):
    USES_TEXT = True
    SUPPORTS_POS_GATE = True  # iAdapter+PosGate 多位置 dict-z 注入支持

    def __init__(self, seq_len, pred_len, enc_in, text_dim=64,
                 patch_size=16, stride=8, e_layers=6,
                 moving_avg=25, low_freq_ratio=0.25,
                 mem_size=64, hidden=64, device="cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.text_dim = text_dim

        self.patch_size = pred_len if patch_size > pred_len else patch_size
        self.stride = stride
        self.patch_num = infer_patch_num(seq_len, pred_len, patch_size, stride)
        self.padding_patch_layer = nn.ReplicationPad1d((0, stride))

        from transformers import GPT2Config, GPT2Model
        self.gpt_layers = min(max(e_layers, 1), 12)
        gpt2_config = load_pretrained_hf(GPT2Config, GPT2_HF_NAME)
        gpt2_config.output_attentions = False
        gpt2_config.output_hidden_states = False
        self.gpt2 = load_pretrained_hf(GPT2Model, GPT2_HF_NAME, config=gpt2_config,
                                       trust_remote_code=True)
        self.gpt2.h = self.gpt2.h[:self.gpt_layers]
        for name, param in self.gpt2.named_parameters():
            param.requires_grad = ("ln" in name or "wpe" in name)
        self.gpt2.to(self.device)

        llm_dim = gpt2_config.n_embd
        self.llm_dim = llm_dim
        self.relu = nn.ReLU()
        self.in_layer = nn.Linear(self.patch_size, llm_dim)
        self.out_layer = nn.Linear(llm_dim * self.patch_num, pred_len)
        self.prompt_proj = nn.Linear(text_dim, llm_dim)

        # ── 1a. 趋势/季节分解 (Multi-level Alignment 输入侧) ──
        self.decomp = series_decomp(moving_avg)

        # ── 1b. 频率分解残差分支 ──
        self.low_freq_ratio = low_freq_ratio
        self.freq_encoder = nn.Sequential(
            nn.Linear(2 * seq_len, hidden), nn.ReLU(),
            nn.Linear(hidden, pred_len))
        self.freq_scale = nn.Parameter(torch.tensor(0.1))

        # ── 2. Event-driven Reasoning 表征代理 ──
        self.event_head = nn.Sequential(
            nn.Linear(text_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, pred_len * enc_in))
        self.hist_gate = nn.Parameter(torch.tensor(0.0))    # sigmoid(0)=0.5

        self.mem_size = mem_size
        self.register_buffer("mem_text", torch.zeros(mem_size, text_dim))
        self.register_buffer("mem_y", torch.zeros(mem_size, pred_len, enc_in))
        self.register_buffer("mem_filled", torch.zeros(mem_size, dtype=torch.bool))
        self.register_buffer("mem_ptr", torch.zeros((), dtype=torch.long))

        # ── 3a. Endogenous Text Alignment (完整版) ──
        self.align = MultiComponentVICRegAlign(d_z=llm_dim, text_dim=text_dim)

        # ── 3b. Adaptive Frequency Fusion ──
        self.freq_fusion = AdaptiveFrequencyFusion(pred_len, enc_in, conf_dim=1)

        for layer in (self.in_layer, self.out_layer, self.prompt_proj,
                     self.freq_encoder, self.event_head, self.align,
                     self.freq_fusion):
            layer.to(self.device)

    def get_patch(self, x):
        x = rearrange(x, "b l m -> b m l")
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x = rearrange(x, "b m n p -> (b m) n p")
        return x

    def get_emb(self, x, prompt_tokens=None):
        if prompt_tokens is None:
            return self.gpt2(inputs_embeds=x).last_hidden_state
        b = x.shape[1]
        x_all = torch.cat((prompt_tokens, x), dim=1)
        out = self.gpt2(inputs_embeds=x_all).last_hidden_state
        return out[:, -b:, :]

    @torch.no_grad()
    def update_memory(self, text_emb, y):
        """text_emb: (B, text_dim), y: (B, pred_len, enc_in) —— 由外部
        (OnlineMM) 在每个真实训练步之后调用, 写入 FIFO 历史记忆缓冲区
        (循环覆盖最旧条目)。不在 forward 内部自动调用, 避免验证/梯度探测
        前向污染记忆。"""
        B = text_emb.size(0)
        idx = (self.mem_ptr + torch.arange(B, device=self.mem_text.device)) % self.mem_size
        self.mem_text[idx] = text_emb.detach().to(self.mem_text.dtype)
        self.mem_y[idx] = y.detach().to(self.mem_y.dtype)
        self.mem_filled[idx] = True
        self.mem_ptr.fill_(int((self.mem_ptr.item() + B) % self.mem_size))

    def _retrieve_history(self, text_emb):
        """text_emb: (B, text_dim) -> (B, pred_len, enc_in) 按余弦相似度
        加权的历史 y 检索结果; 缓冲区全空时返回全 0 (退化为纯 EventHead)。"""
        B = text_emb.size(0)
        if not bool(self.mem_filled.any()):
            return text_emb.new_zeros(B, self.pred_len, self.enc_in)
        valid = self.mem_filled                              # (K,)
        mem_text = self.mem_text[valid]                       # (K', text_dim)
        mem_y = self.mem_y[valid]                              # (K', pred_len, enc_in)
        q = torch.nn.functional.normalize(text_emb, dim=-1)
        k = torch.nn.functional.normalize(mem_text, dim=-1)
        sim = q @ k.t()                                        # (B, K')
        w = torch.softmax(sim, dim=-1)                         # (B, K')
        return torch.einsum("bk,kpn->bpn", w, mem_y)

    def _inject(self, base, pos, z, z_loc, is_multi, B, M):
        """在候选注入点 `pos` 处把 `z` 加性注入 `base` (若命中)；未命中原样
        返回。`is_multi=True` 时 `z` 是 `{pos: tensor}` dict (PosGate 多位置
        自适应注入)，否则 `z` 是单张量、仅当 `z_loc == pos` 时注入 (固定
        位置消融)。"""
        if is_multi:
            zi = z.get(pos)
        elif z is not None and z_loc == pos:
            zi = z
        else:
            zi = None
        if zi is None:
            return base
        return base + torch.repeat_interleave(zi[:B], M, dim=0)

    def forward(self, x, z=None, z_loc=None, text_emb=None, text_conf=None):
        """x: (B, seq_len, enc_in) -> {"pred", "feature", "aux_loss"}。
        text_emb/text_conf 语义与 GPT4MTS_z 一致。z 注入位置见模块 docstring
        (POS_TREND/POS_SEASONAL/POS_PRE_GPT2/POS_POST_GPT2)。aux_loss 是
        Endogenous Text Alignment 的标量损失 (text_emb 为 None 或 batch<2
        时为 0), 由调用方累加进主训练损失。"""
        B, L, M = x.shape

        means = x.mean(1, keepdim=True).detach()
        x_c = x - means
        stdev = torch.sqrt(torch.var(x_c, dim=1, keepdim=True,
                                     unbiased=False) + 1e-5).detach()
        x_n = x_c / stdev

        # ---- 1a. 趋势/季节分解, 分别 patch-embed ----
        seasonal, trend = self.decomp(x_n)
        h_trend = self.in_layer(self.get_patch(trend))        # (B*M, patch_num, llm_dim)
        h_seasonal = self.in_layer(self.get_patch(seasonal))

        # feature 对外契约是 (B, patch_num, llm_dim), 见 GPT4MTS_z.py 同一处的
        # 注释: 按通道维聚合(均值)掉 (b m) 里的 m, 而不是暴露 B*M。始终从
        # "两支相加、任何位置注入之前"的干净张量算, 不受选用哪个候选注入点
        # 影响, 保证消融时门控网络看到的参考信号一致。
        feature = (h_trend + h_seasonal).clone().view(
            B, M, self.patch_num, -1).mean(dim=1)

        is_multi = isinstance(z, dict)
        h_trend_z = self._inject(h_trend, POS_TREND, z, z_loc, is_multi, B, M)
        h_seasonal_z = self._inject(h_seasonal, POS_SEASONAL, z, z_loc,
                                    is_multi, B, M)
        h = self._inject(h_trend_z + h_seasonal_z, POS_PRE_GPT2, z, z_loc,
                         is_multi, B, M)

        prompt_tokens = None
        if text_emb is not None:
            prompt = self.relu(self.prompt_proj(text_emb))
            if text_conf is not None:
                prompt = prompt * text_conf
            prompt = torch.repeat_interleave(prompt, M, dim=0).unsqueeze(1)
            prompt_tokens = prompt

        out = self.get_emb(h, prompt_tokens)
        out = self._inject(out, POS_POST_GPT2, z, z_loc, is_multi, B, M)
        out = self.out_layer(out.reshape(B * M, -1))
        pred_numeric = rearrange(out, "(b m) l -> b l m", b=B)  # 标准化空间

        # ---- 1b. 频率分解残差 ----
        low, high = split_low_high(x_n, low_ratio=self.low_freq_ratio)
        freq_in = torch.cat([low, high], dim=1)                 # (B, 2*seq_len, M)
        freq_in = rearrange(freq_in, "b l m -> (b m) l")
        freq_res = self.freq_encoder(freq_in)                   # (B*M, pred_len)
        freq_res = rearrange(freq_res, "(b m) l -> b l m", b=B)
        pred_numeric = pred_numeric + self.freq_scale * freq_res

        # ---- 2. Event-driven Reasoning (+ Historical In-Context) ----
        if text_emb is not None:
            event_out = self.event_head(text_emb).view(B, self.pred_len, M)
            hist_out = self._retrieve_history(text_emb)
            gate = torch.sigmoid(self.hist_gate)
            pred_event = gate * hist_out + (1.0 - gate) * event_out
        else:
            pred_event = pred_numeric

        # ---- 3b. Adaptive Frequency Fusion ----
        pred = self.freq_fusion(pred_event, pred_numeric, conf=text_conf)

        pred = pred * stdev + means

        # ---- 3a. Endogenous Text Alignment ----
        aux_loss = pred.new_zeros(())
        if text_emb is not None and B >= 2:
            trend_feat = h_trend.mean(dim=1).view(B, M, self.llm_dim)
            seasonal_feat = h_seasonal.mean(dim=1).view(B, M, self.llm_dim)
            aux_loss = self.align(trend_feat, seasonal_feat, text_emb, text_conf)

        return {"pred": pred, "feature": feature, "aux_loss": aux_loss}
