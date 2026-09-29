"""
GPT4MTS_z —— 修复版 GPT4MTS, 接入本仓库 OnlineMM 的 z/z_loc 在线自适应契约。

相对 `models/GPT4MTS.py` 原版的改动 (原版三个 import 均指向仓库中不存在的包,
且硬编码本地 GPT2 绝对路径, 完全不可用):
  · `utils.rev_in.RevIn`                 -> 删除。原版 `self.revin` 恒为 False,
    RevIn 分支从未被实际执行, 是死代码, 直接去掉。
  · `layers.Autoformer_EncDec.series_decomp` -> 改用
    `models/blocks/trendDecomposition.py::series_decomp` (同名同语义)。
  · `layers.Embed.TimeFeatureEmbedding`  -> 删除。本仓库在线管线的 backbone
    forward 契约里没有 `x_mark_enc` (其余 backbone 如 TimesNet_z/SOFTS_z 均不
    消费时间特征), 保留会破坏统一调用签名, 故 `use_time_feat` 分支一并去掉。
  · GPT2 硬编码路径 `/home/ubuntu/strongmodel/gpt2` -> 改用
    `models/text_latent.py::load_pretrained_hf` 的"本地目录优先/HF Hub 兜底"
    加载策略, 与 `LLMTextEncoder` 共用同一套路径解析逻辑。
  · 原版多 token prompt (`summary` 需要在数据侧预先做与 patch 对齐的 GPT2
    embedding) -> 简化为单 token prompt: 直接消费 `OnlineMM._encode_text()`
    产出的池化文本向量 `text_emb` (B, text_dim), 用一层可训练
    `prompt_proj: Linear(text_dim, llm_dim)` 投影后当成一个 prompt token 拼在
    patch token 序列前面 (`get_emb` 逻辑与原版一致地取输出的最后 patch_num
    个位置)。不再需要为 backbone 单独维护一份冻结文本 LLM —— 复用 OnlineMM
    已算好的 text_emb, 避免重复加载/编码文本, 语义上仍是 GPT4MTS 论文的
    "prompt-prepend 冻结 GPT2" 思路。
  · z 注入 [O8 更新]: 原版固定单点 (patch embedding 之后、冻结 GPT2 之前),
    `z_loc` 形参完全被忽略 —— 是仓库里唯一一个 "z_loc 名存实亡" 的 backbone。
    现改为手动展开冻结 GPT2 的 block 循环 (`_gpt2_layerwise`), 复刻 HF
    `GPT2Model.forward` 内部逻辑 (wpe 位置编码只在输入处加一次 -> drop ->
    依次跑 `self.gpt2.h` 每个 block -> 最后 `ln_f`), 从而可以在任意一个
    block 的输出处截住并注入 z, 与 `models/TimesNet_z.py` 的逐层 z_loc
    语义对齐: `z_loc=0` = 旧版单点固定注入 (字节级复现旧行为, 作为对照组
    保留); `z_loc=k` (1..gpt_layers) = 第 k 个 GPT2 block 输出之后注入,
    `z_loc=gpt_layers` 是最深的可用点 (紧邻 ln_f 之前)。`USES_TEXT = True`
    告诉 `OnlineMM` 直接把 `_encode_text()` 的 (text_emb, conf) 传进来,
    不必再经外部融合适配器。
"""
import torch
import torch.nn as nn
from einops import rearrange

from models.text_latent import load_pretrained_hf, GPT2_HF_NAME
from models.blocks.trendDecomposition import series_decomp


def infer_patch_num(seq_len, pred_len, patch_size=16, stride=8):
    """GPT4MTS_z/VoT_z 共用的 patch 数推导 (与 `_PatchTSTZAdapter` 同理,
    `_backbone_dims()` 需要在不构造模型的情况下预知 z 的注入张量形状,
    故抽成独立函数, 避免与 `Model.__init__` 内部逻辑重复/漂移)。"""
    eff_patch = pred_len if patch_size > pred_len else patch_size
    return (seq_len - eff_patch) // stride + 1 + 1   # +1: 右侧 replication pad


class Model(nn.Module):
    USES_TEXT = True

    def __init__(self, seq_len, pred_len, enc_in, text_dim=64,
                 patch_size=16, stride=8, e_layers=6,
                 use_decomp=False, moving_avg=25, device="cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in

        # 与原版一致的既有 quirk: patch 不应比预测步长还长
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

        self.use_decomp = use_decomp
        self.decomp = series_decomp(moving_avg) if use_decomp else None

        for layer in (self.in_layer, self.out_layer, self.prompt_proj):
            layer.to(self.device)

    def get_patch(self, x):
        x = rearrange(x, "b l m -> b m l")
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x = rearrange(x, "b m n p -> (b m) n p")
        return x

    def _gpt2_layerwise(self, h, prompt_tokens, z, z_loc):
        """手动展开冻结 GPT2 的逐 block 前向 (复刻 HF GPT2Model.forward 内部
        逻辑: wpe 位置编码只在输入处加一次 -> drop -> 依次跑 self.gpt2.h 的
        每个 block -> 最后 ln_f), 从而可以在任意一个 block 的输出处截住并
        注入 z (h: (B*M, patch_num[+1 prompt], llm_dim) 已完成 patch/prompt
        拼接前的纯 patch 部分)。

        z_loc 语义与 models/TimesNet_z.py 的逐层 z_loc 对齐: 0 = 旧版单点
        固定注入 (GPT2 之前, 字节级复现旧行为); k (1..gpt_layers) = 第 k 个
        GPT2 block 输出之后注入, k=gpt_layers 是最深的可用点 (紧邻 ln_f
        之前)。返回 (hidden_state, feature) —— feature 是注入点前一刻的
        快照 (z=None 时也照样计算, 供纯梯度探测场景使用, 与 TimesNet_z 同一
        约定)。

        z 只加在 patch token 那一段 (最后 b=patch_num 个位置), 不碰可能被
        拼接在最前面的 prompt token —— 否则 USES_TEXT=True 时序列长度会变成
        patch_num+1, 与 z_shape=(B, patch_num, llm_dim) 的对外契约错位、
        加法直接 shape mismatch。"""
        b = h.shape[1]                      # patch_num (不含 prompt token)

        def _inject(hs):
            patch_part = hs[:, -b:, :]
            feature = patch_part.clone().requires_grad_(True)
            injected = feature + z if z is not None else feature + torch.zeros_like(feature)
            hs = torch.cat([hs[:, :-b, :], injected], dim=1) if hs.shape[1] > b else injected
            return hs, feature

        feature = None
        if z_loc == 0:
            h, feature = _inject(h)

        if prompt_tokens is not None:
            h = torch.cat((prompt_tokens, h), dim=1)

        pos_ids = torch.arange(h.shape[1], device=h.device).unsqueeze(0)
        hidden = self.gpt2.drop(h + self.gpt2.wpe(pos_ids))

        for i, block in enumerate(self.gpt2.h):
            hidden = block(hidden)[0]
            if z_loc == i + 1:
                hidden, feature = _inject(hidden)
        hidden = self.gpt2.ln_f(hidden)

        if feature is None:                # z_loc 越界兜底: 等价于不注入
            feature = hidden[:, -b:, :].clone()
        if prompt_tokens is not None:
            hidden = hidden[:, -b:, :]
        return hidden, feature

    def forward(self, x, z=None, z_loc=0, text_emb=None, text_conf=None):
        """x: (B, seq_len, enc_in) -> {"pred": (B, pred_len, enc_in),
        "feature": (B, patch_num, llm_dim) 注入点前一刻的快照 (已对通道维取
        均值聚合, 与 _backbone_dims() 声明的 z_shape=(B, patch_num, llm_dim)
        契约一致)}。
        text_emb: (B, text_dim) 或 None; text_conf: (B,1) 或 None (低置信度
        样本按 conf 软降权 prompt 贡献, 而非硬屏蔽)。z_loc: 0 = 旧版
        pre-GPT2 单点注入 (对照组); 1..gpt_layers = 冻结 GPT2 内部第 k 个
        block 输出之后注入, 越大越深 (见 _gpt2_layerwise)。"""
        B, L, M = x.shape

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True,
                                     unbiased=False) + 1e-5).detach()
        x = x / stdev

        if self.use_decomp:
            seasonal, trend = self.decomp(x)
            h = self.in_layer(self.get_patch(seasonal)) + \
                self.in_layer(self.get_patch(trend))
        else:
            h = self.in_layer(self.get_patch(x))          # (B*M, patch_num, llm_dim)

        z_bm = None
        if z is not None:
            z_bm = torch.repeat_interleave(z[:B], M, dim=0)  # 对齐 'b m -> (b m)'

        prompt_tokens = None
        if text_emb is not None:
            prompt = self.relu(self.prompt_proj(text_emb))    # (B, llm_dim)
            if text_conf is not None:
                prompt = prompt * text_conf
            prompt = torch.repeat_interleave(prompt, M, dim=0).unsqueeze(1)
            prompt_tokens = prompt                             # (B*M, 1, llm_dim)

        out, feat_bm = self._gpt2_layerwise(h, prompt_tokens, z_bm, z_loc)

        # feature 对外契约是 (B, patch_num, llm_dim) (与 _backbone_dims() 里
        # z_n_dim=patch_num 对应, 供 adapter/梯度探测使用) —— 按 'b m -> (b m)'
        # 的逆操作把逐通道 (B*M,...) 张量的通道维聚合(均值)掉, 而不是暴露
        # B*M 这个内部实现细节。
        feature = feat_bm.clone().view(B, M, feat_bm.shape[-2], -1).mean(dim=1)

        out = self.out_layer(out.reshape(B * M, -1))
        out = rearrange(out, "(b m) l -> b l m", b=B)

        out = out * stdev + means
        return {"pred": out, "feature": feature}
