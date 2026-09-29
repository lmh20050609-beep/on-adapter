"""
共享的隐变量(文本侧)提取模块 —— 从 adapt_z_multimodal.py 抽出, 供
adapt_z_multimodal.py / models/GPT4MTS_z.py / models/VoT_z.py 三方共用。

  · SimpleTextEncoder : 轻量可训练文本编码器 (Embedding + Transformer + mean-pool)。
  · LLMTextEncoder     : 冻结 LLM 文本编码器 (BERT/GPT2/ModernBERT/T5/BART/
                          Qwen3-Embedding), 仅投影 MLP 可训练。
  · load_pretrained_hf : LLMTextEncoder._load 的独立版本, 供 GPT4MTS_z/VoT_z
                          自己的 (与文本编码器无关的) 冻结 GPT2 预测骨干复用同一套
                          "本地目录优先 / HF Hub 兜底" 加载策略。
"""
import os

import torch
import torch.nn as nn


# 本地 GPT2 权重目录 (本仓库运行环境的既有路径约定, 见下方 LLMTextEncoder
# 的 "GPT2" spec)。GPT4MTS_z/VoT_z 各自的冻结 GPT2 预测骨干与
# LLMTextEncoder 的 "GPT2" 文本编码器共用同一份本地权重, 用同一个常量避免
# 两处硬编码该路径导致漂移/其中一处忘记同步。
GPT2_HF_NAME = "/home/ubuntu/strongmodel/gpt2"


def load_pretrained_hf(loader, name, **extra):
    """本地目录优先 (脚本所在目录 / cwd 下的 basename 子目录), 否则回退到
    HF Hub name (先尝试 local_files_only, 再允许联网下载)。"""
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _parent_dir = os.path.dirname(_script_dir)
    _basename = name.split("/")[-1]
    for _local in [os.path.join(_parent_dir, _basename),
                   os.path.join(os.getcwd(), _basename)]:
        if os.path.isdir(_local):
            try:
                return loader.from_pretrained(_local, local_files_only=True,
                                              **extra)
            except Exception:
                pass
    try:
        return loader.from_pretrained(name, local_files_only=True, **extra)
    except Exception:
        pass
    return loader.from_pretrained(name, **extra)


class SimpleTextEncoder(nn.Module):
    """轻量可训练文本编码器 (Embedding + Transformer + mean-pool)。"""
    def __init__(self, vocab_size, embed_dim=64, output_dim=64, max_len=48,
                 nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        enc = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead,
                                         dim_feedforward=embed_dim * 2,
                                         dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.proj = nn.Linear(embed_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, ids):
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        B, L = ids.shape
        pos = torch.arange(L, device=ids.device).unsqueeze(0)
        x = self.token_embed(ids) + self.pos_embed(pos)
        mask = (ids == 0)
        x = self.transformer(x, src_key_padding_mask=mask)
        m = (~mask).float().unsqueeze(-1)
        x = (x * m).sum(1) / m.sum(1).clamp(min=1)
        return self.norm(self.proj(x))


class LLMTextEncoder(nn.Module):
    """
    冻结 LLM 文本编码器 (BERT / GPT2 / ModernBERT / T5 / BART), 仅投影 MLP
    可训练 —— 不微调。
    [O3] 拆分为 pool() (no_grad, 结果可按行缓存) 与 project() (可训练):
         forward(texts) == project(pool(texts))。
    新增编码器说明:
      · MODERNBERT: encoder-only, 支持 8192 长上下文, 推理更快 (2024)。
      · T5:         仅取 T5EncoderModel (编码器部分), 丢弃解码器。
      · BART:       仅取 BartModel.get_encoder() (编码器部分), 丢弃解码器。
    """
    _SPECS = {
        "GPT2":       ("gpt2",       GPT2_HF_NAME,                                 768),
        "GPT2M":      ("gpt2",       "/home/ubuntu/strongmodel/gpt2-medium",      1024),
        "BERT":       ("bert",       "/home/ubuntu/strongmodel/bert-base-uncased", 768),
        # 新增: 更新的文本编码器 (本地目录缺失时按 basename 走在线 HF Hub id)
        "MODERNBERT": ("modernbert", "answerdotai/ModernBERT-base", 768),
        "T5":         ("t5",         "t5-base",                    768),
        "BART":       ("bart",       "facebook/bart-base",         768),
        # Qwen3-Embedding-0.6B: 因果解码器架构专用嵌入模型, 采用末位 token
        # 池化 (last-token pooling), 而非均值池化。
        "QWEN3EMB":   ("qwen3emb",   "/home/ubuntu/strongmodel/Qwen3-Embedding-0.6B",  1024),
    }

    def __init__(self, llm_tag="BERT", output_dim=64, use_fullmodel=True,
                 llm_layers=6, max_length=64, dropout=0.1, device="cpu"):
        super().__init__()
        from transformers import (BertConfig, BertModel, BertTokenizer,
                                  GPT2Config, GPT2Model, GPT2Tokenizer,
                                  AutoConfig, AutoModel, AutoTokenizer,
                                  T5Config, T5EncoderModel,
                                  BartConfig, BartModel)
        if llm_tag not in self._SPECS:
            raise ValueError(f"Unknown llm_tag '{llm_tag}'")
        kind, hf_name, hidden = self._SPECS[llm_tag]
        self.kind = kind
        self.hidden = hidden
        self.use_fullmodel = use_fullmodel
        self.max_length = max_length
        self._device = torch.device(device)

        def _load(loader, name, **extra):
            return load_pretrained_hf(loader, name, **extra)

        if kind == "bert":
            cfg = _load(BertConfig, hf_name)
            cfg.num_hidden_layers = llm_layers
            self.llm = _load(BertModel, hf_name, config=cfg,
                             trust_remote_code=True)
            try:
                self.tok = _load(BertTokenizer, hf_name)
            except Exception:
                self.tok = _load(AutoTokenizer, hf_name)
            self._embed = self.llm.get_input_embeddings()
        elif kind == "gpt2":
            cfg = _load(GPT2Config, hf_name)
            cfg.num_hidden_layers = llm_layers
            self.llm = _load(GPT2Model, hf_name, config=cfg,
                             trust_remote_code=True)
            try:
                self.tok = _load(GPT2Tokenizer, hf_name)
            except Exception:
                self.tok = _load(AutoTokenizer, hf_name)
            self._embed = self.llm.get_input_embeddings()
        elif kind == "modernbert":
            # encoder-only, 与 BERT 接口一致, 走 AutoModel (较新架构,
            # 未必有专用 XxxModel 类导出)。
            cfg = _load(AutoConfig, hf_name, trust_remote_code=True)
            cfg.num_hidden_layers = llm_layers
            self.llm = _load(AutoModel, hf_name, config=cfg,
                             trust_remote_code=True)
            self.tok = _load(AutoTokenizer, hf_name)
            self._embed = self.llm.get_input_embeddings()
        elif kind == "t5":
            # 仅取编码器 (T5EncoderModel), 层数属性是 num_layers 而非
            # num_hidden_layers。
            cfg = _load(T5Config, hf_name)
            cfg.num_layers = llm_layers
            self.llm = _load(T5EncoderModel, hf_name, config=cfg,
                             trust_remote_code=True)
            self.tok = _load(AutoTokenizer, hf_name)
            self._embed = self.llm.get_input_embeddings()
        elif kind == "bart":
            # 编码器-解码器架构, 只保留 encoder 子模块, 解码器随 full
            # 局部变量在 __init__ 返回后被回收 (节省显存/内存)。
            cfg = _load(BartConfig, hf_name)
            cfg.encoder_layers = llm_layers
            cfg.decoder_layers = llm_layers
            full = _load(BartModel, hf_name, config=cfg,
                        trust_remote_code=True)
            self.llm = full.get_encoder()
            self.tok = _load(AutoTokenizer, hf_name)
            self._embed = self.llm.embed_tokens
        elif kind == "qwen3emb":
            # Qwen3-Embedding: 因果解码器 (Qwen3ForCausalLM 主干), 官方推荐
            # 末位 token 池化, 左侧 padding 保证末位即为真实末 token。
            cfg = _load(AutoConfig, hf_name, trust_remote_code=True)
            cfg.num_hidden_layers = llm_layers
            self.llm = _load(AutoModel, hf_name, config=cfg,
                             trust_remote_code=True)
            self.tok = _load(AutoTokenizer, hf_name, trust_remote_code=True,
                             padding_side="left")
            self.tok.padding_side = "left"
            self._embed = self.llm.get_input_embeddings()
        else:
            raise ValueError(f"Unknown encoder kind '{kind}'")

        # 池化方式: 因果嵌入模型 (Qwen3-Embedding) 用末位 token, 其余用均值。
        self.pooling = "last" if kind == "qwen3emb" else "mean"

        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token or "[PAD]"
        for p in self.llm.parameters():
            p.requires_grad = False               # ← frozen, never fine-tuned
        self.llm.eval().to(self._device)

        self.proj = nn.Sequential(
            nn.Linear(hidden, max(hidden // 8, output_dim)), nn.ReLU(),
            nn.Linear(max(hidden // 8, output_dim), output_dim), nn.ReLU(),
            nn.Dropout(dropout))
        self.norm = nn.LayerNorm(output_dim)

    @torch.no_grad()
    def pool(self, texts):
        """texts -> (B, hidden) 冻结池化表征 (逐行缓存的对象)。"""
        if isinstance(texts, str):
            texts = [texts]
        enc = self.tok(texts, return_tensors="pt", padding=True,
                       truncation=True, max_length=self.max_length)
        input_ids = enc["input_ids"].to(self._device)
        attn = enc["attention_mask"].to(self._device)
        emb = self._embed(input_ids)
        if self.use_fullmodel:
            if self.kind == "qwen3emb":
                # 因果模型必须显式传 attention_mask, 否则左侧 padding 的
                # pad token 会被真实 token 错误 attend 到。
                emb = self.llm(inputs_embeds=emb,
                               attention_mask=attn).last_hidden_state
            else:
                emb = self.llm(inputs_embeds=emb).last_hidden_state
        if self.pooling == "last":
            # tokenizer 左 padding, 末位 (-1) 即为每行真实的最后一个 token。
            return emb[:, -1, :]
        mask = attn.unsqueeze(-1).expand_as(emb)
        return (emb * mask).sum(1) / mask.sum(1).clamp(min=1)

    def project(self, pooled):
        """(B, hidden) -> (B, output_dim), 可训练投影。
        pooled 来自冻结 LLM (可能是 bf16/fp16), proj/norm 恒为 fp32
        —— 显式对齐 dtype, 否则 F.linear 会因 dtype 不一致报错。"""
        pooled = pooled.to(self.proj[0].weight.dtype)
        return self.norm(self.proj(pooled))

    def forward(self, texts):
        return self.project(self.pool(texts))
