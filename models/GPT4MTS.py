import numpy as np
import torch
import torch.nn as nn
from torch import optim

# from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from einops import rearrange
# from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
from utils.rev_in import RevIn
from layers.Autoformer_EncDec import series_decomp
from layers.Embed import TimeFeatureEmbedding

class Model(nn.Module):
    
    def __init__(self, configs, patch_size = 16, stride=8):
        super().__init__()
        # self.is_gpt = configs.is_gpt
        self.revin = False
        if patch_size > configs.pred_len :
            self.patch_size = configs.pred_len
        else:
            self.patch_size = patch_size
        # self.pretrain = configs.pretrain
        self.stride = stride
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1

        self.gpt_layers = 6
        # if configs.is_gpt:
        self.gpt2_config = GPT2Config.from_pretrained('/home/ubuntu/strongmodel/gpt2')

        # self.gpt2_config.num_hidden_layers = configs.llm_layers
        self.gpt2_config.output_attentions = True
        self.gpt2_config.output_hidden_states = True

        self.gpt2 = GPT2Model.from_pretrained('/home/ubuntu/strongmodel/gpt2',
                trust_remote_code=True,
                local_files_only=True,
                config=self.gpt2_config,
            )

        self.gpt2.h = self.gpt2.h[:self.gpt_layers]
        print("gpt2 = {}".format(self.gpt2))

        self.relu = nn.ReLU()
        llm_dim = self.gpt2_config.n_embd  # 768
        self.in_layer = nn.Linear(self.patch_size, llm_dim)
        self.out_layer = nn.Linear(llm_dim * self.patch_num, configs.pred_len)
        # 若保留多模态版,再加:
        self.prompt_layer = nn.Linear(llm_dim, llm_dim)
        # self.in_layer = nn.Linear(self.patch_size, configs.d_model)
        #
        # self.prompt_layer = nn.Linear(configs.d_model, configs.d_model)
        # self.out_layer = nn.Linear(configs.d_model * (self.patch_num), configs.pred_len)
        
        # if configs.freeze and configs.pretrain:
        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'ln' in name or 'wpe' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # 多模态时的趋势/季节双分支 prompt 投影 + decomp_w（与 PatchTST_clip/iTransformer_clip 对齐）
        self.multimodal = configs.multimodal
        if configs.multimodal:
            self.tr_sea = configs.tr_sea
            if self.tr_sea:
                self.prompt_layer_seasonal = nn.Linear(llm_dim, llm_dim)
            self.decomp_w = nn.Parameter((torch.tensor([0.5] * 6) / 10).reshape(6, 1, 1, 1))

        # 消融开关：趋势/季节分解 与 频率时间特征，默认关闭，行为与原模型完全一致
        self.use_decomp = getattr(configs, 'use_decomp', 0)
        self.use_time_feat = getattr(configs, 'use_time_feat', 0)
        if self.use_decomp:
            self.decomp = series_decomp(configs.moving_avg)
        if self.use_time_feat:
            self.time_feat_layer = TimeFeatureEmbedding(d_model=llm_dim, embed_type=configs.embed, freq=configs.freq)

        layers_to_device = [self.gpt2, self.in_layer, self.out_layer, self.prompt_layer]
        if configs.multimodal and self.tr_sea:
            layers_to_device.append(self.prompt_layer_seasonal)
        if self.use_time_feat:
            layers_to_device.append(self.time_feat_layer)
        for layer in layers_to_device:
            layer.to(device=configs.gpu)
            layer.train()

        self.device = configs.gpu
        self.rev_in = RevIn(num_features=1) # channel independent

    def get_emb(self, x, tokens=None):
        if tokens is None:
            x = self.gpt2(inputs_embeds=x).last_hidden_state
            return x
        else:
            [a, b, c] = x.shape
            # print(x.shape, tokens.shape)
            tokens = tokens.to(self.prompt_layer.weight.device)
            prompt_x = self.relu(self.prompt_layer(tokens))
            if self.multimodal and self.tr_sea:
                prompt_x = prompt_x + self.relu(self.prompt_layer_seasonal(tokens))
            x_all = torch.cat((prompt_x, x), dim=1)
            x = self.gpt2(inputs_embeds=x_all).last_hidden_state
            return x[: , -b:, :]

    def get_patch(self, x):
        x = rearrange(x, 'b l m -> b m l')
        x = self.padding_patch_layer(x) # b, 1, seq_len
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride) #b, 1, patch_num, patch_size
        x = rearrange(x, 'b m n p -> (b m) n p') # b, patch_num, patch_size

        return x

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, summary, forecast=None):
        B, L, M = x.shape  # 4, 512(or 36), 1

        # ================== Summary / Prompt 处理 (forecast=2 专用) ==================
        if summary is None:
            prompt_tokens = None
        else:
            # 对 prompt (summary) 进行 patching 和平均
            summary = rearrange(summary, 'b l m -> b m l')  # [B, 768, seq_len]
            summary = self.padding_patch_layer(summary)
            summary = summary.unfold(dimension=-1, size=self.patch_size, step=self.stride)
            summary = summary.mean(dim=-1)  # [B, 768, patch_num]
            prompt_tokens = rearrange(summary, 'b l m -> b m l')  # [B, patch_num, 768]

        # ====================== Time Series 主干 ======================
        if self.revin:
            x = self.rev_in(x, 'norm').to(self.device)
        else:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            x /= stdev

        # ---- 趋势/季节分解消融开关 ----
        if self.use_decomp:
            seasonal, trend = self.decomp(x)  # 均为 [B, L, M]
            x = self.in_layer(self.get_patch(seasonal)) + self.in_layer(self.get_patch(trend))  # [B*M, patch_num, 768]
        else:
            x = self.get_patch(x)  # [B*M, patch_num, patch_size]
            x = self.in_layer(x)  # [B*M, patch_num, 768]

        # ---- 频率时间特征消融开关 ----
        if self.use_time_feat and x_mark_enc is not None:
            time_feat = rearrange(x_mark_enc, 'b l m -> b m l')  # [B, num_time_feat, seq_len]
            time_feat = self.padding_patch_layer(time_feat)
            time_feat = time_feat.unfold(dimension=-1, size=self.patch_size, step=self.stride)
            time_feat = time_feat.mean(dim=-1)  # [B, num_time_feat, patch_num]
            time_feat = rearrange(time_feat, 'b m l -> b l m')  # [B, patch_num, num_time_feat]
            x = x + self.time_feat_layer(time_feat)  # [B*M, patch_num, 768]

        # 传入 prompt_tokens
        outputs = self.get_emb(x, prompt_tokens)

        # 输出层
        outputs = self.out_layer(outputs.reshape(B * M, -1))
        outputs = rearrange(outputs, '(b m) l -> b l m', b=B)

        if self.revin:
            outputs = self.rev_in(outputs, 'denorm').to(self.device)
        else:
            outputs = outputs * stdev
            outputs = outputs + means

        return outputs