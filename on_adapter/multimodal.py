import ast
from contextlib import contextmanager, redirect_stdout
import io
import math
import re
from pathlib import Path
import sys
import torch
from .config import ROOT
from .native import source_module, model_config, NativeForecast
from .prepared_text import PreparedText

@contextmanager
def local_gpt2():
    import transformers as t
    classes = [t.GPT2Model, t.GPT2Config, t.GPT2Tokenizer, t.AutoTokenizer]
    originals = {}
    for cls in classes:
        originals[cls] = (cls.__dict__.get('from_pretrained'), cls.from_pretrained)
        original = cls.from_pretrained
        def replacement(c, location, *args, _original=original, **kwargs):
            kwargs.pop('cache_dir', None)
            kwargs['local_files_only'] = True
            if c.__name__.endswith('Model'):
                kwargs['attn_implementation'] = 'eager'
            return _original(str(ROOT / 'third_party/weights/gpt2'), *args, **kwargs)
        cls.from_pretrained = classmethod(replacement)
    try:
        yield
    finally:
        for cls, (descriptor, bound) in originals.items():
            if descriptor is None:
                delattr(cls, 'from_pretrained')
            else:
                setattr(cls, 'from_pretrained', descriptor)

def author_class(path, name, namespace):
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    definition = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    tree = ast.fix_missing_locations(ast.Module(body=[definition], type_ignores=[]))
    exec(compile(tree, str(path), 'exec'), namespace)
    return namespace[name]

class GPTForecast(torch.nn.Module):
    def __init__(self, model, horizon, data=None, lookback=None):
        super().__init__()
        self.author_model, self.horizon, self.data, self.lookback = model, horizon, data, lookback
        self.text = getattr(data, 'prepared_text', None) if data is not None else None

    def forward(self, batch):
        if self.data is None:
            return self.author_model(batch['x'], 0)[..., -1]
        if self.text is None:
            raise RuntimeError('Prepare frozen GPT4MTS features before constructing the model')
        summary = self.text.gpt(batch['origin'], self.lookback, batch['x'].device)
        B, L, C = batch['x'].shape
        x = batch['x'].permute(0, 2, 1).reshape(B*C, L, 1)
        summary = summary.repeat_interleave(C, dim=0)
        out = self.author_model(x, 0, summary)
        return out.reshape(B, C, self.horizon).permute(0, 2, 1)

class MMD(torch.nn.Module):
    def __init__(self, numeric, mlp_class, horizon, data, lookback):
        super().__init__()
        self.numeric, self.data, self.lookback = numeric, data, lookback
        self.text = data.prepared_text
        self.mlp = mlp_class([768, 96, horizon], dropout_rate=.3)
        self.prompt_weight = .01  # author run.py default

    def forward(self, batch):
        numeric = self.numeric(batch)
        token_embeddings = self.text.tokens(batch['origin'], numeric.device)
        text = torch.stack([self.mlp(tokens).mean(0) for tokens in token_embeddings])
        # Author active final_sum path. Recompute its prior from known history,
        # because the original loader reads future rows of prior_history_avg.
        text = text - text.mean(1, keepdim=True).detach()
        text = text / torch.sqrt(text.var(1, keepdim=True, unbiased=False) + 1e-5)
        prior = batch['x'].mean(1, keepdim=True)
        text = text.unsqueeze(-1)
        return (1 - self.prompt_weight) * numeric + self.prompt_weight * (text + prior)

class AutoTimesForecast(torch.nn.Module):
    def __init__(self, model, horizon, data, lookback):
        super().__init__()
        self.author_model, self.horizon, self.data, self.lookback = model, horizon, data, lookback

    def training_objective(self, batch):
        length = self.lookback
        token_len = self.author_model.token_len
        pred = self.author_model(batch['x'], None, None, None)[..., -1]
        joined = torch.cat([batch['history_y'][..., -1], batch['y'][..., -1]], dim=1)
        joined_mask = torch.cat([batch['history_mask'][..., -1], batch['mask'][..., -1]], dim=1)
        # A short H can supply fewer labels than one token; unsupplied labels stay masked.
        if joined.shape[1] < token_len + length:
            pad = token_len + length - joined.shape[1]
            joined = torch.nn.functional.pad(joined, (0, pad))
            joined_mask = torch.nn.functional.pad(joined_mask, (0, pad), value=False)
        truth = joined[:, token_len:token_len + length]
        mask = joined_mask[:, token_len:token_len + length]
        return ((pred[mask] - truth[mask]) ** 2).mean(), int(mask.sum())

    def forward(self, batch):
        x = batch['x']
        token_len = self.author_model.token_len
        outputs = []
        for _ in range(math.ceil(self.horizon / token_len)):
            out = self.author_model(x, None, None, None)[:, -token_len:]
            outputs.append(out)
            x = torch.cat([x[:, token_len:], out], dim=1)
        return torch.cat(outputs, dim=1)[:, :self.horizon, -1]

def build_multimodal(method, settings, data, lookback, horizon, device):
    args = model_config(settings, data, lookback, horizon, method)
    if method == 'GPT4TS':
        base = ROOT / 'third_party/code/gpt4ts/Long-term_Forecasting'
        sys.path.insert(0, str(base / 'models'))
        sys.path.insert(0, str(base))
        args.d_model = 768
        with local_gpt2(), redirect_stdout(io.StringIO()):
            module = source_module('gpt4ts', 'Long-term_Forecasting/models/GPT4TS.py')
            model = module.GPT4TS(args, device)
        return GPTForecast(model.to(device), horizon), vars(args)
    if method == 'Time-LLM':
        args.d_ff = 32  # author's flattened LLM output subspace; not an LLM hidden-size replacement
        with local_gpt2():
            module = source_module('timellm', 'models/TimeLLM.py')
            model = module.Model(args)
        wrapper = NativeForecast(model.to(device), horizon, args.label_len)
        wrapper.bf16 = True
        args.precision = 'author-compatible bfloat16 autocast'
        return wrapper, vars(args)
    if method == 'AutoTimes':
        args.llm_layers = 12  # This author branch uses the full GPT2 model.
        args.token_len = math.gcd(lookback, 12)
        args.use_multi_gpu = False; args.local_rank = 0; args.mix_embeds = False
        args.mlp_hidden_layers = 2; args.mlp_hidden_dim = 256; args.mlp_activation = 'gelu'
        module = source_module('autotimes', 'models/AutoTimes_Gpt2.py')
        with local_gpt2():
            model = module.Model(args)
        args.note = 'Author GPT2 mixFalse branch; shifted-token masked training; autoregressive generation'
        return AutoTimesForecast(model.to(device), horizon, data, lookback), vars(args)
    if method == 'GPT4MTS':
        args.d_model = 768
        module = source_module('gpt4mts', 'models/GPT4MTS.py')
        # A declared batch=1 compatibility fix: mean leaves B,768,patch_num; squeeze wrongly removes B.
        source = (ROOT / 'third_party/code/gpt4mts/models/GPT4MTS.py').read_text(encoding='utf-8')
        old = 'summary.mean(dim=-1).squeeze()'
        if source.count(old) != 1:
            raise RuntimeError('Pinned GPT4MTS no longer matches its documented batch compatibility fix')
        namespace = dict(module.__dict__)
        exec(compile(source.replace(old, 'summary.mean(dim=-1)'), str(ROOT / 'third_party/code/gpt4mts/models/GPT4MTS.py'), 'exec'), namespace)
        with local_gpt2(), redirect_stdout(io.StringIO()):
            model = namespace['GPT4MTS'](args, device)
        model.gpt2.config.output_attentions = False
        model.gpt2.config.output_hidden_states = False
        model.gpt2.config.use_cache = False
        args.compatibility = 'Local GPT2 path; preserve batch dimension after summary mean'
        return GPTForecast(model.to(device), horizon, data, lookback), vars(args)
    if method == 'MM-TSFlib':
        module = source_module('mmtslib', 'models/iTransformer.py')
        numeric = NativeForecast(module.Model(args).to(device), horizon, args.label_len)
        mlp = author_class(ROOT / 'third_party/code/mmtslib/exp/exp_long_term_forecasting.py', 'MLP',
                           {'nn': torch.nn, 'F': torch.nn.functional})
        args.variant = 'MM-TSFlib iTransformer/GPT2 final_sum; causal history-only prior; contextual fact text max256'
        return MMD(numeric, mlp, horizon, data, lookback).to(device), vars(args)
    if method in {'ADAPT-Z', 'Text-Enhanced', 'On-Adapter', 'iAdapter-Frozen', 'DSOF', 'Proceed'}:
        from .online import build_online
        return build_online(method, settings, data, lookback, horizon, device)
    if method == 'VoT':
        raise RuntimeError('VoT event-prior provenance/backend is pending; no proxy score is substituted')
    raise ValueError(method)
