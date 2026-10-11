"""Frozen text encoding runs before model training, never inside forward."""
import json
import re
from pathlib import Path
import numpy as np
import torch
from .config import ROOT
from .feature_store import Budget, build_table
from .resources import digest, identity, isolated_rng

SPECS = {
    'GPT4MTS': dict(model='bert-base-uncased', layers=None, max_length=512, pooling='cls'),
    'On-Adapter': dict(model='bert-base-uncased', layers=6, max_length=64, pooling='mean'),
    'MM-TSFlib': dict(model='gpt2', layers=6, max_length=256, pooling='tokens'),
}


def origin_text(data, origin):
    rows = [r for r in range(max(0, origin - 1), origin + 1)
            if data.available_dates[r] <= data.available_dates[origin] and data.texts[r].strip()]
    return '\n'.join(data.texts[r] for r in rows[-2:])


def text_plan(data, method):
    if method == 'GPT4MTS':
        mapped = [re.sub(r'[^a-zA-Z0-9 ]+', '', text) if text.strip()
                  else 'Predict the future time step given the past' for text in data.texts]
    else:
        mapped = [origin_text(data, i) for i in range(len(data.values))]
    # Stable first-appearance order; identities do not depend on H or seed.
    lookup = {}; texts = []; ids = []
    for text in mapped:
        if text not in lookup:
            lookup[text] = len(texts); texts.append(text)
        ids.append(lookup[text])
    return texts, np.asarray(ids, dtype=np.int64)


def descriptor(data, method):
    spec = SPECS[method]
    weights = ROOT / 'weights' / spec['model']
    files = {p.name: digest(p) for p in sorted(weights.iterdir())
             if p.is_file() and p.suffix in {'.bin', '.safetensors', '.json', '.txt', '.model'}}
    if not any(n.endswith(('.bin', '.safetensors')) for n in files):
        raise FileNotFoundError('Offline pretrained weights are required: ' + str(weights))
    import transformers
    return dict(spec, format='domain-table-v1', domain=data.domain, data_sha256=digest(data.path),
                preprocessing={name: digest(Path(__file__).parent/name) for name in ['data.py', 'data_cache.py', 'text_clean.py', 'prepared_text.py', 'feature_store.py']},
                weights=files, storage='FP32-to-FP16-to-FP32, before downstream projection',
                padding='right; attention mask; valid tokens only; no cached trainable layers',
                transformers_version=transformers.__version__, torch_version=torch.__version__,
                numerical_policy='batched encoder; equivalence must be measured; distinct campaign')


class FrozenBatchEncoder:
    def __init__(self, spec, device='cpu'):
        self.spec = spec; self.device = torch.device(device)
        self.tokenizer = None; self.model = None

    def tokenizer_load(self):
        if self.tokenizer is None:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(ROOT/'weights'/self.spec['model'], local_files_only=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = 'right'

    def lengths(self, texts, batch_size):
        self.tokenizer_load()
        result = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start+batch_size]
            ids = self.tokenizer(batch, truncation=True, max_length=self.spec['max_length'], padding=False)['input_ids']
            result.extend(len(v) if t.strip() else 1 for t, v in zip(batch, ids))
        return result

    def load(self):
        self.tokenizer_load()
        if self.model is None:
            from transformers import AutoModel
            # Preserve the pinned eager kernel and original load-then-truncate path.
            self.model = AutoModel.from_pretrained(ROOT/'weights'/self.spec['model'], local_files_only=True,
                                                   attn_implementation='eager').eval().to(self.device)
            layers = self.spec['layers']
            if self.spec['model'] == 'gpt2': self.model.h = self.model.h[:layers or 6]
            elif layers is not None: self.model.encoder.layer = self.model.encoder.layer[:layers]
            self.model.requires_grad_(False)

    @torch.no_grad()
    def encode(self, texts):
        result = [np.zeros((1, 768), dtype=np.float32) for _ in texts]
        active = [i for i, text in enumerate(texts) if text.strip()]
        if not active: return result
        self.load()
        encoded = self.tokenizer([texts[i] for i in active], return_tensors='pt', padding=True,
                                 truncation=True, max_length=self.spec['max_length'])
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        kwargs = dict(output_attentions=False, output_hidden_states=False)
        if self.spec['model'] == 'gpt2': kwargs['use_cache'] = False
        output = self.model(**encoded, **kwargs).last_hidden_state
        lengths = encoded['attention_mask'].sum(-1).cpu().tolist()
        for row, index in enumerate(active):
            valid = output[row, :lengths[row]]
            pooling = self.spec['pooling']
            if pooling == 'cls': valid = valid[:1]
            elif pooling == 'mean': valid = valid.mean(0, keepdim=True)
            result[index] = valid.float().cpu().numpy()
        return result

    def release(self):
        self.model = None; self.tokenizer = None
        if self.device.type == 'cuda': torch.cuda.empty_cache()


class PreparedText:
    def __init__(self, data, method, table, ids, stats):
        self.data, self.method, self.table, self.ids, self.stats = data, method, table, ids, stats
        self.available = np.array([bool(origin_text(data, i)) for i in range(len(data.values))]) if method == 'On-Adapter' else None
        self._history = {}
        confidence = []; c = 0.
        for text in data.texts:
            available = float(bool(text.strip())); c = .7*c + .3*available
            confidence.append(max(c, .25*available))
        self.confidence = np.asarray(confidence, dtype=np.float32)

    def gpt(self, origins, lookback, device):
        origins = cpu_origins(origins)
        if lookback not in self._history:
            # Table indices are prepared once, not rebuilt or hashed per forward.
            rows = np.arange(len(self.ids))[:, None] + np.arange(1-lookback, 1)[None, :]
            valid = rows >= 0; clipped = rows.clip(min=0)
            visible = self.data.available_dates[clipped] <= self.data.available_dates[:, None]
            # Original loader requires monotone availability; a violation is explicit.
            if not np.all(visible[valid]): raise ValueError('Noncausal historical text mapping')
            self._history[lookback] = self.ids[clipped]
        if np.any(origins < lookback-1): raise ValueError('Insufficient historical text')
        return self.table.gather(self._history[lookback][origins], device)

    def pooled(self, origins, device):
        origins = cpu_origins(origins)
        return self.table.gather(self.ids[origins], device)

    def tokens(self, origins, device):
        origins = cpu_origins(origins)
        return self.table.gather(self.ids[origins], device)


def cpu_origins(origins):
    if isinstance(origins, torch.Tensor):
        if origins.device.type != 'cpu': raise ValueError('Origins must remain on CPU')
        return origins.detach().numpy().astype(np.int64, copy=False)
    return np.asarray(origins, dtype=np.int64)


def prepare_text(data, method, settings, device):
    # Include lazy library imports, tokenizer loading and metadata setup in the
    # isolation boundary, not only the neural-network encode calls.
    with isolated_rng():
        return _prepare_text(data, method, settings, device)


def _prepare_text(data, method, settings, device):
    if method not in SPECS: return None
    options = settings.get('execution', {})
    root = options['cache_root']
    budget = Budget(**options.get('cache_budget', {}))
    texts, ids = text_plan(data, method)
    desc = descriptor(data, method)
    desc.update(encoding_device=str(device),encoding_batch_size=options.get('text_encoding_batch_size',16),
                encoding_precision='float32',allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
                deterministic_cudnn=bool(torch.backends.cudnn.deterministic))
    encoder = FrozenBatchEncoder(SPECS[method], device)
    try:
        with isolated_rng():
            table, stats = build_table(Path(root)/'text', desc, texts, encoder, budget,
                                       options.get('text_encoding_batch_size', 16))
    finally: encoder.release()
    data.prepared_text = PreparedText(data, method, table, ids, stats)
    return data.prepared_text
