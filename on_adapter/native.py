"""Load actual author model classes, without importing their training entrypoints."""
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import torch
from .config import ROOT, METHODS

NUMERIC = {'iTransformer', 'PatchTST', 'TimesNet', 'DLinear', 'Autoformer', 'TimeXer', 'SOFTS'}

def source_module(repo, relative):
    base = ROOT / 'third_party/code' / repo
    if not (base / 'SOURCE.json').is_file():
        raise FileNotFoundError('Run third_party/fetch_code.sh for ' + repo)
    # A worker process handles one author namespace, avoiding cross-repository layers collisions.
    for name in list(sys.modules):
        if name == 'layers' or name.startswith('layers.'):
            del sys.modules[name]
    external_prefix = str(ROOT / 'third_party/code').lower()
    active_prefix = str(base).lower()
    sys.path[:] = [p for p in sys.path if not str(p).lower().startswith(external_prefix)
                   or str(p).lower().startswith(active_prefix)]
    sys.path.insert(0, str(base))
    path = base / relative
    spec = importlib.util.spec_from_file_location('iadapter_author_' + repo, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def model_config(settings, data, lookback, horizon, method):
    n = len(data.columns)
    args = dict(settings)
    args.update(task_name='long_term_forecast', seq_len=lookback, label_len=lookback // 2,
                pred_len=horizon, enc_in=n, dec_in=n, c_out=n,
                features='MS' if method == 'TimeXer' else 'M', embed='timeF', freq='h',
                factor=3, activation='gelu', moving_avg=25, output_attention=False,
                individual=False, use_norm=True, d_core=settings['d_model'],
                top_k=3, num_kernels=6, num_class=2,
                patch_len=math.gcd(lookback, settings['patch_len']) if method == 'TimeXer' else settings['patch_len'],
                patch_size=settings['patch_len'], llm_dim=768,
                llm_path=str(ROOT / 'third_party/weights/gpt2'), llm_ckp_dir=str(ROOT / 'third_party/weights/gpt2'),
                prompt_domain=True, content='Historical ' + data.domain + ' observations; forecast all selected numerical variables.',
                pretrain=1, freeze=1, gpt_layers=settings['llm_layers'], is_gpt=1,
                revin=True, mlp_hidden=256, mlp_layers=2, covariate=False,
                token_len=12, use_gpu=True, gpu=0)
    return SimpleNamespace(**args)

def build(method, settings, data, lookback, horizon, device):
    if method not in NUMERIC:
        from .multimodal import build_multimodal
        return build_multimodal(method, settings, data, lookback, horizon, device)
    repo = METHODS[method][0]
    module = source_module(repo, 'models/' + method + '.py')
    args = model_config(settings, data, lookback, horizon, method)
    model = (module.Model(args, patch_len=args.patch_len, stride=args.stride)
             if method == 'PatchTST' else module.Model(args))
    return NativeForecast(model.to(device), horizon, args.label_len), vars(args)

class NativeForecast(torch.nn.Module):
    def __init__(self, model, horizon, label_len):
        super().__init__()
        self.author_model, self.horizon, self.label_len = model, horizon, label_len
        self.bf16 = False

    def forward(self, batch):
        x, mark = batch['x'], batch['mark']
        decoder = torch.cat([x[:, -self.label_len:], x.new_zeros((len(x), self.horizon, x.shape[-1]))], dim=1)
        # Future observation timestamps are not read; the decoder gets a repeated last historical mark.
        decoder_mark = torch.cat([mark[:, -self.label_len:], mark[:, -1:].expand(-1, self.horizon, -1)], dim=1)
        with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=self.bf16):
            output = self.author_model(x, mark, decoder, decoder_mark)
        if isinstance(output, tuple):
            output = output[0]
        if output is None or output.ndim != 3 or output.shape[1] != self.horizon:
            raise ValueError('Author model returned an unexpected forecasting shape')
        return output.float()
