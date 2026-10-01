"""Pinned author VoT PatchTST and mathematical helpers; no CLI or data-loader import."""
import ast
import json
import sys
from types import SimpleNamespace
from .config import ROOT

SOURCE = ROOT / 'external/vot'
WEIGHTS = ROOT / 'weights/gpt2'
COMMIT = 'f769621da9efde3e82a1975f478efdec83dcbd10'
HELPERS = ['get_model', 'get_text_emb', 'frequency_decomposition', 'correction',
           'contrastiveLoss_tr_sea', 'contrastiveLoss']

def source_methods():
    manifest = json.loads((SOURCE / 'SOURCE.json').read_text(encoding='utf-8-sig'))
    if manifest['commit'] != COMMIT:
        raise ValueError('Unexpected upstream VoT source revision')
    path = SOURCE / 'exp/exp_long_term_forecasting_clip.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == 'Exp_Long_Term_Forecast_Clip')
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    if not all(name in methods for name in HELPERS):
        raise ValueError('Author VoT helpers are incomplete')
    return path, [methods[name] for name in HELPERS]

def model_class():
    source_methods()  # verifies the existing upstream revision record
    path = SOURCE / 'models/PatchTST_clip.py'
    # Source imports only torch and the author's layers; no top-level experiments.
    sys.path.insert(0, str(SOURCE))
    try:
        namespace = {'__name__': 'iadapter.author_vot_patchtst'}
        exec(compile(path.read_text(encoding='utf-8'), str(path), 'exec'), namespace)
    finally:
        sys.path.pop(0)
    return namespace['Model']

def native_config(lookback=8, horizon=6):
    return SimpleNamespace(task_name='long_term_forecast', seq_len=lookback,
                           pred_len=horizon, enc_in=1, d_model=512, n_heads=8,
                           e_layers=2, d_ff=2048, factor=1, embed='timeF', freq='h',
                           dropout=.1, activation='gelu', output_attention=False,
                           multimodal=1, tr_sea=1, clip_t=0., llm_dim=768)

def helper_class():
    import numpy as np
    import torch
    from torch import nn
    path, methods = source_methods()
    # Executing only these exact definitions avoids unrelated sktime/CLI imports.
    cls = ast.ClassDef(name='AuthorVoTHelpers', bases=[], keywords=[], body=methods,
                       decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    namespace = {'__name__': 'iadapter.author_vot_helpers', 'np': np,
                 'torch': torch, 'nn': nn}
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace['AuthorVoTHelpers']

def build(lookback=8, horizon=6, device='cuda'):
    model = model_class()(native_config(lookback, horizon)).float().to(device)
    return model

def build_helpers(model, device='cuda'):
    from transformers import GPT2Config, GPT2Model, GPT2Tokenizer
    if not (WEIGHTS / 'model.safetensors').is_file():
        raise FileNotFoundError('Existing local GPT2 weights are required')
    config = GPT2Config.from_pretrained(WEIGHTS, local_files_only=True)
    config.num_hidden_layers = 6
    config.output_attentions = True
    config.output_hidden_states = True
    gpt = GPT2Model.from_pretrained(WEIGHTS, config=config, local_files_only=True,
                                  attn_implementation='eager').eval().to(device)
    gpt.requires_grad_(False)
    tokenizer = GPT2Tokenizer.from_pretrained(WEIGHTS, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    helper = helper_class()()
    helper.model, helper.llm_model = model, gpt
    helper.tokenizer, helper.device = tokenizer, device
    return helper
