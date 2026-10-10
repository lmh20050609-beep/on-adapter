"""Shared On-Adapter replay harness for online-rule diagnostics (2026-10-08).

Read-only with respect to historical runs and caches: a cache miss aborts instead of writing.
Rebuilds one historical On-Adapter cell from its own config.json / best.pt, then runs the
origin-by-origin online loop of online.evaluate_online on a chosen split with a chosen
online learning rate, recording every forecast and every update.

The loop body mirrors evaluate_online for ResidualOnline line by line. Extra mechanism
measurements use torch.autograd.grad (no RNG use, no .grad side effects), and the
equivalence with the package function is checked bitwise in replay_consistency_20261008.py.
"""
import importlib
import json
import random
import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

METHOD = 'On-Adapter'
# execution_version recorded in each run's config.json -> package that produced it
PACKAGES = {'textrepair-v1': 'methods', 'v1conf': 'on_adapter'}


def cvar(origin_losses, q=.9):
    v = np.sort(np.asarray(origin_losses, dtype=np.float64))[::-1]
    mass = (1. - q) * len(v)
    whole = int(np.floor(mass))
    total = v[:whole].sum() + (mass - whole) * (v[whole] if whole < len(v) else 0.)
    return float(total / mass)


def scores(pred, truth, mask):
    pred, truth = np.asarray(pred, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    mask = np.asarray(mask).astype(bool)
    err = pred - truth
    per_origin = np.array([np.abs(e[m]).mean() for e, m in zip(err, mask) if m.any()])
    return dict(mse=float(np.mean(err[mask] ** 2)), mae=float(np.mean(np.abs(err[mask]))),
                cvar90=cvar(per_origin, .9), cvar80=cvar(per_origin, .8), cvar95=cvar(per_origin, .95))


def module(ctx, name):
    return importlib.import_module(ctx['pkg'] + '.' + name)


def setup(run_dir, device='cpu', threads=2):
    torch.set_num_threads(threads)
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / 'config.json').read_text('utf-8'))
    result = json.loads((run_dir / 'result.json').read_text('utf-8'))
    pkg = PACKAGES[meta['execution_version']]
    m = lambda name: importlib.import_module(pkg + '.' + name)
    data_mod, config, provenance, resources = m('data'), m('config'), m('provenance'), m('resources')
    settings = meta['settings_input']
    domain, horizon, seed = meta['domain'], meta['horizon'], meta['seed']

    descriptor = dict(domain=domain, data_sha256=resources.digest(Path(data_mod.DATA_ROOT) / config.DOMAIN_FILES[domain]),
                      split=list(settings['split']),
                      code={n: resources.digest(Path(data_mod.__file__).parent / n) for n in ['data.py', 'data_cache.py', 'text_clean.py']})
    folder = Path(settings['execution']['cache_root']) / 'data' / resources.identity(descriptor)
    if not (folder / 'manifest.json').exists():
        raise SystemExit('Prepared data cache missing; refusing to write historical cache')
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    data = data_mod.load_series(domain, split=tuple(settings['split']), cache_root=settings['execution']['cache_root'])
    L, hs = config.configured_windows(settings)[domain]
    assert L == meta['lookback'] and horizon in hs
    cfg = provenance.resolve_settings(settings, METHOD, len(data.values))
    cfg.update(run_kind=settings['campaign'], active_seed=seed, active_method=METHOD, _settings_input=settings)
    saved = {k: v for k, v in meta['settings'].items() if k != '_settings_input'}
    if saved != json.loads(json.dumps({k: v for k, v in cfg.items() if k != '_settings_input'})):
        raise SystemExit('Resolved settings differ from the run record')
    dev = torch.device(device)
    cfg['device'] = str(dev)
    sets = {s: data_mod.ForecastWindows(data, s, L, horizon, method=METHOD) for s in ['train', 'val', 'test']}
    encode_device = torch.device(settings['device'])  # cache identity only; a hit loads no encoder
    if not text_table_exists(m, data, settings, encode_device):
        raise SystemExit('Text cache missing; refusing to write historical cache')
    text = m('prepared_text').prepare_text(data, METHOD, settings, encode_device)
    assert text.stats['table_hit'] == 1
    m('batching').resident_vectors(text, dev, settings['execution'].get('device_data_bytes', 128 * 1024 ** 2))
    model, _ = m('native').build(METHOD, cfg, data, L, horizon, dev)
    best = torch.load(run_dir / 'best.pt', map_location=dev, weights_only=True)
    missing, unexpected = model.load_state_dict(best, strict=False)
    if unexpected or any(k.startswith('adaptation.') for k in missing):
        raise SystemExit('best.pt does not match the rebuilt adapter')
    return dict(run_dir=run_dir, meta=meta, result=result, data=data, sets=sets, cfg=cfg, model=model,
                best=best, L=L, horizon=horizon, seed=seed, domain=domain, pkg=pkg, device=dev)


def text_table_exists(m, data, settings, device):
    """Same descriptor/key as the package's _prepare_text (not its text_cache_ready helper,
    which in textrepair-v1 still names the chunk field encoding_batch_size)."""
    pt, res = m('prepared_text'), m('resources')
    source = Path(pt.__file__).read_text('utf-8')
    field = 'table_build_chunk_size' if 'table_build_chunk_size=options' in source else 'encoding_batch_size'
    options = settings['execution']
    with res.isolated_rng():
        texts, _ = pt.text_plan(data, METHOD)
        desc = pt.descriptor(data, METHOD)
        desc.update({'encoding_device': str(device), field: options.get('text_encoding_batch_size', 16),
                     'encoding_precision': 'float32', 'allow_tf32': bool(torch.backends.cuda.matmul.allow_tf32),
                     'deterministic_cudnn': bool(torch.backends.cudnn.deterministic)})
        kind = 'tokens' if desc['pooling'] == 'tokens' else 'vectors'
        key = res.identity({'descriptor': desc, 'texts': texts})
    return (Path(options['cache_root']) / 'text' / kind / key / 'manifest.json').exists()


def entry_state(ctx, seed=None):
    """Controlled entry: fixed seeds; returns the exact CPU torch RNG state used."""
    seed = ctx['seed'] if seed is None else seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    return torch.get_rng_state().numpy().copy()


def flat_params(params):
    return torch.cat([p.detach().reshape(-1) for p in params])


def run_online(ctx, split, lr, measure=True, rng_seed=None, online_aux=True, anchor_lambda=0.0):
    """Delayed-feedback online loop of evaluate_online on `split`.

    lr == 0: gradients and hg/ge feedback are computed exactly as usual but the
    parameter step is not executed (no parameter update).
    Returns arrays dict and per-update event list.
    """
    batching, loss_ot = module(ctx, 'batching'), module(ctx, 'worker').loss_ot
    model, dev, H = ctx['model'], ctx['device'], ctx['horizon']
    model.load_state_dict(ctx['best'], strict=False)
    rng = entry_state(ctx, rng_seed)
    model.reset_online()
    params = [p for p in model.parameters() if p.requires_grad]
    named = [n for n, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)
    theta0 = flat_params(params)
    anchors = [p.detach().clone() for p in params]  # offline weights for the optional L2-SP anchor
    pending = deque()
    rec = {k: [] for k in ['pred', 'base', 'y', 'mask', 'origin', 'hg_norm', 'ge_norm', 'correction']}
    events = []
    windows = ctx['sets'][split]
    for item in range(len(windows)):
        host = batching.host_batch(windows, [item])
        batch = batching.transfer(host, dev)
        origin = int(batch['origin'][0])
        while pending and pending[0][0] + H <= origin:
            old_origin, old_batch, old_feedback = pending.popleft()
            model.train()
            saved = (model.hg.clone(), model.ge.clone())
            model.hg.copy_(old_feedback[0]); model.ge.copy_(old_feedback[1])
            optimizer.zero_grad(set_to_none=True)
            loss, _ = loss_ot(model, old_batch)
            probe = batching.observed_error(model.last_base, old_batch).square().mean()
            z_grad = torch.autograd.grad(probe, model.last_feature, retain_graph=True)[0].mean(0)
            text_grad = None
            if model.last_text is not None and model.last_text.requires_grad:
                text_grad = torch.autograd.grad(loss, model.last_text, retain_graph=True, allow_unused=True)[0]
            aux = model.auxiliary_loss() if online_aux else loss.new_zeros(())  # online_aux=False: task loss only
            event = dict(forecast_origin=origin, updated_origin=old_origin, last_label_row=old_origin + H,
                         task_loss=float(loss.detach()), aux_loss=float(aux.detach()))
            if measure:
                g_task = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
                event['task_grad_norm'] = float(torch.sqrt(sum((g.double() ** 2).sum() for g in g_task if g is not None)))
                if aux.requires_grad:
                    g_aux = torch.autograd.grad(aux, params, retain_graph=True, allow_unused=True)
                    event['aux_grad_norm'] = float(torch.sqrt(sum((g.double() ** 2).sum() for g in g_aux if g is not None)))
                else:
                    event['aux_grad_norm'] = 0.
            loss = loss + aux
            if anchor_lambda:
                loss = loss + anchor_lambda * sum(((p - q) ** 2).sum() for p, q in zip(params, anchors))
            loss.backward()
            event['total_grad_norm_preclip'] = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.))
            before = flat_params(params) if measure else None
            if lr > 0:
                optimizer.step()
            if measure:
                after = flat_params(params)
                event['step_norm'] = float((after - before).norm())
                event['drift_norm'] = float((after - theta0).norm())
                event['relative_drift'] = event['drift_norm'] / float(theta0.norm())
            model.hg.copy_(model.momentum * saved[0] + (1 - model.momentum) * z_grad.detach())
            model.ge.copy_(text_grad.detach().mean(0) if text_grad is not None else saved[1])
            events.append(event)
        snapshot = (model.hg.detach().clone(), model.ge.detach().clone())
        rec['hg_norm'].append(float(model.hg.norm())); rec['ge_norm'].append(float(model.ge.norm()))
        model.eval()
        with torch.no_grad():
            safe = dict(batch); safe['y'] = torch.zeros_like(batch['y']); safe['mask'] = torch.zeros_like(batch['mask'])
            prediction = model(safe)
            base = model.last_base.detach()
        pending.append((origin, batch, snapshot))
        rec['pred'].append(prediction.cpu().numpy()); rec['base'].append(base.cpu().numpy())
        rec['y'].append(host['y'].numpy()); rec['mask'].append(host['mask'].numpy())
        rec['origin'].append(host['origin'].numpy())
        rec['correction'].append(float((prediction - base).abs().mean()))
    arrays = {k: (np.concatenate(v) if k in ('pred', 'base', 'y', 'mask', 'origin') else np.asarray(v))
              for k, v in rec.items()}
    arrays['entry_torch_rng'] = rng
    arrays['final_params'] = flat_params(params).cpu().numpy()
    return arrays, events, named
