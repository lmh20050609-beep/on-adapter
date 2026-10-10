"""Isolate why the future-text perturbation check differs for On-Adapter."""
import json
import pathlib

import numpy as np
import torch

BASE = pathlib.Path(__file__).resolve().parent
ROOT = BASE.parent
CONFIG = ROOT / 'configs/benchmark.json'


def stack(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def main():
    from methods.config import load_config
    from methods.data import load_series, ForecastWindows
    from methods.prepared_text import prepare_text
    from methods.batching import host_batch, resident_vectors
    from methods.native import build
    from methods.provenance import resolve_settings
    from methods.worker import to_device

    cfg = load_config(CONFIG)
    domain, horizon, method = 'Traffic', 6, 'On-Adapter'
    device = torch.device('cuda:0')
    L = cfg['dataset_windows'][domain]['lookback']

    def fresh(cache, texts=None):
        data = load_series(domain, split=tuple(cfg['split']), cache_root=cache)
        if texts is not None:
            data.texts = list(texts)
        per_run = resolve_settings(cfg, method, len(data.values))
        per_run.update(active_method=method, active_seed=2025, _settings_input=cfg)
        per_run['execution'] = dict(per_run['execution'])
        per_run['execution']['cache_root'] = cache
        text = prepare_text(data, method, per_run, device)
        resident_vectors(text, device, cfg['execution'].get('device_data_bytes', 128 * 1024 ** 2))
        model, _ = build(method, per_run, data, L, horizon, device)
        model.eval()
        return data, text, model

    cache = 'E:/FedNoise/tpc_diag'
    data_a, text_a, model_a = fresh(cache)
    windows = ForecastWindows(data_a, 'test', L, horizon, method=method)
    probe = int(windows.indices[len(windows.indices) // 2])
    item = int(np.where(windows.indices == probe)[0][0])
    host = host_batch(windows, [item])
    batch = to_device(host, device)
    minimal = lambda b: {k: v for k, v in b.items() if k in ('x', 'mark', 'origin')}

    with torch.no_grad():
        p1 = model_a(minimal(batch)).detach().clone()
        p2 = model_a(minimal(batch)).detach().clone()
    print('repeat on same model identical:', torch.equal(p1, p2))

    texts = list(data_a.texts)
    perturbed = list(texts)
    for row in range(probe + 1, len(perturbed)):
        perturbed[row] = 'PERTURBED FUTURE CONTENT 2099'

    data_b, text_b, model_b = fresh('E:/FedNoise/tpc_diag_rebuild', texts=texts)
    model_b.load_state_dict(stack(model_a), strict=True)
    model_b.eval()
    with torch.no_grad():
        p3 = model_b(minimal(batch)).detach().clone()
    print('rebuilt-same-text identical:', torch.equal(p1, p3))

    data_c, text_c, model_c = fresh('E:/FedNoise/tpc_diag_pert', texts=perturbed)
    model_c.load_state_dict(stack(model_a), strict=True)
    model_c.eval()
    with torch.no_grad():
        p4 = model_c(minimal(batch)).detach().clone()
    print('perturbed identical:', torch.equal(p1, p4), 'maxdiff', float((p1 - p4).abs().max()))

    # Which part of the input differs?
    print('ids probe a/b/c:', int(text_a.ids[probe]), int(text_b.ids[probe]), int(text_c.ids[probe]))
    va = text_a.table.gather(np.asarray([probe]), device)
    vb = text_b.table.gather(np.asarray([probe]), device)
    vc = text_c.table.gather(np.asarray([probe]), device)
    print('gather equal a/b:', bool(torch.equal(va, vb)), 'a/c:', bool(torch.equal(va, vc)))
    print('available probe a/c:', text_a.available[probe], text_c.available[probe])
    print('confidence probe a/c:', text_a.confidence[probe], text_c.confidence[probe])
    print('texts[probe] equal:', texts[probe] == perturbed[probe])
    print('feature equal:', bool(torch.equal(va, vc)))





def replicate_checks_sequence():
    """Mirror checks.check_future_text_perturbation exactly to locate the flaw."""
    from methods.config import load_config
    from methods.data import load_series, ForecastWindows
    from methods.prepared_text import prepare_text
    from methods.batching import host_batch, resident_vectors
    from methods.native import build
    from methods.provenance import resolve_settings
    from methods.worker import to_device
    cfg = load_config(CONFIG)
    domain, horizon, method = 'Traffic', 6, 'On-Adapter'
    device = torch.device('cuda:0')
    L = cfg['dataset_windows'][domain]['lookback']
    data = load_series(domain, split=tuple(cfg['split']), cache_root=cfg['execution']['cache_root'])
    per_run = resolve_settings(cfg, method, len(data.values))
    per_run.update(active_method=method, active_seed=2025, _settings_input=cfg)
    text = prepare_text(data, method, per_run, device)
    resident_vectors(text, device, cfg['execution'].get('device_data_bytes', 128 * 1024 ** 2))
    model, _ = build(method, per_run, data, L, horizon, device)
    windows = ForecastWindows(data, 'test', L, horizon, method=method)
    probe = int(windows.indices[len(windows.indices) // 2])
    item = int(np.where(windows.indices == probe)[0][0])
    host = host_batch(windows, [item])
    batch = to_device(host, device)
    minimal = lambda b: {k: v for k, v in b.items() if k in ('x', 'mark', 'origin')}
    model.eval()
    with torch.no_grad():
        before = model(minimal(batch)).detach().clone()
    original = list(data.texts)
    for row in range(probe + 1, len(data.texts)):
        data.texts[row] = 'PERTURBED FUTURE CONTENT 2099'
    per_run['execution'] = dict(per_run['execution'])
    per_run['execution']['cache_root'] = 'E:/FedNoise/tpc_diag_repl'
    text2 = prepare_text(data, method, per_run, device)
    resident_vectors(text2, device, cfg['execution'].get('device_data_bytes', 128 * 1024 ** 2))
    model2, _ = build(method, per_run, data, L, horizon, device)
    model2.load_state_dict(model.state_dict(), strict=False)
    model2.eval()
    with torch.no_grad():
        after = model2(minimal(batch)).detach().clone()
    print('replica identical:', bool(torch.equal(before, after)),
          'maxdiff', float((before - after).abs().max()) if not torch.equal(before, after) else 0.0)
    # Diagnose: are the two models structurally identical after the copy?
    sd_a, sd_b = model.state_dict(), model2.state_dict()
    diff = [k for k in sd_a if k not in sd_b or not torch.equal(sd_a[k], sd_b[k])]
    print('state_dict keys differing after copy:', diff[:10], 'count', len(diff))
    print('encoder is data.prepared_text:', model.encoder is data.prepared_text,
          model2.encoder is data.prepared_text, text is text2)
    print('model.encoder is text2:', model.encoder is text2, 'model2.encoder is text2:', model2.encoder is text2)


if __name__ == '__main__':
    main()
    replicate_checks_sequence()
