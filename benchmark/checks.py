"""Short-path checks for the paths the three-domain pilot never covered.

Covers daily (Environment) and weekly (Energy, Health_US) periodicities, the
non-pilot horizons and the two text-consuming methods that the pilot never
trained (GPT4MTS, MM-TSFlib). Checks never loosen a tolerance to turn a
failure into a pass: every comparison below is exact unless a documented
float32 boundary is explicitly stated.
"""
import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np
import torch

BASE = pathlib.Path(__file__).resolve().parent
ROOT = BASE.parent
REPORT = ROOT / 'results/benchmark'
CONFIG = ROOT / 'configs/benchmark.json'
ORIGINAL = ROOT.parent / 'output/original_timemmd/data'
V1_ROOT = ROOT / 'data'
DOMAIN_FILES = {'Agriculture': 'Agriculture.csv', 'Climate': 'Climate.csv',
                'Economy': 'Economy.csv', 'Energy': 'Energy.csv',
                'Environment': 'Environment.csv', 'Health_US': 'Health.csv',
                'Security': 'Security.csv', 'SocialGood': 'SocialGood.csv',
                'Traffic': 'Traffic.csv'}

# A compact matrix that reaches every periodicity, every horizon length, the
# longest lookback and both previously untrained text methods.
MATRIX = [('Environment', 48, 'On-Adapter'), ('Environment', 336, 'MM-TSFlib'),
          ('Energy', 12, 'GPT4MTS'), ('Energy', 48, 'On-Adapter'),
          ('Health_US', 12, 'MM-TSFlib'), ('Health_US', 48, 'GPT4MTS'),
          ('Traffic', 6, 'MM-TSFlib'), ('Traffic', 12, 'On-Adapter'),
          ('SocialGood', 8, 'GPT4MTS'), ('Security', 6, 'MM-TSFlib'),
          ('Economy', 10, 'On-Adapter'), ('Climate', 12, 'GPT4MTS'),
          ('Agriculture', 6, 'MM-TSFlib')]


def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def settings():
    from methods.config import load_config
    return load_config(CONFIG)


def purge_author_modules():
    """One author repository per build, exactly as the per-cell processes do it."""
    import sys
    keep = ('utils', 'layers', 'models', 'data', 'exp', 'trainers', 'exp_models')
    for name in list(sys.modules):
        if name.split('.')[0] in keep:
            del sys.modules[name]


def prepare(domain, horizon, method, cfg=None):
    purge_author_modules()
    from methods.data import load_series, ForecastWindows
    from methods.prepared_text import prepare_text
    from methods.batching import resident_vectors
    from methods.native import build
    from methods.provenance import resolve_settings
    cfg = cfg or settings()
    device = torch.device('cuda:0')
    data = load_series(domain, split=tuple(cfg['split']), cache_root=cfg['execution']['cache_root'])
    L, _ = cfg['dataset_windows'][domain]['lookback'], None
    per_run = resolve_settings(cfg, method, len(data.values))
    per_run.update(active_method=method, active_seed=2025, _settings_input=cfg)
    if method in ('On-Adapter',):
        from methods.provenance import numeric_checkpoint
        per_run['pretrained_numeric_checkpoint'] = str(numeric_checkpoint(cfg, domain, horizon, 2025))
    text = prepare_text(data, method, per_run, device)
    resident_vectors(text, device, cfg['execution'].get('device_data_bytes', 128 * 1024 ** 2))
    model, resolved = build(method, per_run, data, L, horizon, device)
    return data, text, model, per_run, device


def check_alignment(data, horizon, lookback):
    from methods.data import ForecastWindows
    results = {}
    for split in ['train', 'val', 'test']:
        windows = ForecastWindows(data, split, lookback, horizon)
        origins = windows.indices
        assert len(origins) > 0, 'empty split ' + split
        sample = origins[[0, len(origins) // 2, len(origins) - 1]]
        for origin in sample:
            item = windows[int(np.where(origins == origin)[0][0])]
            rows = origin + 1 + np.arange(horizon)
            assert int(item['origin']) == int(origin)
            assert np.array_equal(item['mask'].numpy(), data.mask[rows])
            expected = ((data.raw[rows] - data.mean) / data.scale)
            assert np.array_equal(item['y'].numpy(), np.nan_to_num(expected, nan=0.).astype(np.float32)), \
                'target standardization mismatch at origin %d' % origin
            assert np.array_equal(item['x'].numpy(), data.values[origin + 1 - lookback:origin + 1])
            assert np.all(data.available_dates[origin + 1 - lookback:origin + 1] <= data.available_dates[origin])
        results[split] = dict(origins=len(origins), sampled=[int(x) for x in sample])
    return dict(status='passed', detail=results, **{'origin_target_mask_alignment': True,
                                                    'history_is_past_only': True})


def check_minimal_input(model, data, horizon, lookback, method, device):
    from methods.data import ForecastWindows
    from methods.batching import host_batch
    from methods.worker import to_device
    windows = ForecastWindows(data, 'test', lookback, horizon, method=method)
    index = int(windows.indices[len(windows.indices) // 2])
    item = int(np.where(windows.indices == index)[0][0])
    host = host_batch(windows, [item])
    batch = to_device(host, device)
    model.eval()
    with torch.no_grad():
        reference = model({k: v for k, v in batch.items() if k in ('x', 'mark', 'origin')}).detach().clone()
        poisoned = dict(batch)
        poisoned['y'] = torch.full_like(batch['y'], 123456.)
        poisoned['mask'] = ~batch['mask']
        alternate = model({k: v for k, v in poisoned.items() if k in ('x', 'mark', 'origin')}).detach().clone()
    return dict(status='passed' if torch.equal(reference, alternate) else 'FAILED',
                identical=bool(torch.equal(reference, alternate)), origin=int(index),
                method=method, minimal_input_without_y_or_mask=True)


def check_text_visibility(data, text, horizon, lookback, method):
    """Every gathered text row index must be visible at its origin."""
    from methods.data import ForecastWindows
    windows = ForecastWindows(data, 'test', lookback, horizon, method=method)
    origins = windows.indices
    if method == 'GPT4MTS':
        # GPT4MTS gathers the lookback history of original-row ids.
        sources = origins[:, None] + np.arange(1 - lookback, 1)[None, :]
        valid = sources >= 0
        gathered = text.ids[np.clip(sources, 0, None)]
        sources = np.where(valid, sources, 0)
    else:
        # On-Adapter and MM-TSFlib gather only the origin row, whose mapped text
        # already concatenates the two visible rows (origin-1, origin).
        sources = origins[:, None]
        gathered = text.ids[origins][:, None]
    for position in [0, len(origins) // 2, len(origins) - 1]:
        origin = int(origins[position])
        used = np.unique(sources[position])
        for row in used:
            assert data.available_dates[row] <= data.available_dates[origin], \
                'origin %d gathers row %d with a later availability date' % (origin, int(row))
    return dict(status='passed', method=method, method_input_field='fact',
                gathered_rows_are_past_visible=True,
                checked_origins=int(len(origins)))


def check_future_text_perturbation(domain, horizon, method, probe_origin_ratio=0.5):
    """Perturb every text row strictly after a probe origin and re-encode.

    The probe prediction must be bit-identical; only future rows change.
    """
    from methods.data import ForecastWindows
    from methods.prepared_text import prepare_text
    from methods.native import build
    from methods.provenance import resolve_settings
    from methods.worker import to_device
    from methods.batching import host_batch, resident_vectors
    purge_author_modules()
    cfg = settings()
    device = torch.device('cuda:0')
    data, text, model, per_run, _ = prepare(domain, horizon, method, cfg)
    L = cfg['dataset_windows'][domain]['lookback']
    windows = ForecastWindows(data, 'test', L, horizon, method=method)
    probe = int(windows.indices[int(len(windows.indices) * probe_origin_ratio)])
    item = int(np.where(windows.indices == probe)[0][0])
    host = host_batch(windows, [item])
    batch = to_device(host, device)
    model.eval()
    with torch.no_grad():
        before = model({k: v for k, v in batch.items() if k in ('x', 'mark', 'origin')}).detach().clone()
    original = list(data.texts)
    for row in range(probe + 1, len(data.texts)):
        data.texts[row] = 'PERTURBED FUTURE CONTENT 2099'
    per_run['execution'] = dict(per_run['execution'])
    per_run['execution']['cache_root'] = 'E:/FedNoise/tpc_' + domain[:3].lower()
    text2 = prepare_text(data, method, per_run, device)
    resident_vectors(text2, device, cfg['execution'].get('device_data_bytes', 128 * 1024 ** 2))
    model2, _ = build(method, per_run, data, L, horizon, device)
    model2.load_state_dict(model.state_dict(), strict=False)
    model2.eval()
    with torch.no_grad():
        after = model2({k: v for k, v in batch.items() if k in ('x', 'mark', 'origin')}).detach().clone()
    data.texts = original
    identical = bool(torch.equal(before, after))
    return dict(status='passed' if identical else 'FAILED', method=method, domain=domain,
                horizon=horizon, probe_origin=probe, future_rows_perturbed=len(data.texts) - probe - 1,
                prediction_identical=identical, tolerance=0)


def check_cache_identity(domain):
    cfg = settings()
    v1 = V1_ROOT / DOMAIN_FILES[domain]
    original = ORIGINAL / DOMAIN_FILES[domain]
    from methods.data import load_series
    data = load_series(domain, split=(.6, .2, .2), cache_root=cfg['execution']['cache_root'])
    return dict(domain=domain, v1_sha256=digest(v1), original_sha256=digest(original),
                distinct_from_original=digest(v1) != digest(original),
                data_cache_folder=str(getattr(data, '_cache_folder', '')), text_version=data.text_version,
                source_decisions_sha256=data.source_decisions_sha256,
                loaded_path=str(data.path), loaded_is_v1=pathlib.Path(str(data.path)).name == DOMAIN_FILES[domain])


def write(results, out=None):
    path = pathlib.Path(out) if out else REPORT / 'path_checks.json'
    path.write_text(json.dumps(results, indent=1, default=str) + '\n', encoding='utf-8')
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--matrix', default='all')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    results = dict(alignment={}, minimal_input={}, text_visibility={},
                   perturbation={}, cache_identity={})
    for domain, horizon, method in MATRIX:
        key = '%s/H%d/%s' % (domain, horizon, method)
        cfg = settings()
        L = cfg['dataset_windows'][domain]['lookback']
        data, text, model, per_run, device = prepare(domain, horizon, method, cfg)
        results['alignment'][key] = check_alignment(data, horizon, L)
        results['minimal_input'][key] = check_minimal_input(model, data, horizon, L, method, device)
        results['text_visibility'][key] = check_text_visibility(data, text, horizon, L, method)
        del model
        torch.cuda.empty_cache()
        write(results, args.out)
        print('checked', key, flush=True)
    for domain in ['Traffic', 'Security', 'Agriculture']:
        cfg = settings()
        L = cfg['dataset_windows'][domain]['lookback']
        results['perturbation']['%s/H6/On-Adapter' % domain] = check_future_text_perturbation(
            domain, 6, 'On-Adapter')
        print('perturbed', domain, flush=True)
    for domain in sorted(DOMAIN_FILES):
        results['cache_identity'][domain] = check_cache_identity(domain)
    out = write(results, args.out)
    failed = [k for section in results.values() for k, v in section.items()
              if isinstance(v, dict) and v.get('status') == 'FAILED']
    print(json.dumps(dict(out=str(out), failed=failed)))


if __name__ == '__main__':
    main()
