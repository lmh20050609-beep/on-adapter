"""Preprocessing is computed once per CSV/split/code identity, never refitted."""
import json
import os
import shutil
from pathlib import Path
import numpy as np
from .resources import digest, identity, atomic_json, exclusive

ARRAYS = ['raw', 'values', 'mask', 'mean', 'scale', 'dates', 'available_dates', 'marks', 'label_values', 'history_values', 'standardized_truth']


def augment(data):
    standardized = ((data.raw - data.mean) / data.scale).astype(np.float32)
    data.standardized_truth = standardized
    data.label_values = np.nan_to_num(standardized, nan=0., posinf=0., neginf=0.)
    data.history_values = np.nan_to_num(standardized)
    data._window_indices = {}
    return data


def load_prepared_series(domain, data_root, split, cache_root):
    from .data import SeriesData, _load_series_uncached
    from .config import DOMAIN_FILES
    if cache_root is None:
        return augment(_load_series_uncached(domain, data_root, split))
    descriptor = dict(domain=domain, data_sha256=digest(Path(data_root)/DOMAIN_FILES[domain]), split=list(split),
                      code={n: digest(Path(__file__).parent/n) for n in ['data.py','data_cache.py','text_clean.py']})
    folder = Path(cache_root)/'data'/identity(descriptor)
    if not (folder/'manifest.json').exists():
        folder.parent.mkdir(parents=True, exist_ok=True)
        with exclusive(folder.with_suffix('.lock')):
            data = augment(_load_series_uncached(domain, data_root, split))
            temp = folder.with_name(folder.name+'.tmp.'+str(os.getpid())); temp.mkdir()
            try:
                for name in ARRAYS: np.save(temp/(name+'.npy'), getattr(data, name))
                meta = {n: getattr(data,n) for n in ['domain','texts','columns','train_end','val_end','path']}
                atomic_json(temp/'metadata.json', meta)
                atomic_json(temp/'manifest.json', dict(descriptor=descriptor,
                    files={p.name:digest(p) for p in temp.iterdir() if p.is_file()}))
                os.replace(temp, folder)
            except BaseException:
                shutil.rmtree(temp, ignore_errors=True); raise
    manifest = json.loads((folder/'manifest.json').read_text('utf-8'))
    if manifest['descriptor'] != descriptor: raise ValueError('Prepared data identity mismatch')
    for name, value in manifest['files'].items():
        if digest(folder/name) != value: raise ValueError('Prepared data corrupted: '+name)
    meta = json.loads((folder/'metadata.json').read_text('utf-8'))
    arrays = {name:np.load(folder/(name+'.npy')) for name in ARRAYS}
    labels = arrays.pop('label_values'); history = arrays.pop('history_values')
    standardized_truth = arrays.pop('standardized_truth')
    data = SeriesData(**meta, **arrays)
    data.label_values, data.history_values = labels, history
    data.standardized_truth = standardized_truth
    data._cache_folder = folder; data._window_indices = {}
    return data


def window_indices(data, split, lookback, horizon):
    key = (split, lookback, horizon)
    if key not in data._window_indices:
        folder = getattr(data, '_cache_folder', None)
        path = folder/('origins_%s_L%d_H%d.npy' % key) if folder else None
        if path and path.exists():
            indices = np.load(path)
        else:
            candidates = data.origins(split, lookback, horizon)
            cumulative = np.r_[0, np.cumsum(data.mask.any(axis=1), dtype=np.int64)]
            indices = candidates[(cumulative[candidates+1+horizon]-cumulative[candidates+1]) > 0]
            if path:
                temp = path.with_name(path.name+'.tmp.'+str(os.getpid()))
                with temp.open('wb') as stream: np.save(stream, indices)
                os.replace(temp, path)
        # Verify cached indices against the same observed-label rule, no blind trust.
        expected = data.origins(split, lookback, horizon)
        cumulative = np.r_[0, np.cumsum(data.mask.any(axis=1), dtype=np.int64)]
        expected = expected[(cumulative[expected+1+horizon]-cumulative[expected+1]) > 0]
        if not np.array_equal(indices, expected): raise ValueError('Window-index cache mismatch')
        data._window_indices[key] = indices
    return data._window_indices[key].copy()
