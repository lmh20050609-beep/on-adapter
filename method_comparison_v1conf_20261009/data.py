"""Shared chronological data and OT scoring; missing truth is never imputed."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from .config import DATA_ROOT, DOMAIN_FILES

MEASUREMENTS = {
    'Agriculture': ['Wholesale broiler composite', 'Retail-wholesale spread for broiler composite'],
    'Climate': ['Anomaly'], 'Economy': ['Exports', 'Imports'],
    'Environment': ['Number of Sites Reporting'],
    'Health_US': ['% WEIGHTED ILI', 'AGE 0-4', 'AGE 25-49', 'AGE 25-64', 'AGE 5-24',
                  'AGE 50-64', 'AGE 65', 'ILITOTAL', 'NUM. OF PROVIDERS', 'TOTAL PATIENTS'],
    'Weather': ['MINTEMP', 'MAXTEMP', 'HUMIDITY'],
    'Security': [], 'SocialGood': [], 'Traffic': [],
}

@dataclass
class SeriesData:
    domain: str
    raw: np.ndarray
    values: np.ndarray
    mask: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    dates: np.ndarray
    available_dates: np.ndarray
    marks: np.ndarray
    texts: list
    columns: list
    train_end: int
    val_end: int
    path: str

    @property
    def target_index(self):
        return self.columns.index('OT')

    def origins(self, split, lookback, horizon):
        if split == 'train':
            start, end = lookback - 1, self.train_end
        elif split == 'val':
            start, end = self.train_end - 1, self.val_end
        elif split == 'test':
            start, end = self.val_end - 1, len(self.values)
        else:
            raise ValueError(split)
        if start + horizon >= end or start < lookback - 1:
            raise ValueError('Not enough records for %s L%d H%d' % (split, lookback, horizon))
        return np.arange(start, end - horizon, dtype=np.int64)

    def metadata(self):
        return {'domain': self.domain, 'source': self.path, 'rows': len(self.values),
                'columns': self.columns, 'target': 'all_selected_columns', 'train_end': self.train_end,
                'val_end': self.val_end, 'mean': self.mean.tolist(), 'scale': self.scale.tolist(),
                'missing_OT': int((~self.mask[:, self.target_index]).sum()),
                'text_cutoff': 'end_date proxy at numerical observation availability',
                'calendar_intervals': 'reported-record sequence; five Health YEAR/WEEK calendar pairs overlap; Weather is not resampled',
                'metric_space': 'training-standardized all selected numeric columns; OT diagnostic separately'}

def _load_series_uncached(domain, data_root=DATA_ROOT, split=(.6, .2, .2)):
    path = Path(data_root) / DOMAIN_FILES[domain]
    frame = pd.read_csv(path, low_memory=False)
    dates = pd.to_datetime(frame['end_date'] if 'end_date' in frame else frame['date'], errors='raise')
    frame = frame.assign(_date=dates).sort_values('_date', kind='stable').reset_index(drop=True)
    dates = frame['_date']
    if dates.duplicated().any():
        if domain != 'Health_US' or int(dates.duplicated().sum()) != 5:
            raise ValueError('%s contains undeclared duplicate observation dates' % domain)
        duplicated = frame[dates.duplicated(keep=False)]
        for _, group in duplicated.groupby('_date'):
            if group['WEEK'].tolist() != [53, 1] or len(group) != 2:
                raise ValueError('Health duplicates no longer match the declared YEAR/WEEK pairs')
    n = len(frame)
    train_end, val_end = int(n * split[0]), int(n * (split[0] + split[1]))
    # Native GPT4MTS/VoT and numerical baselines share identical OT histories.
    drop = {'date','Date','start_date','end_date','fact','pred','preds','Final_Output','Final_Search_8','Final_Search_16','prior_history_avg','prior_history_std'}
    columns = [c for c in frame if c not in drop and pd.api.types.is_numeric_dtype(frame[c])]
    if any(c not in frame.columns for c in columns):
        raise ValueError('Missing measurement columns for ' + domain)
    raw = frame[columns].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float64)
    mask = np.isfinite(raw)
    raw[~mask] = np.nan
    mean = np.nanmean(raw[:train_end], axis=0)
    scale = np.nanstd(raw[:train_end], axis=0)
    if not np.isfinite(mean).all():
        raise ValueError('A measurement has no training observations')
    scale[scale < 1e-8] = 1.
    # Only past values fill model INPUTS; never use these filled values as targets.
    filled = pd.DataFrame(raw).ffill().fillna(dict(enumerate(mean))).to_numpy()
    values = ((filled - mean) / scale).astype(np.float32)
    available = (pd.to_datetime(frame['end_date'], errors='raise') if 'end_date' in frame else dates)
    if not available.is_monotonic_increasing:
        raise ValueError('Numerical availability dates are not chronological')
    marks = np.stack([dates.dt.month.to_numpy() / 12. - .5,
                      dates.dt.day.to_numpy() / 31. - .5,
                      dates.dt.dayofweek.to_numpy() / 6. - .5,
                      dates.dt.hour.to_numpy() / 23. - .5], axis=-1).astype(np.float32)
    # Fact is used; generated preds and numeric prior_history_* are deliberately not model inputs.
    texts = frame['fact'].fillna('').astype(str).tolist() if 'fact' in frame else [''] * n
    from .text_clean import clean_fact
    texts = [clean_fact(s) for s in texts]
    data = SeriesData(domain, raw, values, mask, mean, scale,
                      dates.to_numpy(), available.to_numpy(), marks, texts, columns,
                      train_end, val_end, str(path))
    # Frozen text-version identity; the frozen encoder descriptor is keyed by both.
    from .config import TEXT_VERSION, source_decisions_sha256
    data.text_version = TEXT_VERSION
    data.source_decisions_sha256 = source_decisions_sha256()
    return data

def load_series(domain, data_root=DATA_ROOT, split=(.6, .2, .2), cache_root=None):
    from .data_cache import load_prepared_series
    return load_prepared_series(domain, data_root, split, cache_root)


class ForecastWindows(Dataset):
    def __init__(self, data, split, lookback, horizon, limit=None, method=None):
        self.data, self.lookback, self.horizon = data, lookback, horizon
        self.method = method
        from .data_cache import window_indices
        self.indices = window_indices(data, split, lookback, horizon)
        if limit is not None:
            self.indices = self.indices[:limit]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        i = int(self.indices[item])
        sl = slice(i + 1, i + 1 + self.horizon)
        hist = slice(i + 1 - self.lookback, i + 1)
        result = {'x': torch.from_numpy(self.data.values[hist]),
                  'y': torch.from_numpy(self.data.label_values[sl]),
                  'mask': torch.from_numpy(self.data.mask[sl]), 'origin': i}
        if self.method != 'GPT4MTS':
            result['mark'] = torch.from_numpy(self.data.marks[hist])
        if self.method is None:
            result['history_y'] = torch.from_numpy(self.data.history_values[hist])
            result['history_mask'] = torch.from_numpy(self.data.mask[hist])
        return result

def masked_errors(pred, target, mask):
    pred, target, mask = np.asarray(pred), np.asarray(target), np.asarray(mask, dtype=bool)
    if pred.shape != target.shape or mask.shape != target.shape:
        raise ValueError('Prediction/target/mask shapes differ')
    if not mask.any():
        raise ValueError('No observed target elements')
    if not np.isfinite(pred[mask]).all() or not np.isfinite(target[mask]).all():
        raise ValueError('Non-finite prediction or observed target')
    error = pred[mask] - target[mask]
    return {'mse': float(np.mean(error ** 2)), 'mae': float(np.mean(np.abs(error))),
            'valid_target_elements': int(mask.sum())}
