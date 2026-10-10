"""Dev screen H1 (validation streams of seeds 2025-2027 only): online per-variable/per-step residual-bias
correction from fully matured labels, applied on top of a base forecast. No test data is read."""
import json, math, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
VAL = ROOT / 'reports/online_lr_val9_20261008/cells'
DOM = {'Environment': [48, 96, 192, 336], 'Energy': [12, 24, 36, 48], 'Health_US': [12, 24, 36, 48]}
DOMAINS = ['Environment', 'Energy', 'Health_US', 'Traffic', 'SocialGood', 'Security', 'Economy', 'Climate', 'Agriculture']

def cvar90(pred, y, m):
    e = np.abs(pred - y); v = np.array([x[k].mean() for x, k in zip(e, m)])
    v = np.sort(v)[::-1]; mass = .1 * len(v); w = int(mass)
    return (v[:w].sum() + (mass - w) * (v[w] if w < len(v) else 0)) / mass

def mse(pred, y, m): return float(((pred - y) ** 2)[m].mean())

def bias_correct(base, y, m, origins, H, gamma, shrink):
    out = base.copy(); s = np.zeros(base.shape[1:]); w = np.zeros(base.shape[1:]); j = 0
    for i in range(len(origins)):
        while j < i and origins[j] + H <= origins[i]:
            r = np.where(m[j], y[j] - base[j], 0.); s = gamma * s + r; w = gamma * w + m[j]; j += 1
        ok = w > 0
        out[i][ok] = base[i][ok] + shrink * s[ok] / w[ok]
    return out

variants = [(g, a) for g in (0.9, 0.97, 1.0) for a in (0.5, 1.0)]
res = {}
for d in DOMAINS:
    for h in DOM.get(d, [6, 8, 10, 12]):
        acc = {}
        for s in (2025, 2026, 2027):
            for src in ('lr0',):
                with np.load(VAL / ('%s_H%d_seed%d' % (d, h, s)) / ('%s_r0.npz' % src)) as z:
                    base, on, y, m, o = z['base'].astype(float), z['pred'].astype(float), z['y'].astype(float), z['mask'].astype(bool), z['origin']
            b = cvar90(base, y, m); bm = mse(base, y, m)
            acc.setdefault('on_lr0', []).append((cvar90(on, y, m) / b, mse(on, y, m) / bm))
            for g, a in variants:
                p = bias_correct(base, y, m, o, h, g, a)
                acc.setdefault('bb+bias g%.2f a%.1f' % (g, a), []).append((cvar90(p, y, m) / b, mse(p, y, m) / bm))
        res['%s H%d' % (d, h)] = {k: (float(np.mean([x[0] for x in v])), float(np.mean([x[1] for x in v]))) for k, v in acc.items()}
names = list(next(iter(res.values())))
gm = lambda xs: math.exp(np.mean(np.log(xs)))
print('VALIDATION (seeds 2025-27), geometric mean over 36 tasks of seed-mean ratio to backbone')
for n in names:
    print('%-22s CVaR90 %.4f  MSE %.4f  | CVaR90 by domain: %s' % (n, gm([res[t][n][0] for t in res]), gm([res[t][n][1] for t in res]),
          ' '.join('%s:%.3f' % (d[:4], gm([res[t][n][0] for t in res if t.startswith(d)])) for d in DOMAINS)))
json.dump(res, open(Path(__file__).with_suffix('.json'), 'w'), indent=1)
