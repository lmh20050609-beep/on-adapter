"""Dev screen H1c (validation streams, seeds 2025-2027 only)."""
import json, math
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
VAL = ROOT / 'reports/online_lr_val9_20261008/cells'
DOM = {'Environment': [48, 96, 192, 336], 'Energy': [12, 24, 36, 48], 'Health_US': [12, 24, 36, 48]}
DOMAINS = ['Environment', 'Energy', 'Health_US', 'Traffic', 'SocialGood', 'Security', 'Economy', 'Climate', 'Agriculture']
SHRINKS = np.array([0., .25, .5, .75, 1.])

def per_origin_mae(pred, y, m): return np.array([x[k].mean() for x, k in zip(np.abs(pred - y), m)])
def cvar_of(v):
    v = np.sort(v)[::-1]; mass = .1 * len(v); w = int(mass); return (v[:w].sum() + (mass - w) * (v[w] if w < len(v) else 0)) / mass

def bias_path(base, y, m, origins, H, gamma, pooled):
    """bias available at each origin from fully matured residuals (no future labels)."""
    n = len(origins); B = np.zeros_like(base); s = np.zeros(base.shape[1:]); w = np.zeros(base.shape[1:]); j = 0
    for i in range(n):
        while j < i and origins[j] + H <= origins[i]:
            s = gamma * s + np.where(m[j], y[j] - base[j], 0.); w = gamma * w + m[j]; j += 1
        if pooled:
            sv, wv = s.sum(0), w.sum(0); b = np.where(wv > 0, sv / np.maximum(wv, 1e-12), 0.)[None, :].repeat(base.shape[1], 0)
        else:
            b = np.where(w > 0, s / np.maximum(w, 1e-12), 0.)
        B[i] = b
    return B

def select_online(base, B, y, m, origins, H, gamma_sel, loss):
    """at each origin choose shrink from SHRINKS by discounted matured loss of each candidate (ties -> smaller shrink)."""
    cand = base[None] + SHRINKS[:, None, None, None] * B[None]
    if loss == 'mae': L = np.stack([per_origin_mae(c, y, m) for c in cand])
    else: L = np.stack([np.array([((x - t) ** 2)[k].mean() for x, t, k in zip(c, y, m)]) for c in cand])
    out = np.empty_like(base); s = np.zeros(len(SHRINKS)); j = 0; chosen = []
    for i in range(len(origins)):
        while j < i and origins[j] + H <= origins[i]:
            s = gamma_sel * s + L[:, j]; j += 1
        k = 0 if not s.any() else int(np.flatnonzero(s == s.min())[0]); chosen.append(SHRINKS[k]); out[i] = cand[k, i]
    return out, float(np.mean(chosen))

res = {}; share = {}
for d in DOMAINS:
    for h in DOM.get(d, [6, 8, 10, 12]):
        acc = {}
        for s in (2025, 2026, 2027):
            with np.load(VAL / ('%s_H%d_seed%d' % (d, h, s)) / 'lr0_r0.npz') as z:
                bb, on, y, m, o = z['base'].astype(float), z['pred'].astype(float), z['y'].astype(float), z['mask'].astype(bool), z['origin']
            ref = cvar_of(per_origin_mae(bb, y, m)); r = lambda p: cvar_of(per_origin_mae(p, y, m)) / ref
            acc.setdefault('on_lr0', []).append(r(on))
            for pooled in (False, True):
                for g in (.97, 1.):
                    B = bias_path(on, y, m, o, h, g, pooled)
                    acc.setdefault('fix.5 pool%d g%.2f' % (pooled, g), []).append(r(on + .5 * B))
                    for gs in (.95, 1.):
                        for loss in ('mae', 'mse'):
                            p, mean_k = select_online(on, B, y, m, o, h, gs, loss)
                            acc.setdefault('sel pool%d g%.2f gs%.2f %s' % (pooled, g, gs, loss), []).append(r(p))
        res['%s H%d' % (d, h)] = {k: float(np.mean(v)) for k, v in acc.items()}
gm = lambda xs: math.exp(np.mean(np.log(xs)))
for n in res['Energy H12']:
    vals = [res[t][n] for t in res]
    print('%-28s %.4f worst %.3f | %s' % (n, gm(vals), max(vals), ' '.join('%s:%.3f' % (d[:4], gm([res[t][n] for t in res if t.startswith(d)])) for d in DOMAINS)))
json.dump(res, open(Path(__file__).with_suffix('.json'), 'w'), indent=1)
