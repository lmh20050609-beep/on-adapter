"""Dev screen H1b (validation streams, seeds 2025-2027 only). Online residual calibration on top of a base forecast:
correction = shrink * m, m = discounted mean of fully matured residuals per (step, variable);
shrink = fixed a, or positive-part James-Stein max(0, 1 - c / t^2), t^2 = m^2 / (var / n_eff)."""
import json, math
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
VAL = ROOT / 'reports/online_lr_val9_20261008/cells'
DOM = {'Environment': [48, 96, 192, 336], 'Energy': [12, 24, 36, 48], 'Health_US': [12, 24, 36, 48]}
DOMAINS = ['Environment', 'Energy', 'Health_US', 'Traffic', 'SocialGood', 'Security', 'Economy', 'Climate', 'Agriculture']

def cvar90(pred, y, m):
    v = np.array([x[k].mean() for x, k in zip(np.abs(pred - y), m)]); v = np.sort(v)[::-1]
    mass = .1 * len(v); w = int(mass); return (v[:w].sum() + (mass - w) * (v[w] if w < len(v) else 0)) / mass

def calibrate(base, y, m, origins, H, gamma, rule, c):
    out = base.copy(); s1 = np.zeros(base.shape[1:]); s2 = np.zeros_like(s1); w = np.zeros_like(s1); w2 = np.zeros_like(s1); j = 0
    for i in range(len(origins)):
        while j < i and origins[j] + H <= origins[i]:
            r = np.where(m[j], y[j] - base[j], 0.); k = m[j].astype(float)
            s1 = gamma * s1 + r; s2 = gamma * s2 + r * r; w = gamma * w + k; w2 = gamma * gamma * w2 + k; j += 1
        ok = w > 1
        mean = np.where(ok, s1 / np.maximum(w, 1e-12), 0.)
        var = np.where(ok, np.maximum(s2 / np.maximum(w, 1e-12) - mean ** 2, 1e-12), np.inf)
        neff = np.where(ok, w ** 2 / np.maximum(w2, 1e-12), 0.)
        if rule == 'fixed':
            shrink = np.where(ok, c, 0.)
        else:
            t2 = mean ** 2 / (var / np.maximum(neff, 1e-12))
            shrink = np.where(ok, np.clip(1 - c / np.maximum(t2, 1e-12), 0, 1), 0.)
        out[i] = base[i] + shrink * mean
    return out

variants = [('fixed', g, .5) for g in (.97, 1.)] + [('js', g, c) for g in (.97, 1.) for c in (1., 2., 4.)]
res = {}
for d in DOMAINS:
    for h in DOM.get(d, [6, 8, 10, 12]):
        acc = {}
        for s in (2025, 2026, 2027):
            with np.load(VAL / ('%s_H%d_seed%d' % (d, h, s)) / 'lr0_r0.npz') as z:
                bb, on, y, m, o = z['base'].astype(float), z['pred'].astype(float), z['y'].astype(float), z['mask'].astype(bool), z['origin']
            ref = cvar90(bb, y, m)
            acc.setdefault('on_lr0', []).append(cvar90(on, y, m) / ref)
            for base_name, base in (('bb', bb), ('on', on)):
                for rule, g, c in variants:
                    acc.setdefault('%s+%s g%.2f c%.1f' % (base_name, rule, g, c), []).append(cvar90(calibrate(base, y, m, o, h, g, rule, c), y, m) / ref)
        res['%s H%d' % (d, h)] = {k: float(np.mean(v)) for k, v in acc.items()}
gm = lambda xs: math.exp(np.mean(np.log(xs)))
print('VALIDATION CVaR90 ratio to backbone (seed-mean, GM over 36 tasks); worst task; per-domain GM')
for n in res['Energy H12']:
    vals = [res[t][n] for t in res]
    print('%-22s %.4f worst %.3f | %s' % (n, gm(vals), max(vals), ' '.join('%s:%.3f' % (d[:4], gm([res[t][n] for t in res if t.startswith(d)])) for d in DOMAINS)))
json.dump(res, open(Path(__file__).with_suffix('.json'), 'w'), indent=1)
