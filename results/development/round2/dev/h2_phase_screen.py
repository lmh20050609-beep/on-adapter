"""H2 screen on VALIDATION streams only (seed 2 round-1 val trajectories + seeds 2025-2027 val9 trajectories).
Calibration b_t[step,var] = discounted mean of fully matured residuals whose TARGET phase equals the phase of
target (t+step); phase = target_row mod P, P = 12 monthly, 52 weekly, 7 daily. Compared with step-indexed RC."""
import json, math, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[4]
S2 = Path('E:/FedNoise/iadapter-baselines/runs/seed2_round1_rc_20261009/val')
DEV = ROOT / 'reports/online_lr_val9_20261008/cells'
HOR = {'Environment': [48, 96, 192, 336], 'Energy': [12, 24, 36, 48], 'Health_US': [12, 24, 36, 48]}
PER = {'Environment': 7, 'Energy': 52, 'Health_US': 52}
DOMAINS = ['Environment', 'Energy', 'Health_US', 'Traffic', 'SocialGood', 'Security', 'Economy', 'Climate', 'Agriculture']

def pomae(p, y, m): return np.array([x[k].mean() for x, k in zip(np.abs(p - y), m)])
def cvar(v): v = np.sort(v)[::-1]; mass = .1 * len(v); w = int(mass); return (v[:w].sum() + (mass - w) * (v[w] if w < len(v) else 0)) / mass

def calib(base, y, m, o, H, a, gamma, P):
    """P=None: step-indexed (round 1). Otherwise phase-indexed by target row."""
    n, Hs, N = base.shape; out = base.copy(); j = 0
    K = Hs if P is None else P
    s = np.zeros((K, N)); w = np.zeros((K, N))
    steps = np.arange(1, Hs + 1)
    for i in range(n):
        while j < i and o[j] + H <= o[i]:
            r = np.where(m[j], y[j] - base[j], 0.)
            idx = np.arange(Hs) if P is None else (o[j] + steps) % P
            s *= gamma; w *= gamma
            np.add.at(s, idx, r); np.add.at(w, idx, m[j].astype(float)); j += 1
        idx = np.arange(Hs) if P is None else (o[i] + steps) % P
        b = np.where(w[idx] > 0, s[idx] / np.maximum(w[idx], 1e-300), 0.)
        out[i] = base[i] + a * b
    return out

def cells():
    for d in DOMAINS:
        for h in HOR.get(d, [6, 8, 10, 12]):
            yield d, h

def load_s2(d, h, r):
    with np.load(S2 / d / ('H%d' % h) / ('lr0_r%d.npz' % r)) as z:
        return z['backbone'].astype(float), z['predictions'].astype(float), z['targets'].astype(float), z['mask'].astype(bool), z['origins']
def load_dev(d, h, s):
    with np.load(DEV / ('%s_H%d_seed%d' % (d, h, s)) / 'lr0_r0.npz') as z:
        return z['base'].astype(float), z['pred'].astype(float), z['y'].astype(float), z['mask'].astype(bool), z['origin']

variants = [('step', None, .75, 1.)] + [('phase', 'P', a, g) for a in (.5, .75, 1.) for g in (1., .97)]
out = {}
for src, loader, keys in (('seed2_val', load_s2, [0]), ('dev_val', load_dev, [2025, 2026, 2027])):
    res = {}
    for d, h in cells():
        acc = {}
        for k in keys:
            bb, on, y, m, o = loader(d, h, k); ref = cvar(pomae(bb, y, m))
            acc.setdefault('lr0', []).append(cvar(pomae(on, y, m)) / ref)
            for name, P, a, g in variants:
                p = calib(on, y, m, o, h, a, g, None if P is None else PER.get(d, 12))
                acc.setdefault('%s a%.2f g%.2f' % (name, a, g), []).append(cvar(pomae(p, y, m)) / ref)
        res['%s H%d' % (d, h)] = {kk: float(np.mean(v)) for kk, v in acc.items()}
    out[src] = res
    gm = lambda x: math.exp(np.mean(np.log(x)))
    print('==', src)
    for n in res['Energy H12']:
        print('%-18s %.4f | %s' % (n, gm([res[t][n] for t in res]), ' '.join('%s:%.3f' % (d[:4], gm([res[t][n] for t in res if t.startswith(d)])) for d in DOMAINS)))
json.dump(out, open(Path(__file__).with_suffix('.json'), 'w'), indent=1)
