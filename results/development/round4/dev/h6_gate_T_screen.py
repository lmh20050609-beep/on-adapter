"""H6 screen, seed-2 VALIDATION only (round-3 a0e15 lr0 r0 trajectories): in-stream gating over experts drawn from
{T (a0e15 adapter), backbone, T+phase(.5), T+step(.5)} by discounted matured per-origin MAE; warm start T."""
import json, math, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / 'scripts'))
import seed2_round1_rc as r1, seed2_round2_pc as r2, seed2_round3_train as r3

def gate(ex, y, m, o, H, gs):
    L = np.stack([r1.per_origin_mae(e, y, m) for e in ex]); s = np.zeros(len(ex)); w = 0.; j = 0; out = np.empty_like(ex[0])
    for i in range(len(o)):
        while j < i and o[j] + H <= o[i]:
            s = gs * s + L[:, j]; w = gs * w + 1; j += 1
        k = 0 if w == 0 else int(np.flatnonzero(s == s.min())[0]); out[i] = ex[k][i]
    return out

sets = {'T': [0], 'G{T,bb}': [0, 1], 'G{T,bb,ph,st}': [0, 1, 2, 3], 'G{T,ph,st}': [0, 2, 3]}
res = {}
for d, h in r1.tasks():
    t, _ = r3.load('a0e15', 'val', d, h)
    T, bb, y, m, o = t['predictions'].astype(float), t['backbone'].astype(float), t['targets'].astype(float), t['mask'], t['origins']
    ex = [T, bb, r2.calibrate(T, y, m, o, h, 'phase', .5, 1., r2.period(d)), r2.calibrate(T, y, m, o, h, 'step', .5, 1., r2.period(d))]
    eb = r1.per_origin_mae(bb, y, m); ix = np.array_split(np.arange(len(o)), 3); acc = {}
    for n, idx in sets.items():
        for gs in ([1.] if len(idx) == 1 else [.7, .8, .9, 1.]):
            p = ex[0] if len(idx) == 1 else gate([ex[k] for k in idx], y, m, o, h, gs)
            ep = r1.per_origin_mae(p, y, m)
            acc['%s gs%.1f' % (n, gs)] = [r1.cvar90(ep) / r1.cvar90(eb)] + [r1.cvar90(ep[i]) / r1.cvar90(eb[i]) for i in ix]
    res['%s H%d' % (d, h)] = acc
gm = lambda x: math.exp(np.mean(np.log(x)))
print('%-22s %7s %7s %7s %7s %7s' % ('candidate', 'whole', 't1', 't2', 't3', 'worst'))
for n in res['Energy H12']:
    c = [gm([res[k][n][i] for k in res]) for i in range(4)]
    print('%-22s %7.4f %7.4f %7.4f %7.4f %7.4f' % (n, *c, max(c[1:])))
json.dump(res, open(Path(__file__).with_suffix('.json'), 'w'), indent=1)
