"""H5 screen, seed-2 VALIDATION only: in-stream expert gating with fast discount.
Experts: lr0, lr3e-4 (original online updates), lr0+phase(.5), lr0+step(.5). Gate picks the expert with the lowest
discounted mean per-origin MAE on fully matured origins (warm: lr0). Reports whole-stream and thirds (GM over 36 tasks, mean of 3 streams)."""
import json, math, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / 'scripts'))
import seed2_round1_rc as r1, seed2_round2_pc as r2

def pomae(p, y, m): return r1.per_origin_mae(p, y, m)
def gate(ex, y, m, o, H, gs):
    L = np.stack([pomae(e, y, m) for e in ex]); s = np.zeros(len(ex)); w = 0.; j = 0; out = np.empty_like(ex[0]); ch = []
    for i in range(len(o)):
        while j < i and o[j] + H <= o[i]:
            s = gs * s + L[:, j]; w = gs * w + 1; j += 1
        k = 0 if w == 0 else int(np.flatnonzero(s == s.min())[0]); out[i] = ex[k][i]; ch.append(k)
    return out, np.array(ch)

configs = {'lr0': None, 'lr3e-4': None, 'phase.5': None,
           'G{lr0,3e-4} gs.9': ([0, 1], .9), 'G{lr0,3e-4} gs.8': ([0, 1], .8),
           'G{lr0,3e-4,ph,st} gs.9': ([0, 1, 2, 3], .9), 'G{lr0,3e-4,ph,st} gs.8': ([0, 1, 2, 3], .8),
           'G{lr0,ph,st} gs.9': ([0, 2, 3], .9), 'G{lr0,ph,st} gs.8': ([0, 2, 3], .8)}
res = {}
for d, h in r1.tasks():
    acc = {}
    for r in r1.STREAMS:
        a, _ = r1.load('val', d, h, 'lr0', r); b, _ = r1.load('val', d, h, 'lr3e-4', r)
        y, m, o, bb = a['targets'].astype(float), a['mask'], a['origins'], a['backbone'].astype(float)
        ex = [a['predictions'].astype(float), b['predictions'].astype(float),
              r2.calibrate(a['predictions'], y, m, o, h, 'phase', .5, 1., r2.period(d)),
              r2.calibrate(a['predictions'], y, m, o, h, 'step', .5, 1., r2.period(d))]
        eb = pomae(bb, y, m); ix = np.array_split(np.arange(len(o)), 3)
        for n, c in configs.items():
            p = {'lr0': ex[0], 'lr3e-4': ex[1], 'phase.5': ex[2]}.get(n)
            if p is None:
                p = gate([ex[k] for k in c[0]], y, m, o, h, c[1])[0]
            ep = pomae(p, y, m)
            acc.setdefault(n, []).append([r1.cvar90(ep) / r1.cvar90(eb)] + [r1.cvar90(ep[i]) / r1.cvar90(eb[i]) for i in ix])
    res['%s H%d' % (d, h)] = {n: np.mean(v, 0).tolist() for n, v in acc.items()}
gm = lambda x: math.exp(np.mean(np.log(x)))
print('%-24s %7s %7s %7s %7s %7s' % ('candidate', 'whole', 't1', 't2', 't3', 'worst'))
for n in configs:
    c = [gm([res[t][n][k] for t in res]) for k in range(4)]
    print('%-24s %7.4f %7.4f %7.4f %7.4f %7.4f' % (n, *c, max(c[1:])))
json.dump(res, open(Path(__file__).with_suffix('.json'), 'w'), indent=1)
