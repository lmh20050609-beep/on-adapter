"""H9 screen, seed-2 VALIDATION only: output-space residual shrinkage  F = bb + lam * (T - bb)
(= scaling the adapter residual by lam, since the head is linear), alone and combined with the round-5 abstaining gate."""
import math, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / 'scripts'))
import seed2_round1_rc as r1, seed2_round3_train as r3, seed2_round5_ta as r5
res = {}
for d, h in r1.tasks():
    t, _ = r3.load('a0e15', 'val', d, h)
    bb, T, y, m, o = t['backbone'].astype(float), t['predictions'].astype(float), t['targets'].astype(float), t['mask'], t['origins']
    eb = r1.per_origin_mae(bb, y, m); ix = np.array_split(np.arange(len(o)), 3)
    for lam in (.25, .5, .75, 1.):
        F = bb + lam * (T - bb)
        for name, p in (('shrink', F), ('shrink+gate', r5.abstain(bb, F, y, m, o, h, .9, 0.)[0])):
            ep = r1.per_origin_mae(p, y, m)
            res.setdefault('%s lam%.2f' % (name, lam), []).append([r1.cvar90(ep) / r1.cvar90(eb)] + [r1.cvar90(ep[i]) / r1.cvar90(eb[i]) for i in ix])
gm = lambda x: math.exp(np.mean(np.log(x)))
print('%-20s %7s %7s %7s %7s %7s' % ('candidate', 'whole', 't1', 't2', 't3', 'score'))
for n, v in res.items():
    v = np.array(v); c = [gm(v[:, i]) for i in range(4)]
    print('%-20s %7.4f %7.4f %7.4f %7.4f %7.4f' % (n, *c, max(c[0], max(c[1:]))))
