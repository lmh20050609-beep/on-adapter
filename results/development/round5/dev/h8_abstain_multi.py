"""H8 screen, seed-2 VALIDATION only: backbone-first abstaining gate over several On experts.
Experts: T = round-3 a0e15 (lr0, r0); O = original-training On with its original online rule lr3e-4 (round-1 val r0).
At forecast t: among experts whose discounted matured MAE beats the backbone's by margin mu (mu=0 for the currently
active expert, hysteresis), emit the lowest; otherwise emit the backbone. Requires >= kmin matured origins."""
import json, math, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / 'scripts'))
import seed2_round1_rc as r1, seed2_round3_train as r3

def abstain_multi(bb, ex, y, m, o, H, gs, mu, kmin):
    lb = r1.per_origin_mae(bb, y, m); L = np.stack([r1.per_origin_mae(e, y, m) for e in ex])
    sb = 0.; s = np.zeros(len(ex)); n = 0; j = 0; act = -1; out = bb.copy(); ch = np.full(len(o), -1)
    for i in range(len(o)):
        while j < i and o[j] + H <= o[i]:
            sb = gs * sb + lb[j]; s = gs * s + L[:, j]; n += 1; j += 1
        if n >= kmin:
            thr = np.where(np.arange(len(ex)) == act, sb, (1 - mu) * sb)
            ok = np.flatnonzero(s < thr)
            act = int(ok[np.argmin(s[ok])]) if len(ok) else -1
        if act >= 0: out[i] = ex[act][i]; ch[i] = act
    return out, ch

res = {}
for d, h in r1.tasks():
    t, _ = r3.load('a0e15', 'val', d, h); oo, _ = r1.load('val', d, h, 'lr3e-4', 0)
    assert np.array_equal(t['backbone'], oo['backbone'])
    bb, y, m, o = t['backbone'].astype(float), t['targets'].astype(float), t['mask'], t['origins']
    T, O = t['predictions'].astype(float), oo['predictions'].astype(float)
    eb = r1.per_origin_mae(bb, y, m); ix = np.array_split(np.arange(len(o)), 3)
    for name, ex in (('{T}', [T]), ('{O}', [O]), ('{T,O}', [T, O])):
        for gs in (.8, .9, 1.):
            for mu in (0., .01, .03):
                p, ch = abstain_multi(bb, ex, y, m, o, h, gs, mu, 4); ep = r1.per_origin_mae(p, y, m)
                res.setdefault('%s gs%.1f mu%.2f' % (name, gs, mu), []).append(
                    [r1.cvar90(ep) / r1.cvar90(eb)] + [r1.cvar90(ep[i]) / r1.cvar90(eb[i]) for i in ix] + [(ch >= 0).mean()])
gm = lambda x: math.exp(np.mean(np.log(x)))
print('%-22s %7s %7s %7s %7s %7s %6s' % ('candidate', 'whole', 't1', 't2', 't3', 'score', 'used%'))
for n, v in res.items():
    v = np.array(v); c = [gm(v[:, i]) for i in range(4)]
    print('%-22s %7.4f %7.4f %7.4f %7.4f %7.4f %6.1f' % (n, *c, max(c[0], max(c[1:])), 100 * v[:, 4].mean()))
