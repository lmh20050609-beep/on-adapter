"""H7 screen, seed-2 VALIDATION only: backbone-first abstaining gate. Emit the backbone unless the adapter's
gs-discounted mean per-origin MAE over >= k fully matured origins is lower than the backbone's by a relative margin mu;
hysteresis: once on the adapter, return to the backbone only when the adapter is no longer better (mu=0)."""
import json, math, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / 'scripts'))
import seed2_round1_rc as r1, seed2_round3_train as r3

def abstain(bb, ad, y, m, o, H, gs, mu, kmin):
    lb, la = r1.per_origin_mae(bb, y, m), r1.per_origin_mae(ad, y, m)
    sb = sa = w = 0.; n = 0; j = 0; on = False; out = bb.copy(); ch = np.zeros(len(o), int)
    for i in range(len(o)):
        while j < i and o[j] + H <= o[i]:
            sb = gs * sb + lb[j]; sa = gs * sa + la[j]; w = gs * w + 1; n += 1; j += 1
        if n >= kmin:
            on = (sa < (1 - mu) * sb) if not on else (sa < sb)
        if on: out[i] = ad[i]; ch[i] = 1
    return out, ch

res = {}
for src in ('orig', 'a0e15'):
    for d, h in r1.tasks():
        t, _ = r3.load(src, 'val', d, h)
        bb, ad, y, m, o = t['backbone'].astype(float), t['predictions'].astype(float), t['targets'].astype(float), t['mask'], t['origins']
        eb = r1.per_origin_mae(bb, y, m); ix = np.array_split(np.arange(len(o)), 3)
        for gs in (.9, 1.):
            for mu in (0., .01, .03):
                for kmin in (4, 12):
                    p, ch = abstain(bb, ad, y, m, o, h, gs, mu, kmin); ep = r1.per_origin_mae(p, y, m)
                    res.setdefault('%s gs%.1f mu%.2f k%d' % (src, gs, mu, kmin), []).append(
                        [r1.cvar90(ep) / r1.cvar90(eb)] + [r1.cvar90(ep[i]) / r1.cvar90(eb[i]) for i in ix] + [ch.mean()])
gm = lambda x: math.exp(np.mean(np.log(x)))
print('%-26s %7s %7s %7s %7s %7s %6s' % ('candidate', 'whole', 't1', 't2', 't3', 'score', 'adapt%'))
for n, v in res.items():
    v = np.array(v); c = [gm(v[:, i]) for i in range(4)]
    print('%-26s %7.4f %7.4f %7.4f %7.4f %7.4f %6.1f' % (n, *c, max(c[0], max(c[1:])), 100 * v[:, 4].mean()))
