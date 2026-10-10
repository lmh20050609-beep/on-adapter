"""H3 screen on seed-2 VALIDATION streams only: fixed calibrations vs an in-stream self-validating gate.
Gate: experts E = {lr0, lr0+step(a), lr0+phase(a)}; at forecast t emit the expert with the lowest gamma_s-discounted
mean per-origin MAE over fully matured origins (o + H <= t); before any maturity emit lr0.
Reported per candidate: whole-stream GM ratio and the worst of the three chronological thirds (each third's CVaR90
ratio computed on that third's origins; calibration state runs through the whole stream)."""
import json, math, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / 'scripts'))
import seed2_round1_rc as r1, seed2_round2_pc as r2

def pomae(p, y, m): return np.array([x[k].mean() for x, k in zip(np.abs(p - y), m.astype(bool))])
def cvar(v): return r1.cvar90(v)

def gate(experts, y, m, o, H, gs):
    L = np.stack([pomae(e, y, m) for e in experts]); s = np.zeros(len(experts)); w = 0.; j = 0
    out = np.empty_like(experts[0]); ch = np.zeros(len(o), int)
    for i in range(len(o)):
        while j < i and o[j] + H <= o[i]:
            s = gs * s + L[:, j]; w = gs * w + 1; j += 1
        k = 0 if w == 0 else int(np.flatnonzero(s == s.min())[0]); ch[i] = k; out[i] = experts[k][i]
    return out, ch

cands = {'lr0': None, 'step.75': ('step', .75), 'phase.75': ('phase', .75), 'step.5': ('step', .5), 'phase.5': ('phase', .5)}
gates = {'gate(.75) gs1': (.75, 1.), 'gate(.75) gs.95': (.75, .95), 'gate(.5) gs1': (.5, 1.), 'gate(.5) gs.95': (.5, .95)}
res = {}
for d, h in r1.tasks():
    acc = {}
    for r in r1.STREAMS:
        t, _ = r1.load('val', d, h, 'lr0', r)
        on, bb, y, m, o = t['predictions'].astype(float), t['backbone'].astype(float), t['targets'].astype(float), t['mask'], t['origins']
        thirds = np.array_split(np.arange(len(o)), 3)
        preds = {}
        for n, c in cands.items():
            preds[n] = on if c is None else r2.calibrate(on, y, m, o, h, c[0], c[1], 1., r2.period(d))
        for n, (a, gs) in gates.items():
            ex = [on, r2.calibrate(on, y, m, o, h, 'step', a, 1., r2.period(d)), r2.calibrate(on, y, m, o, h, 'phase', a, 1., r2.period(d))]
            preds[n] = gate(ex, y, m, o, h, gs)[0]
        eb = pomae(bb, y, m)
        for n, p in preds.items():
            ep = pomae(p, y, m)
            acc.setdefault(n, []).append([cvar(ep) / cvar(eb)] + [cvar(ep[ix]) / cvar(eb[ix]) for ix in thirds])
    res['%s H%d' % (d, h)] = {n: np.mean(v, 0).tolist() for n, v in acc.items()}
gm = lambda x: math.exp(np.mean(np.log(x)))
print('%-16s %7s %7s %7s %7s %8s' % ('candidate', 'whole', 'third1', 'third2', 'third3', 'worst3'))
for n in res['Energy H12']:
    cols = [gm([res[t][n][k] for t in res]) for k in range(4)]
    print('%-16s %7.4f %7.4f %7.4f %7.4f %8.4f' % (n, *cols, max(cols[1:])))
json.dump(res, open(Path(__file__).with_suffix('.json'), 'w'), indent=1)
