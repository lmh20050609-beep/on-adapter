"""Metric scan for the tail-risk table: every metric below is computed for all ten main-table methods on the same cells
(seeds 2025-2027). Writes results/paper/metric_scan.json."""
import json, sys
from pathlib import Path
ROOT_ = Path(__file__).resolve().parents[1]
for _p in (Path(__file__).resolve().parent, ROOT_ / 'final', ROOT_ / 'new_baselines', ROOT_ / 'confirmation'): sys.path.insert(0, str(_p))
import numpy as np
from scipy.stats import binomtest
import scoring as F
import solid as SO
import adcsd_ogd as AO
import cosa as CO
r1, fz, f5 = F.r1, F.fz, F.f5; SEEDS = [2025, 2026, 2027]; OURS = F.OURS; R = SO.R; BB = 'iTransformer'
lr = json.loads((R / 'solid_selection.json').read_text())['selected_lr']; rows = {}
for d, h in r1.tasks():
    for s in SEEDS:
        for m in F.METHODS: rows[(d, h, s, m)], _ = F.baseline(m, d, h, s)
        cv = f5.traj(F.tfolder('cv', d, h, s)); rows[(d, h, s, OURS)] = fz.cell_metrics(f5.gated(cv, h)[0], cv['targets'], cv['mask'])
        a, _ = SO.load('test', lr, d, h, s); rows[(d, h, s, 'SOLID')] = fz.cell_metrics(a['predictions'], a['targets'], a['mask'])
        a, _ = AO.load('ADCSD', d, h, s); rows[(d, h, s, 'ADCSD')] = fz.cell_metrics(a['predictions'], a['targets'], a['mask'])
        a, _ = CO.load(d, h, s); rows[(d, h, s, 'COSA')] = fz.cell_metrics(a['predictions'], a['targets'], a['mask'])
cmp = [OURS] + [m for m in F.METHODS if m != 'On-Adapter'] + ['SOLID', 'ADCSD', 'COSA']
T = list(r1.tasks()); cells = [(d, h, s) for d, h in T for s in SEEDS]
out = {}
for m in cmp:
    v = {}
    for k in ('cvar80', 'cvar90', 'cvar95', 'mae', 'mse'):
        task = np.array([np.mean([rows[(d, h, s, m)][k] / rows[(d, h, s, BB)][k] for s in SEEDS]) for d, h in T])
        v['S_' + k] = float(np.exp(np.log(task).mean()))
        rk = [1 + sorted(cmp, key=lambda x: rows[c + (x,)][k]).index(m) for c in cells]
        v['rank_' + k] = float(np.mean(rk))
        if k == 'cvar90':
            v.update({'top1': sum(r == 1 for r in rk), 'top3': sum(r <= 3 for r in rk), 'top5': sum(r <= 5 for r in rk), 'last': sum(r == len(cmp) for r in rk)})
            for t in (0.0, 0.01, 0.02, 0.05):
                v['gain>%g' % t] = int((task < 1 - t).sum()); v['harm>%g' % t] = int((task > 1 + t).sum())
            v['best_task'] = float(task.min()); v['worst_task'] = float(task.max())
            v['mean_abs_change_pct'] = float(np.mean(np.abs(task - 1)) * 100)
            v['mean_gain_when_gain_pct'] = float(np.mean(1 - task[task < 1]) * 100) if (task < 1).any() else 0.
            v['mean_harm_when_harm_pct'] = float(np.mean(task[task > 1] - 1) * 100) if (task > 1).any() else 0.
            v['sum_gain_pct'] = float(np.sum(np.clip(1 - task, 0, None)) * 100); v['sum_harm_pct'] = float(np.sum(np.clip(task - 1, 0, None)) * 100)
    if m != OURS:
        for k in ('cvar80', 'cvar90', 'cvar95', 'mae'):
            w = sum(rows[c + (OURS,)][k] < rows[c + (m,)][k] for c in cells); l = sum(rows[c + (OURS,)][k] > rows[c + (m,)][k] for c in cells)
            v['ours_vs_%s' % k] = [int(w), int(l), float(binomtest(w, w + l).pvalue) if w + l else 1.]
    out[m] = v
(F.ROOT / 'results/paper/metric_scan.json').write_text(json.dumps(out, indent=1), 'utf-8')
keys = [k for k in out[OURS]]
for k in keys:
    print('%-24s' % k, ' '.join('%s=%s' % (m[:8], ('%.4f' % out[m][k]) if isinstance(out[m][k], float) else out[m][k]) for m in cmp))
for k in ('cvar80', 'cvar90', 'cvar95', 'mae'):
    print('vs', k, {m: out[m]['ours_vs_' + k] for m in cmp if m != OURS})
