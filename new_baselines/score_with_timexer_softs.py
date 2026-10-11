"""Ten-method main table on seeds 2025-2027 (results/new_baselines/PLAN.json): On-Adapter + the six V1 baselines
+ TimeXer, SOFTS (method_comparison_v1ext_20261011) + SOLID (selected online lr). Original On-Adapter as reference row.
Writes results/new_baselines/results.json. TimeXer/SOFTS cells come from run directories (package not included)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'final'))
import numpy as np
import scoring as F
import solid as SO
r1, fz, f5 = F.r1, F.fz, F.f5
SEEDS = [2025, 2026, 2027]; R = F.ROOT / 'results/new_baselines'
EXT = F.RUNS / 'method_comparison_v1ext_20261011'
OURS = F.OURS; BASE7 = F.METHODS; NEW = ['TimeXer', 'SOFTS', 'SOLID']
lr = json.loads((R / 'solid_selection.json').read_text())['selected_lr']
rows, bbdiff = {}, []
for d, h in r1.tasks():
    for s in SEEDS:
        for m in BASE7: rows[(d, h, s, m)], _ = F.baseline(m, d, h, s)
        cv = f5.traj(F.tfolder('cv', d, h, s)); rows[(d, h, s, OURS)] = fz.cell_metrics(f5.gated(cv, h)[0], cv['targets'], cv['mask'])
        for m in ['TimeXer', 'SOFTS']:
            f = EXT / d / ('H%d' % h) / m / ('seed%d' % s)
            if r1.admitted(f) is None: raise SystemExit('missing ' + str(f))
            with np.load(f / 'predictions.npz') as z: rows[(d, h, s, m)] = fz.cell_metrics(z['predictions'], z['targets'], z['mask'])
        a, meta = SO.load('test', lr, d, h, s); rows[(d, h, s, 'SOLID')] = fz.cell_metrics(a['predictions'], a['targets'], a['mask'])
        bbdiff.append(meta['max_abs_backbone_vs_main_table'])
cmp = [OURS] + [m for m in BASE7 if m != 'On-Adapter'] + NEW
out = {m: F.summary(rows, m, cmp, SEEDS) for m in cmp}; out['Original On-Adapter'] = F.summary(rows, 'On-Adapter', [m if m != OURS else 'On-Adapter' for m in cmp], SEEDS)
res = dict(plan_sha256=F.sha(R / 'PLAN.json'), seeds=SEEDS, solid_lr=lr, methods_ranked=cmp, main=out,
           strict_first=all(out[OURS]['S'] < out[m]['S'] for m in cmp if m != OURS), solid_max_abs_backbone_vs_main_table=max(bbdiff))
(R / 'results.json').write_text(json.dumps(res, indent=1), 'utf-8')
for m in sorted(out, key=lambda m: out[m]['S']):
    v = out[m]; print('%-22s S=%.6f rank=%s better=%2d harm>2%%=%2d worst=%.3f mae=%.4f mse=%.3f seeds %s' % (m, v['S'], '%.3f' % v['mean_rank'] if v['mean_rank'] else '  -  ',
          v['tasks_better_than_backbone'], v['harm_tasks_over_2pct'], v['worst_task_ratio'], v['mean_mae'], v['mean_mse'], ' '.join('%.4f' % x for x in v['per_seed_S'].values())))
print('strict first:', res['strict_first'], '| SOLID lr', lr, '| SOLID backbone rebuild max diff %.2e' % max(bbdiff))
