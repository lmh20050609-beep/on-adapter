"""Main table on seeds 2025-2027: On-Adapter + the six V1 baselines + SOLID + ADCSD + OGD (ranked among these ten);
Original On-Adapter as reference row. TimeXer/SOFTS results stay recorded in results/new_baselines/results.json.
Writes results/new_baselines/results_main10.json."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'final'))
import numpy as np
import scoring as F
import solid as SO
import adcsd_ogd as AO
r1, fz, f5 = F.r1, F.fz, F.f5; SEEDS = [2025, 2026, 2027]; OURS = F.OURS; R = SO.R
lr = json.loads((R / 'solid_selection.json').read_text())['selected_lr']; rows = {}
for d, h in r1.tasks():
    for s in SEEDS:
        for m in F.METHODS: rows[(d, h, s, m)], _ = F.baseline(m, d, h, s)
        cv = f5.traj(F.tfolder('cv', d, h, s)); rows[(d, h, s, OURS)] = fz.cell_metrics(f5.gated(cv, h)[0], cv['targets'], cv['mask'])
        a, _ = SO.load('test', lr, d, h, s); rows[(d, h, s, 'SOLID')] = fz.cell_metrics(a['predictions'], a['targets'], a['mask'])
        for m in ('ADCSD', 'OGD'):
            a, _ = AO.load(m, d, h, s); rows[(d, h, s, m)] = fz.cell_metrics(a['predictions'], a['targets'], a['mask'])
cmp = [OURS] + [m for m in F.METHODS if m != 'On-Adapter'] + ['SOLID', 'ADCSD', 'OGD']
out = {m: F.summary(rows, m, cmp, SEEDS) for m in cmp}
out['Original On-Adapter'] = F.summary(rows, 'On-Adapter', [m if m != OURS else 'On-Adapter' for m in cmp], SEEDS)
json.dump(dict(seeds=SEEDS, methods_ranked=cmp, solid_lr=lr, main=out, plans=['PLAN.json', 'PLAN_adcsd_ogd.json']),
          open(R / 'results_main10.json', 'w'), indent=1)
for m in sorted(out, key=lambda m: out[m]['S']):
    v = out[m]; print('%-20s S=%.4f rank=%s better=%2d harm>2%%=%2d worst=%.3f mae=%.4f mse=%.3f' % (m, v['S'], '%.3f' % v['mean_rank'] if v['mean_rank'] else '  -  ',
          v['tasks_better_than_backbone'], v['harm_tasks_over_2pct'], v['worst_task_ratio'], v['mean_mae'], v['mean_mse']))
