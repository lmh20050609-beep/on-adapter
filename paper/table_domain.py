"""Paper Table 2 data: per-domain MSE / MAE / CVaR90 averaged over the four horizons and seeds 2025-2027 for the ten
main-table methods (+ Original On-Adapter reference). Same cells as v1ext_main_cosa_20261011.py.
Writes results/paper/table_domain.json."""
import json, sys
from pathlib import Path
ROOT_ = Path(__file__).resolve().parents[1]
for _p in (Path(__file__).resolve().parent, ROOT_ / 'final', ROOT_ / 'new_baselines', ROOT_ / 'confirmation'): sys.path.insert(0, str(_p))
import numpy as np
import scoring as F
import solid as SO
import adcsd_ogd as AO
import cosa as CO
r1, fz, f5 = F.r1, F.fz, F.f5; SEEDS = [2025, 2026, 2027]; OURS = F.OURS; R = SO.R
lr = json.loads((R / 'solid_selection.json').read_text())['selected_lr']; rows = {}
for d, h in r1.tasks():
    for s in SEEDS:
        for m in F.METHODS: rows[(d, h, s, m)], _ = F.baseline(m, d, h, s)
        cv = f5.traj(F.tfolder('cv', d, h, s)); rows[(d, h, s, OURS)] = fz.cell_metrics(f5.gated(cv, h)[0], cv['targets'], cv['mask'])
        a, _ = SO.load('test', lr, d, h, s); rows[(d, h, s, 'SOLID')] = fz.cell_metrics(a['predictions'], a['targets'], a['mask'])
        a, _ = AO.load('ADCSD', d, h, s); rows[(d, h, s, 'ADCSD')] = fz.cell_metrics(a['predictions'], a['targets'], a['mask'])
        a, _ = CO.load(d, h, s); rows[(d, h, s, 'COSA')] = fz.cell_metrics(a['predictions'], a['targets'], a['mask'])
meths = [OURS] + [m for m in F.METHODS if m != 'On-Adapter'] + ['SOLID', 'ADCSD', 'COSA', 'On-Adapter']
out = {m: {dd: {k: float(np.mean([rows[(d, h, s, m)][k] for d, h in r1.tasks() if d == dd for s in SEEDS])) for k in ('mse', 'mae', 'cvar90')}
           for dd in dict.fromkeys(d for d, _ in r1.tasks())} for m in meths}
(F.ROOT / 'results/paper/table_domain.json').write_text(json.dumps(dict(seeds=SEEDS, ours_key=OURS, methods=meths, per_domain=out), indent=1), 'utf-8')
for m in meths: print(m, {dd: '%.3f/%.3f/%.3f' % (v['mse'], v['mae'], v['cvar90']) for dd, v in out[m].items()})
