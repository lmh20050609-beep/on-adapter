"""Main table on seeds 2025-2027 with the user's method set (MAIN_TABLE_DECISION.json, OGD removed) plus COSA:
On-Adapter + iTransformer, ADAPT-Z, DSOF, Proceed, GPT4MTS, MM-TSFlib, SOLID, ADCSD, COSA (ranked among these ten);
Original On-Adapter as reference row. Writes results/new_baselines/results_main_cosa.json."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'final'))
import scoring as F
import solid as SO
import adcsd_ogd as AO
import cosa as CO
r1, fz, f5 = F.r1, F.fz, F.f5; SEEDS = [2025, 2026, 2027]; OURS = F.OURS; R = SO.R
lr = json.loads((R / 'solid_selection.json').read_text())['selected_lr']; rows = {}; upd = []
for d, h in r1.tasks():
    for s in SEEDS:
        for m in F.METHODS: rows[(d, h, s, m)], _ = F.baseline(m, d, h, s)
        cv = f5.traj(F.tfolder('cv', d, h, s)); rows[(d, h, s, OURS)] = fz.cell_metrics(f5.gated(cv, h)[0], cv['targets'], cv['mask'])
        a, _ = SO.load('test', lr, d, h, s); rows[(d, h, s, 'SOLID')] = fz.cell_metrics(a['predictions'], a['targets'], a['mask'])
        a, _ = AO.load('ADCSD', d, h, s); rows[(d, h, s, 'ADCSD')] = fz.cell_metrics(a['predictions'], a['targets'], a['mask'])
        a, meta = CO.load(d, h, s); rows[(d, h, s, 'COSA')] = fz.cell_metrics(a['predictions'], a['targets'], a['mask']); upd.append(meta['updates'])
cmp = [OURS] + [m for m in F.METHODS if m != 'On-Adapter'] + ['SOLID', 'ADCSD', 'COSA']
out = {m: F.summary(rows, m, cmp, SEEDS) for m in cmp}
out['Original On-Adapter'] = F.summary(rows, 'On-Adapter', [m if m != OURS else 'On-Adapter' for m in cmp], SEEDS)
json.dump(dict(seeds=SEEDS, methods_ranked=cmp, solid_lr=lr, main=out, cosa_mean_updates=float(sum(upd) / len(upd)),
               plans=['PLAN.json', 'PLAN_adcsd_ogd.json', 'PLAN_cosa.json']), open(R / 'results_main_cosa.json', 'w'), indent=1)
for m in sorted(out, key=lambda m: out[m]['S']):
    v = out[m]; print('%-20s S=%.5f rank=%s better=%2d harm>2%%=%2d worst=%.3f mae=%.4f mse=%.3f seeds %s' % (m, v['S'], '%.3f' % v['mean_rank'] if v['mean_rank'] else '  -  ',
          v['tasks_better_than_backbone'], v['harm_tasks_over_2pct'], v['worst_task_ratio'], v['mean_mae'], v['mean_mse'], ' '.join('%.4f' % x for x in v['per_seed_S'].values())))
print('COSA per-domain:', {k: round(v, 3) for k, v in out['COSA']['per_domain'].items()})
print('ours per-domain:', {k: round(v, 3) for k, v in out[OURS]['per_domain'].items()})
