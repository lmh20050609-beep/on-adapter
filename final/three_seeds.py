"""Three-seed summary (seeds 2025, 2026, 2027) of the main table and every ablation, from already computed
trajectories (no new runs). Writes results/final/three_seeds/results.json."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'confirmation'))
import scoring as F
SEEDS = [2025, 2026, 2027]; F.ALL = SEEDS
R = F.ROOT / 'results/final/three_seeds'
rows = F.cells()
main_cmp = [m if m != 'On-Adapter' else F.OURS for m in F.METHODS]
main = {m: F.summary(rows, m, main_cmp, SEEDS) for m in main_cmp}; main['Original On-Adapter'] = F.summary(rows, 'On-Adapter', F.METHODS, SEEDS)
abl = {v: F.summary(rows, v, [m if m != 'On-Adapter' else v for m in F.METHODS], SEEDS) for v in F.VARIANTS}; abl['L0 Original On-Adapter'] = main['Original On-Adapter']
out = dict(seeds=SEEDS, main=main, ablation=abl, strict_first=all(main[F.OURS]['S'] < main[m]['S'] for m in main_cmp if m != F.OURS))
(R / 'results.json').write_text(json.dumps(out, indent=1), 'utf-8')
fmt = lambda n, v: '%-44s S=%.6f rank=%s better=%2d harm>2%%=%2d worst=%.3f mae=%.4f mse=%.3f seeds %s' % (n, v['S'], '%.3f' % v['mean_rank'] if v['mean_rank'] else '  -  ',
      v['tasks_better_than_backbone'], v['harm_tasks_over_2pct'], v['worst_task_ratio'], v['mean_mae'], v['mean_mse'], ' '.join('%.4f' % x for x in v['per_seed_S'].values()))
print('== main'); [print(fmt(m, main[m])) for m in sorted(main, key=lambda m: main[m]['S'])]
print('== ablation'); [print(fmt(n, abl[n])) for n in ['L0 Original On-Adapter'] + F.VARIANTS]
print('strict first:', out['strict_first'])
