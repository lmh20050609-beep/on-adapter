"""Scan for illustrative success/failure cases of On-Adapter among the ten main-table methods (seeds 2025-2027).
Read-only; writes results/paper/showcase.json."""
import json, sys
from pathlib import Path
ROOT_ = Path(__file__).resolve().parents[1]
for _p in (Path(__file__).resolve().parent, ROOT_ / 'final', ROOT_ / 'new_baselines', ROOT_ / 'confirmation'): sys.path.insert(0, str(_p))
import numpy as np
import scoring as F, solid as SO, adcsd_ogd as AO
r1, fz, f5 = F.r1, F.fz, F.f5; SEEDS = [2025, 2026, 2027]
lr = json.loads((SO.R / 'solid_selection.json').read_text())['selected_lr']
OTHERS = ['iTransformer', 'ADAPT-Z', 'DSOF', 'Proceed', 'GPT4MTS', 'MM-TSFlib', 'SOLID', 'ADCSD', 'OGD']
def per_origin(p, y, m):
    e = np.abs(p.astype(np.float64) - y.astype(np.float64)); return np.array([x[k].mean() for x, k in zip(e, m.astype(bool))])
cells = []
for d, h in r1.tasks():
    for s in SEEDS:
        P = {}
        for m in ['iTransformer', 'ADAPT-Z', 'DSOF', 'Proceed', 'GPT4MTS', 'MM-TSFlib']:
            _, arr = F.baseline(m, d, h, s); P[m] = arr
        cv = f5.traj(F.tfolder('cv', d, h, s)); g, ch = f5.gated(cv, h)
        y, mk, o = cv['targets'], cv['mask'], cv['origins']
        P['ours'] = dict(predictions=g, origins=o)
        P['SOLID'] = SO.load('test', lr, d, h, s)[0]
        for m in ('ADCSD', 'OGD'): P[m] = AO.load(m, d, h, s)[0]
        for m, a in P.items():
            if not np.array_equal(a['origins'], o): raise SystemExit('origin mismatch %s %s H%d s%d' % (m, d, h, s))
        E = {m: per_origin(a['predictions'], y, mk) for m, a in P.items()}
        cv90 = {m: fz.cell_metrics(P[m]['predictions'], y, mk)['cvar90'] for m in P}
        best_other = min(OTHERS, key=lambda m: cv90[m])
        rank = 1 + sorted(['ours'] + OTHERS, key=lambda m: cv90[m]).index('ours')
        # best window: 12 consecutive origins where ours beats every other method by the largest margin in mean per-origin MAE
        W = min(12, len(o)); best = None
        for i in range(len(o) - W + 1):
            ours_w = E['ours'][i:i + W].mean(); other_w = min(E[m][i:i + W].mean() for m in OTHERS)
            gap = (other_w - ours_w) / other_w
            if best is None or gap > best[0]: best = (gap, i)
        gap, i = best
        cells.append(dict(domain=d, H=h, seed=s, rank=rank, ours_cvar90_ratio_to_backbone=cv90['ours'] / cv90['iTransformer'],
                          margin_to_best_other=(cv90[best_other] - cv90['ours']) / cv90[best_other], best_other=best_other,
                          window_start_origin=int(o[i]), window_len=W, window_gap=float(gap), window_adapter_share=float(ch[i:i + W].mean()),
                          adapter_share=float(ch.mean())))
json.dump(cells, open(F.ROOT / 'results/paper/showcase.json', 'w'), indent=1)
print('cells where ours ranks 1st among ten: %d / %d' % (sum(c['rank'] == 1 for c in cells), len(cells)))
print('top success cells (ours 1st, largest CVaR90 margin over the best other method):')
for c in sorted([c for c in cells if c['rank'] == 1], key=lambda c: -c['margin_to_best_other'])[:8]:
    print('  %-12s H%-3d s%d margin %.1f%% vs %-12s ratio-to-backbone %.3f | best 12-origin window @%d: %.0f%% lower MAE than every other method, adapter used %.0f%%'
          % (c['domain'], c['H'], c['seed'], 100 * c['margin_to_best_other'], c['best_other'], c['ours_cvar90_ratio_to_backbone'], c['window_start_origin'], 100 * c['window_gap'], 100 * c['window_adapter_share']))
print('worst cells (failure candidates):')
for c in sorted(cells, key=lambda c: c['margin_to_best_other'])[:4]:
    print('  %-12s H%-3d s%d rank %d, %.1f%% worse than %s, ratio-to-backbone %.3f' % (c['domain'], c['H'], c['seed'], c['rank'], -100 * c['margin_to_best_other'], c['best_other'], c['ours_cvar90_ratio_to_backbone']))
