"""Five-seed final scoring (results/final/five_seeds/PLAN.json): seeds 2025, 2026, 2027, 5, 6.
Main table (7 methods with On-Adapter in place of Original On-Adapter) and every ablation of ABLATION_PLAN.json.
Read-only on all trajectories and campaign cells; writes results/final/five_seeds/results.json.
"""
import hashlib, json, math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'confirmation'))
import numpy as np
import round1_calibration as r1, round5_gate as r5
ROOT = r1.ROOT; sys.path.insert(0, str(ROOT))
import benchmark.score as fz
R = ROOT / 'results/final/five_seeds'; RUNS = Path('E:/FedNoise/iadapter-baselines/runs')
SEEDS = [2025, 2026, 2027, 5, 6]; OLD = {2025, 2026, 2027}
METHODS = ['iTransformer', 'ADAPT-Z', 'DSOF', 'Proceed', 'On-Adapter', 'GPT4MTS', 'MM-TSFlib']
OURS = 'On-Adapter (ours)'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest = json.loads((ROOT / 'benchmark/legacy_numeric_reuse_manifest.json').read_text('utf-8'))


def traj(folder):
    meta = json.loads((folder / 'r0.json').read_text('utf-8'))
    for n, w in meta['artifacts'].items():
        if sha(folder / n) != w: raise SystemExit('artifact changed: %s' % folder)
    with np.load(folder / 'r0.npz') as z: return {k: z[k] for k in z.files}


def tfolder(kind, d, h, s):
    if s in OLD:
        base = {'cv': 'v1conf_traj_20261009/cv', 'ta': 'v1conf_traj_20261009/ta'}.get(kind, 'final5_ablation_traj_20261010/' + kind)
        return RUNS / base / d / ('H%d' % h) / ('seed%d' % s)
    base = 'seed%d_final_traj_20261010/cv' % s if kind == 'cv' else 'seed%d_ablation_traj_20261010/%s' % (s, kind)
    return RUNS / base / d / ('H%d' % h)


def baseline(m, d, h, s):
    if s in OLD:
        v = fz.load_cell(m, d, h, s, manifest); folder = Path(v['directory'])
    else:
        folder = RUNS / ('method_comparison_seed%d_v1_20261010' % s) / d / ('H%d' % h) / m / ('seed%d' % s)
        if r1.admitted(folder) is None: raise SystemExit('missing ' + str(folder))
    with np.load(folder / 'predictions.npz') as z: arr = {k: z[k] for k in z.files}
    return fz.cell_metrics(arr['predictions'], arr['targets'], arr['mask']), arr


def gated(t, h, gs=.9):
    return r5.abstain(t['backbone'], t['predictions'].astype(np.float64), t['targets'].astype(np.float64), t['mask'], t['origins'], h, gs, 0.)


VARIANTS = ['L1 Original + online updates off', 'L2 + VICReg off, 15 epochs', 'L3 + backbone-first gate', OURS,
            'ours - gate', 'ours - online updates off (lr 3e-4)', 'ours - VICReg off (VICReg on)', 'ours - 15 epochs (3 epochs)',
            'ours, gate gs 0.8', 'ours, gate gs 1.0', 'Original On-Adapter + gate']


def main():
    rows, share = {}, {}
    for d, h in r1.tasks():
        for s in SEEDS:
            arrs = {}
            for m in METHODS: rows[(d, h, s, m)], arrs[m] = baseline(m, d, h, s)
            orig = arrs['On-Adapter']
            T = {k: traj(tfolder(k, d, h, s)) for k in ['cv', 'ta', 'va', 'e3', 'orig_lr0', 'cv_lr']}
            if not np.array_equal(orig['origins'], T['orig_lr0']['origins']): raise SystemExit('origin mismatch %s H%d s%d' % (d, h, s))
            put = lambda name, p, t: rows.__setitem__((d, h, s, name), fz.cell_metrics(p, t['targets'], t['mask']))
            p, ch = gated(T['cv'], h); put(OURS, p, T['cv']); share[(d, h, s)] = float(np.mean(ch))
            put('L1 Original + online updates off', T['orig_lr0']['predictions'], T['orig_lr0'])
            put('L2 + VICReg off, 15 epochs', T['ta']['predictions'], T['ta'])
            put('L3 + backbone-first gate', gated(T['ta'], h)[0], T['ta'])
            put('ours - gate', T['cv']['predictions'], T['cv'])
            put('ours - online updates off (lr 3e-4)', gated(T['cv_lr'], h)[0], T['cv_lr'])
            put('ours - VICReg off (VICReg on)', gated(T['va'], h)[0], T['va'])
            put('ours - 15 epochs (3 epochs)', gated(T['e3'], h)[0], T['e3'])
            put('ours, gate gs 0.8', gated(T['cv'], h, .8)[0], T['cv'])
            put('ours, gate gs 1.0', gated(T['cv'], h, 1.)[0], T['cv'])
            put('Original On-Adapter + gate', gated(dict(T['orig_lr0'], predictions=orig['predictions']), h)[0], T['orig_lr0'])

    def ratio(m, d, h, s): return rows[(d, h, s, m)]['cvar90'] / rows[(d, h, s, 'iTransformer')]['cvar90']
    def summary(m, cmp):
        task = {(d, h): float(np.mean([ratio(m, d, h, s) for s in SEEDS])) for d, h in r1.tasks()}
        ranks = []
        if cmp:
            for d, h in r1.tasks():
                for s in SEEDS: ranks.append(1 + sorted(cmp, key=lambda x: rows[(d, h, s, x)]['cvar90']).index(m))
        v = list(task.values())
        return dict(S=fz.geometric_mean(v), mean_rank=float(np.mean(ranks)) if ranks else None,
                    per_seed_S={str(s): fz.geometric_mean([ratio(m, d, h, s) for d, h in r1.tasks()]) for s in SEEDS},
                    mean_mae=float(np.mean([rows[(d, h, s, m)]['mae'] for d, h in r1.tasks() for s in SEEDS])),
                    mean_mse=float(np.mean([rows[(d, h, s, m)]['mse'] for d, h in r1.tasks() for s in SEEDS])),
                    tasks_better_than_backbone=int(sum(x < 1 for x in v)), harm_tasks_over_2pct=int(sum(x > 1.02 for x in v)),
                    worst_task_ratio=float(max(v)),
                    per_domain={dd: fz.geometric_mean([task[(d, h)] for d, h in r1.tasks() if d == dd]) for dd in dict.fromkeys(d for d, _ in r1.tasks())},
                    per_task={'%s H%d' % k: x for k, x in task.items()})
    main_cmp = [m if m != 'On-Adapter' else OURS for m in METHODS]
    main_tab = {m: summary(m, main_cmp) for m in main_cmp}
    main_tab['Original On-Adapter'] = summary('On-Adapter', METHODS)
    abl = {v: summary(v, [m if m != 'On-Adapter' else v for m in METHODS]) for v in VARIANTS}
    abl['L0 Original On-Adapter'] = main_tab['Original On-Adapter']
    out = dict(plan_sha256=sha(R / 'PLAN.json'), seeds=SEEDS, main=main_tab, ablation=abl,
               strict_first=all(main_tab[OURS]['S'] < main_tab[m]['S'] for m in main_cmp if m != OURS),
               adapter_share=float(np.mean(list(share.values()))))
    (R / 'results.json').write_text(json.dumps(out, indent=1), 'utf-8')
    fmt = lambda n, v: '%-40s S=%.6f rank=%s better=%2d harm>2%%=%2d worst=%.3f mae=%.4f seeds %s' % (
        n, v['S'], '%.3f' % v['mean_rank'] if v['mean_rank'] else '  -  ', v['tasks_better_than_backbone'], v['harm_tasks_over_2pct'],
        v['worst_task_ratio'], v['mean_mae'], ' '.join('%.4f' % x for x in v['per_seed_S'].values()))
    print('== main'); [print(fmt(m, main_tab[m])) for m in sorted(main_tab, key=lambda m: main_tab[m]['S'])]
    print('== ablation'); [print(fmt(n, abl[n])) for n in ['L0 Original On-Adapter'] + VARIANTS]
    print('strict first:', out['strict_first'])


if __name__ == '__main__':
    main()
