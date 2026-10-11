"""Final scoring on seeds 2023-2027 (primary) and on all seven seeds {5, 6, 2023..2027} (results/final/seven_seeds/PLAN.json).
  --list <nofb|val> ; --traj <nofb|val> <domain> <H> <seed>   (online-component trajectories for seeds 2023, 2024)
  --score                                                      (main table, all ablations, online-component ablations)
"""
import argparse, hashlib, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'confirmation'))
import numpy as np
import five_seeds as f5
import online_ablation as oa
hx, r1, r5, fz = oa.hx, f5.r1, f5.r5, f5.fz
for s in (2023, 2024): hx.PACKAGES.setdefault('seed%d-on' % s, 'method_comparison_seed%d_on_20261011' % s)
ROOT = r1.ROOT; R = ROOT / 'results/final/seven_seeds'; RUNS = f5.RUNS
PRIMARY = [2023, 2024, 2025, 2026, 2027]; ALL = [5, 6] + PRIMARY; OLD = {2025, 2026, 2027}; DATE = {5: '20261010', 6: '20261010', 2023: '20261011', 2024: '20261011'}
OUT = RUNS / 'final_2023_2027_online_traj_20261011'; OURS = f5.OURS; METHODS = f5.METHODS
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tfolder(kind, d, h, s):
    if s in OLD: return f5.tfolder(kind, d, h, s)
    base = 'seed%d_final_traj_%s/cv' % (s, DATE[s]) if kind == 'cv' else 'seed%d_ablation_traj_%s/%s' % (s, DATE[s], kind)
    return RUNS / base / d / ('H%d' % h)


def online_folder(kind, d, h, s):
    if s in (2023, 2024): return OUT / kind / d / ('H%d' % h) / ('seed%d' % s)
    return oa.OUT / kind / d / ('H%d' % h) / ('seed%d' % s)


def baseline(m, d, h, s):
    if s in OLD: return f5.baseline(m, d, h, s)
    folder = RUNS / ('method_comparison_seed%d_v1_%s' % (s, DATE[s])) / d / ('H%d' % h) / m / ('seed%d' % s)
    if r1.admitted(folder) is None: raise SystemExit('missing ' + str(folder))
    with np.load(folder / 'predictions.npz') as z: arr = {k: z[k] for k in z.files}
    return fz.cell_metrics(arr['predictions'], arr['targets'], arr['mask']), arr


def ckpt(d, h, s): return RUNS / ('method_comparison_seed%d_on_cv_20261011' % s) / d / ('H%d' % h) / 'On-Adapter' / ('seed%d' % s)


def traj(kind, d, h, s):
    assert s in (2023, 2024)
    rd = ckpt(d, h, s)
    if r1.admitted(rd) is None: raise SystemExit('cell not complete')
    out = online_folder(kind, d, h, s); out.mkdir(parents=True, exist_ok=True)
    ident = dict(plan_sha256=sha(R / 'PLAN.json'), kind=kind, best_pt_sha256=sha(rd / 'best.pt'), code_sha256={p.name: sha(p) for p in [Path(__file__), Path(oa.__file__), Path(hx.__file__)]})
    done = out / 'r0.json'
    if done.is_file() and json.loads(done.read_text('utf-8')).get('identity') == ident: return
    ctx = hx.setup(rd, 'cpu'); t0 = time.time()
    if kind == 'nofb': a = oa.no_feedback(ctx, 'test')
    else: a, _, _ = hx.run_online(ctx, 'val', 0.0, measure=False, rng_seed=100 * s)
    np.savez_compressed(out / 'r0.npz', predictions=a['pred'], backbone=a['base'], targets=a['y'], mask=a['mask'], origins=a['origin'])
    done.write_text(json.dumps(dict(identity=ident, seconds=time.time() - t0, artifacts={'r0.npz': sha(out / 'r0.npz')}), indent=1), 'utf-8')


VARIANTS = f5.VARIANTS + ['ours - feedback (hg = ge = 0)', 'ours with static gate (validation choice)']


def cells():
    rows = {}
    for d, h in r1.tasks():
        for s in ALL:
            arrs = {}
            for m in METHODS: rows[(d, h, s, m)], arrs[m] = baseline(m, d, h, s)
            orig = arrs['On-Adapter']
            T = {k: f5.traj(tfolder(k, d, h, s)) for k in ['cv', 'ta', 'va', 'e3', 'orig_lr0', 'cv_lr']}
            nf, va = oa._load(online_folder('nofb', d, h, s)), oa._load(online_folder('val', d, h, s))
            if not np.array_equal(orig['origins'], T['orig_lr0']['origins']): raise SystemExit('origin mismatch %s H%d s%d' % (d, h, s))
            if not (np.array_equal(nf['backbone'], T['cv']['backbone']) and np.array_equal(nf['predictions'][0], T['cv']['predictions'][0])):
                raise SystemExit('feedback-off replay inconsistent %s H%d s%d' % (d, h, s))
            put = lambda n, p, t: rows.__setitem__((d, h, s, n), fz.cell_metrics(p, t['targets'], t['mask']))
            g = f5.gated
            put(OURS, g(T['cv'], h)[0], T['cv'])
            put('L1 Original + online updates off', T['orig_lr0']['predictions'], T['orig_lr0'])
            put('L2 + VICReg off, 15 epochs', T['ta']['predictions'], T['ta']); put('L3 + backbone-first gate', g(T['ta'], h)[0], T['ta'])
            put('ours - gate', T['cv']['predictions'], T['cv']); put('ours - online updates off (lr 3e-4)', g(T['cv_lr'], h)[0], T['cv_lr'])
            put('ours - VICReg off (VICReg on)', g(T['va'], h)[0], T['va']); put('ours - 15 epochs (3 epochs)', g(T['e3'], h)[0], T['e3'])
            put('ours, gate gs 0.8', g(T['cv'], h, .8)[0], T['cv']); put('ours, gate gs 1.0', g(T['cv'], h, 1.)[0], T['cv'])
            put('Original On-Adapter + gate', g(dict(T['orig_lr0'], predictions=orig['predictions']), h)[0], T['orig_lr0'])
            put('ours - feedback (hg = ge = 0)', g(nf, h)[0], nf)
            use = fz.cell_metrics(va['predictions'], va['targets'], va['mask'])['cvar90'] < fz.cell_metrics(va['backbone'], va['targets'], va['mask'])['cvar90']
            put('ours with static gate (validation choice)', T['cv']['predictions'] if use else T['cv']['backbone'], T['cv'])
    return rows


def summary(rows, m, cmp, seeds):
    ratio = lambda d, h, s: rows[(d, h, s, m)]['cvar90'] / rows[(d, h, s, 'iTransformer')]['cvar90']
    task = {(d, h): float(np.mean([ratio(d, h, s) for s in seeds])) for d, h in r1.tasks()}
    ranks = [1 + sorted(cmp, key=lambda x: rows[(d, h, s, x)]['cvar90']).index(m) for d, h in r1.tasks() for s in seeds] if cmp else []
    v = list(task.values())
    return dict(S=fz.geometric_mean(v), mean_rank=float(np.mean(ranks)) if ranks else None,
                per_seed_S={str(s): fz.geometric_mean([ratio(d, h, s) for d, h in r1.tasks()]) for s in seeds},
                mean_mae=float(np.mean([rows[(d, h, s, m)]['mae'] for d, h in r1.tasks() for s in seeds])),
                mean_mse=float(np.mean([rows[(d, h, s, m)]['mse'] for d, h in r1.tasks() for s in seeds])),
                tasks_better_than_backbone=int(sum(x < 1 for x in v)), harm_tasks_over_2pct=int(sum(x > 1.02 for x in v)), worst_task_ratio=float(max(v)),
                per_domain={dd: fz.geometric_mean([task[(d, h)] for d, h in r1.tasks() if d == dd]) for dd in dict.fromkeys(d for d, _ in r1.tasks())},
                per_task={'%s H%d' % k: x for k, x in task.items()})


def score():
    rows = cells(); out = dict(plan_sha256=sha(R / 'PLAN.json'))
    for label, seeds in (('primary_2023_2027', PRIMARY), ('all_seven_seeds', ALL)):
        main_cmp = [m if m != 'On-Adapter' else OURS for m in METHODS]
        main = {m: summary(rows, m, main_cmp, seeds) for m in main_cmp}; main['Original On-Adapter'] = summary(rows, 'On-Adapter', METHODS, seeds)
        abl = {v: summary(rows, v, [m if m != 'On-Adapter' else v for m in METHODS], seeds) for v in VARIANTS}; abl['L0 Original On-Adapter'] = main['Original On-Adapter']
        out[label] = dict(seeds=seeds, main=main, ablation=abl, strict_first=all(main[OURS]['S'] < main[m]['S'] for m in main_cmp if m != OURS))
        fmt = lambda n, v: '%-44s S=%.6f rank=%s better=%2d harm>2%%=%2d worst=%.3f mae=%.4f seeds %s' % (n, v['S'], '%.3f' % v['mean_rank'] if v['mean_rank'] else '  -  ',
              v['tasks_better_than_backbone'], v['harm_tasks_over_2pct'], v['worst_task_ratio'], v['mean_mae'], ' '.join('%.4f' % x for x in v['per_seed_S'].values()))
        print('==== %s %s' % (label, seeds)); print('== main'); [print(fmt(m, main[m])) for m in sorted(main, key=lambda m: main[m]['S'])]
        print('== ablation'); [print(fmt(n, abl[n])) for n in ['L0 Original On-Adapter'] + VARIANTS]
        print('strict first:', out[label]['strict_first'])
    (R / 'results.json').write_text(json.dumps(out, indent=1), 'utf-8')


def listing(kind):
    for d, h in r1.tasks():
        for s in (2023, 2024):
            if (ckpt(d, h, s) / 'result.json').is_file() and not (online_folder(kind, d, h, s) / 'r0.json').is_file(): print('%s %s %d %d' % (kind, d, h, s))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--traj', nargs=4); ap.add_argument('--score', action='store_true'); ap.add_argument('--list')
    a = ap.parse_args()
    if a.traj: traj(a.traj[0], a.traj[1], int(a.traj[2]), int(a.traj[3]))
    elif a.score: score()
    elif a.list: listing(a.list)
