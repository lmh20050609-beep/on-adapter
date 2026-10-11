"""Online-component ablations (results/final/five_seeds/ONLINE_ABLATION_PLAN.json).
  --list <kind> ; --traj <kind> <domain> <H> <seed> ; --score
kinds: nofb (test, feedback hg = ge = 0, no updates), val (validation trajectory with feedback, lr 0, for the static gate).
"""
import argparse, hashlib, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'confirmation'))
import numpy as np
import torch
import harness as hx
import five_seeds as f5
r1, r5, fz = f5.r1, f5.r5, f5.fz
hx.PACKAGES.setdefault('seed5-on', 'method_comparison_seed5_on_20261010'); hx.PACKAGES.setdefault('seed6-on', 'method_comparison_seed6_on_20261010')
R, RUNS, SEEDS = f5.R, f5.RUNS, f5.SEEDS; PLAN = R / 'ONLINE_ABLATION_PLAN.json'
OUT = RUNS / 'final5_online_ablation_traj_20261010'; CODE = [Path(__file__), Path(hx.__file__)]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ckpt(d, h, s):
    camp = 'method_comparison_v1conf_cv_20261009' if s in f5.OLD else 'method_comparison_seed%d_on_cv_20261010' % s
    return RUNS / camp / d / ('H%d' % h) / 'On-Adapter' / ('seed%d' % s)


def no_feedback(ctx, split):
    """Origin-by-origin forecasts with hg = ge = 0 throughout; same safe batch as the online loop, no updates."""
    batching = hx.module(ctx, 'batching'); model, dev = ctx['model'], ctx['device']
    model.load_state_dict(ctx['best'], strict=False); model.reset_online(); model.eval()
    rec = {k: [] for k in ['pred', 'base', 'y', 'mask', 'origin']}; windows = ctx['sets'][split]
    with torch.no_grad():
        for item in range(len(windows)):
            host = batching.host_batch(windows, [item]); batch = batching.transfer(host, dev)
            assert float(model.hg.abs().sum()) == 0 and float(model.ge.abs().sum()) == 0
            safe = dict(batch); safe['y'] = torch.zeros_like(batch['y']); safe['mask'] = torch.zeros_like(batch['mask'])
            rec['pred'].append(model(safe).cpu().numpy()); rec['base'].append(model.last_base.detach().cpu().numpy())
            rec['y'].append(host['y'].numpy()); rec['mask'].append(host['mask'].numpy()); rec['origin'].append(host['origin'].numpy())
    return {k: np.concatenate(v) for k, v in rec.items()}


def traj(kind, d, h, s):
    rd = ckpt(d, h, s)
    if r1.admitted(rd) is None: raise SystemExit('cell not complete')
    out = OUT / kind / d / ('H%d' % h) / ('seed%d' % s); out.mkdir(parents=True, exist_ok=True)
    ident = dict(plan_sha256=sha(PLAN), kind=kind, best_pt_sha256=sha(rd / 'best.pt'), code_sha256={p.name: sha(p) for p in CODE})
    done = out / 'r0.json'
    if done.is_file() and json.loads(done.read_text('utf-8')).get('identity') == ident: return
    ctx = hx.setup(rd, 'cpu'); t0 = time.time()
    if kind == 'nofb': a = no_feedback(ctx, 'test')
    else: a, _, _ = hx.run_online(ctx, 'val', 0.0, measure=False, rng_seed=100 * s)
    np.savez_compressed(out / 'r0.npz', predictions=a['pred'], backbone=a['base'], targets=a['y'], mask=a['mask'], origins=a['origin'])
    done.write_text(json.dumps(dict(identity=ident, seconds=time.time() - t0, artifacts={'r0.npz': sha(out / 'r0.npz')}), indent=1), 'utf-8')


def load(kind, d, h, s):
    f = OUT / kind / d / ('H%d' % h) / ('seed%d' % s)
    if json.loads((f / 'r0.json').read_text('utf-8'))['identity']['plan_sha256'] != sha(PLAN): raise SystemExit('plan changed')
    return _load(f)


def _load(f):
    meta = json.loads((f / 'r0.json').read_text('utf-8'))
    for n, w in meta['artifacts'].items():
        if sha(f / n) != w: raise SystemExit('artifact changed: %s' % f)
    with np.load(f / 'r0.npz') as z: return {k: z[k] for k in z.files}


def score():
    rows, choice = {}, {}
    names = ['On-Adapter (ours)', 'ours - gate', 'ours - feedback (hg = ge = 0)', 'ours - feedback, no gate', 'ours with static gate (validation choice)']
    for d, h in r1.tasks():
        for s in SEEDS:
            for m in f5.METHODS: rows[(d, h, s, m)], _ = f5.baseline(m, d, h, s)
            cv = f5.traj(f5.tfolder('cv', d, h, s)); nf = load('nofb', d, h, s); va = load('val', d, h, s)
            if not (np.array_equal(cv['origins'], nf['origins']) and np.array_equal(cv['backbone'], nf['backbone'])):
                raise SystemExit('feedback-off replay does not share origins/backbone with On-Adapter: %s H%d s%d' % (d, h, s))
            if not np.array_equal(cv['predictions'][0], nf['predictions'][0]):   # first origin: no feedback has matured in either replay
                raise SystemExit('feedback-off replay differs before any feedback: %s H%d s%d' % (d, h, s))
            put = lambda n, p, t: rows.__setitem__((d, h, s, n), fz.cell_metrics(p, t['targets'], t['mask']))
            put('On-Adapter (ours)', f5.gated(cv, h)[0], cv); put('ours - gate', cv['predictions'], cv)
            put('ours - feedback (hg = ge = 0)', f5.gated(nf, h)[0], nf); put('ours - feedback, no gate', nf['predictions'], nf)
            use = fz.cell_metrics(va['predictions'], va['targets'], va['mask'])['cvar90'] < fz.cell_metrics(va['backbone'], va['targets'], va['mask'])['cvar90']
            choice[(d, h, s)] = bool(use)
            put('ours with static gate (validation choice)', cv['predictions'] if use else cv['backbone'], cv)
    out = {}
    for n in names:
        cmp = [m if m != 'On-Adapter' else n for m in f5.METHODS]
        task = {(d, h): float(np.mean([rows[(d, h, s, n)]['cvar90'] / rows[(d, h, s, 'iTransformer')]['cvar90'] for s in SEEDS])) for d, h in r1.tasks()}
        ranks = [1 + sorted(cmp, key=lambda x: rows[(d, h, s, x)]['cvar90']).index(n) for d, h in r1.tasks() for s in SEEDS]
        v = list(task.values())
        out[n] = dict(S=fz.geometric_mean(v), mean_rank=float(np.mean(ranks)), tasks_better_than_backbone=int(sum(x < 1 for x in v)),
                      harm_tasks_over_2pct=int(sum(x > 1.02 for x in v)), worst_task_ratio=float(max(v)),
                      per_seed_S={str(s): fz.geometric_mean([rows[(d, h, s, n)]['cvar90'] / rows[(d, h, s, 'iTransformer')]['cvar90'] for d, h in r1.tasks()]) for s in SEEDS},
                      per_domain={dd: fz.geometric_mean([task[(d, h)] for d, h in r1.tasks() if d == dd]) for dd in dict.fromkeys(d for d, _ in r1.tasks())})
    out['static gate: share of task-seed cells using the adapter'] = float(np.mean(list(choice.values())))
    (R / 'results_online_ablation.json').write_text(json.dumps(dict(plan_sha256=sha(PLAN), summary=out), indent=1), 'utf-8')
    for n in names:
        v = out[n]; print('%-44s S=%.6f rank=%.3f better=%2d harm>2%%=%2d worst=%.3f seeds %s' % (n, v['S'], v['mean_rank'], v['tasks_better_than_backbone'],
              v['harm_tasks_over_2pct'], v['worst_task_ratio'], ' '.join('%.4f' % x for x in v['per_seed_S'].values())))
    print('static gate uses the adapter in %.0f%% of task-seed cells' % (100 * out['static gate: share of task-seed cells using the adapter']))


def listing(kind):
    for d, h in r1.tasks():
        for s in SEEDS:
            if (ckpt(d, h, s) / 'result.json').is_file() and not (OUT / kind / d / ('H%d' % h) / ('seed%d' % s) / 'r0.json').is_file():
                print('%s %s %d %d' % (kind, d, h, s))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--traj', nargs=4); ap.add_argument('--score', action='store_true'); ap.add_argument('--list')
    a = ap.parse_args()
    if a.traj: traj(a.traj[0], a.traj[1], int(a.traj[2]), int(a.traj[3]))
    elif a.score: score()
    elif a.list: listing(a.list)
