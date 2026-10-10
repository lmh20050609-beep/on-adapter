"""Seed-2 round 3 (reports/seed2_goal_20261009/round3/PLAN.json): On-Adapter training variants, lr0 online rule.

  --traj <tag> <split> <domain> <H> : one lr0 trajectory (r=0, entry seed 200) for a variant checkpoint
  --select                          : validation selection over orig + variants
  --score                           : test scoring of the selected candidate against fixed seed-2 baselines
"""
import argparse, hashlib, json, math, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import onadapter_harness_20261008 as hx
import seed2_round1_rc as r1

ROOT = hx.ROOT; sys.path.insert(0, str(ROOT))
R3 = ROOT / 'reports/seed2_goal_20261009/round3'
OUT = Path('E:/FedNoise/iadapter-baselines/runs/seed2_round3_20261009')
TAGS = ['a0e3', 'a0e15', 'a0z0e15', 'a1z0e15']
CODE = [Path(__file__), Path(hx.__file__)]


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def psha(): return sha(R3 / 'PLAN.json')
def ckpt(tag, d, h): return Path('E:/FedNoise/iadapter-baselines/runs/method_comparison_seed2_onv3_%s_20261009' % tag) / d / ('H%d' % h) / 'On-Adapter' / 'seed2'


def traj(tag, split, d, h):
    rd = ckpt(tag, d, h)
    if r1.admitted(rd) is None:
        raise SystemExit('variant cell not complete: %s' % rd)
    out = OUT / split / tag / d / ('H%d' % h); out.mkdir(parents=True, exist_ok=True)
    ident = dict(plan_sha256=psha(), tag=tag, split=split, lr=0.0, entry_seed=200, best_pt_sha256=sha(rd / 'best.pt'),
                 config_sha256=sha(rd / 'config.json'), code_sha256={p.name: sha(p) for p in CODE})
    if split == 'test':
        ident['selection_sha256'] = sha(R3 / 'selection.json')
    done = out / 'r0.json'
    if done.is_file() and json.loads(done.read_text('utf-8')).get('identity') == ident:
        return
    ctx = hx.setup(rd, 'cpu')
    t0 = time.time()
    a, ev, _ = hx.run_online(ctx, split, 0.0, measure=False, rng_seed=200)
    np.savez_compressed(out / 'r0.npz', predictions=a['pred'], backbone=a['base'], targets=a['y'], mask=a['mask'], origins=a['origin'],
                        correction=a['correction'], entry_torch_rng=a['entry_torch_rng'])
    (out / 'events_r0.json').write_text(json.dumps(ev), 'utf-8')
    done.write_text(json.dumps(dict(identity=ident, seconds=time.time() - t0, updates=len(ev), finite=bool(np.isfinite(a['pred']).all()),
                                    artifacts={n: sha(out / n) for n in ['r0.npz', 'events_r0.json']}), indent=1), 'utf-8')
    print('%s %s %s H%d %.1fs' % (tag, split, d, h, time.time() - t0), flush=True)


def load(tag, split, d, h):
    if tag == 'orig':
        t, meta = r1.load(split, d, h, 'lr0', 0); return t, meta
    folder = OUT / split / tag / d / ('H%d' % h); meta = json.loads((folder / 'r0.json').read_text('utf-8'))
    for n, w in meta['artifacts'].items():
        if sha(folder / n) != w:
            raise SystemExit('artifact changed %s' % (folder / n))
    with np.load(folder / 'r0.npz') as z:
        return {k: z[k] for k in z.files}, meta


def ratio(t):
    ref = r1.cvar90(r1.per_origin_mae(t['backbone'], t['targets'], t['mask']))
    return r1.cvar90(r1.per_origin_mae(t['predictions'], t['targets'], t['mask'])) / ref


def select():
    cands = ['orig'] + TAGS
    per = {c: {} for c in cands}; thirds = {c: [] for c in cands}
    for d, h in r1.tasks():
        for c in cands:
            t, _ = load(c, 'val', d, h)
            if not np.isfinite(t['predictions']).all():
                raise SystemExit('non-finite %s %s H%d' % (c, d, h))
            per[c]['%s H%d' % (d, h)] = ratio(t)
            ix = np.array_split(np.arange(len(t['origins'])), 3)
            eb = r1.per_origin_mae(t['backbone'], t['targets'], t['mask']); ep = r1.per_origin_mae(t['predictions'], t['targets'], t['mask'])
            thirds[c].append([r1.cvar90(ep[i]) / r1.cvar90(eb[i]) for i in ix])
    gm = lambda x: math.exp(np.mean(np.log(x)))
    score = {c: gm(list(per[c].values())) for c in cands}
    win = sorted(cands, key=lambda c: (score[c], cands.index(c)))[0]
    th = {c: [gm([x[k] for x in thirds[c]]) for k in range(3)] for c in cands}
    (R3 / 'selection.json').write_text(json.dumps(dict(plan_sha256=psha(), winner=win, score=score, thirds=th, per_task=per), indent=1), 'utf-8')
    for c in cands:
        print('%-8s val %.4f  thirds %s' % (c, score[c], ' '.join('%.4f' % x for x in th[c])))
    print('winner', win)


def score():
    import benchmark_textrepair_v1.score as fz
    sel = json.loads((R3 / 'selection.json').read_text('utf-8'))
    if sel['plan_sha256'] != psha():
        raise SystemExit('plan changed')
    w = sel['winner']; methods = ['iTransformer', 'ADAPT-Z', 'DSOF', 'Proceed', 'On-Adapter', 'GPT4MTS', 'MM-TSFlib']
    rows = {}
    for d, h in r1.tasks():
        for m in methods:
            f = r1.RUNS / d / ('H%d' % h) / m / 'seed2'; res = r1.admitted(f)
            with np.load(f / 'predictions.npz') as z:
                v = fz.cell_metrics(z['predictions'], z['targets'], z['mask'])
            v['update_events'] = res.get('update_events', 0); rows[(d, h, m)] = v
        t, meta = load(w, 'test', d, h)
        v = fz.cell_metrics(t['predictions'], t['targets'], t['mask']); v['update_events'] = meta['updates']; rows[(d, h, 'On-Adapter-T')] = v
    cmp = [m if m != 'On-Adapter' else 'On-Adapter-T' for m in methods]
    S = {m: fz.geometric_mean([rows[(d, h, m)]['cvar90'] / rows[(d, h, 'iTransformer')]['cvar90'] for d, h in r1.tasks()]) for m in cmp + ['On-Adapter']}
    rk = {m: [] for m in cmp}
    for d, h in r1.tasks():
        for pos, m in enumerate(sorted(cmp, key=lambda m: rows[(d, h, m)]['cvar90']), 1):
            rk[m].append(pos)
    summ = {m: dict(S=S[m], mean_rank=float(np.mean(rk[m])) if m in rk else None,
                    mean_mae=float(np.mean([rows[(d, h, m)]['mae'] for d, h in r1.tasks()])),
                    mean_mse=float(np.mean([rows[(d, h, m)]['mse'] for d, h in r1.tasks()])),
                    mean_cvar90=float(np.mean([rows[(d, h, m)]['cvar90'] for d, h in r1.tasks()])),
                    better_than_backbone=sum(rows[(d, h, m)]['cvar90'] < rows[(d, h, 'iTransformer')]['cvar90'] for d, h in r1.tasks()))
            for m in cmp + ['On-Adapter']}
    per_task = [dict(task='%s H%d' % (d, h), **{m: dict(rel_cvar90=rows[(d, h, m)]['cvar90'] / rows[(d, h, 'iTransformer')]['cvar90'],
                     mae=rows[(d, h, m)]['mae'], mse=rows[(d, h, m)]['mse'], cvar90=rows[(d, h, m)]['cvar90'],
                     origins=rows[(d, h, m)]['effective_origins'], updates=rows[(d, h, m)]['update_events']) for m in cmp + ['On-Adapter']})
                for d, h in r1.tasks()]
    achieved = all(S['On-Adapter-T'] < S[m] for m in cmp if m != 'On-Adapter-T')
    (R3 / 'results.json').write_text(json.dumps(dict(plan_sha256=psha(), winner=w, achieved_strict_first=achieved, summary=summ, per_task=per_task), indent=1), 'utf-8')
    for m in sorted(summ, key=lambda m: summ[m]['S']):
        v = summ[m]; print('%-14s S=%.6f rank=%s mae=%.4f mse=%.4f cvar90=%.4f better=%d' % (m, v['S'], '%.3f' % v['mean_rank'] if v['mean_rank'] else '-', v['mean_mae'], v['mean_mse'], v['mean_cvar90'], v['better_than_backbone']))
    print('achieved strict first:', achieved)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--traj', nargs=4); ap.add_argument('--select', action='store_true'); ap.add_argument('--score', action='store_true')
    a = ap.parse_args()
    if a.traj: traj(a.traj[0], a.traj[1], a.traj[2], int(a.traj[3]))
    elif a.select: select()
    elif a.score: score()
