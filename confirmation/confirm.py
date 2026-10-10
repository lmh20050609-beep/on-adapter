"""V1 three-seed confirmation (results/confirmation/PLAN.json). --traj <tag> <d> <H> <seed> ; --score ; --list <tag>"""
import argparse, hashlib, json, math, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import harness as hx
import round1_calibration as r1, round5_gate as r5
ROOT = hx.ROOT; sys.path.insert(0, str(ROOT)); R = ROOT / 'results/confirmation'
OUT = Path('E:/FedNoise/iadapter-baselines/runs/v1conf_traj_20261009'); SEEDS = [2025, 2026, 2027]; CODE = [Path(__file__), Path(hx.__file__)]
NAME = {'cv': 'On-Adapter-CV', 'ta': 'On-Adapter-TA'}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ckpt(tag, d, h, s): return Path('E:/FedNoise/iadapter-baselines/runs/method_comparison_v1conf_%s_20261009' % tag) / d / ('H%d' % h) / 'On-Adapter' / ('seed%d' % s)


def traj(tag, d, h, s):
    rd = ckpt(tag, d, h, s)
    if r1.admitted(rd) is None: raise SystemExit('cell not complete')
    out = OUT / tag / d / ('H%d' % h) / ('seed%d' % s); out.mkdir(parents=True, exist_ok=True)
    ident = dict(plan_sha256=sha(R / 'PLAN.json'), tag=tag, split='test', lr=0.0, entry_seed=100 * s, best_pt_sha256=sha(rd / 'best.pt'), code_sha256={p.name: sha(p) for p in CODE})
    done = out / 'r0.json'
    if done.is_file() and json.loads(done.read_text('utf-8')).get('identity') == ident: return
    ctx = hx.setup(rd, 'cpu'); t0 = time.time(); a, ev, _ = hx.run_online(ctx, 'test', 0.0, measure=False, rng_seed=100 * s)
    np.savez_compressed(out / 'r0.npz', predictions=a['pred'], backbone=a['base'], targets=a['y'], mask=a['mask'], origins=a['origin'], entry_torch_rng=a['entry_torch_rng'])
    (out / 'events_r0.json').write_text(json.dumps(ev), 'utf-8')
    done.write_text(json.dumps(dict(identity=ident, seconds=time.time() - t0, updates=len(ev), artifacts={n: sha(out / n) for n in ['r0.npz', 'events_r0.json']}), indent=1), 'utf-8')


def load(tag, d, h, s):
    f = OUT / tag / d / ('H%d' % h) / ('seed%d' % s); meta = json.loads((f / 'r0.json').read_text('utf-8'))
    if meta['identity']['plan_sha256'] != sha(R / 'PLAN.json'): raise SystemExit('plan changed')
    for n, w in meta['artifacts'].items():
        if sha(f / n) != w: raise SystemExit('artifact changed')
    with np.load(f / 'r0.npz') as z: return {k: z[k] for k in z.files}


def score():
    import benchmark.score as fz
    manifest = json.loads((ROOT / 'benchmark/legacy_numeric_reuse_manifest.json').read_text('utf-8'))
    tags = [t for t in ('cv', 'ta') if all((OUT / t / d / ('H%d' % h) / ('seed%d' % s) / 'r0.json').is_file() for d, h in r1.tasks() for s in SEEDS)]
    if 'cv' not in tags: raise SystemExit('primary not complete')
    rows, share, bbdiff = {}, {}, []
    for d, h in r1.tasks():
        for s in SEEDS:
            for m in fz.METHODS: rows[(d, h, s, m)] = fz.load_cell(m, d, h, s, manifest)
            for t in tags:
                x = load(t, d, h, s); y, mk, o, bb = x['targets'], x['mask'], x['origins'], x['backbone']; T = x['predictions'].astype(np.float64)
                p, ch = r5.abstain(bb, T, y.astype(np.float64), mk, o, h, .9, 0.)
                rows[(d, h, s, NAME[t])] = fz.cell_metrics(p, y, mk); rows[(d, h, s, NAME[t] + '(pre-gate)')] = fz.cell_metrics(T, y, mk)
                share[(t, d, h, s)] = float(np.mean(ch))
                with np.load(Path(rows[(d, h, s, 'iTransformer')]['directory']) / 'predictions.npz') as z: bbdiff.append(float(np.abs(bb.astype(np.float64) - z['predictions']).max()))
                f = OUT / 'final' / t / d / ('H%d' % h) / ('seed%d' % s); f.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(f / 'predictions.npz', predictions=p, adapter_predictions=T, backbone=bb, targets=y, mask=mk, origins=o, choice=ch)
    def S(m): return fz.geometric_mean([float(np.mean([rows[(d, h, s, m)]['cvar90'] / rows[(d, h, s, 'iTransformer')]['cvar90'] for s in SEEDS])) for d, h in r1.tasks()])
    out = {}
    for t in tags:
        main = NAME[t]; cmp = [m if m != 'On-Adapter' else main for m in fz.METHODS]
        rk = {m: [] for m in cmp}
        for d, h in r1.tasks():
            for s in SEEDS:
                for pos, m in enumerate(sorted(cmp, key=lambda m: rows[(d, h, s, m)]['cvar90']), 1): rk[m].append(pos)
        allm = cmp + ['On-Adapter', main + '(pre-gate)']
        summ = {m: dict(S=S(m), mean_rank=float(np.mean(rk[m])) if m in rk else None,
                        mean_mae=float(np.mean([rows[(d, h, s, m)]['mae'] for d, h in r1.tasks() for s in SEEDS])),
                        mean_mse=float(np.mean([rows[(d, h, s, m)]['mse'] for d, h in r1.tasks() for s in SEEDS])),
                        mean_cvar90=float(np.mean([rows[(d, h, s, m)]['cvar90'] for d, h in r1.tasks() for s in SEEDS])),
                        tasks_better_than_backbone=int(sum(np.mean([rows[(d, h, s, m)]['cvar90'] / rows[(d, h, s, 'iTransformer')]['cvar90'] for s in SEEDS]) < 1 for d, h in r1.tasks())))
                for m in allm}
        per_seed = {s: fz.geometric_mean([rows[(d, h, s, main)]['cvar90'] / rows[(d, h, s, 'iTransformer')]['cvar90'] for d, h in r1.tasks()]) for s in SEEDS}
        per_task = {'%s H%d' % (d, h): dict(rel=float(np.mean([rows[(d, h, s, main)]['cvar90'] / rows[(d, h, s, 'iTransformer')]['cvar90'] for s in SEEDS])),
                    pre=float(np.mean([rows[(d, h, s, main + '(pre-gate)')]['cvar90'] / rows[(d, h, s, 'iTransformer')]['cvar90'] for s in SEEDS])),
                    adapter_share=float(np.mean([share[(t, d, h, s)] for s in SEEDS]))) for d, h in r1.tasks()}
        out[main] = dict(achieved_strict_first=all(summ[main]['S'] < summ[m]['S'] for m in cmp if m != main), summary=summ, per_seed_S=per_seed, per_task=per_task)
        print('== %s' % main)
        for m in sorted(summ, key=lambda m: summ[m]['S']):
            v = summ[m]; print('%-26s S=%.6f rank=%s mae=%.4f mse=%.4f cvar90=%.4f better=%d' % (m, v['S'], '%.3f' % v['mean_rank'] if v['mean_rank'] else '-', v['mean_mae'], v['mean_mse'], v['mean_cvar90'], v['tasks_better_than_backbone']))
        print('per-seed S:', {k: round(v, 5) for k, v in per_seed.items()}, '| achieved strict first:', out[main]['achieved_strict_first'])
    out['max_internal_vs_table_backbone_abs'] = max(bbdiff)
    (R / 'results.json').write_text(json.dumps(dict(plan_sha256=sha(R / 'PLAN.json'), **out), indent=1), 'utf-8')
    print('max |internal backbone - table iTransformer|: %.5f' % max(bbdiff))


def listing(tag):
    for d, h in r1.tasks():
        for s in SEEDS:
            if (ckpt(tag, d, h, s) / 'result.json').is_file() and not (OUT / tag / d / ('H%d' % h) / ('seed%d' % s) / 'r0.json').is_file(): print('%s %s %d %d' % (tag, d, h, s))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--traj', nargs=4); ap.add_argument('--score', action='store_true'); ap.add_argument('--list')
    a = ap.parse_args()
    if a.traj: traj(a.traj[0], a.traj[1], int(a.traj[2]), int(a.traj[3]))
    elif a.score: score()
    elif a.list: listing(a.list)
