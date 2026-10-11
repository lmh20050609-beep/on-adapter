"""Ablation trajectories for seeds 2025-2027 (results/final/five_seeds/PLAN.json). --list <kind> ; --traj <kind> <domain> <H> <seed>
kinds: orig_lr0 (V1 main-table Original On-Adapter checkpoint, online updates off), cv_lr (confirmation On-Adapter checkpoint,
original online rule Adam lr 3e-4), va / e3 (variants trained with the frozen confirmation package, lr 0).
The w/o-CVaR variant reuses the confirmation On-Adapter-TA trajectories (runs/v1conf_traj_20261009/ta).
"""
import argparse, hashlib, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'confirmation'))
import numpy as np
import harness as hx
import round1_calibration as r1
ROOT = hx.ROOT; R = ROOT / 'results/final/five_seeds'; RUNS = Path('E:/FedNoise/iadapter-baselines/runs')
OUT = RUNS / 'final5_ablation_traj_20261010'; SEEDS = [2025, 2026, 2027]; CODE = [Path(__file__), Path(hx.__file__)]
SRC = {'orig_lr0': ('method_comparison_textrepair_v1_20261008', 0.0), 'cv_lr': ('method_comparison_v1conf_cv_20261009', 3e-4),
       'va': ('method_comparison_v1conf_va_20261010', 0.0), 'e3': ('method_comparison_v1conf_e3_20261010', 0.0)}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ckpt(kind, d, h, s): return RUNS / SRC[kind][0] / d / ('H%d' % h) / 'On-Adapter' / ('seed%d' % s)


def traj(kind, d, h, s):
    rd = ckpt(kind, d, h, s); lr = SRC[kind][1]
    if r1.admitted(rd) is None: raise SystemExit('cell not complete')
    out = OUT / kind / d / ('H%d' % h) / ('seed%d' % s); out.mkdir(parents=True, exist_ok=True)
    ident = dict(plan_sha256=sha(R / 'PLAN.json'), kind=kind, split='test', lr=lr, entry_seed=100 * s, best_pt_sha256=sha(rd / 'best.pt'),
                 code_sha256={p.name: sha(p) for p in CODE})
    done = out / 'r0.json'
    if done.is_file() and json.loads(done.read_text('utf-8')).get('identity') == ident: return
    ctx = hx.setup(rd, 'cpu'); t0 = time.time(); a, ev, _ = hx.run_online(ctx, 'test', lr, measure=False, rng_seed=100 * s)
    np.savez_compressed(out / 'r0.npz', predictions=a['pred'], backbone=a['base'], targets=a['y'], mask=a['mask'], origins=a['origin'])
    (out / 'events_r0.json').write_text(json.dumps(ev), 'utf-8')
    done.write_text(json.dumps(dict(identity=ident, seconds=time.time() - t0, updates=len(ev), artifacts={n: sha(out / n) for n in ['r0.npz', 'events_r0.json']}), indent=1), 'utf-8')


def load(kind, d, h, s):
    f = OUT / kind / d / ('H%d' % h) / ('seed%d' % s); meta = json.loads((f / 'r0.json').read_text('utf-8'))
    if meta['identity']['plan_sha256'] != sha(R / 'PLAN.json'): raise SystemExit('plan changed')
    for n, w in meta['artifacts'].items():
        if sha(f / n) != w: raise SystemExit('artifact changed')
    with np.load(f / 'r0.npz') as z: return {k: z[k] for k in z.files}


def listing(kind):
    for d, h in r1.tasks():
        for s in SEEDS:
            if (ckpt(kind, d, h, s) / 'result.json').is_file() and not (OUT / kind / d / ('H%d' % h) / ('seed%d' % s) / 'r0.json').is_file():
                print('%s %s %d %d' % (kind, d, h, s))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--traj', nargs=4); ap.add_argument('--list')
    a = ap.parse_args()
    if a.traj: traj(a.traj[0], a.traj[1], int(a.traj[2]), int(a.traj[3]))
    elif a.list: listing(a.list)
