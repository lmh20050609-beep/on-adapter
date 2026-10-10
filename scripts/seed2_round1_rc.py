"""Seed-2 round 1 (reports/seed2_goal_20261009/round1/PLAN.json): On-Adapter-RC.

  validation trajectories : --val-cell  E:/.../method_comparison_seed2_v1_20261009/<d>/H<h>/On-Adapter/seed2
  global selection        : --select
  test trajectories       : --test-cell E:/.../<d>/H<h>/On-Adapter/seed2
  score                   : --score
  ready cells (dispatch)  : --list-ready val|test
"""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import onadapter_harness_20261008 as hx

ROOT = hx.ROOT
sys.path.insert(0, str(ROOT))
R1 = ROOT / 'reports/seed2_goal_20261009/round1'
RUNS = Path('E:/FedNoise/iadapter-baselines/runs/method_comparison_seed2_v1_20261009')
OUT = Path('E:/FedNoise/iadapter-baselines/runs/seed2_round1_rc_20261009')
DOMAINS = ['Environment', 'Energy', 'Health_US', 'Traffic', 'SocialGood', 'Security', 'Economy', 'Climate', 'Agriculture']
HOR = {'Environment': [48, 96, 192, 336], 'Energy': [12, 24, 36, 48], 'Health_US': [12, 24, 36, 48]}
BASES = {'lr0': 0.0, 'lr3e-4': 3e-4}
STREAMS = [0, 1, 2]
SEED = 2
CODE = [Path(__file__), Path(hx.__file__)]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def plan():
    return json.loads((R1 / 'PLAN.json').read_text('utf-8')), sha(R1 / 'PLAN.json')


def tasks():
    return [(d, h) for d in DOMAINS for h in HOR.get(d, [6, 8, 10, 12])]


def per_origin_mae(pred, y, m):
    e = np.abs(pred.astype(np.float64) - y)
    return np.array([x[k].mean() for x, k in zip(e, m.astype(bool))])


def cvar90(v):
    v = np.sort(np.asarray(v, dtype=np.float64))[::-1]; mass = .1 * len(v); w = int(math.floor(mass))
    return float((v[:w].sum() + ((mass - w) * v[w] if mass - w else 0.)) / mass)


def calibrate(base, y, mask, origins, horizon, a, gamma):
    """forecast_t = base_t + a * b_t; b_t from fully matured residuals only (o + H <= t)."""
    base = base.astype(np.float64); out = base.copy()
    if a == 0:
        return out
    m = mask.astype(bool); s = np.zeros(base.shape[1:]); w = np.zeros(base.shape[1:]); j = 0
    for i in range(len(origins)):
        while j < i and origins[j] + horizon <= origins[i]:
            s = gamma * s + np.where(m[j], y[j] - base[j], 0.); w = gamma * w + m[j]; j += 1
        out[i] = base[i] + a * np.where(w > 0, s / np.maximum(w, 1e-300), 0.)
    return out


def admitted(run_dir):
    r = Path(run_dir) / 'result.json'
    if not r.is_file():
        return None
    res = json.loads(r.read_text('utf-8'))
    for n, want in res['artifacts'].items():
        if sha(Path(run_dir) / n) != want:
            raise SystemExit('campaign artifact changed: %s' % n)
    return res


def run_split(run_dir, split, bases):
    spec, psha = plan()
    run_dir = Path(run_dir); res = admitted(run_dir)
    if res is None:
        raise SystemExit('not admitted: %s' % run_dir)
    d, h = res['domain'], res['horizon']
    out = OUT / split / d / ('H%d' % h); out.mkdir(parents=True, exist_ok=True)
    ctx = None
    for b in bases:
        for r in STREAMS:
            tag = '%s_r%d' % (b, r)
            ident = dict(plan_sha256=psha, split=split, base=b, online_learning_rate=BASES[b], stream=r, entry_seed=100 * SEED + r,
                         best_pt_sha256=sha(run_dir / 'best.pt'), config_sha256=sha(run_dir / 'config.json'),
                         code_sha256={p.name: sha(p) for p in CODE})
            if split == 'test':
                ident['selection_sha256'] = sha(R1 / 'selection.json')
            done = out / (tag + '.json')
            if done.is_file() and json.loads(done.read_text('utf-8')).get('identity') == ident:
                continue
            if ctx is None:
                ctx = hx.setup(run_dir, 'cpu')
                if split == 'val' and int(ctx['sets']['val'].indices.max()) + h >= ctx['data'].val_end:
                    raise SystemExit('validation labels cross val_end')
            t0 = time.time()
            arrays, events, _ = hx.run_online(ctx, split, BASES[b], measure=True, rng_seed=ident['entry_seed'])
            np.savez_compressed(out / (tag + '.npz'), predictions=arrays['pred'], backbone=arrays['base'], targets=arrays['y'],
                                mask=arrays['mask'], origins=arrays['origin'], hg_norm=arrays['hg_norm'], ge_norm=arrays['ge_norm'],
                                correction=arrays['correction'], entry_torch_rng=arrays['entry_torch_rng'], final_params=arrays['final_params'])
            (out / ('events_' + tag + '.json')).write_text(json.dumps(events), 'utf-8')
            done.write_text(json.dumps(dict(identity=ident, finite=bool(np.isfinite(arrays['pred']).all()), seconds=time.time() - t0,
                                            origins=int(len(arrays['origin'])), updates=len(events),
                                            artifacts={n: sha(out / n) for n in [tag + '.npz', 'events_' + tag + '.json']}), indent=1), 'utf-8')
            print(json.dumps(dict(cell='%s H%d' % (d, h), split=split, traj=tag, sec=round(time.time() - t0, 1))), flush=True)
    (out / ('done_%s.json' % '_'.join(bases))).write_text(json.dumps(dict(plan_sha256=psha)), 'utf-8')


def load(split, d, h, b, r):
    folder = OUT / split / d / ('H%d' % h)
    meta = json.loads((folder / ('%s_r%d.json' % (b, r))).read_text('utf-8'))
    for n, want in meta['artifacts'].items():
        if sha(folder / n) != want:
            raise SystemExit('trajectory artifact changed: %s' % (folder / n))
    with np.load(folder / ('%s_r%d.npz' % (b, r))) as z:
        return {k: z[k] for k in z.files}, meta


def candidates(spec):
    out = []
    for b in BASES:
        for a in spec['candidates']['a']:
            for g in (spec['candidates']['gamma'] if a else [1.0]):
                out.append((b, a, g))
    return out


def select():
    spec, psha = plan()
    cands = candidates(spec)
    per_task = {c: {} for c in cands}; companions = {c: {} for c in cands}; ineligible = set()
    for d, h in tasks():
        vals = {c: [] for c in cands}; comp = {c: [] for c in cands}
        for r in STREAMS:
            traj = {b: load('val', d, h, b, r)[0] for b in BASES}
            bb = traj['lr0']['backbone']; y = traj['lr0']['targets']; m = traj['lr0']['mask']; o = traj['lr0']['origins']
            assert np.array_equal(bb, traj['lr3e-4']['backbone'])
            ref = cvar90(per_origin_mae(bb, y, m))
            for c in cands:
                p = calibrate(traj[c[0]]['predictions'], y, m, o, h, c[1], c[2])
                if not np.isfinite(p).all():
                    ineligible.add(c); continue
                vals[c].append(cvar90(per_origin_mae(p, y, m)) / ref)
                comp[c].append(dict(mse=float(((p - y) ** 2)[m.astype(bool)].mean()), mae=float(np.abs(p - y)[m.astype(bool)].mean())))
        for c in cands:
            if len(vals[c]) == len(STREAMS):
                per_task[c]['%s H%d' % (d, h)] = float(np.mean(vals[c]))
                companions[c]['%s H%d' % (d, h)] = {k: float(np.mean([x[k] for x in comp[c]])) for k in ('mse', 'mae')}
    score = {c: math.exp(np.mean([math.log(x) for x in per_task[c].values()])) for c in cands
             if c not in ineligible and len(per_task[c]) == 36}
    order = sorted(score, key=lambda c: (score[c], c[1], -c[2], list(BASES).index(c[0])))
    win = order[0]
    payload = dict(plan_sha256=psha, winner=dict(base=win[0], a=win[1], gamma=win[2]), winner_score=score[win],
                   ranking=[dict(base=c[0], a=c[1], gamma=c[2], val_gm_cvar90_ratio=score[c]) for c in order],
                   ineligible=[list(c) for c in ineligible], per_task={'%s|%s|%s' % c: v for c, v in per_task.items()},
                   companions={'%s|%s|%s' % c: v for c, v in companions.items()})
    (R1 / 'selection.json').write_text(json.dumps(payload, indent=1), 'utf-8')
    for c in order[:8]:
        print('%-7s a=%.2f g=%.2f  val GM CVaR90 ratio %.4f' % (c[0], c[1], c[2], score[c]))


def score():
    import benchmark_textrepair_v1.score as fz
    spec, psha = plan()
    sel = json.loads((R1 / 'selection.json').read_text('utf-8')); w = sel['winner']
    methods = ['iTransformer', 'ADAPT-Z', 'DSOF', 'Proceed', 'On-Adapter', 'GPT4MTS', 'MM-TSFlib']
    rows, missing = {}, []
    for d, h in tasks():
        for m in methods:
            folder = RUNS / d / ('H%d' % h) / m / ('seed%d' % SEED)
            res = admitted(folder)
            if res is None:
                missing.append('%s/H%d/%s' % (d, h, m)); continue
            with np.load(folder / 'predictions.npz') as z:
                v = fz.cell_metrics(z['predictions'], z['targets'], z['mask'])
            v['update_events'] = res.get('update_events', 0); rows[(d, h, m)] = v
        for r in STREAMS:
            try:
                traj, meta = load('test', d, h, w['base'], r)
            except FileNotFoundError:
                missing.append('%s/H%d/RC/r%d' % (d, h, r)); continue
            p = calibrate(traj['predictions'], traj['targets'], traj['mask'], traj['origins'], h, w['a'], w['gamma'])
            v = fz.cell_metrics(p, traj['targets'], traj['mask']); v['update_events'] = meta['updates']
            rows[(d, h, 'On-Adapter-RC/r%d' % r)] = v
            np.savez_compressed(OUT / 'test' / d / ('H%d' % h) / ('rc_r%d_predictions.npz' % r), predictions=p,
                                targets=traj['targets'], mask=traj['mask'], origins=traj['origins'])
    if missing:
        raise SystemExit('missing: %d e.g. %s' % (len(missing), missing[:6]))
    report = {}
    for variant in ['On-Adapter-RC/r0', 'On-Adapter']:
        cmp_methods = [m if m != 'On-Adapter' else variant for m in methods]
        S, rk = {}, {m: [] for m in cmp_methods}
        for m in cmp_methods + (['On-Adapter'] if variant != 'On-Adapter' else []):
            S[m] = fz.geometric_mean([rows[(d, h, m)]['cvar90'] / rows[(d, h, 'iTransformer')]['cvar90'] for d, h in tasks()])
        for d, h in tasks():
            for pos, m in enumerate(sorted(cmp_methods, key=lambda m: rows[(d, h, m)]['cvar90']), 1):
                rk[m].append(pos)
        report[variant] = dict(S=S, mean_rank={m: float(np.mean(v)) for m, v in rk.items()})
    S = report['On-Adapter-RC/r0']['S']; others = {m: s for m, s in S.items() if m not in ('On-Adapter-RC/r0', 'On-Adapter')}
    achieved = all(S['On-Adapter-RC/r0'] < s for s in others.values())
    per_task = []
    for d, h in tasks():
        e = dict(task='%s H%d' % (d, h))
        for m in methods + ['On-Adapter-RC/r0', 'On-Adapter-RC/r1', 'On-Adapter-RC/r2']:
            v = rows[(d, h, m)]
            e[m] = dict(rel_cvar90=v['cvar90'] / rows[(d, h, 'iTransformer')]['cvar90'], mae=v['mae'], mse=v['mse'], cvar90=v['cvar90'],
                        origins=v['effective_origins'], tail_mass=.1 * v['effective_origins'], updates=v['update_events'])
        per_task.append(e)
    streams = {('r%d' % r): fz.geometric_mean([rows[(d, h, 'On-Adapter-RC/r%d' % r)]['cvar90'] / rows[(d, h, 'iTransformer')]['cvar90']
                                               for d, h in tasks()]) for r in STREAMS}
    payload = dict(plan_sha256=psha, selection=w, achieved_strict_first=achieved, main=report, rc_streams=streams, per_task=per_task)
    (R1 / 'results.json').write_text(json.dumps(payload, indent=1), 'utf-8')
    for m, s in sorted(S.items(), key=lambda kv: kv[1]):
        print('%-18s S=%.6f mean_rank=%s' % (m, s, '%.3f' % report['On-Adapter-RC/r0']['mean_rank'][m] if m in report['On-Adapter-RC/r0']['mean_rank'] else '-'))
    print('achieved strict first:', achieved, '| RC streams', streams)


def list_ready(split):
    for d, h in tasks():
        rd = RUNS / d / ('H%d' % h) / 'On-Adapter' / ('seed%d' % SEED)
        if split == 'val':
            done = OUT / 'val' / d / ('H%d' % h) / 'done_lr0_lr3e-4.json'
        else:
            if not (R1 / 'selection.json').is_file():
                return
            b = json.loads((R1 / 'selection.json').read_text('utf-8'))['winner']['base']
            done = OUT / 'test' / d / ('H%d' % h) / ('done_%s.json' % b)
        if (rd / 'result.json').is_file() and not done.is_file():
            print(str(rd).replace('\\', '/'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--val-cell'); ap.add_argument('--test-cell'); ap.add_argument('--select', action='store_true')
    ap.add_argument('--score', action='store_true'); ap.add_argument('--list-ready')
    a = ap.parse_args()
    if a.val_cell:
        run_split(a.val_cell, 'val', list(BASES))
    elif a.test_cell:
        sel = json.loads((R1 / 'selection.json').read_text('utf-8'))
        run_split(a.test_cell, 'test', [sel['winner']['base']])
    elif a.select:
        select()
    elif a.score:
        score()
    elif a.list_ready:
        list_ready(a.list_ready)


if __name__ == '__main__':
    main()
