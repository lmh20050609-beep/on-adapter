"""Seed-2 round 5 (results/development/round5/PLAN.json): backbone-first abstaining gate over the trained adapter.
  --select (validation) ; --score (one test computation)."""
import argparse, hashlib, json, math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import round1_calibration as r1, round3_training as r3

ROOT = r1.ROOT; R5 = ROOT / 'results/development/round5'
OUT = Path('E:/FedNoise/iadapter-baselines/runs/seed2_round5_ta_20261009')


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def psha(): return sha(R5 / 'PLAN.json')


def abstain(bb, ad, y, m, o, H, gs, mu, kmin=4):
    """emit backbone unless the adapter's discounted matured per-origin MAE beats it by margin mu (hysteresis)."""
    lb, la = r1.per_origin_mae(bb, y, m), r1.per_origin_mae(ad, y, m)
    sb = sa = 0.; n = 0; j = 0; on = False; out = bb.astype(np.float64).copy(); ch = np.zeros(len(o), dtype=np.int64)
    for i in range(len(o)):
        while j < i and o[j] + H <= o[i]:
            sb = gs * sb + lb[j]; sa = gs * sa + la[j]; n += 1; j += 1
        if n >= kmin:
            on = (sa < sb) if on else (sa < (1 - mu) * sb)
        if on:
            out[i] = ad[i]; ch[i] = 1
    return out, ch


def candidates():
    c = [('T', None, None)]
    return c + [('gate', gs, mu) for gs in (0.8, 0.9, 1.0) for mu in (0.0, 0.01, 0.03)]


def forecast(t, h, c):
    ad = t['predictions'].astype(np.float64)
    if c[0] == 'T':
        return ad, np.ones(len(t['origins']), dtype=np.int64)
    return abstain(t['backbone'], ad, t['targets'].astype(np.float64), t['mask'], t['origins'], h, c[1], c[2])


def select():
    cands = candidates(); whole = {c: [] for c in cands}; th = {c: [] for c in cands}
    for d, h in r1.tasks():
        t, _ = r3.load('a0e15', 'val', d, h); eb = r1.per_origin_mae(t['backbone'], t['targets'], t['mask'])
        ix = np.array_split(np.arange(len(t['origins'])), 3)
        for c in cands:
            p, _ = forecast(t, h, c); ep = r1.per_origin_mae(p, t['targets'], t['mask'])
            whole[c].append(r1.cvar90(ep) / r1.cvar90(eb)); th[c].append([r1.cvar90(ep[i]) / r1.cvar90(eb[i]) for i in ix])
    gm = lambda x: math.exp(np.mean(np.log(x)))
    sc = {c: (max(gm(whole[c]), max(gm([x[k] for x in th[c]]) for k in range(3))), gm(whole[c]), [gm([x[k] for x in th[c]]) for k in range(3)]) for c in cands}
    order = sorted(cands, key=lambda c: (sc[c][0], -(c[2] or 0), -(c[1] or 0)))
    w = order[0]
    (R5 / 'selection.json').write_text(json.dumps(dict(plan_sha256=psha(), winner=dict(kind=w[0], gs=w[1], mu=w[2]),
        ranking=[dict(kind=c[0], gs=c[1], mu=c[2], score=sc[c][0], whole=sc[c][1], thirds=sc[c][2]) for c in order]), indent=1), 'utf-8')
    for c in order[:6]:
        print(c, 'score %.4f whole %.4f thirds %s' % (sc[c][0], sc[c][1], ' '.join('%.4f' % x for x in sc[c][2])))


def score():
    import benchmark.score as fz
    sel = json.loads((R5 / 'selection.json').read_text('utf-8'))
    if sel['plan_sha256'] != psha():
        raise SystemExit('plan changed')
    c = (sel['winner']['kind'], sel['winner']['gs'], sel['winner']['mu'])
    methods = ['iTransformer', 'ADAPT-Z', 'DSOF', 'Proceed', 'On-Adapter', 'GPT4MTS', 'MM-TSFlib']; rows = {}; used = {}
    for d, h in r1.tasks():
        for m in methods:
            f = r1.RUNS / d / ('H%d' % h) / m / 'seed2'; res = r1.admitted(f)
            with np.load(f / 'predictions.npz') as z:
                v = fz.cell_metrics(z['predictions'], z['targets'], z['mask'])
            v['update_events'] = res.get('update_events', 0); rows[(d, h, m)] = v
            if m == 'iTransformer':
                bb_campaign = z_pred = None
        t, meta = r3.load('a0e15', 'test', d, h)
        with np.load(r1.RUNS / d / ('H%d' % h) / 'iTransformer' / 'seed2' / 'predictions.npz') as z:
            camp_bb = z['predictions']
        p, ch = forecast(t, h, c)
        v = fz.cell_metrics(p, t['targets'], t['mask']); v['update_events'] = meta['updates']; rows[(d, h, 'On-Adapter-TA')] = v
        used['%s H%d' % (d, h)] = dict(adapter_share=float(ch.mean()),
                                        internal_vs_campaign_backbone_max_abs=float(np.abs(t['backbone'].astype(np.float64) - camp_bb).max()))
        f = OUT / d / ('H%d' % h); f.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(f / 'ta_r0_predictions.npz', predictions=p, targets=t['targets'], mask=t['mask'], origins=t['origins'], choice=ch)
    cmp = [m if m != 'On-Adapter' else 'On-Adapter-TA' for m in methods]
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
    per_task = [dict(task='%s H%d' % (d, h), gate=used['%s H%d' % (d, h)],
                     **{m: dict(rel_cvar90=rows[(d, h, m)]['cvar90'] / rows[(d, h, 'iTransformer')]['cvar90'], mae=rows[(d, h, m)]['mae'],
                                mse=rows[(d, h, m)]['mse'], cvar90=rows[(d, h, m)]['cvar90'], origins=rows[(d, h, m)]['effective_origins'],
                                tail_mass=.1 * rows[(d, h, m)]['effective_origins'], updates=rows[(d, h, m)]['update_events']) for m in cmp + ['On-Adapter']})
                for d, h in r1.tasks()]
    achieved = all(S['On-Adapter-TA'] < S[m] for m in cmp if m != 'On-Adapter-TA')
    (R5 / 'results.json').write_text(json.dumps(dict(plan_sha256=psha(), selection=sel['winner'], achieved_strict_first=achieved, summary=summ, per_task=per_task), indent=1), 'utf-8')
    for m in sorted(summ, key=lambda m: summ[m]['S']):
        v = summ[m]; print('%-14s S=%.6f rank=%s mae=%.4f mse=%.4f cvar90=%.4f better=%d' % (m, v['S'], '%.3f' % v['mean_rank'] if v['mean_rank'] else '-', v['mean_mae'], v['mean_mse'], v['mean_cvar90'], v['better_than_backbone']))
    print('achieved strict first:', achieved, '| max |internal - campaign backbone|:', max(u['internal_vs_campaign_backbone_max_abs'] for u in used.values()))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--select', action='store_true'); ap.add_argument('--score', action='store_true')
    a = ap.parse_args(); select() if a.select else score() if a.score else None
