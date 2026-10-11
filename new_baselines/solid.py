"""SOLID (KDD 2024) on the V1 benchmark (results/new_baselines/PLAN.json).
  --list <split> <lr> ; --traj <split> <lr> <domain> <H> <seed> ; --select
Backbone: the V1 main-table iTransformer checkpoint of the same seed (rebuilt with the package that trained it).
Per origin t: candidates = past origins c with c + H <= t (targets fully observed), lag = t - c in [H, H + N - 1],
lag mod period <= lambda * period; the k nearest input windows (L2); one SGD step on the final linear head with the
masked MSE of the selected samples (train mode, as in the author code); predict in eval mode; restore the head.
"""
import argparse, hashlib, importlib, json, math, random, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
R = ROOT / 'results/new_baselines'; OUT = Path('E:/FedNoise/iadapter-baselines/runs/solid_v1ext_20261011')
PKGS = {'periodic-v3': 'backbones.periodic_v3', 'periodic-v4': 'backbones.periodic_v4'}; SEEDS = [2025, 2026, 2027]; LRS = [1e-4, 1e-3, 1e-2]
N_TRAIN, K, LAMBDA = 500, 5, 0.1
PERIOD = {'Environment': 7., 'Energy': 52.142857, 'Health_US': 52.142857}
DOMAINS = ['Environment', 'Energy', 'Health_US', 'Traffic', 'SocialGood', 'Security', 'Economy', 'Climate', 'Agriculture']
HOR = {'Environment': [48, 96, 192, 336], 'Energy': [12, 24, 36, 48], 'Health_US': [12, 24, 36, 48]}
TASKS = [(d, h) for d in DOMAINS for h in HOR.get(d, [6, 8, 10, 12])]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
MANIFEST = json.loads((ROOT / 'benchmark/legacy_numeric_reuse_manifest.json').read_text('utf-8'))['cells']


def cell_dir(d, h, s): return Path(MANIFEST['%s/H%d/iTransformer/seed%d' % (d, h, s)]['directory'])


def setup(d, h, s):
    rd = cell_dir(d, h, s); meta = json.loads((rd / 'config.json').read_text('utf-8'))
    if meta['execution_version'] not in PKGS or sha(rd / 'best.pt') != MANIFEST['%s/H%d/iTransformer/seed%d' % (d, h, s)]['best_pt_sha256']:
        raise SystemExit('unexpected backbone cell')
    m = lambda n: importlib.import_module(PKGS[meta['execution_version']] + '.' + n)
    torch.set_num_threads(2)
    data = m('data').load_series(d, split=tuple(meta['settings_input']['split']), cache_root=None)
    L = meta['lookback']; cfg = {k: v for k, v in meta['settings'].items() if k != '_settings_input'}; cfg['device'] = 'cpu'
    model, _ = m('native').build('iTransformer', cfg, data, L, h, torch.device('cpu'))
    missing, unexpected = model.load_state_dict(torch.load(rd / 'best.pt', map_location='cpu', weights_only=True), strict=False)
    if unexpected or [k for k in missing if not k.endswith('num_batches_tracked')]: raise SystemExit('checkpoint mismatch %s %s' % (missing[:3], unexpected[:3]))
    return rd, data, L, model, m


def run(split, lr, d, h, s):
    rd, data, L, model, m = setup(d, h, s)
    random.seed(100 * s); np.random.seed(100 * s); torch.manual_seed(100 * s)
    head = model.author_model.projection; model.requires_grad_(False); head.requires_grad_(True)
    windows = m('data').ForecastWindows(data, split, L, h, method='iTransformer')
    X, Y, MK, MR = data.values, data.label_values, data.mask, data.marks
    item = lambda o: (torch.from_numpy(X[o + 1 - L:o + 1]), torch.from_numpy(Y[o + 1:o + 1 + h]), torch.from_numpy(MK[o + 1:o + 1 + h]), torch.from_numpy(MR[o + 1 - L:o + 1]))
    period = PERIOD.get(d, 12.); thr = LAMBDA * period
    rec = {k: [] for k in ['pred', 'base', 'y', 'mask', 'origin']}; n_sel = []
    saved = {k: v.detach().clone() for k, v in head.state_dict().items()}
    for o in windows.indices:
        o = int(o); x, y, mk, mr = item(o)
        cands = [c for c in range(max(L - 1, o - h - N_TRAIN + 1), o - h + 1) if ((o - c) % period) <= thr and MK[c + 1:c + 1 + h].any()]
        model.eval()
        with torch.no_grad(): base = model({'x': x[None], 'mark': mr[None]})[0]
        if cands:
            cx = torch.from_numpy(np.stack([X[c + 1 - L:c + 1] for c in cands]))
            dist = torch.cdist(x.reshape(1, -1), cx.reshape(len(cands), -1))[0]
            sel = [cands[i] for i in dist.topk(min(K, len(cands)), largest=False).indices.tolist()]
            b = [item(c) for c in sel]; bx, by, bm, br = (torch.stack([t[i] for t in b]) for i in range(4))
            model.train(); pred = model({'x': bx, 'mark': br}); w = bm.float()
            loss = ((pred - by) ** 2 * w).sum() / w.sum().clamp_min(1.)
            head.zero_grad(set_to_none=True); loss.backward()
            with torch.no_grad():
                for p in head.parameters(): p -= lr * p.grad
            n_sel.append(len(sel))
        else: n_sel.append(0)
        model.eval()
        with torch.no_grad(): out = model({'x': x[None], 'mark': mr[None]})[0]
        head.load_state_dict(saved)
        for k, v in zip(['pred', 'base', 'y', 'mask', 'origin'], [out, base, y, mk, o]): rec[k].append(np.asarray(v))
    return {k: np.stack(v) for k, v in rec.items()}, float(np.mean(n_sel)), rd


def folder(split, lr, d, h, s): return OUT / split / ('lr%g' % lr) / d / ('H%d' % h) / ('seed%d' % s)


def traj(split, lr, d, h, s):
    out = folder(split, lr, d, h, s); out.mkdir(parents=True, exist_ok=True)
    ident = dict(plan_sha256=sha(R / 'PLAN.json'), split=split, lr=lr, code_sha256=sha(__file__))
    if split == 'test': ident['selection_sha256'] = sha(R / 'solid_selection.json')
    done = out / 'r0.json'
    if done.is_file() and json.loads(done.read_text('utf-8')).get('identity') == ident: return
    t0 = time.time(); a, nsel, rd = run(split, lr, d, h, s); check = None
    if split == 'test':
        with np.load(rd / 'predictions.npz') as z:
            if not np.array_equal(z['origins'], a['origin']): raise SystemExit('origin grid differs from the main table')
            check = float(np.abs(z['predictions'] - a['base']).max())
    np.savez_compressed(out / 'r0.npz', predictions=a['pred'], backbone=a['base'], targets=a['y'], mask=a['mask'], origins=a['origin'])
    done.write_text(json.dumps(dict(identity=ident, seconds=time.time() - t0, mean_selected=nsel, max_abs_backbone_vs_main_table=check,
                                    artifacts={'r0.npz': sha(out / 'r0.npz')}), indent=1), 'utf-8')


def load(split, lr, d, h, s):
    f = folder(split, lr, d, h, s); meta = json.loads((f / 'r0.json').read_text('utf-8'))
    if sha(f / 'r0.npz') != meta['artifacts']['r0.npz']: raise SystemExit('artifact changed')
    with np.load(f / 'r0.npz') as z: return {k: z[k] for k in z.files}, meta


def select():
    import benchmark.score as fz
    sc = {}
    for lr in LRS:
        rat = []
        for d, h in TASKS:
            per = []
            for s in SEEDS:
                a, _ = load('val', lr, d, h, s)
                per.append(fz.cell_metrics(a['predictions'], a['targets'], a['mask'])['cvar90'] / fz.cell_metrics(a['backbone'], a['targets'], a['mask'])['cvar90'])
            rat.append(float(np.mean(per)))
        sc[lr] = math.exp(np.mean(np.log(rat)))
    best = min(LRS, key=lambda lr: (sc[lr], lr))
    (R / 'solid_selection.json').write_text(json.dumps(dict(plan_sha256=sha(R / 'PLAN.json'), validation_score={str(k): v for k, v in sc.items()}, selected_lr=best), indent=1))
    print(sc, 'selected', best)


def listing(split, lr):
    lrs = [json.loads((R / 'solid_selection.json').read_text())['selected_lr']] if split == 'test' else [lr]
    for l in lrs:
        for d, h in TASKS:
            for s in SEEDS:
                if not (folder(split, l, d, h, s) / 'r0.json').is_file(): print(split, l, d, h, s)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--traj', nargs=5); ap.add_argument('--list', nargs=2); ap.add_argument('--select', action='store_true')
    a = ap.parse_args()
    if a.traj: traj(a.traj[0], float(a.traj[1]), a.traj[2], int(a.traj[3]), int(a.traj[4]))
    elif a.list: listing(a.list[0], float(a.list[1]))
    elif a.select: select()
