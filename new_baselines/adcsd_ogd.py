"""ADCSD and OGD on the V1 benchmark (results/new_baselines/PLAN_adcsd_ogd.json). --list ; --traj <ADCSD|OGD> <domain> <H> <seed>
Ports of external/adaptz/ADCSD.py and parameter_finetune.py onto the delayed-feedback origin stream, with the same seed's
V1 main-table iTransformer checkpoint as frozen backbone (rebuilt exactly as for SOLID)."""
import argparse, hashlib, json, random, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
import solid as SO
R = SO.R; PLAN = R / 'PLAN_adcsd_ogd.json'; OUT = Path('E:/FedNoise/iadapter-baselines/runs/adcsd_ogd_v1ext_20261011')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class MovingAvg(torch.nn.Module):          # author moving_avg (AvgPool2d over the horizon, edge padding)
    def __init__(self, k):
        super().__init__(); self.k = k; self.avg = torch.nn.AvgPool2d(kernel_size=(1, k), stride=1, padding=0)
    def forward(self, x):
        front = x[:, 0:1].repeat(1, (self.k - 1) // 2, 1, 1); end = x[:, -1:].repeat(1, (self.k - 1) // 2, 1, 1)
        return self.avg(torch.cat([front, x, end], dim=1).permute(0, 3, 2, 1)).permute(0, 3, 2, 1)


class ADCSDModule(torch.nn.Module):        # author ADCSDModule, output_dim 1, hidden 128, moving_avg 5
    def __init__(self, H, N, k=5):
        super().__init__(); self.mavg = MovingAvg(k)
        mk = lambda: torch.nn.Sequential(torch.nn.Linear(1, 128), torch.nn.LayerNorm([H, N, 128]), torch.nn.GELU(), torch.nn.Linear(128, 1))
        self.FWL_1, self.FWL_2 = mk(), mk()
        self.learned_lambda_1 = torch.nn.Parameter(torch.zeros(N, 1)); self.learned_lambda_2 = torch.nn.Parameter(torch.zeros(N, 1))
    def forward(self, x):
        trend = self.mavg(x); res = x - trend
        return x + self.learned_lambda_1 * self.FWL_1(res) + self.learned_lambda_2 * self.FWL_2(trend)


def masked_mse(p, y, m):
    w = m.float(); return ((p - y) ** 2 * w).sum() / w.sum().clamp_min(1.)


def run(method, d, h, s):
    rd, data, L, model, m = SO.setup(d, h, s)
    meta = json.loads((rd / 'config.json').read_text('utf-8')); bs = int(meta['settings']['batch_size'])
    random.seed(100 * s); np.random.seed(100 * s); torch.manual_seed(100 * s)
    model.eval(); model.requires_grad_(False)
    windows = m('data').ForecastWindows(data, 'test', L, h, method='iTransformer')
    X, Y, MK, MR = data.values, data.label_values, data.mask, data.marks
    N = X.shape[1]
    if method == 'ADCSD':
        fwl = ADCSDModule(h, N); opt = torch.optim.Adam(fwl.parameters(), lr=1e-4, eps=1e-8, weight_decay=0, amsgrad=False)
    else:
        head = model.author_model.projection; head.requires_grad_(True); opt = torch.optim.SGD(head.parameters(), lr=3e-6)
    hist = []; rec = {k: [] for k in ['pred', 'base', 'y', 'mask', 'origin']}; n_upd = 0
    for o in windows.indices:
        o = int(o); x = torch.from_numpy(X[o + 1 - L:o + 1])[None]; mr = torch.from_numpy(MR[o + 1 - L:o + 1])[None]
        with torch.no_grad():
            base = model({'x': x, 'mark': mr})[0]
            pred = fwl(base[None, ..., None])[0, ..., 0] if method == 'ADCSD' else base
        y, mk = Y[o + 1:o + 1 + h], MK[o + 1:o + 1 + h]
        for k, v in zip(['pred', 'base', 'y', 'mask', 'origin'], [pred.numpy(), base.numpy(), y, mk, o]): rec[k].append(np.asarray(v))
        hist.append(o)
        ready = [c for c in hist if c + h <= o and MK[c + 1:c + 1 + h].any()][-bs:]
        if len(ready) >= bs:
            bx = torch.from_numpy(np.stack([X[c + 1 - L:c + 1] for c in ready])); br = torch.from_numpy(np.stack([MR[c + 1 - L:c + 1] for c in ready]))
            by = torch.from_numpy(np.stack([Y[c + 1:c + 1 + h] for c in ready])); bm = torch.from_numpy(np.stack([MK[c + 1:c + 1 + h] for c in ready]))
            opt.zero_grad(set_to_none=True)
            if method == 'ADCSD':
                with torch.no_grad(): out = model({'x': bx, 'mark': br})
                loss = masked_mse(fwl(out[..., None])[..., 0], by, bm)
            else:
                loss = masked_mse(model({'x': bx, 'mark': br}), by, bm)
            loss.backward(); opt.step(); n_upd += 1
    return {k: np.stack(v) for k, v in rec.items()}, n_upd, rd


def folder(method, d, h, s): return OUT / method / d / ('H%d' % h) / ('seed%d' % s)


def traj(method, d, h, s):
    out = folder(method, d, h, s); out.mkdir(parents=True, exist_ok=True)
    ident = dict(plan_sha256=sha(PLAN), method=method, code_sha256={'self': sha(__file__), 'solid': sha(SO.__file__)})
    done = out / 'r0.json'
    if done.is_file() and json.loads(done.read_text('utf-8')).get('identity') == ident: return
    t0 = time.time(); a, n_upd, rd = run(method, d, h, s)
    with np.load(rd / 'predictions.npz') as z:
        if not np.array_equal(z['origins'], a['origin']): raise SystemExit('origin grid differs from the main table')
        check = float(np.abs(z['predictions'] - a['base']).max())
    np.savez_compressed(out / 'r0.npz', predictions=a['pred'], backbone=a['base'], targets=a['y'], mask=a['mask'], origins=a['origin'])
    done.write_text(json.dumps(dict(identity=ident, seconds=time.time() - t0, updates=n_upd, max_abs_backbone_vs_main_table=check,
                                    artifacts={'r0.npz': sha(out / 'r0.npz')}), indent=1), 'utf-8')


def load(method, d, h, s):
    f = folder(method, d, h, s); meta = json.loads((f / 'r0.json').read_text('utf-8'))
    if sha(f / 'r0.npz') != meta['artifacts']['r0.npz']: raise SystemExit('artifact changed')
    with np.load(f / 'r0.npz') as z: return {k: z[k] for k in z.files}, meta


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--traj', nargs=4); ap.add_argument('--list', action='store_true')
    a = ap.parse_args()
    if a.traj: traj(a.traj[0], a.traj[1], int(a.traj[2]), int(a.traj[3]))
    elif a.list:
        for meth in ('ADCSD', 'OGD'):
            for d, h in SO.TASKS:
                for s in SO.SEEDS:
                    if not (folder(meth, d, h, s) / 'r0.json').is_file(): print(meth, d, h, s)
