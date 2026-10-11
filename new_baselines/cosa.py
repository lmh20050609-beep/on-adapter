"""COSA (ICLR 2026) on the V1 benchmark (results/new_baselines/PLAN_cosa.json). --list ; --traj <domain> <H> <seed>
Port of external/cosa/tta/cosa.py (SimpleOutputAdapter + SimpleAdapter.adapt_simple) with the authors' script settings,
forecasts issued at every origin (batch size 1), frozen V1 main-table iTransformer backbone."""
import argparse, hashlib, json, random, sys, time
from collections import deque
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
import torch.nn as nn
import solid as SO
R = SO.R; PLAN = R / 'PLAN_cosa.json'; OUT = Path('E:/FedNoise/iadapter-baselines/runs/cosa_v1ext_20261011')
BASE_LR, WD, CTX, STEPS, MAX_LR, MIN_LR, L2 = 1e-3, 1e-4, 10, 3, 5e-3, 1e-4, 1e-4
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class SimpleOutputAdapter(nn.Module):        # author class, Linear + var-wise gating
    def __init__(self, H, ctx, N):
        super().__init__()
        self.fc_layers = nn.ModuleList([nn.Linear(H + ctx, H) for _ in range(N)]); self.gate = nn.Parameter(torch.zeros(N))
        for m in self.modules():
            if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight, gain=0.1); nn.init.zeros_(m.bias)
    def forward(self, y, c):                  # y (B,H,N), c (B,ctx)
        corr = torch.cat([self.fc_layers[v](torch.cat([y[:, :, v], c], dim=-1)).unsqueeze(-1) for v in range(y.shape[-1])], dim=-1)
        return y + torch.tanh(self.gate)[None, None] * corr


class LR:                                     # author _adaptive_learning_rate (FAST_ADAPTATION, ADAPTIVE_LR, PER_BATCH_LR_RESET)
    def __init__(self): self.hist = deque(maxlen=5); self.lr = BASE_LR
    def __call__(self, loss, step):
        if step == 0:
            self.lr = BASE_LR; self.hist.append(loss); return self.lr
        self.hist.append(loss); r = list(self.hist)[-3:]
        if len(r) < 2: return self.lr
        trend = r[-1] - r[0]; var = torch.tensor(r).var().item()
        if trend > 0 and var < 1e-6: self.lr = min(self.lr * 1.2, MAX_LR)
        elif trend < -0.01: self.lr = min(self.lr * 1.05, MAX_LR)
        elif abs(trend) < 1e-6: self.lr = max(self.lr * 0.8, MIN_LR)
        if step >= 1: self.lr = MIN_LR + (self.lr - MIN_LR) * 0.5 * (1 + torch.cos(torch.tensor(step * 3.14159 / STEPS)).item())
        return self.lr


def run(d, h, s):
    rd, data, L, model, m = SO.setup(d, h, s)
    random.seed(100 * s); np.random.seed(100 * s); torch.manual_seed(100 * s)
    model.eval(); model.requires_grad_(False)
    windows = m('data').ForecastWindows(data, 'test', L, h, method='iTransformer')
    X, Y, MK, MR = data.values, data.label_values, data.mask, data.marks; N = X.shape[1]
    ad = SimpleOutputAdapter(h, CTX, N); opt = torch.optim.Adam(ad.parameters(), lr=BASE_LR, weight_decay=WD); sched = LR()
    history, pending = deque(maxlen=200), deque()
    rec = {k: [] for k in ['pred', 'base', 'y', 'mask', 'origin']}; n_upd = 0
    def context():
        if not history: return torch.zeros(CTX)
        vals = [history[-(i + 1)] for i in range(min(CTX, len(history)))]
        vals += [vals[-1]] * (CTX - len(vals)); return torch.tensor(vals, dtype=torch.float32)
    for o in windows.indices:
        o = int(o); x = torch.from_numpy(X[o + 1 - L:o + 1])[None]; mr = torch.from_numpy(MR[o + 1 - L:o + 1])[None]
        with torch.no_grad(): base = model({'x': x, 'mark': mr})
        while pending and pending[0][0] + h <= o:          # _drain_observed_batches: push history, POGT full replay
            oc, pb, ctx_c = pending.popleft()
            y_c = torch.from_numpy(Y[oc + 1:oc + 1 + h])[None]; m_c = torch.from_numpy(MK[oc + 1:oc + 1 + h])[None].float()
            history.append(float((y_c * m_c).sum() / m_c.sum().clamp_min(1.)))
            for step in range(min(STEPS, 5)):
                ad.train(); out = ad(pb, ctx_c[None])
                loss = ((out - y_c) ** 2 * m_c).sum() / m_c.sum().clamp_min(1.) + L2 * sum(p.pow(2).sum() for p in ad.parameters())
                for g in opt.param_groups: g['lr'] = sched(loss.item(), step)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(ad.parameters(), max_norm=max(0.05, min(0.5, loss.item()))); opt.step(); ad.eval()
                if step > 2 and len(sched.hist) >= 2 and abs(sched.hist[-1] - sched.hist[-2]) < 1e-4: break
            n_upd += 1
        c = context()
        with torch.no_grad(): pred = ad(base, c[None])[0]
        pending.append((o, base.detach(), c))
        for k, v in zip(['pred', 'base', 'y', 'mask', 'origin'], [pred.numpy(), base[0].numpy(), Y[o + 1:o + 1 + h], MK[o + 1:o + 1 + h], o]): rec[k].append(np.asarray(v))
    return {k: np.stack(v) for k, v in rec.items()}, n_upd, rd


def folder(d, h, s): return OUT / d / ('H%d' % h) / ('seed%d' % s)


def traj(d, h, s):
    out = folder(d, h, s); out.mkdir(parents=True, exist_ok=True)
    ident = dict(plan_sha256=sha(PLAN), code_sha256={'self': sha(__file__), 'solid': sha(SO.__file__)})
    done = out / 'r0.json'
    if done.is_file() and json.loads(done.read_text('utf-8')).get('identity') == ident: return
    t0 = time.time(); a, n_upd, rd = run(d, h, s)
    with np.load(rd / 'predictions.npz') as z:
        if not np.array_equal(z['origins'], a['origin']): raise SystemExit('origin grid differs from the main table')
        check = float(np.abs(z['predictions'] - a['base']).max())
    np.savez_compressed(out / 'r0.npz', predictions=a['pred'], backbone=a['base'], targets=a['y'], mask=a['mask'], origins=a['origin'])
    done.write_text(json.dumps(dict(identity=ident, seconds=time.time() - t0, updates=n_upd, max_abs_backbone_vs_main_table=check,
                                    artifacts={'r0.npz': sha(out / 'r0.npz')}), indent=1), 'utf-8')


def load(d, h, s):
    f = folder(d, h, s); meta = json.loads((f / 'r0.json').read_text('utf-8'))
    if sha(f / 'r0.npz') != meta['artifacts']['r0.npz']: raise SystemExit('artifact changed')
    with np.load(f / 'r0.npz') as z: return {k: z[k] for k in z.files}, meta


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--traj', nargs=3); ap.add_argument('--list', action='store_true')
    a = ap.parse_args()
    if a.traj: traj(a.traj[0], int(a.traj[1]), int(a.traj[2]))
    elif a.list:
        for d, h in SO.TASKS:
            for s in SO.SEEDS:
                if not (folder(d, h, s) / 'r0.json').is_file(): print(d, h, s)
