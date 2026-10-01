"""Small forecast mixers consuming only original predictions and mature labels."""
from collections import deque
import copy
import time
import numpy as np

EPS = 1e-8
FLOOR = 1e-6
LS_GRID = [(1., 1.), (1., 10.), (.98, 1.), (.98, 10.)]
ETA_GRID = [.01, .1, 1., 10.]
MIX_GRID = np.asarray([0., .25, .5, .75, 1.])


def check_ledger(ledger):
    o = np.asarray(ledger['origins'])
    n, m, y = [np.asarray(ledger[k]) for k in ['numeric', 'text', 'targets']]
    mask = np.asarray(ledger['valid_mask'])
    available = np.asarray(ledger['available'])
    assert len(o) and o.ndim == 1 and np.issubdtype(o.dtype, np.integer)
    assert np.all(np.diff(o) > 0) and n.shape == m.shape == y.shape == mask.shape
    assert n.ndim == 3 and n.shape[0] == len(o) and n.shape[1] >= 1
    assert mask.dtype == bool and available.dtype == bool and available.shape == (len(o),)
    assert np.isfinite(n).all() and np.isfinite(m).all() and np.isfinite(y[mask]).all()
    return n.shape[1]


def statistics(n, m, y, mask, shared=False):
    valid = np.asarray(mask, bool)
    # Index valid cells before arithmetic, so missing placeholders never enter statistics.
    r = np.where(valid, y, 0.) - np.where(valid, n, 0.)
    d = np.where(valid, m, 0.) - np.where(valid, n, 0.)
    A = (r*d).sum(axis=-1)
    B = (d*d).sum(axis=-1)
    count = valid.sum(axis=-1)
    if shared:
        A, B, count = [np.asarray([v.sum()]) for v in [A, B, count]]
    return A, B, count


def moment_prior(calibration, cutoff, shared=False):
    H = check_ledger(calibration)
    size = 1 if shared else H
    A, B, count = np.zeros(size), np.zeros(size), np.zeros(size, dtype=np.int64)
    consumed = []
    for i, origin in enumerate(calibration['origins']):
        assert int(origin)+H <= cutoff, 'Calibration target exceeds allowed cutoff'
        consumed.append(int(origin))
        if not calibration['available'][i]:
            continue
        a, b, k = statistics(calibration['numeric'][i], calibration['text'][i],
                             calibration['targets'][i], calibration['valid_mask'][i], shared)
        A += a; B += b; count += k
    alpha0 = np.full(size, .5)
    np.divide(A, B, out=alpha0, where=B>EPS)
    alpha0 = np.clip(alpha0, 0., 1.)
    mean_d2 = np.divide(B, count, out=np.zeros(size), where=count>0)
    return dict(H=H, shared=shared, alpha0=alpha0, mean_d2=mean_d2,
                count=count, calibration_origins=consumed, cutoff=int(cutoff))


class FixedGate:
    def __init__(self, alpha):
        self.value = np.asarray(alpha, dtype=np.float64).copy()
    def alpha(self):
        return self.value.copy()
    def advance(self, t):
        pass
    def observe(self, n, m, y, mask):
        pass


class MomentGate:
    def __init__(self, prior, rho=1., kappa=1.):
        assert 0 < rho <= 1 and kappa > 0
        self.H, self.shared = prior['H'], prior['shared']
        self.alpha0 = np.asarray(prior['alpha0'], dtype=np.float64).copy()
        self.lam = kappa*np.asarray(prior['mean_d2'], dtype=np.float64) + EPS
        self.A, self.B = np.zeros_like(self.alpha0), np.zeros_like(self.alpha0)
        self.rho, self.last = rho, None
    def alpha(self):
        value = np.clip((self.A+self.lam*self.alpha0)/(self.B+self.lam), 0., 1.)
        return np.full(self.H, value[0]) if self.shared else value.copy()
    def advance(self, t):
        if self.last is not None:
            assert t > self.last, 'Decay clock repeated or reversed'
            decay = self.rho**(t-self.last)
            self.A *= decay; self.B *= decay
        self.last = int(t)
    def observe(self, n, m, y, mask):
        a, b, _ = statistics(n, m, y, mask, self.shared)
        self.A += a; self.B += b


class HedgeGate:
    def __init__(self, H, eta, candidates=None, weights=None):
        assert eta > 0
        self.H, self.eta = H, eta
        self.candidates = np.asarray([0., 1.] if candidates is None else candidates, dtype=np.float64)
        assert np.all((self.candidates >= 0) & (self.candidates <= 1))
        shape = (H, len(self.candidates))
        self.weights = np.ones(shape)/shape[1] if weights is None else np.asarray(weights, dtype=np.float64).copy()
        assert self.weights.shape == shape and np.isfinite(self.weights).all()
        self._floor()
    def _floor(self):
        self.weights = np.maximum(self.weights, FLOOR)
        self.weights /= self.weights.sum(axis=1, keepdims=True)
    def alpha(self):
        return self.weights @ self.candidates
    def advance(self, t):
        pass
    def observe(self, n, m, y, mask):
        valid = np.asarray(mask, bool)
        n, m, y = [np.where(valid, z, 0.) for z in [n, m, y]]
        forecast = n[:, None, :]+self.candidates[None, :, None]*(m-n)[:, None, :]
        loss = (((forecast-y[:, None, :])**2)*valid[:, None, :]).sum(axis=-1)
        logits = np.log(self.weights)-self.eta*loss
        logits -= logits.max(axis=1, keepdims=True)
        self.weights = np.exp(logits)
        self.weights /= self.weights.sum(axis=1, keepdims=True)
        self._floor()


def warm_hedge(calibration, cutoff, eta, mixed=False):
    H = check_ledger(calibration)
    gate = HedgeGate(H, eta, MIX_GRID if mixed else None)
    consumed = []
    for i, origin in enumerate(calibration['origins']):
        assert int(origin)+H <= cutoff, 'Hedge calibration exceeds cutoff'
        consumed.append(int(origin))
        if calibration['available'][i] and calibration['valid_mask'][i].any():
            gate.observe(*[calibration[k][i] for k in ['numeric','text','targets','valid_mask']])
    return gate.weights.copy(), consumed


def replay(ledger, gate, *, update=True):
    """Emit first, then decay once per observed origin and consume ready records."""
    H = check_ledger(ledger)
    gate = copy.deepcopy(gate)
    pending, consumed = deque(), []
    predictions, alphas, events, pred_seconds, update_seconds = [], [], [], [], []
    last_update = -1
    feedback_origins = []
    for i, origin in enumerate(ledger['origins']):
        t = int(origin)
        begin = time.perf_counter()
        alpha = gate.alpha()
        assert alpha.shape == (H,) and np.isfinite(alpha).all() and np.all((alpha>=0)&(alpha<=1))
        applied = alpha if ledger['available'][i] else np.zeros(H)
        n, m = ledger['numeric'][i], ledger['text'][i]
        predictions.append(n+applied[:, None]*(m-n))
        alphas.append(applied.copy()); feedback_origins.append(last_update)
        pred_seconds.append(time.perf_counter()-begin)
        pending.append(i)
        begin = time.perf_counter()
        if update:
            gate.advance(t)
        while pending and int(ledger['origins'][pending[0]])+H <= t:
            j = pending.popleft(); saved_origin = int(ledger['origins'][j])
            assert saved_origin not in consumed
            consumed.append(saved_origin)
            valid = ledger['valid_mask'][j]
            informative = bool(ledger['available'][j]) and bool(valid.any())
            if update and informative:
                gate.observe(*[ledger[k][j] for k in ['numeric','text','targets','valid_mask']])
                last_update = saved_origin
            events.append(dict(origin=saved_origin,update_at=t,mature_at=saved_origin+H,
                               value_updated=bool(update and informative),valid_cells=int(valid.sum()),
                               available=bool(ledger['available'][j])))
        update_seconds.append(time.perf_counter()-begin)
    expected = [int(o) for o in ledger['origins'] if int(o)+H<=int(ledger['origins'][-1])]
    assert consumed == expected, 'Missing or repeated mature origin'
    return dict(origins=np.asarray(ledger['origins']).copy(),predictions=np.stack(predictions),
                targets=np.asarray(ledger['targets']).copy(),valid_mask=np.asarray(ledger['valid_mask']).copy(),
                alpha=np.stack(alphas),feedback_origins=np.asarray(feedback_origins),
                prediction_seconds=np.asarray(pred_seconds),update_seconds=np.asarray(update_seconds)), events
