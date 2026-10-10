"""Author DSOF teacher/student losses, with explicit origin-indexed label release.

This is a disclosed evaluation bridge, not the author's implicit buffer indexing.
TD uses only the just-observed prefix and teacher pseudo-labels; ER uses complete
observed historical windows. Missing truths are masked in both loss paths.
"""
from collections import deque
import numpy as np
import torch
from .online import DSOFBridge


class ObservedMSE(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, prediction, truth):
        mask = torch.isfinite(truth)
        safe = torch.where(mask, truth, prediction.detach())
        error = (prediction - safe).square()
        if self.reduction == 'none':
            return error * (error.numel() / mask.sum().clamp(min=1))
        return error.sum() / mask.sum().clamp(min=1)


class CausalDSOF(DSOFBridge):
    def __init__(self, *args):
        super().__init__(*args)
        self.framework.criterion = ObservedMSE()
        self.framework.MSEReductNone = ObservedMSE('none')

    def args_from_batch(self, batch):
        args = super().args_from_batch(batch)
        args[2] = batch['y'].masked_fill(~batch['mask'], float('nan'))
        return args

    def native_training_step(self, batch):
        return float(self.framework.train_step(*self.args_from_batch(batch))['loss'])

    def reset_online(self):
        self.pending, self.replay = deque(), deque(maxlen=300)
        self.previous = None
        self.last_event = {}

    def observed_window(self, origin, cutoff, device, partial=False):
        end = min(origin + self.horizon, cutoff)
        if not partial:
            assert end == origin + self.horizon
        rows = np.arange(origin + 1, end + 1)
        x = torch.from_numpy(self.data.values[origin + 1 - self.lookback:origin + 1]).unsqueeze(0).to(device)
        mark = torch.from_numpy(self.data.marks[origin + 1 - self.lookback:origin + 1]).unsqueeze(0).to(device)
        truth = x.new_full((1, self.horizon, x.shape[-1]), float('nan'))
        if len(rows):
            y = self.data.standardized_truth[rows]
            truth[:, :len(rows)] = torch.from_numpy(y).to(device)
        return [None, x, truth, mark, mark[:, -1:].expand(-1, self.horizon, -1)]

    def stream_step(self, batch):
        import time
        update_started = time.perf_counter()
        origin, device = int(batch['origin'][0]), batch['x'].device
        f = self.framework
        while self.pending and self.pending[0] + self.horizon <= origin:
            self.replay.append(self.pending.popleft())
        er_count = 0
        if len(self.replay) >= 32:
            chosen = np.random.choice(list(self.replay), 32, replace=False)
            args = [self.observed_window(int(o), origin, device) for o in chosen]
            merged = [None] + [torch.cat([a[i] for a in args]) for i in range(1, 5)]
            for opt in f.opt['test']['batch']:
                opt.zero_grad()
            prev = f._process_one_batch(*merged, mode='test')
            f.backward_lossT_test(prev)
            f.backward_lossFT(prev, prev)
            for opt in f.opt['test']['batch']:
                opt.step(); opt.zero_grad()
            f.store_grad(); f.update_ocp(prev, prev)
            er_count = len(chosen)
        td_count = 0
        if self.previous is not None and self.previous == origin - 1:
            for opt in f.opt['test']['online']:
                opt.zero_grad()
            prev = f._process_one_batch(*self.observed_window(self.previous, origin, device, partial=True), mode='test')
            with torch.no_grad():
                curr = f._process_one_batch(*super().args_from_batch(batch), mode='test')
            f.backward_tdlossFT(prev, curr)
            for opt in f.opt['test']['online']:
                opt.step(); opt.zero_grad()
            f.store_grad(); f.update_ocp(prev, curr)
            td_count = 1
        self.last_update_host_seconds = time.perf_counter() - update_started
        with torch.no_grad():
            out = f._process_one_batch(*super().args_from_batch(batch), mode='test')
        self.pending.append(origin)
        self.previous = origin
        self.last_event = {'er_samples': er_count, 'td_updates': td_count,
                           'max_observed_label_row': origin if td_count or er_count else None}
        return out
