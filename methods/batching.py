"""Vectorized windows and known-shape observed-label indexing, no CUDA nonzero."""
import numpy as np
import torch


def host_batch(dataset, items):
    ids = dataset.indices[np.asarray(items, dtype=np.int64)]
    data, L, H = dataset.data, dataset.lookback, dataset.horizon
    history = ids[:, None] + np.arange(1-L, 1)[None, :]
    future = ids[:, None] + np.arange(1, H+1)[None, :]
    result = dict(x=torch.from_numpy(data.values[history]),
                  y=torch.from_numpy(data.label_values[future]),
                  mask=torch.from_numpy(data.mask[future]), origin=torch.from_numpy(ids.copy()))
    if dataset.method != 'GPT4MTS': result['mark'] = torch.from_numpy(data.marks[history])
    if dataset.method is None:
        result['history_y'] = torch.from_numpy(data.history_values[history])
        result['history_mask'] = torch.from_numpy(data.mask[history])
    return result


def transfer(batch, device):
    """Preserve flattened observed-element order without device-side nonzero."""
    result = dict(batch)
    if '_observed_count' not in result:
        mask = result['mask']
        if mask.device.type != 'cpu':
            raise ValueError('Prepare observed indices on CPU before transfer')
        flat = mask.numpy().reshape(-1)
        count = int(np.count_nonzero(flat))
        result['_observed_count'] = count
        result['_valid_index'] = None if count == len(flat) else torch.from_numpy(np.flatnonzero(flat))
    return {k: (v.to(device) if isinstance(v, torch.Tensor) and k != 'origin' else v)
            for k, v in result.items()}


def observed_error(prediction, batch):
    if prediction.shape != batch['y'].shape: raise ValueError('Prediction/label shape mismatch')
    if '_observed_count' not in batch:
        if batch['mask'].device.type != 'cpu': raise ValueError('Missing host-prepared observed index')
        batch = transfer(batch, prediction.device)
    if not batch['_observed_count']: raise ValueError('No observed labels')
    pred, truth = prediction.reshape(-1), batch['y'].reshape(-1)
    index = batch['_valid_index']
    if index is not None:
        pred, truth = pred.index_select(0, index), truth.index_select(0, index)
    return pred - truth


class DeviceVectors:
    """Byte-bounded immutable vectors resident on one device, before projection."""
    def __init__(self, table, device, byte_limit):
        required = int(table.values.size) * 4
        if required > byte_limit: raise MemoryError('Resident vectors exceed explicit byte budget')
        self.source, self.device = table, torch.device(device)
        self.metadata, self.stats = table.metadata, table.stats
        self.values = torch.as_tensor(table.values, dtype=torch.float32, device=device)
        self.stats['resident_device_bytes'] = required
        self.stats['device_transfers'] += int(self.device.type == 'cuda')

    def gather(self, indices, device):
        if torch.device(device) != self.device: raise ValueError('Wrong vector-cache device')
        indices = np.asarray(indices, dtype=np.int64)
        if np.any(indices < 0) or np.any(indices >= len(self.values)):
            self.stats['runtime_misses'] += 1
            raise IndexError('Unprepared text ID; runtime encoding forbidden')
        self.stats['batch_hits'] += 1
        self.stats['requested_rows'] += indices.size
        ids = torch.as_tensor(indices.reshape(-1), device=device)
        return self.values.index_select(0, ids).reshape(*indices.shape, 768)


def resident_vectors(text, device, byte_limit):
    if text is not None and text.table.metadata['kind'] == 'vectors':
        text.table = DeviceVectors(text.table, device, byte_limit)
    return text
