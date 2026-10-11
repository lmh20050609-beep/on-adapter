"""Atomic domain tables with separate vector/token quotas and no online encoding.

Pooled tables are resident; ragged token payloads use one read-only mapped array.
Capacity failures occur before encoding, and are recorded rather than ignored.
"""
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from .resources import atomic_json, identity, exclusive, digest, isolated_rng


class CacheCapacityError(RuntimeError):
    pass


@dataclass
class Budget:
    vector_disk_bytes: int = 512 * 1024**2
    token_disk_bytes: int = 12 * 1024**3
    vector_ram_bytes: int = 128 * 1024**2
    token_batch_ram_bytes: int = 256 * 1024**2


class Table:
    def __init__(self, folder, budget):
        self.folder = Path(folder); self.budget = budget
        self.metadata = json.loads((self.folder / 'manifest.json').read_text('utf-8'))
        self.stats = {'batch_hits': 0, 'requested_rows': 0, 'runtime_encoder_calls': 0,
                      'runtime_misses': 0, 'device_transfers': 0}
        # Hash payload once on open. Subsequent batches perform indexed reads only.
        for name, expected in self.metadata['files'].items():
            if digest(self.folder / name) != expected:
                raise ValueError('Feature table integrity failure: ' + name)
        self.offsets = np.load(self.folder / 'offsets.npy')
        if len(self.offsets)!=self.metadata['text_count']+1 or self.offsets[0]!=0 or np.any(np.diff(self.offsets)<1):
            raise ValueError('Invalid cached ragged offsets')
        if self.metadata['kind'] == 'vectors':
            size = self.metadata['payload_bytes']
            if size > budget.vector_ram_bytes:
                raise CacheCapacityError('Vector table requires %d resident bytes, budget=%d; no LRU fallback' % (size, budget.vector_ram_bytes))
            self.values = np.load(self.folder / 'values.npy')
        else:
            self.values = np.load(self.folder / 'values.npy', mmap_mode='r')
        if self.values.shape!=(int(self.offsets[-1]),768) or self.values.dtype!=np.float16:
            raise ValueError('Invalid cached representation payload')

    def gather(self, indices, device):
        indices = np.asarray(indices, dtype=np.int64)
        if np.any(indices < 0) or np.any(indices >= len(self.offsets) - 1):
            self.stats['runtime_misses'] += 1
            raise IndexError('Unprepared text ID; runtime encoding is forbidden')
        self.stats['batch_hits'] += 1; self.stats['requested_rows'] += indices.size
        if self.metadata['kind'] == 'vectors':
            result = torch.from_numpy(np.ascontiguousarray(self.values[indices])).to(device=device, dtype=torch.float32)
        else:
            lengths = self.offsets[indices + 1] - self.offsets[indices]
            # Account for packed FP16 host data and FP32 staging, not entry count.
            required = int(lengths.sum()) * self.values.shape[1] * 6
            if required > self.budget.token_batch_ram_bytes:
                raise CacheCapacityError('Token batch staging requires %d bytes; budget=%d' % (required, self.budget.token_batch_ram_bytes))
            packed = np.empty((int(lengths.sum()), self.values.shape[1]), np.float16)
            cursor = 0
            for idx, length in zip(indices.tolist(), lengths.tolist()):
                packed[cursor:cursor+length] = self.values[self.offsets[idx]:self.offsets[idx+1]]
                cursor += length
            joined = torch.from_numpy(packed).to(device=device, dtype=torch.float32)
            result = list(joined.split(lengths.tolist()))
        self.stats['device_transfers'] += int(torch.device(device).type != 'cpu')
        return result


def build_table(root, descriptor, texts, encoder, budget=None, batch_size=16):
    """encoder.lengths and encoder.encode are deliberately separate from lookup.

    A completed table is published only after all payload and offset hashes exist.
    A single quota lock serializes writers for one representation class; readers
    never acquire it. Incomplete temporary directories also consume the quota.
    """
    budget = budget or Budget()
    kind = 'tokens' if descriptor['pooling'] == 'tokens' else 'vectors'
    parent = Path(root) / kind; parent.mkdir(parents=True, exist_ok=True)
    key = identity({'descriptor': descriptor, 'texts': texts})
    folder = parent / key
    if (folder / 'manifest.json').is_file():
        return Table(folder, budget), {'table_hit': 1, 'table_miss': 0, 'encoded_texts': 0, 'encoder_batches': 0, 'rejected_writes': 0, 'key': key}
    stats = {'table_hit': 0, 'table_miss': 1, 'encoded_texts': 0, 'encoder_batches': 0, 'rejected_writes': 0, 'key': key}
    with exclusive(parent / 'writer.lock'), isolated_rng():
        if (folder / 'manifest.json').is_file():
            return Table(folder, budget), dict(stats, table_hit=1, table_miss=0)
        lengths = np.asarray(encoder.lengths(texts, batch_size), dtype=np.int64)
        if kind == 'vectors': lengths[:] = 1
        offsets = np.r_[0, np.cumsum(lengths)]
        payload = int(offsets[-1]) * 768 * 2
        needed = payload + offsets.nbytes + 65536  # array headers and metadata
        cap = budget.vector_disk_bytes if kind == 'vectors' else budget.token_disk_bytes
        used = sum(p.stat().st_size for p in parent.rglob('*') if p.is_file())
        reason = None
        if used + needed > cap: reason = 'disk quota'
        free_bytes=shutil.disk_usage(parent).free
        if needed > free_bytes: reason = 'physical free disk capacity'
        if kind == 'vectors' and payload > budget.vector_ram_bytes: reason = 'resident vector RAM quota'
        if kind == 'tokens' and int(lengths.max(initial=0)) * 768 * 6 > budget.token_batch_ram_bytes:
            reason = 'single token representation exceeds staging RAM quota'
        if reason:
            stats['rejected_writes'] = 1
            event = dict(stats, reason=reason, required_bytes=needed, used_bytes=used, cap_bytes=cap, free_bytes=free_bytes, time=time.time())
            atomic_json(Path(root) / ('capacity_failure_' + kind + '.json'), event)
            raise CacheCapacityError(json.dumps(event) + '; enlarge the named budget or explicitly relocate regenerable optimized cache. No encoding was started.')
        temp = parent / (key + '.building.' + str(os.getpid()))
        temp.mkdir()
        try:
            values = np.lib.format.open_memmap(temp / 'values.npy', mode='w+', dtype=np.float16, shape=(int(offsets[-1]), 768))
            for start in range(0, len(texts), batch_size):
                batch = texts[start:start+batch_size]
                encoded = encoder.encode(batch)
                if len(encoded) != len(batch): raise ValueError('Encoder batch cardinality changed')
                for row, value in enumerate(encoded, start):
                    value = np.asarray(value, dtype=np.float32).reshape(-1, 768)
                    if value.shape[0] != lengths[row] or not np.isfinite(value).all():
                        raise ValueError('Invalid encoded shape or non-finite representation')
                    values[offsets[row]:offsets[row+1]] = value.astype(np.float16)
                stats['encoded_texts'] += sum(bool(t.strip()) for t in batch)
                stats['encoder_batches'] += int(any(t.strip() for t in batch))
            values.flush(); del values
            np.save(temp / 'offsets.npy', offsets)
            metadata = {'key': key, 'kind': kind, 'descriptor': descriptor, 'text_count': len(texts),
                        'payload_bytes': payload, 'build_stats': stats,
                        'files': {n: digest(temp / n) for n in ['values.npy', 'offsets.npy']}}
            atomic_json(temp / 'manifest.json', metadata)
            os.replace(temp, folder)
        except BaseException:
            # Only this process's unpublished work is removable. User results are never touched.
            shutil.rmtree(temp, ignore_errors=True)
            raise
    return Table(folder, budget), stats
