"""Bounded, versioned read-only resources. No experiment is started here."""
import contextlib
import hashlib
import json
import os
import random
import time
from pathlib import Path
import numpy as np
import torch

VERSION = 'optimized-v1'
_HASHES = {}


def digest(path):
    path = Path(path).resolve()
    stat = path.stat()
    signature = (str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    if signature not in _HASHES:
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(chunk)
        _HASHES[signature] = h.hexdigest()
    return _HASHES[signature]


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), default=str).encode()).hexdigest()


from .state_io import atomic_json, replace, unlink


def atomic_torch(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with temp.open('wb') as stream:
        torch.save(value, stream); stream.flush(); os.fsync(stream.fileno())
    replace(temp, path)


@contextlib.contextmanager
def exclusive(path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    # A stale lock requires explicit inspection; never silently steal it.
    with path.open('x', encoding='utf-8') as stream:
        json.dump({'pid': os.getpid(), 'time': time.time()}, stream)
    try:
        yield
    finally:
        unlink(path)


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda']])


@contextlib.contextmanager
def isolated_rng():
    state = rng_state()
    try:
        yield
    finally:
        restore_rng(state)


class Timings:
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.seconds = {}; self.counters = {}; self.active = []

    @contextlib.contextmanager
    def phase(self, name):
        # Synchronize only phase boundaries, never every batch.
        if self.device.type == 'cuda': torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        self.active.append((name,start))
        try: yield
        finally:
            if self.device.type == 'cuda': torch.cuda.synchronize(self.device)
            self.seconds[name] = self.seconds.get(name, 0.) + time.perf_counter() - start
            self.active.remove((name,start))

    def snapshot(self):
        seconds=dict(self.seconds)
        for name,start in self.active: seconds[name]=seconds.get(name,0.)+time.perf_counter()-start
        return {'seconds': seconds, 'counters': dict(self.counters),
                'semantics':'phase wall time; nested checkpoint/online subtimers are inclusive; active phase heartbeat unsynchronized'}
