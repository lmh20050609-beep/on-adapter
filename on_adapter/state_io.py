"""Bounded Windows sharing-conflict retries; no stale-state fallback."""
import json
import os
import threading
import time
import uuid
from pathlib import Path

_stats = {}
_mutex = threading.Lock()


def _count(name):
    with _mutex:
        _stats[name] = _stats.get(name, 0) + 1


def statistics():
    with _mutex:
        return dict(_stats)


def retry(operation, name, timeout=8., missing=False, malformed=False):
    deadline = time.monotonic() + timeout
    delay = .005
    while True:
        try:
            return operation()
        except (PermissionError, FileNotFoundError, json.JSONDecodeError) as exc:
            if isinstance(exc, FileNotFoundError) and not missing:
                raise
            if isinstance(exc, json.JSONDecodeError) and not malformed:
                raise
            _count(name + '_retries')
            if time.monotonic() >= deadline:
                _count(name + '_exhausted')
                raise
            time.sleep(min(delay, max(0., deadline - time.monotonic())))
            delay = min(.2, delay * 1.5)


def read_json(path, timeout=8.):
    path = Path(path)
    return retry(lambda: json.loads(path.read_text('utf-8-sig')), 'read', timeout,
                 missing=True, malformed=True)


def replace(source, destination, timeout=8.):
    return retry(lambda: os.replace(source, destination), 'replace', timeout)


def unlink(path):
    return retry(lambda: Path(path).unlink(), 'unlink')


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp.' + uuid.uuid4().hex)
    try:
        with temp.open('x', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, default=str)
            stream.flush()
            os.fsync(stream.fileno())
        replace(temp, path)
    finally:
        if temp.exists():
            unlink(temp)
