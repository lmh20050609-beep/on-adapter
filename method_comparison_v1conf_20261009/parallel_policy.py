"""Admission and lease rules for one controller and isolated configuration workers."""
import contextlib
import json
import os
import subprocess
import time
from pathlib import Path
from .state_io import read_json, unlink, retry
import psutil

MIB = 1024 ** 2
WORKER_MODULES = {'v1': 'method_comparison_optimized.worker',
                  'v2': 'method_comparison_optimized_v2.worker',
                  'v3': 'method_comparison_optimized_v3.worker',
                  'periodic': 'method_comparison_periodic_v3.worker'}


class PreparationCancelled(RuntimeError):
    pass


def job_key(job):
    return job['domain'], int(job['horizon']), int(job['seed']), job['method']


def argument(command, name):
    try:
        return command[command.index(name) + 1]
    except (ValueError, IndexError):
        return None


def validate_policy(settings):
    options = settings['execution']
    n = options.get('parallel_workers', 1)
    if type(n) is not int or not 1 <= n <= 6:
        raise ValueError('This single-card version supports one to six workers')
    slots = options.get('gpu_compute_workers', 1)
    if type(slots) is not int or not 1 <= slots <= n:
        raise ValueError('Compute slots must be between one and worker concurrency')
    if settings['threads'] != 2 or settings.get('num_workers', 0) != 0:
        raise ValueError('Parallel scheduling preserves threads=2 and loader workers=0')
    for name in ['gpu_headroom_mib', 'ram_headroom_mib', 'worker_ram_reservation_mib']:
        if options.get(name, 0) <= 0:
            raise ValueError('Missing positive admission budget: ' + name)


def resources():
    probe = subprocess.run(['nvidia-smi.exe', '--query-gpu=memory.total,memory.free',
                            '--format=csv,noheader,nounits'], capture_output=True,
                           text=True, timeout=15, check=True, creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    total, free = (float(x.strip()) for x in probe.stdout.strip().splitlines()[0].split(','))
    return dict(gpu_total_mib=total, gpu_free_mib=free,
                ram_available_mib=psutil.virtual_memory().available / MIB)


def reservation(job, settings):
    options = settings['execution']
    return dict(gpu_mib=options['gpu_job_reservations_mib'].get(job['method'], 1536),
                ram_mib=options.get('worker_ram_reservations_mib', {}).get(
                    job['method'], options['worker_ram_reservation_mib']))


def can_admit(job, active, known, settings, observed, runtime_limit=None):
    """Reserve loading peaks as well as resident tensors; never oversubscribe slots."""
    limit = min(settings['execution'].get('parallel_workers', 1), runtime_limit or settings['execution'].get('parallel_workers', 1))
    if len(active) >= limit or job_key(job) in {job_key(a['job']) for a in active}:
        return False, 'worker slots'
    dependency = (*job_key(job)[:3], 'iTransformer')
    if job['method'] != 'iTransformer' and dependency not in known:
        return False, 'numerical checkpoint dependency'
    needed = reservation(job, settings)
    options = settings['execution']
    reserved_gpu = sum(a['reservation']['gpu_mib'] for a in active)
    if reserved_gpu + needed['gpu_mib'] + options['gpu_headroom_mib'] > observed['gpu_total_mib']:
        return False, 'GPU reservation budget'
    if needed['gpu_mib'] + options['gpu_headroom_mib'] > observed['gpu_free_mib']:
        return False, 'current GPU free memory'
    if needed['ram_mib'] + options['ram_headroom_mib'] > observed['ram_available_mib']:
        return False, 'current host free memory'
    # A just-spawned process may not have materialized its loading peak yet.
    # Resident RSS is already deducted from observed available RAM. Reserve the
    # unmaterialized loading peak only while that worker is initializing.
    pending_ram = sum(max(0, a['reservation']['ram_mib'] - a.get('rss_mib', 0))
                      for a in active if a.get('initializing', True))
    if pending_ram + needed['ram_mib'] + options['ram_headroom_mib'] > observed['ram_available_mib']:
        return False, 'host loading reservations'
    return True, 'admitted'


def command_matches_lease(process, lease, controller_pid, ancestor_pids):
    command = process['cmdline'] or []
    if WORKER_MODULES[lease['backend']] not in command:
        return False
    if controller_pid not in ancestor_pids:
        return False
    if argument(command, '--config') != lease['config_path']:
        return False
    job = lease['job']
    for field, flag in [('method', '--method'), ('domain', '--domain'),
                        ('horizon', '--horizon'), ('seed', '--seed')]:
        if argument(command, flag) != str(job[field]):
            return False
    if lease['backend'] in {'v2', 'v3', 'periodic'}:
        return (argument(command, '--owner-pid') == str(controller_pid)
                and argument(command, '--lease-token') == lease['token'])
    # Immutable v1 accepts standalone authorized workers. Its wrapper PID is
    # bound by this controller, and v2 never treats it as a v2 measurement.
    return process['pid'] == lease.get('launcher_pid') or process['ppid'] == lease.get('launcher_pid')


def text_cache_ready(data,method,settings,device):
    """Inspect the unchanged representation identity without consuming training RNG."""
    from .prepared_text import SPECS,descriptor,text_plan
    from .resources import isolated_rng,identity
    import torch
    if method not in SPECS: return True
    with isolated_rng():
        texts,_=text_plan(data,method);desc=descriptor(data,method)
        desc.update(encoding_device=str(device),encoding_batch_size=settings['execution']['text_encoding_batch_size'],
            encoding_precision='float32',allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
            deterministic_cudnn=bool(torch.backends.cudnn.deterministic))
        kind='tokens' if desc['pooling']=='tokens' else 'vectors'
        key=identity({'descriptor':desc,'texts':texts})
        return (Path(settings['execution']['cache_root'])/'text'/kind/key/'manifest.json').exists()


@contextlib.contextmanager
def preparation_lock(path, timeout=900, cancelled=None):
    """Serialize quota writers outside unchanged cache code, with bounded waiting."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    while True:
        if cancelled is not None and cancelled():
            raise PreparationCancelled('Cancelled before acquiring execution/preparation slot')
        try:
            with path.open('x', encoding='utf-8') as stream:
                json.dump(dict(pid=os.getpid(), create_time=psutil.Process().create_time()), stream)
            break
        except FileExistsError:
            if time.monotonic() - started > timeout:
                raise TimeoutError('Preparation lock wait exceeded budget: ' + str(path))
            try:
                owner = read_json(path)
            except (json.JSONDecodeError, FileNotFoundError):
                time.sleep(.05)
                continue
            try:
                p = psutil.Process(owner['pid'])
                if p.create_time() != owner['create_time']:
                    raise RuntimeError('Stale preparation lock: inspect before recovery: ' + str(path))
            except psutil.NoSuchProcess:
                raise RuntimeError('Exited preparation lock owner: inspect before recovery: ' + str(path))
            time.sleep(.2)
    try:
        yield time.monotonic() - started
    finally:
        unlink(path)


@contextlib.contextmanager
def gpu_compute_lock(report_root, cancelled=None, slots=1):
    """Exclusive numbered leases permit exactly N simultaneous compute owners."""
    if type(slots) is not int or not 1 <= slots <= 6:
        raise ValueError('Invalid compute slot count')
    root = Path(report_root); root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    owned = None
    while owned is None:
        if cancelled is not None and cancelled():
            raise PreparationCancelled('Cancelled before acquiring GPU slot')
        if time.monotonic() - started > 86400:
            raise TimeoutError('GPU compute slot wait exceeded one day')
        for slot in range(slots):
            path = root / ('gpu_compute_%d.lock' % slot)
            try:
                with path.open('x', encoding='utf-8') as stream:
                    json.dump(dict(pid=os.getpid(), create_time=psutil.Process().create_time(),
                                   slot=slot, capacity=slots), stream)
                owned = path
                break
            except FileExistsError:
                try:
                    owner = read_json(path, timeout=.5)
                except FileNotFoundError:
                    continue  # released while inspecting
                except (PermissionError, json.JSONDecodeError):
                    continue  # writing owner header; bounded outer wait
                try:
                    process = psutil.Process(owner['pid'])
                    if process.create_time() != owner['create_time']:
                        raise RuntimeError('PID-reused compute lease: ' + str(path))
                except psutil.NoSuchProcess:
                    raise RuntimeError('Exited compute lease owner; inspect before recovery: ' + str(path))
            except PermissionError:
                # Windows can deny create while the previous lease is deleting.
                if time.monotonic() - started > 8.:
                    raise
        if owned is None:
            time.sleep(.05)
    try:
        yield time.monotonic() - started
    finally:
        owner = read_json(owned)
        if owner['pid'] != os.getpid() or owner['create_time'] != psutil.Process().create_time():
            raise RuntimeError('Compute lease ownership changed')
        unlink(owned)
