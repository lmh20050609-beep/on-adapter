"""Sequential controller for the frozen nine-domain V1 text-repair campaign.

Only the three methods whose forward/training path reads the `fact` column are
trained here (324 cells). The four text-free methods reuse their completed
legacy cells, each pinned by `legacy_numeric_reuse_manifest.json`.

Every cell is a separate OS process. Completed cells are re-verified by digest
and skipped, so the campaign is resumable without retraining.
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

BASE = pathlib.Path(__file__).resolve().parent
ROOT = BASE.parent
REPORT = ROOT / 'results/benchmark'
CONFIG = ROOT / 'configs/benchmark.json'
TEXT_METHODS = ['On-Adapter', 'GPT4MTS', 'MM-TSFlib']
DOMAINS = ['Environment', 'Energy', 'Health_US', 'Traffic', 'SocialGood',
           'Security', 'Economy', 'Climate', 'Agriculture']
SEEDS = [2025, 2026, 2027]
APPROVED = {'Environment': (96, [48, 96, 192, 336]), 'Energy': (24, [12, 24, 36, 48]),
            'Health_US': (24, [12, 24, 36, 48]), 'Traffic': (8, [6, 8, 10, 12]),
            'SocialGood': (8, [6, 8, 10, 12]), 'Security': (8, [6, 8, 10, 12]),
            'Economy': (8, [6, 8, 10, 12]), 'Climate': (8, [6, 8, 10, 12]),
            'Agriculture': (8, [6, 8, 10, 12])}


def digest(path):
    import hashlib
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def load_settings():
    from methods.config import load_config
    return load_config(CONFIG)


def cell_folder(settings, method, domain, horizon, seed):
    return (pathlib.Path(settings['runs_root']) / settings['campaign'] / domain /
            ('H%d' % horizon) / method / ('seed%d' % seed))


def new_cells():
    ordered = []
    for domain in DOMAINS:
        for horizon in APPROVED[domain][1]:
            for seed in SEEDS:
                for method in TEXT_METHODS:
                    ordered.append((method, domain, horizon, seed))
    return ordered


def completed(settings, method, domain, horizon, seed):
    folder = cell_folder(settings, method, domain, horizon, seed)
    result_path = folder / 'result.json'
    if not result_path.is_file():
        return None
    result = json.loads(result_path.read_text('utf-8'))
    for name, wanted in result['artifacts'].items():
        if digest(folder / name) != wanted:
            raise ValueError('Completed cell artifact changed: %s' % (folder / name))
    return result


def cell(method, domain, horizon, seed):
    from methods.worker import run
    settings = load_settings()
    return run(method, domain, horizon, seed, settings, authorized=True, owner_pid=os.getpid())


def plan():
    settings = load_settings()
    todo = [c for c in new_cells() if completed(settings, *c) is None]
    done = len(new_cells()) - len(todo)
    print(json.dumps(dict(cells=len(new_cells()), completed=done, remaining=len(todo),
                          runs_root=settings['runs_root'], campaign=settings['campaign'],
                          order=todo[:5]), indent=1))


def run_all():
    settings = load_settings()
    control = pathlib.Path(settings['execution']['control_file'])
    control.parent.mkdir(parents=True, exist_ok=True)
    if not control.exists():
        control.write_text(json.dumps({'pause': False, 'created': time.time()}), encoding='utf-8')
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / 'controller_pid.json').write_text(
        json.dumps({'pid': os.getpid(), 'started': time.time()}), encoding='utf-8')
    order = new_cells()
    log = REPORT / 'campaign.log'
    failures = []
    for index, (method, domain, horizon, seed) in enumerate(order, 1):
        key = '%s %s H%d seed%d' % (method, domain, horizon, seed)
        already = completed(settings, method, domain, horizon, seed)
        if already is not None:
            print('[%d/%d] skip completed %s' % (index, len(order), key), flush=True)
            continue
        (REPORT / 'controller_state.json').write_text(json.dumps(
            dict(index=index, total=len(order), cell=key, pid=os.getpid(),
                 started=time.time()), indent=1), encoding='utf-8')
        command = [sys.executable, '-m', 'benchmark.runner', 'cell',
                   '--method', method, '--domain', domain, '--horizon', str(horizon),
                   '--seed', str(seed)]
        with log.open('a', encoding='utf-8') as stream:
            stream.write('\n=== %s ===\n' % key)
            stream.flush()
            done = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        if done.returncode:
            failures.append(dict(cell=key, returncode=done.returncode))
            (REPORT / 'failures.json').write_text(json.dumps(failures, indent=1), encoding='utf-8')
            if len(failures) >= 5:
                raise RuntimeError('Five cell failures; stopping for inspection')
        else:
            print('[%d/%d] completed %s' % (index, len(order), key), flush=True)
    (REPORT / 'controller_state.json').write_text(json.dumps(
        dict(status='completed', failures=failures, finished=time.time()), indent=1), encoding='utf-8')
    print(json.dumps(dict(status='completed', failures=len(failures))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['plan', 'cell', 'run'])
    parser.add_argument('--method', choices=TEXT_METHODS)
    parser.add_argument('--domain', choices=DOMAINS)
    parser.add_argument('--horizon', type=int)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    if args.mode == 'plan':
        plan()
    elif args.mode == 'run':
        run_all()
    else:
        cell(args.method, args.domain, args.horizon, args.seed)


if __name__ == '__main__':
    main()
