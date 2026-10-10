"""Sequential controller for the V1 three-seed confirmation of frozen On-Adapter variants.
python -m confirmation.runner run --config configs/confirmation_<tag>.json"""
import argparse, hashlib, json, os, pathlib, subprocess, sys, time
ROOT = pathlib.Path(__file__).resolve().parents[1]
DOMAINS = ['Environment', 'Energy', 'Health_US', 'Traffic', 'SocialGood', 'Security', 'Economy', 'Climate', 'Agriculture']
HOR = {'Environment': [48, 96, 192, 336], 'Energy': [12, 24, 36, 48], 'Health_US': [12, 24, 36, 48]}
SEEDS = [2025, 2026, 2027]
def digest(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def load(config):
    from on_adapter.config import load_config
    return load_config(ROOT / config)
def folder(s, d, h, seed): return pathlib.Path(s['runs_root']) / s['campaign'] / d / ('H%d' % h) / 'On-Adapter' / ('seed%d' % seed)
def completed(s, d, h, seed):
    f = folder(s, d, h, seed)
    if not (f / 'result.json').is_file(): return False
    r = json.loads((f / 'result.json').read_text('utf-8'))
    for n, w in r['artifacts'].items():
        if digest(f / n) != w: raise ValueError('artifact changed: %s' % (f / n))
    return True
def run_all(config):
    s = load(config); rep = pathlib.Path(s['execution']['report_root']); rep.mkdir(parents=True, exist_ok=True)
    ctl = pathlib.Path(s['execution']['control_file'])
    if not ctl.exists(): ctl.write_text(json.dumps({'pause': False}), encoding='utf-8')
    failures = []
    for d in DOMAINS:
        for h in HOR.get(d, [6, 8, 10, 12]):
            for seed in SEEDS:
                if completed(s, d, h, seed): continue
                (rep / 'controller_state.json').write_text(json.dumps(dict(cell='%s H%d seed%d' % (d, h, seed), pid=os.getpid(), started=time.time())), encoding='utf-8')
                cmd = [sys.executable, '-m', 'confirmation.runner', 'cell', '--config', config, '--domain', d, '--horizon', str(h), '--seed', str(seed)]
                with (rep / 'campaign.log').open('a', encoding='utf-8') as log:
                    log.write('\n=== %s H%d seed%d ===\n' % (d, h, seed)); log.flush()
                    done = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                if done.returncode:
                    failures.append('%s H%d seed%d rc=%d' % (d, h, seed, done.returncode)); (rep / 'failures.json').write_text(json.dumps(failures, indent=1), encoding='utf-8')
    (rep / 'controller_state.json').write_text(json.dumps(dict(status='completed', failures=failures, finished=time.time())), encoding='utf-8')
def cell(config, d, h, seed):
    from on_adapter.worker import run
    return run('On-Adapter', d, h, seed, load(config), authorized=True, owner_pid=os.getpid())
if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['run', 'cell']); ap.add_argument('--config', required=True)
    ap.add_argument('--domain'); ap.add_argument('--horizon', type=int); ap.add_argument('--seed', type=int)
    a = ap.parse_args(); run_all(a.config) if a.mode == 'run' else cell(a.config, a.domain, a.horizon, a.seed)
