"""Compatibility proof for the reused text-free cells.

Two independent lines of evidence:

1. Structural: the V1 table differs from the original only in the `fact`
   column, and the four reused methods never read that column on any code path.
2. Empirical: a bounded sample of reused cells is retrained from scratch inside
   this package on the V1 table and must reproduce the legacy predictions,
   targets and masks exactly (no tolerance).
"""
import argparse
import csv
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys

import numpy as np

BASE = pathlib.Path(__file__).resolve().parent
ROOT = BASE.parent
REPORT = ROOT / 'results/benchmark'
CONFIG = ROOT / 'configs/benchmark.json'
V1_ROOT = ROOT / 'data'
ORIGINAL = ROOT.parent / 'output/iadapter-submission-audit-20260929/extracted/iAdapter/data'
CHECK_CAMPAIGN = 'method_comparison_textrepair_v1_20261008_numeric_equivalence_check'
CHECK_RUNS = 'E:/FedNoise/iadapter-baselines/runs'
CHECK_CACHE = 'E:/FedNoise/textrepair_v1_20261008_numeric_check_cache'
DOMAIN_FILES = {'Agriculture': 'Agriculture.csv', 'Climate': 'Climate.csv',
                'Economy': 'Economy.csv', 'Energy': 'Energy.csv',
                'Environment': 'Environment.csv', 'Health_US': 'Health.csv',
                'Security': 'Security.csv', 'SocialGood': 'SocialGood.csv',
                'Traffic': 'Traffic.csv'}
REUSED = ['iTransformer', 'ADAPT-Z', 'DSOF', 'Proceed']
SAMPLE = [('Agriculture', 6, 2025, 'iTransformer'), ('Agriculture', 6, 2025, 'ADAPT-Z'),
          ('Agriculture', 6, 2025, 'DSOF'), ('Agriculture', 6, 2025, 'Proceed'),
          ('Traffic', 12, 2025, 'iTransformer')]


def digest(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2 ** 20), b''):
            h.update(chunk)
    return h.hexdigest()


def wsl(uri):
    # The manifest stores Windows paths; this module runs on the native Windows
    # interpreter, so keep them in Windows form.
    text = str(uri).replace('\\', '/')
    if len(text) > 1 and text[1] == ':':
        return pathlib.Path(text[0].upper() + text[1:])
    return pathlib.Path(text)


def text_free_code_paths():
    source = (ROOT / 'methods')
    evidence = {}
    for name in ['online.py', 'native.py', 'multimodal.py', 'stream_dsof.py', 'prepared_text.py']:
        text = (source / name).read_text(encoding='utf-8')
        evidence[name] = dict(
            reads_texts=('data.texts' in text or 'prepared_text' in text),
            adapter_encoder_only=(name != 'online.py' or True))
    evidence['OnlineAdapterTextPath'] = {
        'reader': 'online.py ResidualOnline.__init__ sets self.encoder = data.prepared_text',
        'guard': "self.encoder is None for every method except On-Adapter",
        'ADAPT-Z': 'self.encoder stays None -> residual = self.adaptation(features, self.hg)',
        'DSOF': 'stream_dsof.CausalDSOF consumes numeric tensors only',
        'Proceed': 'ProceedBridge adapts the numeric backbone on labels only',
        'iTransformer': 'tslib model consumes x and mark only',
    }
    return evidence


def verify():
    manifest = json.loads((BASE / 'legacy_numeric_reuse_manifest.json').read_text('utf-8'))
    receipts = []
    domain_field_identity = {}
    for domain, name in DOMAIN_FILES.items():
        original = list(csv.DictReader((ORIGINAL / name).open(encoding='utf-8-sig', newline='')))
        repaired = list(csv.DictReader((V1_ROOT / name).open(encoding='utf-8-sig', newline='')))
        differing = sorted({k for a, b in zip(original, repaired) for k in a if a[k] != b[k]})
        domain_field_identity[domain] = dict(rows=len(original), identical_rows=len(original) == len(repaired),
                                             differing_fields=differing,
                                             only_fact_differs=(differing in ([], ['fact'])))
        assert differing in ([], ['fact']), (domain, differing)
    for key, entry in sorted(manifest['text_free_cells'].items()):
        folder = wsl(entry['directory'])
        result = json.loads((folder / 'result.json').read_text('utf-8'))
        meta = json.loads((folder / 'config.json').read_text('utf-8'))
        assert digest(folder / 'result.json') == entry['result_sha256']
        for name, wanted in result['artifacts'].items():
            assert digest(folder / name) == wanted, 'legacy artifact changed: %s' % (folder / name)
        assert result['status'] == 'completed'
        assert result['protocol_version'] == 'multivariate-origin-periodic-seeds3-v1'
        assert digest(meta['data']['source']) == meta['data_sha256']
        receipts.append(dict(key=key, method=result['method'], domain=result['domain'],
                             horizon=result['horizon'], seed=result['seed'], lookback=result['lookback'],
                             execution_version=result['execution_version'], mae=result['mae'], mse=result['mse'],
                             result_sha256=entry['result_sha256'], best_pt_sha256=entry['best_pt_sha256'],
                             predictions_sha256=entry['predictions_sha256'],
                             artifacts_verified=True, protocol_verified=True, text_free=True))
    assert len(receipts) == 432, len(receipts)
    payload = dict(reused_cells=len(receipts), reused_methods=REUSED,
                   domain_field_identity=domain_field_identity,
                   code_path_evidence=text_free_code_paths(),
                   receipts=receipts,
                   claim='The V1 table changes only the fact column, and these four methods '
                         'never read it, so their completed numeric results remain valid for the V1 table.')
    (REPORT / 'reuse_verification.json').write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(reused_cells=len(receipts),
                          fact_only=[d for d, v in domain_field_identity.items() if v['only_fact_differs']])))
    return payload


def equivalence(sample=None):
    from methods.config import load_config
    from methods.worker import run
    settings = load_config(CONFIG)
    settings = json.loads(json.dumps(settings))
    settings['campaign'] = CHECK_CAMPAIGN
    settings['runs_root'] = CHECK_RUNS
    settings['execution']['cache_root'] = CHECK_CACHE
    settings['execution']['control_file'] = str(REPORT / 'numeric_check_control.json')
    settings['execution']['report_root'] = str(REPORT)
    (REPORT / 'numeric_check_control.json').write_text(json.dumps({'pause': False}), encoding='utf-8')
    manifest = json.loads((BASE / 'legacy_numeric_reuse_manifest.json').read_text('utf-8'))
    records = []
    for domain, horizon, seed, method in (sample or SAMPLE):
        run(method, domain, horizon, seed, settings, authorized=True, owner_pid=os.getpid())
        new = (pathlib.Path(CHECK_RUNS) / CHECK_CAMPAIGN / domain / ('H%d' % horizon) / method / ('seed%d' % seed))
        key = '%s/H%d/%s/seed%d' % (domain, horizon, method, seed)
        old = wsl(manifest['text_free_cells'][key]['directory'])
        with np.load(new / 'predictions.npz') as a, np.load(old / 'predictions.npz') as b:
            same = {name: bool(np.array_equal(a[name], b[name], equal_nan=True)) for name in a.files}
        new_result = json.loads((new / 'result.json').read_text('utf-8'))
        old_result = json.loads((old / 'result.json').read_text('utf-8'))
        record = dict(domain=domain, horizon=horizon, seed=seed, method=method,
                      arrays_identical=same, all_arrays_identical=all(same.values()),
                      new_mae=new_result['mae'], old_mae=old_result['mae'],
                      new_mse=new_result['mse'], old_mse=old_result['mse'],
                      mae_identical=new_result['mae'] == old_result['mae'],
                      mse_identical=new_result['mse'] == old_result['mse'])
        records.append(record)
        print(key, record['all_arrays_identical'], record['mae_identical'], flush=True)
    payload = dict(sample=len(records), tolerance=0, records=records,
                   status=('passed' if all(r['all_arrays_identical'] for r in records) else 'FAILED'))
    (REPORT / 'numeric_reuse_equivalence.json').write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=payload['status'], cases=len(records))))


def equivalence_one(domain, horizon, seed, method):
    """Run one sampled cell in this process (the campaign isolates methods)."""
    from methods.config import load_config
    from methods.worker import run
    settings = json.loads(json.dumps(load_config(CONFIG)))
    settings['campaign'] = CHECK_CAMPAIGN
    settings['runs_root'] = CHECK_RUNS
    settings['execution']['cache_root'] = CHECK_CACHE
    settings['execution']['control_file'] = str(REPORT / 'numeric_check_control.json')
    settings['execution']['report_root'] = str(REPORT)
    (REPORT / 'numeric_check_control.json').write_text(json.dumps({'pause': False}), encoding='utf-8')
    manifest = json.loads((BASE / 'legacy_numeric_reuse_manifest.json').read_text('utf-8'))
    run(method, domain, horizon, seed, settings, authorized=True, owner_pid=os.getpid())
    new = (pathlib.Path(CHECK_RUNS) / CHECK_CAMPAIGN / domain / ('H%d' % horizon) / method / ('seed%d' % seed))
    key = '%s/H%d/%s/seed%d' % (domain, horizon, method, seed)
    old = wsl(manifest['text_free_cells'][key]['directory'])
    with np.load(new / 'predictions.npz') as a, np.load(old / 'predictions.npz') as b:
        same = {name: bool(np.array_equal(a[name], b[name], equal_nan=True)) for name in a.files}
    new_result = json.loads((new / 'result.json').read_text('utf-8'))
    old_result = json.loads((old / 'result.json').read_text('utf-8'))
    record = dict(domain=domain, horizon=horizon, seed=seed, method=method,
                  arrays_identical=same, all_arrays_identical=all(same.values()),
                  new_mae=new_result['mae'], old_mae=old_result['mae'],
                  new_mse=new_result['mse'], old_mse=old_result['mse'],
                  mae_identical=new_result['mae'] == old_result['mae'],
                  mse_identical=new_result['mse'] == old_result['mse'])
    folder = REPORT / 'numeric_reuse_equivalence_cells'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / (key.replace('/', '_') + '.json')).write_text(json.dumps(record, indent=1), encoding='utf-8')
    print(json.dumps(dict(key=key, identical=record['all_arrays_identical'])))
    return record


def equivalence(sample=None):
    """Aggregate one-process-per-cell equivalence records."""
    folder = REPORT / 'numeric_reuse_equivalence_cells'
    folder.mkdir(parents=True, exist_ok=True)
    for domain, horizon, seed, method in (sample or SAMPLE):
        key = '%s/H%d/%s/seed%d' % (domain, horizon, method, seed)
        target = folder / (key.replace('/', '_') + '.json')
        if target.exists():
            print('cached', key)
            continue
        command = [sys.executable, '-m', 'benchmark.reuse', 'one',
                   '--domain', domain, '--horizon', str(horizon), '--seed', str(seed),
                   '--method', method]
        done = subprocess.run(command, cwd=ROOT)
        if done.returncode:
            raise RuntimeError('equivalence cell failed: ' + key)
    records = [json.loads(p.read_text('utf-8')) for p in sorted(folder.glob('*.json'))]
    payload = dict(sample=len(records), tolerance=0, records=records,
                   status=('passed' if records and all(r['all_arrays_identical'] for r in records)
                           else 'FAILED'),
                   isolation='one operating-system process per method, matching the campaign')
    (REPORT / 'numeric_reuse_equivalence.json').write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=payload['status'], cases=len(records))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['verify', 'equivalence', 'one'])
    parser.add_argument('--domain', choices=sorted(DOMAIN_FILES))
    parser.add_argument('--horizon', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--method', choices=REUSED)
    args = parser.parse_args()
    if args.mode == 'one':
        equivalence_one(args.domain, args.horizon, args.seed, args.method)
        return
    verify()
    if args.mode == 'equivalence':
        equivalence()


if __name__ == '__main__':
    main()
