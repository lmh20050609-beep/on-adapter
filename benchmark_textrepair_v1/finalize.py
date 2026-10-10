"""Protection snapshot, artifact manifest and completion audit for the V1 campaign."""
import argparse
import collections
import hashlib
import json
import pathlib

BASE = pathlib.Path(__file__).resolve().parent
ROOT = BASE.parent
REPORT = ROOT / 'reports/benchmark_textrepair_v1_20261008'
RUNS = pathlib.Path('E:/FedNoise/iadapter-baselines/runs/method_comparison_textrepair_v1_20261008')
REPAIR_REPORT = ROOT / 'reports/benchmark_text_repair_all9_v1_20261008'
LEGACY_RUNS = pathlib.Path('E:/FedNoise/iadapter-baselines/runs')
LEGACY_CAMPAIGNS = ['method_comparison_opt_periodic_v3_seeds3_20261006',
                    'method_comparison_opt_periodic_v4_seeds3_20261007']
ORIGINAL = ROOT.parent / 'output/iadapter-submission-audit-20260929/extracted/iAdapter/data'
PRIOR_REPORTS = ['benchmark_text_pair_v1_20261008', 'encoding_independent_v2_20261008',
                 'benchmark_text_pair_independent_v2_20261008', 'benchmark_text_audit_v1',
                 'text_truncation_audit_20261007', 'periodic_guard_repair_20261007',
                 'socialgood_real_source_v1_20261007', 'frequency_combinations_20261007']
DOMAINS = ['Environment', 'Energy', 'Health_US', 'Traffic', 'SocialGood',
           'Security', 'Economy', 'Climate', 'Agriculture']
WINDOWS = {'Environment': [48, 96, 192, 336], 'Energy': [12, 24, 36, 48],
           'Health_US': [12, 24, 36, 48], 'Traffic': [6, 8, 10, 12],
           'SocialGood': [6, 8, 10, 12], 'Security': [6, 8, 10, 12],
           'Economy': [6, 8, 10, 12], 'Climate': [6, 8, 10, 12],
           'Agriculture': [6, 8, 10, 12]}
METHODS = ['On-Adapter', 'GPT4MTS', 'MM-TSFlib']
SEEDS = [2025, 2026, 2027]


def digest(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2 ** 20), b''):
            h.update(chunk)
    return h.hexdigest()


def protected_paths():
    paths = []
    for campaign in LEGACY_CAMPAIGNS:
        for result in (LEGACY_RUNS / campaign).glob('*/*/*/seed*/result.json'):
            paths += [p for p in result.parent.iterdir() if p.is_file()]
    paths += [p for p in ORIGINAL.iterdir() if p.is_file()]
    for name in PRIOR_REPORTS:
        folder = ROOT / 'reports' / name
        if folder.exists():
            paths += [p for p in folder.rglob('*') if p.is_file()
                      and 'audit_traces' not in p.parts]
    return sorted(set(paths))


def snapshot(name):
    paths = protected_paths()
    record = {str(p): digest(p) for p in paths}
    (REPORT / name).write_text(json.dumps(dict(files=len(record), digests=record), indent=1) + '\n',
                               encoding='utf-8')
    print(json.dumps(dict(snapshot=name, files=len(record))))
    return record


def compare():
    before = json.loads((REPORT / 'protected_before.json').read_text('utf-8'))['digests']
    after = {str(p): digest(p) for p in protected_paths()}
    changed = sorted(k for k in before if k not in after or before[k] != after[k])
    added = sorted(k for k in after if k not in before)
    recomputed = {str(p): digest(p) for p in protected_paths()}
    assert all(recomputed[k] == after[k] for k in after)
    payload = dict(protected_files=len(before), unchanged=len(before) - len(changed),
                   changed=changed, added_under_protected_roots=added)
    (REPORT / 'protection_check.json').write_text(json.dumps(payload, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(dict(protected=payload['protected_files'], changed=len(changed))))
    return payload


SELF_OUTPUTS = {'artifact_manifest.json', 'artifact_manifest_verification.json'}


def manifest():
    """Hash every produced artifact except this step's own two outputs."""
    files = {}
    for folder in [REPAIR_REPORT, REPORT]:
        for path in folder.rglob('*'):
            # Live run logs are retained as files but not hash-pinned: the
            # controller keepalive log is appended to while this step runs.
            if path.is_file() and path.name not in SELF_OUTPUTS and path.suffix != '.log':
                files[str(path)] = digest(path)
    for domain in DOMAINS:
        for horizon in WINDOWS[domain]:
            for seed in SEEDS:
                for method in METHODS:
                    folder = RUNS / domain / ('H%d' % horizon) / method / ('seed%d' % seed)
                    for path in folder.glob('*'):
                        if path.is_file():
                            files[str(path)] = digest(path)
    payload = dict(files=len(files), digests=files)
    (REPORT / 'artifact_manifest.json').write_text(json.dumps(payload, indent=1) + '\n', encoding='utf-8')
    (REPORT / 'artifact_manifest_verification.json').write_text(json.dumps(dict(
        files=len(files), verified=sum(1 for p, d in files.items() if digest(p) == d)),
        indent=1) + '\n', encoding='utf-8')
    print(json.dumps(dict(artifacts=len(files))))


def audit():
    completed = collections.Counter()
    failures = []
    folders = {}
    for domain in DOMAINS:
        for horizon in WINDOWS[domain]:
            for seed in SEEDS:
                for method in METHODS:
                    folder = RUNS / domain / ('H%d' % horizon) / method / ('seed%d' % seed)
                    result = folder / 'result.json'
                    key = '%s/H%d/%s/seed%d' % (domain, horizon, method, seed)
                    if not result.is_file():
                        failures.append(dict(cell=key, reason='no result.json'))
                        continue
                    record = json.loads(result.read_text('utf-8'))
                    if record.get('status') != 'completed':
                        failures.append(dict(cell=key, reason='status ' + str(record.get('status'))))
                        continue
                    for name, wanted in record['artifacts'].items():
                        if digest(folder / name) != wanted:
                            failures.append(dict(cell=key, reason='artifact changed: ' + name))
                            break
                    completed[(domain, method)] += 1
    lock = json.loads((REPORT / 'execution_lock.json').read_text('utf-8'))
    lock_ok = all(digest(ROOT / p) == d for p, d in lock.items())
    payload = dict(status='completed' if not failures else 'completed_with_failures',
                   new_training_cells_completed=sum(completed.values()),
                   new_training_cells_expected=324,
                   reused_legacy_text_free_cells=432,
                   matrix_cells=756,
                   failures=failures,
                   execution_lock_unchanged=lock_ok,
                   per_domain_new={d: sum(completed[(d, m)] for m in METHODS) for d in DOMAINS},
                   text_version='V1',
                   metric_recomputation='see results.json / results.csv from benchmark_textrepair_v1.score')
    (REPORT / 'completion_audit.json').write_text(json.dumps(payload, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=payload['status'], completed=payload['new_training_cells_completed'],
                          failures=len(failures), lock_ok=lock_ok)))



def text_tables():
    """Record the frozen encoded-input table identity of every text cell."""
    records = []
    for domain in DOMAINS:
        for horizon in WINDOWS[domain]:
            for seed in SEEDS:
                for method in METHODS:
                    folder = RUNS / domain / ('H%d' % horizon) / method / ('seed%d' % seed)
                    meta_path = folder / 'config.json'
                    if not meta_path.is_file():
                        continue
                    meta = json.loads(meta_path.read_text('utf-8'))
                    table = meta.get('feature_table')
                    if not table:
                        continue
                    records.append(dict(domain=domain, horizon=horizon, seed=seed, method=method,
                                        table_key=table.get('key'),
                                        kind=table.get('kind'),
                                        text_count=table.get('text_count'),
                                        payload_bytes=table.get('payload_bytes'),
                                        descriptor={k: table['descriptor'][k] for k in
                                                    ['model', 'max_length', 'pooling', 'layers',
                                                     'text_version', 'source_decisions_sha256',
                                                     'effective_model_batch_size',
                                                     'independent_entry_encoding']}
                                        if 'descriptor' in table else None,
                                        data_sha256=meta.get('data_sha256')))
    (REPORT / 'encoded_input_tables.json').write_text(json.dumps(dict(cells=len(records), records=records),
                                                                 indent=1) + '\n', encoding='utf-8')
    print(json.dumps(dict(text_cells=len(records),
                          distinct_tables=len({r['table_key'] for r in records}))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['protect', 'compare', 'manifest', 'audit', 'tables'])
    args = parser.parse_args()
    {'protect': lambda: snapshot('protected_before.json'),
     'compare': compare, 'manifest': manifest, 'audit': audit,
     'tables': text_tables}[args.mode]()


if __name__ == '__main__':
    main()
