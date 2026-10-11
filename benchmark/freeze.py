"""Freeze the nine-domain V1 execution identity before any cell is executed.

Produces
--------
* legacy_numeric_reuse_manifest.json : every reusable text-free cell, pinned by
  the digest of its result and selected checkpoint.
* frozen_matrix_textrepair_v1.json   : all 756 cells, their per-run resolved
  configuration and whether the cell is newly trained or reused.
* execution_lock.json                : digests of every frozen dependency.
"""
import csv
import hashlib
import json
import pathlib
import re

BASE = pathlib.Path(__file__).resolve().parent
ROOT = BASE.parent
REPORT = ROOT / 'results/benchmark'
LEGACY_CAMPAIGNS = ['method_comparison_opt_periodic_v3_seeds3_20261006',
                    'method_comparison_opt_periodic_v4_seeds3_20261007']
LEGACY_RUNS = pathlib.Path('E:/FedNoise/iadapter-baselines/runs')
DOMAINS = ['Environment', 'Energy', 'Health_US', 'Traffic', 'SocialGood',
           'Security', 'Economy', 'Climate', 'Agriculture']
DOMAIN_FILES = {'Agriculture': 'Agriculture.csv', 'Climate': 'Climate.csv',
                'Economy': 'Economy.csv', 'Energy': 'Energy.csv',
                'Environment': 'Environment.csv', 'Health_US': 'Health.csv',
                'Security': 'Security.csv', 'SocialGood': 'SocialGood.csv',
                'Traffic': 'Traffic.csv'}
TEXT_METHODS = ['On-Adapter', 'GPT4MTS', 'MM-TSFlib']
TEXT_FREE_METHODS = ['iTransformer', 'ADAPT-Z', 'DSOF', 'Proceed']
ALL_METHODS = ['iTransformer', 'ADAPT-Z', 'DSOF', 'Proceed', 'On-Adapter', 'GPT4MTS', 'MM-TSFlib']
SEEDS = [2025, 2026, 2027]
APPROVED = {'Environment': (96, [48, 96, 192, 336]), 'Energy': (24, [12, 24, 36, 48]),
            'Health_US': (24, [12, 24, 36, 48]), 'Traffic': (8, [6, 8, 10, 12]),
            'SocialGood': (8, [6, 8, 10, 12]), 'Security': (8, [6, 8, 10, 12]),
            'Economy': (8, [6, 8, 10, 12]), 'Climate': (8, [6, 8, 10, 12]),
            'Agriculture': (8, [6, 8, 10, 12])}
ORIGINAL_DIR = ROOT.parent / 'output/original_timemmd/data'


def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def windows_path(path):
    """Express a WSL path the way the historical Windows runs recorded it."""
    text = str(path).replace('\\', '/')
    match = re.match(r'^/mnt/([a-z])/(.*)$', text)
    if match:
        return match.group(1).upper() + ':/' + match.group(2)
    return text


def resolve_settings(settings, method, rows):
    cfg = dict(settings)
    rules = ['base configuration']
    if rows < 500:
        cfg.update(d_model=32, dropout=.2, batch_size=8, epochs=50); rules.append('legacy rows<500 rule')
    if method in {'ADAPT-Z', 'On-Adapter'}:
        cfg['epochs'] = 3; rules.append('legacy adapter epochs=3')
    if method == 'GPT4MTS':
        cfg.update(epochs=10, batch_size=4); rules.append('legacy GPT4MTS epochs=10,batch=4')
    if method == 'MM-TSFlib':
        cfg['batch_size'] = 8; rules.append('legacy MM-TSFlib batch=8')
    cfg['effective_execution'] = dict(batch_size=cfg['batch_size'], epochs=cfg['epochs'],
                                      threads=cfg['threads'], rules=rules)
    return cfg


def row_counts():
    counts = {}
    for domain, name in DOMAIN_FILES.items():
        with (ORIGINAL_DIR / name).open(encoding='utf-8-sig', newline='') as stream:
            counts[domain] = sum(1 for _ in csv.DictReader(stream))
    return counts


def scan_legacy(methods):
    found = {}
    for campaign in LEGACY_CAMPAIGNS:
        for result_path in (LEGACY_RUNS / campaign).glob('*/*/*/seed*/result.json'):
            domain, horizon_dir, method, seed_dir = result_path.parts[-5:-1]
            if method not in methods:
                continue
            result = json.loads(result_path.read_text('utf-8'))
            if result.get('status') != 'completed':
                continue
            horizon = int(horizon_dir[1:]); seed = int(seed_dir[4:])
            entry = dict(directory=windows_path(result_path.parent), campaign=campaign,
                         execution_version=result.get('execution_version'),
                         lookback=result['lookback'], mse=result['mse'], mae=result['mae'],
                         result_sha256=digest(result_path),
                         config_sha256=digest(result_path.parent / 'config.json'),
                         best_pt_sha256=digest(result_path.parent / 'best.pt'),
                         predictions_sha256=digest(result_path.parent / 'predictions.npz'))
            found['%s/H%d/%s/seed%d' % (domain, horizon, method, seed)] = entry
    return found


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    settings = json.loads((ROOT / 'configs/benchmark.json')
                          .read_text(encoding='utf-8-sig'))
    rows = row_counts()
    backbone = scan_legacy({'iTransformer'})
    textfree = scan_legacy(set(TEXT_FREE_METHODS))

    for domain in DOMAINS:
        for horizon in APPROVED[domain][1]:
            for seed in SEEDS:
                if '%s/H%d/iTransformer/seed%d' % (domain, horizon, seed) not in backbone:
                    raise FileNotFoundError('missing legacy numeric backbone: %s/H%d/seed%d'
                                            % (domain, horizon, seed))
                for method in TEXT_FREE_METHODS:
                    if '%s/H%d/%s/seed%d' % (domain, horizon, method, seed) not in textfree:
                        raise FileNotFoundError('missing legacy text-free cell %s/H%d/%s/seed%d'
                                                % (domain, horizon, method, seed))

    cells = []
    for domain in DOMAINS:
        L, horizons = APPROVED[domain]
        for horizon in horizons:
            for seed in SEEDS:
                for method in ALL_METHODS:
                    entry = dict(domain=domain, horizon=horizon, seed=seed, method=method,
                                 lookback=L, text_version='V1')
                    if method in TEXT_FREE_METHODS:
                        key = '%s/H%d/%s/seed%d' % (domain, horizon, method, seed)
                        entry.update(kind='reused_legacy_text_free', reuse_key=key,
                                     reuse_directory=textfree[key]['directory'],
                                     reuse_result_sha256=textfree[key]['result_sha256'])
                    else:
                        entry.update(kind='new_training_required', text_reading=True,
                                     resolved=resolve_settings(settings, method, rows[domain]))
                    cells.append(entry)
    assert len(cells) == 756
    new_cells = [c for c in cells if c['kind'] == 'new_training_required']
    reused = [c for c in cells if c['kind'].startswith('reused')]

    (BASE / 'legacy_numeric_reuse_manifest.json').write_text(
        json.dumps(dict(text_free_methods=TEXT_FREE_METHODS, backbone_method='iTransformer',
                        cells=backbone, text_free_cells=textfree,
                        note='Reuse is limited to methods whose forward/training path never reads the '
                             'fact text column; the V1 table changes only that column.'), indent=2) + '\n',
        encoding='utf-8')
    (REPORT / 'frozen_matrix_textrepair_v1.json').write_text(
        json.dumps(dict(campaign=settings['campaign'], text_version='V1', cells_total=len(cells),
                        new_training_count=len(new_cells), reused_count=len(reused),
                        new_training=new_cells, reused=reused, cells=cells), indent=2) + '\n',
        encoding='utf-8')

    lock_paths = [ROOT / 'configs/benchmark.json',
                  BASE / 'legacy_numeric_reuse_manifest.json',
                  ROOT / 'data_repair/source_evidence/decision_rules_all9.json']
    lock_paths += sorted((ROOT / 'methods').glob('*.py'))
    lock = {str(p.relative_to(ROOT)): digest(p) for p in lock_paths}
    (REPORT / 'execution_lock.json').write_text(json.dumps(lock, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(cells=len(cells), new=len(new_cells), reused=len(reused),
                          backbone=len(backbone), lock_files=len(lock)), indent=1))


if __name__ == '__main__':
    main()
