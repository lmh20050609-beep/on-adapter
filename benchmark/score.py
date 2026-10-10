"""Compose and independently recompute the frozen 756-cell comparison table.

Metric definitions are frozen before any result is inspected:

* MAE / MSE: all observed target elements of all selected numeric columns, in the
  training-standardised space, pooled per cell.
* CVaR_q: exact worst-q probability mass of the uniform per-origin MAE
  distribution, with a fractional boundary origin when q*N is not an integer.
  Cross-checked against the variational definition.
* Main statistic: cross-task geometric mean of the per-(domain,H) seed-averaged
  CVaR90 ratio against the common iTransformer backbone. Equal weight per
  domain/H; monthly-group and frequency-combination summaries are declared
  exploratory in advance.
"""
import argparse
import csv
import hashlib
import json
import math
import pathlib

import numpy as np

BASE = pathlib.Path(__file__).resolve().parent
ROOT = BASE.parent
REPORT = ROOT / 'results/benchmark'
NEW_RUNS = pathlib.Path('E:/FedNoise/iadapter-baselines/runs/method_comparison_textrepair_v1_20261008')
DOMAINS = ['Environment', 'Energy', 'Health_US', 'Traffic', 'SocialGood',
           'Security', 'Economy', 'Climate', 'Agriculture']
APPROVED = {'Environment': (96, [48, 96, 192, 336]), 'Energy': (24, [12, 24, 36, 48]),
            'Health_US': (24, [12, 24, 36, 48]), 'Traffic': (8, [6, 8, 10, 12]),
            'SocialGood': (8, [6, 8, 10, 12]), 'Security': (8, [6, 8, 10, 12]),
            'Economy': (8, [6, 8, 10, 12]), 'Climate': (8, [6, 8, 10, 12]),
            'Agriculture': (8, [6, 8, 10, 12])}
METHODS = ['iTransformer', 'ADAPT-Z', 'DSOF', 'Proceed', 'On-Adapter', 'GPT4MTS', 'MM-TSFlib']
BACKBONE = 'iTransformer'
REUSED_METHODS = ['iTransformer', 'ADAPT-Z', 'DSOF', 'Proceed']
NEW_METHODS = ['On-Adapter', 'GPT4MTS', 'MM-TSFlib']
SEEDS = [2025, 2026, 2027]
MONTHLY = ['Traffic', 'SocialGood', 'Security', 'Economy', 'Climate', 'Agriculture']


def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def cvar(values, q):
    values = np.sort(np.asarray(values, dtype=np.float64))[::-1]
    mass = len(values) * q
    whole = int(math.floor(mass))
    fraction = mass - whole
    return float((values[:whole].sum() + (values[whole] * fraction if fraction else 0.0)) / mass)


def cvar_variational(values, q):
    x = np.asarray(values, dtype=np.float64)
    return float(min(t + np.maximum(x - t, 0).mean() / q for t in x))


def cell_metrics(predictions, targets, mask):
    prediction = predictions.astype(np.float64)
    truth = targets.astype(np.float64)
    mask = mask.astype(bool)
    error = prediction - truth
    counts = mask.sum(axis=(1, 2))
    if not np.all(counts > 0):
        raise ValueError('an origin has no observed target elements')
    if not np.isfinite(error[mask]).all():
        raise ValueError('non-finite prediction or target on observed elements')
    absolute = np.where(mask, np.abs(error), 0.0)
    squared = np.where(mask, error ** 2, 0.0)
    origin_loss = absolute.sum(axis=(1, 2)) / counts
    values = dict(
        mae=float(absolute.sum() / counts.sum()),
        mse=float(squared.sum() / counts.sum()),
        cvar80=cvar(origin_loss, .2), cvar90=cvar(origin_loss, .1), cvar95=cvar(origin_loss, .05),
        effective_origins=int(len(origin_loss)), valid_target_elements=int(counts.sum()),
        origin_mae=origin_loss)
    for q in (.2, .1, .05):
        assert abs(values['cvar%d' % int(round((1 - q) * 100))] - cvar_variational(origin_loss, q)) < 1e-12
    return values


def load_cell(method, domain, horizon, seed, manifest):
    if method in REUSED_METHODS:
        key = '%s/H%d/%s/seed%d' % (domain, horizon, method, seed)
        entry = manifest['text_free_cells'][key]
        folder = windows_to_wsl(entry['directory'])
        if digest(folder / 'result.json') != entry['result_sha256']:
            raise ValueError('reused legacy result changed: ' + key)
        source = 'reused_legacy_text_free'
        legacy = json.loads((folder / 'result.json').read_text('utf-8'))
    else:
        folder = NEW_RUNS / domain / ('H%d' % horizon) / method / ('seed%d' % seed)
        if not (folder / 'result.json').is_file():
            return None
        legacy = json.loads((folder / 'result.json').read_text('utf-8'))
        for name, wanted in legacy['artifacts'].items():
            if digest(folder / name) != wanted:
                raise ValueError('new result artifact changed: ' + str(folder / name))
        source = 'new_v1_training'
    with np.load(folder / 'predictions.npz') as archive:
        arrays = {name: archive[name] for name in archive.files}
    values = cell_metrics(arrays['predictions'], arrays['targets'], arrays['mask'])
    for field in ('mae', 'mse'):
        if not math.isclose(values[field], float(legacy[field]), rel_tol=1e-6, abs_tol=1e-9):
            raise ValueError('recomputed %s disagrees with saved result for %s %s H%d seed%d'
                             % (field, method, domain, horizon, seed))
    values.update(method=method, domain=domain, horizon=horizon, seed=seed, source=source,
                  directory=str(folder), epochs=legacy.get('epochs'),
                  update_events=legacy.get('update_events', 0),
                  execution_version=legacy.get('execution_version'))
    return values


def windows_to_wsl(uri):
    # Native Windows interpreter: keep the manifest's Windows path form.
    text = str(uri).replace('\\', '/')
    if len(text) > 1 and text[1] == ':':
        return pathlib.Path(text[0].upper() + text[1:])
    return pathlib.Path(text)



def oracle_relative(by_key, method, metric='cvar90'):
    """Geometric mean over 36 tasks of the seed-averaged ratio to the per-task best."""
    task_ratios = []
    for domain in DOMAINS:
        for horizon in APPROVED[domain][1]:
            per_seed = []
            for seed in SEEDS:
                best = min(by_key[(domain, horizon, seed, m)][metric] for m in METHODS)
                per_seed.append(by_key[(domain, horizon, seed, method)][metric] / best)
            task_ratios.append(float(np.mean(per_seed)))
    return geometric_mean(task_ratios)


def geometric_mean(values):
    if any(v <= 0 for v in values):
        return None
    return float(math.exp(sum(math.log(v) for v in values) / len(values)))



def historical_reference():
    """Same statistic computed from the completed legacy 756 cells.

    This is a clearly separated reference, not the main table. The legacy
    text methods used the batched encoder and the original (V0) fact column,
    so a difference against V1 mixes the text repair with the encoding fix.
    """
    manifest = json.loads((BASE / 'legacy_numeric_reuse_manifest.json').read_text('utf-8'))
    legacy = json.loads((BASE / 'legacy_all_methods_manifest.json').read_text('utf-8'))
    rows = []
    for domain in DOMAINS:
        for horizon in APPROVED[domain][1]:
            for seed in SEEDS:
                for method in METHODS:
                    key = '%s/H%d/%s/seed%d' % (domain, horizon, method, seed)
                    entry = legacy['cells'].get(key)
                    if entry is None:
                        return None
                    folder = windows_to_wsl(entry['directory'])
                    with np.load(folder / 'predictions.npz') as archive:
                        arrays = {name: archive[name] for name in archive.files}
                    values = cell_metrics(arrays['predictions'], arrays['targets'], arrays['mask'])
                    values.update(method=method, domain=domain, horizon=horizon, seed=seed,
                                  execution_version=entry['execution_version'])
                    rows.append(values)
    by_key = {(r['domain'], r['horizon'], r['seed'], r['method']): r for r in rows}
    ratios = {k: v['cvar90'] / by_key[(k[0], k[1], k[2], BACKBONE)]['cvar90'] for k, v in by_key.items()}
    summary = {}
    for method in METHODS:
        task_ratios = []
        for domain in DOMAINS:
            for horizon in APPROVED[domain][1]:
                task_ratios.append(float(np.mean([ratios[(domain, horizon, s, method)] for s in SEEDS])))
        ranks = []
        for domain in DOMAINS:
            for horizon in APPROVED[domain][1]:
                for seed in SEEDS:
                    order = sorted(METHODS, key=lambda m: by_key[(domain, horizon, seed, m)]['cvar90'])
                    ranks.append(order.index(method) + 1)
        summary[method] = dict(geometric_mean_relative_cvar90=geometric_mean(task_ratios),
                               geometric_mean_relative_oracle_cvar90=oracle_relative(by_key, method),
                               mean_rank_cvar90=float(np.mean(ranks)),
                               mean_mae=float(np.mean([by_key[(d, h, s, method)]['mae']
                                                       for d in DOMAINS for h in APPROVED[d][1] for s in SEEDS])),
                               mean_mse=float(np.mean([by_key[(d, h, s, method)]['mse']
                                                       for d in DOMAINS for h in APPROVED[d][1] for s in SEEDS])))
    payload = dict(cells=len(rows), summary=summary,
                   label='historical reference: legacy periodic-v3/v4 cells, original (V0) fact column '
                         'and the pre-fix batched text encoder for the three text methods',
                   not_the_main_table=True)
    (REPORT / 'historical_reference.json').write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({m: summary[m]['geometric_mean_relative_cvar90'] for m in METHODS}, indent=1))
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--historical', action='store_true')
    args = parser.parse_args()
    if args.historical:
        historical_reference()
        return
    manifest = json.loads((BASE / 'legacy_numeric_reuse_manifest.json').read_text('utf-8'))
    rows = []
    missing = []
    for domain in DOMAINS:
        for horizon in APPROVED[domain][1]:
            for seed in SEEDS:
                for method in METHODS:
                    cell = load_cell(method, domain, horizon, seed, manifest)
                    if cell is None:
                        missing.append('%s/H%d/%s/seed%d' % (domain, horizon, method, seed))
                    else:
                        rows.append(cell)
    if missing:
        raise SystemExit('missing cells: %d %s' % (len(missing), missing[:5]))

    by_key = {(r['domain'], r['horizon'], r['seed'], r['method']): r for r in rows}
    ratios = {}
    for r in rows:
        backbone = by_key[(r['domain'], r['horizon'], r['seed'], BACKBONE)]
        ratios[(r['domain'], r['horizon'], r['seed'], r['method'])] = \
            r['cvar90'] / backbone['cvar90']

    task_rows = []
    for domain in DOMAINS:
        for horizon in APPROVED[domain][1]:
            entry = dict(domain=domain, horizon=horizon, periodicity=('daily' if domain == 'Environment'
                          else 'weekly' if domain in ('Energy', 'Health_US') else 'monthly'))
            for method in METHODS:
                values = [by_key[(domain, horizon, s, method)] for s in SEEDS]
                entry[method] = dict(
                    mae=float(np.mean([v['mae'] for v in values])),
                    mse=float(np.mean([v['mse'] for v in values])),
                    cvar80=float(np.mean([v['cvar80'] for v in values])),
                    cvar90=float(np.mean([v['cvar90'] for v in values])),
                    cvar95=float(np.mean([v['cvar95'] for v in values])),
                    relative_cvar90=float(np.mean([ratios[(domain, horizon, s, method)] for s in SEEDS])),
                    effective_origins=int(np.mean([v['effective_origins'] for v in values])),
                    valid_target_elements=int(np.mean([v['valid_target_elements'] for v in values])))
            task_rows.append(entry)

    summary = {}
    for method in METHODS:
        rel = [t[method]['relative_cvar90'] for t in task_rows]
        summary[method] = dict(
            tasks=len(rel),
            geometric_mean_relative_oracle_cvar90=oracle_relative(by_key, method),
            geometric_mean_relative_cvar90=geometric_mean(rel),
            arithmetic_mean_relative_cvar90=float(np.mean(rel)),
            mean_mae=float(np.mean([t[method]['mae'] for t in task_rows])),
            mean_mse=float(np.mean([t[method]['mse'] for t in task_rows])),
            mean_cvar80=float(np.mean([t[method]['cvar80'] for t in task_rows])),
            mean_cvar95=float(np.mean([t[method]['cvar95'] for t in task_rows])),
            worst_relative_cvar90=float(np.max(rel)), best_relative_cvar90=float(np.min(rel)))

    ranks = {m: [] for m in METHODS}
    for domain in DOMAINS:
        for horizon in APPROVED[domain][1]:
            for seed in SEEDS:
                order = sorted(METHODS, key=lambda m: by_key[(domain, horizon, seed, m)]['cvar90'])
                for position, method in enumerate(order, 1):
                    ranks[method].append(position)
    for method in METHODS:
        summary[method]['mean_rank_cvar90'] = float(np.mean(ranks[method]))
        summary[method]['median_rank_cvar90'] = float(np.median(ranks[method]))

    monthly_tasks = [t for t in task_rows if t['domain'] in MONTHLY]
    exploratory = {}
    for method in METHODS:
        exploratory[method] = dict(
            monthly_six_domains_geometric_mean_relative_cvar90=geometric_mean(
                [t[method]['relative_cvar90'] for t in monthly_tasks]))
    frequencies = {}
    for method in METHODS:
        groups = {}
        for domain in DOMAINS:
            periodicity = ('daily' if domain == 'Environment' else
                           'weekly' if domain in ('Energy', 'Health_US') else 'monthly')
            groups.setdefault(periodicity, []).extend(
                t[method]['relative_cvar90'] for t in task_rows if t['domain'] == domain)
        frequencies[method] = {k: geometric_mean(v) for k, v in groups.items()}
        frequencies[method]['frequency_equal_weight_geometric_mean'] = geometric_mean(
            list(frequencies[method][k] for k in ['daily', 'weekly', 'monthly']))

    REPORT.mkdir(parents=True, exist_ok=True)
    fields = ['method', 'domain', 'horizon', 'seed', 'source', 'mae', 'mse', 'cvar80', 'cvar90',
              'cvar95', 'effective_origins', 'valid_target_elements', 'relative_cvar90',
              'epochs', 'update_events', 'execution_version', 'directory']
    with (REPORT / 'results.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for r in sorted(rows, key=lambda x: (DOMAINS.index(x['domain']), x['horizon'], x['seed'], x['method'])):
            r['relative_cvar90'] = ratios[(r['domain'], r['horizon'], r['seed'], r['method'])]
            writer.writerow(r)
    (REPORT / 'results.json').write_text(json.dumps(dict(
        rows=len(rows), cells=len(rows), missing=missing, summary=summary, tasks=task_rows,
        exploratory_monthly=exploratory, frequency_groups=frequencies,
        metric_definition=dict(space='training-standardised observed elements of all selected numeric columns',
                               cvar='exact worst-q probability mass of the uniform per-origin MAE distribution, fractional boundary origin',
                               relative='cell CVaR90 divided by the same domain/H/seed iTransformer cell',
                               main_statistic='equal-weight geometric mean over the 36 domain/H tasks of the seed-averaged relative CVaR90'),
        geometric_mean_undefined='reported as null when any task ratio is non-positive'), indent=2) + '\n',
        encoding='utf-8')
    print(json.dumps({m: dict(gm=summary[m]['geometric_mean_relative_cvar90'],
                              rank=summary[m]['mean_rank_cvar90']) for m in METHODS}, indent=1))


if __name__ == '__main__':
    main()
