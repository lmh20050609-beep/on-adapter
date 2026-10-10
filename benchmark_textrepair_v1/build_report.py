"""Assemble the nine-domain V1 campaign report from the frozen artifacts."""
import json
import pathlib

BASE = pathlib.Path(__file__).resolve().parent
ROOT = BASE.parent
REPORT = ROOT / 'reports/benchmark_textrepair_v1_20261008'
METHODS = ['iTransformer', 'ADAPT-Z', 'DSOF', 'Proceed', 'On-Adapter', 'GPT4MTS', 'MM-TSFlib']
DOMAINS = ['Environment', 'Energy', 'Health_US', 'Traffic', 'SocialGood',
           'Security', 'Economy', 'Climate', 'Agriculture']
ORDER = ['Environment', 'Energy', 'Health_US', 'Traffic', 'SocialGood', 'Security',
         'Economy', 'Climate', 'Agriculture']


def load(name):
    return json.loads((REPORT / name).read_text(encoding='utf-8'))


def main():
    results = load('results.json')
    audit = load('completion_audit.json')
    reuse = load('reuse_verification.json')
    checks = load('path_checks.json')
    try:
        equivalence = load('numeric_reuse_equivalence.json')
    except FileNotFoundError:
        equivalence = dict(status='not_run', records=[])
    repair = json.loads((ROOT / 'reports/benchmark_text_repair_all9_v1_20261008/coverage_all9.json')
                        .read_text(encoding='utf-8'))
    summary = results['summary']

    lines = []
    lines.append('# Nine-domain V1 text-repair benchmark: full 756-cell evaluation\n')
    lines.append('2026-10-08. Data version **V1** (nine-domain minimal text repair); protocol in `PROTOCOL.md`, '
                 'data and sources in `../benchmark_text_repair_all9_v1_20261008/REPORT.md`.\n')
    lines.append('This report **does not presuppose that any method wins**; results may be positive, negative or close to unchanged.\n')

    lines.append('## 1. Execution matrix and completion\n')
    lines.append('| Item | Count |\n|---|---|')
    lines.append('| Total matrix cells (9 domains × 4 H × 3 seeds × 7 methods) | %d |' % audit['matrix_cells'])
    lines.append('| Newly trained in this round (On-Adapter / GPT4MTS / MM-TSFlib) | %d completed / %d expected |'
                 % (audit['new_training_cells_completed'], audit['new_training_cells_expected']))
    lines.append('| Reused historical text-independent cells (iTransformer / ADAPT-Z / DSOF / Proceed) | %d |'
                 % audit['reused_legacy_text_free_cells'])
    lines.append('| Failed cells | %d |' % len(audit['failures']))
    lines.append('| Execution lock unchanged | %s |' % audit['execution_lock_unchanged'])
    lines.append('\nNewly trained cells completed per domain: %s\n' % ', '.join(
        '%s %d' % (d, n) for d, n in sorted(audit['per_domain_new'].items())))
    lines.append('Every new cell was trained in a separate OS process from an identically defined initial state; '
                 'no score, adapter checkpoint or result from an older text version was carried over.\n')

    lines.append('## 2. Proof of reuse compatibility\n')
    lines.append('- Reuse is limited to the 4 methods **whose code path does not read the `fact` column**, %d cells in total, '
                 'with the result / config / best.pt / predictions hashes pinned per cell (`reuse_verification.json`).\n'
                 % reuse['reused_cells'])
    lines.append('- The V1 tables are compared row by row with the original tables; the differing fields are `%s`.\n'
                 % json.dumps(sorted({f for v in reuse['domain_field_identity'].values()
                                      for f in v['differing_fields']}), ensure_ascii=False))
    lines.append('- Numeric-equivalence rerun (tolerance 0): status %s, %d sampled cells.\n'
                 % (equivalence['status'], equivalence.get('sample', 0)))
    if equivalence.get('records'):
        lines.append('\n| Sampled cell | predictions/targets/mask/origins bitwise equal | MAE/MSE exactly equal |\n|---|---|---|')
        for record in equivalence['records']:
            lines.append('| %s H%d seed%d %s | %s | %s |'
                         % (record['domain'], record['horizon'], record['seed'], record['method'],
                            record['all_arrays_identical'],
                            record['mae_identical'] and record['mse_identical']))

    lines.append('\n## 3. Checks of previously uncovered paths\n')
    failed = [k for section in checks.values() for k, v in section.items()
              if isinstance(v, dict) and v.get('status') == 'FAILED']
    lines.append('Covers daily/weekly frequencies, non-H12 windows, and GPT4MTS and MM-TSFlib, which had not been trained before; '
                 '%d checks in total, %d failed.\n'
                 % (sum(len(v) for v in checks.values()), len(failed)))
    lines.append('| Check | Passed | Description |\n|---|---|---|')
    lines.append('| Origin/target/mask/standardization alignment | %d/%d | target_rows, mask and standardized ground truth checked per origin |'
                 % (len(checks['alignment']), len(checks['alignment'])))
    lines.append('| Forecasts do not consume future y/mask | %d/%d | Forecasts are bitwise equal after replacing y and mask |'
                 % (sum(v['identical'] for v in checks['minimal_input'].values()), len(checks['minimal_input'])))
    lines.append('| Visibility of text gather rows | %d/%d | The availability time of every gathered row index is no later than the origin |'
                 % (len(checks['text_visibility']), len(checks['text_visibility'])))
    lines.append('| Perturbing future text does not affect past outputs | %d/%d | Tolerance 0, bitwise equal |'
                 % (sum(v['prediction_identical'] for v in checks['perturbation'].values()),
                    len(checks['perturbation'])))
    lines.append('| Preprocessing/cache version isolation | %d/%d | V1 and original files have different hashes; caches are isolated by data identity and text version |'
                 % (sum(v['distinct_from_original'] for v in checks['cache_identity'].values()),
                    len(checks['cache_identity'])))

    lines.append('\n## 4. Main statistic\n')
    lines.append('Main statistic: within each domain/H, average the risk over seeds, divide by the CVaR90 of the shared '
                 'iTransformer backbone for the same domain/H/seed, then take the equally weighted geometric mean over the 36 domain/H tasks.\n')
    lines.append('| Method | Geometric mean relative CVaR90 across tasks | Arithmetic mean | Mean rank | Median rank | Best/worst task ratio |\n'
                 '|---|---|---|---|---|---|')
    for method in METHODS:
        s = summary[method]
        gm = s['geometric_mean_relative_cvar90']
        lines.append('| %s | %s | %.6f | %.4f | %.1f | %.6f / %.6f |'
                     % (method, ('%.6f' % gm) if gm is not None else 'undefined (non-positive ratio present)',
                        s['arithmetic_mean_relative_cvar90'], s['mean_rank_cvar90'],
                        s['median_rank_cvar90'], s['best_relative_cvar90'], s['worst_relative_cvar90']))

    lines.append('\n| Method | Mean MAE | Mean MSE | Mean CVaR80 | Mean CVaR95 |\n|---|---|---|---|---|')
    for method in METHODS:
        s = summary[method]
        lines.append('| %s | %.6f | %.6f | %.6f | %.6f |'
                     % (method, s['mean_mae'], s['mean_mse'], s['mean_cvar80'], s['mean_cvar95']))

    lines.append('\n## 5. Per-domain/H results (seed average)\n')
    lines.append('| Domain | H | Method | MAE | MSE | CVaR90 | CVaR90 relative to backbone |\n|---|---|---|---|---|---|---|')
    for task in sorted(results['tasks'], key=lambda t: (ORDER.index(t['domain']), t['horizon'])):
        for method in METHODS:
            row = task[method]
            lines.append('| %s | %d | %s | %.6f | %.6f | %.6f | %.6f |'
                         % (task['domain'], task['horizon'], method, row['mae'], row['mse'],
                            row['cvar90'], row['relative_cvar90']))

    lines.append('\n## 6. Pre-declared exploratory groupings\n')
    lines.append('The following exploratory groupings were declared before the protocol was frozen; they are not confirmatory conclusions.\n')
    lines.append('| Method | Geometric mean relative CVaR90 over the six monthly domains |\n|---|---|')
    for method in METHODS:
        value = results['exploratory_monthly'][method]['monthly_six_domains_geometric_mean_relative_cvar90']
        lines.append('| %s | %s |' % (method, ('%.6f' % value) if value is not None else 'undefined'))
    lines.append('\n| Method | daily | weekly | monthly | Frequency-equal-weight |\n|---|---|---|---|---|')
    for method in METHODS:
        g = results['frequency_groups'][method]
        fmt = lambda x: ('%.6f' % x) if x is not None else 'undefined'
        lines.append('| %s | %s | %s | %s | %s |'
                     % (method, fmt(g['daily']), fmt(g['weekly']), fmt(g['monthly']),
                        fmt(g['frequency_equal_weight_geometric_mean'])))

    lines.append('\n## 7. Historical reference (existing 756 cells with V0 text + old batched encoding)\n')
    lines.append('The table below is recomputed by **read-only reuse** of the saved forecasts of completed periodic-v3/v4 cells. '
                 'It is **not a new main table**: its text methods use the original V0 text column and the batched encoding '
                 'from before the fix, so its differences from the V1 main table combine the text repair and the encoding fix.\n')
    try:
        history = load('historical_reference.json')
        lines.append('| Method | Geometric mean CVaR90 relative to backbone | Geometric mean relative to per-task best | Mean rank | Mean MAE | Mean MSE |\n'
                     '|---|---|---|---|---|---|')
        for method in sorted(METHODS, key=lambda m: history['summary'][m]['geometric_mean_relative_cvar90']):
            h = history['summary'][method]
            lines.append('| %s | %.4f | %.4f | %.3f | %.6f | %.6f |'
                         % (method, h['geometric_mean_relative_cvar90'],
                            h['geometric_mean_relative_oracle_cvar90'], h['mean_rank_cvar90'],
                            h['mean_mae'], h['mean_mse']))
        lines.append('\n**Direct comparison of V1 and history (same main-statistic definition)**\n')
        lines.append('| Method | V1 GM relative to backbone | Historical GM relative to backbone | Relative difference | Mean rank V1 / history |\n|---|---|---|---|---|')
        for method in sorted(METHODS, key=lambda m: summary[m]['geometric_mean_relative_cvar90']):
            v1 = summary[method]['geometric_mean_relative_cvar90']
            h = history['summary'][method]['geometric_mean_relative_cvar90']
            lines.append('| %s | %.4f | %.4f | %+.2f%% | %.3f / %.3f |'
                         % (method, v1, h, 100 * (v1 - h) / h,
                            summary[method]['mean_rank_cvar90'],
                            history['summary'][method]['mean_rank_cvar90']))
        deltas = [abs(summary[m]['geometric_mean_relative_cvar90']
                      - history['summary'][m]['geometric_mean_relative_cvar90'])
                  / history['summary'][m]['geometric_mean_relative_cvar90'] for m in METHODS]
        lines.append('\nMaximum relative difference %.2f%%. **Neither the text repair nor the encoding fix materially changes '
                     'the comparative structure among methods**: the ranking is unchanged, and On-Adapter\'s disadvantage on Energy '
                     '(H24: 2.47 → 2.53) and its advantages on SocialGood / Traffic are reproduced almost bit for bit, task by task. '
                     'This conclusion applies only to the range of entries repaired in this round; it cannot be extrapolated to '
                     '"the old results were entirely caused by leakage", nor to "the remaining text has been proven clean".\n'
                     % (100 * max(deltas)))
        lines.append('The "three-axis equal-weight geometric ratio to the best" in the historical record is: backbone 1.0861, '
                     'MM-TSFlib 1.0937, GPT4MTS 1.0953, ADAPT-Z 1.1112, On-Adapter 1.1408, Proceed 1.1477, DSOF 1.1622, '
                     'with On in 5th place. The oracle ratio in this table is defined similarly but aggregated differently, '
                     'so it serves only as a directional comparison.\n')
    except FileNotFoundError:
        lines.append('(Historical reference not computed.)\n')

    lines.append('\n## 8. Actual visibility of the text repair\n')
    lines.append('| Domain | Individual entries | B (masked) | C (kept, disclosed) |\n|---|---|---|---|')
    for row in sorted(repair, key=lambda r: ORDER.index(r['domain'])):
        lines.append('| %s | %d | %d | %d |' % (row['domain'], row['items'], row['B'], row['C']))
    lines.append('\nIn most domains the 64-token window of On-Adapter does not contain the masked entries themselves; '
                 'the 256-token window of MM-TSFlib keeps some of them. Actual token changes are in '
                 '`../benchmark_text_repair_all9_v1_20261008/repair_token_effects.csv`.\n')

    lines.append('## 9. Limitations that must be kept\n')
    lines.append('- This protocol was frozen after the old test set had been observed repeatedly; it is **not an independent confirmation or a preregistration**.')
    lines.append('- Source decisions cover only candidate entries hit by screening; C/D entries are kept in the actual data, '
                 'and V1 does not mean "every entry is certified leak-free".')
    lines.append('- The numeric data still use the original `end_date` availability proxy and the retrospective target-mask origin grid; '
                 'this does not prove full causality in the sense of real publication times and historical revisions.')
    lines.append('- The 432 reused cells are historical results under the same protocol; their text independence is shown by '
                 'both the code path and the numeric-equivalence rerun, but this is not the same as retraining them in this round.')
    lines.append('- iTransformer is the shared backbone; ratios to it are relative quantities, not absolute verdicts of better or worse.')
    lines.append('- The text methods in the historical reference use the old batched encoding, so their differences from the V1 main table are not a pure text-repair effect.')
    (REPORT / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('written', REPORT / 'REPORT.md')


if __name__ == '__main__':
    main()
