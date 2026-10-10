"""Measured effect of V1 on the tokens each text-reading method actually sees.

For every row the model input is compared between the original (V0) and the
minimally repaired (V1) table, using the frozen per-method tokenizer, token
limit and text-mapping rule. Nothing here reads scores.
"""
import csv
import hashlib
import json
import pathlib
import re

import numpy as np

BASE = pathlib.Path(__file__).resolve().parent
ROOT = BASE.parent
REPORT = ROOT / 'results/data_repair'
V1_ROOT = ROOT / 'data'
ORIGINAL = ROOT.parent / 'output/iadapter-submission-audit-20260929/extracted/iAdapter/data'
DOMAIN_FILES = {'Agriculture': 'Agriculture.csv', 'Climate': 'Climate.csv',
                'Economy': 'Economy.csv', 'Energy': 'Energy.csv',
                'Environment': 'Environment.csv', 'Health_US': 'Health.csv',
                'Security': 'Security.csv', 'SocialGood': 'SocialGood.csv',
                'Traffic': 'Traffic.csv'}
DOMAINS = ['Environment', 'Energy', 'Health_US', 'Traffic', 'SocialGood',
           'Security', 'Economy', 'Climate', 'Agriculture']
WINDOWS = {'Environment': (96, [48, 96, 192, 336]), 'Energy': (24, [12, 24, 36, 48]),
           'Health_US': (24, [12, 24, 36, 48]), 'Traffic': (8, [6, 8, 10, 12]),
           'SocialGood': (8, [6, 8, 10, 12]), 'Security': (8, [6, 8, 10, 12]),
           'Economy': (8, [6, 8, 10, 12]), 'Climate': (8, [6, 8, 10, 12]),
           'Agriculture': (8, [6, 8, 10, 12])}
METHODS = ['On-Adapter', 'GPT4MTS', 'MM-TSFlib']
BOUNDARY = re.compile(r';\s*(?=\d{4}-\d{2}-\d{2}:)')


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def sorted_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return sorted(csv.DictReader(stream), key=lambda r: r.get('end_date') or r.get('date') or '')


def main():
    from methods.data import load_series, ForecastWindows
    from methods.prepared_text import (
        FrozenBatchEncoder, text_plan, SPECS)
    tokenizers = {}
    for method in METHODS:
        encoder = FrozenBatchEncoder(SPECS[method]); encoder.tokenizer_load()
        tokenizers[method] = encoder.tokenizer
    rules = json.loads((BASE / 'source_evidence/decision_rules_all9.json').read_text('utf-8'))
    results = []
    for domain in DOMAINS:
        L, horizons = WINDOWS[domain]
        v0 = load_series(domain, data_root=ORIGINAL, split=(.6, .2, .2), cache_root=None)
        v1 = load_series(domain, data_root=V1_ROOT, split=(.6, .2, .2), cache_root=None)
        assert np.array_equal(v0.mask, v1.mask) and np.array_equal(v0.values, v1.values)
        raw0 = sorted_rows(ORIGINAL / DOMAIN_FILES[domain])
        raw1 = sorted_rows(V1_ROOT / DOMAIN_FILES[domain])
        assert len(raw0) == len(raw1) == len(v0.values)
        masked = {r['complete_item_sha256'] for r in rules
                  if r['domain'] == domain and r['classification'] == 'B'}
        row_items = {}
        for row, record in enumerate(raw0):
            cell = (record.get('fact') or '').strip()
            row_items[row] = [x.strip() for x in BOUNDARY.split(cell)] if cell else []
        rows_with_masked_item = {row for row, items in row_items.items()
                                 if any(sha(x) in masked for x in items)}
        spans = {}
        for split in ['train', 'val', 'test']:
            spans[split] = set(int(x) for x in
                               ForecastWindows(v1, split, L, min(horizons)).indices)
        per_method = {}
        for method in METHODS:
            tokenizer = tokenizers[method]
            max_length = SPECS[method]['max_length']
            texts0, ids0 = text_plan(v0, method)
            texts1, ids1 = text_plan(v1, method)
            changed_text, changed_token = [], []
            for row in range(len(v0.values)):
                before, after = texts0[ids0[row]], texts1[ids1[row]]
                if before == after:
                    continue
                changed_text.append(row)
                tokens_before = tokenizer(before, truncation=True, max_length=max_length,
                                          padding=False)['input_ids'] if before.strip() else []
                tokens_after = tokenizer(after, truncation=True, max_length=max_length,
                                         padding=False)['input_ids'] if after.strip() else []
                if tokens_before != tokens_after:
                    changed_token.append(row)
            decoded = {}
            for row in rows_with_masked_item:
                text = texts0[ids0[row]]
                ids = tokenizer(text, truncation=True, max_length=max_length,
                                padding=False)['input_ids'] if text.strip() else []
                decoded[row] = tokenizer.decode(ids, skip_special_tokens=True)
            visible = [row for row in rows_with_masked_item
                       if any(sha(x) in masked and x[:24] in decoded[row] for x in row_items[row])]
            per_split = {}
            for split in ['train', 'val', 'test']:
                origins = spans[split]
                touching = {o for o in origins
                            if any(r in rows_with_masked_item for r in (o - 1, o))}
                per_split[split] = dict(
                    origins=len(origins),
                    origins_with_changed_tokens=len(origins & set(changed_token)),
                    origins_whose_visible_rows_contain_a_masked_item=len(touching))
            per_method[method] = dict(
                distinct_texts_before=len(texts0), distinct_texts_after=len(texts1),
                rows_with_changed_text=len(changed_text), rows_with_changed_tokens=len(changed_token),
                rows_whose_raw_cell_contains_a_masked_item=len(rows_with_masked_item),
                of_those_rows_the_masked_item_survives_into_the_token_window=len(visible),
                per_split=per_split)
        results.append(dict(domain=domain, rows=len(v0.values), masked_items=len(masked),
                            v0_sha256=digest(ORIGINAL / DOMAIN_FILES[domain]),
                            v1_sha256=digest(V1_ROOT / DOMAIN_FILES[domain]),
                            methods=per_method))
    (REPORT / 'token_effects.json').write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    header = ('domain,method,masked_items,rows_with_changed_text,rows_with_changed_tokens,'
              'rows_whose_raw_cell_contains_a_masked_item,'
              'of_those_rows_the_masked_item_survives_into_the_token_window,'
              'test_origins,test_origins_with_changed_tokens,'
              'test_origins_whose_visible_rows_contain_a_masked_item\n')
    with (REPORT / 'repair_token_effects.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        stream.write(header)
        for entry in results:
            for method in METHODS:
                m = entry['methods'][method]
                stream.write('%s,%s,%d,%d,%d,%d,%d,%d,%d,%d\n' % (
                    entry['domain'], method, entry['masked_items'], m['rows_with_changed_text'],
                    m['rows_with_changed_tokens'], m['rows_whose_raw_cell_contains_a_masked_item'],
                    m['of_those_rows_the_masked_item_survives_into_the_token_window'],
                    m['per_split']['test']['origins'],
                    m['per_split']['test']['origins_with_changed_tokens'],
                    m['per_split']['test']['origins_whose_visible_rows_contain_a_masked_item']))
    print(json.dumps({e['domain']: {m: e['methods'][m]['rows_with_changed_tokens'] for m in METHODS}
                      for e in results}, indent=1))


if __name__ == '__main__':
    main()
