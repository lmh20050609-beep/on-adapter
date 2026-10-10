"""Complete nine-domain source inventory.

Reuses the frozen four-domain (Agriculture/Economy/Security/Energy) decision
rules verbatim; every other item stays D until an independent source decision
is recorded. Screening hits are never promoted to A by themselves.
Never loads run results, predictions or checkpoints.
"""
import collections
import csv
import hashlib
import json
import pathlib
import re

BASE = pathlib.Path(__file__).resolve().parent
ROOT = BASE.parent
DATA_ROOT = ROOT.parent / 'output/iadapter-submission-audit-20260929/extracted/iAdapter/data'
DOMAIN_FILES = {'Agriculture': 'Agriculture.csv', 'Climate': 'Climate.csv',
                'Economy': 'Economy.csv', 'Energy': 'Energy.csv',
                'Environment': 'Environment.csv', 'Health_US': 'Health.csv',
                'Security': 'Security.csv', 'SocialGood': 'SocialGood.csv',
                'Traffic': 'Traffic.csv'}
DOMAINS = ['Environment', 'Energy', 'Health_US', 'Traffic', 'SocialGood',
           'Security', 'Economy', 'Climate', 'Agriculture']
FACT_FIELD = 'fact'
NEVER_READ = ['preds', 'pred', 'Final_Output', 'Final_Search_8', 'Final_Search_16']
INHERITED_DOMAINS = ['Agriculture', 'Economy', 'Security', 'Energy']

# Only these three methods read text, and all three read the cleaned `fact`
# column through SeriesData.texts (see config.METHODS / prepared_text.SPECS).
TEXT_METHODS = ['On-Adapter', 'GPT4MTS', 'MM-TSFlib']
NUMERIC_ONLY_METHODS = ['iTransformer', 'ADAPT-Z', 'DSOF', 'Proceed']

ITEM_SPLIT = re.compile(r';\s*(?=\d{4}-\d{2}-\d{2}:)')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(text):
    return re.sub(r'\s+', ' ', text.strip().rstrip('.;')).strip()


def load_rules():
    path = BASE / 'source_evidence' / 'inherited_decision_rules.json'
    rules = json.loads(path.read_text(encoding='utf-8'))
    return rules, sha(path.read_bytes())


def scan(domains=DOMAINS):
    rules, rules_sha = load_rules()
    by_key = {(r['domain'], r['field'], canonical(r['pattern'])): r for r in rules}
    entries = []
    cells = []
    summary = {}
    for domain in domains:
        path = DATA_ROOT / DOMAIN_FILES[domain]
        raw = path.read_bytes()
        file_sha = sha(raw)
        with path.open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            prior = reader.line_num
            rows = []
            for idx, row in enumerate(reader):
                rows.append((idx, prior + 1, reader.line_num, row))
                prior = reader.line_num
        order = {old: new for new, (old, _, _, _) in enumerate(
            sorted(rows, key=lambda x: (x[3].get('end_date') or x[3].get('date') or '',
                                        x[3].get('start_date') or '')))}
        counts = collections.Counter()
        for idx, line_start, line_end, row in rows:
            for field in [FACT_FIELD] + [c for c in NEVER_READ if c in row]:
                text = row.get(field) or ''
                cells.append(dict(
                    domain=domain, original_file=str(path), original_file_sha256=file_sha,
                    original_row_index_zero_based=idx, physical_line_start=line_start,
                    physical_line_end=line_end, sorted_row_index_zero_based=order[idx],
                    record_start=row.get('start_date'), record_end=row.get('end_date') or row.get('date'),
                    field=field, read_by_model=(field == FACT_FIELD),
                    read_by_methods=TEXT_METHODS if field == FACT_FIELD else [],
                    text_sha256=sha(text.encode()), nonempty=bool(text.strip()),
                    character_count=len(text)))
                counts[field + '_cells'] += 1
                counts[field + '_nonempty_cells'] += bool(text.strip())
            text = row.get(FACT_FIELD) or ''
            if not text.strip():
                continue
            for item_idx, item in enumerate(ITEM_SPLIT.split(text)):
                if not item.strip():
                    continue
                item = item.strip()
                match = re.match(r'^\s*(\d{4}-\d{2}-\d{2}):\s*(.*)', item, re.S)
                if match:
                    described_at = match.group(1)
                    basis = 'independent dated item prefix'
                    body = match.group(2)
                else:
                    described_at = row.get('end_date') or row.get('date')
                    basis = 'record end; item lacks own date'
                    body = item
                rule = by_key.get((domain, FACT_FIELD, canonical(item)), {})
                future_yrs = [y for y in re.findall(r'\b(?:19|20)\d{2}\b', body)
                              if y > str(described_at)[:4]]
                entries.append(dict(
                    domain=domain, original_file=str(path), original_file_sha256=file_sha,
                    original_row_index_zero_based=idx, csv_record_line_one_based=idx + 2,
                    physical_line_start=line_start, physical_line_end=line_end,
                    sorted_row_index_zero_based=order[idx],
                    record_start=row.get('start_date'), record_end=row.get('end_date') or row.get('date'),
                    field=FACT_FIELD, item_index_zero_based=item_idx, text=item,
                    text_sha256=sha(item.encode()), complete_item_sha256=sha(item.encode()),
                    described_at=described_at, described_at_basis=basis,
                    class_source=('inherited_frozen_rule' if rule else 'pending_new_decision'),
                    classification=rule.get('classification', 'D'),
                    available_at=rule.get('available_at'), decision_rule_id=rule.get('id'),
                    source_urls=rule.get('source_urls', []),
                    evidence=rule.get('evidence', 'No source/body availability proof completed. Not certified merely because the screen did not flag it.'),
                    version_limit=rule.get('version_limit', 'Individual raw search URL/body history not present in the public processed CSV.'),
                    treatment_v1=rule.get('action', 'keep'),
                    treatment_v2=('keep' if rule.get('classification') == 'A' else 'mask_item'),
                    screening_only_future_years=future_yrs))
                counts[FACT_FIELD + '_entries'] += 1
                counts[FACT_FIELD + '_' + entries[-1]['classification']] += 1
                counts[FACT_FIELD + '_future_year_screen_hits'] += bool(future_yrs)
        summary[domain] = dict(original_file=str(path), original_file_sha256=file_sha,
                               rows=len(rows), fact_nonempty_cells=counts[FACT_FIELD + '_nonempty_cells'],
                               counts=dict(counts))
    return entries, cells, summary, rules_sha


def main():
    entries, cells, summary, rules_sha = scan()
    out = BASE / 'source_evidence'
    out.mkdir(parents=True, exist_ok=True)
    for name, values in [('all_text_entries.jsonl', entries), ('all_text_cells.jsonl', cells)]:
        with (out / name).open('w', encoding='utf-8', newline='\n') as handle:
            for value in values:
                handle.write(json.dumps(value, ensure_ascii=False) + '\n')
    payload = dict(inherited_decision_rules_sha256=rules_sha,
                   rules_source='reports/benchmark_text_pair_v1_20261008/source_evidence/decision_rules.json',
                   text_methods=TEXT_METHODS, numeric_only_methods=NUMERIC_ONLY_METHODS,
                   text_field=FACT_FIELD, never_read_fields=NEVER_READ,
                   default_classification='D', entries=len(entries), cells=len(cells),
                   domains=summary)
    (out / 'inventory_summary.json').write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({d: dict(entries=v['counts'].get('fact_entries'),
                              hits=v['counts'].get('fact_future_year_screen_hits'),
                              inherited=sum(1 for e in entries if e['domain'] == d and e['class_source'] == 'inherited_frozen_rule'))
                      for d, v in summary.items()}, ensure_ascii=False))


if __name__ == '__main__':
    main()
