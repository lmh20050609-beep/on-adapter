"""Build the nine-domain V1 `fact` text tables.

Only the `fact` column changes. Numeric rows, splits, targets, masks and origin
grids are untouched. `preds` and every other field stay byte-identical.
"""
import collections
import csv
import hashlib
import json
import pathlib
import re
import shutil

BASE = pathlib.Path(__file__).resolve().parent
ROOT = BASE.parent
ORIGINAL = ROOT.parent / 'output/iadapter-submission-audit-20260929/extracted/iAdapter/data'
REPORT = ROOT / 'results/data_repair'
FROZEN_THREE = ROOT / 'reports/benchmark_text_pair_v1_20261008/data/V1'
DOMAIN_FILES = {'Agriculture': 'Agriculture.csv', 'Climate': 'Climate.csv',
                'Economy': 'Economy.csv', 'Energy': 'Energy.csv',
                'Environment': 'Environment.csv', 'Health_US': 'Health.csv',
                'Security': 'Security.csv', 'SocialGood': 'SocialGood.csv',
                'Traffic': 'Traffic.csv'}
DOMAINS = ['Environment', 'Energy', 'Health_US', 'Traffic', 'SocialGood',
           'Security', 'Economy', 'Climate', 'Agriculture']
BOUNDARY = re.compile(r';\s*(?=\d{4}-\d{2}-\d{2}:)')


def sha(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    rules = json.loads((BASE / 'source_evidence/decision_rules_all9.json').read_text(encoding='utf-8'))
    rules_sha = sha((BASE / 'source_evidence/decision_rules_all9.json').read_bytes())
    templates = {}
    for rule in rules:
        templates.setdefault((rule['domain'], rule['classification']), 0)
    index = {}
    for rule in rules:
        index.setdefault((rule['domain'], rule['complete_item_sha256']), rule)

    data_root = ROOT / 'data'
    inventory = []
    coverage = []
    invariance = []
    for domain in DOMAINS:
        source = ORIGINAL / DOMAIN_FILES[domain]
        raw = source.read_bytes()
        rows = []
        with source.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            line = 2
            for position, row in enumerate(reader):
                rows.append((position, line, reader.line_num, row))
                line = reader.line_num + 1
        n = len(rows)
        train_end, val_end = int(n * .6), int(n * .8)
        order = {pos: rank for rank, (pos, *_) in enumerate(
            sorted(rows, key=lambda x: (x[3].get('end_date') or x[3].get('date') or '',)))} if False else \
            {pos: rank for rank, (pos, *_) in enumerate(sorted(rows))}
        ordered = sorted(rows, key=lambda x: (x[3].get('end_date') or x[3].get('date') or ''))
        order = {pos: rank for rank, (pos, *_) in enumerate(ordered)}
        output = []
        counts = collections.Counter()
        for position, first_line, last_line, row in rows:
            new = dict(row)
            text = row.get('fact') or ''
            items = BOUNDARY.split(text) if text.strip() else []
            kept = []
            for item_index, item in enumerate(items):
                stripped = item.strip()
                rule = index.get((domain, sha(stripped)))
                classification = 'D'
                if rule is not None:
                    assert rule['pattern'] == stripped, 'rule text mismatch'
                    classification = rule['classification']
                masked = rule is not None and rule['classification'] == 'B'
                inventory.append(dict(
                    domain=domain, source_file=str(source), source_sha256=sha(raw),
                    original_row_index_zero_based=position, csv_record_line_one_based=position + 2,
                    physical_line_start=first_line, physical_line_end=last_line,
                    sorted_row_index_zero_based=order[position],
                    split=('train' if order[position] < train_end else
                           'val' if order[position] < val_end else 'test'),
                    field='fact', item_index_zero_based=item_index, text=stripped,
                    text_sha256=sha(stripped), complete_item_sha256=sha(stripped),
                    described_at=(re.match(r'(\d{4}-\d{2}-\d{2}):', stripped).group(1)
                                  if re.match(r'(\d{4}-\d{2}-\d{2}):', stripped) else row.get('end_date')),
                    record_start=row.get('start_date'), record_end=row.get('end_date') or row.get('date'),
                    classification=classification,
                    source_urls=rule['source_urls'] if rule else [],
                    evidence=rule['evidence'] if rule else 'No source/body availability proof; screening did not flag a later-dated fact. Not certified.',
                    version_limit=rule['version_limit'] if rule else 'Individual raw search URL/body history is absent from the public processed CSV.',
                    decision_rule_id=rule['id'] if rule else None,
                    action='mask_item' if masked else 'keep',
                    later_fact_year=rule.get('later_fact_year') if rule else None))
                counts['items'] += 1
                counts[classification] += 1
                if masked:
                    counts['masked_items'] += 1
                else:
                    kept.append(item)
            new['fact'] = text if len(kept) == len(items) else '; '.join(kept)
            output.append(new)
            counts['rows'] += 1
            counts['original_nonempty'] += bool(text.strip())
            counts['nonempty'] += bool(new['fact'].strip())
            counts['modified'] += new['fact'] != text
            counts['emptied'] += bool(text.strip()) and not new['fact'].strip()
        # Use the package's own file naming (Health_US -> Health.csv).
        target = data_root / DOMAIN_FILES[domain]
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0][3]))
            writer.writeheader()
            writer.writerows(output)
        for (_, _, _, before), after in zip(rows, output):
            assert {k: v for k, v in before.items() if k != 'fact'} == \
                   {k: v for k, v in after.items() if k != 'fact'}, 'non-fact field changed'
        v0_sha = sha(raw)
        v1_sha = sha(target.read_bytes())
        invariance.append(dict(domain=domain, version='V1', rows=n, v0_sha256=v0_sha,
                               v1_sha256=v1_sha, only_fact_column_changed=True,
                               text_identical_to_v0=v0_sha == v1_sha))
        coverage.append(dict(domain=domain, rows=counts['rows'], items=counts['items'],
                             A=counts['A'], B=counts['B'], C=counts['C'], D=counts['D'],
                             masked_items=counts['masked_items'], modified_rows=counts['modified'],
                             emptied_rows=counts['emptied'],
                             original_nonempty_rows=counts['original_nonempty'],
                             nonempty_rows_after=counts['nonempty'],
                             item_retention=1 - counts['masked_items'] / max(counts['items'], 1)))
    save(REPORT / 'text_inventory_all9.json', inventory)
    save(REPORT / 'coverage_all9.json', coverage)
    save(REPORT / 'data_field_invariance_all9.json', invariance)

    # The three previously frozen V1 tables must reproduce byte-for-byte.
    reproduced = {}
    for domain in ['Agriculture', 'Economy', 'Security']:
        frozen = (FROZEN_THREE / (domain + '.csv')).read_bytes()
        built = (data_root / (domain + '.csv')).read_bytes()
        reproduced[domain] = dict(frozen_sha256=sha(frozen), built_sha256=sha(built),
                                  identical=frozen == built)
        assert frozen == built, 'frozen V1 not reproduced for ' + domain
    save(REPORT / 'frozen_v1_reproduction.json', reproduced)

    print(json.dumps(dict(
        rules_sha256=rules_sha,
        masked_items_total=sum(c['masked_items'] for c in coverage),
        per_domain={c['domain']: dict(items=c['items'], A=c['A'], B=c['B'], C=c['C'], D=c['D'],
                                      masked=c['masked_items'], modified_rows=c['modified_rows'],
                                      retention=round(c['item_retention'], 6)) for c in coverage},
        frozen_reproduced=all(v['identical'] for v in reproduced.values())), ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
