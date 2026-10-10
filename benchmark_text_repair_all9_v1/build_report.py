"""Assemble the nine-domain repair report from the frozen artifacts."""
import collections
import csv
import json
import pathlib

BASE = pathlib.Path(__file__).resolve().parent
ROOT = BASE.parent
REPORT = ROOT / 'reports/benchmark_text_repair_all9_v1_20261008'
DOMAINS = ['Environment', 'Energy', 'Health_US', 'Traffic', 'SocialGood',
           'Security', 'Economy', 'Climate', 'Agriculture']


def load(name):
    return json.loads((REPORT / name).read_text(encoding='utf-8'))


def main():
    inventory = load('text_inventory_all9.json')
    coverage = {row['domain']: row for row in load('coverage_all9.json')}
    invariance = {row['domain']: row for row in load('data_field_invariance_all9.json')}
    reproduction = load('frozen_v1_reproduction.json')
    summary = json.loads((BASE / 'source_evidence/decision_summary.json').read_text(encoding='utf-8'))
    effects = {row['domain']: row for row in load('token_effects.json')}

    per_split = collections.defaultdict(collections.Counter)
    for item in inventory:
        per_split[(item['domain'], item['split'])]['items'] += 1
        per_split[(item['domain'], item['split'])][item['classification']] += 1
        per_split[(item['domain'], item['split'])]['masked'] += item['action'] == 'mask_item'

    lines = []
    lines.append('# Nine-domain minimal text repair (V1): data and source report\n')
    lines.append('2026-10-08. Goal: with numeric data, splits, targets, masks and feedback rules unchanged, '
                 'extend the minimal repair of confirmed text timing mismatches to all 9 benchmark domains '
                 'and freeze the full experimental protocol.\n')
    lines.append('**V1 masks only individual entries confirmed to postdate the time they describe**; '
                 'entries graded C/D by source are always kept and disclosed, '
                 'and a screening miss does not mean certified. This report reads no model scores.\n')

    lines.append('## 1. Text fields the models actually read\n')
    lines.append('| Method | Reads `fact` | Input path |\n|---|---|---|')
    lines.append('| On-Adapter | yes | `online.ResidualOnline` via `data.prepared_text.pooled(origin)`, '
                 'concatenation of visible rows (origin-1, origin), 64 tokens, mean pooling |')
    lines.append('| GPT4MTS | yes | `multimodal.GPTForecast` via `text.gpt(origin, lookback)`, 512 tokens, cls |')
    lines.append('| MM-TSFlib | yes | `multimodal.MMD` via `text.tokens(origin)`, 6 GPT2 layers, 256 tokens |')
    lines.append('| iTransformer / ADAPT-Z / DSOF / Proceed | no | The code path does not access `data.texts` / `prepared_text` |')
    lines.append('\nNo method reads `preds`, `Final_Output` or `Final_Search_*`.\n')

    lines.append('## 2. Source grading and masking\n')
    lines.append('Rule file `source_evidence/decision_rules_all9.json`, SHA256 `%s`; '
                 '%d frozen rules inherited from the 4-domain pilot (SHA256 `%s`), '
                 'plus %d new rules from independent review of 5 domains.\n'
                 % (summary['sha256'], summary['inherited_rules'],
                    summary['inherited_sha256'], summary['new_rules']))
    lines.append('| Domain | Individual entries | A | B (masked) | C (kept, disclosed) | D (kept, unverified) '
                 '| Masked share of domain | Modified rows |\n'
                 '|---|---|---|---|---|---|---|---|')
    for domain in DOMAINS:
        row = coverage[domain]
        lines.append('| %s | %d | %d | %d | %d | %d | %.3f%% | %d |'
                     % (domain, row['items'], row['A'], row['B'], row['C'], row['D'],
                        100 * row['masked_items'] / row['items'], row['modified_rows']))
    total_items = sum(coverage[d]['items'] for d in DOMAINS)
    total_masked = sum(coverage[d]['masked_items'] for d in DOMAINS)
    lines.append('| **Total** | %d | %d | %d | %d | %d | %.3f%% | %d |'
                 % (total_items, sum(coverage[d]['A'] for d in DOMAINS),
                    sum(coverage[d]['B'] for d in DOMAINS), sum(coverage[d]['C'] for d in DOMAINS),
                    sum(coverage[d]['D'] for d in DOMAINS),
                    100 * total_masked / total_items,
                    sum(coverage[d]['modified_rows'] for d in DOMAINS)))
    lines.append('\nMasking counts entry occurrences; when the same raw text repeats across rows, '
                 'it is masked separately in each row. No numeric row, forecast origin or scoring target was deleted.\n')

    lines.append('## 3. Coverage per domain and split\n')
    lines.append('| Domain | Split | Entries | A | B | C | D | Masked | Entry retention |\n|---|---|---|---|---|---|---|---|---|')
    for domain in DOMAINS:
        for split in ['train', 'val', 'test']:
            cell = per_split[(domain, split)]
            if not cell['items']:
                continue
            lines.append('| %s | %s | %d | %d | %d | %d | %d | %d | %.4f |'
                         % (domain, split, cell['items'], cell['A'], cell['B'], cell['C'], cell['D'],
                            cell['masked'], 1 - cell['masked'] / cell['items']))

    lines.append('\n## 4. Actual token changes\n')
    lines.append('Using each method\'s frozen tokenizer, token limit and text mapping rule, '
                 'the actual model input of every row is compared between V0 and V1.\n')
    lines.append('| Domain | Method | Rows containing a masked entry | Of those, masked entry inside the token window '
                 '| Rows with changed tokens | Test origins | Test origins with changed tokens |\n'
                 '|---|---|---|---|---|---|---|')
    for domain in DOMAINS:
        for method in ['On-Adapter', 'GPT4MTS', 'MM-TSFlib']:
            m = effects[domain]['methods'][method]
            lines.append('| %s | %s | %d | %d | %d | %d | %d |'
                         % (domain, method, m['rows_whose_raw_cell_contains_a_masked_item'],
                            m['of_those_rows_the_masked_item_survives_into_the_token_window'],
                            m['rows_with_changed_tokens'],
                            m['per_split']['test']['origins'],
                            m['per_split']['test']['origins_with_changed_tokens']))
    lines.append('\nIn most domains the 64-token window of On-Adapter does not contain the masked entry itself; '
                 'its token changes come mainly from the trailing 1200-character window of `clean_fact` shifting '
                 'after the deletion. The 256-token window of MM-TSFlib keeps a sizeable share of the masked entries.\n')

    lines.append('## 5. Identity and reproduction\n')
    lines.append('| Domain | V1 file SHA256 | Non-fact columns identical |\n|---|---|---|')
    for domain in DOMAINS:
        row = invariance[domain]
        lines.append('| %s | `%s` | %s |' % (domain, row['v1_sha256'][:32] + '…',
                                            row['only_fact_column_changed']))
    lines.append('\nV1 of Agriculture/Economy/Security is byte-identical to the frozen three-domain pilot files: %s.\n'
                 % all(v['identical'] for v in reproduction.values()))

    lines.append('## 6. Limitations that must be kept\n')
    lines.append('- Source review covers only candidate entries whose body mentions a year later than the described time; '
                 'a miss does not mean there is no problem, nor that the entry is certified A.')
    lines.append('- All C/D entries are kept in the actual data and only disclosed; V1 does not mean "every entry is certified".')
    lines.append('- The numeric data still use the original `end_date` availability proxy and the retrospective '
                 'target-mask origin grid; this does not prove full causality in the sense of real publication times '
                 'and historical revisions.')
    lines.append('- Source decisions read no model scores; masking rules are not adjusted based on results.')
    lines.append('- This stage does not extend to other seeds or to a SocialGood main table; SocialGood V1 is included '
                 'in the 9-domain data, but no separate extra pilot is run for it.')
    (REPORT / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('report written', REPORT / 'REPORT.md')


if __name__ == '__main__':
    main()
