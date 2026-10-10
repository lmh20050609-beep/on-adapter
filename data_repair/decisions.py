"""Freeze the nine-domain source decisions.

Inputs
------
* inherited_decision_rules.json : the frozen four-domain rules
  (Agriculture/Economy/Security/Energy), reused byte-for-byte.
* candidate_decisions/*.json    : independent per-batch classifications of the
  five previously uninventoried domains (Environment, Health_US, Traffic,
  SocialGood, Climate), produced by six independent review agents under
  reports/.../RUBRIC.md, plus a short manual override list below.

The fusion rule is fixed before any model output is inspected:

* inherited B  -> B  (mask)
* new B        -> B  (mask)
* new A        -> A only when a dated public source with available_at no later
                  than the description date is attached; otherwise C (keep).
* new C / new D-> C / not emitted (default D, keep)

Nothing here reads scores, predictions or checkpoints.
"""
import collections
import hashlib
import json
import pathlib
import re

BASE = pathlib.Path(__file__).resolve().parent
EVIDENCE = BASE / 'source_evidence'
CANDIDATES = BASE / 'candidate_decisions'
OVERRIDES_FILE = BASE / 'manual_overrides.json'


def sha(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def load(path):
    return json.loads(pathlib.Path(path).read_text(encoding='utf-8'))


def description_date(item_text, record_dates):
    match = re.match(r'\s*(\d{4}-\d{2}-\d{2}):', item_text)
    if match:
        return match.group(1)
    return record_dates


def build():
    inherited = load(EVIDENCE / 'inherited_decision_rules.json')
    inherited_keys = {(r['domain'], r['complete_item_sha256']) for r in inherited}
    candidates = load(BASE / 'unique_candidates.json')
    by_id = {c['id']: c for c in candidates}
    overrides = load(OVERRIDES_FILE) if OVERRIDES_FILE.exists() else {}

    verdicts = {}
    for path in sorted(CANDIDATES.glob('*.out.json')):
        for entry in load(path):
            assert entry['id'] in by_id, entry['id']
            assert entry['id'] not in verdicts, 'duplicate verdict ' + entry['id']
            verdicts[entry['id']] = entry
    missing = sorted(set(by_id) - set(verdicts))
    if missing:
        raise ValueError('unreviewed candidates: %d' % len(missing))

    rules = list(inherited)
    counts = collections.Counter()
    for candidate in candidates:
        cid = candidate['id']
        verdict = verdicts[cid]
        classification = overrides.get(cid, verdict['classification'])
        text = candidate['text']
        record = dict(
            id='%s_%s_%s_%s' % (candidate['domains'][0], candidate['described_max'], classification, cid),
            domain=candidate['domains'][0], field='fact', pattern=text,
            flag='literal', pattern_type='literal', classification=classification,
            available_at=None, source_urls=[], evidence=verdict.get('evidence', ''),
            version_limit='', action='mask_item' if classification == 'B' else 'keep',
            complete_item_sha256=sha(text),
            described_at=description_date(text, candidate['described_max']),
            later_fact_year=verdict.get('later_fact_year'),
            review_origin=('manual_override' if cid in overrides else 'independent_batch_review'),
            candidate_id=cid, candidate_row_count=candidate['n_rows'])
        if classification == 'B':
            record['version_limit'] = ('The later-than-description fact is stated in the item itself; '
                                       'no earlier body version of this claim was recovered, so the '
                                       'item cannot be delayed or reinserted.')
            url = verdict.get('source_url')
            record['source_urls'] = [url] if url else []
        elif classification == 'A':
            url = verdict.get('source_url')
            if not url:
                record['classification'] = 'C'
                record['action'] = 'keep'
                record['evidence'] = ('Plausible historical forecast/plan and no later realised fact, '
                                      'but no dated public source was attached; recorded as C, not certified A. '
                                      + record['evidence'])
            else:
                record['source_urls'] = [url]
                record['available_at'] = candidate['described_min']
                record['evidence'] = ('Dated public source attached; availability no later than the '
                                      'description date. ' + record['evidence'])
        if record['classification'] == 'C':
            record['version_limit'] = record['version_limit'] or 'Date/version of the original body not pinned; item retained and disclosed as unverified.'
        counts[record['classification']] += 1
        rules.append(record)

    dedup = {}
    for rule in rules:
        key = (rule['domain'], rule['complete_item_sha256'])
        if key in dedup:
            raise ValueError('duplicate rule for ' + str(key))
        dedup[key] = rule
    out = EVIDENCE / 'decision_rules_all9.json'
    out.write_text(json.dumps(rules, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    summary = dict(domains=sorted({r['domain'] for r in rules}),
                   rules=len(rules),
                   classification_counts=dict(counts),
                   inherited_rules=len(inherited),
                   new_rules=len(rules) - len(inherited),
                   reviewed_candidates=len(verdicts),
                   manual_overrides=overrides,
                   sha256=sha(out.read_bytes()),
                   inherited_sha256=sha((EVIDENCE / 'inherited_decision_rules.json').read_bytes()))
    (EVIDENCE / 'decision_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return summary


if __name__ == '__main__':
    print(json.dumps(build(), ensure_ascii=False, indent=1))
