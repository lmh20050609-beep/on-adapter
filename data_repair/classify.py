"""Deterministic first-pass classification of screening candidates.

Rule (frozen before any model output is inspected):

* described_at = the independent dated item prefix, else the record end date.
* For every sentence in the item that mentions a calendar year later than
  described_at's year, decide whether that sentence states a *realised* fact
  dated at or after that year, or whether it states a forward-looking
  prediction/plan/goal/projection.
* An item is a B candidate when at least one later-year sentence is realised
  and no forward marker governs that sentence.
* Items whose later-year sentences are all forward-looking are A/C candidates.
* Items with neither are not candidates at all.

This is a screen only: it produces candidates for human/agent source review.
It never certifies A and never reads model results.
"""
import json
import re

FORWARD = re.compile(
    r'\b('
    r'predict\w*|forecast\w*|project(?:ed|ion|ions|ing)?|expect\w*|anticipat\w*|'
    r'plan(?:s|ned|ning)?|goal\w*|target\w*|aim(?:s|ed)?|intend\w*|hope[ds]?|'
    r'scheduled|schedules|propose[ds]?|proposing|vision|outlook|scenario\w*|'
    r'estimate[ds]?|estimat\w*|likely to|slated|due to be|set to|on track|'
    r'will\b|would\b|shall\b|to be\b|timetable|trajectory|roadmap|'
    r'is expected|are expected|were expected|is anticipated|are anticipated|'
    r'by the year|by 20\d\d|through 20\d\d|envision\w*|prospect\w*'
    r')', re.I)

REALISED = re.compile(
    r'\b('
    r'was|were|has been|have been|had been|reached|stood at|stood|'
    r'rose|fell|increased to|decreased to|declined to|grew by|'
    r'published|released|enacted|adopted|approved|signed|issued|'
    r'opened|closed|began|ended|appointed|elected|established|announced|'
    r'reported|recorded|observed|measured|counted|found|showed|listed|'
    r'occurred|happened|took place|won|died|launched|implemented|'
    r'as of|effective|valid from|remains|is now|are now'
    r')\b', re.I)

SENTENCE = re.compile(r'(?<=[.!?])\s+|\n+')
YEAR = re.compile(r'\b(19|20)\d{2}\b')


def sentences(text):
    return [s.strip() for s in SENTENCE.split(text) if s.strip()]


def year_of(value):
    m = re.match(r'\s*(\d{4})', str(value or ''))
    return int(m.group(1)) if m else None


def classify_item(described_at, body):
    base = year_of(described_at)
    if base is None:
        return dict(verdict='not_candidate', later_sentences=[], reason='no described year')
    later, forward_only, realised = [], [], []
    for sentence in sentences(body):
        years = [int(y) for y in re.findall(r'\b((?:19|20)\d{2})\b', sentence)]
        later_years = [y for y in years if y > base]
        if not later_years:
            continue
        later.append(sentence)
        has_forward = bool(FORWARD.search(sentence))
        has_realised = bool(REALISED.search(sentence))
        if has_forward and not has_realised:
            forward_only.append(sentence)
        elif has_realised:
            realised.append(sentence)
        else:
            # A bare later-year assertion with neither marker stays ambiguous.
            forward_only.append(sentence)
    if realised:
        verdict = 'B_candidate'
    elif later:
        verdict = 'AC_candidate'
    else:
        verdict = 'not_candidate'
    return dict(verdict=verdict, later_sentences=later,
                forward_only_sentences=forward_only, realised_sentences=realised)


def load_entries(path):
    with open(path, encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


if __name__ == '__main__':
    import collections
    import pathlib
    base = pathlib.Path(__file__).resolve().parent
    entries = load_entries(base / 'source_evidence/all_text_entries.jsonl')
    counts = collections.Counter()
    checked = 0
    for entry in entries:
        if entry['domain'] in {'Agriculture', 'Economy', 'Security', 'Energy'}:
            continue
        if not entry['screening_only_future_years']:
            continue
        checked += 1
        result = classify_item(entry['described_at'], entry['text'])
        counts[(entry['domain'], result['verdict'])] += 1
    for key in sorted(counts):
        print(key, counts[key])
    print('checked', checked)
