import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Frozen nine-domain minimal text repair (V1). Only the `fact` column differs
# from the original distribution; every numeric field is byte-identical.
REPAIR_REPORT = ROOT / 'reports/benchmark_text_repair_all9_v1_20261008'
DATA_ROOT = REPAIR_REPORT / 'data/V1'
TEXT_VERSION = 'V1'
DECISION_RULES = ROOT / 'benchmark_text_repair_all9_v1/source_evidence/decision_rules_all9.json'
DOMAIN_FILES = {'Agriculture': 'Agriculture.csv', 'Climate': 'Climate.csv',
                'Economy': 'Economy.csv', 'Energy': 'Energy.csv',
                'Environment': 'Environment.csv', 'Health_US': 'Health.csv',
                'Security': 'Security.csv', 'SocialGood': 'SocialGood.csv',
                'Traffic': 'Traffic.csv', 'Weather': 'new_metr.csv'}

APPROVED_TABLE = {'Environment': {'periodicity': 'daily', 'lookback': 96, 'horizons': [48, 96, 192, 336]}, 'Energy': {'periodicity': 'weekly', 'lookback': 24, 'horizons': [12, 24, 36, 48]}, 'Health_US': {'periodicity': 'weekly', 'lookback': 24, 'horizons': [12, 24, 36, 48]}, 'Traffic': {'periodicity': 'monthly', 'lookback': 8, 'horizons': [6, 8, 10, 12]}, 'SocialGood': {'periodicity': 'monthly', 'lookback': 8, 'horizons': [6, 8, 10, 12]}, 'Security': {'periodicity': 'monthly', 'lookback': 8, 'horizons': [6, 8, 10, 12]}, 'Economy': {'periodicity': 'monthly', 'lookback': 8, 'horizons': [6, 8, 10, 12]}, 'Climate': {'periodicity': 'monthly', 'lookback': 8, 'horizons': [6, 8, 10, 12]}, 'Agriculture': {'periodicity': 'monthly', 'lookback': 8, 'horizons': [6, 8, 10, 12]}}
WINDOWS = {d: (v["lookback"], v["horizons"]) for d, v in APPROVED_TABLE.items()}
METHODS = {'iTransformer': ('tslib','static',2024), 'ADAPT-Z': ('adaptz','online',2025), 'DSOF': ('dsof','online',2025), 'Proceed': ('proceed','online',2025), 'On-Adapter': ('local','online',2026), 'GPT4MTS': ('gpt4mts','static',2024), 'MM-TSFlib': ('mmtslib','static',2024)}

# Methods whose forward pass consumes the frozen `fact` text table. The other
# four methods are provably text-free, so their historical cells are reusable.
TEXT_METHODS = ['On-Adapter', 'GPT4MTS', 'MM-TSFlib']
TEXT_FREE_METHODS = ['iTransformer', 'ADAPT-Z', 'DSOF', 'Proceed']


def text_version(settings):
    version = settings.get('text_version')
    if version != TEXT_VERSION:
        raise ValueError('The main campaign runs the V1 text table only')
    return version


def source_decisions_sha256():
    import hashlib
    return hashlib.sha256(DECISION_RULES.read_bytes()).hexdigest()


def configured_windows(settings):
    table = settings.get('dataset_windows')
    if table != APPROVED_TABLE or settings.get('datasets') != list(APPROVED_TABLE):
        raise ValueError('Dataset L/H/periodicity does not match the user-approved nine-domain table')
    if settings.get('protocol_version') != 'multivariate-origin-periodic-seeds3-v1':
        raise ValueError('Wrong dataset protocol')
    if settings.get('expected_matrix_cells') != 756:
        raise ValueError('Wrong periodic matrix size')
    if settings['seeds'] != [2025,2026,2027]:
        raise ValueError('Seed budget changed')
    if settings.get('methods') != TEXT_METHODS:
        raise ValueError('The main campaign trains exactly the three text-consuming methods')
    if any(settings['execution'].get('resume_'+v+'_cells',False) for v in ['v1','v2','v3','v4']):
        raise ValueError('Old L36 recovery cannot be migrated to this protocol')
    if settings.get('prior_campaign_reuse', False):
        raise ValueError('Old text-version scores cannot count toward this table')
    text_version(settings)
    return {d: (v['lookback'], list(v['horizons'])) for d,v in table.items()}


def load_config(path=None):
    source = Path(path) if path else ROOT/'configs/method_comparison_textrepair_v1_20261008.json'
    settings = json.loads(source.read_text(encoding='utf-8-sig'))
    configured_windows(settings)
    return settings

def save_json(path, value):
    from .state_io import atomic_json
    atomic_json(path, value)
