import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / 'data'   # numeric columns are byte-identical to the original Time-MMD files
DOMAIN_FILES = {'Agriculture': 'Agriculture.csv', 'Climate': 'Climate.csv',
                'Economy': 'Economy.csv', 'Energy': 'Energy.csv',
                'Environment': 'Environment.csv', 'Health_US': 'Health.csv',
                'Security': 'Security.csv', 'SocialGood': 'SocialGood.csv',
                'Traffic': 'Traffic.csv', 'Weather': 'new_metr.csv'}

APPROVED_TABLE = {'Environment': {'periodicity': 'daily', 'lookback': 96, 'horizons': [48, 96, 192, 336]}, 'Energy': {'periodicity': 'weekly', 'lookback': 24, 'horizons': [12, 24, 36, 48]}, 'Health_US': {'periodicity': 'weekly', 'lookback': 24, 'horizons': [12, 24, 36, 48]}, 'Traffic': {'periodicity': 'monthly', 'lookback': 8, 'horizons': [6, 8, 10, 12]}, 'SocialGood': {'periodicity': 'monthly', 'lookback': 8, 'horizons': [6, 8, 10, 12]}, 'Security': {'periodicity': 'monthly', 'lookback': 8, 'horizons': [6, 8, 10, 12]}, 'Economy': {'periodicity': 'monthly', 'lookback': 8, 'horizons': [6, 8, 10, 12]}, 'Climate': {'periodicity': 'monthly', 'lookback': 8, 'horizons': [6, 8, 10, 12]}, 'Agriculture': {'periodicity': 'monthly', 'lookback': 8, 'horizons': [6, 8, 10, 12]}}
WINDOWS = {d: (v["lookback"], v["horizons"]) for d, v in APPROVED_TABLE.items()}
METHODS = {'iTransformer': ('tslib','static',2024), 'ADAPT-Z': ('adaptz','online',2025), 'DSOF': ('dsof','online',2025), 'Proceed': ('proceed','online',2025), 'On-Adapter': ('local','online',2026), 'GPT4MTS': ('gpt4mts','static',2024), 'MM-TSFlib': ('mmtslib','static',2024)}

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
    if set(settings['methods']) != set(METHODS)-{'iTransformer'} or len(settings['methods']) != 6:
        raise ValueError('This prepared matrix contains six runnable comparison methods plus iTransformer')
    if any(settings['execution'].get('resume_'+v+'_cells',False) for v in ['v1','v2','v3','v4']):
        raise ValueError('Old L36 recovery cannot be migrated to this protocol')
    if settings.get('prior_campaign_reuse', False):
        raise ValueError('Old results cannot count toward the periodic table')
    if settings['execution'].get('reuse_periodic_v1',False) or settings['execution'].get('resume_periodic_v1',False):
        raise ValueError('This fresh three-seed campaign cannot reuse old results or checkpoints')
    return {d: (v['lookback'], list(v['horizons'])) for d,v in table.items()}

def load_config(path=None):
    source = Path(path) if path else ROOT/'configs/method_comparison_periodic_v4_20261007.json'
    settings = json.loads(source.read_text(encoding='utf-8-sig'))
    configured_windows(settings)
    return settings

def save_json(path, value):
    from .state_io import atomic_json
    atomic_json(path, value)
