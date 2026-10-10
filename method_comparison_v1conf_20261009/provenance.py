"""Separate result lineage; legacy artifacts are read-only verified dependencies."""
import json
from pathlib import Path
from .config import ROOT
from .resources import digest, identity


def fingerprint():
    paths = list(Path(__file__).parent.glob('*.py')) + [Path(__file__).parent/'FORK.json']
    paths += [ROOT/'original-paper-replay/uniform-l36-source'/n for n in ['adapt_z_multimodal.py','models/text_alignment.py']]
    paths += list((ROOT/'external').glob('*/SOURCE.json')) + list((ROOT/'weights').glob('*/SOURCE.json'))
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


class PriorIndex:
    """L36 artifacts stay read-only and never count in this new L/H protocol."""
    def __init__(self, settings):
        from .config import configured_windows
        configured_windows(settings)
        self.cells={}; self.legacy_count=0
        self.v1_keys=set(); self.v2_keys=set(); self.v3_keys=set()
    def references(self): return []

def verify_v1(folder, settings):
    raise ValueError('L36 result reuse is forbidden in the periodic protocol')

LEGACY_MANIFEST = ROOT / 'benchmark_textrepair_v1/legacy_numeric_reuse_manifest.json'


def legacy_manifest():
    if not LEGACY_MANIFEST.is_file():
        raise FileNotFoundError('Frozen legacy numeric reuse manifest is required')
    return json.loads(LEGACY_MANIFEST.read_text(encoding='utf-8'))


def numeric_checkpoint(settings, domain, horizon, seed):
    """Frozen shared numeric backbone for the text-consuming methods.

    The four text-free methods (iTransformer, ADAPT-Z, DSOF, Proceed) never read
    `fact`, so their completed cells are reusable once the numeric path, the
    columns/splits/masks and the selected checkpoint are verified unchanged.
    Each reuse is pinned by the frozen manifest and re-verified here.
    """
    manifest = legacy_manifest()
    key = '%s/H%d/iTransformer/seed%d' % (domain, horizon, seed)
    entry = manifest['cells'].get(key)
    if entry is None:
        raise FileNotFoundError('No verified legacy numeric backbone for ' + key)
    folder = Path(entry['directory'])
    result = json.loads((folder/'result.json').read_text('utf-8'))
    meta = json.loads((folder/'config.json').read_text('utf-8'))
    if digest(folder/'result.json') != entry['result_sha256']:
        raise ValueError('Legacy numeric result changed: ' + key)
    if digest(folder/'best.pt') != entry['best_pt_sha256']:
        raise ValueError('Legacy numeric checkpoint changed: ' + key)
    if result.get('status') != 'completed' or result.get('method') != 'iTransformer':
        raise ValueError('Legacy numeric cell is not a completed iTransformer result')
    if (meta['domain'], meta['horizon'], meta['seed'], meta['lookback']) != (domain, horizon, seed, entry['lookback']):
        raise ValueError('Legacy numeric cell identity mismatch: ' + key)
    if result['artifacts'].get('best.pt') != entry['best_pt_sha256']:
        raise ValueError('Legacy numeric checkpoint disagrees with its own result manifest')
    if digest(meta['data']['source']) != meta['data_sha256']:
        raise ValueError('Legacy numeric dependency data changed: ' + key)
    return folder/'best.pt'

def resolve_settings(settings, method, rows):
    cfg = dict(settings)
    rules = ['base configuration']
    if rows < 500:
        cfg.update(d_model=32, dropout=.2, batch_size=8, epochs=50); rules.append('legacy rows<500 rule')
    if method in {'ADAPT-Z','On-Adapter'}:
        cfg['epochs']=3; rules.append('legacy adapter epochs=3')
    if method == 'On-Adapter' and (settings.get('on_variant') or {}).get('epochs') is not None:
        cfg['epochs']=int(settings['on_variant']['epochs']); rules.append('v1conf variant epochs=%d' % cfg['epochs'])
    if method == 'GPT4MTS':
        cfg.update(epochs=10,batch_size=4); rules.append('legacy GPT4MTS epochs=10,batch=4')
    if method == 'MM-TSFlib':
        cfg['batch_size']=8; rules.append('legacy MM-TSFlib batch=8')
    if cfg.get('num_workers',0) != 0:
        raise ValueError('Nonzero loader workers require a separately validated performance configuration')
    cfg['effective_execution'] = dict(batch_size=cfg['batch_size'],epochs=cfg['epochs'],threads=cfg['threads'],
        num_workers=0,pin_memory=False,precision='float32',rules=rules,
        ignored_legacy_fields=['minimum_free_gpu_mib (no historical scheduler gate)'],
        gradient_accumulation=1, training_budget_changed=False)
    return cfg
