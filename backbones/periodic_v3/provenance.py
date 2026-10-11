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

def numeric_checkpoint(settings, domain, horizon, seed):
    from .config import configured_windows
    original=settings.get('_settings_input',settings)
    if seed not in original['seeds']:
        raise ValueError('Numerical prerequisite uses an unapproved seed')
    L, horizons=configured_windows(original)[domain]
    if horizon not in horizons: raise ValueError('Numerical prerequisite outside periodic matrix')
    folder=Path(original['runs_root'])/original['campaign']/domain/('H%d'%horizon)/'iTransformer'/('seed%d'%seed)
    if not (folder/'result.json').is_file():
        raise FileNotFoundError('Matching new-L numerical prerequisite has not completed')
    meta=json.loads((folder/'config.json').read_text('utf-8'))
    result=json.loads((folder/'result.json').read_text('utf-8'))
    expected=(domain,horizon,seed,'iTransformer',L)
    for value in [meta,result]:
        actual=tuple(value[k] for k in ['domain','horizon','seed','method','lookback'])
        if actual!=expected: raise ValueError('Numerical dependency has a different L/H/job')
    if meta['settings_input']!=original or meta['source_fingerprint']!=fingerprint():
        raise ValueError('New-L numerical dependency settings/source mismatch')
    if meta['protocol_version']!=original['protocol_version'] or result['execution_version']!='periodic-v3':
        raise ValueError('Numerical dependency protocol/version mismatch')
    if digest(meta['data']['source'])!=meta['data_sha256']:
        raise ValueError('New-L dependency data changed')
    if digest(folder/'best.pt')!=result['artifacts']['best.pt']:
        raise ValueError('New-L dependency checkpoint changed')
    return folder/'best.pt'

def resolve_settings(settings, method, rows):
    cfg = dict(settings)
    rules = ['base configuration']
    if rows < 500:
        cfg.update(d_model=32, dropout=.2, batch_size=8, epochs=50); rules.append('legacy rows<500 rule')
    if method in {'ADAPT-Z','On-Adapter'}:
        cfg['epochs']=3; rules.append('legacy adapter epochs=3')
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
