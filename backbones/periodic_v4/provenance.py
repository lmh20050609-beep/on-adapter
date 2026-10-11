"""Verified same-protocol reuse; L36 artifacts remain excluded."""
import json
from pathlib import Path
from .config import ROOT, configured_windows
from .resources import digest, identity
from .state_io import read_json

PARENT_CONFIG = ROOT / 'configs/method_comparison_periodic_v3_20261006.json'


def protocol(value):
    return {k:v for k,v in value.items() if k not in
            {'execution','campaign','run_kind','_settings_input'}}


def parent_settings(settings):
    old=read_json(PARENT_CONFIG)
    configured_windows(old); configured_windows(settings)
    if protocol(old)!=protocol(settings):
        raise ValueError('Parent periodic experiment definition changed')
    return old


def fingerprint():
    paths=list(Path(__file__).parent.glob('*.py'))+[Path(__file__).parent/'FORK.json']
    paths+=list((ROOT/'method_comparison_periodic_v3').glob('*.py'))
    paths+=[ROOT/'method_comparison_periodic_v3/FORK.json',PARENT_CONFIG]
    paths+=[ROOT/'original-paper-replay/uniform-l36-source'/n for n in
            ['adapt_z_multimodal.py','models/text_alignment.py']]
    paths+=list((ROOT/'external').glob('*/SOURCE.json'))+list((ROOT/'weights').glob('*/SOURCE.json'))
    return {str(p.relative_to(ROOT)):digest(p) for p in paths}


def verify_parent(folder, settings, artifacts=True):
    from method_comparison_periodic_v3.provenance import fingerprint as original_fingerprint
    old=parent_settings(settings)
    folder=Path(folder);meta=read_json(folder/'config.json');result=read_json(folder/'result.json')
    if meta['settings_input']!=old or meta['source_fingerprint']!=original_fingerprint():
        raise ValueError('Parent periodic settings/source mismatch')
    L,hs=configured_windows(old)[result['domain']]
    for value in [meta,result]:
        if (value['lookback']!=L or value['horizon'] not in hs or
            value['protocol_version']!=old['protocol_version'] or
            value['execution_version']!='periodic-v3'):
            raise ValueError('Parent result has incompatible L/H/protocol')
    if any(meta[k]!=result[k] for k in ['domain','horizon','seed','method','lookback']):
        raise ValueError('Parent result identity mismatch')
    if result['status']!='completed' or digest(meta['data']['source'])!=meta['data_sha256']:
        raise ValueError('Parent data/result changed')
    if artifacts:
        for name,expected in result['artifacts'].items():
            if digest(folder/name)!=expected:raise ValueError('Parent artifact changed: '+name)
    return result,meta


class PriorIndex:
    def __init__(self, settings):
        configured_windows(settings)
        self.cells={};self.legacy_count=0
        self.v1_keys=set();self.v2_keys=set();self.v3_keys=set()
        if not settings['execution'].get('reuse_periodic_v3',False):return
        from .parallel_policy import job_key
        old=parent_settings(settings)
        for path in (Path(old['runs_root'])/old['campaign']).glob('*/*/*/*/result.json'):
            result,_=verify_parent(path.parent,settings);key=job_key(result)
            if key in self.cells:raise ValueError('Duplicate parent periodic result')
            self.cells[key]=dict(domain=result['domain'],horizon=result['horizon'],
                lookback=result['lookback'],seed=result['seed'],method=result['method'],
                directory=str(path.parent),result=str(path),sha256=digest(path),
                lineage='periodic-v3',protocol_version=result['protocol_version'])
            self.v1_keys.add(key)
    def references(self):return list(self.cells.values())


def verify_v1(folder, settings):
    raise ValueError('Uniform-L36 reuse remains forbidden')


def numeric_checkpoint(settings,domain,horizon,seed):
    original=settings.get('_settings_input',settings)
    if seed not in original['seeds']:
        raise ValueError('Numerical prerequisite uses an unapproved seed')
    L,hs=configured_windows(original)[domain]
    if horizon not in hs:raise ValueError('Prerequisite outside periodic matrix')
    folder=Path(original['runs_root'])/original['campaign']/domain/('H%d'%horizon)/'iTransformer'/('seed%d'%seed)
    if not (folder/'result.json').exists():
        old=parent_settings(original)
        folder=Path(old['runs_root'])/old['campaign']/domain/('H%d'%horizon)/'iTransformer'/('seed%d'%seed)
        result,meta=verify_parent(folder,original,artifacts=False)
    else:
        result=read_json(folder/'result.json');meta=read_json(folder/'config.json')
        if meta['settings_input']!=original or meta['source_fingerprint']!=fingerprint():
            raise ValueError('Current dependency settings/source mismatch')
        if result['execution_version']!='periodic-v4':raise ValueError('Dependency execution lineage mismatch')
        if digest(meta['data']['source'])!=meta['data_sha256']:raise ValueError('Dependency data changed')
    expected=(domain,horizon,seed,'iTransformer',L)
    for value in [meta,result]:
        if tuple(value[k] for k in ['domain','horizon','seed','method','lookback'])!=expected:
            raise ValueError('Dependency has incompatible L/H/job')
    if digest(folder/'best.pt')!=result['artifacts']['best.pt']:
        raise ValueError('Dependency checkpoint changed')
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
