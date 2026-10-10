"""Read-only prerequisites and a bounded engineering entrypoint.

This runner has no offline-training or formal-scoring command.
"""
import argparse
import copy
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch
from .config import ROOT
from .resources import digest
from .data import load_series,ForecastWindows

REPORT=ROOT/'reports/encoding_independent_v2_20261008'
OLD=ROOT/'reports/benchmark_text_pair_v1_20261008'
RUNS=Path('E:/FedNoise/iadapter-baselines/runs/benchmark_text_pair_v1_20261008')
DOMAINS=['Agriculture','Economy','Security']
VERSIONS=['V0','V1','V2']
def seeded():
    torch.set_num_threads(2)
    random.seed(2025);np.random.seed(2025);torch.manual_seed(2025)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
def same_array(a,b):return np.array_equal(a,b,equal_nan=True) if a.dtype.kind in 'fc' else np.array_equal(a,b)
def get_cell(domain,version):
    return next(x for x in json.loads((REPORT/'source_frozen_matrix.json').read_text('utf-8'))['cells'] if x['domain']==domain and x['version']==version)
def data_for(domain,version):
    data=load_series(domain,data_root=REPORT/'data'/version,cache_root=None)
    data.text_version=version
    data.source_decisions_sha256=digest(OLD/'source_evidence/decision_rules.json')
    data._fixed_origin_grid={(s,8,12):ForecastWindows(data,s,8,12).indices.copy() for s in ['train','val','test']}
    return data
def settings(domain,version):
    cfg=copy.deepcopy(get_cell(domain,version)['resolved_settings'])
    cfg['active_method']='On-Adapter'
    cfg['execution']=copy.deepcopy(cfg['execution'])
    cfg['execution']['cache_root']=str(Path('E:/FedNoise/textenc_v2_20261008')/version/domain)
    path=RUNS/domain/'numeric/best.pt'
    cfg['_pair_numeric_checkpoint']=str(path);cfg['_pair_numeric_sha256']=digest(path)
    return cfg
def state_hash(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def verify_lock():
    for name,wanted in json.loads((REPORT/'execution_lock.json').read_text('utf-8')).items():
        if digest(name)!=wanted:raise ValueError('Independent-stage frozen input changed: '+name)
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['check'])
    parser.parse_args()
    from benchmark_text_encoding_v2.engineering import main as run
    run()
if __name__=='__main__':main()
