"""Single guarded queue, domain coverage first; plan-only unless explicitly enabled."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from .config import ROOT, WINDOWS, load_config, configured_windows
from .provenance import PriorIndex, fingerprint
from .resources import atomic_json, exclusive, digest, identity
from .process_guard import require_execution, status

METHOD_ORDER=['iTransformer','ADAPT-Z','DSOF','Proceed','On-Adapter','GPT4MTS','MM-TSFlib']
DOMAIN_ORDER=['Economy','Energy','Environment','Weather','Climate','Traffic','Security','SocialGood','Agriculture','Health_US']


def matrix(settings):
    windows=configured_windows(settings)
    return [dict(domain=d,lookback=windows[d][0],periodicity=settings['dataset_windows'][d]['periodicity'],horizon=windows[d][1][h],seed=s,method=m)
            for s in settings['seeds'] for h in range(4) for m in METHOD_ORDER
            for d in DOMAIN_ORDER if d in settings['datasets']]


def key(job): return (job['domain'],job['horizon'],job['seed'],job['method'])


def cell_dir(settings,job):
    return Path(settings['runs_root'])/settings['campaign']/job['domain']/('H%d'%job['horizon'])/job['method']/('seed%d'%job['seed'])


def milestone(cells, settings):
    windows=configured_windows(settings)
    main=set(METHOD_ORDER)-{'iTransformer'};by={}
    for domain,h,s,m in cells: by.setdefault((domain,h,s),set()).add(m)
    paired={k for k,v in by.items() if main<=v}
    return {'domains':sorted({d for d,h,s in paired}),
            'full_seed_blocks':sorted((d,s) for d in windows for s in {x[2] for x in paired}
                if all((d,h,s) in paired for h in windows[d][1]))}


def plan(settings):
    legacy=PriorIndex(settings);jobs=matrix(settings)
    if len(jobs)!=settings['expected_matrix_cells'] or len({key(j) for j in jobs})!=settings['expected_matrix_cells']: raise ValueError('Approved experiment matrix changed')
    return dict(matrix=jobs,legacy_references=legacy.references(),
                pending=[j for j in jobs if key(j) not in legacy.cells],
                legacy_completed=len(legacy.cells),original_legacy_completed=legacy.legacy_count,
                prior_optimized_completed=len(legacy.v1_keys)+len(legacy.v2_keys)+len(legacy.v3_keys),remaining=len(jobs)-len(legacy.cells),
                source_fingerprint=fingerprint(),runtime=status())


def execute(settings, authorized, config_path):
    from .parallel_executor import run
    run(settings, authorized, config_path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default=str(ROOT/'configs/method_comparison_periodic_v4_20261007.json'))
    p.add_argument('--execute-authorized',action='store_true');p.add_argument('--plan-output')
    args=p.parse_args();settings=load_config(args.config)
    if args.execute_authorized: execute(settings,True,args.config)
    else:
        value=plan(settings)
        if args.plan_output: atomic_json(args.plan_output,value)
        print(json.dumps({k:value[k] for k in ['legacy_completed','remaining']}|{'mode':'plan_only','first_jobs':value['pending'][:12]},indent=2))


if __name__=='__main__':main()
