"""Local serial, persistent 190-job queue; stops on any failed experiment."""
import argparse,json,time,traceback
import numpy as np
from .config import ROOT,Config,matrix,job_id
from .run import run_one
from .util import write_json,revision

def verify_group(jobs):
    reference=None
    for job in jobs:
        path=ROOT/'runs'/job_id(job)
        with np.load(path/'predictions.npz') as z:
            current=(z['origins'].copy(),z['targets'].copy(),z['valid_mask'].copy())
        if reference is None: reference=current
        else:
            for a,b in zip(reference,current): np.testing.assert_array_equal(a,b)

def main(limit=None):
    jobs=matrix(); state_path=ROOT/'queue_state.json'; current_revision=revision()
    if current_revision['dirty']: raise RuntimeError('Commit experiment code before launching')
    if state_path.exists():
        state=json.loads(state_path.read_text())
        if state['code']!=current_revision: raise RuntimeError('Queue belongs to another code revision; do not mix results')
    else:
        state={'code':current_revision,'started_at':time.time(),'jobs':{job_id(j):dict(j,status='pending') for j in jobs}}
        write_json(ROOT/'refine-logs/manifest.json',{'code':current_revision,'jobs':jobs,'max_parallel':1,'backend':'local WSL CUDA','retry_policy':'stop on failure; inspect before retry; never alter model or seed budget'})
    completed=0
    for index,job in enumerate(jobs):
        key=job_id(job); entry=state['jobs'][key]
        if entry['status']=='completed':
            path=ROOT/'runs'/key
            assert json.loads((path/'status.json').read_text())['status']=='completed'
            continue
        if entry['status']!='pending': raise RuntimeError(f'{key}: unresolved {entry["status"]}; inspect saved traceback')
        entry.update(status='running',started_at=time.time()); write_json(state_path,state)
        print(f'QUEUE {index+1}/190 {key}',flush=True)
        try:
            stats=run_one(Config(job['dataset'],job['horizon'],job['seed'],job['variant']),ROOT/'runs'/key)
            entry.update(status='completed',seconds=stats['seconds'],mse=stats['mse'],mae=stats['mae'])
            completed+=1
            if index==2:
                verify_group(jobs[:3]); state['sanity_passed']=True
                total=sum(state['jobs'][job_id(j)]['seconds'] for j in jobs[:3])
                state['sanity_seconds']=total
                print(f'SANITY_PASSED first_three_seconds={total:.1f}; larger domains have longer streams',flush=True)
            group=[j for j in jobs[:index+1] if (j['dataset'],j['horizon'],j['seed'])==(job['dataset'],job['horizon'],job['seed']) and state['jobs'][job_id(j)]['status']=='completed']
            verify_group(group)
        except Exception:
            entry.update(status='failed',error=traceback.format_exc()); write_json(state_path,state); raise
        state['completed']=sum(j['status']=='completed' for j in state['jobs'].values()); state['updated_at']=time.time(); write_json(state_path,state)
        lines=['# Experiment tracker','',f"Completed: {state['completed']}/190",'', '| Run | Stage | Status | MSE | MAE |','|---|---|---|---:|---:|']
        lines.extend(f"| {k} | {v['stage']} | {v['status']} | {v.get('mse','')} | {v.get('mae','')} |" for k,v in state['jobs'].items())
        (ROOT/'refine-logs/EXPERIMENT_TRACKER.md').write_text('\n'.join(lines)+'\n')
        if limit and completed>=limit: return
    state['finished_at']=time.time(); write_json(state_path,state)
    print('CAMPAIGN_COMPLETED 190/190',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--limit',type=int); main(p.parse_args().limit)
