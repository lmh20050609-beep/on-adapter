"""Local serial, persistent 190-job queue; stops on any failed experiment."""
import argparse,json,time,traceback,os
from pathlib import Path
import numpy as np
from .config import ROOT,Config,matrix,job_id
from .run import run_one
from .util import write_json,revision
from .audit import audit_all,AuditError

def queue_status(state,jobs,root=ROOT):
    """Read evidence markers without trusting queue counters or launching work."""
    if state is None:
        return {'status':'not_started','planned':len(jobs),'completed':0,'running':0,'failed':0,'pending':len(jobs),'artifact_errors':[]}
    expected={job_id(j) for j in jobs}
    if set(state.get('jobs',{}))!=expected: raise AuditError('Queue manifest does not match the planned matrix')
    counts={name:0 for name in ['pending','running','completed','failed']}; errors=[]
    for key,entry in state['jobs'].items():
        status=entry.get('status')
        if status not in counts: raise AuditError(f'Unknown queue status for {key}: {status}')
        counts[status]+=1
        if status=='completed':
            directory=Path(root)/'runs'/key
            for filename in ['status.json','config.json','metrics.json','predictions.npz','updates.jsonl','warm_updates.jsonl','prediction_context.json','warm_adapter.pt','final_adapter.pt']:
                if not (directory/filename).is_file(): errors.append(f'{key}: missing {filename}')
            if (directory/'status.json').is_file():
                if json.loads((directory/'status.json').read_text()).get('status')!='completed': errors.append(f'{key}: completion marker contradicts queue')
    pid=state.get('process_id'); process={'recorded_pid':pid,'verified_campaign_process':False}
    if pid:
        path=Path('/proc')/str(pid)/'cmdline'
        process['pid_exists']=path.exists()
        if path.exists(): process['verified_campaign_process']=b'iadapter.campaign' in path.read_bytes()
    else: process['liveness']='not_recorded'
    return dict(status='artifact_error' if errors else ('completed' if counts['completed']==len(jobs) else 'incomplete'),planned=len(jobs),**counts,artifact_errors=errors,experiment_code=state.get('code'),finished_at=state.get('finished_at'),process=process,evidence_validation='markers_only; --audit-completed verifies original CSVs')

def verify_group(jobs):
    reference=None
    for job in jobs:
        path=ROOT/'runs'/job_id(job)
        with np.load(path/'predictions.npz') as z:
            current=(z['origins'].copy(),z['targets'].copy(),z['valid_mask'].copy())
        if reference is None: reference=current
        else:
            for a,b in zip(reference,current): np.testing.assert_array_equal(a,b)

def main(limit=None,status_only=False,audit_completed=False):
    if limit is not None and limit<1: raise ValueError('--limit must be positive')
    jobs=matrix(); state_path=ROOT/'queue_state.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else None
    summary=queue_status(state,jobs,ROOT)
    if status_only:
        print(json.dumps(summary,ensure_ascii=False,indent=2))
        if summary['artifact_errors']: raise AuditError('Queue completion markers have missing or contradictory artifacts')
        return summary
    if summary['status']=='completed':
        report=audit_all(root=ROOT,jobs=jobs)
        if report['status']!='passed': raise AuditError(json.dumps(report['failures'],ensure_ascii=False))
        print(f'CAMPAIGN_ALREADY_COMPLETED {report["passed"]}/{report["planned"]}; original CSV audit passed; no files or timestamps changed')
        return report
    if audit_completed: raise AuditError('The planned campaign is not complete')
    if summary['artifact_errors']: raise AuditError('Repair incomplete artifacts before continuing')
    current_revision=revision()
    if current_revision['dirty']: raise RuntimeError('Commit experiment code before launching')
    if state_path.exists():
        if state['code']!=current_revision: raise RuntimeError('Queue belongs to another code revision; do not mix results')
    else:
        state={'code':current_revision,'started_at':time.time(),'jobs':{job_id(j):dict(j,status='pending') for j in jobs}}
        write_json(ROOT/'refine-logs/manifest.json',{'code':current_revision,'jobs':jobs,'max_parallel':1,'backend':'local WSL CUDA','retry_policy':'stop on failure; inspect before retry; never alter model or seed budget'})
    state['process_id']=os.getpid()
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
    p=argparse.ArgumentParser(); p.add_argument('--limit',type=int)
    group=p.add_mutually_exclusive_group(); group.add_argument('--status',action='store_true'); group.add_argument('--audit-completed',action='store_true')
    args=p.parse_args(); main(args.limit,args.status,args.audit_completed)
