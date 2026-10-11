"""One durable controller, at most six compute workers, no duplicated queues."""
import contextlib
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
import psutil
from .config import ROOT
from .state_io import read_json, statistics as io_statistics
from .resources import atomic_json, digest, identity, exclusive
from .provenance import fingerprint, verify_v1


def reduced_concurrency(limit):
    return max(1, limit // 2)
from .parallel_policy import job_key, can_admit, reservation, resources, validate_policy, gpu_compute_lock, WORKER_MODULES


def verify_result(folder, settings, sources):
    metadata=json.loads((folder/'config.json').read_text('utf-8'))
    result=json.loads((folder/'result.json').read_text('utf-8'))
    from .config import configured_windows
    L, horizons=configured_windows(settings)[result['domain']]
    if result['horizon'] not in horizons or result['lookback']!=L or metadata['lookback']!=L or result['execution_version']!='periodic-v3':
        raise ValueError('Completed result belongs to a different L/H/protocol')
    if metadata['settings_input']!=settings or metadata['source_fingerprint']!=sources:
        raise ValueError('Completed v3 configuration/source changed: '+str(folder))
    if digest(metadata['data']['source'])!=metadata['data_sha256']:
        raise ValueError('Completed v3 data changed')
    for name,expected in result['artifacts'].items():
        if digest(folder/name)!=expected: raise ValueError('Completed v3 artifact changed: '+name)
    return result


def v1_continuations(settings, known, approved):
    """Continue committed v1 states through unchanged v1 code/configuration."""
    if not settings['execution'].get('resume_v1_cells',False): return {},None
    config=ROOT/'configs/method_comparison_optimized_20261005.json'
    old=json.loads(config.read_text('utf-8')); found={}
    for manifest in (Path(old['runs_root'])/old['campaign']).glob('*/*/*/*/resume.json'):
        folder=manifest.parent
        if (folder/'result.json').exists(): continue
        meta=json.loads((folder/'config.json').read_text('utf-8')); key=job_key(meta)
        if key in known: raise ValueError('Unfinished v1 overlaps completed configuration')
        if key not in approved: raise ValueError('Unfinished v1 outside approved matrix')
        if meta['settings_input']!=old: raise ValueError('Unfinished v1 configuration changed')
        for relative,expected in meta['source_fingerprint'].items():
            if digest(ROOT/relative)!=expected: raise ValueError('Unfinished v1 code changed')
        if digest(meta['data']['source'])!=meta['data_sha256']: raise ValueError('Unfinished v1 data changed')
        record=json.loads(manifest.read_text('utf-8'))
        if digest(folder/('resume_%d.pt'%record['slot']))!=record['sha256']:
            raise ValueError('Unfinished v1 checkpoint changed')
        found[key]=dict(folder=folder,config=config,manifest=record)
    return found,old


def run(settings, authorized, config_path):
    from .process_guard import require_execution
    from .scheduler import plan, cell_dir, milestone
    require_execution(settings,authorized)
    validate_policy(settings)
    report=Path(settings['execution']['report_root']);report.mkdir(parents=True,exist_ok=True)
    control=Path(settings['execution']['control_file'])
    v1_report=ROOT/'reports/method_comparison_optimized_20261005'
    v1_control=v1_report/'control.json'
    active=[]; pending=[]; known={}; index=[]; admission=[]
    runtime_limit=settings['execution']['parallel_workers']; retries={}; failure=None
    config_path=Path(config_path).resolve()
    session=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),
                 config_path=str(config_path),config_sha256=digest(config_path),
                 settings_identity=identity(settings),session_token=uuid.uuid4().hex,active=[])

    def set_pause(path,value,reason):
        if path==v1_control and not settings['execution'].get('resume_v1_cells',False): return
        old=read_json(path)
        if old.get('pause')==value and old.get('reason')==reason: return
        old.update(pause=value,requested_at=time.time(),reason=reason)
        atomic_json(path,old)

    def journal():
        session['active']=[{k:v for k,v in item.items() if k in
            {'job','backend','token','config_path','launcher_pid','reservation','folder','started_at','rss_mib','initializing'}}
            for item in active]
        session['updated_at']=time.time()
        atomic_json(report/'active_workers.json',session)

    def publish(status='running',observed=None):
        workers=[]
        for item in active:
            path=Path(item['folder'])/'state.json'
            progress=read_json(path) if path.exists() else None
            workers.append(dict(job=item['job'],backend=item['backend'],launcher_pid=item['launcher_pid'],
                                directory=item['folder'],reservation=item['reservation'],progress=progress))
        atomic_json(report/'queue_state.json',dict(status=status,pid=session['pid'],create_time=session['create_time'],
            jobs=workers,job=workers[0]['job'] if workers else None,active_workers=len(workers),
            maximum_workers=runtime_limit,compute_slots=settings['execution']['gpu_compute_workers'],state_io=io_statistics(),completed=len(known),total=settings['expected_matrix_cells'],remaining=settings['expected_matrix_cells']-len(known),
            pending=len(pending),updated_at=time.time(),resources=observed,
            admission_denials=admission[-12:],failure=failure))

    with contextlib.ExitStack() as locks:
        locks.enter_context(exclusive(ROOT/'reports/method_comparison_execution.lock'))
        locks.enter_context(exclusive(report/'queue.lock'))
        # Also block immutable v1's launcher, which predates the v2 inventory.
        # v1 checkpoint continuations remain workers of this sole controller.
        locks.enter_context(exclusive(v1_report/'queue.lock'))
        locks.enter_context(exclusive(ROOT/'reports/method_comparison_optimized_v2_20261005/queue.lock'))
        locks.enter_context(exclusive(ROOT/'reports/method_comparison_optimized_v3_20261006/queue.lock'))
        locks.enter_context(exclusive(ROOT/'reports/method_comparison_optimized_v4_20261006/queue.lock'))
        locks.enter_context(exclusive(ROOT/'reports/periodic_configuration_20261006/queue.lock'))
        locks.enter_context(exclusive(ROOT/'reports/periodic_parallel6_20261006/queue.lock'))
        try:
            p=plan(settings);atomic_json(report/'execution_plan.json',p)
            sources=p['source_fingerprint'];known={job_key(j):j for j in p['legacy_references']}
            root=Path(settings['runs_root'])/settings['campaign'];root.mkdir(parents=True,exist_ok=True)
            for result in root.glob('*/*/*/*/result.json'):
                r=verify_result(result.parent,settings,sources);key=job_key(r)
                if key in known: raise ValueError('Duplicate cross-version configuration')
                known[key]=dict(domain=r['domain'],horizon=r['horizon'],seed=r['seed'],method=r['method'],
                                directory=str(result.parent),lineage='periodic-v3')
                index.append(dict(known[key],result=str(result),sha256=digest(result)))
            continuations,old=v1_continuations(settings,known,{job_key(j) for j in p['matrix']})
            from .migration import continuation_jobs
            migrated=continuation_jobs(settings,known,{job_key(j) for j in p['matrix']})
            pending=[j for j in p['matrix'] if job_key(j) not in known]
            pending.sort(key=lambda j:0 if job_key(j) in continuations or job_key(j) in migrated else 1)
            used=sum(f.stat().st_size for f in root.rglob('*') if f.is_file())
            signature=identity(milestone(known,settings));journal()
            while pending or active:
                if fingerprint()!=sources or digest(config_path)!=session['config_sha256']:
                    raise ValueError('Running controller source/configuration changed')
                pausing=read_json(control).get('pause',True)
                if pausing: set_pause(v1_control,True,'Controller pause: preserve v1 continuation checkpoint')
                for item in list(active):
                    code=item['child'].poll()
                    if code is None:
                        try:
                            launcher=psutil.Process(item['launcher_pid'])
                            procs=[launcher]+launcher.children(recursive=True)
                            item['rss_mib']=sum(a.memory_info().rss for a in procs)/1024**2
                            progress_path=Path(item['folder'])/'state.json'
                            if progress_path.exists():
                                value=read_json(progress_path)
                                if value.get('updated_at',0)>=item['started_at']:
                                    item['initializing']=value.get('status')!='running'
                        except psutil.NoSuchProcess: pass
                        continue
                    item['log'].close();item['claim'].__exit__(None,None,None);active.remove(item)
                    if item.get('gpu_claim'):item['gpu_claim'].__exit__(None,None,None)
                    folder=Path(item['folder'])
                    if item['backend']=='periodic':
                        used+=sum(f.stat().st_size for f in folder.rglob('*') if f.is_file())-item['bytes_before']
                    if code==75:
                        pending.insert(0,item['job'])
                        if not pausing: raise RuntimeError('Worker paused unexpectedly; inspect controller/checkpoint')
                    elif code or not (folder/'result.json').exists():
                        tail=(folder/'worker.log').read_text('utf-8',errors='replace')[-12000:]
                        key=job_key(item['job'])
                        if 'out of memory' in tail.lower() and retries.get(key,0)==0 and runtime_limit>1:
                            runtime_limit=reduced_concurrency(runtime_limit);retries[key]=1;pending.insert(0,item['job'])
                            atomic_json(report/'memory_fallback.json',dict(job=item['job'],returncode=code,
                                reason='OOM: retry once from committed state after halving worker admission',concurrency=runtime_limit,log_tail=tail,time=time.time()))
                        else: raise RuntimeError('Worker failed: '+str(item['job'])+' returncode='+str(code))
                    else:
                        if item['backend']=='v1':
                            result,_=verify_v1(folder,settings)
                        else: result=verify_result(folder,settings,sources)
                        key=job_key(result)
                        if key in known: raise ValueError('Duplicate completion')
                        ref=dict(item['job'],directory=str(folder),lineage='optimized-'+item['backend'])
                        known[key]=ref
                        index.append(dict(ref,result=str(folder/'result.json'),sha256=digest(folder/'result.json')))
                        atomic_json(report/'optimized_index.json',dict(results=index,result_bytes=used,
                            prior_completed=p['legacy_completed'],completed=len(known)))
                        current=milestone(known,settings);new_signature=identity(current)
                        if new_signature!=signature:
                            atomic_json(report/'analysis_request.json',dict(milestone=current,status='new_data_ready',
                                prior_references=p['legacy_references'],optimized_index=str(report/'optimized_index.json'),
                                policy='Preserve execution versions; figures only on explicit analysis request'))
                            signature=new_signature
                    journal()
                if not any(a['backend']=='v1' for a in active):
                    set_pause(v1_control,True,'v1 retired; continuation controlled by optimized v3')
                observed=resources()
                admission=[]
                if not pausing:
                    # Preserve domain/method matrix order instead of allowing
                    # all light methods to starve the large-model baselines.
                    candidates=sorted(pending,key=lambda j:0 if job_key(j) in migrated else 1)
                    for job in candidates:
                        if len(active)>=runtime_limit: break
                        large={'GPT4MTS','MM-TSFlib'}
                        ready_light=any(j['method'] not in large and
                            (j['method']=='iTransformer' or (*job_key(j)[:3],'iTransformer') in known)
                            for j in pending)
                        if (job['method'] in large and ready_light and
                                any(a['job']['method'] in large for a in active)):
                            admission.append(dict(job=job,reason='Mix one large model with ready light jobs'));continue
                        backend='v1' if job_key(job) in continuations else 'periodic'
                        if backend=='v1' and any(a['backend']=='v1' for a in active): continue
                        if backend=='v1' and (report/'gpu_compute.lock').exists():
                            admission.append(dict(job=job,reason='GPU compute slot held'));continue
                        allowed,reason=can_admit(job,active,known,settings,observed,runtime_limit)
                        if not allowed:
                            admission.append(dict(job=job,reason=reason));continue
                        reserve=settings['execution']['checkpoint_reserve_bytes']
                        if used+(len(active)+1)*reserve>settings['storage_limit_gib']*1024**3:
                            raise RuntimeError('Result/checkpoint byte budget would be exceeded')
                        if shutil.disk_usage(root).free<settings['minimum_disk_free_gib']*1024**3+(len(active)+1)*reserve:
                            raise RuntimeError('Insufficient disk capacity for atomic checkpoints')
                        folder=continuations[job_key(job)]['folder'] if backend=='v1' else cell_dir(settings,job)
                        folder.mkdir(parents=True,exist_ok=True)
                        claim=exclusive(folder/'run.lock');claim.__enter__()
                        config=continuations[job_key(job)]['config'] if backend=='v1' else config_path
                        entry=dict(job=job,backend=backend,token=uuid.uuid4().hex,config_path=str(config.resolve()),
                            folder=str(folder),reservation=reservation(job,settings),launcher_pid=None,
                            started_at=time.time(),rss_mib=0,initializing=True,claim=claim,
                            bytes_before=sum(f.stat().st_size for f in folder.rglob('*') if f.is_file()))
                        active.append(entry);journal()
                        command=[sys.executable,'-u','-m',WORKER_MODULES[backend],
                            '--method',job['method'],'--domain',job['domain'],'--horizon',str(job['horizon']),
                            '--seed',str(job['seed']),'--config',str(config.resolve()),'--execute-authorized']
                        if backend=='periodic': command+=['--owner-pid',str(os.getpid()),'--lease-token',entry['token']]
                        else: set_pause(v1_control,False,'Authorized immutable v1 checkpoint continuation under sole v3 controller')
                        if backend=='v1':
                            entry['gpu_claim']=gpu_compute_lock(report)
                            entry['gpu_claim'].__enter__()
                        env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONUTF8='1')
                        entry['log']=(folder/'worker.log').open('a',encoding='utf-8')
                        entry['child']=subprocess.Popen(command,cwd=ROOT,stdout=entry['log'],stderr=subprocess.STDOUT,
                            env=env,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                        entry['launcher_pid']=entry['child'].pid
                        pending.remove(job);journal()
                        with (report/'dispatch.jsonl').open('a',encoding='utf-8') as stream:
                            stream.write(json.dumps(dict(job=job,backend=backend,pid=entry['launcher_pid'],
                                time=time.time(),active_workers=len(active),resources=observed))+'\n')
                        observed=resources()
                journal();publish('pausing_at_checkpoints' if pausing and active else
                    ('paused_at_checkpoint' if pausing else ('running' if active else 'waiting_resources')),observed)
                if pausing and not active: return
                if not active and pending and not admission:
                    raise RuntimeError('No ready configuration despite pending matrix entries')
                time.sleep(settings['execution'].get('scheduler_poll_seconds',3))
            publish('completed')
        except BaseException as exc:
            failure=str(exc);set_pause(control,True,'Controller failure: '+failure)
            set_pause(v1_control,True,'Controller failure: preserve v1 continuation')
            # Keep ownership locks while workers reach complete recovery boundaries.
            while any(a.get('child') is not None and a['child'].poll() is None for a in active):
                publish('failure_draining_to_checkpoints');time.sleep(3)
            publish('failed');raise
        finally:
            set_pause(v1_control,True,'Immutable v1 queue retired; controller finished or paused')
            for item in active:
                if item.get('log'): item['log'].close()
                if item.get('gpu_claim'):item['gpu_claim'].__exit__(None,None,None)
                item['claim'].__exit__(None,None,None)
            session['active']=[];session['finished_at']=time.time()
            atomic_json(report/'active_workers.json',session)
