"""Independent execution version. Training requires explicit execution authorization."""
import argparse
import contextlib
import json
import os
import random
import time
import sys
from importlib.metadata import version as package_version
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import default_collate
from .config import ROOT, METHODS, WINDOWS, load_config, configured_windows
from .data import load_series, ForecastWindows, masked_errors
from .native import build
from .state_io import read_json, replace, statistics as io_statistics
from .resources import atomic_json as save_json, atomic_torch, digest, identity, Timings
from .provenance import fingerprint, resolve_settings
from .prepared_text import prepare_text
from .batching import host_batch, transfer, observed_error, resident_vectors
from .checkpoint import ResumeFile, epoch_plan, consume_sequential_loader_seed
from .parallel_policy import preparation_lock, text_cache_ready, PreparationCancelled, gpu_compute_lock


class PausedAtCheckpoint(RuntimeError):
    pass


def to_device(batch, device):
    return transfer(batch, device)


def loss_ot(model, batch):
    prediction=model(batch);truth,mask=batch['y'],batch['mask']
    if prediction.shape!=truth.shape: raise ValueError('Prediction shape mismatch')
    loss=observed_error(prediction,batch).square().mean()
    torch._assert_async(torch.isfinite(loss),'Non-finite observed multivariate loss')
    return loss,prediction


@torch.no_grad()
def evaluate(model, dataset, device, batch_size, generator=None, full=False):
    model.eval()
    if generator is not None: consume_sequential_loader_seed(generator)
    arrays=[[],[],[],[]] if full else None
    sums=torch.zeros(3,dtype=torch.float64,device=device)
    finite=torch.ones((),dtype=torch.bool,device=device)
    for start in range(0,len(dataset),batch_size):
        host=host_batch(dataset,range(start,min(start+batch_size,len(dataset))))
        batch=to_device(host,device)
        safe=dict(batch);safe['y']=torch.zeros_like(batch['y']);safe['mask']=torch.zeros_like(batch['mask'])
        prediction=model(safe);truth,mask=batch['y'],batch['mask']
        error=observed_error(prediction,batch)
        finite.logical_and_(torch.isfinite(error).all())
        # FP64 reduction is disclosed; float32 element errors are unchanged.
        sums += torch.stack([error.square().double().sum(),error.abs().double().sum(),
                             error.new_tensor(batch['_observed_count'],dtype=torch.float64)])
        if full:
            for dest,value in zip(arrays,[prediction,host['y'],host['mask'],host['origin']]):
                dest.append(value.detach().cpu().numpy())
    if not bool(finite) or sums[2].item()==0: raise FloatingPointError('Invalid validation/test values')
    a,b,n=sums.cpu().tolist()
    scores={'mse':a/n,'mae':b/n,'valid_target_elements':int(n)}
    if full:
        arrays=[np.concatenate(v) for v in arrays]
        scores=masked_errors(*arrays[:3])  # original final test score definition
    return scores,arrays


def checkpoint_keys(model, method):
    keys={n for n,p in model.named_parameters() if p.requires_grad}
    keys|={n for n,_ in model.named_buffers()}
    if method=='Proceed': keys|={n for n,_ in model.named_parameters()}
    return keys


def run(method, domain, horizon, seed, settings, authorized=False, owner_pid=None, lease_token=None):
    from .process_guard import require_execution
    require_execution(settings,authorized,owner_pid,
                      dict(method=method,domain=domain,horizon=horizon,seed=seed),lease_token)
    if seed not in settings['seeds']:
        raise ValueError('Seed outside the approved fresh three-seed experiment')
    if settings['campaign']=='method_comparison_20261004': raise ValueError('Legacy campaign is immutable')
    folder=Path(settings['runs_root'])/settings['campaign']/domain/('H%d'%horizon)/method/('seed%d'%seed)
    folder.mkdir(parents=True,exist_ok=True)
    sources=fingerprint()
    if (folder/'result.json').exists():
        old=json.loads((folder/'config.json').read_text('utf-8'))
        if old['source_fingerprint']!=sources or old['settings_input']!=settings or digest(old['data']['source'])!=old['data_sha256']:
            raise ValueError('Completed optimized result has incompatible provenance')
        result=json.loads((folder/'result.json').read_text('utf-8'))
        for name,value in result['artifacts'].items():
            if digest(folder/name)!=value: raise ValueError('Completed optimized artifact changed')
        return result
    device=torch.device(settings['device'])
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA is unavailable')
    torch.set_num_threads(settings['threads'])
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    timer=Timings(device);started=time.perf_counter()
    options=settings['execution']
    def cancelled():
        from .process_guard import controller_alive
        return (read_json(options['control_file']).get('pause',True)
                or not controller_alive(settings,owner_pid))
    def initialization_state(phase):
        save_json(folder/'state.json',dict(status='initializing',phase=phase,pid=os.getpid(),updated_at=time.time()))
    initialization_state('data_prepare')
    with timer.phase('data_prepare'):
        with preparation_lock(Path(options['cache_root'])/'parallel_data_prepare.lock'):
            data=load_series(domain,split=tuple(settings['split']),cache_root=options['cache_root'])
        L,hs=configured_windows(settings)[domain]
        if horizon not in hs: raise ValueError('Outside approved L/H matrix')
        cfg=resolve_settings(settings,method,len(data.values))
        cfg.update(run_kind=settings['campaign'],active_seed=seed,active_method=method,_settings_input=settings)
        sets={s:ForecastWindows(data,s,L,horizon,method=method) for s in ['train','val','test']}
        if method=='Proceed':
            indices=sets['train'].indices;indices=indices[indices>=L-1+horizon]
            sets['train'].indices=indices[-max(1,int(len(indices)*.2)):]
        if not all(len(ds) for ds in sets.values()): raise ValueError('Empty split')
    initialization_state('text_prepare_wait')
    with timer.phase('text_encode'):
        # The table implementation and representation key stay unchanged.
        # Quota writers are serialized outside it; ready training workers proceed.
        with preparation_lock(Path(options['cache_root'])/'parallel_text_prepare.lock') as waited:
            timer.seconds['text_prepare_lock_wait']=waited
            initialization_state('text_encode')
            cold=not text_cache_ready(data,method,settings,device)
            gpu_slot=(gpu_compute_lock(options['report_root'],cancelled=cancelled,slots=options['gpu_compute_workers'])
                if device.type=='cuda' and cold else contextlib.nullcontext())
            if cold:initialization_state('text_encode_gpu_wait')
            with gpu_slot as waited_gpu:
                timer.seconds['text_encode_gpu_wait']=float(waited_gpu or 0.)
                initialization_state('text_encode')
                text=prepare_text(data,method,settings,device)
        resident_vectors(text,device,options.get('device_data_bytes',128*1024**2))
    initialization_state('model_load')
    with timer.phase('model_load'):
        model,resolved=build(method,cfg,data,L,horizon,device)
        if method=='On-Adapter' and options.get('cache_frozen_numeric',False):
            from .numeric_features import prepare_numeric
            prepare_numeric(model,data,sets,cfg,device)
    initialization_state('gpu_compute_wait')
    compute_slot=(gpu_compute_lock(options['report_root'],cancelled=cancelled,slots=options['gpu_compute_workers'])
        if device.type=='cuda' else contextlib.nullcontext())
    with compute_slot as waited_gpu:
        timer.seconds['gpu_compute_wait']=float(waited_gpu or 0.)
        keys=checkpoint_keys(model,method)
        optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
            lr=cfg['llm_learning_rate'] if method=='GPT4MTS' else cfg['learning_rate'])
        generator=torch.Generator().manual_seed(seed)
        metadata=dict(method=method,domain=domain,horizon=horizon,lookback=L,seed=seed,smoke=False,
            settings_input=settings,settings=cfg,resolved=resolved,dataset_configuration=settings['dataset_windows'][domain],data=data.metadata(),data_sha256=digest(data.path),
            source_fingerprint=sources,protocol_version=cfg['protocol_version'],execution_version='textrepair-v1',
            feature_table=text.table.metadata if text else None,
            environment=dict(python=sys.version,torch=str(torch.__version__),numpy=np.__version__,
                             transformers=package_version('transformers'),cuda_build=torch.version.cuda),
            trajectory_policy='distinct campaign: batched frozen encoding, isolated precompute RNG, streaming validation reduction; not presumed bitwise legacy-identical',
            checkpoint='best.pt selects validation minimum; resume.json atomically selects one complete training/online snapshot',
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            parameters=sum(p.numel() for p in model.parameters()),
            numeric_dependency_sha256=digest(resolved['pretrained_numeric_checkpoint']) if resolved.get('pretrained_numeric_checkpoint') else None)
        from .migration import lineage, load_migrated
        recovery_lineage=lineage(folder)
        if recovery_lineage: metadata['resume_lineage']=recovery_lineage
        if (folder/'config.json').exists():
            previous=json.loads((folder/'config.json').read_text('utf-8'))
            if identity(previous)!=identity(metadata): raise ValueError('Unfinished optimized cell metadata changed')
        save_json(folder/'config.json',metadata)
        resume=ResumeFile(folder,identity(metadata),keys)
        checkpoint=resume.load()
        if checkpoint is None: checkpoint=load_migrated(folder,metadata)
        progress=dict(phase='train',epoch=0,cursor=0,plan=None,total=0.,count=0,history=[],best=float('inf'),stale=0)
        if checkpoint: progress=resume.restore(checkpoint,model,optimizer,generator,device)
        if progress.get('timings'):
            for name,value in progress['timings']['seconds'].items(): timer.seconds[name]=timer.seconds.get(name,0.)+value
            timer.counters.update(progress['timings']['counters'])
        previous_elapsed=progress.get('active_elapsed_seconds',0.)
        interval=options.get('checkpoint_every_batches',64)
        control=Path(options['control_file'])

        def checkpoint_save(online_optimizer=None):
            progress['timings']=timer.snapshot()
            progress['active_elapsed_seconds']=previous_elapsed+time.perf_counter()-started
            with timer.phase('checkpoint_save'):
                resume.save(model,optimizer,progress,generator,online_optimizer)
            save_json(folder/'state.json',dict(status='running',phase=progress['phase'],epoch=progress['epoch'],
                cursor=progress['cursor'],updated_at=time.time(),pid=os.getpid(),timings=timer.snapshot(),
                cache=dict(build=text.stats,runtime=text.table.stats) if text else None,
                effective=cfg['effective_execution']))
            from .process_guard import controller_alive
            if read_json(control).get('pause',True) or not controller_alive(settings,owner_pid):
                save_json(folder/'state.json',dict(status='paused_at_checkpoint',phase=progress['phase'],
                           epoch=progress['epoch'],cursor=progress['cursor'],pid=os.getpid(),updated_at=time.time()))
                raise PausedAtCheckpoint('Paused after atomic recovery checkpoint')

        while progress['phase']=='train' and progress['epoch']<cfg['epochs']:
            if progress['plan'] is None:
                model.train()
                if hasattr(model,'begin_training_epoch'): model.begin_training_epoch()
                progress.update(plan=epoch_plan(len(sets['train']),cfg['batch_size'],
                    method not in {'ADAPT-Z','On-Adapter','Proceed'},generator),cursor=0,total=0.,count=0)
                checkpoint_save()
            with timer.phase('train'):
                total=torch.as_tensor(progress['total'],dtype=torch.float64,device=device)
                for index in range(progress['cursor'],len(progress['plan'])):
                    host=host_batch(sets['train'],progress['plan'][index])
                    n=int(host['mask'].sum())
                    batch=to_device(host,device)
                    if hasattr(model,'native_training_step') and method in {'ADAPT-Z','On-Adapter','DSOF','Proceed'}:
                        value=model.native_training_step(batch)
                    else:
                        optimizer.zero_grad(set_to_none=True)
                        value,_=loss_ot(model,batch)
                        if hasattr(model,'auxiliary_loss'): value=value+model.auxiliary_loss()
                        value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                    value=torch.as_tensor(value,device=device).detach()
                    torch._assert_async(torch.isfinite(value),'Non-finite training objective')
                    total+=value.double()*n;progress['count']+=n
                    timer.counters['train_windows']=timer.counters.get('train_windows',0)+len(host['origin'])
                    progress.update(cursor=index+1,total=total.detach())
                    if progress['cursor']%interval==0: checkpoint_save()
                progress['total']=float(total)
            with timer.phase('validation'):
                if hasattr(model,'prepare_evaluation'): model.prepare_evaluation()
                val,_=evaluate(model,sets['val'],device,cfg['batch_size'],generator)
            row=dict(epoch=progress['epoch']+1,train_objective=progress['total']/progress['count'],val_mse=val['mse'],
                     timings=timer.snapshot(),train_windows=len(sets['train']),batch_size=cfg['batch_size'])
            progress['history'].append(row)
            if val['mse']<progress['best']:
                progress['best']=val['mse'];progress['stale']=0
                with timer.phase('best_checkpoint_save'):
                    atomic_torch(folder/'best.pt',{k:v.detach().cpu().clone() for k,v in model.state_dict().items() if k in keys})
            else: progress['stale']+=1
            progress.update(epoch=progress['epoch']+1,plan=None,cursor=0,total=0.,count=0)
            if progress['stale']>=cfg['patience'] or progress['epoch']>=cfg['epochs']:
                progress['phase']='test_ready'
            save_json(folder/'history.json',progress['history'])
            print(method,domain,horizon,seed,json.dumps(row),flush=True)
            checkpoint_save()

        if progress['phase']=='test_ready':
            model.load_state_dict(torch.load(folder/'best.pt',map_location=device,weights_only=True),strict=False)
            progress['phase']='online' if METHODS[method][1]=='online' else 'static_test'
            # Online runtime is reset by evaluate_online; no mid-stream state yet.
            progress['online']=None
            checkpoint_save()
            checkpoint=None
        if progress['phase']=='online':
            from .online import evaluate_online
            def save_online(state,online_optimizer):
                progress['online']=state;progress['cursor']=state['cursor']
                checkpoint_save(online_optimizer)
            def restore_online(online_optimizer):
                nonlocal progress
                if checkpoint is None or checkpoint['progress'].get('online') is None: return None
                progress=resume.restore(checkpoint,model,optimizer,generator,device,online_optimizer)
                return progress['online']
            with timer.phase('online_total'):
                scores,arrays,events=evaluate_online(model,data,sets['test'],device,cfg,
                    checkpoint_callback=save_online,resume_loader=restore_online,
                    checkpoint_every=options.get('checkpoint_every_origins',64),timings=timer)
        else:
            with timer.phase('test'):
                scores,arrays=evaluate(model,sets['test'],device,cfg['batch_size'],generator,full=True)
            events=[]
        with timer.phase('save'):
            pred,target,mask,origins=arrays
            target_rows=origins[:,None]+np.arange(1,horizon+1)[None,:]
            actual=data.raw[target_rows]
            if not np.array_equal(mask,np.isfinite(actual)): raise ValueError('Final label mask mismatch')
            if not np.allclose(target[mask],((actual-data.mean)/data.scale)[mask],rtol=1e-5,atol=1e-5):
                raise ValueError('Final label standardization mismatch')
            for event in events:
                if 'last_label_row' in event: assert event['last_label_row']<=event['label_cutoff']
                if event.get('max_observed_label_row') is not None: assert event['max_observed_label_row']<=event['label_cutoff']
            tmp=folder/'predictions.tmp.npz'
            np.savez_compressed(tmp,predictions=pred,targets=target,mask=mask,origins=origins,target_rows=target_rows)
            replace(tmp,folder/'predictions.npz')
            save_json(folder/'update_events.json',events)
            ot=data.target_index
            scores.update(method=method,domain=domain,horizon=horizon,lookback=L,seed=seed,smoke=False,status='completed',
                metric_space=cfg['metric_space'],columns=data.columns,ot_diagnostic=masked_errors(pred[...,ot],target[...,ot],mask[...,ot]),
                elapsed_seconds=previous_elapsed+time.perf_counter()-started,epochs=len(progress['history']),best_val_mse=progress['best'],
                update_events=len(events),ds_of_td_updates=sum(e.get('td_updates',0) for e in events),
                ds_of_er_samples=sum(e.get('er_samples',0) for e in events),result_directory=str(folder),
                source_fingerprint=sources,protocol_version=cfg['protocol_version'],execution_version='textrepair-v1',
                peak_gpu_mib=torch.cuda.max_memory_allocated()/1024**2 if device.type=='cuda' else 0,
                artifacts={n:digest(folder/n) for n in ['best.pt','config.json','predictions.npz','update_events.json']})
        scores['timings']=timer.snapshot()
        scores['cache']=dict(build=text.stats,runtime=text.table.stats) if text else None
        scores['frozen_numeric_cache']=model.frozen_numeric.stats if hasattr(model,'frozen_numeric') else None
        scores['effective_execution']=cfg['effective_execution']
        scores['state_io']=io_statistics()
        if recovery_lineage: scores['resume_lineage']=recovery_lineage
        scores['throughput_train_windows_per_second']=timer.counters.get('train_windows',0)/max(timer.seconds.get('train',0),1e-9)
        save_json(folder/'result.json',scores)
        save_json(folder/'state.json',dict(status='completed',finished_at=time.time(),pid=os.getpid()))
        return scores


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--method',required=True,choices=METHODS);p.add_argument('--domain',required=True,choices=WINDOWS)
    p.add_argument('--horizon',required=True,type=int);p.add_argument('--seed',type=int,choices=[2025,2026,2027],default=2025)
    p.add_argument('--config',default=str(ROOT/'configs/method_comparison_periodic_v3_20261006.json'))
    p.add_argument('--execute-authorized',action='store_true');p.add_argument('--owner-pid',type=int)
    p.add_argument('--lease-token')
    args=p.parse_args()
    try: run(args.method,args.domain,args.horizon,args.seed,load_config(args.config),args.execute_authorized,args.owner_pid,args.lease_token)
    except (PausedAtCheckpoint,PreparationCancelled) as exc: print(str(exc),flush=True);raise SystemExit(75)


if __name__=='__main__':main()
