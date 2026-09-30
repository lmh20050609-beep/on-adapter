"""B1 declared OT controls and deployment U/F factorial, preserving prior results."""
import argparse,copy,gc,json,time,traceback,subprocess
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .config import ROOT
from .models import Backbone
from .external_static import ot_data,pilot_config,evaluate
from .external_static_matrix import OUTPUT as STATIC,group_path
from .external_online import load_start
from .review_sprint import initialize_adapter,write_events
from .engine import Stream
from .shared_online import InternalBackend,OnlineRunner
from .util import metrics,seed_all,write_json,revision
from .audit import original_reference,same,require
from .analyze import md_table
from .replan_audit import OUTPUT

METHODS=['frozen_backbone','text_add_static','text_add_online','iadapter_U1F0','iadapter_U0F1']

def jobs():return [dict(dataset=d,seed=s,method=m) for d in ['Energy','SocialGood'] for s in [2023,2024,2025] for m in METHODS]
def key(job):return f"{job['dataset']}_s{job['seed']}_{job['method']}"

def full_start(job):
    return load_start({name:job[name] for name in ['dataset','seed']},'iadapter_online')

def load_bone(job,with_text=False,device='cuda'):
    source=group_path(STATIC,job)/'runs/numeric_static'
    c=pilot_config('text_add' if with_text else 'numeric_add',dataset=job['dataset'],seed=job['seed'],device=device)
    data,bounds=ot_data(c.dataset,with_embeddings=with_text)
    previous=json.loads((source/'config.json').read_text());result=json.loads((source/'metrics.json').read_text())
    assert previous['data']==data.metadata() and previous['boundaries']==bounds
    checkpoint=Path(result['backbone_checkpoint']);blob=torch.load(checkpoint,map_location='cpu',weights_only=True)
    bc=blob['config'];assert bc['dataset']==c.dataset and bc['seed']==c.seed and bc['columns']==['OT'] and bc['horizon']==6 and bc['lookback']==c.lookback and bc['d_model']==32 and bc['e_layers']==2 and bc['batch']==16
    model=Backbone(c.lookback,c.horizon,1,d_model=c.d_model,e_layers=c.e_layers,dropout=c.dropout).to(device)
    model.load_state_dict(blob['model']);model.requires_grad_(False);model.eval()
    return c,data,model,checkpoint,bounds

@torch.no_grad()
def frozen_predictions(c,data,model):
    origins=np.asarray(data.origins('test',c.lookback,c.horizon));values=[]
    for t in origins:values.append(model(torch.from_numpy(data.x[t-c.lookback+1:t+1]).unsqueeze(0).to(c.device))[0].cpu().numpy())
    return dict(origins=origins,predictions=np.stack(values),targets=np.stack([data.target[t+1:t+7] for t in origins]),valid_mask=np.stack([data.valid[t+1:t+7] for t in origins]))

def text_warm(c,data,model,base):
    directory=base/'warm'/f'{c.dataset}_s{c.seed}'
    adapter=initialize_adapter(c,1)
    if directory.exists():
        status=json.loads((directory/'status.json').read_text());cfg=json.loads((directory/'config.json').read_text())
        assert status['status']=='completed' and cfg['config']==c.to_dict() and cfg['data']==data.metadata() and cfg['code']==revision()
        adapter.load_state_dict(torch.load(directory/'warm_adapter.pt',map_location=c.device,weights_only=True));return adapter,directory
    directory.mkdir(parents=True,exist_ok=False);write_json(directory/'status.json',{'status':'running'})
    write_json(directory/'config.json',dict(config=c.to_dict(),data=data.metadata(),code=revision()))
    seed_all(c.seed+11000);_,events,stats=Stream(c,data,model,adapter).run('train')
    write_events(directory/'warm_updates.jsonl',events);torch.save(adapter.state_dict(),directory/'warm_adapter.pt')
    write_json(directory/'warm_metrics.json',stats);write_json(directory/'status.json',{'status':'completed'})
    return adapter,directory

def run_cell(job,base):
    out=base/'runs'/key(job);out.mkdir(parents=True,exist_ok=False);write_json(out/'status.json',{'status':'running'});start=time.perf_counter()
    try:
        m=job['method'];initial=None;adapter=None;events=[];contexts=[];warm=None
        if m.startswith('iadapter_'):
            c,data,backend,source,previous,initial,frozen=full_start(job)
            checkpoint=Path(json.loads((source/'metrics.json').read_text())['backbone_checkpoint']);model=backend.model;adapter=backend.adapter
            u=m=='iadapter_U1F0';f=m=='iadapter_U0F1';seed_all(c.seed+20000);torch.cuda.reset_peak_memory_stats()
            arrays,events,contexts,stats=OnlineRunner(c,data,backend,update_parameters=u,refresh_feedback=f).run()
            with np.load(source/'predictions.npz') as z:np.testing.assert_allclose(arrays['predictions'][:7],z['predictions'][:7],atol=1e-6,rtol=1e-6)
        else:
            c,data,model,checkpoint,bounds=load_bone(job,m!='frozen_backbone')
            frozen={n:p.detach().cpu().clone() for n,p in model.named_parameters()};u=m=='text_add_online';f=u
            if m!='frozen_backbone':adapter,warm=text_warm(c,data,model,base);initial={n:v.detach().cpu().clone() for n,v in adapter.state_dict().items()}
            torch.cuda.reset_peak_memory_stats();seed_all(c.seed+20000)
            if m=='frozen_backbone':arrays=frozen_predictions(c,data,model);stats=metrics(arrays['predictions'],arrays['targets'],arrays['valid_mask'])
            elif u:
                backend=InternalBackend(c,data,model,adapter);arrays,events,contexts,stats=OnlineRunner(c,data,backend).run()
                reference=base/'runs'/key(dict(job,method='text_add_static'))/'predictions.npz'
                with np.load(reference) as z:np.testing.assert_allclose(arrays['predictions'][:7],z['predictions'][:7],atol=1e-6,rtol=1e-6)
            else:arrays,contexts=evaluate(data,c,model,adapter);stats=metrics(arrays['predictions'],arrays['targets'],arrays['valid_mask'])
        config=dict(dataset=c.dataset,seed=c.seed,method=m,horizon=6,lookback=c.lookback,config=c.to_dict(),data=data.metadata(),code=revision(),parameter_updates=u,feedback_refresh=f,checkpoint=str(checkpoint),warm_source=str(warm) if warm is not None else (str(source) if m.startswith('iadapter_') else None),optimizer=dict(lr=.001,weight_decay=.0001,clip=1,batch=1) if u else None)
        write_json(out/'config.json',config)
        assert all(p.grad is None and torch.equal(p.detach().cpu(),frozen[n]) for n,p in model.named_parameters() if not p.requires_grad),'Frozen backbone changed'
        delta=0.
        if initial is not None:
            final={n:v.detach().cpu().clone() for n,v in adapter.state_dict().items()}
            delta=sum(float((final[n].double()-v.double()).square().sum()) for n,v in initial.items())**.5
            if not u:assert delta==0.,'Parameter-free strategy changed weights'
            elif stats.get('updated',0):assert delta>0.,'Adapter never updated'
            torch.save(initial,out/'initial_adaptive_state.pt');torch.save(final,out/'final_adaptive_state.pt')
        if m=='iadapter_U1F0':assert not arrays['numeric_feedback_snapshots'].any() and not arrays['text_feedback_snapshots'].any()
        if m=='frozen_backbone':baseline=arrays['predictions']
        else:
            with np.load(base/'runs'/key(dict(job,method='frozen_backbone'))/'predictions.npz') as z:baseline=z['predictions'].copy()
        arrays['frozen_backbone_predictions']=baseline
        residual=float(np.linalg.norm((arrays['predictions']-baseline)[arrays['valid_mask']])/(np.linalg.norm(baseline[arrays['valid_mask']])+1e-8))
        np.savez_compressed(out/'predictions.npz',**arrays);write_events(out/'updates.jsonl',events);write_json(out/'contexts.json',contexts)
        stats.update(config=config,seconds=time.perf_counter()-start,parameter_displacement_l2=delta,frozen_parameters_unchanged=True,trainable_parameters=sum(p.numel() for p in adapter.parameters()) if adapter is not None else 0,peak_allocated_bytes=torch.cuda.max_memory_allocated(),residual_to_frozen_prediction_norm=residual)
        if 'numeric_feedback_snapshots' in arrays:stats.update(numeric_feedback_norm_mean=float(np.linalg.norm(arrays['numeric_feedback_snapshots'].reshape(len(arrays['origins']),-1),axis=1).mean()),text_feedback_norm_mean=float(np.linalg.norm(arrays['text_feedback_snapshots'],axis=1).mean()))
        write_json(out/'metrics.json',stats);write_json(out/'status.json',{'status':'completed'})
        audit_cell(out,job);print(f"B1_COMPLETE {key(job)} MSE={stats['mse']:.7f} MAE={stats['mae']:.7f}",flush=True)
        del model,adapter;gc.collect();torch.cuda.empty_cache();return stats
    except Exception:
        write_json(out/'status.json',{'status':'failed','error':traceback.format_exc()});raise

def audit_cell(out,job):
    out=Path(out);ref=original_reference(job['dataset']);k=ref['columns'].index('OT');n=ref['rows'];a,b,end=int(.6*n),int(.7*n),int(.8*n)
    origins=np.arange(b-1,end-6);target=np.stack([ref['target'][t+1:t+7,k:k+1] for t in origins]);mask=np.stack([ref['valid'][t+1:t+7,k:k+1] for t in origins]);cfg=json.loads((out/'config.json').read_text());stats=json.loads((out/'metrics.json').read_text())
    require(json.loads((out/'status.json').read_text())['status']=='completed','Incomplete cell')
    require(cfg==stats['config'] and not cfg['code']['dirty'],'Bad provenance')
    with np.load(out/'predictions.npz') as z:
        same(z['origins'],origins,'Origins',exact=True);same(z['targets'],target,'Original target');same(z['valid_mask'],mask,'Original mask',exact=True)
        score=metrics(z['predictions'],target,mask)
        for metric in ['mse','mae']:require(abs(score[metric]-stats[metric])<1e-12,'Saved score')
        if 'feedback_origins' in z:
            events=[json.loads(x) for x in (out/'updates.jsonl').read_text().splitlines()];expected=origins[:-6].tolist();require([e['origin'] for e in events]==expected,'Missing/repeated mature origin')
            latest=[-1,-1];last_parameter=-1;at={e['update_at']:e for e in events}
            for index,t in enumerate(origins):
                same(z['feedback_origins'][index],latest,'Feedback chronology',exact=True);require(int(z['parameter_update_origins'][index])==last_parameter,'Parameter chronology')
                e=at.get(int(t))
                if e:
                    s=e['origin'];count=int(mask[origins.tolist().index(s)].sum());require(e['mature_at']==s+6==int(t) and e['text_origin']==s==e['state_origin'] and e['valid_targets']==count,'Maturity/context mismatch')
                    require(e['text_ids']==ref['text_ids'][s],'Wrong original text')
                    if count:
                        if cfg['parameter_updates']:last_parameter=s
                        if cfg['feedback_refresh']:
                            latest[0]=s
                            if cfg['config']['variant']=='full':latest[1]=s
            if not cfg['feedback_refresh']:require(not z['numeric_feedback_snapshots'].any() and not z['text_feedback_snapshots'].any(),'Disabled feedback active')
    require(stats['frozen_parameters_unchanged'],'Frozen check failed')
    if not cfg['parameter_updates']:require(stats['parameter_displacement_l2']==0.,'Unexpected adaptive parameter update')
    write_json(out/'validation.json',dict(status='passed',original_csv_target_mask=True,update_feedback_flags=True))
    return dict(dataset=job['dataset'],seed=job['seed'],method=job['method'],mse=score['mse'],mae=score['mae'],source=str(out),new_evaluation=True)

def campaign(base=OUTPUT,limit=None):
    base=Path(base)/'b1';base.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);code=revision()
    if code['dirty']:raise RuntimeError('Commit source before B1 campaign')
    subprocess.run(['git','ls-files','--error-unmatch','iadapter/feedback_factorial.py','iadapter/shared_online.py'],cwd=ROOT,capture_output=True,check=True)
    statefile=base/'queue_state.json';planned=jobs()
    state=json.loads(statefile.read_text()) if statefile.exists() else dict(code=code,planned_jobs=planned,jobs={key(j):{'status':'pending'} for j in planned})
    require(state['code']==code and state['planned_jobs']==planned,'Queue source/plan changed')
    done=0
    for job in planned:
        entry=state['jobs'][key(job)]
        if entry['status']=='completed':continue
        require(entry['status']=='pending','Unresolved job '+key(job));entry.update(status='running',started_at=time.time());write_json(statefile,state)
        try:result=run_cell(job,base);entry.update(status='completed',mse=result['mse'],mae=result['mae'])
        except Exception:entry.update(status='failed',error=traceback.format_exc());write_json(statefile,state);raise
        write_json(statefile,state);done+=1
        if limit is not None and done>=limit:break
    print('B1_QUEUE',sum(e['status']=='completed' for e in state['jobs'].values()),'/30',flush=True)

def analyze(base=OUTPUT):
    base=Path(base)/'b1';rows=[audit_cell(base/'runs'/key(j),j) for j in jobs()]
    manifest=json.loads((ROOT/'refine-logs/REPLAN_MANIFEST.json').read_text())
    for x in manifest['reused_b1_cells']:
        src=Path(x['source']);saved=json.loads((src/'metrics.json').read_text())
        ref=original_reference(x['dataset']);k=ref['columns'].index('OT');origins=np.arange(int(.7*ref['rows'])-1,int(.8*ref['rows'])-6)
        target=np.stack([ref['target'][t+1:t+7,k:k+1] for t in origins]);mask=np.stack([ref['valid'][t+1:t+7,k:k+1] for t in origins])
        with np.load(src/'predictions.npz') as z:
            same(z['origins'],origins,'Reused origins',exact=True);same(z['targets'],target,'Reused target');same(z['valid_mask'],mask,'Reused mask',exact=True);score=metrics(z['predictions'],target,mask)
        for metric in ['mse','mae']:require(abs(score[metric]-saved[metric])<1e-12,'Reused score')
        rows.append(dict(dataset=x['dataset'],seed=x['seed'],method=x['system'],mse=score['mse'],mae=score['mae'],source=str(src),new_evaluation=False))
    runs=pd.DataFrame(rows);require(len(runs)==60 and not runs.duplicated(['dataset','seed','method']).any(),'Wrong B1 matrix')
    out=base/'reports';out.mkdir(exist_ok=True);runs.to_csv(out/'runs.csv',index=False)
    summary=runs.groupby(['dataset','method'],sort=False).agg(n=('seed','count'),mse_mean=('mse','mean'),mse_sd=('mse','std'),mae_mean=('mae','mean'),mae_sd=('mae','std')).reset_index();summary.to_csv(out/'summary.csv',index=False)
    pairs=[]
    for (d,s),g in runs.groupby(['dataset','seed']):
        index=g.set_index('method')
        for candidate,control in [('iadapter_online','iadapter_U1F0'),('iadapter_U0F1','iadapter_static'),('text_add_online','numeric_online'),('text_add_online','text_add_static')]:
            pairs.append(dict(dataset=d,seed=s,candidate=candidate,control=control,delta_mse=index.loc[candidate,'mse']-index.loc[control,'mse'],delta_mae=index.loc[candidate,'mae']-index.loc[control,'mae']))
    pairs=pd.DataFrame(pairs);pairs.to_csv(out/'paired_runs.csv',index=False)
    gates=[]
    for candidate,control,claim in [('iadapter_online','iadapter_U1F0','C1_feedback'),('text_add_online','numeric_online','C2_initial_text')]:
        for d in ['Energy','SocialGood']:
            a=summary[(summary.dataset==d)&(summary.method==candidate)].iloc[0];b=summary[(summary.dataset==d)&(summary.method==control)].iloc[0];p=pairs[(pairs.dataset==d)&(pairs.candidate==candidate)&(pairs.control==control)]
            mse=(a.mse_mean/b.mse_mean-1)*100;mae=(a.mae_mean/b.mae_mean-1)*100;wins=int((p.delta_mse<0).sum())
            gates.append(dict(claim=claim,dataset=d,mse_change_percent=mse,mae_change_percent=mae,mse_improved_seeds=wins,passes=mse<=-2 and mae<=1 and wins>=2))
    pd.DataFrame(gates).to_csv(out/'gates.csv',index=False);write_json(out/'validation.json',dict(status='passed',new_evaluations=30,reused_evaluations=30,original_csv_verified_cells=60))
    (out/'RESULTS.md').write_text('# B1 common-OT controls and deployment feedback\n\n'+md_table(summary.round(6))+'\n\n## Predeclared screening gates\n\n'+md_table(pd.DataFrame(gates).round(6))+'\n\nDevelopment only; same warmed full weights. U/F contrasts isolate deployment changes, not entire-training feedback necessity. Static GPT4MTS retained.\n')
    print('B1_VALIDATED 60 cells; 30 new',flush=True);print(pd.DataFrame(gates).to_string(index=False),flush=True)
    return gates

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--limit',type=int);p.add_argument('--analyze',action='store_true');a=p.parse_args();analyze(a.output) if a.analyze else campaign(a.output,a.limit)
