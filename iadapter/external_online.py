"""Online extensions starting exactly from the completed static checkpoints."""
import argparse,gc,json,time,traceback,subprocess
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from .config import ROOT
from .models import Backbone
from .external_static import ot_data,pilot_config,OUTPUT as PILOT
from .external_static_matrix import OUTPUT as STATIC_MATRIX,groups,group_id,group_path
from .native_gpt4mts import build as native_build,native_config
from .review_sprint import initialize_adapter,write_events
from .shared_online import InternalBackend,NativeBackend,OnlineRunner
from .util import seed_all,revision,write_json

OUTPUT=ROOT/'experiments/external_online_20260929'
METHODS={'numeric_online':'numeric_static','iadapter_online':'iadapter_static','gpt4mts_online_extension':'gpt4mts_static'}

def require_tracked_sources():
    files=['iadapter/shared_online.py','iadapter/external_online.py','iadapter/external_online_analysis.py']
    result=subprocess.run(['git','ls-files','--error-unmatch','--']+files,cwd=ROOT,capture_output=True,text=True)
    if result.returncode:raise RuntimeError('Online implementation modules must be tracked and committed before deployment')

def load_start(job,method):
    static_method=METHODS[method];source=group_path(STATIC_MATRIX,job)/'runs'/static_method
    assert json.loads((source/'status.json').read_text())['status']=='completed'
    static_cfg=json.loads((source/'config.json').read_text());static_stats=json.loads((source/'metrics.json').read_text())
    c=replace(pilot_config('numeric_add' if method=='numeric_online' else 'full',**job),protocol='development_60_10_10_online_extension_v1')
    data,bounds=ot_data(job['dataset'],with_embeddings=method=='iadapter_online')
    assert static_cfg['data']==data.metadata() and static_cfg['boundaries']==bounds
    assert static_cfg['seed']==c.seed and static_cfg['horizon']==c.horizon and static_cfg['lookback']==c.lookback
    checkpoint=Path(static_stats['backbone_checkpoint']);blob=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if method=='gpt4mts_online_extension':
        assert blob['model_config']==vars(native_config(c.lookback,c.horizon))
        model=native_build(c.lookback,c.horizon,c.device)
        named={n:p for n,p in model.named_parameters() if p.requires_grad}
        assert set(named)==set(blob['trainable'])
        with torch.no_grad():
            for name,p in named.items():p.copy_(blob['trainable'][name].to(c.device))
        cache=(PILOT if c.dataset=='Economy' else STATIC_MATRIX/'text_cache'/c.dataset)/'gpt4mts_cls512.npz'
        with np.load(cache,allow_pickle=False) as z:
            assert z['texts'].tolist()==data.texts and z['embeddings'].shape==(len(data.x),768)
            embeddings=z['embeddings'].copy()
        backend=NativeBackend(c,data,model,embeddings);expected=blob['trainable']
        adaptive={n:p for n,p in model.named_parameters() if p.requires_grad}
    else:
        bc=blob['config'];assert bc['dataset']==c.dataset and bc['seed']==c.seed and bc['columns']==['OT']
        model=Backbone(c.lookback,c.horizon,1,d_model=c.d_model,e_layers=c.e_layers,dropout=c.dropout).to(c.device)
        model.load_state_dict(blob['model']);adapter=initialize_adapter(c,1)
        expected=torch.load(source/'warm_adapter.pt',map_location='cpu',weights_only=True);adapter.load_state_dict(expected)
        backend=InternalBackend(c,data,model,adapter);adaptive=adapter.state_dict()
    initial={n:p.detach().cpu().clone() for n,p in adaptive.items()}
    assert set(initial)==set(expected) and all(torch.equal(initial[n],v) for n,v in expected.items()),'Starting parameters differ from static experiment'
    frozen={n:p.detach().cpu().clone() for n,p in model.named_parameters() if not p.requires_grad}
    return c,data,backend,source,static_cfg,initial,frozen

def run_cell(job,method,base):
    directory=base/'groups'/group_id(job)/method;directory.mkdir(parents=True,exist_ok=False)
    write_json(directory/'status.json',{'status':'running'});started=time.perf_counter()
    try:
        c,data,backend,source,static_cfg,initial,frozen=load_start(job,method)
        config=dict(dataset=c.dataset,seed=c.seed,horizon=c.horizon,lookback=c.lookback,method=method,protocol=c.protocol,data=data.metadata(),static_source=str(source),static_source_config=static_cfg,code=revision(),optimizer=dict(name='Adam',lr=c.lr,weight_decay=c.weight_decay,grad_clip=c.grad_clip,batch=1),prediction_before_update=True,has_gradient_feedback_cache=backend.has_feedback)
        write_json(directory/'config.json',config);torch.save(initial,directory/'initial_adaptive_state.pt')
        seed_all(c.seed+20000);torch.cuda.reset_peak_memory_stats()
        arrays,events,contexts,stats=OnlineRunner(c,data,backend).run()
        assert all(p.grad is None and torch.equal(p.detach().cpu(),frozen[n]) for n,p in backend.model.named_parameters() if not p.requires_grad),'Frozen parameters changed'
        final={n:p.detach().cpu().clone() for n,p in (backend.adapter.state_dict().items() if backend.has_feedback else backend.model.named_parameters()) if n in initial}
        assert set(final)==set(initial)
        if stats['updated']:assert any(not torch.equal(initial[n],final[n]) for n in initial),'No adaptive parameter changed'
        with np.load(source/'predictions.npz') as z:
            np.testing.assert_array_equal(arrays['origins'],z['origins']);np.testing.assert_array_equal(arrays['targets'],z['targets']);np.testing.assert_array_equal(arrays['valid_mask'],z['valid_mask'])
            prefix=c.horizon+1
            np.testing.assert_allclose(arrays['predictions'][:prefix],z['predictions'][:prefix],rtol=1e-6,atol=1e-6)
            first_difference=float(np.max(np.abs(arrays['predictions'][:prefix]-z['predictions'][:prefix])))
        np.savez_compressed(directory/'predictions.npz',**arrays);write_events(directory/'updates.jsonl',events);write_json(directory/'contexts.json',contexts)
        torch.save(final,directory/'final_adaptive_state.pt')
        stats.update(config=config,seconds=time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),initial_parameters_match_static=True,frozen_parameters_unchanged=True,first_unupdated_prediction_max_difference=first_difference,trainable_parameters=sum(p.numel() for p in backend.parameters()),static_mse=json.loads((source/'metrics.json').read_text())['mse'])
        write_json(directory/'metrics.json',stats);write_json(directory/'status.json',{'status':'completed'})
        print(f'ONLINE_COMPLETE {group_id(job)}/{method} mse={stats["mse"]:.6f} mae={stats["mae"]:.6f} updates={stats["updated"]}',flush=True)
        del backend,initial,frozen,final;gc.collect();torch.cuda.empty_cache();return stats
    except Exception:
        write_json(directory/'status.json',{'status':'failed','error':traceback.format_exc()});raise

def campaign(base=OUTPUT,limit_groups=None):
    require_tracked_sources()
    torch.set_num_threads(1);base=Path(base);code=revision()
    assert not code['dirty'],'Commit online implementation first'
    base.mkdir(parents=True,exist_ok=True)
    validated=json.loads((STATIC_MATRIX/'reports/validation.json').read_text());assert validated['status']=='passed' and validated['completed_cells']==75
    path=base/'queue_state.json';planned=groups()
    if path.exists():
        state=json.loads(path.read_text());assert state['code']==code and state['groups']==planned and state['methods']==METHODS
    else:state=dict(code=code,groups=planned,methods=METHODS,jobs={group_id(j)+'/'+m:{'status':'pending'} for j in planned for m in METHODS})
    completed_now=0
    for job in planned:
        needed=any(state['jobs'][group_id(job)+'/'+m]['status']!='completed' for m in METHODS)
        for method in METHODS:
            key=group_id(job)+'/'+method;entry=state['jobs'][key]
            if entry['status']=='completed':continue
            assert entry['status']=='pending','Unresolved online cell '+key
            entry.update(status='running',started_at=time.time());write_json(path,state)
            try:
                result=run_cell(job,method,base);entry.update(status='completed',mse=result['mse'],mae=result['mae'])
            except Exception:
                entry.update(status='failed',error=traceback.format_exc());write_json(path,state);raise
            write_json(path,state)
        from .external_online_analysis import audit_group
        audit_group(base,job)
        if needed:completed_now+=1
        if limit_groups is not None and completed_now>=limit_groups:break
    print('ONLINE_QUEUE',sum(v['status']=='completed' for v in state['jobs'].values()),'/75',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=OUTPUT);parser.add_argument('--limit-groups',type=int)
    args=parser.parse_args();campaign(args.output,args.limit_groups)
