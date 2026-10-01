"""Isolated same-bone cost, including both branches; not deployment throughput."""
import argparse,gc,json
from pathlib import Path
from dataclasses import replace
import numpy as np
import pandas as pd
import torch
from .calibration_pilot import OUT,full_ot,features,check_sources,pair_ledger,gate_for,bounds
from .feedback_factorial import load_bone
from .revision_candidate import initialize_five
from .review_sprint import initialize_adapter
from .shared_online import InternalBackend,Record
from .efficiency import measure
from .util import revision,write_json,seed_all
from .audit import require

def profile(base=OUT):
    base=Path(base);require(not revision()['dirty'],'Commit profiler');torch.set_num_threads(1)
    locked=json.loads((base/'selection/locked.json').read_text());require(locked['code']==revision(),'Rule source changed')
    out=base/'cost';out.mkdir(exist_ok=False);rows=[]
    for dataset in ['Energy','SocialGood']:
        j=dict(dataset=dataset,seed=2023);data=full_ot(dataset);vals=features(data,j,base);sources=check_sources(j)
        for policy in ['numeric_only','full_online','numeric_online_text_online','numeric_online_text_static']:
            for repeat in range(3):
                seed_all(2023+repeat);c,old,model,checkpoint,previous=load_bone(j,True)
                frozen={n:p.detach().cpu().clone() for n,p in model.named_parameters()};t=bounds(len(data.x))['selection_end']-1
                specs=[('full',data.embeddings)] if policy=='full_online' else [('numeric',None)] if policy=='numeric_only' else [('numeric',None),('LLM5',vals['LLM5'])]
                backends=[];optimizers=[];records=[];adapters=[];parameters=0
                y=torch.from_numpy(data.target[t+1:t+7]).unsqueeze(0).to(c.device);mask=torch.from_numpy(data.valid[t+1:t+7]).unsqueeze(0).to(c.device)
                for name,emb in specs:
                    cfg=replace(c,variant='full' if name=='full' else 'numeric_add' if name=='numeric' else 'text_add')
                    ad=initialize_five(cfg,1) if name=='LLM5' else initialize_adapter(cfg,1)
                    ad.load_state_dict(torch.load(sources[f'{name}_warm'],map_location='cpu',weights_only=True))
                    backend=InternalBackend(cfg,replace(data,embeddings=emb),model,ad);p,k,ids,g=backend.predict(t)
                    backends.append(backend);adapters.append(ad);records.append(Record(t,t+6,ids,k,p.detach().clone()))
                    train_this=not (policy.endswith('text_static') and name=='LLM5')
                    optimizers.append(torch.optim.Adam(backend.parameters(),lr=c.lr,weight_decay=c.weight_decay) if train_this else None)
                    parameters+=sum(v.numel() for v in ad.parameters())
                cal=pair_ledger(j,'calibration','LLM5','static','static',base)
                gate=gate_for('ls',locked['chosen']['ls']['setting'],cal,bounds(len(data.x))['calibration_end']-1)
                n=records[0].issued_prediction[0].cpu().numpy().astype(np.float64);m=records[-1].issued_prediction[0].cpu().numpy().astype(np.float64)
                y_np=data.target[t+1:t+7].astype(np.float64);mask_np=data.valid[t+1:t+7]
                def predict():
                    results=[b.predict(t)[0][0] for b in backends]
                    if len(results)==2:
                        a=gate.alpha();n_now,m_now=[v.cpu().numpy().astype(np.float64) for v in results]
                        return n_now+a[:,None]*(m_now-n_now)
                    return results[0]
                def update():
                    for b,opt,record in zip(backends,optimizers,records):
                        if opt is None:continue
                        opt.zero_grad(set_to_none=True);total,loss,feedback=b.loss(record,y,mask)
                        require(bool(torch.isfinite(total)),'Nonfinite benchmark loss');total.backward()
                        torch.nn.utils.clip_grad_norm_(b.parameters(),c.grad_clip,error_if_nonfinite=True);opt.step();b.after_update(record,feedback)
                    if len(backends)==2:gate.observe(n,m,y_np,mask_np)
                torch.cuda.reset_peak_memory_stats();p=measure(predict,c.device,50,200);u=measure(update,c.device,50,200)
                require(all(p.grad is None and torch.equal(p.detach().cpu(),frozen[name]) for name,p in model.named_parameters()),'Benchmark backbone changed')
                # The moment state is cheap; this measures CPU computation separately.
                gpred=measure(gate.alpha,'cpu',50,200) if len(backends)==2 else dict(median_ms=0.)
                gup=measure(lambda:gate.observe(n,m,y_np,mask_np),'cpu',50,200) if len(backends)==2 else dict(median_ms=0.)
                rows.append(dict(dataset=dataset,policy=policy,repeat=repeat+1,origin=t,adapter_parameters=parameters,
                    backbone_parameters=sum(v.numel() for v in model.parameters()),predict_median_ms=p['median_ms'],predict_p95_ms=p['p95_ms'],
                    update_median_ms=u['median_ms'],update_p95_ms=u['p95_ms'],peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    gate_predict_median_ms=gpred['median_ms'],gate_update_median_ms=gup['median_ms'],
                    gate_float64_payload_bytes=sum(v.nbytes for v in [gate.A,gate.B,gate.alpha0,gate.lam]) if len(backends)==2 else 0,
                    branch_gradient_payload_bytes=sum(b.hg.numel()*b.hg.element_size()+b.ge.numel()*b.ge.element_size() for b in backends),
                    frozen_feature_cache_bytes=vals['LLM5'].nbytes if len(backends)==2 else data.embeddings.nbytes if policy=='full_online' else 0,
                    code=revision(),warmup=50,samples=200,hardware=torch.cuda.get_device_name()))
                write_json(out/'rounds.json',rows);print('CALIBRATION_COST',dataset,policy,repeat+1,flush=True)
                del backends,adapters,optimizers,records,model,ad,backend,k,y,mask;gc.collect();torch.cuda.empty_cache()
    frame=pd.DataFrame(rows);frame.drop(columns=['code']).to_csv(out/'rounds.csv',index=False)
    summary=frame.groupby(['dataset','policy']).agg(rounds=('repeat','count'),adapter_parameters=('adapter_parameters','first'),backbone_parameters=('backbone_parameters','first'),prediction_median_ms=('predict_median_ms','median'),prediction_p95_ms=('predict_p95_ms','median'),update_median_ms=('update_median_ms','median'),update_p95_ms=('update_p95_ms','median'),peak_allocated_bytes_max=('peak_allocated_bytes','max'),gate_predict_median_ms=('gate_predict_median_ms','median'),gate_update_median_ms=('gate_update_median_ms','median'),gate_float64_payload_bytes=('gate_float64_payload_bytes','first'),branch_gradient_payload_bytes=('branch_gradient_payload_bytes','first'),frozen_feature_cache_bytes=('frozen_feature_cache_bytes','first')).reset_index();summary.to_csv(out/'summary.csv',index=False)
    write_json(out/'validation.json',dict(status='passed',rounds=24,formal_weight_or_prediction_files_modified=False,frozen_backbones_unchanged=True,
        scope='Isolated warmed copies; real first assessment-origin inputs and fixed mature record; repeated updates only for timing. No same-state content probes. Includes both independently computed numerical paths and GPU-to-CPU transfer for mix. Excludes queue/logging/input loading and offline encoding/generation; not deployment throughput.',generation=json.loads((base/'generation_status.json').read_text())))
    print(summary.to_string(index=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,default=OUT);a=p.parse_args();profile(a.base)
