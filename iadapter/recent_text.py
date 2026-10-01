"""N1 origin-traceable same-budget recent text, following existing B1 controls."""
import argparse,gc,json,time,traceback
from pathlib import Path
from dataclasses import replace
import numpy as np
import pandas as pd
import torch
from transformers import BertModel,BertTokenizerFast
from .config import ROOT
from .text_order import recent_entry
from .replan_audit import reconstruct,OUTPUT as OLD
from .prepare import weight_directory,contextual_pool
from .external_static import ot_data,evaluate
from .feedback_factorial import load_bone
from .review_sprint import initialize_adapter,write_events
from .revision_candidate import DonorBackend
from .shared_online import OnlineRunner
from .engine import Stream
from .util import seed_all,write_json,revision,metrics,sync
from .audit import require,same,original_reference
from .analyze import md_table
OUT=ROOT/'experiments/continuation_20261001'

def jobs():return [dict(dataset=d,seed=s) for d in ['Energy','SocialGood'] for s in [2023,2024,2025]]
def key(j):return f"{j['dataset']}_s{j['seed']}"

def input_rows(name):
    data,_=ot_data(name,with_embeddings=True);frame=pd.read_csv(data.source);datecol=next(c for c in ['end_date','date','Date','start_date'] if c in frame);frame=frame.iloc[np.argsort(pd.to_datetime(frame[datecol]).values,kind='stable')].reset_index(drop=True);old=reconstruct(frame,details=True)[:len(data.x)];new=[recent_entry(x) for x in old];require([x['text'] for x in old]==data.texts,'Original text selection changed');return data,old,new

def visible(tok,text,spans):
    encoded=tok(text,truncation=True,max_length=64,return_offsets_mapping=True);offsets=[(a,b) for a,b in encoded['offset_mapping'] if b>a];return [rid for rid,a,b in spans if any(a<j and b>i for i,j in offsets)],len(encoded['input_ids']) if text else 0

@torch.no_grad()
def prepare(base=OUT):
    base=Path(base);out=base/'recent_cache';out.mkdir(parents=True,exist_ok=False);require(not revision()['dirty'],'Commit source');torch.set_num_threads(1);seed_all(2023);tok=BertTokenizerFast.from_pretrained(weight_directory(),local_files_only=True);model=BertModel.from_pretrained(weight_directory(),local_files_only=True).eval().to('cuda');model.requires_grad_(False);require(len(model.encoder.layer)==12,'Incomplete BERT');coverage=[]
    for name in ['Energy','SocialGood']:
        data,old,new=input_rows(name);texts=[e['text'] for e in new];rows=[]
        for t,(a,b) in enumerate(zip(old,new)):
            cur,ct=visible(tok,a['text'],a['spans']);rec,rt=visible(tok,b['text'],b['spans']);latest=max([int(r.split(':')[0]) for r in data.text_ids[t]],default=-1);require(set(rec)<=set(data.text_ids[t]) and all(int(r.split(':')[0])<=t for r in rec),'Future/wrong visible ID');rows.append(dict(origin=t,split='train' if t<data.train_end else 'selection' if t<data.val_end else 'development',selected_ids=data.text_ids[t],current_visible_ids=cur,recent_visible_ids=rec,current_tokens=ct,recent_tokens=rt,reordered=b['reordered'],available=bool(texts[t]),current_latest_visible=any(int(r.split(':')[0])==latest for r in cur),recent_latest_visible=any(int(r.split(':')[0])==latest for r in rec),current_text=a['text'],recent_text=b['text']))
        embeddings=np.zeros((len(texts),768),dtype=np.float32);active=[i for i,t in enumerate(texts) if t];cursor=0;batch=8;sync('cuda');start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        while cursor<len(active):
            ids=active[cursor:cursor+batch]
            try:embeddings[ids]=contextual_pool(model,tok,[texts[i] for i in ids],'cuda')
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache();require(batch>1,'BERT OOM at1');batch=max(1,batch//2);continue
            cursor+=len(ids)
        sync('cuda');seconds=time.perf_counter()-start
        for i in active[:2]:np.testing.assert_allclose(embeddings[i],contextual_pool(model,tok,[texts[i]],'cuda')[0],rtol=3e-4,atol=3e-5)
        require(np.isfinite(embeddings).all(),'Nonfinite encoding');np.savez_compressed(out/f'{name}.npz',embeddings=embeddings,texts=np.asarray(texts));write_json(out/f'{name}_provenance.json',rows);write_json(out/f'{name}_encoding.json',dict(status='completed',seconds=seconds,layers=12,max_length=64,batch=batch,peak_allocated_bytes=torch.cuda.max_memory_allocated(),code=revision()))
        for split in ['train','selection','development']:
            g=[r for r in rows if r['split']==split];a=[r for r in g if r['available']];coverage.append(dict(dataset=name,split=split,rows=len(g),available=len(a),reordered=sum(r['reordered'] for r in g),current_latest_not_visible=sum(not r['current_latest_visible'] for r in a),recent_latest_not_visible=sum(not r['recent_latest_visible'] for r in a)))
        print('ENCODED_RECENT',name,seconds,flush=True)
    pd.DataFrame(coverage).to_csv(out/'coverage.csv',index=False);write_json(out/'validation.json',dict(status='passed',char_inventory_unchanged=True,missing_text_unchanged=True,code=revision()));del model,tok;gc.collect();torch.cuda.empty_cache()

def validate(j,directory,online):
    ref=original_reference(j['dataset']);k=ref['columns'].index('OT');origins=np.arange(int(.7*ref['rows'])-1,int(.8*ref['rows'])-6);ys=np.stack([ref['target'][t+1:t+7,k:k+1] for t in origins]);mask=np.stack([ref['valid'][t+1:t+7,k:k+1] for t in origins]);saved=json.loads((directory/'metrics.json').read_text());require(json.loads((directory/'status.json').read_text())['status']=='completed','Incomplete mode')
    with np.load(directory/'predictions.npz') as z:same(z['origins'],origins,'Origins',exact=True);same(z['targets'],ys,'Original CSV targets');same(z['valid_mask'],mask,'Original masks',exact=True);score=metrics(z['predictions'],ys,mask)
    for m in ['mse','mae']:require(abs(score[m]-saved[m])<1e-12,'Metric mismatch')
    if online:
        events=[json.loads(e) for e in (directory/'updates.jsonl').read_text().splitlines()];require([e['origin'] for e in events]==origins[:-6].tolist(),'Missing/repeated mature record');require(all(e['mature_at']==e['origin']+6==e['update_at'] and e['text_origin']==e['state_origin']==e['origin'] and e['text_ids']==ref['text_ids'][e['origin']] for e in events),'Wrong replay context')
    return score

def save(directory,arrays,events,contexts,stats,ad,warm,frozen,model,backend=None):
    require(all(p.grad is None and torch.equal(p.detach().cpu(),frozen[n]) for n,p in model.named_parameters()),'Backbone changed');delta=sum(float((v.detach().cpu().double()-warm[n].double()).square().sum()) for n,v in ad.state_dict().items())**.5;stats.update(frozen_unchanged=True,parameter_displacement_l2=delta,trainable_parameters=sum(p.numel() for p in ad.parameters()));np.savez_compressed(directory/'predictions.npz',**arrays);write_events(directory/'updates.jsonl',events);write_json(directory/'contexts.json',contexts);write_json(directory/'metrics.json',stats);torch.save(ad.state_dict(),directory/'final_adapter.pt')
    if backend is not None:
        np.savez_compressed(directory/'donor.npz',origins=arrays['origins'],predictions=np.stack([p['prediction'] for p in backend.probes]),targets=arrays['targets'],valid_mask=arrays['valid_mask'],donor_origins=np.asarray([p['donor_origin'] for p in backend.probes]));write_json(directory/'donor_context.json',[{k:v for k,v in p.items() if k!='prediction'} for p in backend.probes])
    write_json(directory/'status.json',{'status':'completed'})

def run_group(j,base):
    out=base/'runs'/key(j);out.mkdir(parents=True,exist_ok=False);write_json(out/'status.json',{'status':'running'})
    try:
        c,data,model,checkpoint,bounds=load_bone(j,with_text=True);frozen={n:p.detach().cpu().clone() for n,p in model.named_parameters()};source=OLD/'b1/runs'/f"{key(j)}_text_add_static";old_cfg=json.loads((source/'config.json').read_text());require(old_cfg['data']==data.metadata() and old_cfg['config']==c.to_dict(),'Different old control');ad=initialize_adapter(c,1);initial={n:v.detach().cpu().clone() for n,v in ad.state_dict().items()};ad.load_state_dict(torch.load(source/'initial_adaptive_state.pt',map_location=c.device,weights_only=True));warm={n:v.detach().cpu().clone() for n,v in ad.state_dict().items()};directory=out/'current_diagnostic';directory.mkdir();seed_all(c.seed+20000);backend=DonorBackend(c,data,model,ad,c.seed);arrays,events,contexts,stats=OnlineRunner(c,data,backend).run()
        with np.load(OLD/'b1/runs'/f"{key(j)}_text_add_online"/'predictions.npz') as z:same(arrays['origins'],z['origins'],'Current original origins',exact=True);same(arrays['predictions'],z['predictions'],'Current probe changed original forecasts')
        save(directory,arrays,events,contexts,stats,ad,warm,frozen,model,backend);validate(j,directory,True)
        with np.load(base/'recent_cache'/f"{j['dataset']}.npz") as z:texts=z['texts'].tolist();embedding=z['embeddings'].copy()
        require(len(texts)==len(data.x) and [bool(t) for t in texts]==[bool(t) for t in data.texts],'Changed missing-text set');data=replace(data,texts=texts,embeddings=embedding);c=replace(c,protocol='same_chars_recent_first_ot_v1');ad=initialize_adapter(c,1);require(all(torch.equal(v.detach().cpu(),initial[n]) for n,v in ad.state_dict().items()),'Different initializer');torch.save(initial,out/'initial_adapter.pt');seed_all(c.seed+11000);_,ev,stats=Stream(c,data,model,ad).run('train');warm={n:v.detach().cpu().clone() for n,v in ad.state_dict().items()};torch.save(warm,out/'warm_adapter.pt');write_events(out/'warm_updates.jsonl',ev);write_json(out/'warm_metrics.json',stats);write_json(out/'config.json',dict(**j,config=c.to_dict(),data=data.metadata(),boundaries=bounds,checkpoint=str(checkpoint),code=revision(),old_source=str(source),input_change='Only retained row-block order; same character inventory/availability/64-token cap'))
        for mode in ['static','online']:
            directory=out/f'recent_{mode}';directory.mkdir();ad.load_state_dict(warm);seed_all(c.seed+20000);backend=None
            if mode=='static':arrays,contexts=evaluate(data,c,model,ad);events=[];stats=metrics(arrays['predictions'],arrays['targets'],arrays['valid_mask'])
            else:
                backend=DonorBackend(c,data,model,ad,c.seed);arrays,events,contexts,stats=OnlineRunner(c,data,backend).run()
                with np.load(out/'recent_static/predictions.npz') as z:same(arrays['predictions'][:7],z['predictions'][:7],'Before-update mismatch')
            save(directory,arrays,events,contexts,stats,ad,warm,frozen,model,backend);require(stats['parameter_displacement_l2']==0. if mode=='static' else stats['parameter_displacement_l2']>0.,'Wrong update mode');validate(j,directory,mode=='online')
        write_json(out/'status.json',{'status':'completed'});del ad,model,backend;gc.collect();torch.cuda.empty_cache();print('RECENT_COMPLETE',key(j),flush=True)
    except Exception:write_json(out/'status.json',dict(status='failed',error=traceback.format_exc()));raise

def analyze(base=OUT):
    base=Path(base);out=base/'reports';out.mkdir(exist_ok=True);rows=[];donors=[]
    for j in jobs():
        directory=base/'runs'/key(j);require(json.loads((directory/'status.json').read_text())['status']=='completed','Incomplete group');require(not json.loads((directory/'config.json').read_text())['code']['dirty'],'Dirty run')
        for rep in ['current','recent']:
            for mode in ['static','online']:
                p=OLD/'b1/runs'/f'{key(j)}_text_add_{mode}' if rep=='current' else directory/f'recent_{mode}';score=validate(j,p,mode=='online');rows.append(dict(**j,representation=rep,mode=mode,reused=rep=='current',**score))
            p=directory/('current_diagnostic' if rep=='current' else 'recent_online');validate(j,p,True)
            with np.load(p/'donor.npz') as z:
                ref=original_reference(j['dataset']);k=ref['columns'].index('OT');origins=np.arange(int(.7*ref['rows'])-1,int(.8*ref['rows'])-6);same(z['origins'],origins,'Donor origins',exact=True);target=np.stack([ref['target'][t+1:t+7,k:k+1] for t in origins]);mask=np.stack([ref['valid'][t+1:t+7,k:k+1] for t in origins]);same(z['targets'],target,'Donor targets');same(z['valid_mask'],mask,'Donor masks',exact=True);require((z['donor_origins']<ref['train_end']).all(),'Future donor');score=metrics(z['predictions'],target,mask)
            donors.append(dict(**j,representation=rep,**score))
    frame=pd.DataFrame(rows);frame.to_csv(out/'runs.csv',index=False);pd.DataFrame(donors).to_csv(out/'donor_runs.csv',index=False);summary=frame.groupby(['dataset','representation','mode']).agg(n=('seed','count'),mse_mean=('mse','mean'),mse_sd=('mse','std'),mae_mean=('mae','mean'),mae_sd=('mae','std')).reset_index();summary.to_csv(out/'summary.csv',index=False);paired=[]
    for d in ['Energy','SocialGood']:
        for s in [2023,2024,2025]:
            i=frame[(frame.dataset==d)&(frame.seed==s)].set_index(['representation','mode'])
            for m in ['static','online']:paired.append(dict(dataset=d,seed=s,mode=m,delta_mse=i.loc[('recent',m),'mse']-i.loc[('current',m),'mse'],delta_mae=i.loc[('recent',m),'mae']-i.loc[('current',m),'mae']))
    p=pd.DataFrame(paired);p.to_csv(out/'paired_runs.csv',index=False);p.groupby(['dataset','mode']).agg(n=('seed','count'),delta_mse_mean=('delta_mse','mean'),delta_mse_sd=('delta_mse','std'),delta_mae_mean=('delta_mae','mean'),delta_mae_sd=('delta_mae','std'),mse_improved_seeds=('delta_mse',lambda v:int((v<0).sum()))).reset_index().to_csv(out/'paired_summary.csv',index=False)
    refs=pd.read_csv(OLD/'b1/reports/runs.csv');refs[refs.method.isin(['frozen_backbone','numeric_online','gpt4mts_static'])].to_csv(out/'strong_references.csv',index=False);llm=pd.read_csv(OLD/'exploratory_fusion/reports/runs.csv');llm[(llm.representation=='LLM5')&(llm['mode']=='online')].to_csv(out/'LLM5_reference.csv',index=False);write_json(out/'validation.json',dict(status='passed',new_evaluations=12,reused_current_evaluations=12,current_diagnostic_replay_executions=6,warmups=6,donor_sequences=12,scope='Post-results exploratory input-order study; not independent confirmation'));(out/'RESULTS.md').write_text('# Same-character recent-first comparison\n\n'+md_table(summary.round(6))+'\n');print(summary.to_string(index=False),flush=True)

def campaign(base=OUT,limit=None):
    base=Path(base);require(not revision()['dirty'],'Commit source');require(json.loads((base/'recent_cache/validation.json').read_text())['status']=='passed','Missing cache');torch.set_num_threads(1);out=base/'runs';out.mkdir(exist_ok=True);planned=jobs();code=revision();path=out/'queue_state.json';state=json.loads(path.read_text()) if path.exists() else dict(code=code,planned=planned,jobs={key(j):'pending' for j in planned});require(state['code']==code and state['planned']==planned,'Queue changed');count=0
    for j in planned:
        if state['jobs'][key(j)]=='completed':continue
        require(state['jobs'][key(j)]=='pending','Unresolved failed group');state['jobs'][key(j)]='running';write_json(path,state)
        try:run_group(j,base);state['jobs'][key(j)]='completed'
        except Exception:state['jobs'][key(j)]='failed';write_json(path,state);raise
        write_json(path,state);count+=1
        if limit is not None and count>=limit:break
    if all(v=='completed' for v in state['jobs'].values()):return analyze(base)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','campaign','analyze'],required=True);p.add_argument('--limit',type=int);a=p.parse_args();campaign(limit=a.limit) if a.stage=='campaign' else globals()[a.stage]()
