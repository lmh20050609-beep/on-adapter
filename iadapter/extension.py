"""Locked five-dimensional text candidate on additional original Time-MMD domains."""
import argparse,gc,json,time,traceback,hashlib
from pathlib import Path
from dataclasses import replace
from unittest import mock
import numpy as np
import pandas as pd
import torch
from .config import ROOT,Config
from .data import load_data
from .prepare import prepare as prepare_bert,weight_directory
from transformers import BertTokenizerFast
from . import task_text_prior as prior
from .text_signal_probe import pca_basis,standardize_train
from .revision_candidate import initialize_five
from .review_sprint import initialize_adapter,write_events
from .run import fit_backbone
from .feedback_factorial import frozen_predictions
from .external_static import evaluate
from .shared_online import InternalBackend,OnlineRunner
from .engine import Stream
from .util import seed_all,revision,write_json,metrics
from .audit import require,same,original_reference
from .analyze import md_table
OUT=ROOT/'experiments/extension_20261001'
CONTEXT={'Agriculture':'U.S. retail broiler composite price, sampled monthly; forecast the next six reported monthly observations.','Health':'U.S. national unweighted percentage of outpatient visits for influenza-like illness (ILI), reported weekly; forecast the next six reported observations.'}

def ot_data(name):
    raw=load_data(name,with_embeddings=True);k=raw.columns.index('OT');sl=slice(k,k+1)
    return replace(raw,x=raw.x[:,sl].copy(),target=raw.target[:,sl].copy(),valid=raw.valid[:,sl].copy(),input_missing=raw.input_missing[:,sl].copy(),columns=['OT'],mean=raw.mean[sl].copy(),scale=raw.scale[sl].copy())

def prepare(base=OUT):
    base=Path(base);base.mkdir(parents=True,exist_ok=True);require(not revision()['dirty'],'Commit source');prepare_bert(list(CONTEXT),batch_size=8);tok=BertTokenizerFast.from_pretrained(weight_directory(),local_files_only=True);rows=[]
    with mock.patch.dict(prior.DOMAIN,CONTEXT):
        for d in CONTEXT:
            data=ot_data(d);contexts=[]
            for t,text in enumerate(data.texts):
                if not text:contexts.append(None);continue
                encoded=tok(text,max_length=64,truncation=True,return_offsets_mapping=True);right=max([b for a,b in encoded['offset_mapping'] if b>a],default=0);visible=text[:right];key=prior.cache_key(d,visible);row=dict(dataset=d,origin=t,split='train' if t<data.train_end else 'selection' if t<data.val_end else 'test',visible_text=visible,selected_ids=data.text_ids[t],key=key);require(all(int(r.split(':')[0])<=t for r in data.text_ids[t]),'Future selected text');contexts.append(key);rows.append(row)
            write_json(base/f'{d}_text_keys.json',contexts);write_json(base/f'{d}_data.json',data.metadata())
    write_json(base/'prior_inputs.json',rows)
    write_json(base/'candidate_lock.json',dict(status='locked',source='E003 fixedLLM5, not recent-first chosen on extension labels',system_prompt=prior.SYSTEM,domain_context=CONTEXT,model_source=json.loads((prior.WEIGHTS/'SOURCE.json').read_text()),max_new_tokens=8,greedy=True,encoder_batch=1,horizon=6,lookback=36,seed_list=list(range(2023,2028)),split='60/20/20 chronological; existing source rows retained',projection='same5->96->64 simple additive structure',code=revision(),Health_calendar='DistinctYEAR/WEEKreports retain originalsource intervals, including5duplicatedcalendarpairs; H6means6reportedobservations'))
    print('EXTENSION_INPUTS',len({r['key'] for r in rows}),flush=True)

def generate(base=OUT):
    base=Path(base);require(not revision()['dirty'],'Commit source');rows=json.loads((base/'prior_inputs.json').read_text());pool={r['key']:r for r in rows};require(len(pool)<=1885,'Extra input cap');cache_path=base/'prior_cache.json';cache=json.loads(cache_path.read_text()) if cache_path.exists() else {};budget_path=base/'generation_budget.json';budget=json.loads(budget_path.read_text()) if budget_path.exists() else dict(limit_seconds=600,used_seconds=0.);require(budget['used_seconds']<600,'Generationbudgetexhausted');seed_all(2023);start=last=time.perf_counter();model=tok=None
    try:
        tok,model,config=prior.load_model();torch.cuda.reset_peak_memory_stats()
        with mock.patch.dict(prior.DOMAIN,CONTEXT):
            for i,row in enumerate(pool.values()):
                if row['key'] in cache:continue
                result=prior.classify_batch([row],tok,model,config)[0];cache[row['key']]=result;write_json(cache_path,cache);now=time.perf_counter();budget['used_seconds']+=now-last;last=now;write_json(budget_path,budget)
                if i%200==0:print('EXTENSION_PRIOR',len(cache),'/',len(pool),flush=True)
                if budget['used_seconds']>=600:write_json(base/'generation_status.json',dict(status='stopped_budget',completed=len(cache),planned=len(pool)));return
        torch.cuda.synchronize();write_json(base/'generation_status.json',dict(status='completed',generated_unique=len(cache),planned=len(pool),invalid=sum(not r['valid'] for r in cache.values()),seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),total_input_tokens=sum(r['input_tokens'] for r in cache.values()),total_output_tokens=sum(r['output_tokens'] for r in cache.values()),encoder_batch=1,code=revision()))
    except Exception:write_json(base/'generation_status.json',dict(status='failed',completed=len(cache),error=traceback.format_exc()));raise
    finally:budget['used_seconds']+=time.perf_counter()-last;write_json(budget_path,budget);del model,tok;gc.collect();torch.cuda.empty_cache()

def representations(data,d,base):
    available=np.asarray([bool(t) for t in data.texts]);raw,pcs=pca_basis(data.embeddings,available,data.train_end,5);keys=json.loads((base/f'{d}_text_keys.json').read_text());cache=json.loads((base/'prior_cache.json').read_text());llm=np.zeros((len(data.x),5),dtype=np.float64)
    for t,key in enumerate(keys):
        if key:llm[t,prior.LABELS.index(cache[key]['label'])]=1.
    llm,ls=standardize_train(llm,np.arange(len(llm))<data.train_end);llm[~available]=0.;return raw.astype(np.float32),llm.astype(np.float32),dict(**{'pca_'+k:v for k,v in pcs.items()},llm_mean=ls['mean'],llm_scale=ls['scale'])

def jobs():return [dict(dataset=d,seed=s) for d in CONTEXT for s in range(2023,2028)]
def key(j):return f"{j['dataset']}_s{j['seed']}"

def save(directory,arrays,events,contexts,stats,c,data):
    ref=original_reference(c.dataset);k=ref['columns'].index('OT');origins=np.arange(int(.8*ref['rows'])-1,ref['rows']-6);ys=np.stack([ref['target'][t+1:t+7,k:k+1] for t in origins]);mask=np.stack([ref['valid'][t+1:t+7,k:k+1] for t in origins]);same(arrays['origins'],origins,'Extensionorigins',exact=True);same(arrays['targets'],ys,'OriginalCSVtarget');same(arrays['valid_mask'],mask,'OriginalCSVmask',exact=True);score=metrics(arrays['predictions'],ys,mask)
    for m in ['mse','mae']:require(abs(score[m]-stats[m])<1e-12,'Metricmismatch')
    if events:require([e['origin'] for e in events]==origins[:-6].tolist() and all(e['mature_at']==e['origin']+6==e['update_at'] and e['text_origin']==e['state_origin']==e['origin'] for e in events),'Maturereplay mismatch')
    np.savez_compressed(directory/'predictions.npz',**arrays);write_events(directory/'updates.jsonl',events);write_json(directory/'contexts.json',contexts);write_json(directory/'metrics.json',stats);write_json(directory/'status.json',{'status':'completed'})

def run_group(j,base):
    out=base/'groups'/key(j);out.mkdir(parents=True,exist_ok=False);write_json(out/'status.json',{'status':'running'})
    try:
        c=Config(j['dataset'],6,j['seed'],'numeric_add',protocol='extension_60_20_20_ot_v1',pretrain_batch=16);data=ot_data(c.dataset);model,checkpoint,training=fit_backbone(c,data,base/'checkpoints');model.requires_grad_(False);model.zero_grad(set_to_none=True);frozen={n:p.detach().cpu().clone() for n,p in model.named_parameters()};write_json(out/'config.json',dict(**j,config=c.to_dict(),data=data.metadata(),code=revision(),checkpoint=checkpoint,candidate_lock=str(base/'candidate_lock.json')))
        directory=out/'frozen_backbone';directory.mkdir();arrays=frozen_predictions(c,data,model);stats=metrics(arrays['predictions'],arrays['targets'],arrays['valid_mask']);save(directory,arrays,[],[],stats,c,data)
        raw,llm,transform=representations(data,c.dataset,base);np.savez_compressed(out/'transforms.npz',**transform);initial_five=None
        for representation in ['numeric','raw5','LLM5']:
            cfg=replace(c,variant='numeric_add' if representation=='numeric' else 'text_add');d=replace(data,embeddings=raw if representation=='raw5' else llm if representation=='LLM5' else None);ad=initialize_adapter(cfg,1) if representation=='numeric' else initialize_five(cfg,1);initial={n:v.detach().cpu().clone() for n,v in ad.state_dict().items()}
            if representation=='raw5':initial_five=initial
            if representation=='LLM5':require(all(torch.equal(v,initial_five[n]) for n,v in initial.items()),'Unequal5dim initializer')
            warmdir=out/f'{representation}_warm';warmdir.mkdir();torch.save(initial,warmdir/'initial_adapter.pt');seed_all(c.seed+11000);_,events,stats=Stream(cfg,d,model,ad).run('train');warm={n:v.detach().cpu().clone() for n,v in ad.state_dict().items()};torch.save(warm,warmdir/'warm_adapter.pt');write_events(warmdir/'updates.jsonl',events);write_json(warmdir/'metrics.json',stats)
            for mode in ['static','online'] if representation=='LLM5' else ['online']:
                directory=out/f'{representation}_{mode}';directory.mkdir();ad.load_state_dict(warm);seed_all(c.seed+20000)
                if mode=='static':arrays,contexts=evaluate(d,cfg,model,ad);events=[];stats=metrics(arrays['predictions'],arrays['targets'],arrays['valid_mask'])
                else:
                    arrays,events,contexts,stats=OnlineRunner(cfg,d,InternalBackend(cfg,d,model,ad)).run()
                    if representation=='LLM5':
                        with np.load(out/'LLM5_static/predictions.npz') as z:same(arrays['predictions'][:7],z['predictions'][:7],'Before-update mismatch')
                require(all(p.grad is None and torch.equal(p.detach().cpu(),frozen[n]) for n,p in model.named_parameters()),'Frozenbone changed');stats.update(trainable_parameters=sum(p.numel() for p in ad.parameters()),frozen_unchanged=True);save(directory,arrays,events,contexts,stats,cfg,d);torch.save(ad.state_dict(),directory/'final_adapter.pt')
        write_json(out/'status.json',{'status':'completed'});print('EXTENSION_COMPLETE',key(j),flush=True);del ad,model;gc.collect();torch.cuda.empty_cache()
    except Exception:write_json(out/'status.json',dict(status='failed',error=traceback.format_exc()));raise

def paired_report(frame):
    require(not frame.duplicated(['dataset','seed','method']).any(),'Duplicateextensioncell');pairs=[]
    for dataset,block in frame.groupby('dataset'):
        candidate=block[block.method=='LLM5_online'].set_index('seed')
        for control in ['raw5_online','numeric_online','LLM5_static','frozen_backbone']:
            reference=block[block.method==control].set_index('seed');require(set(candidate.index)==set(reference.index)==set(range(2023,2028)),'Incompletepairedseeds')
            require(candidate.valid_elements.nunique()==reference.valid_elements.nunique()==1 and candidate.valid_elements.iloc[0]==reference.valid_elements.iloc[0],'Unequalpairedtargets')
            for seed in sorted(candidate.index):
                a,b=candidate.loc[seed],reference.loc[seed];pairs.append(dict(dataset=dataset,seed=int(seed),candidate='LLM5_online',control=control,delta_mse=float(a.mse-b.mse),delta_mae=float(a.mae-b.mae),candidate_mse=float(a.mse),control_mse=float(b.mse),candidate_mae=float(a.mae),control_mae=float(b.mae)))
    pairs=pd.DataFrame(pairs);summary=pairs.groupby(['dataset','candidate','control']).agg(n=('seed','count'),delta_mse_mean=('delta_mse','mean'),delta_mse_sd=('delta_mse','std'),delta_mae_mean=('delta_mae','mean'),delta_mae_sd=('delta_mae','std'),mse_improved_seeds=('delta_mse',lambda x:int((x<0).sum())),mae_improved_seeds=('delta_mae',lambda x:int((x<0).sum())),candidate_mse=('candidate_mse','mean'),control_mse=('control_mse','mean'),candidate_mae=('candidate_mae','mean'),control_mae=('control_mae','mean')).reset_index();summary['mse_change_percent']=100*(summary.candidate_mse/summary.control_mse-1);summary['mae_change_percent']=100*(summary.candidate_mae/summary.control_mae-1);return pairs,summary

def analyze(base=OUT):
    base=Path(base);out=base/'reports';out.mkdir(exist_ok=True);rows=[]
    for j in jobs():
        group=base/'groups'/key(j);require(json.loads((group/'status.json').read_text())['status']=='completed','Incompleteextensiongroup')
        for method in ['frozen_backbone','numeric_online','raw5_online','LLM5_static','LLM5_online']:
            directory=group/method;require(json.loads((directory/'status.json').read_text())['status']=='completed','Incompleteextensioncell');saved=json.loads((directory/'metrics.json').read_text());ref=original_reference(j['dataset']);k=ref['columns'].index('OT');origins=np.arange(int(.8*ref['rows'])-1,ref['rows']-6);target=np.stack([ref['target'][t+1:t+7,k:k+1] for t in origins]);mask=np.stack([ref['valid'][t+1:t+7,k:k+1] for t in origins])
            with np.load(directory/'predictions.npz') as z:same(z['origins'],origins,'Origins',exact=True);same(z['targets'],target,'Target');same(z['valid_mask'],mask,'Mask',exact=True);score=metrics(z['predictions'],target,mask)
            for m in ['mse','mae']:require(abs(score[m]-saved[m])<1e-12,'Savedmetric')
            rows.append(dict(**j,method=method,**score))
    f=pd.DataFrame(rows);f.to_csv(out/'runs.csv',index=False);summary=f.groupby(['dataset','method']).agg(n=('seed','count'),mse_mean=('mse','mean'),mse_sd=('mse','std'),mae_mean=('mae','mean'),mae_sd=('mae','std')).reset_index();summary.to_csv(out/'summary.csv',index=False);pairs,ps=paired_report(f);pairs.to_csv(out/'paired_runs.csv',index=False);ps.to_csv(out/'paired_summary.csv',index=False);write_json(out/'validation.json',dict(status='passed',evaluations=50,shared_backbones=10,warmups=30,source='OriginalCSVtargets',scope='Lockedcandidateadditionaldomains; sourcehealthhas5duplicatecalendarpairs, H6reportedrecords; not originalpaperreconstruction'));(out/'RESULTS.md').write_text('# Additional-domain validation\n\n'+md_table(summary.round(6))+'\n\n## Paired differences\n\nNegative delta/change favors LLM5; SD is across five matched seeds on a fixed stream, not windows or a confidence interval. All controls and adverse outcomes remain.\n\n'+md_table(ps.round(6))+'\n');print(summary.to_string(index=False),flush=True)

def campaign(base=OUT,limit=None):
    base=Path(base);require(not revision()['dirty'],'Commit source');require(json.loads((base/'generation_status.json').read_text())['status']=='completed','Incompletepriorcache');torch.set_num_threads(1);planned=jobs();code=revision();path=base/'queue_state.json';state=json.loads(path.read_text()) if path.exists() else dict(code=code,planned=planned,jobs={key(j):'pending' for j in planned});require(state['code']==code and state['planned']==planned,'Queuechanged');count=0
    for j in planned:
        if state['jobs'][key(j)]=='completed':continue
        require(state['jobs'][key(j)]=='pending','Unresolvedgroup');state['jobs'][key(j)]='running';write_json(path,state)
        try:run_group(j,base);state['jobs'][key(j)]='completed'
        except Exception:state['jobs'][key(j)]='failed';write_json(path,state);raise
        write_json(path,state);count+=1
        if limit is not None and count>=limit:break
    if all(v=='completed' for v in state['jobs'].values()):return analyze(base)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','generate','campaign','analyze'],required=True);p.add_argument('--limit',type=int);a=p.parse_args();campaign(limit=a.limit) if a.stage=='campaign' else globals()[a.stage]()
