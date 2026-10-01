"""Exploratory 60/10/5/5/20 pilot: clean calibration, shared issued forecasts."""
import argparse,copy,gc,json,time,traceback,zlib
from pathlib import Path
from dataclasses import replace
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizerFast
from .config import ROOT
from .data import load_data
from .feedback_factorial import load_bone
from .external_static import pilot_config
from .external_online import OUTPUT as ONLINE
from .replan_audit import OUTPUT as OLD
from .revision_candidate import initialize_five
from .review_sprint import initialize_adapter,write_events
from .shared_online import InternalBackend,OnlineRunner
from .prepare import weight_directory
from . import task_text_prior as prior
from .online_mix import (LS_GRID,ETA_GRID,FixedGate,MomentGate,HedgeGate,MIX_GRID,
                         moment_prior,warm_hedge,replay)
from .util import seed_all,write_json,revision,metrics,sync
from .audit import require,same,original_reference
from .analyze import md_table

OUT=ROOT/'experiments/online_calibration_20261001'
DOMAINS=['Energy','SocialGood']
SEEDS=[2023,2024,2025]
REPRESENTATIONS=['LLM5','raw5']
SCENARIOS=[('online','online'),('online','static'),('static','static')]

def jobs():return [dict(dataset=d,seed=s) for d in DOMAINS for s in SEEDS]
def key(j):return f"{j['dataset']}_s{j['seed']}"
def bounds(n):return dict(fit_end=int(.6*n),checkpoint_end=int(.7*n),calibration_end=int(.75*n),selection_end=int(.8*n),assessment_end=n)
def region_origins(n,L,H,phase):
    b=bounds(n);lo,hi={'calibration':(b['checkpoint_end'],b['calibration_end']),
        'selection':(b['calibration_end'],b['selection_end']),
        'assessment':(b['selection_end'],n)}[phase]
    result=np.arange(max(L-1,lo-1),hi-H,dtype=np.int64)
    require(len(result)>0 and (result+1>=lo).all() and (result+H<hi).all(),'Phase boundary violation')
    return result

def full_ot(d):
    raw=load_data(d,with_embeddings=True);k=raw.columns.index('OT');sl=slice(k,k+1)
    return replace(raw,x=raw.x[:,sl].copy(),target=raw.target[:,sl].copy(),valid=raw.valid[:,sl].copy(),
                   input_missing=raw.input_missing[:,sl].copy(),columns=['OT'],mean=raw.mean[sl].copy(),scale=raw.scale[sl].copy(),val_end=int(.7*len(raw.x)))

def static_source(j):
    p=ONLINE/'groups'/key(j)/'numeric_online/config.json'
    return Path(json.loads(p.read_text())['static_source'])

def check_sources(j):
    c,old,model,checkpoint,previous=load_bone(j,with_text=True,device='cpu')
    data=full_ot(j['dataset']);b=bounds(len(data.x));blob=torch.load(checkpoint,map_location='cpu',weights_only=True)
    require(blob['config']['train_end']==b['fit_end'] and blob['config']['val_end']==b['checkpoint_end'],'Backbone boundary mismatch')
    for split,end in [('train',b['fit_end']),('val',b['checkpoint_end'])]:
        origins=np.asarray(old.origins(split,c.lookback,c.horizon));require((origins+c.horizon<end).all(),'Source targets cross next stage')
    same(data.mean,old.mean,'Train mean');same(data.scale,old.scale,'Train scale')
    same(data.x[:len(old.x)],old.x,'Old numerical prefix');same(data.target[:len(old.x)],old.target,'Old target prefix')
    same(data.valid[:len(old.x)],old.valid,'Old mask prefix',exact=True)
    require(data.texts[:len(old.x)]==old.texts,'Old text prefix differs')
    return dict(**j,checkpoint=str(checkpoint),backbone_config=blob['config'],source_revision=blob.get('source_revision'),
                boundaries=b,numeric_warm=str(static_source(j)/'warm_adapter.pt'),
                LLM5_warm=str(OLD/'exploratory_fusion'/f'{key(j)}_LLM5/warm_adapter.pt'),
                raw5_warm=str(OLD/'exploratory_fusion'/f'{key(j)}_raw_BERT_PCA5/warm_adapter.pt'),
                full_warm=str(static_source(j).parent/'iadapter_static/warm_adapter.pt'),
                fit_target_max=int(old.origins('train',c.lookback,c.horizon)[-1]+c.horizon),
                checkpoint_target_max=int(old.origins('val',c.lookback,c.horizon)[-1]+c.horizon))

def prepare(base=OUT):
    base=Path(base);require(not revision()['dirty'],'Commit source before preparation');torch.set_num_threads(1)
    out=base/'inputs';out.mkdir(parents=True,exist_ok=False)
    tok=BertTokenizerFast.from_pretrained(weight_directory(),local_files_only=True);rows=[];checks=[]
    for d in DOMAINS:
        data=full_ot(d);old=json.loads((OLD/'m0'/f'{d}_token_provenance.json').read_text());b=bounds(len(data.x))
        for t,text in enumerate(data.texts):
            encoded=tok(text,max_length=64,truncation=True,return_offsets_mapping=True)
            right=max([v for u,v in encoded['offset_mapping'] if v>u],default=0)
            visible=text[:right] if text else ''
            if t<len(old):require(visible==old[t]['visible_text'],'Visible rule changed')
            require(all(int(x.split(':')[0])<=t for x in data.text_ids[t]),'Future text ID')
            split=next(name for name,end in [('fit',b['fit_end']),('checkpoint',b['checkpoint_end']),('calibration',b['calibration_end']),('selection',b['selection_end']),('assessment',len(data.x))] if t<end)
            rows.append(dict(dataset=d,origin=t,split=split,available=bool(text),visible_text=visible,
                             selected_ids=data.text_ids[t],key=prior.cache_key(d,visible) if text else None))
        for s in SEEDS:checks.append(check_sources(dict(dataset=d,seed=s)))
    cache=json.loads((OLD/'b3/cache.json').read_text())
    pool={r['key']:r for r in rows if r['available']}
    reused={k:cache[k] for k in pool if k in cache}
    require(all(v['dataset']==pool[k]['dataset'] and v['visible_text']==pool[k]['visible_text'] and v['label'] in prior.LABELS for k,v in reused.items()),'Wrong reused prior')
    write_json(out/'rows.json',rows);write_json(out/'source_checks.json',checks)
    write_json(base/'cache.json',reused)
    write_json(base/'budget.json',dict(limit_seconds=600,used_seconds=0.))
    write_json(out/'validation.json',dict(status='passed',code=revision(),unique_inputs=len(pool),reused_unique=len(reused),missing_unique=len(pool)-len(reused),source_checks=6,scope='Post-results exploratory repartition; no untouched confirmation'))
    print('CALIBRATION_PREPARED',len(pool),'unique;',len(pool)-len(reused),'new',flush=True)

def generate(base=OUT):
    base=Path(base);require(not revision()['dirty'],'Commit source');rows=json.loads((base/'inputs/rows.json').read_text())
    pool={r['key']:r for r in rows if r['available']};cache=json.loads((base/'cache.json').read_text())
    pending=[v for k,v in pool.items() if k not in cache];budget=json.loads((base/'budget.json').read_text())
    require(budget['used_seconds']<budget['limit_seconds'],'Generation budget exhausted');torch.set_num_threads(1)
    start=last=time.perf_counter();model=tok=None
    try:
        seed_all(2023);tok,model,cfg=prior.load_model();torch.cuda.reset_peak_memory_stats()
        for i,row in enumerate(pending):
            result=prior.classify_batch([row],tok,model,cfg)[0];cache[row['key']]=result
            now=time.perf_counter();budget['used_seconds']+=now-last;last=now
            write_json(base/'cache.json',cache);write_json(base/'budget.json',budget)
            if i%100==0:print('CALIBRATION_PRIOR',i+1,'/',len(pending),flush=True)
            require(budget['used_seconds']<=budget['limit_seconds'],'Generation exceeded budget')
        sync('cuda');invalid=sum(not cache[k]['valid'] for k in pool)
        require(invalid==0,'Invalid fixed-prior output; stop before expansion')
        write_json(base/'generation_status.json',dict(status='completed',new_unique=len(pending),planned_unique=len(pool),invalid=invalid,
            seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),code=revision(),
            weights=str(prior.WEIGHTS),prompt=prior.SYSTEM,labels=prior.LABELS,semantic_correctness_verified=False))
    except Exception:write_json(base/'generation_status.json',dict(status='failed',error=traceback.format_exc()));raise
    finally:
        budget['used_seconds']+=time.perf_counter()-last;write_json(base/'budget.json',budget)
        del model,tok;gc.collect();torch.cuda.empty_cache()

def features(data,j,base):
    path=OLD/'b3/readouts'/key(j)/'transforms.npz'
    rows=[r for r in json.loads((base/'inputs/rows.json').read_text()) if r['dataset']==j['dataset']]
    cache=json.loads((base/'cache.json').read_text());available=np.asarray([bool(t) for t in data.texts]);llm=np.zeros((len(data.x),5),dtype=np.float64)
    require(len(rows)==len(data.x),'Rows incomplete')
    for t,r in enumerate(rows):
        require(r['origin']==t and r['selected_ids']==data.text_ids[t],'Origin mismatch')
        if r['available']:
            result=cache[r['key']];require(result['visible_text']==r['visible_text'],'Wrong prior');llm[t,prior.LABELS.index(result['label'])]=1.
    with np.load(path,allow_pickle=False) as z:
        raw=((data.embeddings.astype(np.float64)-z['pca_center'])@z['pca_basis'].T-z['pca_mean'])/z['pca_scale']
        llm=(llm-z['llm_mean'])/z['llm_scale']
    for a in [raw,llm]:a[~available]=0.;require(np.isfinite(a).all(),'Nonfinite features')
    return dict(raw5=raw.astype(np.float32),LLM5=llm.astype(np.float32))

class ContentBackend(InternalBackend):
    """Same-state current-content probe; learning still uses original text."""
    def __init__(self,c,data,model,adapter):
        super().__init__(c,data,model,adapter);self.probes=[]
        self.pool=np.flatnonzero(np.asarray([bool(t) for t in data.texts[:data.train_end]]));require(len(self.pool)>0,'No training donor')
    @torch.no_grad()
    def predict(self,t):
        prediction,k,ids,gate=super().predict(t);sync(self.c.device);start=time.perf_counter()
        donor=int(self.pool[zlib.crc32(f'{self.c.dataset}:{self.c.seed}:{t}'.encode())%len(self.pool)]);require(donor<t,'Future donor')
        values={}
        for name,e in [('historical_donor',self.adapter.text(self.emb[donor:donor+1],k['available'],self.c.device)),
                       ('zero_current_text',self.adapter.text(k['pooled'],False,self.c.device))]:
            residual=self.adapter(k['feature'],k['hg'],e,k['ge'],k['offset'])
            values[name]=self.model.head(k['feature']+residual,k['mu'],k['sigma'])[0].cpu().numpy().copy()
        sync(self.c.device);self.probes.append(dict(origin=t,donor_origin=donor,donor_ids=self.data.text_ids[donor],seconds=time.perf_counter()-start,**values))
        return prediction,k,ids,gate

def frozen_run(c,data,backend,origins):
    predictions=[];contexts=[];seconds=[]
    for t in origins:
        sync(c.device);start=time.perf_counter();p,k,ids,gate=backend.predict(int(t));sync(c.device)
        require(bool(torch.isfinite(p).all()),'Nonfinite frozen prediction');predictions.append(p[0].cpu().numpy().copy())
        seconds.append(time.perf_counter()-start);contexts.append(dict(origin=int(t),text_ids=ids,feedback_origin=-1))
    arrays=dict(origins=origins.copy(),predictions=np.stack(predictions),targets=np.stack([data.target[t+1:t+c.horizon+1] for t in origins]),
                valid_mask=np.stack([data.valid[t+1:t+c.horizon+1] for t in origins]),prediction_seconds=np.asarray(seconds))
    return arrays,[],contexts,metrics(arrays['predictions'],arrays['targets'],arrays['valid_mask'])

def audit_arrays(j,data,phase,arrays):
    c=pilot_config(dataset=j['dataset'],seed=j['seed']);expected=region_origins(len(data.x),c.lookback,c.horizon,phase)
    ref=original_reference(j['dataset']);k=ref['columns'].index('OT');same(arrays['origins'],expected,'Common origins',exact=True)
    y=np.stack([ref['target'][t+1:t+7,k:k+1] for t in expected]);mask=np.stack([ref['valid'][t+1:t+7,k:k+1] for t in expected])
    same(arrays['targets'],y,'Original targets');same(arrays['valid_mask'],mask,'Original masks',exact=True)
    return metrics(arrays['predictions'],y,mask)

def ledger_group(j,base,assessment=False):
    out=base/'groups'/key(j)
    if assessment:
        require(json.loads((out/'status.json').read_text())['status']=='completed','Calibration/selection group incomplete')
        status=out/'assessment/status.json';status.parent.mkdir(exist_ok=False)
    else:
        out.mkdir(parents=True,exist_ok=False);status=out/'status.json'
    write_json(status,dict(status='running'));started=time.perf_counter()
    try:
        proof=check_sources(j);c,old,model,checkpoint,previous=load_bone(j,True);data=full_ot(j['dataset']);values=features(data,j,base)
        c=replace(c,protocol='exploratory_clean_calibration_60_10_5_5_20_v1');frozen={n:p.detach().cpu().clone() for n,p in model.state_dict().items()}
        metadata=dict(**j,boundaries=bounds(len(data.x)),source_checks=proof,config=c.to_dict(),code=revision(),data=data.metadata())
        if assessment:require(json.loads((out/'config.json').read_text())==metadata,'Assessment source/config changed')
        else:write_json(out/'config.json',metadata)
        branches=[('numeric',None),('LLM5',values['LLM5']),('raw5',values['raw5'])]
        for phase in (['assessment'] if assessment else ['calibration','selection']):
            origins=region_origins(len(data.x),c.lookback,c.horizon,phase)
            specs=branches+([('full',data.embeddings)] if phase=='assessment' else [])
            for name,emb in specs:
                cfg=replace(c,variant='numeric_add' if name=='numeric' else 'full' if name=='full' else 'text_add')
                branch_data=replace(data,embeddings=emb)
                adapter=initialize_five(cfg,1) if name in REPRESENTATIONS else initialize_adapter(cfg,1)
                warm=torch.load(proof[f'{name}_warm'],map_location='cpu',weights_only=True)
                for mode in (['online'] if name=='full' else ['static'] if phase=='calibration' else ['static','online']):
                    directory=out/phase/f'{name}_{mode}';directory.mkdir(parents=True,exist_ok=False)
                    adapter.load_state_dict(warm);seed_all(cfg.seed+20000);torch.cuda.reset_peak_memory_stats()
                    backend=(ContentBackend if name in REPRESENTATIONS and phase!='calibration' else InternalBackend)(cfg,branch_data,model,adapter)
                    start=time.perf_counter()
                    arrays,events,contexts,stats=frozen_run(cfg,branch_data,backend,origins) if mode=='static' else OnlineRunner(cfg,branch_data,backend).run(origins.tolist())
                    score=audit_arrays(j,data,phase,arrays);require(stats['mse']==score['mse'] and stats['mae']==score['mae'],'Metrics differ')
                    require(all(torch.equal(v.detach().cpu(),frozen[n]) for n,v in model.state_dict().items()),'Frozen backbone changed')
                    displacement=sum(float((v.detach().cpu().double()-warm[n].double()).square().sum()) for n,v in adapter.state_dict().items())**.5
                    if mode=='static':require(displacement==0.,'Static parameters changed')
                    else:
                        require(displacement>0.,'Online branch never updates')
                        require([e['origin'] for e in events]==origins[:-6].tolist(),'Mature origins missing/repeated')
                    if isinstance(backend,ContentBackend):
                        np.savez_compressed(directory/'content_probes.npz',origins=origins,
                            historical_donor=np.stack([p['historical_donor'] for p in backend.probes]),
                            zero_current_text=np.stack([p['zero_current_text'] for p in backend.probes]),
                            donor_origins=np.asarray([p['donor_origin'] for p in backend.probes]),probe_seconds=np.asarray([p['seconds'] for p in backend.probes]))
                        write_json(directory/'donor_context.json',[{k:v for k,v in p.items() if k in ['origin','donor_origin','donor_ids']} for p in backend.probes])
                    if phase=='calibration':
                        reference=static_source(j)/'predictions.npz' if name=='numeric' else OLD/'exploratory_fusion'/f"{key(j)}_{'LLM5' if name=='LLM5' else 'raw_BERT_PCA5'}"/'static/predictions.npz'
                        with np.load(reference,allow_pickle=False) as z:
                            indices={int(t):i for i,t in enumerate(z['origins'])}
                            np.testing.assert_allclose(arrays['predictions'],np.stack([z['predictions'][indices[int(t)]] for t in origins]),atol=1e-6,rtol=1e-6,err_msg='Reused static prefix')
                    np.savez_compressed(directory/'predictions.npz',**arrays);write_events(directory/'updates.jsonl',events);write_json(directory/'contexts.json',contexts)
                    stats.update(seconds=time.perf_counter()-start,trainable_parameters=sum(p.numel() for p in adapter.parameters()),peak_allocated_bytes=torch.cuda.max_memory_allocated(),parameter_displacement_l2=displacement,frozen_backbone_unchanged=True,all_phase_state_reset=True)
                    write_json(directory/'metrics.json',stats);write_json(directory/'status.json',dict(status='completed'))
                    del backend;gc.collect();torch.cuda.empty_cache()
                del adapter
        write_json(status,dict(status='completed',seconds=time.perf_counter()-started));print('ASSESSMENT_LEDGERS_COMPLETE' if assessment else 'CALIBRATION_LEDGERS_COMPLETE',key(j),flush=True)
    except Exception:write_json(status,dict(status='failed',error=traceback.format_exc()));raise

def ledgers(base=OUT,limit=None,assessment=False):
    base=Path(base);require(not revision()['dirty'],'Commit source');require(json.loads((base/'generation_status.json').read_text())['status']=='completed','Incomplete text cache')
    torch.set_num_threads(1);code=revision();planned=jobs()
    if assessment:
        require(json.loads((base/'selection/locked.json').read_text())['code']==code,'Rules/code changed')
        previous=json.loads((base/'queue_state.json').read_text());require(previous['code']==code and all(v=='completed' for v in previous['jobs'].values()),'Selection wave incomplete')
    path=base/('assessment_queue_state.json' if assessment else 'queue_state.json')
    state=json.loads(path.read_text()) if path.exists() else dict(code=code,planned=planned,jobs={key(j):'pending' for j in planned})
    require(state['code']==code and state['planned']==planned,'Queue/source changed');count=0
    for j in planned:
        if state['jobs'][key(j)]=='completed':continue
        require(state['jobs'][key(j)]=='pending','Unresolved failed/running group')
        state['jobs'][key(j)]='running';write_json(path,state)
        try:ledger_group(j,base,assessment);state['jobs'][key(j)]='completed'
        except Exception:state['jobs'][key(j)]='failed';write_json(path,state);raise
        write_json(path,state);count+=1
        if limit is not None and count>=limit:break

def pair_ledger(j,phase,representation,nmode,tmode,base):
    out=base/'groups'/key(j)/phase
    entries=[]
    for name in [f'numeric_{nmode}',f'{representation}_{tmode}']:
        require(json.loads((out/name/'status.json').read_text())['status']=='completed','Incomplete branch')
        with np.load(out/name/'predictions.npz',allow_pickle=False) as z:entries.append({k:z[k].copy() for k in ['origins','predictions','targets','valid_mask']})
    n,m=entries
    for k in ['origins','targets','valid_mask']:same(n[k],m[k],'Matched pair '+k,exact=True)
    data=load_data(j['dataset'])
    return dict(origins=n['origins'],numeric=n['predictions'].astype(np.float64),text=m['predictions'].astype(np.float64),
        targets=n['targets'].astype(np.float64),valid_mask=n['valid_mask'],available=np.asarray([bool(data.texts[t]) for t in n['origins']]))

def gate_for(family,setting,cal,cutoff):
    p=moment_prior(cal,cutoff,shared=family=='shared_ls')
    if family=='equal':return FixedGate(np.full(p['H'],.5))
    if family=='fixed':return FixedGate(moment_prior(cal,cutoff)['alpha0'])
    if family in ['ls','shared_ls']:return MomentGate(p,**setting)
    weights,_=warm_hedge(cal,cutoff,setting['eta'],mixed=family=='mixed_hedge')
    return HedgeGate(p['H'],setting['eta'],MIX_GRID if family=='mixed_hedge' else None,weights)

def settings(family):
    return [dict(rho=r,kappa=k) for r,k in LS_GRID] if family in ['ls','shared_ls'] else [dict(eta=x) for x in ETA_GRID]

def select(base=OUT):
    base=Path(base);out=base/'selection';rows=[]
    state=json.loads((base/'queue_state.json').read_text());require(state['code']==revision() and not revision()['dirty'],'Queue/code changed')
    require(all(v=='completed' for v in state['jobs'].values()),'Incomplete ledgers')
    out.mkdir(exist_ok=False)
    for j in jobs():
        b=json.loads((base/'groups'/key(j)/'config.json').read_text())['boundaries'];cutoff=b['calibration_end']-1
        for rep in REPRESENTATIONS:
            cal=pair_ledger(j,'calibration',rep,'static','static',base)
            for nm,tm in SCENARIOS:
                ledger=pair_ledger(j,'selection',rep,nm,tm,base);control=metrics(ledger['numeric'],ledger['targets'],ledger['valid_mask'])['mse']
                for family in ['ls','shared_ls','endpoint_hedge','mixed_hedge']:
                    for index,setting in enumerate(settings(family)):
                        arrays,events=replay(ledger,gate_for(family,setting,cal,cutoff));score=metrics(arrays['predictions'],arrays['targets'],arrays['valid_mask'])
                        rows.append(dict(**j,representation=rep,numeric_mode=nm,text_mode=tm,family=family,setting_index=index,ratio=score['mse']/control,**score))
    frame=pd.DataFrame(rows);frame.to_csv(out/'runs.csv',index=False);rank=frame.groupby(['family','setting_index'])['ratio'].mean().reset_index();rank.to_csv(out/'candidates.csv',index=False)
    chosen={f:dict(index=int(g.sort_values(['ratio','setting_index'],kind='stable').iloc[0].setting_index),setting=settings(f)[int(g.sort_values(['ratio','setting_index'],kind='stable').iloc[0].setting_index)]) for f,g in rank.groupby('family')}
    write_json(out/'locked.json',dict(status='locked',chosen=chosen,code=revision(),selection_only=True,scope='One global setting per family; validation states discarded; no assessment metric used'))
    print('CALIBRATION_RULES_LOCKED',chosen,flush=True)

def evaluate_mix(base=OUT):
    base=Path(base);locked=json.loads((base/'selection/locked.json').read_text());require(locked['status']=='locked','Rules not locked')
    require(locked['code']==revision() and not revision()['dirty'],'Locked/code changed')
    state=json.loads((base/'assessment_queue_state.json').read_text());require(state['code']==revision() and all(v=='completed' for v in state['jobs'].values()),'Assessment wave incomplete')
    out=base/'mixes';out.mkdir(exist_ok=False);rows=[]
    for j in jobs():
        cfg=json.loads((base/'groups'/key(j)/'config.json').read_text());cutoff=cfg['boundaries']['calibration_end']-1
        for rep in REPRESENTATIONS:
            cal=pair_ledger(j,'calibration',rep,'static','static',base)
            for nm,tm in SCENARIOS:
                ledger=pair_ledger(j,'assessment',rep,nm,tm,base)
                for family in ['equal','fixed','ls','shared_ls','endpoint_hedge','mixed_hedge']:
                    setting=locked['chosen'][family]['setting'] if family in locked['chosen'] else {}
                    gate=gate_for(family,setting,cal,cutoff);initial=gate.alpha().copy();arrays,events=replay(ledger,gate,update=family not in ['equal','fixed'])
                    directory=out/key(j)/rep/f'numeric_{nm}_text_{tm}'/family;directory.mkdir(parents=True,exist_ok=False)
                    score=metrics(arrays['predictions'],arrays['targets'],arrays['valid_mask']);audit_arrays(j,full_ot(j['dataset']),'assessment',arrays)
                    same(gate.alpha(),initial,'Initial gate mutated')
                    np.savez_compressed(directory/'predictions.npz',**arrays);write_events(directory/'updates.jsonl',events)
                    write_json(directory/'config.json',dict(**j,representation=rep,numeric_mode=nm,text_mode=tm,family=family,setting=setting,calibration_cutoff=cutoff,initial_alpha=initial.tolist(),source=cfg,selection_source=str(base/'selection/locked.json')))
                    write_json(directory/'metrics.json',score);write_json(directory/'status.json',dict(status='completed'))
                    rows.append(dict(**j,representation=rep,numeric_mode=nm,text_mode=tm,family=family,**score))
    pd.DataFrame(rows).to_csv(out/'runs.csv',index=False);print('CALIBRATION_MIXES_COMPLETE',len(rows),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True,choices=['prepare','generate','ledgers','select','assessment','evaluate_mix']);p.add_argument('--base',type=Path,default=OUT);p.add_argument('--limit',type=int);a=p.parse_args()
    ledgers(a.base,a.limit,a.stage=='assessment') if a.stage in ['ledgers','assessment'] else globals()[a.stage](a.base)
