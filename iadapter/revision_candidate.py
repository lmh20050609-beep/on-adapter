"""User-revised local-progress follow-up: same five-dimensional simple fusion."""
import argparse,copy,gc,json,time,traceback,zlib
from pathlib import Path
from dataclasses import replace
import numpy as np
import pandas as pd
import torch
from torch import nn
from .replan_audit import OUTPUT
from .feedback_factorial import load_bone
from .review_sprint import initialize_adapter,write_events
from .external_static import evaluate
from .shared_online import InternalBackend,OnlineRunner
from .task_text_prior import prior_arrays
from .util import seed_all,revision,write_json,metrics
from .audit import original_reference,same,require
from .analyze import md_table

REPRESENTATIONS=['raw_BERT_PCA5','LLM5']
def jobs():return [dict(dataset=d,seed=s,representation=r) for d in ['Energy','SocialGood'] for s in [2023,2024,2025] for r in REPRESENTATIONS]
def key(j):return f"{j['dataset']}_s{j['seed']}_{j['representation']}"

def initialize_five(c,N):
    ad=initialize_adapter(c,N)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed((c.seed+10000+zlib.crc32(b'project.0'))%(2**31))
        ad.project[0]=nn.Linear(5,96)
    return ad.to(c.device)

def five_data(data,d,s,base):
    path=base/'b3/readouts'/f'{d}_s{s}'/'transforms.npz'
    with np.load(path) as z:
        available=np.asarray([bool(t) for t in data.texts]);raw=((data.embeddings.astype(np.float64)-z['pca_center'])@z['pca_basis'].T-z['pca_mean'])/z['pca_scale']
        provenance=json.loads((base/'m0'/f'{d}_token_provenance.json').read_text());cache=json.loads((base/'b3/cache.json').read_text());_,llm=prior_arrays(data,d,provenance,cache);llm=(llm-z['llm_mean'])/z['llm_scale']
        for a in [raw,llm]:a[~available]=0.;require(np.isfinite(a).all() and a.shape==(len(data.x),5),'Invalid5dim input')
    return {'raw_BERT_PCA5':raw.astype(np.float32),'LLM5':llm.astype(np.float32)},path

class DonorBackend(InternalBackend):
    def __init__(self,c,data,model,adapter,seed):
        super().__init__(c,data,model,adapter);self.seed=seed;self.pool=np.flatnonzero(np.asarray([bool(t) for t in data.texts[:data.train_end]]));self.probes=[]
        require(len(self.pool)>0,'No informative training donor')
    @torch.no_grad()
    def predict(self,t):
        issued,context,ids,gate=super().predict(t);donor=int(self.pool[zlib.crc32(f'{self.c.dataset}:{self.seed}:{t}'.encode())%len(self.pool)]);require(donor<t,'Nonhistorical donor')
        e=self.adapter.text(self.emb[donor:donor+1],context['available'],self.c.device)
        z=self.adapter(context['feature'],context['hg'],e,context['ge'],context['offset']);prediction=self.model.head(context['feature']+z,context['mu'],context['sigma'])
        self.probes.append(dict(origin=t,donor_origin=donor,donor_ids=self.data.text_ids[donor],prediction=prediction[0].cpu().numpy()))
        return issued,context,ids,gate

def run_group(j,base):
    out=base/'exploratory_fusion'/key(j);out.mkdir(parents=True,exist_ok=False);write_json(out/'status.json',{'status':'running'});start=time.perf_counter()
    try:
        c,data,model,checkpoint,bounds=load_bone(j,with_text=True);c=replace(c,protocol='exploratory_development_five_dim_text_v1')
        representations,transform=five_data(data,j['dataset'],j['seed'],base);data=replace(data,embeddings=representations[j['representation']])
        # Identical topology/initial parameters across representations, training differs only in input.
        ad=initialize_five(c,1);initial={n:v.detach().cpu().clone() for n,v in ad.state_dict().items()};torch.save(initial,out/'initial_adapter.pt')
        frozen={n:v.detach().cpu().clone() for n,v in model.state_dict().items()};seed_all(c.seed+11000)
        from .engine import Stream
        _,events,warm_stats=Stream(c,data,model,ad).run('train');write_events(out/'warm_updates.jsonl',events);torch.save(ad.state_dict(),out/'warm_adapter.pt');write_json(out/'warm_metrics.json',warm_stats)
        warm={n:v.detach().cpu().clone() for n,v in ad.state_dict().items()};config=dict(**j,config=c.to_dict(),data=data.metadata(),boundaries=bounds,checkpoint=str(checkpoint),input_dimension=5,transform_source=str(transform),priors_source=str(base/'b3/cache.json'),code=revision(),selection='Representation chosen post-results for local content signal; no LR/prompt/width search',projection='5->96->64; same original ReLU/dropout0.1/LayerNorm64',mode_start='identical warm weights; empty feedback/queue/optimizer')
        write_json(out/'config.json',config)
        for mode in ['static','online']:
            directory=out/mode;directory.mkdir(exist_ok=False);write_json(directory/'status.json',{'status':'running'});ad.load_state_dict(warm);seed_all(c.seed+20000);torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
            if mode=='static':arrays,contexts=evaluate(data,c,model,ad);events=[];stats=metrics(arrays['predictions'],arrays['targets'],arrays['valid_mask'])
            else:
                backend=DonorBackend(c,data,model,ad,c.seed);arrays,events,contexts,stats=OnlineRunner(c,data,backend).run()
                with np.load(out/'static/predictions.npz') as z:same(arrays['predictions'][:7],z['predictions'][:7],'Preupdate predictions')
                donor=np.stack([p['prediction'] for p in backend.probes]);donors=np.asarray([p['donor_origin'] for p in backend.probes]);np.savez_compressed(directory/'donor.npz',origins=arrays['origins'],predictions=donor,targets=arrays['targets'],valid_mask=arrays['valid_mask'],donor_origins=donors);write_json(directory/'donor_context.json',[{k:v for k,v in p.items() if k!='prediction'} for p in backend.probes])
            require(all(torch.equal(v.detach().cpu(),frozen[n]) for n,v in model.state_dict().items()),'Frozen backbone changed')
            delta=sum(float((v.detach().cpu().double()-warm[n].double()).square().sum()) for n,v in ad.state_dict().items())**.5
            if mode=='static':require(delta==0.,'Static state changed')
            else:require(delta>0.,'Online adapter never updated')
            ref=original_reference(j['dataset']);k=ref['columns'].index('OT');origins=arrays['origins'];target=np.stack([ref['target'][t+1:t+7,k:k+1] for t in origins]);valid=np.stack([ref['valid'][t+1:t+7,k:k+1] for t in origins])
            same(arrays['targets'],target,'Sanity original target');same(arrays['valid_mask'],valid,'Sanity original mask',exact=True);same(origins,np.asarray(data.origins('test',c.lookback,6)),'Sanity origin',exact=True)
            if mode=='online':require([e['origin'] for e in events]==origins[:-6].tolist(),'Sanity mature consumption')
            np.savez_compressed(directory/'predictions.npz',**arrays);write_events(directory/'updates.jsonl',events);write_json(directory/'contexts.json',contexts);torch.save(ad.state_dict(),directory/'final_adapter.pt')
            stats.update(seconds=time.perf_counter()-begin,trainable_parameters=sum(p.numel() for p in ad.parameters()),peak_allocated_bytes=torch.cuda.max_memory_allocated(),parameter_displacement_l2=delta,frozen_backbone_unchanged=True)
            write_json(directory/'metrics.json',stats);write_json(directory/'status.json',{'status':'completed'})
        write_json(out/'status.json',dict(status='completed',seconds=time.perf_counter()-start));print('FUSION_COMPLETE',key(j),flush=True);del model,ad;gc.collect();torch.cuda.empty_cache()
    except Exception:
        write_json(out/'status.json',dict(status='failed',error=traceback.format_exc()));raise

def audit(base=OUTPUT):
    base=Path(base);rows=[]
    for j in jobs():
        out=base/'exploratory_fusion'/key(j);require(json.loads((out/'status.json').read_text())['status']=='completed','Incomplete group');cfg=json.loads((out/'config.json').read_text());require(not cfg['code']['dirty'],'Uncommitted run')
        ref=original_reference(j['dataset']);k=ref['columns'].index('OT');origins=np.arange(int(.7*ref['rows'])-1,int(.8*ref['rows'])-6);target=np.stack([ref['target'][t+1:t+7,k:k+1] for t in origins]);mask=np.stack([ref['valid'][t+1:t+7,k:k+1] for t in origins])
        initial=torch.load(out/'initial_adapter.pt',map_location='cpu',weights_only=True);other=out.parent/key(dict(j,representation='LLM5' if j['representation']=='raw_BERT_PCA5' else 'raw_BERT_PCA5'))/'initial_adapter.pt'
        peer=torch.load(other,map_location='cpu',weights_only=True);require(all(torch.equal(v,peer[n]) for n,v in initial.items()),'Representation initial weights differ')
        for mode in ['static','online']:
            directory=out/mode;require(json.loads((directory/'status.json').read_text())['status']=='completed','Incomplete mode');saved=json.loads((directory/'metrics.json').read_text())
            with np.load(directory/'predictions.npz') as z:
                same(z['origins'],origins,'Original origins',exact=True);same(z['targets'],target,'Original target');same(z['valid_mask'],mask,'Original mask',exact=True);score=metrics(z['predictions'],target,mask)
                if mode=='online':
                    ev=[json.loads(x) for x in (directory/'updates.jsonl').read_text().splitlines()];require([e['origin'] for e in ev]==origins[:-6].tolist(),'Missing/repeated mature origin');require(all(e['mature_at']==e['origin']+6==e['update_at'] and e['text_origin']==e['state_origin']==e['origin'] and e['text_ids']==ref['text_ids'][e['origin']] for e in ev),'Replay context differs')
            for metric in ['mse','mae']:require(abs(score[metric]-saved[metric])<1e-12,'Saved metric differs')
            rows.append(dict(**j,mode=mode,**score))
        with np.load(out/'online/donor.npz') as z:
            same(z['origins'],origins,'Donor origins',exact=True);same(z['targets'],target,'Donor target');same(z['valid_mask'],mask,'Donor mask',exact=True);require((z['donor_origins']<ref['train_end']).all(),'Future donor');score=metrics(z['predictions'],target,mask)
        rows.append(dict(**j,mode='donor_same_state',**score))
    frame=pd.DataFrame(rows);out=base/'exploratory_fusion/reports';out.mkdir(exist_ok=True);frame.to_csv(out/'runs.csv',index=False);summary=frame.groupby(['dataset','representation','mode']).agg(n=('seed','count'),mse_mean=('mse','mean'),mse_sd=('mse','std'),mae_mean=('mae','mean'),mae_sd=('mae','std')).reset_index();summary.to_csv(out/'summary.csv',index=False)
    numeric=pd.read_csv(base/'b1/reports/runs.csv');pairs=[]
    references=numeric[numeric.method.isin(['frozen_backbone','numeric_static','numeric_online','gpt4mts_static','iadapter_online'])].copy();references.to_csv(out/'references.csv',index=False)
    reference_summary=references.groupby(['dataset','method']).agg(n=('seed','count'),mse_mean=('mse','mean'),mse_sd=('mse','std'),mae_mean=('mae','mean'),mae_sd=('mae','std')).reset_index();reference_summary.to_csv(out/'reference_summary.csv',index=False)
    for d in ['Energy','SocialGood']:
        means=summary[summary.dataset==d].set_index(['representation','mode'])
        for s in [2023,2024,2025]:
            i=frame[(frame.dataset==d)&(frame.seed==s)].set_index(['representation','mode']);control=numeric[(numeric.dataset==d)&(numeric.seed==s)&(numeric.method=='numeric_online')].iloc[0]
            for label,x,y in [('LLM_online_minus_raw_online',i.loc[('LLM5','online')],i.loc[('raw_BERT_PCA5','online')]),('LLM_online_minus_numeric_online',i.loc[('LLM5','online')],control),('LLM_online_minus_LLM_static',i.loc[('LLM5','online')],i.loc[('LLM5','static')]),('LLM_donor_minus_correct',i.loc[('LLM5','donor_same_state')],i.loc[('LLM5','online')]),('raw_donor_minus_correct',i.loc[('raw_BERT_PCA5','donor_same_state')],i.loc[('raw_BERT_PCA5','online')])]:
                pairs.append(dict(dataset=d,seed=s,comparison=label,delta_mse=x['mse']-y['mse'],delta_mae=x['mae']-y['mae']))
    pair=pd.DataFrame(pairs);pair.to_csv(out/'paired_runs.csv',index=False);pair.groupby(['dataset','comparison']).agg(n=('seed','count'),delta_mse_mean=('delta_mse','mean'),delta_mse_sd=('delta_mse','std'),delta_mae_mean=('delta_mae','mean'),delta_mae_sd=('delta_mae','std'),mse_improved_seeds=('delta_mse',lambda v:int((v<0).sum()))).reset_index().to_csv(out/'paired_summary.csv',index=False)
    write_json(out/'validation.json',dict(status='passed',new_evaluations=24,warmups=12,donor_sequences=12,original_csv_verified_score_sequences=36,same_initial_parameters_across_representations=True,scope='User-revised post-results local-progress exploration; not independent confirmation'))
    (out/'RESULTS.md').write_text('# Five-dimensional simple fusion exploration\n\n'+md_table(summary.round(6))+'\n\n## Reused strong system references\n\n'+md_table(reference_summary.round(6))+'\n\nReference rows are exact B1 reuse, not new training. GPT4MTS is a system-level comparator with different text history/encoding.\n');print(summary.to_string(index=False),flush=True);return summary

def campaign(base=OUTPUT,limit=None):
    base=Path(base);out=base/'exploratory_fusion';out.mkdir(exist_ok=True);torch.set_num_threads(1);code=revision();require(not code['dirty'],'Commit source first');planned=jobs();path=out/'queue_state.json'
    state=json.loads(path.read_text()) if path.exists() else dict(code=code,jobs={key(j):'pending' for j in planned},planned=planned);require(state['code']==code and state['planned']==planned,'Queue changed');count=0
    for j in planned:
        if state['jobs'][key(j)]=='completed':continue
        require(state['jobs'][key(j)]=='pending','Unresolved failed group');state['jobs'][key(j)]='running';write_json(path,state)
        try:run_group(j,base);state['jobs'][key(j)]='completed'
        except Exception:state['jobs'][key(j)]='failed';write_json(path,state);raise
        write_json(path,state);count+=1
        if limit is not None and count>=limit:break
    print('FUSION_QUEUE',sum(v=='completed' for v in state['jobs'].values()),'/12',flush=True)
    if all(v=='completed' for v in state['jobs'].values()):return audit(base)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--limit',type=int);p.add_argument('--audit',action='store_true');a=p.parse_args();audit(a.output) if a.audit else campaign(a.output,a.limit)
