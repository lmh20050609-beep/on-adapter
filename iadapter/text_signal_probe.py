"""B2 train-prefix PCA/Ridge residual probes and fixed-state content intervention."""
import argparse,json,time,zlib,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .config import ROOT
from .feedback_factorial import load_bone
from .models import compute_offset_features
from .replan_audit import OUTPUT
from .util import metrics,write_json,revision
from .audit import original_reference,same,require
from .analyze import md_table

LAMBDAS=[.1,1.,10.]

def fit_ridge(x,y,mask,lam):
    design=np.column_stack([np.ones(len(x)),x]);coef=np.empty((design.shape[1],y.shape[1]),dtype=np.float64)
    penalty=np.eye(design.shape[1])*lam;penalty[0,0]=0.
    for h in range(y.shape[1]):
        valid=mask[:,h];require(valid.any(),'No training target for horizon')
        a=design[valid];coef[:,h]=np.linalg.solve(a.T@a+penalty,a.T@y[valid,h])
    return coef

def predict_ridge(x,coef,baseline):return (np.column_stack([np.ones(len(x)),x])@coef+baseline).astype(np.float32)[:,:,None]

def pca_basis(embeddings,available,train_end,dimension=16):
    train=embeddings[:train_end][available[:train_end]].astype(np.float64)
    require(len(train)>dimension,'Too few informative training texts')
    center=train.mean(0);_,singular,v=np.linalg.svd(train-center,full_matrices=False)
    basis=v[:dimension];projected=(embeddings.astype(np.float64)-center)@basis.T;projected[~available]=0.
    # Scaling only from training forecast-independent input rows.
    mean=projected[:train_end].mean(0);sd=projected[:train_end].std(0);sd=np.where(sd<1e-8,1.,sd)
    content=(projected-mean)/sd;content[~available]=0.
    return content,dict(center=center,basis=basis,mean=mean,scale=sd,singular_values=singular[:dimension])

def standardize_train(x,train):
    mean=x[train].mean(0);sd=x[train].std(0);sd=np.where(sd<1e-8,1.,sd)
    return (x-mean)/sd,dict(mean=mean,scale=sd)

@torch.no_grad()
def numerical_features(c,data,model,origins):
    features=[];baseline=[]
    for t in origins:
        x=torch.from_numpy(data.x[t-c.lookback+1:t+1]).unsqueeze(0).to(c.device)
        h,mu,sd=model.context(x);features.append(torch.cat([h.flatten(),compute_offset_features(x).flatten()]).cpu().numpy())
        baseline.append(model.head(h,mu,sd)[0,:,0].cpu().numpy())
    return np.stack(features).astype(np.float64),np.stack(baseline)

def _run_group(dataset,seed,base=OUTPUT):
    base=Path(base);out=base/'b2'/f'{dataset}_s{seed}';out.mkdir(parents=True,exist_ok=False);started=time.perf_counter();write_json(out/'status.json',{'status':'running'})
    torch.set_num_threads(1);c,data,model,checkpoint,bounds=load_bone(dict(dataset=dataset,seed=seed),with_text=True)
    provenance=json.loads((base/'m0'/f'{dataset}_token_provenance.json').read_text());require(len(provenance)==len(data.x),'Wrong M0 input rows')
    origins=np.arange(c.lookback-1,len(data.x)-6);train=origins<data.train_end-6;selection=(origins>=data.train_end-1)&(origins<data.val_end-6);dev=origins>=data.val_end-1
    require(np.array_equal(origins[train],data.origins('train',c.lookback,6)) and np.array_equal(origins[selection],data.origins('val',c.lookback,6)) and np.array_equal(origins[dev],data.origins('test',c.lookback,6)),'Split differs')
    nums,baseline=numerical_features(c,data,model,origins);nums,ns=standardize_train(nums,train)
    metadata=np.asarray([x['metadata'] for x in provenance],dtype=np.float64)[origins];metadata,ms=standardize_train(metadata,train)
    available=np.asarray([bool(t) for t in data.texts]);content,pcs=pca_basis(data.embeddings,available,data.train_end,16)
    inputs=dict(P0=np.column_stack([nums,np.zeros((len(origins),20))]),P1=np.column_stack([nums,metadata,np.zeros((len(origins),16))]),P2=np.column_stack([nums,metadata,content[origins]]))
    y=np.stack([data.target[t+1:t+7,0] for t in origins]);mask=np.stack([data.valid[t+1:t+7,0] for t in origins]);residual=y.astype(np.float64)-baseline.astype(np.float64)
    ref=original_reference(dataset);k=ref['columns'].index('OT');expected=np.stack([ref['target'][t+1:t+7,k] for t in origins]);valid=np.stack([ref['valid'][t+1:t+7,k] for t in origins])
    same(y,expected,'Original target');same(mask,valid,'Original mask',exact=True)
    saved={};scores=[];selection_history=[]
    for name,x in inputs.items():
        candidates=[]
        for lam in LAMBDAS:
            coef=fit_ridge(x[train],residual[train],mask[train],lam);prediction=predict_ridge(x[selection],coef,baseline[selection]);score=metrics(prediction,y[selection,:,None],mask[selection,:,None]);candidates.append((score['mse'],lam,coef));selection_history.append(dict(probe=name,lambda_value=lam,selection_mse=score['mse']))
        _,lam,coef=min(candidates,key=lambda q:(q[0],q[1]));prediction=predict_ridge(x[dev],coef,baseline[dev]);score=metrics(prediction,y[dev,:,None],mask[dev,:,None])
        np.savez_compressed(out/(name+'.npz'),origins=origins[dev],predictions=prediction,targets=y[dev,:,None],valid_mask=mask[dev,:,None],coefficient=coef,lambda_value=lam)
        saved[name]=(coef,lam,prediction);scores.append(dict(dataset=dataset,seed=seed,probe=name,lambda_value=lam,**score))
    donor_pool=np.flatnonzero(available[:data.train_end]);require(len(donor_pool)>0,'No historical donor pool')
    donors=np.asarray([int(donor_pool[zlib.crc32(f'{dataset}:{seed}:{t}'.encode())%len(donor_pool)]) for t in origins[dev]])
    require((donors<origins[dev]).all(),'Nonhistorical donor')
    intervention=inputs['P2'][dev].copy();intervention[:,-16:]=content[donors];intervention[~available[origins[dev]],-16:]=0.
    coef,lam,_=saved['P2'];prediction=predict_ridge(intervention,coef,baseline[dev]);score=metrics(prediction,y[dev,:,None],mask[dev,:,None])
    np.savez_compressed(out/'P3.npz',origins=origins[dev],predictions=prediction,targets=y[dev,:,None],valid_mask=mask[dev,:,None],donor_origins=donors)
    scores.append(dict(dataset=dataset,seed=seed,probe='P3',lambda_value=lam,**score));pd.DataFrame(scores).to_csv(out/'scores.csv',index=False)
    write_json(out/'donor_context.json',[dict(origin=int(t),donor=int(d),donor_ids=provenance[d]['selected_ids'],metadata_held_fixed=True) for t,d in zip(origins[dev],donors)])
    np.savez_compressed(out/'transforms.npz',**{'pca_'+n:v for n,v in pcs.items()},numeric_mean=ns['mean'],numeric_scale=ns['scale'],metadata_mean=ms['mean'],metadata_scale=ms['scale'],train_origins=origins[train],selection_origins=origins[selection],development_origins=origins[dev])
    write_json(out/'config.json',dict(dataset=dataset,seed=seed,checkpoint=str(checkpoint),data=data.metadata(),boundaries=bounds,code=revision(),lambda_candidates=LAMBDAS,pca_dimension=16,pca_fit_rows='informative training prefix only',ridge_fit_origins=origins[train].tolist(),selection_only=True,feature_slots=60,metadata_names=['available','visible_characters','visible_records','visible_age_days'],scores_against='original CSV target, not frozen forecast',seconds=time.perf_counter()-started))
    write_json(out/'selection_history.json',selection_history);write_json(out/'status.json',{'status':'completed'});del model;torch.cuda.empty_cache()
    print('B2_COMPLETE',dataset,seed,[(x['probe'],round(x['mse'],7)) for x in scores],flush=True)

def run_group(dataset,seed,base=OUTPUT):
    out=Path(base)/'b2'/f'{dataset}_s{seed}'
    if out.exists():raise FileExistsError('Probe output already exists: '+str(out))
    try:return _run_group(dataset,seed,base)
    except Exception:
        if out.exists():write_json(out/'status.json',{'status':'failed','error':traceback.format_exc()})
        raise

def audit(base=OUTPUT):
    base=Path(base);rows=[]
    for d in ['Energy','SocialGood']:
        ref=original_reference(d);k=ref['columns'].index('OT');origins=np.arange(int(.7*ref['rows'])-1,int(.8*ref['rows'])-6);target=np.stack([ref['target'][t+1:t+7,k:k+1] for t in origins]);mask=np.stack([ref['valid'][t+1:t+7,k:k+1] for t in origins])
        for s in [2023,2024,2025]:
            directory=base/'b2'/f'{d}_s{s}';require(json.loads((directory/'status.json').read_text())['status']=='completed','Incomplete probe group')
            cfg=json.loads((directory/'config.json').read_text());require(not cfg['code']['dirty'] and cfg['lambda_candidates']==LAMBDAS and max(cfg['ridge_fit_origins'])+6<ref['train_end'],'Fit used non-training target')
            saved=pd.read_csv(directory/'scores.csv')
            for name in ['P0','P1','P2','P3']:
                with np.load(directory/(name+'.npz')) as z:
                    same(z['origins'],origins,'Probe origin',exact=True);same(z['targets'],target,'Probe original target');same(z['valid_mask'],mask,'Probe mask',exact=True);score=metrics(z['predictions'],target,mask)
                    if name=='P3':require((z['donor_origins']<ref['train_end']).all(),'Future donor')
                row=saved[saved.probe==name].iloc[0]
                for metric in ['mse','mae']:require(abs(score[metric]-row[metric])<1e-12,'Probe saved score differs')
                rows.append(dict(dataset=d,seed=s,probe=name,**score))
    runs=pd.DataFrame(rows);out=base/'b2/reports';out.mkdir(exist_ok=True);runs.to_csv(out/'runs.csv',index=False)
    summary=runs.groupby(['dataset','probe'],sort=False).agg(n=('seed','count'),mse_mean=('mse','mean'),mse_sd=('mse','std'),mae_mean=('mae','mean'),mae_sd=('mae','std')).reset_index();summary.to_csv(out/'summary.csv',index=False)
    gates=[]
    for d in ['Energy','SocialGood']:
        x=summary[summary.dataset==d].set_index('probe');delta=(x.loc['P2','mse_mean']/x.loc['P1','mse_mean']-1)*100;mae=(x.loc['P2','mae_mean']/x.loc['P1','mae_mean']-1)*100;paired=runs[runs.dataset==d].pivot(index='seed',columns='probe',values='mse');wins=int((paired.P2<paired.P1).sum());donor=x.loc['P3','mse_mean']-x.loc['P2','mse_mean']
        gates.append(dict(dataset=d,mse_change_percent=delta,mae_change_percent=mae,mse_improved_seeds=wins,donor_minus_correct_mse=donor,passes=delta<=-2 and mae<=1 and wins>=2 and donor>0))
    pd.DataFrame(gates).to_csv(out/'gates.csv',index=False);write_json(out/'validation.json',dict(status='passed',selection_tasks=18,lambda_configurations=54,donor_intervention_sequences=6,original_csv_verified_score_sequences=24,scope='exploratory frozen residual readout; no semantic causality or unseen-data confirmation'))
    (out/'RESULTS.md').write_text('# B2 metadata/content residual readout\n\n'+md_table(summary.round(6))+'\n\n'+md_table(pd.DataFrame(gates).round(6))+'\n\nFit/PCA/scalers use training prefix only. Lambda uses selection segment, donor uses training text, development labels only score. P3 is a fixed-state intervention, not a trained model.\n')
    print('B2_VALIDATED',flush=True);print(pd.DataFrame(gates).to_string(index=False),flush=True);return gates

def main(base=OUTPUT):
    base=Path(base);require(not revision()['dirty'],'Commit probe source before deployment')
    for d in ['Energy','SocialGood']:
        for s in [2023,2024,2025]:
            out=base/'b2'/f'{d}_s{s}'
            if out.exists():require(json.loads((out/'status.json').read_text())['status']=='completed','Unresolved probe attempt');continue
            run_group(d,s,base)
    return audit(base)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--audit',action='store_true');a=p.parse_args();audit(a.output) if a.audit else main(a.output)
