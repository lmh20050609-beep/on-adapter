"""Original-CSV audit and four-cell E-01 development comparison."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .audit import original_reference,require,same,verify_trace
from .util import metrics,write_json
from .run import backbone_config
from .review_sprint import DEFAULT_RUN,development_data
from .review_sprint_analysis import analyze as audit_initial
from .review_stability import verify_training_unchanged
from .review_compression import OUTPUT,METHODS,jobs,configuration,cell_id,source_base,initialize_projection
from .analyze import md_table

def analyze(base=OUTPUT,sanity=False):
    base=Path(base);proof=verify_training_unchanged()
    state=json.loads((base/'queue_state.json').read_text());planned=jobs(sanity)
    require(state['planned_jobs']==planned,'Unexpected compression grid')
    initial_proof=audit_initial(source_base(sanity),sanity=sanity)
    refs={};rows=[]
    for j in planned:
        c=configuration(j);directory=base/'runs'/cell_id(j)
        require(state['jobs'][cell_id(j)]['status']=='completed','Incomplete queue cell')
        require(json.loads((directory/'status.json').read_text())['status']=='completed','Incomplete run')
        cfg=json.loads((directory/'config.json').read_text());stat=json.loads((directory/'metrics.json').read_text())
        for k,v in c.to_dict().items():require(cfg[k]==v,'Wrong config: '+k)
        require(cfg['projection_hidden']==8 and cfg==stat['config'] and cfg['code']==state['code'] and not cfg['code']['dirty'],'Config provenance mismatch')
        require(stat['backbone_unchanged'],'Backbone changed')
        if c.dataset not in refs:refs[c.dataset]=original_reference(c.dataset)
        ref=refs[c.dataset];n=ref['rows'];a,b,end=int(n*.6),int(n*.7),int(n*.8)
        require(cfg['boundaries']==dict(raw_rows=n,train_end=a,backbone_selection_end=b,development_end=end,scope='exploratory development; old test not evaluated'),'Wrong boundaries')
        require(cfg['data']['train_end']==a and cfg['data']['val_end']==b and cfg['data']['rows']==end,'Wrong data extent')
        same(cfg['data']['mean'],ref['mean'],'Wrong mean');same(cfg['data']['scale'],ref['scale'],'Wrong scale')
        origins=np.arange(b-1,end-c.horizon)
        target=np.stack([ref['target'][t+1:t+c.horizon+1] for t in origins]);mask=np.stack([ref['valid'][t+1:t+c.horizon+1] for t in origins])
        with np.load(directory/'predictions.npz') as z:
            same(z['origins'],origins,'Origins',exact=True);same(z['targets'],target,'Original targets');same(z['valid_mask'],mask,'Masks',exact=True)
            scored=metrics(z['predictions'],target,mask);feedback=z['feedback_origins'].copy()
        for k in ['mse','mae']:require(abs(scored[k]-stat[k])<=1e-12,'Metric mismatch')
        events=verify_trace(directory/'updates.jsonl',origins,c,ref)
        verify_trace(directory/'warm_updates.jsonl',np.arange(c.lookback-1,a-c.horizon),c,ref)
        by_time={e['update_at']:e for e in events};latest=[-1,-1];expected=[]
        for t in origins:
            expected.append(latest.copy());e=by_time.get(int(t))
            if e and e['status']=='updated':latest=[e['origin'],e['origin']]
        same(feedback,expected,'Feedback chronology',exact=True)
        contexts=json.loads((directory/'prediction_context.json').read_text())
        require([x['origin'] for x in contexts]==origins.tolist(),'Context origins')
        require(all(x['text_ids']==ref['text_ids'][x['origin']] for x in contexts),'Text provenance')
        cp=Path(stat['backbone_checkpoint']);declared=source_base(sanity)/'checkpoints'/f'{c.dataset}_h6_s{c.seed}.pt'
        require(cp.resolve()==declared.resolve(),'Unmatched backbone')
        stored=torch.load(cp,map_location='cpu',weights_only=True)
        data,_=development_data(c.dataset,True)
        require(stored['config']==backbone_config(c,data),'Backbone configuration')
        require(stored['source_revision']['commit']==initial_proof['experiment_commit'],'Backbone source revision')
        initial=torch.load(directory/'initial_adapter.pt',map_location='cpu',weights_only=True)
        # Audit initialization independently of saved summary counts.
        c.device='cpu';fresh=initialize_projection(c,len(data.columns),8)
        require(set(initial)==set(fresh.state_dict()),'Initial state keys')
        for k,v in fresh.state_dict().items():require(torch.equal(initial[k],v),'Initial tensor: '+k)
        require(stat['trainable_parameters']==sum(p.numel() for p in fresh.parameters()),'Parameter count')
        require(stat['projection_parameters']==6856,'Projection count')
        if not sanity:
            control=source_base()/ 'runs'/f'{c.dataset}_h6_s{c.seed}_{c.variant}'
            require(Path(cfg['control_source']).resolve()==control.resolve(),'Wrong reused control')
            other=torch.load(control/'initial_adapter.pt',map_location='cpu',weights_only=True)
            for k,v in initial.items():
                if v.shape==other[k].shape:require(torch.equal(v,other[k]),'Compatible initialization mismatch: '+k)
            wide=json.loads((control/'metrics.json').read_text())
            require(wide['backbone_checkpoint']==str(cp),'Different control backbone')
            rows.append(dict(dataset=c.dataset,seed=c.seed,variant=c.variant,width=96,setting='E0' if c.variant==METHODS[0] else 'E2',mse=wide['mse'],mae=wide['mae'],parameters=wide['trainable_parameters'],source=str(control),commit=wide['code']['commit'],reused=True))
        rows.append(dict(dataset=c.dataset,seed=c.seed,variant=c.variant,width=8,setting='E1' if c.variant==METHODS[0] else 'E3',mse=stat['mse'],mae=stat['mae'],parameters=stat['trainable_parameters'],source=str(directory),commit=cfg['code']['commit'],reused=False))
    out=base/'reports';out.mkdir(exist_ok=True);frame=pd.DataFrame(rows);frame.to_csv(out/'runs.csv',index=False)
    summary=frame.groupby(['dataset','setting'],sort=True).agg(n=('seed','size'),mse_mean=('mse','mean'),mse_sd=('mse','std'),mae_mean=('mae','mean'),mae_sd=('mae','std'),parameters=('parameters','first')).reset_index();summary.to_csv(out/'summary.csv',index=False)
    if not sanity:
        references=pd.read_csv(DEFAULT_RUN/'reports/runs.csv')
        references=references[references.variant.isin(['numeric_add','text_add'])].copy()
        require(len(references)==12,'Missing strong simple reference cells')
        references.to_csv(out/'simple_reference_runs.csv',index=False)
        references.groupby(['dataset','variant']).agg(n=('seed','size'),mse_mean=('mse','mean'),mse_sd=('mse','std'),mae_mean=('mae','mean'),mae_sd=('mae','std'),parameters=('trainable_parameters','first')).to_csv(out/'simple_reference_summary.csv')
        paired=[]
        for (d,s),block in frame.groupby(['dataset','seed']):
            b=block.set_index('setting')
            for candidate,control in [('E1','E0'),('E2','E0'),('E3','E0'),('E3','E2')]:
                paired.append(dict(dataset=d,seed=s,candidate=candidate,control=control,delta_mse=b.loc[candidate,'mse']-b.loc[control,'mse'],delta_mae=b.loc[candidate,'mae']-b.loc[control,'mae']))
        pairs=pd.DataFrame(paired);pairs.to_csv(out/'paired_runs.csv',index=False)
        pairs.groupby(['dataset','candidate','control']).agg(n=('seed','size'),delta_mse_mean=('delta_mse','mean'),delta_mse_sd=('delta_mse','std'),delta_mae_mean=('delta_mae','mean'),delta_mae_sd=('delta_mae','std'),mse_improved_seeds=('delta_mse',lambda x:int((x<0).sum())),mae_improved_seeds=('delta_mae',lambda x:int((x<0).sum()))).to_csv(out/'paired_summary.csv')
        decisions=[]
        for d,block in summary.groupby('dataset'):
            b=block.set_index('setting')
            for v in ['E1','E2','E3']:
                mse=100*(b.loc[v,'mse_mean']/b.loc['E0','mse_mean']-1);mae=100*(b.loc[v,'mae_mean']/b.loc['E0','mae_mean']-1);reduction=100*(1-b.loc[v,'parameters']/b.loc['E0','parameters'])
                decisions.append(dict(dataset=d,setting=v,mse_change_percent=mse,mae_change_percent=mae,parameter_reduction_percent=reduction,accuracy_parameter_gate_pass=bool(mse<=1 and mae<=1 and reduction>=40)))
        pd.DataFrame(decisions).to_csv(out/'decisions.csv',index=False)
    validation=dict(status='passed',new_runs=len(planned),reused_runs=0 if sanity else len(planned),combined_cells=len(frame),original_csv_targets_masks=True,feedback_maturity_verified=True,initial_compatible_tensors_equal=True,backbones_reused=True,projection_parameters=6856,experiment_commit=state['code']['commit'],scope='exploratory development only; three seeds for formal matrix',**proof)
    write_json(out/'validation.json',validation)
    (out/'RESULTS.md').write_text('# E-01 开发精度与参数\n\nE0/E1：五专家、无路由文本梯度和 entropy，投影中间宽96/8；E2/E3：单MLP，投影96/8。原始目标、掩码、成熟反馈和初始化检查通过。\n\n'+md_table(summary.round(6))+'\n\n百分比和门槛见 decisions.csv，隔离投影效应及所有 seed 见 paired_runs.csv。n=3 是优化 seed 数，不是独立数据集；sanity n=1 不估计波动。成本必须另测，不用训练总时长替代延迟。\n',encoding='utf-8')
    print('COMPRESSION_VALIDATED',json.dumps(validation),flush=True)
    return validation

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sanity',action='store_true');p.add_argument('--run_dir',type=Path);a=p.parse_args()
    analyze(a.run_dir or (OUTPUT.with_name(OUTPUT.name+'_sanity') if a.sanity else OUTPUT),a.sanity)
