"""Original-CSV, common-maturity and matched-start audits for online extensions."""
import argparse,json
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .audit import original_reference,require,same,verify_trace
from .external_static import pilot_config
from .external_static_matrix import groups,group_id,group_path,OUTPUT as STATIC_MATRIX
from .external_online import OUTPUT,METHODS
from .util import metrics,write_json
from .analyze import md_table

def audit_group(base,job):
    base=Path(base);state=json.loads((base/'queue_state.json').read_text());ref=original_reference(job['dataset'])
    n=ref['rows'];a,b,end=int(.6*n),int(.7*n),int(.8*n)
    k=ref['columns'].index('OT');origins=np.arange(b-1,end-6)
    target=np.stack([ref['target'][t+1:t+7,k:k+1] for t in origins]);mask=np.stack([ref['valid'][t+1:t+7,k:k+1] for t in origins]);rows=[]
    for method,static_method in METHODS.items():
        directory=base/'groups'/group_id(job)/method;source=group_path(STATIC_MATRIX,job)/'runs'/static_method
        key=group_id(job)+'/'+method
        require(state['jobs'][key]['status']=='completed' and json.loads((directory/'status.json').read_text())['status']=='completed','Incomplete online cell')
        cfg=json.loads((directory/'config.json').read_text());stat=json.loads((directory/'metrics.json').read_text());static_cfg=json.loads((source/'config.json').read_text());static_stat=json.loads((source/'metrics.json').read_text())
        c=replace(pilot_config('numeric_add' if method=='numeric_online' else 'full',**job),protocol='development_60_10_10_online_extension_v1')
        require(cfg==stat['config'] and cfg['code']==state['code'] and not state['code']['dirty'],'Online provenance mismatch')
        require((cfg['dataset'],cfg['seed'],cfg['horizon'],cfg['lookback'],cfg['method'],cfg['protocol'])==(c.dataset,c.seed,6,c.lookback,method,c.protocol),'Wrong task')
        require(Path(cfg['static_source']).resolve()==source.resolve() and cfg['static_source_config']==static_cfg,'Wrong static counterpart')
        require(cfg['data']==static_cfg['data'] and cfg['data']['columns']==['OT'] and cfg['data']['rows']==end and cfg['data']['train_end']==a and cfg['data']['val_end']==b,'Wrong OT boundaries')
        same(cfg['data']['mean'],ref['mean'][k:k+1],'Scaler mean');same(cfg['data']['scale'],ref['scale'][k:k+1],'Scaler scale')
        require(cfg['optimizer']==dict(name='Adam',lr=.001,weight_decay=.0001,grad_clip=1.,batch=1) and cfg['prediction_before_update'],'Optimizer/timeline mismatch')
        native=method=='gpt4mts_online_extension'
        require(cfg['has_gradient_feedback_cache']==(not native),'Wrong feedback interface')
        smallref=dict(ref);smallref['valid']=ref['valid'][:,k:k+1]
        if native:smallref['text_ids']=[[rid for entry in ref['text_ids'][max(0,t-c.lookback+1):t+1] for rid in entry] for t in range(n)]
        events=verify_trace(directory/'updates.jsonl',origins,c,smallref)
        require(stat['consumed']==len(events) and stat['updated']==sum(e['status']=='updated' for e in events) and stat['pending']==6,'Mature counts mismatch')
        require(stat['initial_parameters_match_static'] and stat['frozen_parameters_unchanged'],'Parameter checks failed')
        initial=torch.load(directory/'initial_adaptive_state.pt',map_location='cpu',weights_only=True)
        expected=torch.load(Path(static_stat['backbone_checkpoint']),map_location='cpu',weights_only=True)['trainable'] if native else torch.load(source/'warm_adapter.pt',map_location='cpu',weights_only=True)
        require(set(initial)==set(expected) and all(torch.equal(initial[name],v) for name,v in expected.items()),'Different start weights')
        final=torch.load(directory/'final_adaptive_state.pt',map_location='cpu',weights_only=True)
        require(set(final)==set(initial) and all(torch.isfinite(v).all() for v in final.values()),'Final adaptive state invalid')
        require(sum(v.numel() for v in initial.values())==stat['trainable_parameters'],'Parameter count differs')
        contexts=json.loads((directory/'contexts.json').read_text());require([x['origin'] for x in contexts]==origins.tolist(),'Context origin mismatch')
        with np.load(directory/'predictions.npz') as z,np.load(source/'predictions.npz') as static:
            same(z['origins'],origins,'Origin mismatch',exact=True);same(z['targets'],target,'Raw OT target mismatch');same(z['valid_mask'],mask,'Raw mask mismatch',exact=True)
            same(static['origins'],origins,'Static origin mismatch',exact=True);same(static['targets'],target,'Static targets');same(static['valid_mask'],mask,'Static masks',exact=True)
            require(z['predictions'].shape==target.shape and np.isfinite(z['predictions']).all(),'Invalid forecasts')
            score=metrics(z['predictions'],target,mask)
            for metric in ['mse','mae']:require(abs(score[metric]-stat[metric])<=1e-12,'Saved metric mismatch')
            require(stat['valid_elements']==int(mask.sum()),'Valid count mismatch')
            np.testing.assert_allclose(z['predictions'][:7],static['predictions'][:7],rtol=1e-6,atol=1e-6)
            last=-1;feedback=[-1,-1];event_at={e['update_at']:e for e in events}
            for index,t in enumerate(origins):
                context=contexts[index];ids=smallref['text_ids'][t] if c.uses_text else []
                require(context['text_ids']==ids and all(int(x.split(':')[0])<=t for x in ids),'Future or wrong text')
                require(int(z['parameter_update_origins'][index])==last==context['last_parameter_update_origin'],'New parameters used too early')
                same(z['feedback_origins'][index],feedback,'Feedback chronology',exact=True)
                require(context['numeric_feedback_origin']==feedback[0] and context['text_feedback_origin']==feedback[1],'Feedback context mismatch')
                event=event_at.get(int(t))
                if event and event['status']=='updated':
                    last=event['origin']
                    if not native:
                        if c.numeric_feedback:feedback[0]=last
                        if c.text_feedback:feedback[1]=last
            if native:require('numeric_feedback_snapshots' not in z.files and 'text_feedback_snapshots' not in z.files,'Native method must not claim gradient caches')
            else:
                require(z['numeric_feedback_snapshots'].shape==(len(origins),1,32) and z['text_feedback_snapshots'].shape==(len(origins),64),'Snapshot dimensions')
                require(np.isfinite(z['numeric_feedback_snapshots']).all() and np.isfinite(z['text_feedback_snapshots']).all(),'Invalid feedback snapshot')
                require(not z['numeric_feedback_snapshots'][:7].any() and not z['text_feedback_snapshots'][:7].any(),'Warm feedback leaked into evaluation')
        rows.append(dict(dataset=c.dataset,seed=c.seed,method=method,static_method=static_method,mse=score['mse'],mae=score['mae'],static_mse=static_stat['mse'],static_mae=static_stat['mae'],delta_mse_online_minus_static=score['mse']-static_stat['mse'],delta_mae_online_minus_static=score['mae']-static_stat['mae'],origins=len(origins),valid_elements=int(mask.sum()),updated=stat['updated'],pending=stat['pending'],seconds=stat['seconds'],trainable_parameters=stat['trainable_parameters'],source=str(directory),static_source=str(source)))
    out=base/'groups'/group_id(job);pd.DataFrame(rows).to_csv(out/'summary.csv',index=False)
    proof=dict(status='passed',dataset=job['dataset'],seed=job['seed'],completed=3,original_csv_targets_masks=True,initial_weights_match_static=True,predict_before_mature_update=True,once_only_updates=True,first_7_predictions_match_static=True,code=state['code'])
    write_json(out/'validation.json',proof);print('ONLINE_GROUP_VALIDATED',group_id(job),flush=True)
    return rows,proof

def analyze(base=OUTPUT):
    base=Path(base);state=json.loads((base/'queue_state.json').read_text());require(state['groups']==groups() and state['methods']==METHODS,'Matrix differs')
    rows=[];proofs=[]
    for job in groups():
        group,proof=audit_group(base,job);rows+=group;proofs.append(proof)
    runs=pd.DataFrame(rows);require(len(runs)==75 and not runs.duplicated(['dataset','seed','method']).any(),'Missing/duplicate online cells')
    out=base/'reports';out.mkdir(exist_ok=True);runs.to_csv(out/'runs.csv',index=False)
    summary=runs.groupby(['dataset','method'],sort=False).agg(n=('seed','count'),mse_mean=('mse','mean'),mse_sd=('mse','std'),mae_mean=('mae','mean'),mae_sd=('mae','std'),updated=('updated','first'),origins=('origins','first')).reset_index();summary.to_csv(out/'summary.csv',index=False)
    change=runs.groupby(['dataset','method'],sort=False).agg(n=('seed','count'),delta_mse_mean=('delta_mse_online_minus_static','mean'),delta_mse_sd=('delta_mse_online_minus_static','std'),delta_mae_mean=('delta_mae_online_minus_static','mean'),delta_mae_sd=('delta_mae_online_minus_static','std'),mse_improved_seeds=('delta_mse_online_minus_static',lambda x:int((x<0).sum())),mae_improved_seeds=('delta_mae_online_minus_static',lambda x:int((x<0).sum()))).reset_index();change.to_csv(out/'online_minus_static.csv',index=False)
    paired=[]
    for (d,s),block in runs.groupby(['dataset','seed']):
        b=block.set_index('method')
        for control in ['numeric_online','gpt4mts_online_extension']:
            paired.append(dict(dataset=d,seed=s,candidate='iadapter_online',control=control,delta_mse=b.loc['iadapter_online','mse']-b.loc[control,'mse'],delta_mae=b.loc['iadapter_online','mae']-b.loc[control,'mae']))
    paired=pd.DataFrame(paired);paired.to_csv(out/'paired_runs.csv',index=False)
    paired.groupby(['dataset','control'],sort=False).agg(n=('seed','count'),delta_mse_mean=('delta_mse','mean'),delta_mse_sd=('delta_mse','std'),delta_mae_mean=('delta_mae','mean'),delta_mae_sd=('delta_mae','std'),mse_improved_seeds=('delta_mse',lambda x:int((x<0).sum())),mae_improved_seeds=('delta_mae',lambda x:int((x<0).sum()))).to_csv(out/'paired_summary.csv')
    write_json(out/'validation.json',dict(status='passed',completed=75,group_proofs=proofs,source_code=state['code'],scope='common OT development; native-model online extension; not original Table2 or publication-time confirmation'))
    (out/'RESULTS.md').write_text('# 共同延迟协议在线扩展\n\n75/75运行完成，25组原CSV/起始权重/成熟事件核验通过。GPT4MTS是新定义的online extension；内部方法保留原反馈机制。起始权重与静态表完全相同，前7个尚未更新的预测在fp32容差内一致。\n\n'+md_table(summary.round(6))+'\n\n## Online减Static\n\n负值表示在线误差更低。内部方法同时启用参数更新与反馈刷新，不能单独归因于缓存。\n\n'+md_table(change.round(6))+'\n\n文本历史/编码与训练阶段的跨方法差异仍存在；各方法仅共享原点/目标、可用性代理与成熟更新机会。五seed是固定开发流上的优化重复，无显著性或独立泛化声明。仍无可信VoT来源，不恢复原Table2。\n',encoding='utf-8')
    print('ONLINE_MATRIX_VALIDATED 75 / 75',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUTPUT);a=p.parse_args();analyze(a.output)
