"""Audit and summarize the development sprint against original CSV targets."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .config import Config,job_id
from .audit import original_reference,require,same,verify_trace
from .review_sprint import DEFAULT_RUN,VARIANTS,CONDITIONS,matrix
from .util import metrics,write_json
from .analyze import md_table

def analyze(base,sanity=False,jobs_override=None,factorial=True):
    base=Path(base);state=json.loads((base/'queue_state.json').read_text());rows=[];probes=[];refs={};checkpoints={};shared={};paired_streams={}
    jobs=matrix(sanity) if jobs_override is None else jobs_override
    require(state['planned_jobs']==jobs,'Analysis job list differs from registered queue')
    for job in jobs:
        key=job_id(dict(job,horizon=6));directory=base/'runs'/key
        require(state['jobs'][key]['status']=='completed',f'Incomplete queue cell: {key}')
        require(json.loads((directory/'status.json').read_text())['status']=='completed',f'Incomplete run: {key}')
        cfg=json.loads((directory/'config.json').read_text());saved=json.loads((directory/'metrics.json').read_text())
        c=Config(job['dataset'],6,job['seed'],job['variant'],protocol='development_60_10_10_v1')
        for name,value in c.to_dict().items():require(cfg[name]==value,f'Config mismatch: {key}/{name}')
        require(cfg==saved['config'] and cfg['code']==state['code'] and not cfg['code']['dirty'],'Configuration/provenance mismatch')
        if c.dataset not in refs:refs[c.dataset]=original_reference(c.dataset)
        ref=refs[c.dataset];n=ref['rows'];a=int(.6*n);b=int(.7*n);end=int(.8*n)
        require(cfg['boundaries']['raw_rows']==n and cfg['boundaries']['train_end']==a and cfg['boundaries']['backbone_selection_end']==b and cfg['boundaries']['development_end']==end,'Development boundaries mismatch')
        require(cfg['data']['train_end']==a and cfg['data']['val_end']==b and cfg['data']['rows']==end,'Development data shape mismatch')
        same(cfg['data']['mean'],ref['mean'],'Source training-prefix mean mismatch');same(cfg['data']['scale'],ref['scale'],'Source training-prefix scale mismatch')
        origins=np.arange(b-1,end-c.horizon);target=np.stack([ref['target'][s+1:s+1+c.horizon] for s in origins]);mask=np.stack([ref['valid'][s+1:s+1+c.horizon] for s in origins])
        with np.load(directory/'predictions.npz') as z:
            same(z['origins'],origins,'Origin mismatch',exact=True);same(z['targets'],target,'Original CSV target mismatch');same(z['valid_mask'],mask,'Original CSV mask mismatch',exact=True)
            scored=metrics(z['predictions'],target,mask)
            for name in ['mse','mae']:require(np.isclose(scored[name],saved[name],rtol=0,atol=1e-12),'Saved metric mismatch')
            prediction=z['predictions'].copy();gate=z['gate_weights'].copy();feedback=z['feedback_origins'].copy()
        events=verify_trace(directory/'updates.jsonl',origins,c,ref)
        warm_origins=np.arange(c.lookback-1,a-c.horizon)
        verify_trace(directory/'warm_updates.jsonl',warm_origins,c,ref)
        latest=[-1,-1];expected_feedback=[];by_time={e['update_at']:e for e in events}
        for t in origins:
            expected_feedback.append(latest.copy());event=by_time.get(int(t))
            if event and event['status']=='updated':
                if c.numeric_feedback:latest[0]=event['origin']
                if c.text_feedback:latest[1]=event['origin']
        same(feedback,expected_feedback,'Feedback reset/chronology mismatch',exact=True)
        contexts=json.loads((directory/'prediction_context.json').read_text())
        require([r['origin'] for r in contexts]==origins.tolist(),'Prediction context mismatch')
        require(all(r['text_ids']==(ref['text_ids'][r['origin']] if c.uses_text else []) for r in contexts),'Prediction text IDs mismatch')
        cp=Path(saved['backbone_checkpoint']);require(cp.parent.resolve()==(base/'checkpoints').resolve(),'Checkpoint is outside isolated sprint')
        if cp not in checkpoints:checkpoints[cp]=torch.load(cp,map_location='cpu',weights_only=True)
        bc=checkpoints[cp]['config']
        require(bc['train_end']==a and bc['val_end']==b and checkpoints[cp]['source_revision']==state['code'],'Backbone selected on wrong split/revision')
        require(bc['dataset']==c.dataset and bc['seed']==c.seed and bc['horizon']==6,'Backbone identity mismatch')
        group=(c.dataset,c.seed)
        if group in paired_streams:require(paired_streams[group]==str(cp),'Methods do not share checkpoint')
        else:paired_streams[group]=str(cp)
        initial=torch.load(directory/'initial_adapter.pt',map_location='cpu',weights_only=True)
        if c.variant in ['full','no_route_text_gradient','no_decisiveness','no_route_no_decisiveness','single_mlp']:
            if group in shared:
                for name,value in initial.items():
                    if name in shared[group]:require(torch.equal(value,shared[group][name]),'Shared initialization mismatch: '+name)
            else:shared[group]=initial
        row=dict(job,**scored,seconds=saved['seconds'],trainable_parameters=saved['trainable_parameters'],peak_cuda_allocated_bytes=saved['peak_cuda_allocated_bytes'],residual_ratio_mean=float(np.mean([r['residual_to_feature_norm'] for r in contexts])))
        if np.isfinite(gate).all():
            row['effective_experts_mean']=float((1/(gate.astype(float)**2).sum(-1)).mean());row['dominant_mean_weight']=float(gate.mean((0,1)).max())
        rows.append(row)
        if c.variant in ['full','text_add']:
            with np.load(directory/'text_interventions.npz') as z:
                require(z['conditions'].tolist()==CONDITIONS,'Condition ordering mismatch')
                same(z['origins'],origins,'Probe origin mismatch',exact=True);same(z['predictions'][:,0],prediction,'Probe reference prediction mismatch',exact=True)
                donor=z['donor_origins'];eligible=z['eligible'];values=z['predictions']
                require(np.array_equal(eligible,donor>=0),'Probe eligibility mismatch')
                require(np.all(donor[eligible]<origins[eligible]),'Future donor text')
                details=json.loads((directory/'text_interventions.json').read_text())
                require(len(details)==len(origins),'Missing probe trace')
                for t,d,e,detail in zip(origins,donor,eligible,details):
                    require(detail['origin']==t and detail['donor']==d,'Donor trace mismatch')
                    if e:
                        require(bool(ref['texts'][t]) and bool(ref['texts'][d]),'Empty donor/reference')
                        require(abs(detail['reference_tokens']-detail['donor_tokens'])<=8,'Length matching violated')
                        require(detail['donor_text_ids']==ref['text_ids'][d],'Donor text provenance mismatch')
                common=mask&eligible[:,None,None]
                if common.any():
                    control=metrics(values[:,0],target,common)
                    for index,condition in enumerate(CONDITIONS):
                        actual=metrics(values[:,index],target,common)
                        divergence=float(np.mean(np.abs((values[:,index]-values[:,0])[common]).astype(float)))
                        probes.append(dict(job,condition=condition,eligible_origins=int(eligible.sum()),total_origins=len(origins),**actual,delta_mse_vs_correct=actual['mse']-control['mse'],delta_mae_vs_correct=actual['mae']-control['mae'],prediction_divergence_mae=divergence))
                else:raise ValueError(f'No matched donor coverage: {key}')
    out=base/'reports';out.mkdir(exist_ok=True)
    frame=pd.DataFrame(rows);frame.to_csv(out/'runs.csv',index=False)
    summary=frame.groupby(['dataset','variant'],sort=False).agg(n=('seed','count'),mse_mean=('mse','mean'),mse_sd=('mse','std'),mae_mean=('mae','mean'),mae_sd=('mae','std'),parameters=('trainable_parameters','first'),seconds_mean=('seconds','mean'),residual_ratio_mean=('residual_ratio_mean','mean')).reset_index()
    summary.to_csv(out/'main_summary.csv',index=False)
    probe=pd.DataFrame(probes);probe.to_csv(out/'text_probe_runs.csv',index=False)
    probe_summary=probe.groupby(['dataset','variant','condition'],sort=False).agg(n=('seed','count'),delta_mse_mean=('delta_mse_vs_correct','mean'),delta_mse_sd=('delta_mse_vs_correct','std'),delta_mae_mean=('delta_mae_vs_correct','mean'),delta_mae_sd=('delta_mae_vs_correct','std'),prediction_divergence_mae=('prediction_divergence_mae','mean'),eligible_origins=('eligible_origins','min'),total_origins=('total_origins','first')).reset_index()
    probe_summary.to_csv(out/'text_probe_summary.csv',index=False)
    paired=[];interaction=[]
    if not sanity:
        for (dataset,seed),block in frame.groupby(['dataset','seed']):
            by=block.set_index('variant')
            expected_methods={j['variant'] for j in jobs if j['dataset']==dataset and j['seed']==seed}
            require(set(by.index)==expected_methods,'Missing method in paired cell')
            if factorial:require(set(by.index)==set(VARIANTS),'Missing method in factorial cell')
            for candidate in by.index:
                for control in ['full','numeric_add','text_add']:
                    if candidate==control:continue
                    paired.append(dict(dataset=dataset,seed=int(seed),candidate=candidate,control=control,delta_mse=float(by.loc[candidate,'mse']-by.loc[control,'mse']),delta_mae=float(by.loc[candidate,'mae']-by.loc[control,'mae'])))
            for metric in (['mse','mae'] if factorial else []):
                interaction.append(dict(dataset=dataset,seed=int(seed),metric=metric,route_off_when_entropy_on=float(by.loc['no_route_text_gradient',metric]-by.loc['full',metric]),route_off_when_entropy_off=float(by.loc['no_route_no_decisiveness',metric]-by.loc['no_decisiveness',metric]),entropy_off_when_route_on=float(by.loc['no_decisiveness',metric]-by.loc['full',metric]),entropy_off_when_route_off=float(by.loc['no_route_no_decisiveness',metric]-by.loc['no_route_text_gradient',metric]),interaction=float(by.loc['no_route_no_decisiveness',metric]-by.loc['no_route_text_gradient',metric]-by.loc['no_decisiveness',metric]+by.loc['full',metric])))
        pairs=pd.DataFrame(paired);pairs.to_csv(out/'paired_runs.csv',index=False)
        pairs.groupby(['dataset','candidate','control']).agg(n=('seed','count'),delta_mse_mean=('delta_mse','mean'),delta_mse_sd=('delta_mse','std'),delta_mae_mean=('delta_mae','mean'),delta_mae_sd=('delta_mae','std'),mse_improved_seeds=('delta_mse',lambda s:int((s<0).sum()))).reset_index().to_csv(out/'paired_summary.csv',index=False)
        if factorial:pd.DataFrame(interaction).to_csv(out/'factorial_interaction.csv',index=False)
    proof={'status':'passed','completed':len(frame),'planned':len(jobs),'unique_backbones':len(checkpoints),'original_csv_targets_and_masks':True,'development_boundaries_60_70_80':True,'once_only_mature_updates':True,'shared_factorial_initialization':True,'probe_sequences':len(probe),'future_donors':False,'test_80_100_evaluated':False,'experiment_commit':state['code']['commit'],'total_cell_seconds':float(frame.seconds.sum()),'scope':'development-only; seed counts are stated per table; no significance claim'}
    write_json(out/'validation.json',proof)
    text='# 开发实验结果（不属于未见测试集确认）\n\n'+md_table(summary.round(6))+'\n\n## 文本内容干预\n\n差值为控制文本减正确文本，正值表示正确文本误差更低。只比较有历史长度匹配 donor 的共同原点；零臂保留历史反馈，只去掉当前文本。差值小或预测变化小不能自动证明原始文本无信息。\n\n'+md_table(probe_summary.round(6))+'\n\n原始配对差值和每seed诊断分别保存在 paired_runs.csv、text_probe_runs.csv；完整2×2设计另存factorial_interaction.csv。n按表记录，跨seed波动只作开发诊断，不作显著性保证；sanity的n=1不估计SD。\n'
    (out/'RESULTS.md').write_text(text,encoding='utf-8')
    print('SPRINT_VALIDATED',json.dumps(proof),flush=True)
    return proof

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run_dir',type=Path,default=DEFAULT_RUN);p.add_argument('--sanity',action='store_true');a=p.parse_args();analyze(a.run_dir,a.sanity)
