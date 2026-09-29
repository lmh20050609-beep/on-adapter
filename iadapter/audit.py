"""Read-only verification against original CSVs, not saved targets alone.

Default execution only prints a summary. --output writes a derived audit report,
never a run, checkpoint, queue, prediction or source dataset.
"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .config import ROOT,DATA_ROOT,Config,matrix,job_id
from .data import text_rows
from .util import write_json
from .evidence_paths import resolve_evidence_path

class AuditError(ValueError): pass

def require(condition,message):
    if not condition: raise AuditError(message)

def read_json(path):
    path=Path(path)
    require(path.is_file(),f'Missing required file: {path}')
    return json.loads(path.read_text())

def same(actual,expected,message,exact=False):
    a=np.asarray(actual); b=np.asarray(expected)
    require(a.shape==b.shape,f'{message}: shape {a.shape} != {b.shape}')
    matched=np.array_equal(a,b) if exact else np.allclose(a,b,rtol=1e-6,atol=1e-7,equal_nan=False)
    require(matched,message)

def original_reference(dataset,data_root=DATA_ROOT):
    path=Path(data_root)/f'{dataset}.csv'
    frame=pd.read_csv(path)
    datecol=next(c for c in ['end_date','date','Date','start_date'] if c in frame)
    dates=pd.to_datetime(frame[datecol],errors='raise')
    order=np.argsort(dates.values,kind='stable')
    frame=frame.iloc[order].reset_index(drop=True)
    dates=dates.iloc[order].reset_index(drop=True)
    # Explicit independently audited field policy, not Data.target/load_data.
    excluded={'date','Date','start_date','end_date','fact','pred','preds','Final_Output','Final_Search_8','Final_Search_16','prior_history_avg','prior_history_std'}
    columns=[c for c in frame if c not in excluded and not c.startswith('Unnamed:') and pd.api.types.is_numeric_dtype(frame[c])]
    values=frame[columns].to_numpy(dtype=np.float32)
    observed=np.isfinite(values)
    filled=np.empty_like(values); previous=np.zeros(len(columns),dtype=np.float32)
    for i,row in enumerate(values):
        previous=np.where(observed[i],row,previous)
        filled[i]=previous
    n=len(values); train_end=int(.6*n); val_end=int(.8*n)
    mean=filled[:train_end].mean(axis=0,dtype=np.float64).astype(np.float32)
    scale=filled[:train_end].std(axis=0,dtype=np.float64).astype(np.float32)
    scale=np.where(scale<1e-8,1.,scale)
    target=np.zeros_like(values)
    target[observed]=((values-mean)/scale)[observed]
    texts,ids=text_rows(frame)
    return dict(source=str(path),columns=columns,rows=n,train_end=train_end,val_end=val_end,mean=mean,scale=scale,x=(filled-mean)/scale,target=target,valid=observed,texts=texts,text_ids=ids,dates=[str(d) for d in dates],input_missing=~observed)

def historical_state_reference(ref,lookback):
    cache=ref.setdefault('historical_state',{})
    if lookback not in cache:
        scores=np.zeros(ref['rows'],dtype=np.float32)
        for s in range(lookback-1,ref['rows']):
            window=ref['x'][s-lookback+1:s+1]
            first,last=window[:lookback//2],window[lookback//2:]
            scores[s]=np.mean(np.abs(first.mean(0)-last.mean(0))+np.abs(first.std(0)-last.std(0)))
        thresholds=np.quantile(scores[max(lookback-1,ref['train_end']-1):ref['val_end']-1],[1/3,2/3])
        cache[lookback]=(scores,thresholds)
    return cache[lookback]

def expected_arrays(ref,c,split='test'):
    start,end=(ref['val_end']-1,ref['rows']) if split=='test' else (c.lookback-1,ref['train_end'])
    origins=np.arange(start,end-c.horizon,dtype=np.int64)
    targets=np.stack([ref['target'][s+1:s+c.horizon+1] for s in origins])
    mask=np.stack([ref['valid'][s+1:s+c.horizon+1] for s in origins])
    return origins,targets,mask

def verify_trace(path,origins,c,ref):
    require(Path(path).is_file(),f'Missing required file: {path}')
    events=[json.loads(line) for line in Path(path).read_text().splitlines()]
    eligible=[int(s) for s in origins if s+c.horizon<=origins[-1]]
    require([e['origin'] for e in events]==eligible,'Missing, repeated or reordered mature origins')
    for e in events:
        s=e['origin']; mature=s+c.horizon
        require(e['mature_at']==mature and e['update_at']==mature,'Update does not follow the locked maturity schedule')
        require(e['text_origin']==s and e['state_origin']==s,'Replay origin mismatch')
        require(e['text_ids']==(ref['text_ids'][s] if c.uses_text else []),'Replay selected text differs from origin text')
        count=int(ref['valid'][s+1:mature+1].sum())
        require(e['valid_targets']==count,'Update target count differs from original CSV')
        require(e['status']==('updated' if count else 'skipped_missing_target'),'Missing-target update status mismatch')
    return events

def audit_run(directory,job,reference,check_checkpoint=True,checkpoint_cache=None):
    directory=Path(directory); c=Config(job['dataset'],job['horizon'],job['seed'],job['variant'])
    status=read_json(directory/'status.json'); require(status.get('status')=='completed','Run is not complete')
    cfg=read_json(directory/'config.json'); result=read_json(directory/'metrics.json')
    for key,value in c.to_dict().items():
        require(cfg.get(key)==value,f'Effective configuration mismatch: {key}')
        require(result['config'].get(key)==value,f'Metric configuration mismatch: {key}')
    require(cfg['code']==result['code'] and not result['code']['dirty'],'Run revision mismatch or dirty experiment code')
    require(len(result['code']['commit'])==40,'Missing experiment commit identity')
    meta=cfg['data']; ref=reference
    require(resolve_evidence_path(meta['source'])==Path(ref['source']).resolve(),'Dataset source mismatch')
    for key in ['columns','rows','train_end','val_end']:
        require(meta[key]==ref[key],f'Dataset metadata differs from original CSV: {key}')
    same(meta['mean'],ref['mean'],'Training-only scaler mean mismatch')
    same(meta['scale'],ref['scale'],'Training-only scaler scale mismatch')
    require(meta['missing_targets']==int((~ref['valid']).sum()),'Raw missing-target count mismatch')
    path=directory/'predictions.npz'; require(path.is_file(),f'Missing required file: {path}')
    origin,expected_y,expected_mask=expected_arrays(ref,c)
    state_scores,state_thresholds=historical_state_reference(ref,c.lookback)
    with np.load(path,allow_pickle=False) as saved:
        same(saved['origins'],origin,'Forecast origins differ from prescribed split',exact=True)
        same(saved['targets'],expected_y,'Saved targets differ from original CSV')
        same(saved['valid_mask'],expected_mask,'Saved validity mask differs from original missing targets',exact=True)
        pred=saved['predictions']; require(pred.shape==expected_y.shape and np.isfinite(pred).all(),'Invalid prediction shape/values')
        # util.metrics subtracts float32 before promoting; reproduce its rounding.
        delta=(pred-expected_y)[expected_mask].astype(np.float64)
        recomputed={'mse':float(np.mean(delta**2)),'mae':float(np.mean(np.abs(delta))),'valid_elements':int(expected_mask.sum())}
        for metric in ['mse','mae']:
            require(np.isclose(recomputed[metric],result[metric],rtol=0,atol=1e-12),f'Raw-data metric mismatch: {metric}')
        require(result['valid_elements']==recomputed['valid_elements'],'Scored target count mismatch')
        missing=np.stack([ref['input_missing'][s-c.lookback+1:s+1] for s in origin])
        same(saved['input_missing_mask'],missing,'Input missingness provenance mismatch',exact=True)
        available=np.array([bool(ref['texts'][s]) and c.uses_text for s in origin])
        same(saved['text_available'],available,'Text availability mismatch',exact=True)
        feedback=saved['feedback_origins'].copy()
        same(saved['state_change_scores'],state_scores[origin],'State-change score differs from historical source windows')
        same(saved['state_change_group'],np.digitize(state_scores[origin],state_thresholds),'State-change groups differ from validation-only thresholds',exact=True)
    context=read_json(directory/'prediction_context.json')
    same(context['group_thresholds_validation'],state_thresholds,'State-change thresholds differ from validation-only reference')
    same(context['origin_ids'],origin,'Context origin mismatch',exact=True)
    require(context['text_ids']==[ref['text_ids'][s] if c.uses_text else [] for s in origin],'Prediction text IDs mismatch')
    require(context['dates']==[ref['dates'][s] for s in origin],'Origin timestamps differ from source end dates')
    events=verify_trace(directory/'updates.jsonl',origin,c,ref)
    warm_origin,_,_=expected_arrays(ref,c,'train')
    warm_events=verify_trace(directory/'warm_updates.jsonl',warm_origin,c,ref)
    require(result['consumed']==len(events) and result['warmup']['consumed']==len(warm_events),'Consumed count mismatch')
    require(result['predictions']==len(origin) and result['pending']==c.horizon,'Forecast/pending count mismatch')
    latest=[-1,-1]; expected_feedback=[]; by_time={e['update_at']:e for e in events}
    for t in origin:
        expected_feedback.append(latest.copy())
        e=by_time.get(int(t))
        if e and e['status']=='updated':
            if c.numeric_feedback: latest[0]=e['origin']
            if c.text_feedback: latest[1]=e['origin']
    same(feedback,expected_feedback,'Feedback ages violate predict-before-update or reset',exact=True)
    for filename in ['warm_adapter.pt','final_adapter.pt']:
        require((directory/filename).is_file(),f'Missing required file: {directory/filename}')
    if check_checkpoint:
        path=resolve_evidence_path(result['backbone_checkpoint']); require(path.is_file(),f'Missing backbone: {path}')
        cache=checkpoint_cache if checkpoint_cache is not None else {}
        if str(path) not in cache: cache[str(path)]=torch.load(path,map_location='cpu',weights_only=True)
        checkpoint=cache[str(path)]; bc=checkpoint['config']
        for key in ['dataset','horizon','seed','lookback','d_model','e_layers','dropout','pretrain_epochs','patience','lr','protocol']:
            expected=c.lookback if key=='lookback' else getattr(c,key)
            require(bc[key]==expected,f'Backbone config mismatch: {key}')
        require(bc['columns']==ref['columns'] and bc['train_end']==ref['train_end'] and bc['val_end']==ref['val_end'],'Backbone data split mismatch')
        require(checkpoint['source_revision']==result['code'],'Backbone/run code revision mismatch')
        require(checkpoint['training']==result['backbone_training'],'Backbone training provenance mismatch')
    return dict(job,run_dir=str(directory),**recomputed,original_csv=ref['source'],forecast_origins=len(origin),mature_updates=len(events),code_commit=result['code']['commit'],actual_text_usage='contextual_bert' if c.uses_text else 'none')

def audit_all(root=ROOT,jobs=None,data_root=DATA_ROOT):
    jobs=matrix() if jobs is None else jobs
    references={}; checkpoints={}; checked=[]; failures=[]
    for job in jobs:
        try:
            if job['dataset'] not in references: references[job['dataset']]=original_reference(job['dataset'],data_root)
            checked.append(audit_run(Path(root)/'runs'/job_id(job),job,references[job['dataset']],checkpoint_cache=checkpoints))
        except (AuditError,KeyError,ValueError,OSError,RuntimeError) as error:
            failures.append({'job_id':job_id(job),'error':str(error)})
    return {'status':'passed' if not failures else 'failed','planned':len(jobs),'passed':len(checked),'failed':len(failures),'datasets_checked':len(references),'checkpoints_checked':len(checkpoints),'checks':['original CSV target values','original missing-target masks','training-prefix scaler','configured chronological split','matched origin/text IDs','once-only mature updates','feedback provenance and reset markers','historical-window state scores and validation-only thresholds','backbone configuration and revision','prediction-derived MSE/MAE'],'limits':['Text end_date remains an availability proxy, not verified publication time.','Snapshot tensor values were not serialized in old runs; their numerical equality is supported by implementation tests, not reconstructed from trace IDs.','This does not recover missing original-paper baseline experiments.'],'failures':failures,'runs':checked}

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path)
    args=p.parse_args(); report=audit_all()
    if args.output: write_json(args.output,report)
    print(json.dumps({k:v for k,v in report.items() if k!='runs'},ensure_ascii=False,indent=2))
    if report['status']!='passed': raise SystemExit(1)
