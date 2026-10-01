"""Frozen TimesFM3 issued-forecast ledger for a bounded residual experiment."""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .foundation_preflight import SOURCE,SOURCE_REVISION,WEIGHTS,MODEL_REVISION,historical_context
from .calibration_pilot import full_ot,bounds,region_origins
from .audit import require
from .util import write_json,revision,sync,metrics

def phase_origins(n,lookback,phase):
    b=bounds(n)
    if phase=='fit':return np.arange(lookback-1,b['fit_end']-6,dtype=np.int64)
    if phase=='checkpoint':return np.arange(b['fit_end']-1,b['checkpoint_end']-6,dtype=np.int64)
    return region_origins(n,lookback,6,phase)

def run(run_dir,weights=WEIGHTS):
    run_dir=Path(run_dir);run_dir.mkdir(parents=True,exist_ok=False)
    write_json(run_dir/'status.json',dict(status='running',phase='frozen numerical forecasts'))
    try:
        code=revision();require(not code['dirty'],'Commit source before run')
        receipt=json.loads((weights/'DOWNLOAD_RECEIPT.json').read_text())
        require(receipt['status']=='verified_transport' and receipt['revision']==MODEL_REVISION,'Wrong/incomplete pretrained weights')
        require(json.loads((SOURCE/'SOURCE.json').read_text())['revision']==SOURCE_REVISION,'Wrong source')
        free,total=torch.cuda.mem_get_info();require(free>2.5*2**30,'Wait for adequate free GPU memory')
        sys.path.insert(0,str(SOURCE/'src'))
        from timesfm3 import TimesFM3Forecaster,ModelConfig
        torch.set_num_threads(1);torch.manual_seed(2023);torch.set_float32_matmul_precision('highest')
        model=TimesFM3Forecaster(ModelConfig(checkpoint_path=str(weights),device='cuda',per_core_batch_size=1,local_files_only=True))
        model.model.requires_grad_(False);model.model.eval()
        versions=[p._version for p in model.model.parameters()]
        require(sum(p.numel() for p in model.model.parameters())==330710976,'Unexpected official model size')
        torch.cuda.reset_peak_memory_stats();rows=[]
        for domain,lookback in [('Energy',48),('SocialGood',36)]:
            data=full_ot(domain);boundary=bounds(len(data.x))
            for phase in ['fit','checkpoint','calibration','selection','assessment']:
                output=run_dir/domain/phase;output.mkdir(parents=True,exist_ok=False)
                origins=phase_origins(len(data.x),lookback,phase).tolist()
                lo,hi={'fit':(0,boundary['fit_end']), 'checkpoint':(boundary['fit_end'],boundary['checkpoint_end']),
                    'calibration':(boundary['checkpoint_end'],boundary['calibration_end']),
                    'selection':(boundary['calibration_end'],boundary['selection_end']),
                    'assessment':(boundary['selection_end'],boundary['assessment_end'])}[phase]
                require(all(t+1>=lo and t+6<hi for t in origins),'Target/maturity crosses stage')
                write_json(output/'config.json',dict(dataset=domain,phase=phase,lookback=lookback,horizon=6,code=code,
                    model='TimesFM3 official median, frozen fp32',source_revision=SOURCE_REVISION,model_revision=MODEL_REVISION,
                    boundaries=boundary,phase_target_first=lo,phase_target_end_exclusive=hi,pretrained_optimization_seeds=0,
                    no_future_covariates=True,train_mean=data.mean.tolist(),train_scale=data.scale.tolist(),
                    text_ids_read_by_model=False,foundation_pretraining_overlap_unverified=True))
                write_json(output/'status.json',dict(status='running'));predictions=[];targets=[];masks=[];seconds=[];records=[];persistence=[]
                for i,t in enumerate(origins):
                    ctx=historical_context(data,t,lookback)
                    sync('cuda');start=time.perf_counter()
                    issued=model.predict(ctx,horizon=6,return_quantiles=False,use_symmetric_averaging=False,
                        make_positive=False,sort_quantiles=True,use_znorm=False,padding_mode='none')
                    sync('cuda');seconds.append(time.perf_counter()-start)
                    pred=np.asarray(issued.forecast,dtype=np.float32)
                    require(pred.shape==(6,) and np.isfinite(pred).all(),'Invalid issued forecast')
                    predictions.append(pred[:,None].copy())
                    # Scoring accesses original labels only after issuing the frozen prediction.
                    targets.append(data.target[t+1:t+7].copy());masks.append(data.valid[t+1:t+7].copy())
                    persistence.append(np.full((6,1),ctx[-1],dtype=np.float32))
                    records.append(dict(origin=t,input_first=t-lookback+1,input_last=t,target_first=t+1,
                        target_last=t+6,mature_at=t+6,updated=False,uses_text=False,uses_future_covariates=False))
                    if i%200==0:print('FOUNDATION_LEDGER',domain,phase,i+1,'/',len(origins),flush=True)
                p=np.asarray(predictions);y=np.asarray(targets);mask=np.asarray(masks)
                np.savez_compressed(output/'predictions.npz',origins=np.asarray(origins),predictions=p,targets=y,valid_mask=mask,
                    persistence=np.asarray(persistence),prediction_seconds=np.asarray(seconds))
                write_json(output/'contexts.json',records)
                score=metrics(p,y,mask);naive=metrics(np.asarray(persistence),y,mask)
                write_json(output/'metrics.json',dict(foundation=score,persistence=naive))
                write_json(output/'status.json',dict(status='completed'))
                rows += [dict(dataset=domain,phase=phase,method='TimesFM3_frozen_median',origins=len(origins),**score),
                    dict(dataset=domain,phase=phase,method='persistence',origins=len(origins),**naive)]
        require(versions==[p._version for p in model.model.parameters()],'Model parameters changed')
        require(all(not p.requires_grad and p.grad is None for p in model.model.parameters()),'Gradient state changed')
        pd.DataFrame(rows).to_csv(run_dir/'scores.csv',index=False)
        write_json(run_dir/'validation.json',dict(status='passed',code=code,streams=10,model_frozen=True,
            prediction_before_label_access=True,whole_window_stage_boundaries=True,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            input_preprocessing='same train-only scale and causal historical fill as A',
            limitations=['Previously inspected periods; exploratory','Foundation pretraining overlap not excluded',
                'Median point forecast, not conditional mean','No residual adaptation has been run by this command']))
        write_json(run_dir/'status.json',dict(status='completed'));print('FOUNDATION_LEDGER_COMPLETED',flush=True)
    except Exception:
        write_json(run_dir/'status.json',dict(status='failed',error=traceback.format_exc()));raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--weights',type=Path,default=WEIGHTS)
    a=parser.parse_args();run(a.run_dir,a.weights)
