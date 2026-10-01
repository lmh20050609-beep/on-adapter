"""Bounded local TimesFM-3 integration; no fit or benchmark-ranking claim."""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from .config import ROOT
from .calibration_pilot import full_ot,region_origins
from .util import revision,write_json,sync,metrics
from .audit import require

SOURCE_REVISION='e51928e27119cb17bebc005be2696b75e0a9e688'
MODEL_REVISION='43046b85ec22d584a13f8098c2ed39c889e129c2'
WEIGHTS=ROOT/'weights/timesfm-3.0-pytorch'
SOURCE=ROOT/'external/timesfm'

def historical_context(data,t,lookback):
    require(t-lookback+1>=0,'Short history')
    x=np.array(data.x[t-lookback+1:t+1,0],dtype=np.float32,copy=True)
    require(np.isfinite(x).all(),'Nonfinite causal input')
    return x

def smoke(run_dir,weights=WEIGHTS):
    run_dir=Path(run_dir);run_dir.mkdir(parents=True,exist_ok=False)
    write_json(run_dir/'status.json',dict(status='running',scope='integration only; no training or formal ranking'))
    try:
        require(not revision()['dirty'],'Commit wrapper/plan before model use')
        receipt=json.loads((weights/'DOWNLOAD_RECEIPT.json').read_text())
        require(receipt['revision']==MODEL_REVISION and receipt['status']=='verified_transport','Incomplete weights')
        source=json.loads((SOURCE/'SOURCE.json').read_text())
        require(source['revision']==SOURCE_REVISION,'Source revision changed')
        require(torch.cuda.is_available(),'CUDA unavailable')
        free,total=torch.cuda.mem_get_info()
        require(free>2.5*2**30,'Insufficient currently free GPU memory for bounded fp32 smoke; wait, do not terminate other processes')
        torch.set_num_threads(1);torch.manual_seed(2023)
        torch.set_float32_matmul_precision('highest')
        sys.path.insert(0,str(SOURCE/'src'))
        from timesfm3 import TimesFM3Forecaster,ModelConfig
        cfg=ModelConfig(checkpoint_path=str(weights),device='cuda',per_core_batch_size=1,local_files_only=True)
        torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        model=TimesFM3Forecaster(cfg);model.model.requires_grad_(False);model.model.eval();sync('cuda')
        load_seconds=time.perf_counter()-start
        parameters=sum(p.numel() for p in model.model.parameters())
        require(parameters==330710976,'Unexpected official model parameter count')
        require(len(model.config.quantiles)==9 and model.config.median_quantile_index==4,'Wrong point output quantile')
        versions=[p._version for p in model.model.parameters()]
        records=[];all_pred=[];all_target=[];all_mask=[];all_context=[];max_batch_delta=0.
        for domain,lookback in [('Energy',48),('SocialGood',36)]:
            data=full_ot(domain)
            origins=region_origins(len(data.x),lookback,6,'calibration')[:2].tolist()
            contexts=[historical_context(data,t,lookback) for t in origins]
            prediction=[]
            for t,ctx in zip(origins,contexts):
                # Label/covariate futures are never passed. Preprocessing is already historical-only.
                altered=data.x.copy();altered[t+1:]=999999.
                require(np.array_equal(ctx,altered[t-lookback+1:t+1,0]),'Future values entered input')
                sync('cuda');start=time.perf_counter()
                out=model.predict(ctx,horizon=6,return_quantiles=True,use_symmetric_averaging=False,
                    make_positive=False,sort_quantiles=True,use_znorm=False,padding_mode='none')
                sync('cuda');seconds=time.perf_counter()-start
                pred=np.asarray(out.forecast,dtype=np.float32)
                quant=np.asarray(out.quantiles,dtype=np.float32)
                require(pred.shape==(6,) and quant.shape==(6,9),'Output shape differs')
                require(np.isfinite(pred).all() and np.isfinite(quant).all(),'Nonfinite pretrained output')
                require(np.array_equal(pred,quant[:,4]),'Point forecast is not median')
                prediction.append(pred)
                target=data.target[t+1:t+7].copy();mask=data.valid[t+1:t+7].copy()
                records.append(dict(dataset=domain,origin=t,lookback=lookback,input_first=t-lookback+1,input_last=t,
                    target_first=t+1,target_last=t+6,prediction_seconds=seconds,available_text_ids=data.text_ids[t],
                    uses_text=False,uses_future_covariates=False,phase='calibration',fit=False))
                all_pred.append(pred[:,None]);all_target.append(target);all_mask.append(mask);all_context.append(ctx)
            # Verify singleton inference agrees with a real two-item batch, then restore config.
            model.config.per_core_batch_size=2
            batched=list(model.predict_batch(contexts,horizon=6,return_quantiles=True,
                use_symmetric_averaging=False,make_positive=False,sort_quantiles=True,use_znorm=False,padding_mode='none'))
            model.config.per_core_batch_size=1
            require(len(batched)==len(contexts),'Batch output count differs')
            for out in batched:
                point=np.asarray(out.forecast);quantiles=np.asarray(out.quantiles)
                require(point.shape==(6,) and quantiles.shape==(6,9),'Batch output shape differs')
                require(np.isfinite(point).all() and np.isfinite(quantiles).all(),'Nonfinite batch output')
                require(np.array_equal(point,quantiles[:,4]),'Batch point forecast is not median')
            difference=float(np.max(np.abs(np.stack(prediction)-np.stack([o.forecast for o in batched]))))
            max_batch_delta=max(max_batch_delta,difference)
            require(np.allclose(np.stack(prediction),np.stack([o.forecast for o in batched]),atol=1e-5,rtol=1e-5),'Batch inference differs')
        require(versions==[p._version for p in model.model.parameters()],'Model parameters mutated')
        require(all(not p.requires_grad and p.grad is None for p in model.model.parameters()),'Model gradient state changed')
        pred=np.asarray(all_pred);target=np.asarray(all_target);mask=np.asarray(all_mask)
        np.savez_compressed(run_dir/'predictions.npz',predictions=pred,targets=target,valid_mask=mask,
            origins=np.asarray([r['origin'] for r in records]),datasets=np.asarray([r['dataset'] for r in records]))
        write_json(run_dir/'contexts.json',records)
        write_json(run_dir/'validation.json',dict(status='passed',source_revision=SOURCE_REVISION,model_revision=MODEL_REVISION,
            code=revision(),model_parameters=parameters,model_frozen=True,optimizer_created=False,
            singleton_origins=4,batch_comparison_max_abs=max_batch_delta,history_only_contexts=True,
            masked_ground_truth_diagnostic=metrics(pred,target,mask),load_seconds=load_seconds,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),free_bytes_before=free,total_gpu_bytes=total,
            scope='Four calibration-origin integration forecasts, not a performance comparison or confirmed stronger baseline',
            limitations=['Foundation pretraining overlap with benchmark domains not independently excluded',
                'Official numerical path only; output residual adapter is a future separate experiment',
                'end_date proxy timeline retained; model weights transported via mirror as disclosed in receipt']))
        write_json(run_dir/'status.json',dict(status='completed'))
        print('TIMESFM3_SMOKE_PASSED',parameters,'parameters; peak',torch.cuda.max_memory_allocated(),flush=True)
    except Exception:
        write_json(run_dir/'status.json',dict(status='failed',error=traceback.format_exc()));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--weights',type=Path,default=WEIGHTS)
    a=p.parse_args();smoke(a.run_dir,a.weights)
