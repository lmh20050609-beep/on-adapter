"""Isolated latency measurements; never changes scientific run artifacts."""
import json,time,copy,gc
import numpy as np
import torch
from transformers import BertModel,BertTokenizerFast
from .config import ROOT,Config,MAIN,job_id
from .data import load_data
from .models import Backbone,Adaptation,compute_offset_features
from .util import write_json,masked_loss,sync,seed_all
from .prepare import weight_directory,contextual_pool
from .evidence_paths import resolve_evidence_path

def measure(call,device,warmup=50,repeats=200):
    for _ in range(warmup): call()
    times=[]
    for _ in range(repeats):
        sync(device); start=time.perf_counter(); call(); sync(device); times.append((time.perf_counter()-start)*1000)
    return {'median_ms':float(np.median(times)),'p95_ms':float(np.quantile(times,.95)),'samples':repeats,'warmup':warmup}

def main():
    torch.set_num_threads(1); seed_all(2023); results=[]
    for dataset in ['Economy','Energy']:
        for variant in MAIN:
            c=Config(dataset,6,2023,variant); directory=ROOT/'runs'/job_id(c.to_dict())
            assert json.loads((directory/'status.json').read_text())['status']=='completed'
            scientific=json.loads((directory/'metrics.json').read_text())
            data=load_data(dataset,with_embeddings=c.uses_text); N=len(data.columns)
            model=Backbone(c.lookback,6,N,d_model=32,e_layers=2).to(c.device).eval()
            checkpoint=torch.load(resolve_evidence_path(scientific['backbone_checkpoint']),map_location=c.device,weights_only=True)
            model.load_state_dict(checkpoint['model'])
            for p in model.parameters(): p.requires_grad_(False)
            ad=Adaptation(c,N).to(c.device)
            ad.load_state_dict(torch.load(directory/'warm_adapter.pt',map_location=c.device,weights_only=True))
            s=data.origins('test',c.lookback,6)[0]
            x,y,m=data.window(s,c.lookback,6)
            x=torch.from_numpy(x).unsqueeze(0).to(c.device); y=torch.from_numpy(y).unsqueeze(0).to(c.device); m=torch.from_numpy(m).unsqueeze(0).to(c.device)
            pooled=torch.from_numpy(data.embeddings[s:s+1]).to(c.device) if c.uses_text else None
            available=bool(data.texts[s]) if c.uses_text else False
            hg=torch.zeros(N,32,device=c.device); ge=torch.zeros(64,device=c.device)
            def predict():
                ad.eval()
                with torch.no_grad():
                    f,mu,sd=model.context(x); e=ad.text(pooled,available,c.device)
                    return model.head(f+ad(f,hg,e,ge,compute_offset_features(x)),mu,sd)
            with torch.no_grad(): f,mu,sd=model.context(x); offset=compute_offset_features(x)
            opt=torch.optim.Adam(ad.parameters(),lr=c.lr,weight_decay=c.weight_decay)
            def update():
                ad.train(); opt.zero_grad(set_to_none=True)
                e=ad.text(pooled,available,c.device); r=ad(f,hg,e,ge,offset)
                loss=masked_loss(model.head(f+r,mu,sd),y,m)
                if c.text_feedback: torch.autograd.grad(loss,e,retain_graph=True)
                if c.numeric_feedback:
                    z=torch.zeros_like(f,requires_grad=True)
                    torch.autograd.grad(masked_loss(model.head(f+z,mu,sd),y,m),z)
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite benchmark update')
                (loss+ad.auxiliary).backward(); torch.nn.utils.clip_grad_norm_(ad.parameters(),c.grad_clip,error_if_nonfinite=True); opt.step()
            torch.cuda.reset_peak_memory_stats()
            pred_cost=measure(predict,c.device); update_cost=measure(update,c.device)
            row={'dataset':dataset,'variant':variant,'hardware':torch.cuda.get_device_name(),'torch_version':str(torch.__version__),'prediction_cached_text':pred_cost,'update_fixed_mature_record':update_cost,'adapter_peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'trainable_parameters':scientific['trainable_parameters'],'backbone_parameters':scientific['backbone_parameters'],'frozen_text_parameters':0,'gradient_cache_bytes':scientific['gradient_cache_bytes'],'queue_peak_bytes':scientific['queue_peak_bytes'],'text_cache_bytes':scientific['text_cache_bytes'],'scope':'fixed real test-origin microbenchmark; warmed adapter copy, repeated optimizer updates including task gradients; excludes queue/logging overhead; no scientific metrics computed; formal run unchanged; shared desktop GPU'}
            if c.uses_text:
                path=weight_directory(); bert=BertModel.from_pretrained(path,local_files_only=True).eval().to(c.device); tok=BertTokenizerFast.from_pretrained(path,local_files_only=True)
                for p in bert.parameters(): p.requires_grad_(False)
                row['frozen_text_parameters']=sum(p.numel() for p in bert.parameters())
                def encode(): return contextual_pool(bert,tok,[data.texts[s]],c.device)
                row['text_encode_singleton']=measure(encode,c.device)
                def end_to_end():
                    nonlocal pooled
                    pooled=torch.from_numpy(encode()).to(c.device)
                    return predict()
                torch.cuda.reset_peak_memory_stats()
                row['fresh_text_end_to_end_prediction']=measure(end_to_end,c.device)
                row['end_to_end_peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated()
                del bert,tok
            else: row['fresh_text_end_to_end_prediction']=pred_cost
            row['total_parameters']=row['trainable_parameters']+row['backbone_parameters']+row['frozen_text_parameters']
            results.append(row); write_json(ROOT/'reports/efficiency.json',results)
            print(f'EFFICIENCY {dataset} {variant}: pred={pred_cost["median_ms"]:.3f}ms update={update_cost["median_ms"]:.3f}ms',flush=True)
            del model,ad,opt; gc.collect(); torch.cuda.empty_cache()

if __name__=='__main__': main()
