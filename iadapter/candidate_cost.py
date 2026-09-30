"""18 isolated microbenchmark rounds; no formal prediction/weight file edits."""
import json,time,gc
from pathlib import Path
from dataclasses import replace
import numpy as np
import pandas as pd
import torch
from .config import ROOT
from .replan_audit import OUTPUT
from .revision_candidate import initialize_five,five_data,key
from .feedback_factorial import load_bone
from .external_online import load_start
from .shared_online import InternalBackend,Record
from .efficiency import measure
from .util import write_json,revision,seed_all
from .audit import require

OUT=ROOT/'profile_output/e003_20261001'
def run(base=OUTPUT,out=OUT):
    base=Path(base);out=Path(out);out.mkdir(parents=True,exist_ok=False);require(not revision()['dirty'],'Commit profiler');torch.set_num_threads(1);rows=[]
    for d in ['Energy','SocialGood']:
        for method in ['raw_BERT_PCA5','LLM5','numeric_online']:
            for repeat in range(3):
                seed_all(2023+repeat)
                if method=='numeric_online':
                    c,data,backend,source,previous,initial,frozen=load_start(dict(dataset=d,seed=2023),'numeric_online');model=backend.model;ad=backend.adapter
                else:
                    c,data,model,checkpoint,bounds=load_bone(dict(dataset=d,seed=2023),with_text=True);representations,_=five_data(data,d,2023,base);data=replace(data,embeddings=representations[method]);ad=initialize_five(c,1);source=base/'exploratory_fusion'/key(dict(dataset=d,seed=2023,representation=method));ad.load_state_dict(torch.load(source/'warm_adapter.pt',map_location=c.device,weights_only=True));backend=InternalBackend(c,data,model,ad)
                before={n:p.detach().cpu().clone() for n,p in model.named_parameters()};t=data.origins('test',c.lookback,6)[0];prediction,context,ids,gate=backend.predict(t);record=Record(t,t+6,ids,context,prediction.detach().clone())
                y=torch.from_numpy(data.target[t+1:t+7]).unsqueeze(0).to(c.device);mask=torch.from_numpy(data.valid[t+1:t+7]).unsqueeze(0).to(c.device);opt=torch.optim.Adam(backend.parameters(),lr=c.lr,weight_decay=c.weight_decay)
                def predict():return backend.predict(t)
                def update():
                    opt.zero_grad(set_to_none=True);total,loss,feedback=backend.loss(record,y,mask);require(torch.isfinite(total).item(),'Nonfinite benchmark update');total.backward();torch.nn.utils.clip_grad_norm_(backend.parameters(),c.grad_clip,error_if_nonfinite=True);opt.step();backend.after_update(record,feedback)
                torch.cuda.reset_peak_memory_stats();pred_cost=measure(predict,c.device,50,200);update_cost=measure(update,c.device,50,200)
                require(all(p.grad is None and torch.equal(p.detach().cpu(),before[n]) for n,p in model.named_parameters()),'Frozen benchmark backbone changed')
                rows.append(dict(dataset=d,method=method,repeat=repeat+1,origin=t,trainable_parameters=sum(p.numel() for p in ad.parameters()),backbone_parameters=sum(p.numel() for p in model.parameters()),predict_median_ms=pred_cost['median_ms'],predict_p95_ms=pred_cost['p95_ms'],update_median_ms=update_cost['median_ms'],update_p95_ms=update_cost['p95_ms'],peak_allocated_bytes=torch.cuda.max_memory_allocated(),text_array_cache_bytes=data.embeddings.nbytes if c.uses_text else 0,gradient_cache_bytes=backend.hg.numel()*backend.hg.element_size()+backend.ge.numel()*backend.ge.element_size(),hardware=torch.cuda.get_device_name(),samples=200,warmup=50,code=revision(),source=str(source)))
                write_json(out/'rounds.json',rows);print('COST_ROUND',d,method,repeat+1,pred_cost['median_ms'],update_cost['median_ms'],flush=True);del backend,model,ad,opt;gc.collect();torch.cuda.empty_cache()
    frame=pd.DataFrame(rows);frame.drop(columns=['code']).to_csv(out/'rounds.csv',index=False);summary=frame.groupby(['dataset','method']).agg(rounds=('repeat','count'),parameters=('trainable_parameters','first'),backbone_parameters=('backbone_parameters','first'),prediction_median_ms=('predict_median_ms','median'),prediction_p95_ms=('predict_p95_ms','median'),update_median_ms=('update_median_ms','median'),update_p95_ms=('update_p95_ms','median'),peak_allocated_bytes_max=('peak_allocated_bytes','max'),text_array_cache_bytes=('text_array_cache_bytes','first')).reset_index();summary.to_csv(out/'summary.csv',index=False)
    write_json(out/'validation.json',dict(status='passed',rounds=18,formal_state_files_not_written=True,frozen_backbones_unchanged=True,scope='Fresh warmed copy per round; cached first real development-origin prediction; repeated fixed mature-record optimizer updates with feedback computation/refresh; synchronization included; excludes queue/logging/input loading and encoder/generation. Shared laptop GPU; this is not deployment throughput.',preprocessing=json.loads((base/'b3/generation_status.json').read_text()),cumulative_generation_budget=json.loads((base/'b3/budget.json').read_text())))
    print(summary.to_string(index=False),flush=True)
if __name__=='__main__':run()
