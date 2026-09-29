"""Isolated, repeated cost measurements for the reviewed development candidates."""
import argparse,gc,json,time,random
from dataclasses import fields
from pathlib import Path
import numpy as np
import torch
from transformers import BertModel,BertTokenizerFast
from .config import ROOT,Config
from .models import Backbone,Adaptation,compute_offset_features
from .review_sprint import DEFAULT_RUN,development_data
from .prepare import weight_directory,contextual_pool
from .util import seed_all,masked_loss,sync,write_json,revision

METHODS=['numeric_add','text_add','full','no_decisiveness','single_mlp']
DEFAULT_OUTPUT=ROOT/'profile_output/review_sprint_cost_20260929'

def timed(call,device,warmup,repeats):
    for _ in range(warmup):call()
    elapsed=[]
    for _ in range(repeats):
        sync(device);start=time.perf_counter();call();sync(device)
        elapsed.append((time.perf_counter()-start)*1000)
    return {'median_ms':float(np.median(elapsed)),'p95_ms':float(np.quantile(elapsed,.95)),'sample_ms':elapsed}

def evidence_stamp(base):
    return {str(p.relative_to(base)):(p.stat().st_size,p.stat().st_mtime_ns) for name in ['runs','checkpoints'] for p in (base/name).rglob('*') if p.is_file()}

def profile_case(base,dataset,variant,round_index,warmup,repeats):
    directory=base/'runs'/f'{dataset}_h6_s2023_{variant}'
    assert json.loads((directory/'status.json').read_text())['status']=='completed'
    scientific=json.loads((directory/'metrics.json').read_text())
    c=Config(**{f.name:scientific['config'][f.name] for f in fields(Config)})
    assert c.device=='cuda'
    data,bounds=development_data(dataset,c.uses_text);N=len(data.columns)
    seed_all(2023+round_index)
    model=Backbone(c.lookback,c.horizon,N,d_model=c.d_model,e_layers=c.e_layers,dropout=c.dropout).to(c.device).eval()
    checkpoint=torch.load(scientific['backbone_checkpoint'],map_location='cpu',weights_only=True)
    model.load_state_dict(checkpoint['model']);model.requires_grad_(False)
    ad=Adaptation(c,N).to(c.device)
    warmed=torch.load(directory/'warm_adapter.pt',map_location='cpu',weights_only=True);ad.load_state_dict(warmed)
    origin=data.origins('test',c.lookback,c.horizon)[0]
    x,y,mask=data.window(origin,c.lookback,c.horizon)
    x=torch.from_numpy(x).unsqueeze(0).to(c.device);y=torch.from_numpy(y).unsqueeze(0).to(c.device);mask=torch.from_numpy(mask).unsqueeze(0).to(c.device)
    assert mask.any()
    pooled=torch.from_numpy(data.embeddings[origin:origin+1]).to(c.device) if c.uses_text else None
    available=bool(data.texts[origin]) if c.uses_text else False
    hg=torch.zeros(N,c.d_model,device=c.device);ge=torch.zeros(c.text_dim,device=c.device)
    recorded_hg=hg.clone();recorded_ge=ge.clone()
    @torch.no_grad()
    def predict():
        ad.eval();f,mu,sd=model.context(x);e=ad.text(pooled,available,c.device)
        result=model.head(f+ad(f,hg,e,ge,compute_offset_features(x)),mu,sd)
        if not torch.isfinite(result).all():raise FloatingPointError('Nonfinite benchmark prediction')
        return result
    with torch.no_grad():f,mu,sd=model.context(x);offset=compute_offset_features(x)
    opt=torch.optim.Adam(ad.parameters(),lr=c.lr,weight_decay=c.weight_decay)
    def update():
        nonlocal hg,ge
        ad.train();opt.zero_grad(set_to_none=True)
        e=ad.text(pooled,available,c.device)
        loss=masked_loss(model.head(f+ad(f,recorded_hg,e,recorded_ge,offset),mu,sd),y,mask)
        new_ge=torch.autograd.grad(loss,e,retain_graph=True)[0].mean(0).detach() if c.text_feedback else None
        if c.numeric_feedback:
            z=torch.zeros_like(f,requires_grad=True)
            new_hg=torch.autograd.grad(masked_loss(model.head(f+z,mu,sd),y,mask),z)[0].mean(0).detach()
        else:new_hg=None
        total=loss+ad.auxiliary
        if not torch.isfinite(total):raise FloatingPointError('Nonfinite benchmark update')
        total.backward();torch.nn.utils.clip_grad_norm_(ad.parameters(),c.grad_clip,error_if_nonfinite=True);opt.step()
        if new_hg is not None:hg=c.momentum*hg+(1-c.momentum)*new_hg
        if new_ge is not None:ge=new_ge
    torch.cuda.reset_peak_memory_stats()
    prediction=timed(predict,c.device,warmup,repeats);prediction_peak=torch.cuda.max_memory_allocated()
    update_cost=timed(update,c.device,warmup,repeats)
    cached_peak=torch.cuda.max_memory_allocated()
    assert all(p.grad is None for p in model.parameters())
    for name,value in model.state_dict().items():assert torch.equal(value.cpu(),checkpoint['model'][name])
    # Every fresh-text prediction starts from the scientific warm state, not microbenchmark-trained weights.
    ad.load_state_dict(warmed);ad.zero_grad(set_to_none=True);hg.zero_();ge.zero_()
    del opt;gc.collect();torch.cuda.empty_cache()
    bert_params=0
    if c.uses_text:
        weights=weight_directory();bert=BertModel.from_pretrained(weights,local_files_only=True).eval().to(c.device)
        bert.requires_grad_(False);tok=BertTokenizerFast.from_pretrained(weights,local_files_only=True)
        assert len(bert.encoder.layer)==12
        bert_params=sum(p.numel() for p in bert.parameters())
        def fresh():
            nonlocal pooled
            encoded=contextual_pool(bert,tok,[data.texts[origin]],c.device)
            pooled=torch.from_numpy(encoded).to(c.device)
            return predict()
        torch.cuda.reset_peak_memory_stats();fresh_cost=timed(fresh,c.device,warmup,repeats);fresh_peak=torch.cuda.max_memory_allocated()
        del bert,tok
    else:fresh_cost=prediction;fresh_peak=prediction_peak
    result={'dataset':dataset,'variant':variant,'round':round_index,'seed_checkpoint':2023,'origin':origin,'mature_at':origin+c.horizon,'warmup':warmup,'repeats':repeats,'prediction_cached':prediction,'update_fixed_record':update_cost,'prediction_fresh_text':fresh_cost,'cached_path_peak_allocated_bytes':cached_peak,'fresh_path_peak_allocated_bytes':fresh_peak,'original_development_peak_allocated_bytes':scientific['peak_cuda_allocated_bytes'],'cpu_embedding_cache_bytes':data.embeddings.nbytes if c.uses_text else 0,'gradient_cache_bytes':scientific['gradient_cache_bytes'],'replay_queue_peak_bytes':scientific['queue_peak_bytes'],'trainable_parameters':sum(p.numel() for p in ad.parameters()),'backbone_parameters':sum(p.numel() for p in model.parameters()),'frozen_text_parameters':bert_params,'model_commit':scientific['code']['commit'],'boundaries':bounds}
    assert result['trainable_parameters']==scientific['trainable_parameters']
    del model,ad;gc.collect();torch.cuda.empty_cache()
    return result

def main(args):
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    base=Path(args.source);proof=json.loads((base/'reports/validation.json').read_text());assert proof['status']=='passed' and proof['completed']==42
    torch.set_num_threads(1)
    before=evidence_stamp(base);queue=(base/'queue_state.json').read_bytes();results=[]
    config={'source':str(base),'datasets':args.datasets.split(','),'variants':args.variants.split(','),'rounds':args.rounds,'warmup':args.warmup,'repeats':args.repeats,'hardware':torch.cuda.get_device_name(),'torch':str(torch.__version__),'profile_code':revision(),'model_code':proof['experiment_commit'],'scope':'isolated fixed development-origin microbenchmark; no quality metrics; copied warm state; queue/logging excluded; shared desktop GPU'}
    write_json(output/'config.json',config);write_json(output/'status.json',{'status':'running'})
    try:
        for round_index in range(args.rounds):
            order=[(d,v) for d in config['datasets'] for v in config['variants']]
            random.Random(4040+round_index).shuffle(order)
            for dataset,variant in order:
                row=profile_case(base,dataset,variant,round_index,args.warmup,args.repeats)
                results.append(row);write_json(output/'raw.json',results)
                print(f'PROFILE {len(results)} round={round_index} {dataset} {variant} pred={row["prediction_cached"]["median_ms"]:.3f}ms update={row["update_fixed_record"]["median_ms"]:.3f}ms fresh={row["prediction_fresh_text"]["median_ms"]:.3f}ms',flush=True)
        assert before==evidence_stamp(base) and queue==(base/'queue_state.json').read_bytes()
        write_json(output/'status.json',{'status':'completed','cases':len(results),'scientific_artifacts_unchanged':True,'verified_scientific_files':len(before),'backbone_unchanged':True})
    except Exception:
        import traceback
        write_json(output/'status.json',{'status':'failed','error':traceback.format_exc(),'completed_cases':len(results)});raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=DEFAULT_RUN);p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--datasets',default='Energy,SocialGood');p.add_argument('--variants',default=','.join(METHODS));p.add_argument('--rounds',type=int,default=3);p.add_argument('--warmup',type=int,default=50);p.add_argument('--repeats',type=int,default=200)
    main(p.parse_args())
