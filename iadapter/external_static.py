"""Fixed common-OT static pilot; independent of the original online matrices."""
import argparse,gc,json,time,traceback
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from transformers import BertModel,BertTokenizerFast
from .config import ROOT,Config
from .review_sprint import development_data,initialize_adapter,write_events
from .run import fit_backbone
from .models import compute_offset_features
from .engine import Stream
from .native_gpt4mts import build as native_build,native_config,COMMIT as NATIVE_COMMIT,WEIGHTS
from .prepare import weight_directory
from .util import seed_all,write_json,revision,metrics,masked_loss

OUTPUT=ROOT/'experiments/external_static_pilot_20260929'
METHODS=['numeric_static','iadapter_static','gpt4mts_static']

def ot_data(dataset='Economy',with_embeddings=False):
    d,bounds=development_data(dataset,with_embeddings)
    if 'OT' not in d.columns:raise ValueError('Common target OT is missing')
    k=d.columns.index('OT');sl=slice(k,k+1)
    return replace(d,x=d.x[:,sl].copy(),target=d.target[:,sl].copy(),valid=d.valid[:,sl].copy(),input_missing=d.input_missing[:,sl].copy(),columns=['OT'],mean=d.mean[sl].copy(),scale=d.scale[sl].copy()),bounds

def pilot_config(variant='full',device='cuda',dataset='Economy',seed=2023):
    return Config(dataset,6,seed,variant,protocol='development_60_10_10_static_ot_v1',pretrain_batch=16,device=device)

def encode_cls(data,base,device='cuda'):
    path=base/'gpt4mts_cls512.npz';meta=base/'gpt4mts_cls512.json'
    if path.exists():
        with np.load(path,allow_pickle=False) as z:
            assert z['texts'].tolist()==data.texts and z['embeddings'].shape==(len(data.x),768)
            return z['embeddings'].copy()
    start=time.perf_counter();model=BertModel.from_pretrained(weight_directory(),local_files_only=True).to(device).eval();model.requires_grad_(False)
    tokenizer=BertTokenizerFast.from_pretrained(weight_directory(),local_files_only=True)
    assert len(model.encoder.layer)==12;result=[]
    with torch.no_grad():
        for begin in range(0,len(data.texts),4):
            batch=tokenizer(data.texts[begin:begin+4],padding=True,truncation=True,max_length=512,return_tensors='pt').to(device)
            result.append(model(**batch).last_hidden_state[:,0,:].cpu().numpy())
    result=np.concatenate(result).astype(np.float32);assert np.isfinite(result).all()
    np.savez_compressed(path,embeddings=result,texts=np.asarray(data.texts))
    write_json(meta,dict(pooling='CLS',max_length=512,layers=12,encoder_batch=4,eval=True,frozen=True,rows=len(result),seconds=time.perf_counter()-start,code=revision()))
    del model,tokenizer;gc.collect();torch.cuda.empty_cache();return result

@torch.no_grad()
def evaluate(data,c,model,adapter=None,cls_embeddings=None):
    model.eval()
    if adapter is not None:adapter.eval()
    origins=data.origins('test',c.lookback,c.horizon);predictions=[];contexts=[]
    hg=torch.zeros(1,c.d_model,device=c.device);ge=torch.zeros(c.text_dim,device=c.device)
    for t in origins:
        x=torch.from_numpy(data.x[t-c.lookback+1:t+1]).unsqueeze(0).to(c.device)
        if cls_embeddings is not None:
            summary=torch.from_numpy(cls_embeddings[t-c.lookback+1:t+1]).unsqueeze(0).to(c.device)
            prediction=model(x,0,summary)
            selected=[rid for ids in data.text_ids[t-c.lookback+1:t+1] for rid in ids]
        else:
            feature,mu,sd=model.context(x)
            pooled=torch.from_numpy(data.embeddings[t:t+1]).to(c.device) if c.uses_text else None
            e=adapter.text(pooled,bool(data.texts[t]) if c.uses_text else False,c.device)
            prediction=model.head(feature+adapter(feature,hg,e,ge,compute_offset_features(x)),mu,sd)
            selected=data.text_ids[t] if c.uses_text else []
        assert prediction.shape==(1,c.horizon,1) and torch.isfinite(prediction).all()
        assert all(int(rid.split(':')[0])<=t for rid in selected)
        predictions.append(prediction[0].cpu().numpy())
        contexts.append(dict(origin=t,text_ids=selected,numeric_feedback_origin=-1,text_feedback_origin=-1,updated_after_prediction=False))
    targets=np.stack([data.target[t+1:t+c.horizon+1] for t in origins]);mask=np.stack([data.valid[t+1:t+c.horizon+1] for t in origins])
    return dict(origins=np.asarray(origins),predictions=np.stack(predictions),targets=targets,valid_mask=mask),contexts

def train_native(data,c,embeddings,directory):
    seed_all(c.seed);model=native_build(c.lookback,c.horizon,c.device)
    frozen={n:p.detach().cpu().clone() for n,p in model.named_parameters() if not p.requires_grad}
    trainable={n:p for n,p in model.named_parameters() if p.requires_grad}
    optimizer=torch.optim.Adam(list(trainable.values()),lr=c.lr,weight_decay=0.)
    best=float('inf');best_state=None;best_epoch=None;bad=0;history=[];steps=0;start=time.perf_counter()
    def batches(split,shuffle):
        origins=np.asarray(data.origins(split,c.lookback,c.horizon))
        if shuffle:np.random.shuffle(origins)
        for begin in range(0,len(origins),c.pretrain_batch):
            ids=origins[begin:begin+c.pretrain_batch]
            windows=[data.window(int(t),c.lookback,c.horizon) for t in ids]
            x,y,mask=[torch.from_numpy(np.stack([w[j] for w in windows])).to(c.device) for j in range(3)]
            text=torch.from_numpy(np.stack([embeddings[t-c.lookback+1:t+1] for t in ids])).to(c.device)
            yield x,y,mask,text
    for epoch in range(c.pretrain_epochs):
        model.train();total_train=0.;train_count=0
        for x,y,mask,text in batches('train',True):
            if not mask.any():continue
            optimizer.zero_grad(set_to_none=True);prediction=model(x,0,text);loss=masked_loss(prediction,y,mask)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite native training loss')
            loss.backward()
            if not all(torch.isfinite(p.grad).all() for p in trainable.values() if p.grad is not None):raise FloatingPointError('Nonfinite native gradient')
            optimizer.step();steps+=1;count=int(mask.sum());total_train+=float(loss.detach())*count;train_count+=count
        model.eval();total=0.;count=0
        with torch.no_grad():
            for x,y,mask,text in batches('val',False):
                prediction=model(x,0,text);total+=float(((prediction-y).square()*mask).sum());count+=int(mask.sum())
        score=total/count
        if not np.isfinite(score):raise FloatingPointError('Nonfinite native selection score')
        history.append(dict(epoch=epoch+1,train_mse=total_train/train_count,val_mse=score,steps=steps))
        write_json(directory/'training_history.json',history)
        print(f'NATIVE_EPOCH {epoch+1} selection_mse={score:.6f} steps={steps}',flush=True)
        if score<best:
            best=score;best_epoch=epoch+1;bad=0;best_state={n:p.detach().cpu().clone() for n,p in trainable.items()}
        else:bad+=1
        if bad>=c.patience:break
    assert best_state is not None
    assert all(p.grad is None and torch.equal(p.detach().cpu(),frozen[n]) for n,p in model.named_parameters() if not p.requires_grad)
    with torch.no_grad():
        for n,p in trainable.items():p.copy_(best_state[n].to(c.device))
    info=dict(history=history,best_epoch=best_epoch,best_val_mse=best,optimizer_steps=steps,seconds=time.perf_counter()-start,lr=c.lr,weight_decay=0.,lr_schedule='constant',grad_clip=None,selection='masked MSE on60-70 percent only',frozen_parameters_unchanged=True)
    checkpoint=directory/'selected_native.pt'
    torch.save(dict(trainable=best_state,training=info,runner_config=c.to_dict(),model_config=vars(native_config(c.lookback,c.horizon)),upstream_commit=NATIVE_COMMIT,weights_source=json.loads((WEIGHTS/'SOURCE.json').read_text()),code=revision()),checkpoint)
    model.zero_grad(set_to_none=True);del optimizer,frozen;gc.collect();torch.cuda.empty_cache()
    return model,info,checkpoint

def run_method(method,base,dataset='Economy',seed=2023,cache_dir=None):
    directory=base/'runs'/method;directory.mkdir(parents=True,exist_ok=False);write_json(directory/'status.json',{'status':'running'})
    start=time.perf_counter()
    try:
        c=pilot_config('numeric_add' if method=='numeric_static' else 'full',dataset=dataset,seed=seed)
        data,bounds=ot_data(dataset=dataset,with_embeddings=method=='iadapter_static')
        config=dict(dataset=c.dataset,horizon=c.horizon,lookback=c.lookback,seed=c.seed,protocol=c.protocol,device=c.device,method=method,data=data.metadata(),boundaries=bounds,code=revision(),evaluation='static weights; no development updates',feedback='none' if method=='gpt4mts_static' else 'reset to zero and never refreshed during evaluation',text_interface='none' if method=='numeric_static' else ('current selected text masked-mean64' if method=='iadapter_static' else 'historical sequence of CLS512'),model=vars(native_config(c.lookback,c.horizon)) if method=='gpt4mts_static' else c.to_dict(),training_budget=dict(max_epochs=c.pretrain_epochs,batch_size=c.pretrain_batch,lr=c.lr,patience=c.patience,backbone_weight_decay=0,backbone_grad_clip=None),text_max_length=None if method=='numeric_static' else (512 if method=='gpt4mts_static' else 64))
        write_json(directory/'config.json',config);adapter=None;cls=None;warm_stats=None
        if method=='gpt4mts_static':
            text_base=Path(cache_dir) if cache_dir is not None else base;text_base.mkdir(parents=True,exist_ok=True)
            cls=encode_cls(data,text_base,c.device);model,training,checkpoint=train_native(data,c,cls,directory)
        else:
            model,checkpoint,training=fit_backbone(c,data,base/'checkpoints');model.zero_grad(set_to_none=True)
            adapter=initialize_adapter(c,1);seed_all(c.seed+11000)
            _,events,warm_stats=Stream(c,data,model,adapter).run('train')
            write_events(directory/'warm_updates.jsonl',events);torch.save(adapter.state_dict(),directory/'warm_adapter.pt')
        objects=[model]+([adapter] if adapter is not None else [])
        before=[{n:v.detach().cpu().clone() for n,v in obj.state_dict().items()} for obj in objects]
        torch.cuda.reset_peak_memory_stats();eval_start=time.perf_counter()
        arrays,contexts=evaluate(data,c,model,adapter,cls);evaluation_seconds=time.perf_counter()-eval_start
        assert all(torch.equal(v.detach().cpu(),state[n]) for obj,state in zip(objects,before) for n,v in obj.state_dict().items()),'Static evaluation changed parameters/buffers'
        np.savez_compressed(directory/'predictions.npz',**arrays);write_json(directory/'contexts.json',contexts)
        stats=metrics(arrays['predictions'],arrays['targets'],arrays['valid_mask'])
        if adapter is not None:trainable=sum(p.numel() for p in adapter.parameters())
        else:trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
        stats.update(config=config,training=training,backbone_checkpoint=str(checkpoint),warmup=warm_stats,total_seconds=time.perf_counter()-start,static_evaluation_seconds=evaluation_seconds,static_evaluation_peak_allocated_bytes=torch.cuda.max_memory_allocated(),static_parameters_unchanged=True,development_updates=0,trainable_parameters_before_static_freeze=trainable,total_parameters=sum(p.numel() for obj in objects for p in obj.parameters()),prediction_count=len(arrays['origins']))
        write_json(directory/'metrics.json',stats);write_json(directory/'status.json',{'status':'completed'})
        print(f'STATIC_COMPLETE {method} mse={stats["mse"]:.6f} mae={stats["mae"]:.6f} seconds={stats["total_seconds"]:.1f}',flush=True)
        del model,adapter,objects,before;gc.collect();torch.cuda.empty_cache();return stats
    except Exception:
        write_json(directory/'status.json',{'status':'failed','error':traceback.format_exc()});raise

def main(base=OUTPUT,dataset='Economy',seed=2023,cache_dir=None):
    torch.set_num_threads(1);base=Path(base);base.mkdir(parents=True,exist_ok=True);code=revision()
    if code['dirty']:raise RuntimeError('Commit source before starting pilot')
    state_path=base/'queue_state.json'
    if state_path.exists():
        state=json.loads(state_path.read_text())
        if state['code']!=code or state['methods']!=METHODS:raise RuntimeError('Pilot source/config mismatch')
    else:state=dict(code=code,methods=METHODS,jobs={m:{'status':'pending'} for m in METHODS})
    for method in METHODS:
        entry=state['jobs'][method]
        if entry['status']=='completed':continue
        if entry['status']!='pending':raise RuntimeError('Unresolved prior cell '+method)
        entry.update(status='running',started_at=time.time());write_json(state_path,state)
        try:
            stats=run_method(method,base,dataset=dataset,seed=seed,cache_dir=cache_dir);entry.update(status='completed',mse=stats['mse'],mae=stats['mae'])
        except Exception:
            entry.update(status='failed',error=traceback.format_exc());write_json(state_path,state);raise
        write_json(state_path,state)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUTPUT);main(p.parse_args().output)
