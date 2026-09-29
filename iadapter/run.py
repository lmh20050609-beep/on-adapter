import argparse,copy,json,time,traceback,gc
from pathlib import Path
from dataclasses import fields
import numpy as np
import torch
from .config import Config,ROOT,DOMAINS,VARIANTS
from .data import load_data,change_scores
from .models import Backbone,Adaptation
from .engine import Stream
from .util import seed_all,write_json,revision,masked_loss,metrics,sync

def backbone_config(c,data):
    return dict(dataset=c.dataset,lookback=c.lookback,horizon=c.horizon,seed=c.seed,d_model=c.d_model,e_layers=c.e_layers,d_ff=128,heads=4,dropout=c.dropout,pretrain_epochs=c.pretrain_epochs,batch=c.pretrain_batch,patience=c.patience,lr=c.lr,columns=data.columns,train_end=data.train_end,val_end=data.val_end,protocol=c.protocol)

def fit_backbone(c,data,checkpoint_dir=None):
    expected=backbone_config(c,data)
    directory=Path(checkpoint_dir) if checkpoint_dir is not None else ROOT/'checkpoints'; directory.mkdir(parents=True,exist_ok=True)
    path=directory/f'{c.dataset}_h{c.horizon}_s{c.seed}.pt'
    seed_all(c.seed)
    model=Backbone(c.lookback,c.horizon,len(data.columns),d_model=c.d_model,e_layers=c.e_layers,dropout=c.dropout).to(c.device)
    if path.exists():
        stored=torch.load(path,map_location=c.device,weights_only=True)
        if stored['config']!=expected: raise ValueError('Checkpoint configuration mismatch')
        model.load_state_dict(stored['model']); return model.eval(),str(path),stored['training']
    def batches(split,shuffle):
        origins=np.asarray(data.origins(split,c.lookback,c.horizon))
        if shuffle: np.random.shuffle(origins)
        for start in range(0,len(origins),c.pretrain_batch):
            windows=[data.window(int(s),c.lookback,c.horizon) for s in origins[start:start+c.pretrain_batch]]
            yield tuple(torch.from_numpy(np.stack([w[j] for w in windows])).to(c.device) for j in range(3))
    optimizer=torch.optim.Adam(model.parameters(),lr=c.lr)
    best=float('inf'); best_state=None; bad=0; history=[]; start=time.perf_counter()
    for epoch in range(c.pretrain_epochs):
        model.train()
        for x,y,mask in batches('train',True):
            if not mask.any(): continue
            optimizer.zero_grad(set_to_none=True); loss=masked_loss(model(x),y,mask)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite backbone training')
            loss.backward(); optimizer.step()
        model.eval(); total=0.; count=0
        with torch.no_grad():
            for x,y,mask in batches('val',False):
                pred=model(x); total+=float(((pred-y).square()*mask).sum()); count+=int(mask.sum())
        score=total/count
        history.append({'epoch':epoch+1,'val_mse':score})
        if score<best:
            best=score; bad=0; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else: bad+=1
        if bad>=c.patience: break
    if best_state is None: raise RuntimeError('No backbone checkpoint selected')
    training={'history':history,'best_val_mse':best,'seconds':time.perf_counter()-start}
    model.load_state_dict(best_state)
    temporary=path.with_suffix('.tmp'); torch.save({'model':best_state,'config':expected,'training':training,'source_revision':revision()},temporary); temporary.replace(path)
    print(f'BACKBONE {c.dataset} h={c.horizon} seed={c.seed} epochs={len(history)}',flush=True)
    return model.eval(),str(path),training

def run_one(c,run_dir):
    run_dir=Path(run_dir); run_dir.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    write_json(run_dir/'status.json',{'status':'running','config':c.to_dict()})
    started=time.perf_counter()
    try:
        data=load_data(c.dataset,with_embeddings=c.uses_text)
        code=revision()
        write_json(run_dir/'config.json',dict(c.to_dict(),data=data.metadata(),code=code))
        model,checkpoint,training=fit_backbone(c,data)
        model.zero_grad(set_to_none=True)
        frozen={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        seed_all(c.seed+10000)
        adapter=Adaptation(c,len(data.columns)).to(c.device)
        initial={k:v.detach().cpu().clone() for k,v in adapter.state_dict().items()}
        if c.device.startswith('cuda'): torch.cuda.reset_peak_memory_stats()
        warm_arrays,warm_events,warm_stats=Stream(c,data,model,adapter).run('train',trace=False)
        del warm_arrays
        torch.save(adapter.state_dict(),run_dir/'warm_adapter.pt')
        seed_all(c.seed+20000)
        arrays,events,stats=Stream(c,data,model,adapter).run('test')
        for k,v in model.state_dict().items():
            if not torch.equal(v.detach().cpu(),frozen[k]): raise AssertionError(f'Backbone changed: {k}')
        assert any(not torch.equal(v.detach().cpu(),initial[k]) for k,v in adapter.state_dict().items()),'Adapter never changed'
        recomputed=metrics(arrays['predictions'],arrays['targets'],arrays['valid_mask'])
        assert recomputed['mse']==stats['mse'] and recomputed['mae']==stats['mae']
        scores,thresholds=change_scores(data,c.lookback)
        arrays['state_change_scores']=scores[arrays['origins']]
        arrays['state_change_group']=np.digitize(arrays['state_change_scores'],thresholds)
        arrays['text_available']=np.asarray([bool(data.texts[t]) if c.uses_text else False for t in arrays['origins']])
        arrays['input_missing_mask']=np.stack([data.input_missing[t-c.lookback+1:t+1] for t in arrays['origins']])
        np.savez_compressed(run_dir/'predictions.npz',**arrays)
        with (run_dir/'updates.jsonl').open('w') as f:
            for event in events: f.write(json.dumps(event)+'\n')
        with (run_dir/'warm_updates.jsonl').open('w') as f:
            for event in warm_events: f.write(json.dumps(event)+'\n')
        write_json(run_dir/'prediction_context.json',{'origin_ids':arrays['origins'].tolist(),'text_ids':[data.text_ids[t] if c.uses_text else [] for t in arrays['origins']],'dates':[data.dates[t] for t in arrays['origins']],'group_thresholds_validation':thresholds})
        torch.save(adapter.state_dict(),run_dir/'final_adapter.pt')
        stats.update(config=c.to_dict(),code=code,backbone_checkpoint=checkpoint,backbone_training=training,warmup=warm_stats,seconds=time.perf_counter()-started,backbone_parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in adapter.parameters()),peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if c.device.startswith('cuda') else 0,expert_matching=getattr(adapter.adapter,'expert_matching',{}))
        write_json(run_dir/'metrics.json',stats)
        write_json(run_dir/'status.json',{'status':'completed','seconds':stats['seconds']})
        print(f'COMPLETED {c.dataset} h={c.horizon} seed={c.seed} {c.variant} mse={stats["mse"]:.6f} seconds={stats["seconds"]:.1f}',flush=True)
        del model,adapter; gc.collect()
        if c.device.startswith('cuda'): torch.cuda.empty_cache()
        return stats
    except Exception:
        write_json(run_dir/'status.json',{'status':'failed','traceback':traceback.format_exc(),'seconds':time.perf_counter()-started})
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',choices=DOMAINS,required=True); p.add_argument('--horizon',type=int,required=True); p.add_argument('--seed',type=int,required=True)
    p.add_argument('--variant',choices=VARIANTS,required=True); p.add_argument('--run_dir',required=True); p.add_argument('--device',default='cuda')
    a=p.parse_args(); c=Config(a.dataset,a.horizon,a.seed,a.variant,device=a.device); run_one(c,a.run_dir)
