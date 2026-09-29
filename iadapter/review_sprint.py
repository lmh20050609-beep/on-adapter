"""Predeclared development-only B/C1 experiments, isolated from the 190-run campaign."""
import argparse,copy,gc,json,time,traceback,zlib
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from transformers import BertTokenizerFast
from .config import ROOT,Config,job_id
from .data import load_data
from .models import Adaptation
from .legacy_core import IdentityExpert
from .engine import Stream
from .run import fit_backbone
from .prepare import weight_directory
from .util import seed_all,write_json,revision,metrics

VARIANTS=['numeric_add','text_add','full','no_route_text_gradient','no_decisiveness','no_route_no_decisiveness','single_mlp']
CONDITIONS=['correct','train_mean','historical_donor','zero_current_text']
DEFAULT_RUN=ROOT/'experiments/review_sprint_20260929'

def development_data(dataset,uses_text):
    raw=load_data(dataset,with_embeddings=uses_text)
    return truncate_development(raw)

def truncate_development(raw):
    total=len(raw.x); train_end=int(.6*total); select_end=int(.7*total); dev_end=int(.8*total)
    assert raw.train_end==train_end
    data=replace(raw,x=raw.x[:dev_end],target=raw.target[:dev_end],valid=raw.valid[:dev_end],input_missing=raw.input_missing[:dev_end],texts=raw.texts[:dev_end],text_ids=raw.text_ids[:dev_end],dates=raw.dates[:dev_end],val_end=select_end,embeddings=raw.embeddings[:dev_end] if raw.embeddings is not None else None)
    return data,{'raw_rows':total,'train_end':train_end,'backbone_selection_end':select_end,'development_end':dev_end,'scope':'exploratory development; old test not evaluated'}

def initialize_adapter(c,N):
    seed_all(c.seed+10000);ad=Adaptation(c,N)
    # Name-specific RNG avoids shifting all shared initial weights when a module is removed.
    for name,module in ad.named_modules():
        if hasattr(module,'reset_parameters') and not list(module.children()):
            module_seed=(c.seed+10000+zlib.crc32(name.encode()))%(2**31)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(module_seed);module.reset_parameters()
    for module in ad.modules():
        if isinstance(module,IdentityExpert):
            torch.nn.init.zeros_(module.out.weight);torch.nn.init.zeros_(module.out.bias)
    return ad.to(c.device)

class DevelopmentObserver:
    def __init__(self,data,seed,content_probe=False,token_lengths=None):
        self.data=data;self.content_probe=content_probe;self.records=[];self.probes=[]
        self.rng=np.random.default_rng(seed+30000)
        if content_probe:
            if token_lengths is None:
                tok=BertTokenizerFast.from_pretrained(weight_directory(),local_files_only=True)
                token_lengths=[len(ids) for ids in tok(data.texts,truncation=True,max_length=64)['input_ids']]
            self.lengths=np.asarray(token_lengths,dtype=int)
            valid=np.asarray([bool(t) for t in data.texts])
            self.available=valid
            self.mean=data.embeddings[:data.train_end][valid[:data.train_end]].mean(0).astype(np.float32)
            if not np.isfinite(self.mean).all():raise ValueError('No finite training-only mean text representation')
    @torch.no_grad()
    def __call__(self,stream,rec,pred,residual):
        self.records.append({'origin':rec.origin,'text_ids':rec.text_ids,'available':rec.available,'numeric_feedback_norm':float(rec.hg.norm()),'text_feedback_norm':float(rec.ge.norm()),'residual_to_feature_norm':float(residual.norm()/(rec.feature.norm()+1e-8))})
        if not self.content_probe:return
        t=rec.origin;eligible=np.flatnonzero(self.available[:t]&(np.abs(self.lengths[:t]-self.lengths[t])<=8))
        donor=int(self.rng.choice(eligible)) if rec.available and len(eligible) else -1
        ad=stream.adaptation;gate=ad.adapter.last_gate;aux=ad.adapter.aux_loss
        outputs=[pred[0].cpu().numpy().copy()]
        try:
            for condition in CONDITIONS[1:]:
                if condition=='historical_donor' and donor<0:
                    outputs.append(np.full_like(outputs[0],np.nan));continue
                pooled=self.mean if condition=='train_mean' else (self.data.embeddings[donor] if condition=='historical_donor' else np.zeros(768,dtype=np.float32))
                available=rec.available if condition!='zero_current_text' else False
                e=ad.text(torch.from_numpy(pooled).to(stream.device).unsqueeze(0),available,stream.device)
                z=ad(rec.feature,rec.hg,e,rec.ge,rec.offset)
                outputs.append(stream.backbone.head(rec.feature+z,rec.mu,rec.sigma)[0].cpu().numpy().copy())
        finally:
            ad.adapter.last_gate=gate;ad.adapter.aux_loss=aux
        self.probes.append({'origin':t,'donor':donor,'donor_text_ids':self.data.text_ids[donor] if donor>=0 else [],'reference_tokens':int(self.lengths[t]),'donor_tokens':int(self.lengths[donor]) if donor>=0 else None,'predictions':np.stack(outputs)})

def write_events(path,events):
    path.write_text(''.join(json.dumps(e)+'\n' for e in events))

def run_cell(job,base):
    c=Config(job['dataset'],6,job['seed'],job['variant'],protocol='development_60_10_10_v1')
    directory=base/'runs'/job_id(c.to_dict());directory.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();write_json(directory/'status.json',{'status':'running'})
    try:
        torch.set_num_threads(1);data,bounds=development_data(c.dataset,c.uses_text)
        model,checkpoint,training=fit_backbone(c,data,base/'checkpoints');model.zero_grad(set_to_none=True)
        frozen={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        ad=initialize_adapter(c,len(data.columns));initial={k:v.detach().cpu().clone() for k,v in ad.state_dict().items()}
        torch.save(initial,directory/'initial_adapter.pt')
        config=dict(c.to_dict(),data=data.metadata(),boundaries=bounds,code=revision(),initialization='module-name CRC32-derived seeds; identity output reset to zero')
        write_json(directory/'config.json',config)
        seed_all(c.seed+11000)
        _,warm_events,warm_stats=Stream(c,data,model,ad).run('train')
        torch.save(ad.state_dict(),directory/'warm_adapter.pt')
        observer=DevelopmentObserver(data,c.seed,c.variant in {'full','text_add'})
        seed_all(c.seed+20000)
        if c.device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
        arrays,events,stats=Stream(c,data,model,ad).run('test',observer=observer)
        for k,v in model.state_dict().items():
            if not torch.equal(v.detach().cpu(),frozen[k]):raise AssertionError('Backbone changed: '+k)
        assert any(not torch.equal(v.detach().cpu(),initial[k]) for k,v in ad.state_dict().items())
        np.savez_compressed(directory/'predictions.npz',**arrays)
        write_events(directory/'updates.jsonl',events);write_events(directory/'warm_updates.jsonl',warm_events)
        write_json(directory/'prediction_context.json',observer.records)
        if observer.content_probe:
            values=np.stack([p['predictions'] for p in observer.probes]);donors=np.asarray([p['donor'] for p in observer.probes])
            np.testing.assert_array_equal(values[:,0],arrays['predictions'])
            np.savez_compressed(directory/'text_interventions.npz',predictions=values,origins=arrays['origins'],donor_origins=donors,eligible=donors>=0,conditions=np.asarray(CONDITIONS))
            write_json(directory/'text_interventions.json',[{k:v for k,v in p.items() if k!='predictions'} for p in observer.probes])
        torch.save(ad.state_dict(),directory/'final_adapter.pt')
        stats.update(config=config,code=config['code'],boundaries=bounds,backbone_checkpoint=str(checkpoint),backbone_training=training,warmup=warm_stats,trainable_parameters=sum(p.numel() for p in ad.parameters()),seconds=time.perf_counter()-started,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if c.device.startswith('cuda') else 0)
        write_json(directory/'metrics.json',stats);write_json(directory/'status.json',{'status':'completed'})
        print(f'COMPLETED {directory.name} mse={stats["mse"]:.6f} mae={stats["mae"]:.6f} seconds={stats["seconds"]:.1f}',flush=True)
        del model,ad;gc.collect()
        if c.device.startswith('cuda'):torch.cuda.empty_cache()
        return stats
    except Exception:
        write_json(directory/'status.json',{'status':'failed','error':traceback.format_exc()});raise

def matrix(sanity=False):
    if sanity:return [dict(dataset='Economy',seed=2023,variant=v) for v in ['numeric_add','full']]
    return [dict(dataset=d,seed=s,variant=v) for d in ['Energy','SocialGood'] for s in [2023,2024,2025] for v in VARIANTS]

def campaign(base,sanity=False,limit=None):
    base=Path(base);base.mkdir(parents=True,exist_ok=True);jobs=matrix(sanity);path=base/'queue_state.json';code=revision()
    if code['dirty']:raise RuntimeError('Commit source before running development experiments')
    if path.exists():
        state=json.loads(path.read_text())
        if state['code']!=code or state['planned_jobs']!=jobs:raise RuntimeError('Queue code/config mismatch')
    else:
        state={'code':code,'planned_jobs':jobs,'jobs':{job_id(dict(j,horizon=6)):{'status':'pending'} for j in jobs},'started_at':time.time()}
        write_json(path,state)
    done_now=0
    for job in jobs:
        key=job_id(dict(job,horizon=6));entry=state['jobs'][key]
        if entry['status']=='completed':continue
        if entry['status']!='pending':raise RuntimeError(f'Unresolved {key}: {entry["status"]}')
        entry.update(status='running',started_at=time.time());write_json(path,state)
        try:
            result=run_cell(job,base);entry.update(status='completed',seconds=result['seconds'],mse=result['mse'],mae=result['mae'])
        except Exception:
            entry.update(status='failed',error=traceback.format_exc());write_json(path,state);raise
        write_json(path,state);done_now+=1
        if limit is not None and done_now>=limit:break
    print('QUEUE_STATUS',sum(v['status']=='completed' for v in state['jobs'].values()),'/',len(jobs),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run_dir',type=Path,default=DEFAULT_RUN);p.add_argument('--sanity',action='store_true');p.add_argument('--limit',type=int)
    a=p.parse_args();campaign(a.run_dir,a.sanity,a.limit)
