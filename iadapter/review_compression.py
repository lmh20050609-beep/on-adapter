"""Fixed E-01 projection-width study; original training modules stay unchanged."""
import argparse,gc,json,time,traceback,zlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .config import ROOT,Config,job_id
from .review_sprint import DEFAULT_RUN,development_data,initialize_adapter,DevelopmentObserver,write_events
from .review_stability import verify_training_unchanged
from .run import fit_backbone
from .engine import Stream
from .util import seed_all,revision,write_json

OUTPUT=ROOT/'experiments/review_compression_20260929'
METHODS=['no_route_no_decisiveness','single_mlp']

def jobs(sanity=False):
    return [dict(dataset=d,seed=s,variant=v,projection_hidden=8) for d in (['Economy'] if sanity else ['Energy','SocialGood']) for s in ([2023] if sanity else [2023,2024,2025]) for v in METHODS]

def configuration(job):
    return Config(job['dataset'],6,job['seed'],job['variant'],protocol='development_60_10_10_v1')

def cell_id(job):return job_id(dict(job,horizon=6))+'_p'+str(job['projection_hidden'])

def initialize_projection(c,N,width):
    if width not in (8,96):raise ValueError('Only preregistered widths 8 and 96 are allowed')
    ad=initialize_adapter(c,N)
    if width==96:return ad
    if c.variant not in METHODS:raise ValueError('Unsupported compression variant')
    for index,inputs,outputs in [(0,768,width),(2,width,c.text_dim)]:
        previous=ad.project[index]
        layer=nn.Linear(inputs,outputs)
        module_seed=(c.seed+10000+zlib.crc32(f'project.{index}'.encode()))%(2**31)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(module_seed);layer.reset_parameters()
        if layer.bias.shape==previous.bias.shape:
            with torch.no_grad():layer.bias.copy_(previous.bias.detach().cpu())
        ad.project[index]=layer.to(c.device)
    return ad

def source_base(sanity=False):return DEFAULT_RUN.with_name(DEFAULT_RUN.name+'_sanity') if sanity else DEFAULT_RUN

def run_cell(job,base,sanity=False):
    c=configuration(job);directory=base/'runs'/cell_id(job)
    directory.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    write_json(directory/'status.json',{'status':'running'})
    try:
        torch.set_num_threads(1);data,bounds=development_data(c.dataset,True)
        src=source_base(sanity);cp=src/'checkpoints'/f'{c.dataset}_h6_s{c.seed}.pt'
        if not cp.is_file():raise FileNotFoundError('Declared shared backbone missing: '+str(cp))
        model,checkpoint,training=fit_backbone(c,data,src/'checkpoints');model.zero_grad(set_to_none=True)
        frozen={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        ad=initialize_projection(c,len(data.columns),job['projection_hidden'])
        initial={k:v.detach().cpu().clone() for k,v in ad.state_dict().items()}
        torch.save(initial,directory/'initial_adapter.pt')
        config=dict(c.to_dict(),projection_hidden=job['projection_hidden'],data=data.metadata(),boundaries=bounds,code=revision(),initialization='module-name seeds; only project.0/project.2 dimensions changed',control_source=str(src/'runs'/job_id(c.to_dict())) if not sanity else None)
        write_json(directory/'config.json',config)
        seed_all(c.seed+11000)
        _,warm_events,warm_stats=Stream(c,data,model,ad).run('train')
        torch.save(ad.state_dict(),directory/'warm_adapter.pt')
        observer=DevelopmentObserver(data,c.seed,False);seed_all(c.seed+20000)
        if c.device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
        arrays,events,stats=Stream(c,data,model,ad).run('test',observer=observer)
        assert all(torch.equal(v.detach().cpu(),frozen[k]) for k,v in model.state_dict().items()),'Backbone changed'
        assert any(not torch.equal(v.detach().cpu(),initial[k]) for k,v in ad.state_dict().items()),'Adapter did not update'
        np.savez_compressed(directory/'predictions.npz',**arrays)
        write_events(directory/'updates.jsonl',events);write_events(directory/'warm_updates.jsonl',warm_events)
        write_json(directory/'prediction_context.json',observer.records);torch.save(ad.state_dict(),directory/'final_adapter.pt')
        stats.update(config=config,code=config['code'],boundaries=bounds,backbone_checkpoint=str(checkpoint),backbone_training=training,warmup=warm_stats,trainable_parameters=sum(p.numel() for p in ad.parameters()),projection_parameters=sum(p.numel() for p in ad.project.parameters()),seconds=time.perf_counter()-start,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if c.device.startswith('cuda') else 0,backbone_unchanged=True)
        write_json(directory/'metrics.json',stats);write_json(directory/'status.json',{'status':'completed'})
        print(f'COMPLETED {directory.name} mse={stats["mse"]:.6f} mae={stats["mae"]:.6f}',flush=True)
        del model,ad;gc.collect();torch.cuda.empty_cache()
        return stats
    except Exception:
        write_json(directory/'status.json',{'status':'failed','error':traceback.format_exc()});raise

def campaign(base=OUTPUT,sanity=False):
    verify_training_unchanged();base=Path(base);base.mkdir(parents=True,exist_ok=True)
    code=revision()
    if code['dirty']:raise RuntimeError('Commit source before launching')
    path=base/'queue_state.json';planned=jobs(sanity)
    if path.exists():
        state=json.loads(path.read_text())
        if state['code']!=code or state['planned_jobs']!=planned:raise RuntimeError('Queue source/config mismatch')
    else:state={'code':code,'planned_jobs':planned,'jobs':{cell_id(j):{'status':'pending'} for j in planned}}
    for j in planned:
        key=cell_id(j);entry=state['jobs'][key]
        if entry['status']=='completed':continue
        if entry['status']!='pending':raise RuntimeError('Unresolved prior run: '+key)
        entry.update(status='running',started_at=time.time());write_json(path,state)
        try:
            result=run_cell(j,base,sanity);entry.update(status='completed',mse=result['mse'],mae=result['mae'],seconds=result['seconds'])
        except Exception:
            entry.update(status='failed',error=traceback.format_exc());write_json(path,state);raise
        write_json(path,state)
    print('COMPRESSION_QUEUE_COMPLETE',len(planned),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sanity',action='store_true');p.add_argument('--run_dir',type=Path)
    a=p.parse_args();campaign(a.run_dir or (OUTPUT.with_name(OUTPUT.name+'_sanity') if a.sanity else OUTPUT),a.sanity)
