"""Prepare frozen contextual BERT representations; no forecasting training."""
import argparse,json,time,zipfile,gc
from pathlib import Path
import numpy as np
import torch
from transformers import BertModel,BertTokenizerFast
from .config import ROOT,BERT_ARCHIVE,DOMAINS
from .data import load_data
from .util import write_json,seed_all,sync

def weight_directory():
    target=ROOT/'weights/bert-base-uncased'; target.mkdir(parents=True,exist_ok=True)
    required=['config.json','model.safetensors','tokenizer.json','tokenizer_config.json','vocab.txt']
    with zipfile.ZipFile(BERT_ARCHIVE) as archive:
        for name in required:
            out=target/name
            if not out.exists():
                with archive.open('bert-base-uncased/'+name) as src, out.open('wb') as dst:
                    import shutil
                    shutil.copyfileobj(src,dst)
    return target

@torch.no_grad()
def contextual_pool(model,tokenizer,texts,device):
    encoded=tokenizer(texts,padding=True,truncation=True,max_length=64,return_tensors='pt').to(device)
    hidden=model(**encoded).last_hidden_state
    mask=encoded['attention_mask'].unsqueeze(-1)
    return ((hidden*mask).sum(1)/mask.sum(1).clamp(min=1)).cpu().numpy()

def prepare(domains=DOMAINS,batch_size=8,device='cuda'):
    torch.set_num_threads(1); seed_all(2023)
    weights=weight_directory(); tok=BertTokenizerFast.from_pretrained(weights,local_files_only=True)
    model=BertModel.from_pretrained(weights,local_files_only=True).eval().to(device)
    for p in model.parameters(): p.requires_grad_(False)
    assert len(model.encoder.layer)==12
    ROOT.joinpath('cache').mkdir(exist_ok=True)
    for name in domains:
        data=load_data(name); target=ROOT/'cache'/f'{name}_bert_contextual.npz'
        if target.exists():
            with np.load(target,allow_pickle=False) as z:
                if z['texts'].tolist()==data.texts and z['embeddings'].shape==(len(data.x),768):
                    print(f'CACHE_READY {name}',flush=True); continue
            raise RuntimeError(f'Stale text cache: {target}')
        sync(device); start=time.perf_counter()
        if device.startswith('cuda'): torch.cuda.reset_peak_memory_stats()
        vectors=np.zeros((len(data.x),768),dtype=np.float32)
        ids=[i for i,t in enumerate(data.texts) if t]
        cursor=0; actual_batch=batch_size
        while cursor<len(ids):
            ix=ids[cursor:cursor+actual_batch]
            try: batch=contextual_pool(model,tok,[data.texts[i] for i in ix],device)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if actual_batch==1: raise
                actual_batch=max(1,actual_batch//2); continue
            vectors[ix]=batch; cursor+=len(ix)
        sync(device); elapsed=time.perf_counter()-start
        # Check disk-bound cache against an independent singleton forward.
        max_error=0.
        for i in ids[:2]:
            direct=contextual_pool(model,tok,[data.texts[i]],device)[0]
            max_error=max(max_error,float(np.abs(direct-vectors[i]).max()))
            np.testing.assert_allclose(direct,vectors[i],atol=3e-5,rtol=3e-4)
        np.savez_compressed(target,embeddings=vectors,texts=np.asarray(data.texts),available=np.asarray([bool(t) for t in data.texts]))
        write_json(target.with_suffix('.json'),dict(data=data.metadata(),contextual_layers=12,max_length=64,batch_size=actual_batch,device=device,encoding_seconds=elapsed,encoded_rows=len(ids),seconds_per_encoded_row=elapsed/max(1,len(ids)),cache_direct_max_error=max_error,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device.startswith('cuda') else 0,weights=str(weights)))
        print(f'ENCODED {name} rows={len(ids)} seconds={elapsed:.2f}',flush=True)
    del model; gc.collect()
    if device.startswith('cuda'): torch.cuda.empty_cache()

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--datasets',default=','.join(DOMAINS)); p.add_argument('--batch_size',type=int,default=8); p.add_argument('--device',default='cuda')
    a=p.parse_args(); prepare(a.datasets.split(','),a.batch_size,a.device)
