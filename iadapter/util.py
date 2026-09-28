import json,random,subprocess
from pathlib import Path
import numpy as np
import torch
from .config import ROOT

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False

def write_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8'); temp.replace(path)

def revision():
    def git(*args): return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
    try: return {'commit':git('rev-parse','HEAD'),'dirty':bool(git('status','--porcelain','--untracked-files=no'))}
    except subprocess.CalledProcessError: return {'commit':'uncommitted','dirty':True}

def masked_loss(pred,y,mask):
    count=mask.sum()
    if count==0: return pred.sum()*0
    return ((pred-y).square()*mask).sum()/count

def metrics(pred,y,mask):
    delta=(np.asarray(pred)-np.asarray(y))[np.asarray(mask,dtype=bool)]
    if not len(delta): raise ValueError('No observed targets to score')
    if not np.isfinite(delta).all(): raise ValueError('Non-finite valid predictions')
    return {'mse':float(np.mean(delta.astype(np.float64)**2)),'mae':float(np.mean(np.abs(delta.astype(np.float64)))),'valid_elements':int(len(delta))}

def sync(device):
    if str(device).startswith('cuda'): torch.cuda.synchronize()
