from dataclasses import dataclass
from collections import deque
import time
import numpy as np
import torch
from .models import compute_offset_features
from .util import masked_loss,metrics,sync

@dataclass
class Replay:
    origin: int
    mature: int
    feature: torch.Tensor
    mu: torch.Tensor
    sigma: torch.Tensor
    offset: torch.Tensor
    hg: torch.Tensor
    ge: torch.Tensor
    pooled: torch.Tensor | None
    available: bool
    text_ids: list

class MatureQueue:
    def __init__(self): self.pending=deque(); self.consumed=[]
    def push(self,item):
        if self.pending: assert item.origin>self.pending[-1].origin
        self.pending.append(item)
    def pop_ready(self,t):
        while self.pending and self.pending[0].mature<=t:
            item=self.pending.popleft()
            assert not self.consumed or item.origin>self.consumed[-1]
            self.consumed.append(item.origin); yield item

def tensor_bytes(t): return t.numel()*t.element_size()

class Stream:
    def __init__(self,cfg,data,backbone,adaptation):
        self.cfg=cfg; self.data=data; self.backbone=backbone; self.adaptation=adaptation
        self.device=cfg.device
    def run(self,split,origins=None,trace=True,observer=None):
        c=self.cfg; d=self.data; model=self.backbone; ad=self.adaptation; device=self.device
        N=len(d.columns); L=c.lookback; H=c.horizon
        model.eval()
        for p in model.parameters(): p.requires_grad_(False)
        optimizer=torch.optim.Adam(ad.parameters(),lr=c.lr,weight_decay=c.weight_decay)
        hg=torch.zeros(N,c.d_model,device=device); ge=torch.zeros(c.text_dim,device=device)
        queue=MatureQueue(); preds=[]; targets=[]; masks=[]; gates=[]; events=[]; pred_times=[]; update_times=[]
        numerical_age_origin=None; text_age_origin=None; feedback_origins=[]; maximum_queue_bytes=0
        origins=list(origins if origins is not None else d.origins(split,L,H))
        xall=torch.from_numpy(d.x).to(device)
        targetall=torch.from_numpy(d.target).to(device); maskall=torch.from_numpy(d.valid).to(device)
        pooledall=torch.from_numpy(d.embeddings).to(device) if c.uses_text else None
        for t in origins:
            ad.eval(); sync(device); start=time.perf_counter()
            x=xall[t-L+1:t+1].unsqueeze(0)
            pooled=pooledall[t:t+1] if c.uses_text else None
            available=bool(d.texts[t]) if c.uses_text else False
            selected=list(d.text_ids[t]) if c.uses_text else []
            assert all(int(rid.split(':')[0])<=t for rid in selected)
            with torch.no_grad():
                feat,mu,sigma=model.context(x)
                offset=compute_offset_features(x)
                e=ad.text(pooled,available,device)
                residual=ad(feat,hg,e,ge,offset)
                pred=model.head(feat+residual,mu,sigma)
            sync(device); pred_times.append(time.perf_counter()-start)
            if not torch.isfinite(pred).all(): raise FloatingPointError(f'Nonfinite prediction at {t}')
            preds.append(pred[0].cpu().numpy()); targets.append(d.target[t+1:t+H+1]); masks.append(d.valid[t+1:t+H+1])
            gates.append(ad.gate[0].cpu().numpy() if ad.gate is not None else np.full((N,5),np.nan,dtype=np.float32))
            feedback_origins.append([numerical_age_origin if numerical_age_origin is not None else -1,text_age_origin if text_age_origin is not None else -1])
            item=Replay(t,t+H,feat.detach(),mu.detach(),sigma.detach(),offset.detach(),hg.detach().clone(),ge.detach().clone(),pooled,available,selected)
            queue.push(item)
            if observer is not None:
                observer(self,item,pred.detach(),residual.detach())
            maximum_queue_bytes=max(maximum_queue_bytes,sum(sum(tensor_bytes(getattr(r,name)) for name in ['feature','mu','sigma','offset','hg','ge'])+len(str(r.text_ids).encode()) for r in queue.pending))
            for rec in queue.pop_ready(t):
                assert rec.mature<=t
                y=targetall[rec.origin+1:rec.mature+1].unsqueeze(0)
                mask=maskall[rec.origin+1:rec.mature+1].unsqueeze(0)
                event={'update_at':t,'origin':rec.origin,'mature_at':rec.mature,'text_origin':rec.origin,'state_origin':rec.origin,'text_ids':rec.text_ids,'valid_targets':int(mask.sum()),'status':'updated'}
                if not mask.any():
                    event['status']='skipped_missing_target'; events.append(event); continue
                sync(device); update_start=time.perf_counter(); ad.train()
                optimizer.zero_grad(set_to_none=True)
                e_up=ad.text(rec.pooled,rec.available,device)
                residual=ad(rec.feature,rec.hg,e_up,rec.ge,rec.offset)
                loss_pred=masked_loss(model.head(rec.feature+residual,rec.mu,rec.sigma),y,mask)
                new_ge=None
                if c.text_feedback:
                    new_ge=torch.autograd.grad(loss_pred,e_up,retain_graph=True)[0].mean(0).detach()
                new_hg=None
                if c.numeric_feedback:
                    probe=torch.zeros_like(rec.feature,requires_grad=True)
                    probe_loss=masked_loss(model.head(rec.feature+probe,rec.mu,rec.sigma),y,mask)
                    new_hg=torch.autograd.grad(probe_loss,probe)[0].mean(0).detach()
                total=loss_pred+ad.auxiliary
                if not torch.isfinite(total): raise FloatingPointError(f'Nonfinite update at {t}')
                total.backward(); torch.nn.utils.clip_grad_norm_(ad.parameters(),c.grad_clip,error_if_nonfinite=True); optimizer.step()
                # New feedback is deliberately applied AFTER the replay update.
                if new_hg is not None: hg=c.momentum*hg+(1-c.momentum)*new_hg; numerical_age_origin=rec.origin
                if new_ge is not None: ge=new_ge; text_age_origin=rec.origin
                sync(device); update_times.append(time.perf_counter()-update_start)
                event['loss']=float(loss_pred.detach()); events.append(event)
        expected=[s for s in origins if s+H<=origins[-1]] if origins else []
        assert queue.consumed==expected, 'Missing/repeated mature origins'
        assert all(p.grad is None for p in model.parameters())
        arrays={'predictions':np.asarray(preds),'targets':np.asarray(targets),'valid_mask':np.asarray(masks),'origins':np.asarray(origins),'gate_weights':np.asarray(gates),'feedback_origins':np.asarray(feedback_origins),'prediction_seconds':np.asarray(pred_times),'update_seconds':np.asarray(update_times)}
        stats=metrics(arrays['predictions'],arrays['targets'],arrays['valid_mask'])
        stats.update(predictions=len(origins),updated=sum(e['status']=='updated' for e in events),consumed=len(events),pending=len(queue.pending),queue_peak_bytes=maximum_queue_bytes,gradient_cache_bytes=tensor_bytes(hg)+tensor_bytes(ge),text_cache_bytes=d.embeddings.nbytes if c.uses_text else 0)
        return arrays,events,stats
