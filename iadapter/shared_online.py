"""One prediction/maturity driver for numerical adapters and native-model extensions."""
from dataclasses import dataclass
import time
import numpy as np
import torch
from .engine import MatureQueue,tensor_bytes
from .models import compute_offset_features
from .util import masked_loss,metrics,sync

@dataclass
class Record:
    origin:int
    mature:int
    text_ids:list
    context:dict
    issued_prediction:torch.Tensor

class InternalBackend:
    has_feedback=True
    def __init__(self,c,data,model,adapter):
        self.c=c;self.data=data;self.model=model.eval();self.adapter=adapter
        model.requires_grad_(False);model.zero_grad(set_to_none=True)
        self.x=torch.from_numpy(data.x).to(c.device)
        self.emb=torch.from_numpy(data.embeddings).to(c.device) if c.uses_text else None
        self.hg=torch.zeros(len(data.columns),c.d_model,device=c.device);self.ge=torch.zeros(c.text_dim,device=c.device)
        self.feedback_ids=[-1,-1]
    def parameters(self):return list(self.adapter.parameters())
    @torch.no_grad()
    def predict(self,t):
        c=self.c;self.adapter.eval();x=self.x[t-c.lookback+1:t+1].unsqueeze(0)
        pooled=self.emb[t:t+1] if c.uses_text else None;available=bool(self.data.texts[t]) if c.uses_text else False
        feature,mu,sd=self.model.context(x);offset=compute_offset_features(x)
        e=self.adapter.text(pooled,available,c.device)
        prediction=self.model.head(feature+self.adapter(feature,self.hg,e,self.ge,offset),mu,sd)
        context=dict(feature=feature.detach(),mu=mu.detach(),sigma=sd.detach(),offset=offset.detach(),hg=self.hg.detach().clone(),ge=self.ge.detach().clone(),pooled=pooled,available=available)
        ids=list(self.data.text_ids[t]) if c.uses_text else []
        gate=self.adapter.gate.detach().clone() if self.adapter.gate is not None else None
        return prediction,context,ids,gate
    def loss(self,record,y,mask):
        c=self.c;k=record.context;ad=self.adapter;ad.train()
        e=ad.text(k['pooled'],k['available'],c.device)
        residual=ad(k['feature'],k['hg'],e,k['ge'],k['offset'])
        loss=masked_loss(self.model.head(k['feature']+residual,k['mu'],k['sigma']),y,mask)
        q=torch.autograd.grad(loss,e,retain_graph=True)[0].mean(0).detach() if c.text_feedback else None
        g=None
        if c.numeric_feedback:
            probe=torch.zeros_like(k['feature'],requires_grad=True)
            probe_loss=masked_loss(self.model.head(k['feature']+probe,k['mu'],k['sigma']),y,mask)
            g=torch.autograd.grad(probe_loss,probe)[0].mean(0).detach()
        return loss+ad.auxiliary,loss,(g,q)
    def after_update(self,record,feedback):
        g,q=feedback
        if g is not None:self.hg=self.c.momentum*self.hg+(1-self.c.momentum)*g;self.feedback_ids[0]=record.origin
        if q is not None:self.ge=q;self.feedback_ids[1]=record.origin

class NativeBackend:
    has_feedback=False
    def __init__(self,c,data,model,embeddings):
        self.c=c;self.data=data;self.model=model;self.feedback_ids=[-1,-1]
        self.x=torch.from_numpy(data.x).to(c.device);self.emb=torch.from_numpy(embeddings).to(c.device)
    def parameters(self):return [p for p in self.model.parameters() if p.requires_grad]
    @torch.no_grad()
    def predict(self,t):
        c=self.c;self.model.eval();x=self.x[t-c.lookback+1:t+1].unsqueeze(0).clone();text=self.emb[t-c.lookback+1:t+1].unsqueeze(0).clone()
        prediction=self.model(x,0,text)
        ids=[rid for entry in self.data.text_ids[t-c.lookback+1:t+1] for rid in entry]
        return prediction,dict(x=x,summary=text),ids,None
    def loss(self,record,y,mask):
        self.model.train();k=record.context
        loss=masked_loss(self.model(k['x'],0,k['summary']),y,mask)
        return loss,loss,None
    def after_update(self,record,feedback):pass

class OnlineRunner:
    def __init__(self,c,data,backend,*,update_parameters=True,refresh_feedback=True):
        self.c=c;self.data=data;self.backend=backend
        self.update_parameters=update_parameters;self.refresh_feedback=refresh_feedback
    def run(self,origins=None):
        c=self.c;data=self.data;backend=self.backend
        origins=list(data.origins('test',c.lookback,c.horizon) if origins is None else origins)
        parameters=backend.parameters();optimizer=torch.optim.Adam(parameters,lr=c.lr,weight_decay=c.weight_decay) if self.update_parameters else None
        queue=MatureQueue();predictions=[];targets=[];masks=[];gates=[];feedback_ids=[];parameter_ids=[];contexts=[];events=[];predict_seconds=[];update_seconds=[];g_snap=[];q_snap=[]
        last_parameter_origin=-1;peak_queue_bytes=0
        for t in origins:
            sync(c.device);start=time.perf_counter();prediction,context,ids,gate=backend.predict(t);sync(c.device)
            predict_seconds.append(time.perf_counter()-start)
            if not torch.isfinite(prediction).all():raise FloatingPointError(f'Nonfinite original forecast {t}')
            assert all(int(rid.split(':')[0])<=t for rid in ids)
            predictions.append(prediction[0].detach().cpu().numpy().copy());targets.append(data.target[t+1:t+c.horizon+1]);masks.append(data.valid[t+1:t+c.horizon+1])
            gates.append(gate[0].cpu().numpy() if gate is not None else np.full((len(data.columns),5),np.nan,dtype=np.float32))
            feedback_ids.append(backend.feedback_ids.copy());parameter_ids.append(last_parameter_origin)
            if backend.has_feedback:
                g_snap.append(context['hg'].cpu().numpy().copy());q_snap.append(context['ge'].cpu().numpy().copy())
            record=Record(t,t+c.horizon,ids,context,prediction.detach().clone());queue.push(record)
            contexts.append(dict(origin=t,text_ids=ids,numeric_feedback_origin=backend.feedback_ids[0],text_feedback_origin=backend.feedback_ids[1],last_parameter_update_origin=last_parameter_origin))
            peak_queue_bytes=max(peak_queue_bytes,sum(tensor_bytes(r.issued_prediction)+sum(tensor_bytes(v) for v in r.context.values() if isinstance(v,torch.Tensor))+len(str(r.text_ids).encode()) for r in queue.pending))
            for rec in queue.pop_ready(t):
                y=torch.from_numpy(data.target[rec.origin+1:rec.mature+1]).unsqueeze(0).to(c.device)
                mask=torch.from_numpy(data.valid[rec.origin+1:rec.mature+1]).unsqueeze(0).to(c.device)
                event=dict(update_at=t,origin=rec.origin,mature_at=rec.mature,text_origin=rec.origin,state_origin=rec.origin,text_ids=rec.text_ids,valid_targets=int(mask.sum()),status='updated')
                if not mask.any():event['status']='skipped_missing_target';events.append(event);continue
                sync(c.device);start=time.perf_counter()
                if optimizer is not None:optimizer.zero_grad(set_to_none=True)
                total,loss,feedback=backend.loss(rec,y,mask)
                if not torch.isfinite(total):raise FloatingPointError(f'Nonfinite replay update {t}')
                if optimizer is not None:
                    total.backward();torch.nn.utils.clip_grad_norm_(parameters,c.grad_clip,error_if_nonfinite=True);optimizer.step();last_parameter_origin=rec.origin
                else:event['status']='feedback_only' if self.refresh_feedback else 'consumed_without_update'
                if self.refresh_feedback:backend.after_update(rec,feedback)
                if not (self.update_parameters and self.refresh_feedback):
                    event['parameter_updated']=self.update_parameters;event['feedback_refreshed']=self.refresh_feedback and backend.has_feedback
                sync(c.device);update_seconds.append(time.perf_counter()-start);event['loss']=float(loss.detach());events.append(event)
        assert queue.consumed==[t for t in origins if t+c.horizon<=origins[-1]],'Maturity queue lost/repeated origins'
        arrays=dict(origins=np.asarray(origins),predictions=np.stack(predictions),targets=np.stack(targets),valid_mask=np.stack(masks),gate_weights=np.stack(gates),feedback_origins=np.asarray(feedback_ids),parameter_update_origins=np.asarray(parameter_ids),prediction_seconds=np.asarray(predict_seconds),update_seconds=np.asarray(update_seconds))
        if backend.has_feedback:arrays.update(numeric_feedback_snapshots=np.stack(g_snap),text_feedback_snapshots=np.stack(q_snap))
        stats=metrics(arrays['predictions'],arrays['targets'],arrays['valid_mask'])
        stats.update(predictions=len(origins),updated=sum(e['status']=='updated' for e in events),consumed=len(events),pending=len(queue.pending),queue_tensor_and_id_peak_bytes=peak_queue_bytes,online_lr=c.lr,online_weight_decay=c.weight_decay,online_grad_clip=c.grad_clip)
        if not (self.update_parameters and self.refresh_feedback):stats['feedback_refreshed']=sum(e.get('feedback_refreshed',False) for e in events)
        return arrays,events,contexts,stats
