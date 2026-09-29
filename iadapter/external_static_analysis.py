"""Verify the shared static task against the original CSV and selected state."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .audit import original_reference,same,require,verify_trace
from .external_static import OUTPUT,METHODS,pilot_config
from .native_gpt4mts import native_config
from .util import metrics,write_json
from .analyze import md_table

def analyze(base=OUTPUT,dataset='Economy',seed=2023):
    base=Path(base);state=json.loads((base/'queue_state.json').read_text())
    require(state['methods']==METHODS,'Method list changed')
    ref=original_reference(dataset);n=ref['rows'];a,b,end=int(.6*n),int(.7*n),int(.8*n)
    # Original CSV reference keeps numerical columns in source order.
    from .data import load_data
    raw=load_data(dataset,False);k=raw.columns.index('OT')
    origins=np.arange(b-1,end-6);target=np.stack([ref['target'][t+1:t+7,k:k+1] for t in origins]);mask=np.stack([ref['valid'][t+1:t+7,k:k+1] for t in origins])
    rows=[];checkpoints=[]
    for method in METHODS:
        d=base/'runs'/method
        require(state['jobs'][method]['status']=='completed' and json.loads((d/'status.json').read_text())['status']=='completed','Incomplete pilot method')
        c=pilot_config('numeric_add' if method=='numeric_static' else 'full',dataset=dataset,seed=seed)
        cfg=json.loads((d/'config.json').read_text());saved=json.loads((d/'metrics.json').read_text())
        require(cfg==saved['config'] and cfg['code']==state['code'] and not state['code']['dirty'],'Provenance mismatch')
        require(cfg['method']==method and cfg['dataset']==dataset and cfg['seed']==seed and cfg['horizon']==6 and cfg['lookback']==c.lookback,'Wrong task')
        require(cfg['protocol']==c.protocol and cfg['data']['columns']==['OT'],'Wrong protocol/target')
        require(cfg['boundaries']==dict(raw_rows=n,train_end=a,backbone_selection_end=b,development_end=end,scope='exploratory development; old test not evaluated'),'Wrong boundaries')
        require(cfg['data']['train_end']==a and cfg['data']['val_end']==b and cfg['data']['rows']==end,'Wrong data extent')
        same(cfg['data']['mean'],ref['mean'][k:k+1],'Wrong OT mean');same(cfg['data']['scale'],ref['scale'][k:k+1],'Wrong OT scale')
        require(cfg['training_budget']==dict(max_epochs=30,batch_size=16,lr=.001,patience=5,backbone_weight_decay=0,backbone_grad_clip=None),'Training budget differs')
        require(saved['static_parameters_unchanged'] and saved['development_updates']==0,'Evaluation not static')
        require(saved['prediction_count']==len(origins),'Wrong prediction count')
        with np.load(d/'predictions.npz') as z:
            same(z['origins'],origins,'Mismatched origins',exact=True);same(z['targets'],target,'Wrong original OT target');same(z['valid_mask'],mask,'Wrong target mask',exact=True)
            scored=metrics(z['predictions'],target,mask)
        for metric in ['mse','mae']:require(abs(scored[metric]-saved[metric])<=1e-12,'Metric does not recompute')
        require(saved['valid_elements']==int(mask.sum()),'Valid element count mismatch')
        contexts=json.loads((d/'contexts.json').read_text());require([x['origin'] for x in contexts]==origins.tolist(),'Context origins mismatch')
        for record in contexts:
            t=record['origin']
            ids=[] if method=='numeric_static' else (ref['text_ids'][t] if method=='iadapter_static' else [rid for at in ref['text_ids'][t-c.lookback+1:t+1] for rid in at])
            require(record['text_ids']==ids and all(int(v.split(':')[0])<=t for v in ids),'Wrong/unavailable text IDs')
            require(record['numeric_feedback_origin']==record['text_feedback_origin']==-1 and record['updated_after_prediction']==False,'Feedback refreshed during static evaluation')
        training=saved['training'];history=training['history'];require(0<len(history)<=30,'Epoch budget exceeded')
        best=min(history,key=lambda h:h['val_mse']);require(training['best_val_mse']==best['val_mse'],'Selected score not best validation score')
        checkpoint=Path(saved['backbone_checkpoint']);blob=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if method=='gpt4mts_static':
            require(cfg['model']==vars(native_config(c.lookback,c.horizon)) and cfg['text_max_length']==512,'Native model/text interface changed')
            require(training['best_epoch']==best['epoch'] and training['frozen_parameters_unchanged'],'Native checkpoint mismatch')
            require(blob['training']==training and blob['model_config']==cfg['model'] and blob['code']==state['code'],'Native checkpoint provenance')
            require(sum(v.numel() for v in blob['trainable'].values())==saved['trainable_parameters_before_static_freeze'],'Native parameter count')
            warm_updates=0;steps=training['optimizer_steps']
        else:
            require(cfg['model']==c.to_dict(),'Adapter effective configuration')
            require(blob['config']['columns']==['OT'] and blob['config']['train_end']==a and blob['config']['val_end']==b and blob['config']['batch']==16,'Backbone selection protocol')
            require(blob['source_revision']==state['code'],'Backbone source mismatch')
            smallref=dict(ref);smallref['valid']=ref['valid'][:,k:k+1];smallref['target']=ref['target'][:,k:k+1]
            events=verify_trace(d/'warm_updates.jsonl',np.arange(c.lookback-1,a-c.horizon),c,smallref)
            warm_updates=sum(x['status']=='updated' for x in events);checkpoints.append(checkpoint)
            params=torch.load(d/'warm_adapter.pt',map_location='cpu',weights_only=True)
            require(sum(v.numel() for v in params.values())==saved['trainable_parameters_before_static_freeze'],'Adapter parameter count')
            steps=int(np.ceil(len(np.arange(c.lookback-1,a-c.horizon))/16))*len(history)
        rows.append(dict(dataset=dataset,method=method,seed=seed,origins=len(origins),valid_elements=int(mask.sum()),mse=scored['mse'],mae=scored['mae'],epochs=len(history),best_epoch=best['epoch'],backbone_optimizer_steps=steps,adapter_warm_updates=warm_updates,trainable_parameters_before_static_freeze=saved['trainable_parameters_before_static_freeze'],total_parameters_excluding_BERT=saved['total_parameters'],total_seconds=saved['total_seconds'],evaluation_seconds=saved['static_evaluation_seconds'],static_peak_allocated_MiB=saved['static_evaluation_peak_allocated_bytes']/2**20,text_interface=cfg['text_interface']))
    require(checkpoints[0].resolve()==checkpoints[1].resolve(),'Internal methods did not share backbone')
    out=base/'reports';out.mkdir(exist_ok=True);table=pd.DataFrame(rows);table.to_csv(out/'summary.csv',index=False)
    proof=dict(status='passed',completed=3,dataset=dataset,seed=seed,seed_count=1,origin_count=len(origins),valid_target_elements=int(mask.sum()),original_csv_targets_masks=True,shared_internal_OT_backbone=True,static_evaluation_no_updates=True,selection_uses_60_70_prefix=True,text_ids_historical_only=True,text_consumption_identical=False,code=state['code'],scope='single-seed exploratory common-OT static development; not original Table2 reproduction')
    write_json(out/'validation.json',proof)
    (out/'RESULTS.md').write_text(f'# 共同OT静态对照：单seed开发试验\n\n{dataset}，L{c.lookback}/H6，seed{seed}；训练{a}行、选择段截至{b}、开发段截至{end}。{len(origins)}个预测原点、{int(mask.sum())}个有效OT目标元素（重叠目标不视为独立重复）。三方法均不使用开发标签更新。\n\n'+md_table(table[['method','seed','origins','valid_elements','mse','mae','epochs','best_epoch','trainable_parameters_before_static_freeze','total_parameters_excluding_BERT']].round(6))+'\n\n内部两个方法共享单变量骨干，GPT4MTS保留作者模型。iAdapter在训练流预热，开发反馈保持零；GPT4MTS为批量监督训练后静态评估。文本可用性截止点一致，实际历史长度和编码方式不同，不能称文本输入完全相同。新表不与原多变量在线结果合并。只有一个seed，不计算SD/显著性，不声称普遍优于任何方法。训练阶段、时间和文本接口完整字段见summary.csv，原始预测可重算。\n',encoding='utf-8')
    print('STATIC_PILOT_VALIDATED',json.dumps(proof),flush=True)
    return proof

if __name__=='__main__':analyze()
