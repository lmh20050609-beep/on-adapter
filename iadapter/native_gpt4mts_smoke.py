"""Integration evidence only: not a baseline accuracy evaluation."""
import gc,json,time,types
import numpy as np
import torch
from einops import EinopsError
from transformers import BertModel,BertTokenizerFast
from .config import ROOT
from .native_gpt4mts import build,model_class,native_config,COMMIT,WEIGHTS
from .review_sprint import development_data
from .prepare import weight_directory
from .util import seed_all,masked_loss,revision,write_json

OUTPUT=ROOT/'experiments/native_gpt4mts_smoke_20260929'

def main():
    OUTPUT.mkdir(parents=True,exist_ok=False);write_json(OUTPUT/'status.json',{'status':'running'})
    try:
        torch.set_num_threads(1);seed_all(2023);device='cuda'
        data,bounds=development_data('Economy',False)
        assert 'OT' in data.columns;column=data.columns.index('OT');L=36;H=6;t=data.origins('test',L,H)[0]
        historical_texts=data.texts[t-L+1:t+2]
        assert len(historical_texts)==L+1
        weights=weight_directory();bert=BertModel.from_pretrained(weights,local_files_only=True).to(device).eval();bert.requires_grad_(False)
        tok=BertTokenizerFast.from_pretrained(weights,local_files_only=True);embeddings=[]
        with torch.no_grad():
            for begin in range(0,len(historical_texts),4):
                encoded=tok(historical_texts[begin:begin+4],padding=True,truncation=True,max_length=512,return_tensors='pt').to(device)
                embeddings.append(bert(**encoded).last_hidden_state[:,0,:].cpu())
        embedding=torch.cat(embeddings);assert len(bert.encoder.layer)==12
        del bert,tok;gc.collect();torch.cuda.empty_cache()
        x=torch.from_numpy(np.stack([data.x[s-L+1:s+1,column:column+1] for s in [t,t+1]])).to(device)
        text=torch.stack([embedding[:L],embedding[1:]]).to(device)
        y=torch.from_numpy(data.target[t+1:t+H+1,column:column+1]).unsqueeze(0).to(device)
        mask=torch.from_numpy(data.valid[t+1:t+H+1,column:column+1]).unsqueeze(0).to(device)
        torch.cuda.reset_peak_memory_stats();model=build(L,H,device).eval()
        original=model_class(fix_batch=False).forward
        with torch.no_grad():
            patched=model(x.clone(),0,text.clone());reference=original(model,x.clone(),0,text.clone())
            torch.testing.assert_close(patched,reference,rtol=0,atol=0)
            single=model(x[:1].clone(),0,text[:1].clone());assert single.shape==(1,H,1) and torch.isfinite(single).all()
        failure=None
        try:
            with torch.no_grad():original(model,x[:1].clone(),0,text[:1].clone())
        except EinopsError as e:
            assert 'dimension' in str(e).lower() or 'dim' in str(e).lower()
            failure=type(e).__name__+': '+str(e)
        assert failure is not None,'Upstream batch=1 regression witness did not reproduce'
        trainable=[n for n,p in model.named_parameters() if p.requires_grad]
        for n,p in model.gpt2.named_parameters():assert p.requires_grad==('ln' in n or 'wpe' in n)
        before=model.out_layer.weight.detach().clone();optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.001)
        model.train();optimizer.zero_grad();pred=model(x[:1].clone(),0,text[:1].clone());loss=masked_loss(pred,y,mask)
        assert torch.isfinite(loss);loss.backward()
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        assert model.in_layer.weight.grad is not None and model.prompt_layer.weight.grad is not None and model.out_layer.weight.grad is not None
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step();assert not torch.equal(before,model.out_layer.weight)
        proof={'status':'passed','purpose':'integration only, one update is not performance evidence','upstream_commit':COMMIT,'weight_source':json.loads((WEIGHTS/'SOURCE.json').read_text()),'code':revision(),'model_config':vars(native_config(L,H)),'dataset':'Economy','column':'OT','origins':[t,t+1],'boundaries':bounds,'batch2_exact_forward_equivalence':True,'batch1_fixed_shape':list(single.shape),'upstream_batch1_error':failure,'real_target_masked_update':True,'valid_target_elements':int(mask.sum()),'bert_layers':12,'text_embedding':'frozen eval CLS, max_length512; data.text_rows preprocessing (up to3 clauses per field, current+previous row,1200-character limit), not entire raw documents','frozen_parameter_gradients_absent':True,'trainable_parameter_names':trainable,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'total_parameters':sum(p.numel() for p in model.parameters()),'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated()}
        write_json(OUTPUT/'validation.json',proof);write_json(OUTPUT/'status.json',{'status':'completed'})
        print('NATIVE_GPT4MTS_SMOKE_PASSED',proof['trainable_parameters'],proof['batch1_fixed_shape'],flush=True)
    except Exception:
        import traceback
        write_json(OUTPUT/'status.json',{'status':'failed','error':traceback.format_exc()});raise

if __name__=='__main__':main()
