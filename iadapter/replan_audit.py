"""Exact selected-record to contextual BERT token-span audit, no model training."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from transformers import BertTokenizerFast
from .config import ROOT
from .data import load_data
from .legacy_core import clean_timemmd_text
from .prepare import weight_directory
from .util import write_json,revision

OUTPUT=ROOT/'experiments/replan_20260930'

def reconstruct(frame,details=False):
    records=[]
    for i,row in frame.iterrows():
        parts=[]
        for col,label in [('fact','Facts'),('preds','Outlook')]:
            if col not in frame:continue
            cleaned,count,_=clean_timemmd_text('' if pd.isna(row[col]) else str(row[col]),3,20)
            if count:parts.append((f'{i}:{col}',label+'. ',cleaned))
        records.append(parts)
    output=[]
    for t in range(len(frame)):
        pieces=[];spans=[];cursor=0
        for row in records[max(0,t-1):t+1]:
            for rid,label,text in row:
                if pieces:cursor+=1
                spans.append((rid,cursor+len(label),cursor+len(label)+len(text)))
                pieces.append(label+text);cursor+=len(label)+len(text)
        text=' '.join(pieces);left=0
        if len(text)>1200:
            left=len(text)-1200;clipped=text[left:];at=clipped.find(' ')
            if 0<at<80:left+=at+1
        text=text[left:]
        mapped=[(rid,max(0,a-left),min(len(text),b-left)) for rid,a,b in spans if b>left]
        output.append(dict(text=text,spans=mapped,character_crop=left,selected_characters_before_crop=cursor) if details else (text,mapped))
    return output

def run(base=OUTPUT):
    base=Path(base);out=base/'m0';out.mkdir(parents=True,exist_ok=False)
    tok=BertTokenizerFast.from_pretrained(weight_directory(),local_files_only=True)
    summaries=[]
    for name in ['Energy','SocialGood']:
        data=load_data(name,with_embeddings=True)
        frame=pd.read_csv(data.source);datecol=next(c for c in ['end_date','date','Date','start_date'] if c in frame)
        frame=frame.iloc[np.argsort(pd.to_datetime(frame[datecol]).values,kind='stable')].reset_index(drop=True)
        rebuilt=reconstruct(frame,details=True);rows=[]
        a=int(.6*len(frame));b=int(.7*len(frame));end=int(.8*len(frame))
        for t,entry in enumerate(rebuilt[:end]):
            text,spans=entry['text'],entry['spans']
            assert text==data.texts[t],f'Text reconstruction differs {name}/{t}'
            enc=tok(text,truncation=True,max_length=64,return_offsets_mapping=True)
            offsets=[(a,b) for a,b in enc['offset_mapping'] if b>a]
            visible=[rid for rid,a,b in spans if any(a<j and b>i for i,j in offsets)]
            assert set(visible)<=set(data.text_ids[t]) and all(int(x.split(':')[0])<=t for x in visible)
            right=max([j for _,j in offsets],default=0);whole=tok(text,truncation=False)['input_ids']
            encoded_count=len(enc['input_ids']) if text else 0
            counts={'fact':0,'preds':0,'labels_or_separators':0,'special':len(enc['input_ids'])-len(offsets) if text else 0}
            for i,j in offsets:
                owner=next((rid.split(':')[1] for rid,a,b in spans if a<j and b>i),None)
                counts[owner if owner in ['fact','preds'] else 'labels_or_separators']+=1
            assert sum(counts.values())==encoded_count
            latest=max([int(x.split(':')[0]) for x in data.text_ids[t]],default=-1)
            visible_latest=any(int(x.split(':')[0])==latest for x in visible)
            prior_rows=frame.iloc[max(0,t-1):t+1]
            raw_nonempty=any(pd.notna(v) and bool(str(v).strip()) for c in ['fact','preds'] if c in frame for v in prior_rows[c])
            age=(pd.Timestamp(data.dates[t])-pd.Timestamp(data.dates[max(int(x.split(':')[0]) for x in visible)])).days if visible else 0
            rows.append(dict(origin=t,split='train' if t<a else 'selection' if t<b else 'development',selected_ids=data.text_ids[t],visible_ids=visible,visible_text=text[:right],available=bool(text),raw_nonempty=raw_nonempty,cleaned_no_information=raw_nonempty and not bool(text),truncated=len(whole)>64,token_count=encoded_count,tokenizer_token_count=len(enc['input_ids']),visible_characters=right,latest_selected_row=latest,latest_row_visible=visible_latest,metadata=[int(bool(text)),right,len(visible),age],visible_token_counts=counts,selected_characters_before_crop=entry['selected_characters_before_crop'],selected_characters_after_crop=len(text),character_truncated=entry['character_crop']>0,character_crop=entry['character_crop'],pre_truncation_tokens=len(whole) if text else 0))
        write_json(out/(name+'_token_provenance.json'),rows)
        for split in ['train','selection','development']:
            group=[x for x in rows if x['split']==split];active=[x for x in group if x['available']]
            tokens={field:sum(x['visible_token_counts'][field] for x in group) for field in ['fact','preds','labels_or_separators','special']}
            denominator=tokens['fact']+tokens['preds']
            summaries.append(dict(character_truncated=sum(x['character_truncated'] for x in group),raw_selected_chars_mean=float(np.mean([x['selected_characters_before_crop'] for x in group])),encoded_chars_mean=float(np.mean([x['selected_characters_after_crop'] for x in group])),visible_chars_median=float(np.median([x['visible_characters'] for x in active])) if active else 0.,visible_age_days_median=float(np.median([x['metadata'][3] for x in active])) if active else 0.,pre64_tokens_median=float(np.median([x['pre_truncation_tokens'] for x in group])),fact_content_tokens=tokens['fact'],preds_content_tokens=tokens['preds'],labels_separator_tokens=tokens['labels_or_separators'],special_tokens=tokens['special'],content_token_denominator=denominator,fact_content_token_fraction=tokens['fact']/denominator if denominator else None,preds_content_token_fraction=tokens['preds']/denominator if denominator else None,dataset=name,split=split,rows=len(group),informative=len(active),raw_nonempty=sum(x['raw_nonempty'] for x in group),cleaned_no_information=sum(x['cleaned_no_information'] for x in group),truncated=sum(x['truncated'] for x in group),latest_selected_not_visible=sum(not x['latest_row_visible'] for x in active),visible_fact=sum(any(y.endswith(':fact') for y in x['visible_ids']) for x in active),visible_preds=sum(any(y.endswith(':preds') for y in x['visible_ids']) for x in active)))
    pd.DataFrame(summaries).to_csv(out/'text_coverage.csv',index=False)
    write_json(out/'validation.json',dict(status='passed',rows_reconstructed=True,token_offsets_verified=True,character_crop_verified=True,content_token_fraction_denominator='all visible content tokens; labels/separators and CLS/SEP excluded; dated anchors in content retained',availability='end_date proxy only',code=revision()))
    print(pd.DataFrame(summaries).to_string(index=False),flush=True)
    return summaries

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUTPUT);a=p.parse_args();run(a.output)
