from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .config import DATA_ROOT, ROOT
from .legacy_core import clean_timemmd_text

EXCLUDE={'date','Date','start_date','end_date','fact','pred','preds','Final_Output','Final_Search_8','Final_Search_16','prior_history_avg','prior_history_std'}

def causal_fill(values):
    return pd.DataFrame(values).ffill().fillna(0.).to_numpy(dtype=np.float32)

def text_rows(frame):
    records=[]
    for i,row in frame.iterrows():
        parts=[]; ids=[]
        for col,label in [('fact','Facts'),('preds','Outlook')]:
            if col not in frame: continue
            value=row[col]
            cleaned,k,_=clean_timemmd_text('' if pd.isna(value) else str(value),3,20)
            if k: parts.append(label+'. '+cleaned); ids.append(f'{i}:{col}')
        records.append((' '.join(parts),ids))
    texts=[]; text_ids=[]
    for i in range(len(frame)):
        selected=records[max(0,i-1):i+1]
        text=' '.join(x[0] for x in selected if x[0])
        if len(text)>1200:
            text=text[-1200:]; at=text.find(' ')
            if 0<at<80: text=text[at+1:]
        texts.append(text); text_ids.append([rid for _,ids in selected for rid in ids])
    return texts,text_ids

@dataclass
class Data:
    name: str
    x: np.ndarray
    target: np.ndarray
    valid: np.ndarray
    input_missing: np.ndarray
    texts: list
    text_ids: list
    dates: list
    columns: list
    train_end: int
    val_end: int
    mean: np.ndarray
    scale: np.ndarray
    source: str
    embeddings: np.ndarray | None = None

    def origins(self,split,L,H):
        start,end={'train':(L-1,self.train_end),'val':(self.train_end-1,self.val_end),'test':(self.val_end-1,len(self.x))}[split]
        return list(range(start,end-H))
    def window(self,s,L,H):
        return self.x[s-L+1:s+1],self.target[s+1:s+H+1],self.valid[s+1:s+H+1]
    def metadata(self):
        return dict(dataset=self.name,source=self.source,columns=self.columns,rows=len(self.x),train_end=self.train_end,val_end=self.val_end,mean=self.mean.tolist(),scale=self.scale.tolist(),missing_targets=int((~self.valid).sum()),origin_time_convention='end of observed numeric aggregation period: end_date, not date/start_date',text_availability='merged end_date proxy; not verified publication time')

def load_data(name,with_embeddings=False,root=DATA_ROOT):
    path=Path(root)/f'{name}.csv'
    frame=pd.read_csv(path)
    datecol=next(c for c in ['end_date','date','Date','start_date'] if c in frame)
    dates=pd.to_datetime(frame[datecol],errors='raise')
    order=np.argsort(dates.values,kind='stable')
    frame=frame.iloc[order].reset_index(drop=True)
    dates=dates.iloc[order].reset_index(drop=True)
    columns=[c for c in frame if c not in EXCLUDE and not c.startswith('Unnamed:') and pd.api.types.is_numeric_dtype(frame[c])]
    raw=frame[columns].to_numpy(dtype=np.float32)
    valid=np.isfinite(raw)
    raw=np.where(valid,raw,np.nan)
    filled=causal_fill(raw)
    n=len(raw); train_end=int(.6*n); val_end=int(.8*n)
    mean=filled[:train_end].mean(0,dtype=np.float64).astype(np.float32)
    scale=filled[:train_end].std(0,dtype=np.float64).astype(np.float32)
    scale=np.where(scale<1e-8,1.,scale)
    x=(filled-mean)/scale
    target=np.where(valid,(raw-mean)/scale,0.).astype(np.float32)
    texts,ids=text_rows(frame)
    data=Data(name,x,target,valid,~valid,texts,ids,[str(d) for d in dates],columns,train_end,val_end,mean,scale,str(path))
    if with_embeddings:
        cache=ROOT/'cache'/f'{name}_bert_contextual.npz'
        with np.load(cache,allow_pickle=False) as z:
            assert z['embeddings'].shape==(n,768)
            assert z['texts'].tolist()==texts, 'Text cache does not match current selection'
            data.embeddings=z['embeddings'].copy()
    return data

def change_scores(data,L):
    # Available historical halves only; a descriptive score, not drift truth.
    scores=np.zeros(len(data.x),dtype=np.float32)
    for s in range(L-1,len(data.x)):
        x=data.x[s-L+1:s+1]; a,b=np.array_split(x,2)
        scores[s]=np.mean(np.abs(a.mean(0)-b.mean(0))+np.abs(a.std(0)-b.std(0)))
    val=scores[max(L-1,data.train_end-1):data.val_end-1]
    return scores,np.quantile(val,[1/3,2/3]).tolist()
