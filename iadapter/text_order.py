"""Pure same-character row-order transform; no model/runtime imports."""
from collections import Counter

def recent_entry(entry):
    text=entry['text'];spans=entry['spans'];groups=list(dict.fromkeys(int(rid.split(':')[0]) for rid,_,_ in spans))
    if len(groups)<2:return dict(text=text,spans=spans,reordered=False)
    if len(groups)!=2 or groups!=sorted(groups):raise ValueError('Unexpected chronological row groups')
    latest=groups[-1];rid,a,b=next((rid,a,b) for rid,a,b in spans if int(rid.split(':')[0])==latest)
    label='Facts. ' if rid.endswith(':fact') else 'Outlook. ';start=a-len(label)
    if not(start>0 and text[start:a]==label and text[start-1]==' '):raise ValueError('No intact latest-row boundary')
    older=text[:start-1];newer=text[start:];recent=newer+' '+older
    if len(recent)!=len(text) or Counter(recent)!=Counter(text):raise ValueError('Character inventory changed')
    mapped=[]
    for rid,a,b in spans:
        shift=-start if int(rid.split(':')[0])==latest else len(newer)+1
        mapped.append((rid,a+shift,b+shift))
        if recent[a+shift:b+shift]!=text[a:b]:raise ValueError('Content span changed')
    mapped.sort(key=lambda x:x[1]);return dict(text=recent,spans=mapped,reordered=True)
