"""Bounded On-Adapter history cache. No trainable projection or output is cached."""
import hashlib
import numpy as np
import torch
from .resources import identity, digest, isolated_rng
from .batching import host_batch, transfer


def eligible(model):
    return (getattr(model, 'method', None) == 'On-Adapter' and hasattr(model, 'numeric')
            and not any(p.requires_grad for p in model.numeric.parameters())
            and not model.numeric.training)


def raw_features(model, batch):
    x = batch['x']
    mean = x.mean(1, keepdim=True).detach()
    std = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + 1e-5).detach()
    source = model.numeric.author_model
    features = source.enc_embedding((x-mean)/std, batch['mark'])
    features, _ = source.encoder(features, attn_mask=None)
    return features[:, :x.shape[-1]]


class FrozenHistory:
    def __init__(self, values, present, key, channels, width):
        self.values, self.present, self.key = values, present, key
        self.channels, self.width = channels, width
        self.stats = dict(gathers=0, numeric_encoder_runtime_calls=0,
                          resident_bytes=values.numel()*values.element_size())

    def gather(self, origins):
        from .prepared_text import cpu_origins
        ids = cpu_origins(origins)
        if np.any(ids < 0) or np.any(ids >= len(self.present)) or not self.present[ids].all():
            raise IndexError('Unprepared frozen origin; no runtime encoding fallback')
        if len(ids) and (len(ids) == 1 or np.all(np.diff(ids) == 1)):
            batch = self.values[int(ids[0]):int(ids[-1])+1]
        else:
            batch = self.values.index_select(0, torch.as_tensor(ids, device=self.values.device))
        self.stats['gathers'] += 1
        n, d = self.channels, self.width
        feature, mean, std, offset, text, available, confidence = batch.split(
            [n*d, n, n, 8, 768, 1, 1], dim=1)
        return feature.reshape(-1,n,d), mean[:,None], std[:,None], offset, text, available, confidence


@torch.no_grad()
def prepare_numeric(model, data, sets, settings, device):
    from .original_on_adapter import compute_offset_features
    if getattr(model, 'method', None) != 'On-Adapter':
        raise ValueError('Frozen numerical caching is only valid for On-Adapter')
    model.numeric.eval()
    if not eligible(model): raise ValueError('Backbone must be frozen and eval')
    origins = np.unique(np.concatenate([ds.indices for ds in sets.values()]))
    n, d = len(data.columns), settings['d_model']
    width = n*d + 2*n + 8 + 768 + 2
    required = len(data.values) * width * 4
    cap = settings['execution'].get('numeric_ram_bytes', 128*1024**2)
    if required > cap: raise MemoryError('Packed frozen history exceeds explicit byte budget')
    h = hashlib.sha256()
    for name, value in model.numeric.state_dict().items():
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    key = identity(dict(checkpoint=h.hexdigest(), data=digest(data.path), lookback=model.lookback,
                        representation='frozen-history-v2', dtype='float32', source=digest(__file__),
                        text_key=data.prepared_text.table.metadata['key']))
    values = torch.empty((len(data.values),width), dtype=torch.float32, device=device)
    present = np.zeros(len(data.values),dtype=bool)
    with isolated_rng():
        for dataset in sets.values():
            for start in range(0,len(dataset),settings['batch_size']):
                host = host_batch(dataset,np.arange(start,min(start+settings['batch_size'],len(dataset))))
                batch = transfer(host,device); ids = host['origin'].numpy(); x = batch['x']
                feature = raw_features(model,batch)
                mean = x.mean(1).detach(); std = torch.sqrt(x.var(1,unbiased=False)+1e-5).detach()
                text = model.encoder.pooled(ids,device)
                available = torch.as_tensor(model.encoder.available[ids],dtype=torch.float32,device=device)[:,None]
                confidence = torch.as_tensor(model.encoder.confidence[ids],device=device)[:,None]
                packed = torch.cat([feature.flatten(1),mean,std,compute_offset_features(x),text,available,confidence],dim=1)
                values.index_copy_(0,torch.as_tensor(ids,device=device),packed);present[ids] = True
    assert present[origins].all()
    model.frozen_numeric = FrozenHistory(values,present,key,n,d)
    return dict(key=key,bytes=required,origins=len(origins),dtype='float32',
                note='Frozen encoder/history/text only; trainable adapter and projections always recomputed')
