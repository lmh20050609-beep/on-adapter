"""Training/online continuation, distinct from validation-selected best.pt.

Only local, hash-checked resume files are loaded with weights_only=False, since
Python/NumPy RNG and deque state are essential. No untrusted checkpoint loading.
"""
from collections import deque
import copy
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .resources import atomic_torch, atomic_json, digest, rng_state, restore_rng


def cpu_tree(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value, deque): return deque((cpu_tree(v) for v in value), maxlen=value.maxlen)
    if isinstance(value, tuple): return tuple(cpu_tree(v) for v in value)
    if isinstance(value, list): return [cpu_tree(v) for v in value]
    return copy.deepcopy(value)


def device_tree(value, device):
    if isinstance(value, torch.Tensor): return value.to(device)
    if isinstance(value, dict): return {k:(v if k=='origin' else device_tree(v,device)) for k,v in value.items()}
    if isinstance(value, deque): return deque((device_tree(v,device) for v in value),maxlen=value.maxlen)
    if isinstance(value, tuple): return tuple(device_tree(v,device) for v in value)
    if isinstance(value, list): return [device_tree(v,device) for v in value]
    return value


def optimizer_tree(value):
    if isinstance(value, torch.optim.Optimizer): return cpu_tree(value.state_dict())
    if isinstance(value, dict): return {k:optimizer_tree(v) for k,v in value.items()}
    if isinstance(value, list): return [optimizer_tree(v) for v in value]
    raise TypeError(type(value))


def restore_optimizers(target, state):
    if isinstance(target, torch.optim.Optimizer): target.load_state_dict(state)
    elif isinstance(target, dict):
        for k,v in target.items(): restore_optimizers(v,state[k])
    elif isinstance(target, list):
        for v,s in zip(target,state): restore_optimizers(v,s)
    else: raise TypeError(type(target))


RUNTIME_FIELDS = ('online_active','training_pending','pending','replay','previous','last_event',
                  'flag_online_learning','flag_update','flag_current','flag_basic','recent_concept')


def runtime_state(model):
    state = {'fields':{}, 'nonpersistent_buffers':{}, 'requires_grad':{}, 'training_modes':{}, 'optimizers':{}}
    for name,module in model.named_modules():
        fields = {k:cpu_tree(getattr(module,k)) for k in RUNTIME_FIELDS if hasattr(module,k)}
        if fields: state['fields'][name]=fields
        for k in module._non_persistent_buffers_set:
            state['nonpersistent_buffers'][(name,k)]=cpu_tree(module._buffers[k])
        state['training_modes'][name]=module.training
    state['requires_grad']={n:p.requires_grad for n,p in model.named_parameters()}
    if hasattr(model,'training_optimizer'): state['optimizers']['native']=optimizer_tree(model.training_optimizer)
    if hasattr(model,'framework'):
        f=model.framework
        state['optimizers']['framework']=optimizer_tree(f.opt)
        # Registered teacher/student parameters are in model.state_dict. The
        # remaining mutable framework state belongs to its causal update bridge.
        fields = ['laststepA','laststepB','replayBuffer','count','indices','discountedW',
                  'grad','grads','prev_grad','prev_grads','ocp','ocp_state']
        state['framework']={k:cpu_tree(getattr(f,k)) for k in fields if hasattr(f,k)}
    return state


def restore_runtime(model, state, device):
    modules=dict(model.named_modules())
    for name,fields in state['fields'].items():
        for k,v in fields.items(): setattr(modules[name],k,device_tree(v,device))
    for (name,k),v in state['nonpersistent_buffers'].items(): modules[name]._buffers[k]=device_tree(v,device)
    for name,value in state['training_modes'].items(): modules[name].training=value
    for name,p in model.named_parameters(): p.requires_grad_(state['requires_grad'][name])
    if 'native' in state['optimizers']: restore_optimizers(model.training_optimizer,state['optimizers']['native'])
    if 'framework' in state['optimizers']:
        restore_optimizers(model.framework.opt,state['optimizers']['framework'])
        for k,v in state.get('framework',{}).items(): setattr(model.framework,k,device_tree(v,device))


class ResumeFile:
    def __init__(self, folder, provenance, keys=None):
        self.folder=Path(folder); self.provenance=provenance; self.keys=keys

    def save(self, model, optimizer, progress, generator=None, online_optimizer=None):
        state=model.state_dict()
        if self.keys is not None: state={k:v for k,v in state.items() if k in self.keys}
        value=dict(format='resume-v1',provenance=self.provenance,model=cpu_tree(state),
                   optimizer=optimizer_tree(optimizer),runtime=runtime_state(model),
                   progress=cpu_tree(progress),rng=rng_state(),
                   generator=generator.get_state() if generator else None,
                   online_optimizer=optimizer_tree(online_optimizer) if online_optimizer else None)
        # Two slots + atomic manifest: interruption before manifest replacement
        # leaves the previously committed checkpoint loadable.
        manifest=self.folder/'resume.json'
        old=json.loads(manifest.read_text('utf-8')) if manifest.exists() else {}
        slot=1-old.get('slot',1); path=self.folder/('resume_%d.pt'%slot)
        atomic_torch(path,value)
        atomic_json(manifest,dict(slot=slot,sha256=digest(path),format='resume-v1',
                                phase=progress.get('phase'),epoch=progress.get('epoch'),cursor=progress.get('cursor')))

    def load(self):
        manifest=self.folder/'resume.json'
        if not manifest.exists(): return None
        record=json.loads(manifest.read_text('utf-8'));path=self.folder/('resume_%d.pt'%record['slot'])
        if digest(path)!=record['sha256']: raise ValueError('Resume checkpoint integrity failure')
        value=torch.load(path,map_location='cpu',weights_only=False)
        if value['provenance']!=self.provenance: raise ValueError('Resume checkpoint belongs to different code/config/data/features')
        return value

    def restore(self, value, model, optimizer, generator, device, online_optimizer=None):
        model.load_state_dict(value['model'],strict=False)
        restore_optimizers(optimizer,value['optimizer'])
        restore_runtime(model,value['runtime'],device)
        if generator is not None and value['generator'] is not None: generator.set_state(value['generator'])
        if online_optimizer is not None and value['online_optimizer'] is not None:
            restore_optimizers(online_optimizer,value['online_optimizer'])
        restore_rng(value['rng'])
        return value['progress']


def epoch_plan(length, batch_size, shuffle, generator):
    # DataLoader consumes its base seed before RandomSampler requests randperm.
    # Materializing this independent generator does not change model dropout RNG.
    loader=DataLoader(range(length),batch_size=batch_size,shuffle=shuffle,generator=generator,num_workers=0)
    torch.empty((),dtype=torch.int64).random_(generator=generator)
    return [list(batch) for batch in loader.batch_sampler]


def consume_sequential_loader_seed(generator):
    torch.empty((),dtype=torch.int64).random_(generator=generator)
