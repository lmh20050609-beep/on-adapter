"""Origin-based delayed evaluation and author online-method bridges."""
import copy
import importlib
import json
import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from .config import ROOT
from .data import ForecastWindows, masked_errors
from .native import NativeForecast, model_config, source_module
from .prepared_text import PreparedText, cpu_origins
from .batching import observed_error, host_batch

class ResidualOnline(torch.nn.Module):
    def __init__(self, numeric, method, settings, data, lookback, horizon):
        super().__init__()
        self.numeric, self.method, self.data = numeric, method, data
        self.lookback, self.horizon = lookback, horizon
        self.online_active = False
        self.numeric.requires_grad_(method == 'ADAPT-Z')
        self.encoder = None
        self.register_buffer('hg', torch.zeros(len(data.columns), settings['d_model']))
        self.register_buffer('ge', torch.zeros(64))
        if method == 'ADAPT-Z':
            from .multimodal import author_class
            holder = author_class(ROOT / 'external/adaptz/adapt_z.py', 'Ada_fineture',
                                   {'torch': torch, 'nn': torch.nn})
            args = SimpleNamespace(z_shape=(1, len(data.columns), settings['d_model']), device=str(next(numeric.parameters()).device))
            self.adaptation = holder.__new__(holder).bulid_adapter(settings['d_model'], args)
            self.momentum = 0.  # Author replaces the historical gradient directly.
            self.training_optimizer = torch.optim.Adam(
                [{'params': self.numeric.parameters(), 'lr': .00003},
                 {'params': self.adaptation.parameters(), 'lr': .001}])
            self.training_pending = deque()
        else:
            from .original_on_adapter import OriginalOnAdapter
            self.adaptation = OriginalOnAdapter(settings, len(data.columns))
            self.encoder = data.prepared_text
            self.momentum = .9
            self.training_optimizer = torch.optim.Adam(self.adaptation.parameters(), lr=settings['learning_rate'], weight_decay=.0001)
            self.training_pending = deque()
            confidence, c = [], 0.
            for text in data.texts:
                available = float(bool(text.strip()))
                c = .7 * c + .3 * available
                confidence.append(max(c, .25 * available))
            self.confidence = confidence
        self.last_feature = self.last_text = self.last_base = None

    def train(self, mode=True):
        super().train(mode)
        self.numeric.train(mode if self.method == 'ADAPT-Z' and not self.online_active else False)
        return self

    def begin_training_epoch(self):
        self.hg.zero_(); self.ge.zero_()
        self.training_pending.clear()

    def prepare_evaluation(self):
        self.hg.zero_(); self.ge.zero_()

    def native_training_step(self, batch):
        # The custom chronological lifecycle applies only to ADAPT-Z. The other
        # residual variants use their ordinary masked offline objective.
        from .worker import loss_ot
        first_origin = int(batch['origin'].min())
        while self.training_pending and int(self.training_pending[0]['origin'].max()) + self.horizon <= first_origin:
            old = self.training_pending.popleft()
            old_task, _ = loss_ot(self, old)
            target, mask = old['y'], old['mask']
            probe = observed_error(self.last_base,old).square().mean()
            gradient = torch.autograd.grad(probe, self.last_feature, retain_graph=self.last_text is not None)[0].mean(0)
            if self.last_text is not None:
                text_gradient = torch.autograd.grad(old_task, self.last_text)[0].mean(0)
                self.ge.copy_(text_gradient.detach())
            self.hg.copy_(self.momentum * self.hg + (1-self.momentum) * gradient.detach())
        self.training_optimizer.zero_grad(set_to_none=True)
        loss, _ = loss_ot(self, batch)
        loss = loss + self.auxiliary_loss()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.)
        self.training_optimizer.step()
        self.training_pending.append({k: (v.detach() if isinstance(v,torch.Tensor) else v) for k, v in batch.items()})
        return loss.detach()

    def reset_online(self):
        self.online_active = True
        self.hg.zero_(); self.ge.zero_()
        if self.method == 'ADAPT-Z':
            self.numeric.requires_grad_(False)
            self.numeric.author_model.projection.requires_grad_(True)

    def forward(self, batch):
        from .original_on_adapter import compute_offset_features
        x = batch['x']
        source = self.numeric.author_model
        if hasattr(self, 'frozen_numeric'):
            from .numeric_features import eligible
            if not eligible(self):
                raise RuntimeError('Frozen feature cache used on a mutable or training backbone')
            features,mean,std,offset,pooled,available,confidence=self.frozen_numeric.gather(batch['origin'])
        else:
            mean = x.mean(1, keepdim=True).detach()
            std = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + 1e-5).detach()
            features = source.enc_embedding((x - mean) / std, batch['mark'])
            features, _ = source.encoder(features, attn_mask=None)
            features = features[:, :x.shape[-1]]
        if self.method != 'ADAPT-Z' or not self.training:
            features = features.detach().requires_grad_(torch.is_grad_enabled())
        elif not features.requires_grad and torch.is_grad_enabled():
            features = features.detach().requires_grad_(True)
        def head(z):
            return source.projection(z).permute(0, 2, 1) * std + mean
        self.last_feature, self.last_base = features, head(features)
        if self.encoder is None:
            residual = self.adaptation(features, self.hg)
            self.last_text = None
        else:
            if not hasattr(self,'frozen_numeric'):
                origins = cpu_origins(batch['origin'])
                pooled = self.encoder.pooled(origins, x.device)
                available = torch.from_numpy(self.encoder.available[origins]).to(x.device)[:,None]
                confidence = torch.from_numpy(self.encoder.confidence[origins]).to(x.device)[:,None]
                offset = compute_offset_features(x)
            text = self.adaptation.project(pooled)
            text = text * available * confidence
            self.last_text = text
            residual = self.adaptation(features, self.hg, text, self.ge, offset)
        return head(features + residual)

    def auxiliary_loss(self):
        if self.encoder is None:
            return self.hg.new_zeros(())
        value = self.adaptation.auxiliary
        return value if value is not None else self.hg.new_zeros(())

class ProceedBridge(NativeForecast):
    def __init__(self, model, horizon, label_len, data, lookback, settings):
        super().__init__(model, horizon, label_len)
        self.data, self.lookback = data, lookback
        self.training_optimizer = torch.optim.Adam(self.parameters(), lr=settings['learning_rate'])
        self.training_windows = ForecastWindows(data, 'train', lookback, horizon, method='Proceed')
        self.origin_to_item = {int(origin): i for i, origin in enumerate(self.training_windows.indices)}

    def native_training_step(self, batch):
        """Author two-phase calibration, confined to the training partition."""
        from torch.utils.data import default_collate
        from .worker import loss_ot, to_device
        source = self.author_model
        old_origins = batch['origin'].detach().cpu().numpy() - self.horizon
        old = default_collate([self.training_windows[self.origin_to_item[int(o)]] for o in old_origins])
        old = to_device(old, batch['x'].device)
        source.backbone.requires_grad_(True)
        source.freeze_bias(False); source.freeze_adapter(True)
        source.flag_online_learning = True; source.flag_current = False
        self.training_optimizer.zero_grad(set_to_none=True)
        recent_loss, _ = loss_ot(self, old)
        recent_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.)
        self.training_optimizer.step()
        self.after_training_batch(old)
        source.freeze_bias(True); source.backbone.requires_grad_(False)
        source.freeze_adapter(False); source.flag_current = True
        self.training_optimizer.zero_grad(set_to_none=True)
        current_loss, _ = loss_ot(self, batch)
        current_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.)
        self.training_optimizer.step()
        self.after_training_batch(batch)
        source.backbone.requires_grad_(True); source.freeze_bias(False)
        source.flag_online_learning = False; source.flag_current = False
        return current_loss.detach()

    def after_training_batch(self, batch):
        self.author_model.recent_batch = torch.cat([batch['x'], batch['y']], 1).detach()

    def prepare_evaluation(self):
        source = self.author_model
        # Use only the last fully observed training context; no validation adaptation state.
        end = self.data.train_end
        source.recent_batch = torch.from_numpy(self.data.values[end - self.lookback - self.horizon:end]).unsqueeze(0).to(next(source.parameters()).device)
        source.flag_update = False; source.flag_current = False; source.flag_online_learning = False
        source.freeze_adapter(True)
        source.freeze_bias(False)

    def reset_online(self):
        self.prepare_evaluation()

class DSOFBridge(torch.nn.Module):
    def __init__(self, framework, data, lookback, horizon):
        super().__init__()
        self.framework, self.data, self.lookback, self.horizon = framework, data, lookback, horizon
        self.teacher, self.student = framework.main_model, framework.student_model

    def args_from_batch(self, batch):
        future_marks = batch['mark'][:, -1:].expand(-1, self.horizon, -1)
        return [None, batch['x'], batch['y'], batch['mark'], future_marks]

    def forward(self, batch):
        out = self.framework._process_one_batch(*self.args_from_batch(batch), mode='val')
        return self.framework.studentFFN.get_future(out['pred'])

    def native_training_step(self, batch):
        if not bool(batch['mask'].all()):
            raise ValueError('Native DSOF training requires fully observed labels; no fake imputation is supplied')
        return float(self.framework.train_step(*self.args_from_batch(batch))['loss'])

    def reset_online(self):
        self.framework.laststepA = [None] * 4
        self.framework.laststepB = [None] * 4
        self.framework.replayBuffer = [None] * 4
        self.framework.count = 0
        valid = self.data.mask[:, -1]
        last = np.flatnonzero(valid)[-1]
        if not valid[:last + 1].all():
            raise ValueError('Native DSOF masked-buffer extension is needed for internal missing OT records')

def build_online(method, settings, data, lookback, horizon, device):
    args = model_config(settings, data, lookback, horizon, method)
    if method == 'DSOF':
        src = ROOT / 'external/dsof/src'
        sys.path.insert(0, str(src))
        for name in list(sys.modules):
            if name in {'utils', 'models', 'data'} or name.startswith(('utils.', 'models.', 'data.')):
                del sys.modules[name]
        framework_cls = importlib.import_module('trainers.w_student.trainerFramework_ERTDRes').Exp_TS2VecSupervised
        teacher = importlib.import_module('models.DLinear').Model
        student = importlib.import_module('models.MLP').Model
        args.online_learning = 'full'; args.device = device; args.features = 'M'; args.use_amp = False
        args.main_model = {'model': 'DLinear', 'individual': False}
        args.student_model = {'model': 'MLP', 'mlp_width': 16, 'mlp_depth': 3, 'mlp_dropout': 0., 'act_str': 'Identity'}
        args.test_run = False; args.replayBufferSize = 300; args.batchReplaySize = 32
        args.num_ERepochs = 1; args.freq_ERupdate = 1; args.td_k = [1]; args.discounted = .9
        args.er = True; args.td = True
        args.opt_main = args.opt_student = {'train': {'opt': 'adam', 'lr': settings['learning_rate']},
                                           'test': {'batch': {'opt': 'adam', 'lr': settings['learning_rate']},
                                                    'online': {'opt': 'adam', 'lr': settings['learning_rate']}}}
        framework = framework_cls(args, teacher, student)
        framework.opt = framework._select_optimizer()
        metadata = vars(args).copy(); metadata['device'] = str(device)
        from .stream_dsof import CausalDSOF
        return CausalDSOF(framework, data, lookback, horizon).to(device), metadata
    module = source_module('tslib', 'models/iTransformer.py')
    numeric = NativeForecast(module.Model(args).to(device), horizon, args.label_len)
    # Shared, validation-selected numerical checkpoint is a true prerequisite.
    kind = settings.get('run_kind', settings['campaign'])
    from .provenance import numeric_checkpoint
    path = numeric_checkpoint(settings, data.domain, horizon, settings.get('active_seed', 2023))
    numeric.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    args.pretrained_numeric_checkpoint = str(path)
    if method == 'Proceed':
        proceed = source_module('proceed', 'adapter/proceed.py')
        args.freeze = False; args.concept_dim = 200; args.bottleneck_dim = 32
        args.act = 'sigmoid'; args.individual_generator = False; args.ema = 0.
        args.wo_clip = False; args.do_predict = False; args.tune_mode = 'down_up'; args.merge_weights = True
        args.calibration = 'Author two-phase lifecycle on chronological training-tail windows (last 20%); no validation labels used for adaptation'
        model = proceed.Proceed(numeric.author_model, args).to(device)
        return ProceedBridge(model, horizon, args.label_len, data, lookback, settings), vars(args)
    if method == 'ADAPT-Z':
        args.lifecycle = 'Chronological joint adapter/backbone training with mature historical gradient; direct feedback overwrite; online adapter and projection-head updates'
    else:
        args.lifecycle = 'Frozen shared numerical backbone; masked offline adapter training; delayed test feedback for online variants'
    return ResidualOnline(numeric, method, settings, data, lookback, horizon).to(device), vars(args)

def evaluate_online(model, data, dataset, device, settings, checkpoint_callback=None, resume_loader=None, checkpoint_every=64, timings=None):
    from .worker import to_device, loss_ot
    model.reset_online()
    if isinstance(model, DSOFBridge):
        optimizer = None
    elif isinstance(model, ResidualOnline) and model.method == 'ADAPT-Z':
        optimizer = torch.optim.Adam([{'params': model.adaptation.parameters(), 'lr': .0003},
                                      {'params': model.numeric.author_model.projection.parameters(), 'lr': .00003}])
    else:
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=settings['online_learning_rate'])
    pending = deque()
    preds, truths, masks, origins, events = [], [], [], [], []
    all_windows = ForecastWindows(data, 'test', dataset.lookback, dataset.horizon, method=settings.get('active_method'))
    # Existing data only have terminal missing OT; no scored origin is skipped in the native stream.
    if dataset.indices.size < all_windows.indices.size:
        all_windows.indices = dataset.indices
    start_item = 0
    if resume_loader is not None:
        restored = resume_loader(optimizer)
        if restored is not None:
            from .checkpoint import device_tree
            pending = device_tree(restored['pending'], device)
            preds, truths, masks, origins, events = [restored[k] for k in ['preds','truths','masks','origins','events']]
            start_item = restored['cursor']
    for item in range(start_item, len(all_windows)):
        batch = host_batch(all_windows,[item])
        host = batch
        batch = to_device(batch, device)
        origin = int(batch['origin'][0])
        import time
        update_started = time.perf_counter()
        if isinstance(model, DSOFBridge):
            model.train()
            safe = dict(batch); safe['y'] = torch.zeros_like(batch['y']); safe['mask'] = torch.zeros_like(batch['mask'])
            result = model.stream_step(safe)
            prediction = torch.as_tensor(model.framework.studentFFN.get_future(result['pred']),
                                         device=device).detach()
            events.append({'origin': origin, 'native_er': True, 'native_td_k': [1],
                           'label_cutoff': origin, 'source_lifecycle': 'author losses with origin-indexed observed-label ER/TD bridge', **model.last_event})
            if timings is not None:
                timings.seconds['online_updates_host_wall'] = timings.seconds.get('online_updates_host_wall',0.) + model.last_update_host_seconds
        else:
            while pending and pending[0][0] + dataset.horizon <= origin:
                old_origin, old_batch, old_feedback = pending.popleft()
                model.train()
                saved = None
                if isinstance(model, ResidualOnline):
                    saved = (model.hg.clone(), model.ge.clone())
                    model.hg.copy_(old_feedback[0]); model.ge.copy_(old_feedback[1])
                if isinstance(model, ProceedBridge):
                    model.author_model.flag_online_learning = True
                    model.author_model.flag_current = False
                optimizer.zero_grad(set_to_none=True)
                loss, _ = loss_ot(model, old_batch)
                z_grad, text_grad = None, None
                if isinstance(model, ResidualOnline):
                    target, mask = old_batch['y'], old_batch['mask']
                    probe = observed_error(model.last_base,old_batch).square().mean()
                    z_grad = torch.autograd.grad(probe, model.last_feature, retain_graph=True)[0].mean(0)
                    if model.last_text is not None and model.last_text.requires_grad:
                        text_grad = torch.autograd.grad(loss, model.last_text, retain_graph=True, allow_unused=True)[0]
                    loss = loss + model.auxiliary_loss()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                if saved is not None:
                    model.hg.copy_(model.momentum * saved[0] + (1 - model.momentum) * z_grad.detach())
                    model.ge.copy_(text_grad.detach().mean(0) if text_grad is not None else saved[1])
                if isinstance(model, ProceedBridge):
                    model.after_training_batch(old_batch)
                    model.author_model.flag_online_learning = False
                events.append({'forecast_origin': origin, 'updated_origin': old_origin,
                               'last_label_row': old_origin + dataset.horizon,
                               'label_cutoff': origin, 'replay_origin_context': True,
                               'timing': 'mature labels updated before this origin forecast'})
            if timings is not None:
                timings.seconds['online_updates_host_wall'] = timings.seconds.get('online_updates_host_wall',0.) + time.perf_counter() - update_started
            snapshot = (model.hg.detach().clone(), model.ge.detach().clone()) if isinstance(model, ResidualOnline) else None
            model.eval()
            with torch.no_grad():
                safe = dict(batch); safe['y'] = torch.zeros_like(batch['y']); safe['mask'] = torch.zeros_like(batch['mask'])
                prediction = model(safe)
            pending.append((origin, batch, snapshot))
        preds.append(prediction.cpu().numpy()); truths.append(host['y'].numpy())
        masks.append(host['mask'].numpy()); origins.append(host['origin'].numpy())
        if timings is not None:
            name = 'online_step_wall_seconds'
            timings.seconds[name] = timings.seconds.get(name, 0.) + time.perf_counter() - update_started
            timings.counters['online_origins'] = timings.counters.get('online_origins', 0) + 1
        if checkpoint_callback is not None and ((item+1) % checkpoint_every == 0 or item+1 == len(all_windows)):
            checkpoint_callback(dict(cursor=item+1,pending=pending,preds=preds,truths=truths,
                                     masks=masks,origins=origins,events=events), optimizer)
    arrays = [np.concatenate(values) for values in [preds, truths, masks, origins]]
    return masked_errors(*arrays[:3]), arrays, events
