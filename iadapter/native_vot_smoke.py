"""Real-target VoT integration smoke; never a formal baseline performance result."""
import argparse
import gc
import json
import time
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .config import ROOT
from .data import load_data
from .native_vot import SOURCE, WEIGHTS, COMMIT, build, build_helpers, native_config
from .audit import original_reference, require, same
from .util import seed_all, masked_loss, revision, write_json

OUTPUT = ROOT / 'experiments/native_vot_smoke_20261001'

def main(output=OUTPUT):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'status.json', {'status': 'running'})
    try:
        require(not revision()['dirty'], 'Commit smoke source before GPU execution')
        torch.set_num_threads(1)
        seed_all(2025)
        device, dataset, L, H = 'cuda', 'SocialGood', 8, 6
        data = load_data(dataset)
        k = data.columns.index('OT')
        ref = original_reference(dataset)
        rk = ref['columns'].index('OT')
        author = pd.read_csv(SOURCE / f'data/{dataset}.csv')
        # The released loader indexes priors by numeric-window start position.
        # This proves loader convention, not the original generation provenance.
        require(len(author) <= len(data.x), 'Author CSV exceeds original source')
        require(pd.to_datetime(author.end_date).astype(str).tolist() ==
                [str(pd.Timestamp(d).date()) for d in data.dates[:len(author)]], 'Author prefix row order differs')
        original = pd.read_csv(ref['source'])
        original = original.iloc[np.argsort(pd.to_datetime(original.end_date).values, kind='stable')]
        same(pd.to_numeric(author.OT).to_numpy(dtype=np.float32),
             pd.to_numeric(original.OT).to_numpy(dtype=np.float32)[:len(author)], 'Author/source prefix OT')
        require(not ref['valid'][len(author):, rk].any(), 'Author omitted observed targets')
        origins = [L - 1, L]
        require(all(t + H < data.train_end and t + H < len(author) for t in origins), 'Smoke must use shared training targets')
        x = torch.from_numpy(np.stack([data.x[t-L+1:t+1, k:k+1] for t in origins])).to(device)
        y_np = np.stack([ref['target'][t+1:t+H+1, rk:rk+1] for t in origins])
        mask_np = np.stack([ref['valid'][t+1:t+H+1, rk:rk+1] for t in origins])
        same(y_np, np.stack([data.target[t+1:t+H+1, k:k+1] for t in origins]), 'Real CSV targets')
        same(mask_np, np.stack([data.valid[t+1:t+H+1, k:k+1] for t in origins]), 'Real CSV masks', exact=True)
        y, mask = torch.from_numpy(y_np).to(device), torch.from_numpy(mask_np).to(device)
        require(bool(mask.any()), 'No real target for smoke')
        prior_path = SOURCE / 'data/stat_predictions_rag/SocialGood/SocialGood_length_12_predictions.json'
        released = json.loads(prior_path.read_text(encoding='utf-8'))
        raw_priors, records = [], []
        for t in origins:
            prior_id = t - L + 1
            values = np.asarray(released[str(prior_id)]['Prediction'], dtype=np.float32)
            require(values.ndim == 1 and len(values) >= H and bool(np.isfinite(values).all()), 'Invalid released prior')
            raw_priors.append((values[:H] - data.mean[k]) / data.scale[k])
            records.append(dict(origin=t, window_start=prior_id, released_prior_id=prior_id,
                                original_length=len(values), used_length=H, truncated=len(values)>H,
                                provenance='Author loader convention only; generation availability unverified'))
        prior = torch.from_numpy(np.stack(raw_priors)).float().to(device)
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        model = build(L, H, device)
        helper = build_helpers(model, device)
        gpt_before = {n: p.detach().cpu().clone() for n, p in helper.llm_model.named_parameters()}
        with torch.no_grad():
            prompt = helper.get_text_emb(x.clone(), None).detach()
        require(prompt.shape == (2, 192, 768) and bool(torch.isfinite(prompt).all()), 'Endogenous text shape/finite')
        require(len(helper.llm_model.h) == 6 and all(not p.requires_grad and p.grad is None and
                torch.equal(p.detach().cpu(), gpt_before[n]) for n, p in helper.llm_model.named_parameters()), 'Frozen GPT2 changed')
        del gpt_before
        helper.llm_model = None  # retained prompts are frozen; free encoder before updates
        gc.collect()
        torch.cuda.empty_cache()
        initial = {}
        model.eval()
        with torch.no_grad():
            for batch in [1, 2]:
                for stage in [1, 0, 2]:
                    result = model(x[:batch].clone(), None, None, None, prompt[:batch], forecast=stage)
                    if stage == 0:
                        features, aligned = result
                        require(features.shape[0] == batch and isinstance(aligned, tuple) and
                                len(aligned) == 2 and all(z.shape == features.shape and bool(torch.isfinite(z).all())
                                for z in aligned) and bool(torch.isfinite(features).all()), 'Alignment shape/finite')
                    else:
                        require(result.shape == (batch, H, 1) and bool(torch.isfinite(result).all()), 'Prediction shape/finite')
                        if batch == 2:
                            initial[str(stage)] = result.cpu().numpy()
            fused = helper.correction(model(x.clone(), None, None, None, prompt, forecast=2), prior, [0, 1])
            require(fused.shape == y.shape and bool(torch.isfinite(fused).all()), 'Frequency fusion shape/finite')
        stages = []
        for stage in [1, 0, 2]:
            if stage == 0:
                excluded = set(model.patch_embedding.parameters()) | set(model.encoder.parameters())
                parameters = [p for p in model.parameters() if p not in excluded]
                unchanged = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p in excluded}
                probe = model.ts_text_attn_trend.query_proj.weight
                optimizer = torch.optim.Adam(parameters, lr=.0001)
            else:
                if stage == 2:
                    with torch.no_grad(): model.decomp_w.fill_(.5)
                parameters = list(model.parameters())
                groups = [{'params': [p], 'lr': .0001 if p is model.decomp_w else (.00001 if stage == 1 else .001)}
                          for p in parameters]
                optimizer = torch.optim.Adam(groups)
                unchanged = {}
                probe = model.head.linear.weight if stage == 1 else model.decomp_w
            before = probe.detach().clone()
            model.train()
            model.zero_grad(set_to_none=True)
            result = model(x.clone(), None, None, None, prompt, forecast=stage)
            loss = helper.contrastiveLoss_tr_sea(result[0], result[1], model.t) if stage == 0 else masked_loss(
                helper.correction(result, prior, [0, 1]) if stage == 2 else result, y, mask)
            require(bool(torch.isfinite(loss)), 'Nonfinite smoke loss')
            loss.backward()
            require(probe.grad is not None and bool(torch.isfinite(probe.grad).all()), 'Missing/nonfinite probe gradient')
            require(all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None), 'Nonfinite gradient')
            optimizer.step()
            require(not torch.equal(before, probe), 'Smoke update did not change probe')
            require(all(torch.equal(p.detach().cpu(), unchanged[n]) for n, p in model.named_parameters()
                        if n in unchanged), 'Excluded alignment backbone changed')
            stages.append(dict(stage=stage, loss=float(loss.detach()), update_checked=True,
                               target='Original CSV OT' if stage != 0 else 'Author alignment objective, no future target'))
            del optimizer, before, unchanged
        torch.cuda.synchronize()
        np.savez_compressed(output / 'smoke_arrays.npz', origins=np.asarray(origins),
                            numeric_before=initial['1'], aligned_before=initial['2'],
                            released_prior_used=prior.cpu().numpy(), targets=y_np, valid_mask=mask_np)
        proof = dict(status='passed', purpose='integration only; not a benchmark accuracy evaluation',
                     dataset=dataset, seed=2025, origins=origins, config=vars(native_config(L, H)),
                     upstream_commit=COMMIT, weights_source=json.loads((WEIGHTS/'SOURCE.json').read_text()),
                     code=revision(), stages=stages, frozen_gpt2_unchanged=True, gpt2_layers=6,
                     text_features='Exact author numeric-statistics prompt; frozen GPT2,192tokens',
                     released_prior_records=records, prior_provenance_verified=False,
                     original_rows=len(data.x), author_rows=len(author), original_train_end=data.train_end,
                     author_data_policy='Exact original numeric/date prefix; omitted tail OT missing; original900rows/scaler/mask retained',
                     row_order_checked=True, original_target_mask_checked=True,
                     batch1_and_batch2_shapes_checked=True, excluded_alignment_parameters_unchanged=True,
                     author_math_modified=False, new_benchmark_scores=False,
                     seconds=time.perf_counter()-start, peak_allocated_bytes=torch.cuda.max_memory_allocated())
        write_json(output/'validation.json', proof)
        write_json(output/'status.json', {'status': 'completed'})
        print('NATIVE_VOT_SMOKE_PASSED three_author_stages; no_benchmark_score', flush=True)
    except Exception:
        write_json(output/'status.json', dict(status='failed', error=traceback.format_exc()))
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUTPUT)
    main(parser.parse_args().output)
