# Locked protocol and run plan

This file implements the user-approved 2026-09-28 revision plan.

## Model and data

Five real paired datasets: Economy, Energy, Security, SocialGood and Climate.
Chronological 60/20/20 split, frozen training-only standardized minimal
ITransformerZ (32 dimensions, 2 layers, 4 heads, feedforward 128); L=48 for
Energy, L=36 otherwise. Frozen complete BERT-base (12 layers, max 64 tokens)
embeddings are cached. The adapter/text projection are warmed for one training
stream pass, then online updates use test labels only after maturity. No
validation parameter updates. See DECISIONS.md for all boundary conventions.

## Timeline

For numerical row origin t, issuance is at the END of its observed aggregation
period, end_date(t), not the input CSV's date/start_date. Predict t+1..t+H using
rows t-L+1..t and chosen merged texts with proxy end_date <= end_date(t).
No intra-period availability is claimed. Save the originally issued forecast,
text record IDs, detached numerical/text feedback snapshots. AFTER predicting,
consume each queued origin s with s+H<=t once. Replay its own feature, text and
feedback snapshot with current adapter/projector weights. Compute the task loss
against observed dataset targets; masked targets are excluded. Extract text
feedback from that task loss and numerical feedback at zero residual, apply
adapter optimizer step, then set feedback for subsequent origins. No target
after t is read by optimization. The last H pending test origins are not flushed.

## Matrix

- B: five domains x H6 x numeric_add/text_add/full x five seeds = 75.
- C: Energy/SocialGood x H6 x eight ablations x five seeds = 80.
- C: Security x H6 x no_route_text_gradient x five seeds = 5.
- D: Economy/Energy x H12 x three primary methods x five seeds = 30.
- Seeds 2023..2027. Exactly 190 adaptation evaluations and 35 backbone checkpoints.
- First three B jobs (Economy/H6/2023) are the required sanity experiment.

## Reproducibility and limitations

Each result stores resolved config, source commit, selected checkpoint, predictions,
original valid-target masks, contexts and update events. Failed runs do not count.
The serial local queue persists after each transition and stops on non-OOM
failures. No batch/horizon/model reduction is allowed; only BERT preprocessing
batch size can decrease for memory. Original result tables are not merged.

ADAPT-Z-style denotes the numerical additive adapter, not original ADAPT-Z
reproduction. Text end_date is an availability proxy. No claim of universal SOTA,
actual publication-time verification, causal expert semantics or delayed-feedback
robustness follows from this limited experiment. Negative outcomes are retained.
