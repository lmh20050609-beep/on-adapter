# Declared implementation choices

These choices complete the user's approved plan before experiments begin.

- Source: outer IGSoftMoEFusionAdapter and minimal ITransformerZ; base source has
  no Git metadata. The inner source is a different top-k method and is not merged.
- Five domains: Economy, Energy, Security, SocialGood, Climate. All original
  numeric target columns except date/index/text and prior_history aggregates are
  forecast; missingness masks are retained as metadata, not added as new targets.
- Input imputation is forward-only. For leading missing values use zero in raw
  units, a fixed training-time policy that cannot depend on future rows. Scaler
  uses the imputed training prefix only. Future-invariance checks are conditional
  on a fixed training-fitted scaler; training data legitimately determine it.
- Targets are unfilled original values transformed by the same training scaler;
  invalid targets are zero-filled for tensor storage only and masked in all losses.
- Origin time is the END of the observed numerical aggregation period (end_date),
  not the original date/start_date column. No intra-period forecasting is claimed.
- Text: merged rows are sorted by end_date. Select informative fact and preds
  from the last two known rows, clean with existing cleaning functions, truncate
  to 1200 characters using the original left truncation, max 64 BERT tokens.
  Provenance IDs identify merged row/channel; end_date is only an availability proxy.
- Full BERT-base 12-layer attention-mask-aware contextual encoding, float32,
  frozen/eval, masked mean pooling. CPU cache is reused; no text model is resident
  during online experiments. Missing text uses zero projected representation.
- Backbone: d_model=32, layers=2, heads=4, feedforward=128, dropout=0.1. Energy L=48,
  others L=36. No automatic window/model shrinking. Train 30 epochs max, patience=5,
  batch=32 (retain final batch), Adam lr=1e-3; no backbone weight decay (unspecified
  by plan). Select best validation masked MSE. Store/check checkpoint configuration.
- Adapter warm-up: one chronological pass on training forecast origins; no
  validation parameter updates. Reset optimizer, feedback and pending queue for
  test; retain warmed adapter/projector weights. Stop warm-up at last training
  forecast origin; final H pending origins are not flushed after that point.
- Online and warm-up: predict before update, one origin per step, then consume
  eligible pending origins exactly once. Do not flush pending test targets after
  the final prediction origin. Replay uses the recorded full feedback snapshot.
- Prediction uses eval mode (no dropout); replay training uses dropout=0.1. Text
  and parameter gradients are extracted from the SAME replay loss/graph; numerical
  feedback uses a separate zero-residual probe. New feedback applies only later.
- Feedback with zero valid target count causes no optimizer or feedback update;
  the mature record is consumed once and logged as skipped_missing_target.
- Full model entropy coefficient=.02, temperature=1, alpha initialized .1 and
  learnable, EMA=.9, hidden=64, text_dim=64, rank=2, spectral_modes=8. Adam adaptation
  lr=1e-3, weight_decay=1e-4, gradient clipping=1. Uniform routing has no gate loss.
- no_text retains all five branches; zero text may leave constant learned FiLM
  bias, which uses no text information. Homogeneous experts take common trunk h
  and use independent two-layer GELU MLPs; width minimizes absolute expert-only
  parameter-count difference over positive integer widths (ties use smaller).
- Main additive text baseline has no text gradients; numeric baseline never
  loads text representations. It is labelled ADAPT-Z-style, not a verified
  reproduction of the original ADAPT-Z training recipe.
- Seeds=2023..2027. Models share each dataset/horizon/seed backbone. Separate
  deterministic seed streams initialize backbone, adapter and training shuffle.
- Metrics are normalized-scale MSE/MAE over original valid target elements,
  including overlapping forecast horizons. Seeds, not windows, are repeat units.
- Local serial GPU runs only; no other process is stopped. BERT encoding batch
  may reduce on OOM; online batch/model/seed settings never silently shrink.
- Statement of success means complete reproducible evidence, not positive gains.
