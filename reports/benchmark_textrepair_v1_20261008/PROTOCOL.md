# Frozen protocol: nine-domain V1 minimal text-repair benchmark

Frozen on 2026-10-08, before any result of this round was produced. What is frozen: data identity, code identity,
the execution matrix, reuse rules and statistical definitions. **This freeze happened after the old test set had been observed repeatedly,
so it cannot retroactively be called an independent confirmation or a preregistration of unseen results.**

## 1. Data

- Numeric data, splits, targets, masks and feedback rules are exactly the same as in the original benchmark;
  V1 changes only the `fact` text column.
- Text version: **V1 only** (masks only individual entries confirmed to postdate the time they describe; C/D entries are kept and disclosed).
  The existing three-domain diagnostics of V0/V2 are kept but are not the data version of the main table.
- Domains and windows (no L36 sweep, no domain dropped):

| Domain | Period | L | H |
|---|---|---|---|
| Environment | daily | 96 | 48/96/192/336 |
| Energy | weekly | 24 | 12/24/36/48 |
| Health_US | weekly | 24 | 12/24/36/48 |
| Traffic / SocialGood / Security / Economy / Climate / Agriculture | monthly | 8 | 6/8/10/12 |

- Seeds: 2025, 2026, 2027. Size 9×4×3×7 = 756.

## 2. Text read paths (read-only inventory)

| Method | Reads `fact` | Input path |
|---|---|---|
| On-Adapter | yes | `self.encoder = data.prepared_text` in `online.py`, `pooled(origin)`; the text mapped to row i is the concatenation of visible rows (i-1, i), 64 tokens, mean pooling |
| GPT4MTS | yes | `multimodal.GPTForecast`, `text.gpt(origin, lookback)` takes the history before origin, 512 tokens, cls |
| MM-TSFlib | yes | `multimodal.MMD`, `text.tokens(origin)`, 6 GPT2 layers, 256 tokens |
| iTransformer / ADAPT-Z / DSOF / Proceed | no | The code path does not access `data.texts` / `prepared_text` |

No method reads `preds` or any `Final_*` field.

## 3. Encoding

Reuses the sealed independent encoding v2: each non-empty full text gets its own forward pass, so the actual model
batch size is always 1; weights, number of layers, token limits, pooling, FP32 encoding / FP16 cache
are unchanged. The cache identity includes `data_sha256` and `text_version` and is isolated from the historical cache directories.

## 4. Training and evaluation

- The effective configuration of each run is the resolved output of `resolve_settings`, written to
  each run's `config.json`; the campaign template is not used in its place.
  Rules: rows<500 → d_model=32, dropout=0.2, batch=8, epochs=50;
  On-Adapter/ADAPT-Z → epochs=3; GPT4MTS → epochs=10, batch=4;
  MM-TSFlib → batch=8. patience=3, learning rate 0.0003 (GPT4MTS 0.001).
- Preprocessing is fitted on the training segment (mean/std from the training segment only), models are selected on the validation segment (strict validation MSE),
  and nothing is tuned on test.
- Feedback rule for online methods: `old_origin + H <= current_origin` (same as the original numeric protocol).
- No method's training budget is reduced, and no failed configuration is omitted.

## 5. Reuse rules

- Only the completed cells of the 4 **text-independent** methods (iTransformer, ADAPT-Z, DSOF, Proceed)
  among the historical 756 are reused, 432 cells in total. For each one,
  `benchmark_textrepair_v1/legacy_numeric_reuse_manifest.json` pins the
  result / config / best.pt / predictions hashes, and
  `reuse_verification.json` and `numeric_reuse_equivalence.json` prove compatibility.
- All 324 cells of the 3 text-reading methods are retrained in this package from initial states with identical definitions.
  No score from an older text version is carried over.

## 6. Scoring

- Main space: observed target elements of all selected numeric variables, in the space standardized by the training-set mean/std.
- MAE / MSE are scored over valid target elements.
- CVaR_q: over the distribution of per-origin MAE of observed elements, the worst q probability mass under a uniform distribution over origins,
  with a fractional weight for the boundary origin; independently checked against the variational definition.
- **Main statistic:** within each domain/H, first average the risk over seeds, then divide by the CVaR90 of the shared iTransformer backbone
  for the same domain/H/seed to get a ratio; take the equally weighted geometric mean over the 36 domain/H tasks.
- Also reported: mean rank, MAE, MSE, CVaR80/95, per-domain/H results.
- Pre-declared exploratory groupings: geometric mean over the six monthly domains, all 7 frequency combinations, frequency-equal-weight sensitivity.
- If a zero value makes a geometric mean undefined, this is reported explicitly; no epsilon is added after the fact.

## 7. Limitations that must be kept

- Not extended beyond three seeds; no additional paired pilot for SocialGood was added as a prerequisite of the main experiment.
- Source decisions cover only screening hits; a miss does not mean certified A; C/D are kept but not verified.
- The numeric data still use the original `end_date` availability proxy and the retrospective target-mask origin grid; this does not prove full causality
  in the sense of real publication times and historical revisions.
- This protocol was frozen after the old test set had been observed, so it is not an independent confirmation.
