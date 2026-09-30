# Implementation-aligned method appendix

This text describes the evaluated revision at experiment commit f68027f. It is a replacement appendix, not a claim that the original submission used identical code. Module names indicate architectural choices; their predictive roles are evaluated rather than assumed.

## Outline and paragraph roles

1. Setting: define the observed window, frozen representation and prediction origin.
2. Design: specify text projection, feedback inputs, five experts and gate.
3. Algorithm: separate emitted forecasts, delayed replay and future feedback.
4. Reproducibility: state all fixed choices and exact ablation boundaries.
5. Interpretation: distinguish computational biases from demonstrated semantics.

## Frozen backbone and residual interface

At origin t, the observed standardized window is X_t of shape L × N, with N numerical variables. The origin is the end of the observed aggregation period. The frozen minimal ITransformerZ normalizes each variable by its window mean and population standard deviation with epsilon 1e-5, embeds each length-L variable history into D=32 features, and applies two encoder layers. Each encoder has four attention heads and a 128-wide feedforward network. The resulting representation Z_t has shape N × 32. The adapter adds a residual R_t of the same shape at the tail of the second encoder, before the final layer normalization and horizon projector. The frozen head then restores the saved window scale and mean. This window normalization is additional to the training-prefix standardization of the raw dataset.

The backbone is always in evaluation mode during both prediction and adaptation. Its parameters are frozen, while derivatives through the head with respect to the adapter residual remain available. A saved origin context contains Z_t, window mean and scale, descriptor, text cache reference, text IDs and detached feedback tensors. The original issued prediction is stored separately for scoring. Frozen features can be replayed because the backbone and input preprocessing do not change online.

## Text representation and shared expert inputs

For each origin, cleaned fact and preds fields from the two most recent known merged rows form the candidate text. Each field keeps at most three informative clauses using the documented cleaner with a 20-character minimum clause length. Aggregated text retains its last 1,200 characters, optionally dropping the first partial word, before BERT tokenization truncates to at most 64 tokens. Recorded row/channel IDs describe selected records before these truncations; an ID does not prove that every word of that record survives tokenization. The complete frozen 12-layer BERT-base encoder uses attention masks, and masked mean pooling includes all unmasked token positions, including special tokens. Its 768-dimensional output is cached. The online projection is Linear(768,96), ReLU, Linear(96,64), ReLU, Dropout(0.1), LayerNorm(64). A binary availability multiplier is applied after projection, giving exactly zero projected text when no informative text is available.

Let G_t ∈ R^(N×32) be the numerical feedback cache and q_t ∈ R^64 the latest text feedback. A learned elementwise multiplier A ∈ R^(N×32), initialized to one, forms g_t=A⊙G_t. Full-model text modulation is e'_t=e_t⊙[1+a⊙q_t/(||q_t||_2+1e-6)], where a∈R^64 is initialized to one. Separate affine projections and ReLU map Z_t, g_t and broadcast e'_t to 64-dimensional vectors f, u and v. The common expert input is h=Dropout_0.1(ReLU(W[f;u;v]+b)), with a 192-to-64 affine map. All affine maps have biases unless explicitly stated otherwise.

## Five expert computations

Each branch outputs an N × 32 residual. All five branches are evaluated; soft weights do not implement sparse execution.

| Branch | Exact computation | Interpretation boundary |
|---|---|---|
| Zero-initialized | Linear(64,32)(h), weight and bias initialized to zero | It starts as a zero correction but is trainable; it is not a permanent identity or no-op branch |
| Low-rank | Linear(64,32)(h), then bias-free Linear(32,2), then bias-free Linear(2,32), no intervening activation | A rank-two output bottleneck; temporal trend recovery is not established by its name |
| Spectral | rFFT of Z_t along its 32-dimensional latent axis; retain first 8 modes with learned complex gains 1+gain (gain initially zero), set remaining modes to zero, inverse rFFT to length 32; add Linear(64,32)(h) | Filtering is over latent coordinates, not the observed time axis; it is not a verified seasonal decomposition |
| Fluctuation | Subtract each variable's latent-coordinate mean from Z_t; Linear(32,64)+ReLU; concatenate numerical-feedback projection u; Linear(128,64), GELU, Linear(64,32) | Latent centering is not temporal high-pass filtering; this branch has no direct text/trunk input, although its mixture weight can depend on text |
| TextFiLM | Linear(64,128)(e'_t) gives gamma and beta; form f⊙(1+tanh(gamma))+beta; Linear(64,64), GELU, Linear(64,32) | A direct text-conditioned affine modulation; the common trunk also supplies text to other branches |

## Descriptor and soft gate

The eight-dimensional descriptor is computed from the observed standardized window before per-window backbone normalization. Define m and s as the mean and population standard deviation over all window values, I as the 90th minus 10th percentile, d as the mean of the last quarter of the window minus that of the first quarter, u as the population standard deviation of first temporal differences, and l as the mean of the last numerical row. The descriptor is [m, |m|, s, |s−1|, I, d, |d|, u+0.1|l|]. Quarter length is max(1,floor(L/4)). This descriptor is not a trained or externally validated drift detector.

The gate concatenates five 64-dimensional ReLU-projected inputs: local Z_t, the variable-average Z_t broadcast across variables, g_t, modulated text e'_t broadcast across variables, and LayerNorm(8) of the descriptor. Linear(320,64), ReLU and Linear(64,5) produce base logits. The gate adds alpha times the sum of a Linear(32,5) projection of variable-average g_t and a Linear(64,5) projection of q_t. Alpha is learned and initialized to 0.1. Softmax with temperature 1 gives the five weights. The final residual is their weighted sum, separately for each variable. Effective expert count is 1/sum_k(w_k²), averaged over origins and variables for reporting; it is a concentration diagnostic and does not reduce compute.

The replay objective is masked MSE plus 0.02 times mean gate entropy, where entropy is −sum_k w_k log(w_k+1e-9). Minimizing this positive entropy term encourages concentration. It is not a load-balancing penalty. The numerical probe and text feedback are differentiated from prediction loss only, excluding the entropy term. Observed concentration must therefore be reported rather than interpreted as balanced specialization.

## Algorithm: predict, then consume mature origins

The pseudocode below applies independently to the training warm-up stream and test stream. Both streams start with an empty queue and zero feedback. Test retains the warmed adapter/projection parameters but resets the optimizer. H is the number of future rows predicted, and origin s matures at row s+H.

```text
Initialize G = zeros(N,32), q = zeros(64), queue = empty
Initialize a fresh Adam optimizer over adapter and text projection
For each forecast origin t in chronological order:
    Set adapter/projection to evaluation mode
    Obtain frozen context Z_t, normalization statistics and historical descriptor
    Reproject the selected frozen BERT cache with current projection parameters
    Emit forecast for t+1,...,t+H using current G and q
    Save this original forecast for scoring; never replace it with a replay forecast
    Enqueue (t, t+H, context, selected text IDs/cache, clone(G), clone(q))
    For each queued record s with s+H <= t, in origin order:
        Remove s from the queue exactly once
        If all its original target values are missing: log skip and continue
        Set adapter/projection to training mode
        Reproject s's frozen BERT cache with current projection parameters
        Replay s with its saved G_s and q_s; compute masked prediction loss L_s
        q_new = d L_s / d e_s, averaged over the replay batch (batch size 1)
        At zero residual r=0, G_new = d MSE(head(Z_s+r), y_s) / d r
        Backpropagate L_s + entropy penalty into adapter/projection only
        Clip trainable gradient norm to 1; take one Adam step
        G = 0.9*G + 0.1*G_new; q = q_new
        New feedback is available only for later forecast origins
Stop at the last forecast origin; do not flush the final H pending records
Score only the original emitted forecasts on observed target elements
```

Feedback-disabled variants omit the corresponding derivative and cache refresh. Prediction is deterministic with dropout disabled; replay activates dropout 0.1. The task-loss gradient with respect to projected text and the optimizer backward pass share the same replay graph. The separate zero-residual numerical probe uses the frozen head; it does not evaluate an updated replay prediction for scoring. No prediction at t can use the update performed after that prediction, even if its labels mature at t.

For example, with H=6 and first test origin a, the prediction at a+6 is emitted before the update of origin a. The first forecast that can use that update is a+7. Targets are masked from original CSV observations; missing values are never filled to create supervised truth.

## Exact controls and ablations

| Variant | Changed paths |
|---|---|
| numeric_add | No text encoder cache read and no text projection; feature and numerical-feedback affine projections are added, followed by two 64-wide ReLU/dropout stages and a 32-output linear layer |
| text_add | Same additive structure with a projected-text affine term; no text feedback in any path |
| no_text | Full five-expert structure retained; projected text and text feedback zeroed; learned constant FiLM biases may remain but carry no text information |
| no_text_gradient | No text-feedback extraction/cache update, no expert text modulation, no text-gradient logit term |
| no_route_text_gradient | Expert-side modulated text retained; gate sees unmodulated e and no direct q-to-logit term; text gradients can still influence current parameters through training |
| no_numeric_gradient | No numerical-feedback extraction/cache update; G input zero; no numerical-gradient logit term; fixed affine biases can remain |
| no_distribution | Zero only the descriptor's projected gate input; experts and other gate inputs retained |
| no_decisiveness | Entropy coefficient zero; learned gate retained |
| homogeneous | Replace five branches by independent Linear(64,w), GELU, Linear(w,32) MLPs, all taking h; select integer w minimizing expert-parameter difference and report actual counts; gate unchanged |
| uniform | Five weights fixed to 1/5; gate regularization disabled; expert computations unchanged |

The homogeneous control holds expert count and approximately total expert parameters fixed, but also makes input processing identical across experts. It does not isolate each expert's individual necessity. The approved local matrix omits leave-one-expert-out runs, so individual expert causal roles remain unverified.

## Claim–evidence map and local writing check

| Claim | Source | Status |
|---|---|---|
| Strict forecast-before-update timeline and original replay feedback | engine.py, test_protocol.py, test_future_context.py, original_csv_audit.json | Supported within the declared proxy timeline |
| Concrete expert/gate computations | models.py, legacy_core.py | Supported by implementation; semantic names are not empirical findings |
| Contextual frozen text plus current projection on replay | prepare.py, models.py, engine.py | Supported |
| Concentration diagnostic and parameter/latency costs | routing_summary.csv, efficiency.json | Measured, configuration/device-specific |
| Verified true publication-time text availability | No upstream publication records | Not asserted |
| Necessity or semantic specialization of all five experts | No individual intervention study | Not asserted |

Reverse outline: setting → representation → branch computation → gate/loss → timeline → exact controls. Dimensions, masks, feedback chronology and dropout modes match the executable code. Motivations are limited to inspectable design choices; performance advantages require the reported experiments. Remaining major rejection risks are weak gains, missing matched external baselines and unverified publication-time provenance; clear exposition does not remove those evidence gaps.
