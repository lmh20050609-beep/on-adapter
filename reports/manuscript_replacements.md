# Manuscript replacement text

These are replacement paragraphs for a revised evaluation. They do not assert that the original submission already used this implementation. Numerical tables are supplied separately; no LaTeX manuscript source was available.

## Outline

1. Define period-end forecast origins and delayed supervision.
2. Describe fixed-backbone residual adaptation and immutable replay contexts.
3. Disclose contextual text encoding, missing-target masks and initialization.
4. Present controlled main comparisons, component limits and measured costs.
5. State unresolved original comparisons and availability limitations.

## Abstract: replacement empirical scope

We study text-assisted online adaptation of a frozen time-series forecaster under delayed multi-step supervision. At each forecast origin, the adapter forms a latent residual from numerical features, available text representations and historical gradient feedback. Mature targets update the adapter using the original forecasting context; the resulting feedback is used only for later forecasts. We evaluate the revised implementation on five paired datasets using five matched seeds and a common online protocol. At horizon six, iAdapter obtains the lowest mean MSE on 2 of five datasets and the lowest mean MAE on 1 of five datasets among the three controlled alternatives. Component ablations and measured inference/update costs characterize the method's benefits and limitations. Text availability is approximated by dataset record end dates rather than verified publication timestamps.

## Method: forecasting timeline and replay

A forecast origin is the end of the currently observed numerical aggregation period. The input contains the preceding L observations, and the target comprises the following H observations. The system first emits and records its forecast. It then consumes each previously unused origin whose entire target period has elapsed. Each pending record stores an origin identifier, the selected text record identifiers, the frozen numerical representation and detached numerical/textual feedback snapshots. During replay, the current adapter parameters operate on this saved origin context. A zero-residual probe provides numerical feedback, while the prediction loss provides sensitivity with respect to the projected text representation. After the adapter update, the numerical cache is updated with an exponential average and the latest text feedback is stored for subsequent origins. New feedback is not substituted into the replay that produced it. The final H pending origins are not flushed after the last evaluated forecast.

The complete expert computations, gate equations, eight-dimensional descriptor, optimizer chronology and precise ablation boundaries are provided in method_appendix.md for direct manuscript integration. In particular, the spectral operation and feature centering act along latent coordinates; their names do not establish temporal seasonal or fluctuation decomposition. The zero-initialized expert remains trainable, and all five experts are evaluated under soft routing.

## Related-work addition: Dynamic TMoE

Dynamic TMoE couples drift detection, heterogeneous experts and temporal memory routing, with expert-pool evolution restricted to training. Our revised setting instead evaluates text-assisted parameter updates during deployment, with a frozen numerical backbone and delayed target availability. The presence of heterogeneous experts or historical routing information alone is therefore not treated as a unique contribution. We include a homogeneous expert-pool control and distinguish structural biases from demonstrated semantic specialization. Reference: [Dynamic TMoE, Sections 3.6 and Appendix I](https://arxiv.org/html/2605.20678v1).

## Text and preprocessing

We encode the selected text with the complete frozen 12-layer BERT-base model, using an attention mask and masked mean pooling with a maximum of 64 tokens. Frozen representations are cached and shared across methods; a 64-dimensional projection remains trainable. Text is selected from the two most recent known merged records. If no informative text is available, the projected representation is set to zero. Record end dates serve as an availability proxy; they do not establish the true publication times of the underlying sources. Numerical inputs are forward-filled using observed history, with a fixed zero policy for leading missing values. Standardization is fitted on the training prefix only. Missing original targets are excluded from both the loss and evaluation by a shared validity mask.

## Experimental setup

We use Economy, Energy, Security, SocialGood and Climate, with chronological 60/20/20 splits. The backbone is the disclosed minimal ITransformerZ implementation with dimension 32, two encoder layers, four attention heads and feedforward dimension 128. The lookback is 48 for Energy and 36 otherwise. All adapters use the same tail insertion location and each seed-specific pretrained backbone. Backbone training uses Adam with learning rate 0.001, batch size 32 and at most 30 epochs; validation masked MSE selects the checkpoint with patience five. The adapter is warmed for one chronological training-stream pass. Its optimizer, feedback and queue are reset before testing, while its warmed parameters are retained. No validation-set parameter updates are performed. Online Adam uses learning rate 0.001, weight decay 0.0001 and gradient clipping at one. Seeds are 2023 through 2027. Experiments run serially on an NVIDIA RTX 4060 Laptop GPU with 8 GB memory.

## Results and component interpretation

Table R1 reports mean and sample standard deviation across five matched runs for numerical additive adaptation, additive text fusion and iAdapter. The numerical method is an ADAPT-Z-style controlled adapter and is not presented as a reproduction of the original method's complete recipe. At H=6, iAdapter has the lowest mean MSE on 2/5 datasets and lowest mean MAE on 1/5 datasets. Gains and degradations are both retained. Table R2 isolates text feedback, numerical feedback, the distribution descriptor, entropy regularization, expert heterogeneity and learned fusion. Removing routing text gradients lowers mean MSE on 3/3 evaluated domains; this evidence does not support a universal necessity claim for that component. Additional H=12 results cover Economy and Energy only. We make no statistical significance or universal robustness claim from these experiments.

Routing diagnostics show that 22 of 25 full-model H6 runs assign more than 99% average weight to one expert. This post-hoc descriptive threshold is not used for model selection. The representative mechanism plots therefore illustrate strongly concentrated routing rather than demonstrating adaptive temporal specialization.

The additive text control lowers mean MSE on 0/5 H6 domains and mean MAE on 1/5 relative to the numerical additive control. All three pairwise contrasts are reported with mean and sample standard deviation of matched-seed differences. Historical state-change groups use validation-only thresholds, and group comparisons retain the same forecast origins and valid targets within each seed. These descriptive contrasts test whether gains occur within measured regimes; they do not establish the semantic usefulness of a particular text type.

## Efficiency and limitations

Table R3 distinguishes trainable parameters, cached-text prediction latency, delayed-update latency and fresh-text end-to-end latency. Measurements use isolated copies with 50 warm-up and 200 measured iterations; medians and 95th percentiles are reported. GPU allocated memory, replay/gradient state and CPU text caches are reported separately. These results characterize this implementation on the measured device, not hardware-independent cost. The evaluation does not recover the original GPT4MTS/VoT comparison, and it does not establish true publication-time availability, causal expert specialization or generalization across the original ten-domain, multi-backbone setting.

The trainable adapter and text projection together contain roughly six times as many parameters as the small numerical backbone used here. Their small absolute size does not establish parameter efficiency relative to that backbone, and including the frozen BERT parameters in the denominator would obscure this comparison.

## Claim–evidence map and self-check

| Claim | Evidence | Status |
|---|---|---|
| Origin-aligned mature-label updates | protocol tests; updates.jsonl; validation.json | Supported for implemented protocol |
| Contextual BERT rather than embedding-only pooling | prepare.py; cache/direct checks | Supported |
| H6 performance counts | summary.csv; paired_differences.csv | Descriptive, fixed streams |
| Every module improves performance | component_verdicts.csv | Not asserted |
| Lightweight online deployment | efficiency.json | Limited to measured cost dimensions and hardware |
| Original multimodal SOTA comparison | Missing matched GPT4MTS/VoT runs | Unresolved |
| Publication-time leakage guarantee | Only end_date proxies available | Not asserted |

Self-check: scope and terminology match the implementation; comparisons share data and backbone; negative results are preserved; explanatory hypotheses are not stated as established mechanisms; missing original comparisons remain explicit.
