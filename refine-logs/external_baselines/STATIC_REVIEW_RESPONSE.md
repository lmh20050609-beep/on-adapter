# Static-pilot response supplement

## ECmL F2; LCo1 W3/P2/P5 — matched task and reported scope

We completed an initial common-task static pilot on Economy (OT-only input and target, lookback 36, horizon 6, seed 2023). Numerical adaptation, full iAdapter after training-stream warm-up, and the author GPT4MTS model are evaluated on exactly the same 37 forecast origins and 222 valid target elements. All development predictions use fixed parameters; the internal adapters' feedback states are reset to zero and are not refreshed from development labels. Targets, masks, scaling, and scores were independently reconstructed from the original CSV.

The observed MSE/MAE values are 0.080086/0.220759 for numerical adaptation, 0.116353/0.284588 for static iAdapter, and 0.078934/0.226644 for GPT4MTS. Thus this pilot does not support iAdapter superiority; the lowest MSE and MAE belong to different controls. With one seed, we report neither seed variance nor a significance claim. This static single-variable task is separate from the earlier multivariate online evaluation and does not recover the submitted Table 2.

The two internal methods share one OT backbone and fixed insertion location. GPT4MTS retains the author model's native architecture without adapter-position selection. The same training/selection/development boundaries and text-availability proxy are enforced, but text consumption differs: iAdapter uses its current pooled representation, while GPT4MTS uses a historical sequence with CLS/512 encoding. Training phases also differ because the internal adapters receive a delayed-label warm-up pass. We disclose these distinctions rather than call the inputs or compute budgets identical. Consequently, any eventual whole-system comparison cannot by itself isolate expert structure or text-history effects.

The pilot establishes that the comparison pipeline is executable and traceable. It is not sufficient evidence for cross-seed robustness or broad method rankings. Configuration-locked repetitions are the next stage; the unverified VoT priors remain excluded until their origin/time and generation provenance can be established.

Evidence: `STATIC_PILOT_RESULTS.md`, `static_pilot_evidence/summary.csv`, `validation.json`, per-method configs/selection histories/predictions. Experiment source commit: `6b93caf96fdb5f680fe026d182529af664cae904`.
