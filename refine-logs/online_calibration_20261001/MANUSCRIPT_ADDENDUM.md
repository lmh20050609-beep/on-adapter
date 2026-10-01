# Manuscript addendum: delayed-feedback forecast calibration (exploratory)

We tested whether a low-dimensional forecast combiner can replace complex routing while preserving text-related gains. This experiment uses Energy and SocialGood, OT, H=6, and optimization seeds 2023–2025. The chronological split is 60/10/5/5/20: training and normalization, existing backbone checkpoint selection, frozen-branch calibration, combiner-rule selection, and assessment. Original periods had already been inspected in research; this repartition is exploratory, not untouched confirmation. All full-window targets and their maturity clocks remain within their respective stage; validation state is discarded. Numerical and textual branch predictions are issued before once-only full-H mature updates. The combiner consumes issued forecasts rather than replayed updated forecasts. Existing training/warm weights are reused; the specified online branches undergo real parameter updates.

The constrained least-squares and Hedge combiners are established methods used as interpretable controls, not claimed as new algorithms. Rule selection locks rho=1, kappa=10 for per-horizon/shared LS and eta=10 for both Hedge controls. The predeclared primary scenario is numerical-online plus LLM5-online. Secondary scenarios are retained separately. Identical branch ledgers give 216 combination sequences, not 216 neural training runs.

| dataset | method | n | mse_mean | mse_sd | mae_mean | mae_sd |
| --- | --- | --- | --- | --- | --- | --- |
| Energy | LLM5_online | 3 | 0.092249 | 0.010214 | 0.201771 | 0.005464 |
| Energy | LLM5_static | 3 | 0.068180 | 0.001194 | 0.181446 | 0.002928 |
| Energy | full_online | 3 | 0.167781 | 0.097491 | 0.262765 | 0.056724 |
| Energy | numeric_online | 3 | 0.091155 | 0.006814 | 0.201177 | 0.005236 |
| Energy | numeric_static | 3 | 0.065321 | 0.002251 | 0.175329 | 0.001841 |
| Energy | raw5_online | 3 | 0.099331 | 0.017942 | 0.205888 | 0.015456 |
| Energy | raw5_static | 3 | 0.071797 | 0.009518 | 0.187617 | 0.013987 |
| SocialGood | LLM5_online | 3 | 0.841632 | 0.018035 | 0.399678 | 0.012706 |
| SocialGood | LLM5_static | 3 | 0.829610 | 0.016419 | 0.395310 | 0.008900 |
| SocialGood | full_online | 3 | 0.848562 | 0.017204 | 0.409600 | 0.016207 |
| SocialGood | numeric_online | 3 | 0.848059 | 0.023594 | 0.401397 | 0.013070 |
| SocialGood | numeric_static | 3 | 0.819692 | 0.012413 | 0.392895 | 0.009778 |
| SocialGood | raw5_online | 3 | 0.847003 | 0.018096 | 0.402736 | 0.011315 |
| SocialGood | raw5_static | 3 | 0.826832 | 0.015113 | 0.394932 | 0.007819 |

| dataset | representation | numeric_mode | text_mode | family | n | mse_mean | mse_sd | mae_mean | mae_sd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Energy | LLM5 | online | online | endpoint_hedge | 3 | 0.093588 | 0.009282 | 0.202708 | 0.006162 |
| Energy | LLM5 | online | online | equal | 3 | 0.091346 | 0.008662 | 0.201121 | 0.005524 |
| Energy | LLM5 | online | online | fixed | 3 | 0.092300 | 0.008673 | 0.201339 | 0.005442 |
| Energy | LLM5 | online | online | ls | 3 | 0.093708 | 0.009264 | 0.202854 | 0.006263 |
| Energy | LLM5 | online | online | mixed_hedge | 3 | 0.093154 | 0.008873 | 0.202462 | 0.005969 |
| Energy | LLM5 | online | online | shared_ls | 3 | 0.093897 | 0.009701 | 0.203008 | 0.006783 |
| SocialGood | LLM5 | online | online | endpoint_hedge | 3 | 0.844398 | 0.018923 | 0.400729 | 0.012518 |
| SocialGood | LLM5 | online | online | equal | 3 | 0.844306 | 0.020036 | 0.399841 | 0.012449 |
| SocialGood | LLM5 | online | online | fixed | 3 | 0.844470 | 0.020903 | 0.400311 | 0.013078 |
| SocialGood | LLM5 | online | online | ls | 3 | 0.844964 | 0.018724 | 0.401324 | 0.012315 |
| SocialGood | LLM5 | online | online | mixed_hedge | 3 | 0.844279 | 0.019322 | 0.400391 | 0.012494 |
| SocialGood | LLM5 | online | online | shared_ls | 3 | 0.845274 | 0.020522 | 0.400765 | 0.012389 |

Relative to numerical-online, primary LS reduces SocialGood mean MSE by 0.365%, with one of three seeds improving, and increases Energy MSE by 2.800%, with none improving. It does not improve the mean MSE over equal weighting, fixed calibration or mixed-candidate Hedge on either domain. Secondary numerical-online plus LLM5-static LS reduces MSE by 8.699% and 1.455%, respectively, but underperforms both the static textual branch and the stronger numerical-static control. Static/static LS improves over equal weighting by 0.662% and 0.0434%, yet remains worse than numerical-static. Small gains are reported without an arbitrary minimum-effect threshold and with their paired seed outcomes.

The LLM5-online branch reduces mean MSE relative to capacity-matched raw5-online by 7.130% in Energy (2/3 seeds) and 0.634% in SocialGood (3/3). This is a conditional representation comparison, not evidence of semantic causality. Current-input zeroing probes often improve the original branch, while donor replacements are not verified semantic mismatches. Both probes retain the original learning trajectory and therefore do not evaluate an alternative zero-text learning policy. Numerical-online also underperforms numerical-static by 39.550% and 3.461%; the distinction between text contribution and recovery from online-update deterioration is essential.

The dual adapters have 32,128 parameters compared with 166,832 in the full reference, a reduction of 80.742%. Isolated repeated-input microbenchmarks show cheaper dual-online updates but slower dual-online predictions. The 192-byte gate array payload is not total memory or replay-cache cost. Offline generation of the 470 new fixed inputs takes 84.875 seconds; 1,538 inputs are reused, so this is not full-corpus generation cost.

| dataset | policy | adapter_parameters | prediction_median_ms | prediction_p95_ms | update_median_ms | update_p95_ms |
| --- | --- | --- | --- | --- | --- | --- |
| Energy | full_online | 166832 | 7.488122 | 11.010197 | 21.688836 | 25.888049 |
| Energy | numeric_online_text_online | 32128 | 10.311347 | 16.949801 | 18.717156 | 22.736515 |
| Energy | numeric_online_text_static | 32128 | 8.777867 | 14.527205 | 9.578796 | 12.384347 |
| Energy | numeric_only | 10496 | 3.081689 | 8.268974 | 5.777310 | 9.388045 |
| SocialGood | full_online | 166832 | 8.743700 | 10.806350 | 23.991403 | 29.772691 |
| SocialGood | numeric_online_text_online | 32128 | 9.468509 | 14.386642 | 16.339240 | 20.565151 |
| SocialGood | numeric_online_text_static | 32128 | 8.200586 | 14.502289 | 7.868939 | 10.224369 |
| SocialGood | numeric_only | 10496 | 5.981065 | 7.452177 | 7.981045 | 10.804897 |

These results motivate a frozen strong numerical base and a bounded test of residual-exclusive textual information, rather than additional routing complexity. Neither this study nor the native VoT smoke restores the original Table 2 superiority claim. Publication time, full VoT priors/comparison, source provenance and independent new-period confirmation remain open.
