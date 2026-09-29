# Reviewer-response draft — five-seed development follow-up

Scope: new exploratory evidence on Energy and SocialGood, H=6. This draft supplements the original 190-run revision package; it neither recovers the submitted Table 2 nor presents the development interval as an independent test set. The candidate table contains five seeds (2023–2027); the routing-gradient × entropy factorial retains three seeds.

## ECmL F1 and LCo1 W1/P3: consistency and uncertainty

We agree that the evidence does not justify “consistently and significantly.” We report all seed-level outcomes, mean errors, sample standard deviations, and matched differences, including negative outcomes. In the new five-seed development comparison, removing the decisiveness penalty reduces mean MSE relative to full iAdapter by 5.74% on Energy and 1.03% on SocialGood. However, the improvement occurs in only 3/5 Energy seeds, versus 5/5 SocialGood seeds. In particular, both additional Energy seeds reverse the favorable direction observed in the initial three seeds. These results support a qualified component trade-off rather than a universal improvement claim.

The repeats characterize optimization variability on fixed streams; overlapping forecast windows are not independent repetitions. We have not conducted a confirmatory paired significance test and do not claim statistical significance. The two additional seeds do not undo candidate selection from the initial three-seed development results or establish generalization to unseen data.

Evidence: `five_seed_evidence/five_seed_runs.csv`, `five_seed_paired_summary.csv`, `cohort_paired_summary.csv`.

## ECmL F2 and LCo1 W3/P2: fair comparisons and baseline identity

The original comparison mixed insertion-position variation with run-to-run variation, and we cannot justify those as interchangeable. The new internal comparison fixes the insertion location, seed-specific frozen backbone, numerical/text inputs, delayed-update executor, forecast origins, and scoring masks. Its error bars measure seed variability only. It therefore addresses internal comparison fairness but does not repair the missing original Table 2 provenance.

The supplied package contains a GPT4MTS adaptation and a VoT-inspired local proxy; the latter substitutes a learned event head and local retrieval for the native generative pipeline. We do not label these as faithful native reproductions or claim to have beaten the original methods. A separately specified, task- and information-matched external comparison remains outstanding. Superiority claims based on the unmatched original table should be removed until that comparison exists.

Evidence: `submission/audit/REPORT.md` in the project; matched-run validation in `five_seed_evidence/combined_validation.json`.

## ECmL F3 and LCo1 W2/P4: component necessity

We added a three-seed matched 2×2 study of routing text-gradient conditioning and the decisiveness penalty, and then extended the full, no-penalty, and single-MLP candidates plus numerical/text addition controls to five seeds. These studies remain separate: we do not report five seeds for the incomplete factorial extension. The routing-gradient ablation has different mean-error directions on the earlier test interval and the newer development protocol; the available observations do not isolate a unique cause for that discrepancy.

The five-seed no-penalty model lowers both mean MSE and mean MAE on each development domain, but the Energy MAE improves in only 2/5 seeds. Thus the penalty is not empirically necessary across these settings. The single-MLP candidate also lowers both mean errors relative to full iAdapter, while reducing trainable parameters by approximately 36%. It changes both architecture and capacity, so it is a simple-model comparison rather than an isolated causal test of each heterogeneous expert. We do not infer individual expert semantics or the necessity of all experts from routing weights. The earlier key-input ablations remain separate evidence; a complete leave-one-expert-out study has not been added.

Against the stronger simple text-addition control, the single MLP changes mean MSE by −0.82% on Energy and +0.005% on SocialGood while costing more parameters and latency. This limits the practical argument for retaining its additional complexity. We retain all these outcomes rather than selecting only comparisons against the full model.

Evidence: initial `factorial_interaction.csv` in `experiments/review_sprint_20260929/reports/`; `five_seed_evidence/five_seed_summary.csv`, `five_seed_paired_summary.csv`; `cost_evidence/REPORT.md`.

## ECmL F4 and LCo1 P1: replay, implementation, and efficiency

In the corrected executor, each forecast origin stores its input window, selected text IDs, numerical/text feedback snapshots, maturity time, and original prediction. The current forecast is issued before any updates at that time. Mature records are consumed once, replayed with their original context and the current trainable projection of the frozen text representation; refreshed feedback affects subsequent forecasts. Evaluation always uses originally issued predictions. All internal methods use this executor, and the new audits reconstruct targets and masks from the original CSVs and verify chronological boundaries and once-only mature updates.

The model uses a minimal two-layer iTransformer implementation and full frozen 12-layer BERT with attention masks and masked pooling. Preprocessing statistics are fitted on the training prefix, input filling is causal, and originally missing targets are masked in training and scoring. Text availability is based on the supplied end_date field as a proxy; this is not verification of true publication timestamps. Full architecture and update details are in the method appendix and effective run configurations.

On the RTX 4060 Laptop GPU, we measured separate cached prediction, mature-record update, and fresh-text prediction paths using isolated warm-state copies, three timing rounds, and 50 warm-up plus 200 measured calls per round. For Energy, full versus single-MLP cached prediction takes 3.212 versus 2.153 ms and update takes 9.282 versus 4.894 ms; for SocialGood the corresponding values are 3.160 versus 2.050 ms and 9.363 versus 4.856 ms. These are medians of round medians. The report includes P95, round ranges, parameter counts, allocated GPU peaks, and cache bytes. Repeated updates here are timing operations only and do not modify scientific predictions or produce new accuracy results.

The fresh-text path includes tokenizer and BERT; its reductions are smaller (8.57% and 21.80%), and the Energy round ranges overlap. The benchmark excludes queue scheduling and disk logging and is not an end-to-end service throughput benchmark. Full and no-penalty models have identical parameter counts. We therefore restrict “lightweight” claims to the measured quantities and comparisons.

Evidence: `reports/method_appendix.md`, `docs/PROTOCOL.md`, saved configurations and update logs; `five_seed_evidence/combined_validation.json`; `cost_evidence/REPORT.md` and `status.json`.

## LCo1 W4: useful text and routing interpretation

We compared correct current text with a training-prefix mean, a token-length-matched earlier donor, and a zero representation at the same online state. Diagnostic forwards do not update the model or alter the reference trajectory, and donor selection excludes future records. Across five seeds, mean/donor replacement causes small forecast changes without a consistent error increase. For full iAdapter, the mean absolute prediction changes are approximately 6.4e-5 to 9.2e-5 in standardized output units. These outcomes do not establish effective use of current-text content.

The intervention holds historical feedback fixed and therefore does not remove all historical textual information. Zeroing additionally changes availability, so it cannot by itself demonstrate a semantic effect. The earlier state-group and routing analyses remain descriptive; they do not establish causal adaptation to concept drift. Given the absence of a stable content advantage under the predefined controls, further text-channel or trust-network experiments are not triggered at this stage.

Evidence: `five_seed_evidence/five_seed_text_probe_summary.csv`, all same-origin probe records; earlier `reports/state_paired_analysis.md`.

## LCo1 P5: table scope

The completed revision matrix reports Economy, Energy, Security, SocialGood, and Climate at H=6; its H=12 comparison covers Economy and Energy only. The new development comparison in this draft uses Energy and SocialGood at H=6 only, with a different explicitly recorded 60%/10%/10% chronological protocol. The original 190 evaluations, the 62 unique development evaluations, and the 30 timing round-cases must be identified separately. We do not combine their averages or extend the new findings to ten datasets, multiple backbones, or multiple encoders.

## Claim-evidence and writing check

- Supported conditionally: five-seed component error differences; single-MLP costs relative to full; weak current-content sensitivity under the specified interventions.
- Not established: native baseline superiority, universal component benefit, semantic expert roles, statistical significance, verified publication-time availability, or independent confirmation of selected candidates.
- The numbers above are traceable to the listed tables; all unsuccessful seeds remain included. Terminology distinguishes development accuracy, microbenchmark timing, native methods, and local proxies.
- This is replacement text for a later manuscript/rebuttal integration, not a claim that a complete LaTeX manuscript has already been revised or submitted.
