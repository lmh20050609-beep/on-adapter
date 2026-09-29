# External-comparison response: current verified status

## ECmL F2; LCo1 W3/P2 — baseline identity and controlled comparisons

We agree that averaging baseline insertion positions while selecting iAdapter's tail position does not produce comparable central values or error bars. We do not use that unmatched aggregate to claim superiority. The new internal experiments use a fixed insertion location and report variation across seeds; these do not substitute for a matched external comparison.

We have now pinned and inspected the author repositories for GPT4MTS and VoT. For GPT4MTS, the author model has passed an integration check with locally loaded pretrained GPT2 weights, retaining its six-block architecture and historical prompt sequence. Two explicit compatibility changes localize weight loading and preserve the batch dimension in the prompt-averaging operation. On the same model instance in evaluation mode, the patched and original batch-two forwards agree exactly. Batch-one prediction and a real-target masked gradient update pass; frozen parameters receive no gradients. This is integration evidence, not an accuracy result or a complete reproduction of the author's original benchmark.

For VoT, the released snapshot includes precomputed prediction arrays, but their records contain no generation-time inputs, timestamp provenance, or explicit forecast-origin mapping. The loader also expects a different prediction directory from the published directory, a failure we reproduced without training. Merely renaming the directory would not establish compliance with the shared information cutoff. We therefore have not used those arrays as verified predictions in a new comparison, nor treated the supplied VoT-inspired proxy as the native method. Missing provenance is not evidence that the original method leaks targets.

A shared OT-only static comparison is the next experimental stage. It requires matched forecast origins, target masks, chronology, and text-availability constraints, with architecture-specific text history, encoding, and training budgets explicitly recorded. Results from the existing multivariate online table will not be copied into that static comparison. Native external performance superiority remains unestablished at this stage.

Evidence: `gpt4mts_smoke_validation.json`, `vot_prior_audit.json`, the pinned source manifests, and `INTEGRATION_REPORT.md`. No native external comparison score is reported in this response.
