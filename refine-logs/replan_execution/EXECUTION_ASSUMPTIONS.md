# Declared execution interpretations

2026-09-30. Local WSL/RTX4060, unchanged environment ledger, one forecasting job at a time. No API or remote spend.

- B1 numerical frozen reference uses the same existing OT backbone as the static/online records. Full U/F switches change deployment behavior only; initial warm weights and original forward paths remain unchanged. Default OnlineRunner behavior and event fields are preserved.
- Text_add uses the common-OT initializer and seed+11000 training-stream warm-up, matching the static experiment recipe, not the earlier multivariate experiment. Both modes share that warm state.
- M0 visible record means at least one content-character token span overlaps the record's cleaned text, excluding mere Facts/Outlook labels. Metadata counts visible channel records and age in days using end_date proxies.
- B2 numeric features are the frozen 32-dim OT context plus the existing 8-dim historical descriptor. Slots are numerical40 + metadata4 + content16. PCA fits informative training input rows; content is zero when absent. Numeric/metadata scalers fit training forecast origins.
- Ridge fits true target-minus-frozen-prediction training residuals, separately masks each horizon, and evaluates emitted predictions against actual CSV labels. Lambda candidates are .1/1/10, picked using selection masked MSE only. Donors are fixed CRC32 choices from informative training text rows.
- Lambda configuration count differs from six masked output solvers per configuration. Donor interventions are fixed-state sequences, not new models.
- M0/B1/B2 generated output lives in experiments/replan_20260930, separate from all prior experiment outputs. Original completed queues will not be restarted.
