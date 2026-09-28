# Approved experiment execution plan

See ../docs/PROTOCOL.md and ../docs/DECISIONS.md for resolved scientific and
implementation settings. User approved complete implementation, local GPU only.

Order: CPU protocol checks → contextual BERT cache → Economy/H6/2023 three-method
sanity → 75 total main runs → 85 ablations → 30 H12 extension runs → efficiency,
mechanism analysis, replacement text and reviewer replies. No additional methods.

Successful completion means all 190 predefined cells have valid saved predictions,
paired scoring sets and actual measured metrics. Negative outcomes do not trigger
new modules or retuning. Failed cells are investigated and not counted complete.
