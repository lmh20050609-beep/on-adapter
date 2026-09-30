# B3 settings fixed before any model output

Model: Qwen/Qwen2.5-0.5B-Instruct, public revision7ae557604adf67be50417f59c2c2f167def9a775;9files checked against published git blobs/LFS, safetensors988097824bytes. Existing torch2.6.0+cu124/transformers4.57.6, no installs. Frozen fp16/eager localCUDA, one inference model at a time; greedy maxnew8, freshGenerationConfig with repetition penalty1, no inherited sampling defaults.

Target anchors: Time-MMD official numerical-data description and dataset paper Table1 (https://github.com/AdityaLab/Time-MMD ; https://arxiv.org/abs/2406.08627), plus submitted CSV schemas. Energy is weekly U.S.gasoline-price OT; SocialGood monthly U.S.unemployment-rate OT. No precise regional identity or publication-time provenance is inferred from the alias OT. H6 means six weeks/months respectively.

Input: M0 visible_text exact prefix and fixed domain/horizon only. There are1538 distinct inputs across two domains, below1543cap. Thirty preflight inputs use15fixed evenly spaced informative training rows per domain, no evaluation labels. Invalid exact enum output mapsUNKNOWN;>5% stops. Same source/prompt/text cache is shared across seeds. Rule vocabulary and domain anchors are immutable in task_text_prior.py, not optimized against labels.

Preflight plus all generation calls share3600sec cumulative wall-time budget including model load/tokenization. Input/output tokens and allocated memory recorded. Model checkpoint can contain later historical information; no pretraining-leakage-free or causal inference claim. These settings test one small-model task representation, not complete TiMi or broad LLM capability.
