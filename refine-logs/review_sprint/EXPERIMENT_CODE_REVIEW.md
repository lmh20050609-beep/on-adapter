# Fresh-agent experiment code review

2026-09-29. Reviewer: /root/sprint_code_review, GPT-6-Astra xhigh; same-family provisional.

Verdict: no BLOCKING issues, no required NONBLOCKING corrections; proceed to sanity after committing source.

Reviewer independently inspected review_sprint.py, review_sprint_analysis.py, config.py, models.py, engine.py, data.py, run.py and tests.

- Actual Energy development target indices: 1035..1182; SocialGood: 630..719. No old 80%-100% test targets enter evaluation.
- Full/route-off/entropy-off/both-off and SingleMLP semantics match the plan; shared parameters match initialization.
- Train-only warm-up, fresh optimizer/cache/queue for development, original-context replay and masked ground truth are correct.
- Past-only token-length-matched donors and training-prefix mean are used for same-state counterfactuals.
- Independent 38-test suite passed. Extra CPU synthetic checks for full and text_add found identical original predictions, gates, feedback, updates, final parameters and CPU RNG with diagnostics enabled/disabled.
- All development origins in both actual datasets have nonempty text and a legal prior donor.
- New results/checkpoints are isolated; failures remain recorded and stop the queue.

No training or full reproduction was performed by the reviewer.

## Independent environment witness

/root/sprint_env_witness executed the documented command from /home/wuwa/projects/research/iadapter:

`/home/wuwa/miniconda3/envs/llm/bin/python -m iadapter.witness`

Exit 0: `WITNESS (8, 8) NVIDIA GeForce RTX 4060 Laptop GPU`.
All five required BERT files exist and are nonempty in weights/bert-base-uncased. No installation, downloads, training or process stopping occurred.
