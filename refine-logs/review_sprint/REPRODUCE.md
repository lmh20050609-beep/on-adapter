# Local development sprint commands

Working directory: `/home/wuwa/projects/research/iadapter`.
Interpreter: `/home/wuwa/miniconda3/envs/llm/bin/python`.

```bash
python -m iadapter.review_sprint --sanity --run_dir experiments/review_sprint_20260929_sanity
python -m iadapter.review_sprint_analysis --sanity --run_dir experiments/review_sprint_20260929_sanity
python -m iadapter.review_sprint --run_dir experiments/review_sprint_20260929
python -m iadapter.review_sprint_analysis --run_dir experiments/review_sprint_20260929
```

Here `python` means the exact interpreter above. Start the 42-cell campaign only after sanity passes.
The local serial queue stores planned/running/completed/failed states. A failed cell stops the queue;
inspect its status/trace before any retry, retaining the failed attempt. No automatic scope/seed/model reduction.
Finished cells are skipped on same-revision resume; unresolved states or changed source revision are rejected.
Analysis reads the experiment source revision from its queue, allowing later reporting without retraining.
