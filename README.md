# iAdapter revision evidence package

Independent revision implementation based on the outer IG-SMoE source. Original
files and historical results are not modified. This package does not reproduce
the original ten-domain/multiple-backbone results.

Python: `/home/wuwa/miniconda3/envs/llm/bin/python`.
Working directory: `/home/wuwa/projects/research/iadapter_revision`.

```bash
/home/wuwa/miniconda3/envs/llm/bin/python -m unittest discover -s tests -v
/home/wuwa/miniconda3/envs/llm/bin/python -m iadapter.prepare
/home/wuwa/miniconda3/envs/llm/bin/python -m iadapter.run --dataset Economy --horizon 6 --seed 2023 --variant full --run_dir runs/example
/home/wuwa/miniconda3/envs/llm/bin/python -m iadapter.campaign
```

The campaign implements 190 adaptation evaluations and shares 35 backbone
checkpoints. Every prediction is emitted before mature-label updates at that
origin. Text record end dates are availability proxies, not verified publication
times. Missing targets are masked. Validation never trains adapter parameters.

See `docs/PROTOCOL.md`, `docs/DECISIONS.md`, and `refine-logs/` for details.
