# backbones

Copies of the two benchmark packages that trained the iTransformer checkpoints reused in the V1 main table
(`periodic_v3`: seeds 2025 and 2026; `periodic_v4`: seed 2027). `new_baselines/solid.py` imports their `data` and
`native` modules to rebuild those checkpoints exactly, so that SOLID, ADCSD, OGD and COSA adapt the same backbone.

Changes from the originals: `config.py` reads `data/` of this repository (numeric columns are byte-identical to the
original files), and `native.py` / `text_clean.py` use `third_party/code/` and `on_adapter_original/`. The training
entry points (`worker.py`, `scheduler.py`, `launch.py`, ...) are kept as recorded and still refer to the original
campaign names; they are not needed here.
