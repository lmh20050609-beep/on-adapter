# Existing WSL local runtime

### env: llm@47df2a8b

how: reuse /home/wuwa/miniconda3/envs/llm in WSL Ubuntu; no rebuild or installation in this wave.
tier: local RTX4060 Laptop 8188MiB; one serial job; PyTorch threads=1 in experiment stages.
workspace: /home/wuwa/projects/research/iadapter
weights: BERT-base and Qwen2.5-0.5B directories are recorded in ENV_SPEC.json.
validated: existing N1/N3 and real-target VoT passed; fresh command-following witness pending. Do not infer launch success from this ledger alone.
gotcha: PowerShell uses stdin here-string for multiline Python. WSL Git has no credentials; Windows Git against the same UNC workspace works. No user-process termination.

## Exact fresh witness invocation (PowerShell)

```powershell
$taskWitness = @'
import torch
torch.set_num_threads(1)
torch.manual_seed(2023)
x = torch.arange(16, device='cuda', dtype=torch.float32).reshape(4,4)
y = x @ x.T
print('WITNESS', tuple(y.shape), float(y.sum()), torch.cuda.get_device_name())
'@
$taskWitness | wsl.exe -d Ubuntu --cd /home/wuwa/projects/research/iadapter --exec /home/wuwa/miniconda3/envs/llm/bin/python -
```

Expected: `WITNESS (4, 4) 3680.0 NVIDIA GeForce RTX 4060 Laptop GPU`.
If the exact command differs from reality, report it without silently fixing the invocation. This witness performs no training and modifies no experimental state.
