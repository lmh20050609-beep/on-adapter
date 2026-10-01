# Additional-domain validation

| dataset | method | n | mse_mean | mse_sd | mae_mean | mae_sd |
|---|---|---|---|---|---|---|
| Agriculture | LLM5_online | 5 | 0.365011 | 0.024448 | 0.406278 | 0.017353 |
| Agriculture | LLM5_static | 5 | 0.361745 | 0.031725 | 0.401317 | 0.023022 |
| Agriculture | frozen_backbone | 5 | 0.354534 | 0.022402 | 0.398093 | 0.018375 |
| Agriculture | numeric_online | 5 | 0.348593 | 0.023567 | 0.392263 | 0.01461 |
| Agriculture | raw5_online | 5 | 0.363292 | 0.018837 | 0.406121 | 0.009901 |
| Health | LLM5_online | 5 | 0.796514 | 0.039084 | 0.568591 | 0.024813 |
| Health | LLM5_static | 5 | 0.736614 | 0.020849 | 0.519494 | 0.012026 |
| Health | frozen_backbone | 5 | 0.75028 | 0.046768 | 0.524325 | 0.029341 |
| Health | numeric_online | 5 | 0.865705 | 0.117777 | 0.586012 | 0.050354 |
| Health | raw5_online | 5 | 0.847904 | 0.094628 | 0.59296 | 0.051194 |

## Paired differences

Negative delta/change favors LLM5; SD is across five matched seeds on a fixed stream, not windows or a confidence interval. All controls and adverse outcomes remain.

| dataset | candidate | control | n | delta_mse_mean | delta_mse_sd | delta_mae_mean | delta_mae_sd | mse_improved_seeds | mae_improved_seeds | candidate_mse | control_mse | candidate_mae | control_mae | mse_change_percent | mae_change_percent |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Agriculture | LLM5_online | LLM5_static | 5 | 0.003266 | 0.010653 | 0.004961 | 0.011421 | 2 | 1 | 0.365011 | 0.361745 | 0.406278 | 0.401317 | 0.902758 | 1.236191 |
| Agriculture | LLM5_online | frozen_backbone | 5 | 0.010477 | 0.010037 | 0.008185 | 0.011256 | 1 | 1 | 0.365011 | 0.354534 | 0.406278 | 0.398093 | 2.95523 | 2.056031 |
| Agriculture | LLM5_online | numeric_online | 5 | 0.016418 | 0.01065 | 0.014014 | 0.007665 | 0 | 0 | 0.365011 | 0.348593 | 0.406278 | 0.392263 | 4.709694 | 3.572704 |
| Agriculture | LLM5_online | raw5_online | 5 | 0.001719 | 0.010739 | 0.000157 | 0.010102 | 2 | 3 | 0.365011 | 0.363292 | 0.406278 | 0.406121 | 0.473096 | 0.038629 |
| Health | LLM5_online | LLM5_static | 5 | 0.0599 | 0.035643 | 0.049097 | 0.020718 | 0 | 0 | 0.796514 | 0.736614 | 0.568591 | 0.519494 | 8.131777 | 9.450926 |
| Health | LLM5_online | frozen_backbone | 5 | 0.046234 | 0.055946 | 0.044266 | 0.034658 | 1 | 0 | 0.796514 | 0.75028 | 0.568591 | 0.524325 | 6.162286 | 8.442529 |
| Health | LLM5_online | numeric_online | 5 | -0.069191 | 0.125725 | -0.017421 | 0.053468 | 3 | 3 | 0.796514 | 0.865705 | 0.568591 | 0.586012 | -7.992471 | -2.972731 |
| Health | LLM5_online | raw5_online | 5 | -0.05139 | 0.061348 | -0.024368 | 0.029986 | 4 | 4 | 0.796514 | 0.847904 | 0.568591 | 0.59296 | -6.060776 | -4.109622 |
