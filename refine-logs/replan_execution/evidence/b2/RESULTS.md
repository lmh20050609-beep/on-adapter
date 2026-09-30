# B2 metadata/content residual readout

| dataset | probe | n | mse_mean | mse_sd | mae_mean | mae_sd |
|---|---|---|---|---|---|---|
| Energy | P0 | 3 | 0.019566 | 0.001633 | 0.107624 | 0.005576 |
| Energy | P1 | 3 | 0.019698 | 0.001601 | 0.107954 | 0.005518 |
| Energy | P2 | 3 | 0.018826 | 0.001176 | 0.105322 | 0.004276 |
| Energy | P3 | 3 | 0.01941 | 0.001325 | 0.107254 | 0.004933 |
| SocialGood | P0 | 3 | 0.127419 | 0.006019 | 0.230137 | 0.009742 |
| SocialGood | P1 | 3 | 0.127831 | 0.006059 | 0.229999 | 0.009683 |
| SocialGood | P2 | 3 | 0.136388 | 0.007 | 0.239886 | 0.009607 |
| SocialGood | P3 | 3 | 0.138584 | 0.003675 | 0.244175 | 0.009059 |

| dataset | mse_change_percent | mae_change_percent | mse_improved_seeds | donor_minus_correct_mse | passes |
|---|---|---|---|---|---|
| Energy | -4.42553 | -2.438292 | 3 | 0.000584 | True |
| SocialGood | 6.6942 | 4.298598 | 0 | 0.002196 | False |

Fit/PCA/scalers use training prefix only. Lambda uses selection segment, donor uses training text, development labels only score. P3 is a fixed-state intervention, not a trained model.
