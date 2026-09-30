# All main-method paired contrasts

Delta = candidate minus control; negative values favor candidate. Five matched seeds share a backbone and scoring targets. SD is computed over seed differences. Improved-seed counts do not imply significance.

| dataset | horizon | candidate | control | n | delta_mse_mean | delta_mse_sd | delta_mae_mean | delta_mae_sd | mse_improved_seeds | mae_improved_seeds | valid_elements_per_seed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Climate | 6 | full | numeric_add | 5 | -0.00083 | 0.002267 | 0.000476 | 0.001237 | 2 | 1 | 1140 |
| Climate | 6 | full | text_add | 5 | -0.002108 | 0.002996 | 0.000647 | 0.001802 | 3 | 2 | 1140 |
| Climate | 6 | text_add | numeric_add | 5 | 0.001278 | 0.004206 | -0.000171 | 0.002097 | 1 | 2 | 1140 |
| Economy | 6 | full | numeric_add | 5 | 0.00162 | 0.005593 | 0.001218 | 0.00272 | 2 | 1 | 1440 |
| Economy | 6 | full | text_add | 5 | -0.000456 | 0.005897 | -0.000114 | 0.003367 | 4 | 4 | 1440 |
| Economy | 6 | text_add | numeric_add | 5 | 0.002076 | 0.000929 | 0.001333 | 0.001713 | 0 | 1 | 1440 |
| Economy | 12 | full | numeric_add | 5 | 0.007325 | 0.007689 | 0.003213 | 0.002459 | 1 | 0 | 2664 |
| Economy | 12 | full | text_add | 5 | -0.004504 | 0.006162 | -0.002926 | 0.003625 | 4 | 4 | 2664 |
| Economy | 12 | text_add | numeric_add | 5 | 0.011829 | 0.003695 | 0.006138 | 0.003569 | 0 | 0 | 2664 |
| Energy | 6 | full | numeric_add | 5 | 0.00338 | 0.045146 | 0.009917 | 0.054974 | 4 | 4 | 15714 |
| Energy | 6 | full | text_add | 5 | 4.2e-05 | 0.041648 | 0.005762 | 0.055083 | 4 | 4 | 15714 |
| Energy | 6 | text_add | numeric_add | 5 | 0.003338 | 0.010361 | 0.004155 | 0.005267 | 1 | 1 | 15714 |
| Energy | 12 | full | numeric_add | 5 | 0.022667 | 0.07074 | 0.017724 | 0.040506 | 2 | 2 | 30780 |
| Energy | 12 | full | text_add | 5 | 0.013557 | 0.04943 | 0.008439 | 0.028736 | 3 | 2 | 30780 |
| Energy | 12 | text_add | numeric_add | 5 | 0.00911 | 0.029878 | 0.009285 | 0.023676 | 2 | 2 | 30780 |
| Security | 6 | full | numeric_add | 5 | -0.298745 | 2.00285 | -0.066875 | 0.248176 | 1 | 2 | 330 |
| Security | 6 | full | text_add | 5 | -1.448009 | 2.202844 | -0.131334 | 0.266574 | 4 | 2 | 330 |
| Security | 6 | text_add | numeric_add | 5 | 1.149264 | 0.797152 | 0.064459 | 0.072985 | 0 | 2 | 330 |
| SocialGood | 6 | full | numeric_add | 5 | 0.004498 | 0.008655 | 0.006021 | 0.011503 | 2 | 1 | 1017 |
| SocialGood | 6 | full | text_add | 5 | 0.000713 | 0.007503 | 0.005848 | 0.010122 | 2 | 1 | 1017 |
| SocialGood | 6 | text_add | numeric_add | 5 | 0.003785 | 0.006341 | 0.000173 | 0.004762 | 1 | 2 | 1017 |
