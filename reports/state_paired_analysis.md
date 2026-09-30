# Paired effects by historical state change

Delta = candidate minus control; negative values favor the candidate. Groups 0/1/2 use validation tertiles of historical-window scores. Each row reports five matched seeds on the same valid target elements; SD is across seed differences, not windows. Improved-seed counts are descriptive, not significance tests. Empty test groups are omitted. No concept-drift ground truth or semantic text causality is assumed.

| dataset | horizon | group | candidate | control | n | delta_mse_mean | delta_mse_sd | delta_mae_mean | delta_mae_sd | mse_improved_seeds | mae_improved_seeds | valid_elements_per_seed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Climate | 6 | 0 | full | numeric_add | 5 | -0.00327 | 0.006261 | -0.000733 | 0.001895 | 3 | 4 | 216 |
| Climate | 6 | 0 | full | text_add | 5 | -0.006255 | 0.008113 | -0.000131 | 0.003642 | 4 | 2 | 216 |
| Climate | 6 | 0 | text_add | numeric_add | 5 | 0.002985 | 0.010981 | -0.000602 | 0.004791 | 2 | 4 | 216 |
| Climate | 6 | 1 | full | numeric_add | 5 | -0.001808 | 0.006278 | 0.000783 | 0.002013 | 3 | 2 | 480 |
| Climate | 6 | 1 | full | text_add | 5 | -0.004527 | 0.001787 | 0.000579 | 0.001585 | 5 | 2 | 480 |
| Climate | 6 | 1 | text_add | numeric_add | 5 | 0.002719 | 0.005581 | 0.000205 | 0.002749 | 1 | 3 | 480 |
| Climate | 6 | 2 | full | numeric_add | 5 | 0.001415 | 0.004489 | 0.000733 | 0.001924 | 1 | 1 | 444 |
| Climate | 6 | 2 | full | text_add | 5 | 0.002524 | 0.004723 | 0.0011 | 0.002415 | 2 | 2 | 444 |
| Climate | 6 | 2 | text_add | numeric_add | 5 | -0.001109 | 0.004143 | -0.000367 | 0.002142 | 3 | 3 | 444 |
| Economy | 6 | 0 | full | numeric_add | 5 | 0.003569 | 0.002702 | 0.00553 | 0.004438 | 0 | 0 | 252 |
| Economy | 6 | 0 | full | text_add | 5 | 0.004078 | 0.003979 | 0.003545 | 0.005267 | 1 | 1 | 252 |
| Economy | 6 | 0 | text_add | numeric_add | 5 | -0.000509 | 0.002284 | 0.001984 | 0.002087 | 4 | 1 | 252 |
| Economy | 6 | 1 | full | numeric_add | 5 | 0.002272 | 0.00584 | 0.001864 | 0.001877 | 2 | 1 | 774 |
| Economy | 6 | 1 | full | text_add | 5 | -0.002204 | 0.005769 | -0.001085 | 0.002124 | 4 | 4 | 774 |
| Economy | 6 | 1 | text_add | numeric_add | 5 | 0.004476 | 0.002999 | 0.002949 | 0.0026 | 0 | 1 | 774 |
| Economy | 6 | 2 | full | numeric_add | 5 | -0.000785 | 0.010612 | -0.002614 | 0.007986 | 3 | 3 | 414 |
| Economy | 6 | 2 | full | text_add | 5 | 5.3e-05 | 0.008634 | -0.000528 | 0.006205 | 3 | 4 | 414 |
| Economy | 6 | 2 | text_add | numeric_add | 5 | -0.000837 | 0.007456 | -0.002086 | 0.007923 | 3 | 3 | 414 |
| Energy | 6 | 0 | full | numeric_add | 5 | 0.012741 | 0.030676 | 0.027604 | 0.067974 | 3 | 3 | 648 |
| Energy | 6 | 0 | full | text_add | 5 | 0.012074 | 0.031272 | 0.026242 | 0.069327 | 3 | 3 | 648 |
| Energy | 6 | 0 | text_add | numeric_add | 5 | 0.000667 | 0.001335 | 0.001362 | 0.003238 | 2 | 3 | 648 |
| Energy | 6 | 1 | full | numeric_add | 5 | 0.013711 | 0.044229 | 0.019101 | 0.057157 | 3 | 3 | 3024 |
| Energy | 6 | 1 | full | text_add | 5 | 0.016488 | 0.045039 | 0.021236 | 0.056781 | 3 | 3 | 3024 |
| Energy | 6 | 1 | text_add | numeric_add | 5 | -0.002777 | 0.002976 | -0.002135 | 0.004971 | 3 | 3 | 3024 |
| Energy | 6 | 2 | full | numeric_add | 5 | 0.000282 | 0.046683 | 0.006658 | 0.054243 | 4 | 4 | 12042 |
| Energy | 6 | 2 | full | text_add | 5 | -0.004735 | 0.041381 | 0.000774 | 0.054134 | 4 | 4 | 12042 |
| Energy | 6 | 2 | text_add | numeric_add | 5 | 0.005017 | 0.014014 | 0.005884 | 0.007642 | 1 | 1 | 12042 |
| Security | 6 | 0 | full | numeric_add | 5 | 1.343393 | 2.396037 | 0.070155 | 0.116723 | 2 | 2 | 12 |
| Security | 6 | 0 | full | text_add | 5 | 0.656834 | 3.412799 | 0.033039 | 0.136518 | 2 | 2 | 12 |
| Security | 6 | 0 | text_add | numeric_add | 5 | 0.686559 | 1.121225 | 0.037116 | 0.045661 | 1 | 1 | 12 |
| Security | 6 | 1 | full | numeric_add | 5 | 2.050819 | 6.417848 | 0.06656 | 0.17957 | 2 | 2 | 6 |
| Security | 6 | 1 | full | text_add | 5 | -0.075612 | 4.812754 | 0.02549 | 0.170257 | 3 | 3 | 6 |
| Security | 6 | 1 | text_add | numeric_add | 5 | 2.126432 | 1.932408 | 0.04107 | 0.043983 | 1 | 1 | 6 |
| Security | 6 | 2 | full | numeric_add | 5 | -0.407088 | 2.294848 | -0.074711 | 0.270145 | 1 | 2 | 312 |
| Security | 6 | 2 | full | text_add | 5 | -1.555357 | 2.472829 | -0.140672 | 0.289251 | 4 | 2 | 312 |
| Security | 6 | 2 | text_add | numeric_add | 5 | 1.148269 | 0.851598 | 0.065961 | 0.077995 | 0 | 2 | 312 |
| SocialGood | 6 | 0 | full | numeric_add | 5 | -0.032404 | 0.060968 | 0.002167 | 0.014221 | 3 | 3 | 24 |
| SocialGood | 6 | 0 | full | text_add | 5 | -0.029748 | 0.042133 | 0.000117 | 0.012453 | 4 | 4 | 24 |
| SocialGood | 6 | 0 | text_add | numeric_add | 5 | -0.002656 | 0.03997 | 0.00205 | 0.00602 | 3 | 2 | 24 |
| SocialGood | 6 | 1 | full | numeric_add | 5 | 0.001744 | 0.00918 | 0.003139 | 0.004791 | 3 | 1 | 216 |
| SocialGood | 6 | 1 | full | text_add | 5 | -0.000898 | 0.009987 | 0.003073 | 0.004518 | 4 | 0 | 216 |
| SocialGood | 6 | 1 | text_add | numeric_add | 5 | 0.002641 | 0.005852 | 6.6e-05 | 0.000425 | 1 | 1 | 216 |
| SocialGood | 6 | 2 | full | numeric_add | 5 | 0.006404 | 0.009008 | 0.006941 | 0.013472 | 2 | 1 | 777 |
| SocialGood | 6 | 2 | full | text_add | 5 | 0.002102 | 0.009019 | 0.006796 | 0.011826 | 3 | 1 | 777 |
| SocialGood | 6 | 2 | text_add | numeric_add | 5 | 0.004302 | 0.010755 | 0.000145 | 0.006301 | 1 | 2 | 777 |
