# B3 fixed task text prior

| dataset | representation | n | mse_mean | mse_sd | mae_mean | mae_sd |
|---|---|---|---|---|---|---|
| Energy | raw_BERT_PCA5 | 3 | 0.019203 | 0.000977 | 0.106949 | 0.004602 |
| Energy | rule5 | 3 | 0.01919 | 0.00104 | 0.106974 | 0.004895 |
| Energy | LLM5 | 3 | 0.019478 | 0.001438 | 0.107699 | 0.005257 |
| Energy | LLM5_donor | 3 | 0.020042 | 0.001743 | 0.10868 | 0.005548 |
| Energy | rule5_donor | 3 | 0.01898 | 0.001129 | 0.106279 | 0.004955 |
| Energy | metadata_P1 | 3 | 0.019698 | 0.001601 | 0.107954 | 0.005518 |
| SocialGood | raw_BERT_PCA5 | 3 | 0.130258 | 0.006824 | 0.230301 | 0.010586 |
| SocialGood | rule5 | 3 | 0.129467 | 0.005585 | 0.231586 | 0.008766 |
| SocialGood | LLM5 | 3 | 0.134823 | 0.005983 | 0.236334 | 0.008515 |
| SocialGood | LLM5_donor | 3 | 0.129517 | 0.003799 | 0.233707 | 0.005981 |
| SocialGood | rule5_donor | 3 | 0.128593 | 0.004994 | 0.232076 | 0.00704 |
| SocialGood | metadata_P1 | 3 | 0.127831 | 0.006059 | 0.229999 | 0.009683 |

| dataset | candidate | control | mse_change_percent | mae_change_percent | mse_improved_seeds | candidate_donor_minus_correct_mse | passes |
|---|---|---|---|---|---|---|---|
| Energy | LLM5 | raw_BERT_PCA5 | 1.436067 | 0.701488 | 2 | 0.000564 | False |
| Energy | LLM5 | rule5 | 1.502687 | 0.677332 | 2 | 0.000564 | False |
| Energy | LLM5 | metadata_P1 | -1.114865 | -0.23658 | 3 | 0.000564 | False |
| Energy | rule5 | raw_BERT_PCA5 | -0.065633 | 0.023993 | 1 | -0.00021 | False |
| Energy | rule5 | metadata_P1 | -2.578801 | -0.907764 | 2 | -0.00021 | False |
| SocialGood | LLM5 | raw_BERT_PCA5 | 3.504167 | 2.619296 | 0 | -0.005306 | False |
| SocialGood | LLM5 | rule5 | 4.136727 | 2.050241 | 0 | -0.005306 | False |
| SocialGood | LLM5 | metadata_P1 | 5.469566 | 2.754065 | 0 | -0.005306 | False |
| SocialGood | rule5 | raw_BERT_PCA5 | -0.607432 | 0.557623 | 2 | -0.000874 | False |
| SocialGood | rule5 | metadata_P1 | 1.279893 | 0.689684 | 0 | -0.000874 | False |

Greedy frozen0.5B, exact M0 visible excerpts. Small-model inference is not causal identification; fixed-stream development only.
