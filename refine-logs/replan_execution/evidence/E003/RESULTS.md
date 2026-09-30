# Five-dimensional simple fusion exploration

| dataset | representation | mode | n | mse_mean | mse_sd | mae_mean | mae_sd |
|---|---|---|---|---|---|---|---|
| Energy | LLM5 | donor_same_state | 3 | 0.014238 | 0.000167 | 0.090381 | 0.00022 |
| Energy | LLM5 | online | 3 | 0.014237 | 0.000164 | 0.090377 | 0.000205 |
| Energy | LLM5 | static | 3 | 0.014138 | 0.000327 | 0.090032 | 0.000785 |
| Energy | raw_BERT_PCA5 | donor_same_state | 3 | 0.014685 | 0.000665 | 0.091735 | 0.001659 |
| Energy | raw_BERT_PCA5 | online | 3 | 0.014697 | 0.000694 | 0.091779 | 0.001747 |
| Energy | raw_BERT_PCA5 | static | 3 | 0.014912 | 0.001698 | 0.093366 | 0.006586 |
| SocialGood | LLM5 | donor_same_state | 3 | 0.130844 | 0.01516 | 0.208336 | 0.012311 |
| SocialGood | LLM5 | online | 3 | 0.130672 | 0.015281 | 0.20818 | 0.012376 |
| SocialGood | LLM5 | static | 3 | 0.132819 | 0.017523 | 0.209625 | 0.014407 |
| SocialGood | raw_BERT_PCA5 | donor_same_state | 3 | 0.132773 | 0.014102 | 0.209882 | 0.01163 |
| SocialGood | raw_BERT_PCA5 | online | 3 | 0.13272 | 0.014202 | 0.209849 | 0.011744 |
| SocialGood | raw_BERT_PCA5 | static | 3 | 0.133779 | 0.016167 | 0.210152 | 0.013364 |

## Reused strong system references

| dataset | method | n | mse_mean | mse_sd | mae_mean | mae_sd |
|---|---|---|---|---|---|---|
| Energy | frozen_backbone | 3 | 0.013927 | 3e-05 | 0.090502 | 0.001564 |
| Energy | gpt4mts_static | 3 | 0.015489 | 0.000557 | 0.092406 | 0.001811 |
| Energy | iadapter_online | 3 | 0.023298 | 0.007719 | 0.110256 | 0.015851 |
| Energy | numeric_online | 3 | 0.014097 | 0.000442 | 0.090025 | 0.002486 |
| Energy | numeric_static | 3 | 0.01395 | 0.000316 | 0.089681 | 0.001867 |
| SocialGood | frozen_backbone | 3 | 0.133175 | 0.018801 | 0.211382 | 0.015181 |
| SocialGood | gpt4mts_static | 3 | 0.08809 | 0.005238 | 0.186882 | 0.002546 |
| SocialGood | iadapter_online | 3 | 0.135922 | 0.015729 | 0.213576 | 0.014253 |
| SocialGood | numeric_online | 3 | 0.133797 | 0.016984 | 0.211547 | 0.015039 |
| SocialGood | numeric_static | 3 | 0.135423 | 0.017548 | 0.212558 | 0.015172 |

Reference rows are exact B1 reuse, not new training. GPT4MTS is a system-level comparator with different text history/encoding.
