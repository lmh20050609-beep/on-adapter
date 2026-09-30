# B1 common-OT controls and deployment feedback

| dataset | method | n | mse_mean | mse_sd | mae_mean | mae_sd |
|---|---|---|---|---|---|---|
| Energy | frozen_backbone | 3 | 0.013927 | 3e-05 | 0.090502 | 0.001564 |
| Energy | text_add_static | 3 | 0.014231 | 0.000488 | 0.090432 | 0.001277 |
| Energy | text_add_online | 3 | 0.014408 | 0.000154 | 0.090885 | 0.000563 |
| Energy | iadapter_U1F0 | 3 | 0.022936 | 0.007124 | 0.109534 | 0.014504 |
| Energy | iadapter_U0F1 | 3 | 0.033047 | 0.016929 | 0.132459 | 0.031882 |
| SocialGood | frozen_backbone | 3 | 0.133175 | 0.018801 | 0.211382 | 0.015181 |
| SocialGood | text_add_static | 3 | 0.136406 | 0.014624 | 0.212807 | 0.011335 |
| SocialGood | text_add_online | 3 | 0.132654 | 0.014026 | 0.210331 | 0.011083 |
| SocialGood | iadapter_U1F0 | 3 | 0.135288 | 0.015613 | 0.213256 | 0.013851 |
| SocialGood | iadapter_U0F1 | 3 | 0.139708 | 0.019718 | 0.215882 | 0.016273 |
| Energy | numeric_static | 3 | 0.01395 | 0.000316 | 0.089681 | 0.001867 |
| Energy | numeric_online | 3 | 0.014097 | 0.000442 | 0.090025 | 0.002486 |
| Energy | iadapter_static | 3 | 0.032678 | 0.01631 | 0.132091 | 0.031272 |
| Energy | iadapter_online | 3 | 0.023298 | 0.007719 | 0.110256 | 0.015851 |
| Energy | gpt4mts_static | 3 | 0.015489 | 0.000557 | 0.092406 | 0.001811 |
| SocialGood | numeric_static | 3 | 0.135423 | 0.017548 | 0.212558 | 0.015172 |
| SocialGood | numeric_online | 3 | 0.133797 | 0.016984 | 0.211547 | 0.015039 |
| SocialGood | iadapter_static | 3 | 0.139454 | 0.019646 | 0.215505 | 0.016306 |
| SocialGood | iadapter_online | 3 | 0.135922 | 0.015729 | 0.213576 | 0.014253 |
| SocialGood | gpt4mts_static | 3 | 0.08809 | 0.005238 | 0.186882 | 0.002546 |

## Predeclared screening gates

| claim | dataset | mse_change_percent | mae_change_percent | mse_improved_seeds | passes |
|---|---|---|---|---|---|
| C1_feedback | Energy | 1.576632 | 0.659461 | 1 | False |
| C1_feedback | SocialGood | 0.467938 | 0.1504 | 0 | False |
| C2_initial_text | Energy | 2.208981 | 0.954601 | 1 | False |
| C2_initial_text | SocialGood | -0.854171 | -0.574855 | 2 | False |

Development only; same warmed full weights. U/F contrasts isolate deployment changes, not entire-training feedback necessity. Static GPT4MTS retained.
