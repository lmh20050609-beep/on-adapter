# Static and online results retained together

All six settings are shown; the degraded GPT4MTS online extension never replaces its static reference. Mean and sample SD over five seeds on the fixed development stream. No pooled cross-protocol win count or significance claim.

| dataset | method | n | mse_mean | mse_sd | mae_mean | mae_sd |
|---|---|---|---|---|---|---|
| Economy | numeric_online | 5 | 0.085257 | 0.014368 | 0.233085 | 0.022598 |
| Economy | numeric_static | 5 | 0.087523 | 0.015701 | 0.235855 | 0.024382 |
| Economy | iadapter_online | 5 | 0.097277 | 0.01891 | 0.251902 | 0.028118 |
| Economy | iadapter_static | 5 | 0.105313 | 0.023761 | 0.264798 | 0.03707 |
| Economy | gpt4mts_online_extension | 5 | 0.086951 | 0.012681 | 0.234353 | 0.016382 |
| Economy | gpt4mts_static | 5 | 0.061619 | 0.011277 | 0.198772 | 0.018977 |
| Energy | numeric_online | 5 | 0.014084 | 0.000705 | 0.090779 | 0.002605 |
| Energy | numeric_static | 5 | 0.013982 | 0.000689 | 0.090592 | 0.002622 |
| Energy | iadapter_online | 5 | 0.020119 | 0.007165 | 0.104631 | 0.014458 |
| Energy | iadapter_static | 5 | 0.027068 | 0.014423 | 0.1205 | 0.02908 |
| Energy | gpt4mts_online_extension | 5 | 0.02739 | 0.004321 | 0.111559 | 0.011173 |
| Energy | gpt4mts_static | 5 | 0.015348 | 0.00136 | 0.092643 | 0.005802 |
| Security | numeric_online | 5 | 2.782187 | 0.103676 | 1.22492 | 0.01478 |
| Security | numeric_static | 5 | 2.750218 | 0.096165 | 1.218393 | 0.013255 |
| Security | iadapter_online | 5 | 2.77954 | 0.110431 | 1.226069 | 0.010977 |
| Security | iadapter_static | 5 | 2.718436 | 0.09731 | 1.215638 | 0.008398 |
| Security | gpt4mts_online_extension | 5 | 5.208139 | 0.203605 | 1.781644 | 0.049025 |
| Security | gpt4mts_static | 5 | 2.705181 | 0.11998 | 1.226716 | 0.01483 |
| SocialGood | numeric_online | 5 | 0.144483 | 0.032429 | 0.220996 | 0.022733 |
| SocialGood | numeric_static | 5 | 0.146024 | 0.032361 | 0.221556 | 0.021961 |
| SocialGood | iadapter_online | 5 | 0.146866 | 0.032663 | 0.223024 | 0.022681 |
| SocialGood | iadapter_static | 5 | 0.149492 | 0.033051 | 0.22382 | 0.021836 |
| SocialGood | gpt4mts_online_extension | 5 | 0.420954 | 0.058396 | 0.363127 | 0.024381 |
| SocialGood | gpt4mts_static | 5 | 0.097411 | 0.014844 | 0.196118 | 0.014814 |
| Climate | numeric_online | 5 | 1.012202 | 0.014017 | 0.78221 | 0.00813 |
| Climate | numeric_static | 5 | 0.999289 | 0.012638 | 0.777723 | 0.007251 |
| Climate | iadapter_online | 5 | 1.009755 | 0.012885 | 0.783035 | 0.007228 |
| Climate | iadapter_static | 5 | 1.006596 | 0.013282 | 0.781827 | 0.007056 |
| Climate | gpt4mts_online_extension | 5 | 1.413786 | 0.044122 | 0.939274 | 0.025238 |
| Climate | gpt4mts_static | 5 | 1.046567 | 0.0563 | 0.789762 | 0.03279 |

## iAdapter online minus each control

Negative differences favor iAdapter. Cross-model seed labels align repetitions but do not imply shared parameters.

| dataset | control | n | delta_mse_mean | delta_mse_sd | delta_mae_mean | delta_mae_sd | mse_improved_seeds | mae_improved_seeds |
|---|---|---|---|---|---|---|---|---|
| Climate | numeric_online | 5 | -0.002447 | 0.003572 | 0.000826 | 0.001753 | 4 | 2 |
| Climate | numeric_static | 5 | 0.010466 | 0.00313 | 0.005313 | 0.002273 | 0 | 0 |
| Climate | iadapter_static | 5 | 0.003159 | 0.001986 | 0.001208 | 0.000608 | 0 | 0 |
| Climate | gpt4mts_online_extension | 5 | -0.404031 | 0.053496 | -0.156239 | 0.028146 | 5 | 5 |
| Climate | gpt4mts_static | 5 | -0.036812 | 0.064039 | -0.006726 | 0.037382 | 4 | 1 |
| Economy | numeric_online | 5 | 0.012019 | 0.009454 | 0.018816 | 0.017762 | 1 | 1 |
| Economy | numeric_static | 5 | 0.009754 | 0.00882 | 0.016046 | 0.016854 | 1 | 0 |
| Economy | iadapter_static | 5 | -0.008036 | 0.01052 | -0.012897 | 0.018011 | 5 | 4 |
| Economy | gpt4mts_online_extension | 5 | 0.010325 | 0.023278 | 0.017549 | 0.03396 | 2 | 2 |
| Economy | gpt4mts_static | 5 | 0.035657 | 0.024119 | 0.05313 | 0.037077 | 0 | 0 |
| Energy | numeric_online | 5 | 0.006035 | 0.007082 | 0.013852 | 0.014773 | 1 | 1 |
| Energy | numeric_static | 5 | 0.006137 | 0.007141 | 0.014039 | 0.014674 | 1 | 1 |
| Energy | iadapter_static | 5 | -0.00695 | 0.007417 | -0.015869 | 0.015216 | 4 | 4 |
| Energy | gpt4mts_online_extension | 5 | -0.007271 | 0.007475 | -0.006929 | 0.015962 | 4 | 4 |
| Energy | gpt4mts_static | 5 | 0.004771 | 0.007129 | 0.011988 | 0.014714 | 2 | 0 |
| Security | numeric_online | 5 | -0.002647 | 0.028672 | 0.001149 | 0.006856 | 3 | 2 |
| Security | numeric_static | 5 | 0.029322 | 0.02315 | 0.007676 | 0.004853 | 0 | 0 |
| Security | iadapter_static | 5 | 0.061104 | 0.025035 | 0.010431 | 0.004185 | 0 | 0 |
| Security | gpt4mts_online_extension | 5 | -2.428599 | 0.189408 | -0.555575 | 0.046296 | 5 | 5 |
| Security | gpt4mts_static | 5 | 0.074359 | 0.147529 | -0.000646 | 0.022236 | 2 | 2 |
| SocialGood | numeric_online | 5 | 0.002384 | 0.003298 | 0.002028 | 0.002682 | 1 | 1 |
| SocialGood | numeric_static | 5 | 0.000842 | 0.004483 | 0.001468 | 0.0037 | 2 | 2 |
| SocialGood | iadapter_static | 5 | -0.002625 | 0.00445 | -0.000796 | 0.003922 | 4 | 3 |
| SocialGood | gpt4mts_online_extension | 5 | -0.274088 | 0.066304 | -0.140103 | 0.02731 | 5 | 5 |
| SocialGood | gpt4mts_static | 5 | 0.049455 | 0.022116 | 0.026905 | 0.011709 | 0 | 0 |
