# Review sprint experiment tracker

Update: the table below preserves the initial 42-run, three-seed study. Another 20/20 runs now extend the five fixed candidates to five seeds (50 candidate cells after reusing 30 initial cells; 62 unique development runs overall). Same-state text probes total 80; 30 isolated timing round-cases are complete. See `REVIEW_UPDATE_FIVE_SEED.md` and `five_seed_evidence/combined_validation.json`. The full factorial remains three seeds.

42/42 development runs completed; 2/2 separate sanity runs completed.
All 48 same-state text probe sequences completed. Original CSV audit passed.

| Run | Status | MSE | MAE | Cell seconds |
|---|---|---:|---:|---:|
| Energy_h6_s2023_numeric_add | completed | 0.01708707 | 0.09883628 | 8.78 |
| Energy_h6_s2023_text_add | completed | 0.01755239 | 0.09964954 | 8.83 |
| Energy_h6_s2023_full | completed | 0.01797701 | 0.10242017 | 15.70 |
| Energy_h6_s2023_no_route_text_gradient | completed | 0.01865526 | 0.10201092 | 15.50 |
| Energy_h6_s2023_no_decisiveness | completed | 0.01678954 | 0.09792809 | 15.10 |
| Energy_h6_s2023_no_route_no_decisiveness | completed | 0.01707030 | 0.09799173 | 13.80 |
| Energy_h6_s2023_single_mlp | completed | 0.01659779 | 0.09608364 | 7.92 |
| Energy_h6_s2024_numeric_add | completed | 0.01633684 | 0.09693809 | 7.91 |
| Energy_h6_s2024_text_add | completed | 0.01629717 | 0.09759335 | 7.29 |
| Energy_h6_s2024_full | completed | 0.02038666 | 0.10983474 | 16.16 |
| Energy_h6_s2024_no_route_text_gradient | completed | 0.02473957 | 0.11820511 | 15.21 |
| Energy_h6_s2024_no_decisiveness | completed | 0.01628131 | 0.09761171 | 14.92 |
| Energy_h6_s2024_no_route_no_decisiveness | completed | 0.01615998 | 0.09764542 | 14.26 |
| Energy_h6_s2024_single_mlp | completed | 0.01567891 | 0.09503847 | 8.93 |
| Energy_h6_s2025_numeric_add | completed | 0.02100683 | 0.10836125 | 7.19 |
| Energy_h6_s2025_text_add | completed | 0.01771034 | 0.10231149 | 7.23 |
| Energy_h6_s2025_full | completed | 0.01810527 | 0.10222146 | 15.70 |
| Energy_h6_s2025_no_route_text_gradient | completed | 0.01873737 | 0.10310753 | 14.83 |
| Energy_h6_s2025_no_decisiveness | completed | 0.01789441 | 0.10255030 | 14.91 |
| Energy_h6_s2025_no_route_no_decisiveness | completed | 0.01733775 | 0.10058361 | 14.62 |
| Energy_h6_s2025_single_mlp | completed | 0.01763163 | 0.10207187 | 8.42 |
| SocialGood_h6_s2023_numeric_add | completed | 0.21182916 | 0.26462670 | 5.23 |
| SocialGood_h6_s2023_text_add | completed | 0.20919153 | 0.26356950 | 4.87 |
| SocialGood_h6_s2023_full | completed | 0.20578456 | 0.26757766 | 10.23 |
| SocialGood_h6_s2023_no_route_text_gradient | completed | 0.20761571 | 0.26810617 | 8.91 |
| SocialGood_h6_s2023_no_decisiveness | completed | 0.20249793 | 0.25794750 | 8.87 |
| SocialGood_h6_s2023_no_route_no_decisiveness | completed | 0.20364560 | 0.25942287 | 8.86 |
| SocialGood_h6_s2023_single_mlp | completed | 0.21021318 | 0.26508285 | 5.34 |
| SocialGood_h6_s2024_numeric_add | completed | 0.22877804 | 0.27357423 | 5.09 |
| SocialGood_h6_s2024_text_add | completed | 0.22830214 | 0.27277823 | 4.74 |
| SocialGood_h6_s2024_full | completed | 0.22952280 | 0.27744983 | 9.24 |
| SocialGood_h6_s2024_no_route_text_gradient | completed | 0.24033093 | 0.29142094 | 8.78 |
| SocialGood_h6_s2024_no_decisiveness | completed | 0.22942325 | 0.27442776 | 8.79 |
| SocialGood_h6_s2024_no_route_no_decisiveness | completed | 0.22943860 | 0.27443118 | 9.10 |
| SocialGood_h6_s2024_single_mlp | completed | 0.22968295 | 0.27310298 | 5.57 |
| SocialGood_h6_s2025_numeric_add | completed | 0.13480921 | 0.21022661 | 6.20 |
| SocialGood_h6_s2025_text_add | completed | 0.13235694 | 0.20765795 | 4.85 |
| SocialGood_h6_s2025_full | completed | 0.13485871 | 0.21394292 | 9.42 |
| SocialGood_h6_s2025_no_route_text_gradient | completed | 0.13043446 | 0.20785423 | 8.94 |
| SocialGood_h6_s2025_no_decisiveness | completed | 0.13174469 | 0.20760088 | 9.00 |
| SocialGood_h6_s2025_no_route_no_decisiveness | completed | 0.13184275 | 0.20721139 | 8.72 |
| SocialGood_h6_s2025_single_mlp | completed | 0.13246312 | 0.20687669 | 6.31 |

Cell time includes warm-up and diagnostics; do not interpret it as isolated latency. All results are development-only.
