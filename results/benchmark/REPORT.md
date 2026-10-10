# Nine-domain V1 text-repair benchmark: full 756-cell evaluation

2026-10-08. Data version **V1** (nine-domain minimal text repair); protocol in `PROTOCOL.md`, data and sources in `../data_repair/REPORT.md`.

This report **does not presuppose that any method wins**; results may be positive, negative or close to unchanged.

## 1. Execution matrix and completion

| Item | Count |
|---|---|
| Total matrix cells (9 domains × 4 H × 3 seeds × 7 methods) | 756 |
| Newly trained in this round (On-Adapter / GPT4MTS / MM-TSFlib) | 324 completed / 324 expected |
| Reused historical text-independent cells (iTransformer / ADAPT-Z / DSOF / Proceed) | 432 |
| Failed cells | 0 |
| Execution lock unchanged | True |

Newly trained cells completed per domain: Agriculture 36, Climate 36, Economy 36, Energy 36, Environment 36, Health_US 36, Security 36, SocialGood 36, Traffic 36

Every new cell was trained in a separate OS process from an identically defined initial state; no score, adapter checkpoint or result from an older text version was carried over.

## 2. Proof of reuse compatibility

- Reuse is limited to the 4 methods **whose code path does not read the `fact` column**, 432 cells in total, with the result / config / best.pt / predictions hashes pinned per cell (`reuse_verification.json`).

- The V1 tables are compared row by row with the original tables; the differing fields are `["fact"]`.

- Numeric-equivalence rerun (tolerance 0): status passed, 5 sampled cells.


| Sampled cell | predictions/targets/mask/origins bitwise equal | MAE/MSE exactly equal |
|---|---|---|
| Agriculture H6 seed2025 ADAPT-Z | True | True |
| Agriculture H6 seed2025 DSOF | True | True |
| Agriculture H6 seed2025 iTransformer | True | True |
| Agriculture H6 seed2025 Proceed | True | True |
| Traffic H12 seed2025 iTransformer | True | True |

## 3. Checks of previously uncovered paths

Covers daily/weekly frequencies, non-H12 windows, and GPT4MTS and MM-TSFlib, which had not been trained before; 51 checks in total, 0 failed.

| Check | Passed | Description |
|---|---|---|
| Origin/target/mask/standardization alignment | 13/13 | target_rows, mask and standardized ground truth checked per origin |
| Forecasts do not consume future y/mask | 13/13 | Forecasts are bitwise equal after replacing y and mask |
| Visibility of text gather rows | 13/13 | The availability time of every gathered row index is no later than the origin |
| Perturbing future text does not affect past outputs | 3/3 | Tolerance 0, bitwise equal |
| Preprocessing/cache version isolation | 9/9 | V1 and original files have different hashes; caches are isolated by data identity and text version |

## 4. Main statistic

Main statistic: within each domain/H, average the risk over seeds, divide by the CVaR90 of the shared iTransformer backbone for the same domain/H/seed, then take the equally weighted geometric mean over the 36 domain/H tasks.

| Method | Geometric mean relative CVaR90 across tasks | Arithmetic mean | Mean rank | Median rank | Best/worst task ratio |
|---|---|---|---|---|---|
| iTransformer | 1.000000 | 1.000000 | 3.7037 | 3.0 | 1.000000 / 1.000000 |
| ADAPT-Z | 1.049515 | 1.059844 | 3.6019 | 3.5 | 0.911095 / 1.838433 |
| DSOF | 1.050956 | 1.074622 | 3.8148 | 4.0 | 0.735498 / 1.579297 |
| Proceed | 1.110804 | 1.129717 | 4.9259 | 5.0 | 0.914759 / 1.703989 |
| On-Adapter | 1.093644 | 1.135214 | 4.1667 | 4.0 | 0.863227 / 2.532983 |
| GPT4MTS | 1.035984 | 1.045352 | 3.9444 | 4.0 | 0.833440 / 1.639155 |
| MM-TSFlib | 1.014709 | 1.015553 | 3.8426 | 4.0 | 0.942244 / 1.182166 |

| Method | Mean MAE | Mean MSE | Mean CVaR80 | Mean CVaR95 |
|---|---|---|---|---|
| iTransformer | 1.062820 | 11.978148 | 2.239271 | 2.888627 |
| ADAPT-Z | 1.142148 | 13.389743 | 2.396331 | 3.144579 |
| DSOF | 1.285982 | 13.904060 | 2.406262 | 2.994000 |
| Proceed | 1.175201 | 13.485760 | 2.449535 | 3.209736 |
| On-Adapter | 1.097404 | 12.209680 | 2.328718 | 3.040190 |
| GPT4MTS | 1.087851 | 12.204453 | 2.277366 | 2.944981 |
| MM-TSFlib | 1.069042 | 12.088541 | 2.264087 | 2.961087 |

## 5. Per-domain/H results (seed average)

| Domain | H | Method | MAE | MSE | CVaR90 | CVaR90 relative to backbone |
|---|---|---|---|---|---|---|
| Environment | 48 | iTransformer | 0.213505 | 0.145106 | 0.425975 | 1.000000 |
| Environment | 48 | ADAPT-Z | 0.217488 | 0.149880 | 0.435543 | 1.022528 |
| Environment | 48 | DSOF | 0.222366 | 0.147901 | 0.393683 | 0.924223 |
| Environment | 48 | Proceed | 0.247625 | 0.190147 | 0.515184 | 1.209200 |
| Environment | 48 | On-Adapter | 0.216566 | 0.148562 | 0.432690 | 1.015737 |
| Environment | 48 | GPT4MTS | 0.222873 | 0.155499 | 0.422082 | 0.991066 |
| Environment | 48 | MM-TSFlib | 0.215447 | 0.145489 | 0.430948 | 1.011703 |
| Environment | 96 | iTransformer | 0.220111 | 0.150118 | 0.396297 | 1.000000 |
| Environment | 96 | ADAPT-Z | 0.228431 | 0.156614 | 0.417048 | 1.052394 |
| Environment | 96 | DSOF | 0.250380 | 0.171958 | 0.444764 | 1.122321 |
| Environment | 96 | Proceed | 0.260477 | 0.190122 | 0.472197 | 1.191525 |
| Environment | 96 | On-Adapter | 0.232199 | 0.161828 | 0.437568 | 1.104264 |
| Environment | 96 | GPT4MTS | 0.287453 | 0.233652 | 0.536831 | 1.354671 |
| Environment | 96 | MM-TSFlib | 0.221688 | 0.149784 | 0.398814 | 1.006355 |
| Environment | 192 | iTransformer | 0.227274 | 0.143197 | 0.342570 | 1.000000 |
| Environment | 192 | ADAPT-Z | 0.239322 | 0.157216 | 0.371145 | 1.084686 |
| Environment | 192 | DSOF | 0.273679 | 0.175608 | 0.391299 | 1.142873 |
| Environment | 192 | Proceed | 0.280517 | 0.204050 | 0.471400 | 1.375800 |
| Environment | 192 | On-Adapter | 0.267925 | 0.192729 | 0.446361 | 1.303958 |
| Environment | 192 | GPT4MTS | 0.317912 | 0.270817 | 0.562319 | 1.639155 |
| Environment | 192 | MM-TSFlib | 0.225232 | 0.140314 | 0.339834 | 0.992312 |
| Environment | 336 | iTransformer | 0.220297 | 0.136154 | 0.301587 | 1.000000 |
| Environment | 336 | ADAPT-Z | 0.214557 | 0.130668 | 0.306776 | 1.017206 |
| Environment | 336 | DSOF | 0.273794 | 0.174736 | 0.375448 | 1.244945 |
| Environment | 336 | Proceed | 0.212917 | 0.129295 | 0.300749 | 0.997272 |
| Environment | 336 | On-Adapter | 0.217192 | 0.134298 | 0.310627 | 1.029688 |
| Environment | 336 | GPT4MTS | 0.301956 | 0.252914 | 0.488743 | 1.620162 |
| Environment | 336 | MM-TSFlib | 0.222354 | 0.135952 | 0.306547 | 1.016431 |
| Energy | 12 | iTransformer | 0.261574 | 0.128334 | 0.611671 | 1.000000 |
| Energy | 12 | ADAPT-Z | 0.288719 | 0.182206 | 0.858742 | 1.404337 |
| Energy | 12 | DSOF | 0.305608 | 0.179190 | 0.733057 | 1.198465 |
| Energy | 12 | Proceed | 0.339431 | 0.238763 | 1.041584 | 1.703989 |
| Energy | 12 | On-Adapter | 0.303145 | 0.219826 | 0.990520 | 1.620036 |
| Energy | 12 | GPT4MTS | 0.246707 | 0.121351 | 0.607425 | 0.993131 |
| Energy | 12 | MM-TSFlib | 0.257436 | 0.125272 | 0.599379 | 0.980105 |
| Energy | 24 | iTransformer | 0.385163 | 0.258659 | 0.792788 | 1.000000 |
| Energy | 24 | ADAPT-Z | 0.462848 | 0.468049 | 1.457630 | 1.838433 |
| Energy | 24 | DSOF | 0.443499 | 0.359777 | 0.999264 | 1.260420 |
| Energy | 24 | Proceed | 0.448765 | 0.403761 | 1.316443 | 1.660438 |
| Energy | 24 | On-Adapter | 0.551180 | 0.692185 | 2.008795 | 2.532983 |
| Energy | 24 | GPT4MTS | 0.373508 | 0.255282 | 0.785607 | 0.990913 |
| Energy | 24 | MM-TSFlib | 0.387784 | 0.261593 | 0.816627 | 1.030104 |
| Energy | 36 | iTransformer | 0.442734 | 0.345924 | 0.863316 | 1.000000 |
| Energy | 36 | ADAPT-Z | 0.484247 | 0.424226 | 1.105860 | 1.281100 |
| Energy | 36 | DSOF | 0.522240 | 0.493330 | 1.187127 | 1.375126 |
| Energy | 36 | Proceed | 0.488272 | 0.460529 | 1.264912 | 1.464983 |
| Energy | 36 | On-Adapter | 0.637352 | 0.903563 | 2.011124 | 2.328456 |
| Energy | 36 | GPT4MTS | 0.450392 | 0.361469 | 0.888905 | 1.029728 |
| Energy | 36 | MM-TSFlib | 0.449800 | 0.358063 | 0.885857 | 1.026158 |
| Energy | 48 | iTransformer | 0.511533 | 0.441506 | 0.887634 | 1.000000 |
| Energy | 48 | ADAPT-Z | 0.531404 | 0.470086 | 0.941512 | 1.060718 |
| Energy | 48 | DSOF | 0.590825 | 0.635995 | 1.282953 | 1.445405 |
| Energy | 48 | Proceed | 0.576254 | 0.573038 | 1.216513 | 1.370702 |
| Energy | 48 | On-Adapter | 0.734007 | 0.942181 | 1.817787 | 2.047145 |
| Energy | 48 | GPT4MTS | 0.504561 | 0.452828 | 0.939255 | 1.058396 |
| Energy | 48 | MM-TSFlib | 0.512559 | 0.445951 | 0.906321 | 1.021129 |
| Health_US | 12 | iTransformer | 1.392157 | 10.363014 | 3.928303 | 1.000000 |
| Health_US | 12 | ADAPT-Z | 1.531280 | 15.174954 | 4.620908 | 1.182505 |
| Health_US | 12 | DSOF | 1.562115 | 11.211276 | 4.121989 | 1.054083 |
| Health_US | 12 | Proceed | 1.941413 | 20.243822 | 6.246597 | 1.599678 |
| Health_US | 12 | On-Adapter | 1.468089 | 10.947372 | 4.011073 | 1.021557 |
| Health_US | 12 | GPT4MTS | 1.352115 | 9.987669 | 3.917835 | 1.002008 |
| Health_US | 12 | MM-TSFlib | 1.441136 | 11.379723 | 4.489113 | 1.147932 |
| Health_US | 24 | iTransformer | 1.589811 | 12.677477 | 3.896213 | 1.000000 |
| Health_US | 24 | ADAPT-Z | 1.700616 | 14.310085 | 4.494584 | 1.153294 |
| Health_US | 24 | DSOF | 1.860617 | 14.575460 | 4.025046 | 1.033918 |
| Health_US | 24 | Proceed | 1.948874 | 17.799769 | 4.700301 | 1.206238 |
| Health_US | 24 | On-Adapter | 1.762641 | 14.558588 | 4.275344 | 1.096998 |
| Health_US | 24 | GPT4MTS | 1.637574 | 12.837503 | 4.164464 | 1.069074 |
| Health_US | 24 | MM-TSFlib | 1.663094 | 13.966422 | 4.603145 | 1.182166 |
| Health_US | 36 | iTransformer | 1.665459 | 12.664019 | 3.571555 | 1.000000 |
| Health_US | 36 | ADAPT-Z | 1.750296 | 13.825170 | 4.069015 | 1.139004 |
| Health_US | 36 | DSOF | 1.812483 | 12.593243 | 3.245660 | 0.909029 |
| Health_US | 36 | Proceed | 2.010571 | 16.885262 | 4.867562 | 1.363816 |
| Health_US | 36 | On-Adapter | 1.783515 | 14.118143 | 4.085219 | 1.143156 |
| Health_US | 36 | GPT4MTS | 1.649828 | 11.950456 | 3.485296 | 0.975650 |
| Health_US | 36 | MM-TSFlib | 1.689407 | 13.209502 | 3.818115 | 1.068974 |
| Health_US | 48 | iTransformer | 1.668431 | 12.103991 | 2.980962 | 1.000000 |
| Health_US | 48 | ADAPT-Z | 1.713371 | 12.884506 | 3.378173 | 1.136154 |
| Health_US | 48 | DSOF | 1.778071 | 11.581541 | 2.789021 | 0.937863 |
| Health_US | 48 | Proceed | 1.948058 | 15.830266 | 4.637911 | 1.557276 |
| Health_US | 48 | On-Adapter | 1.772151 | 13.179622 | 3.322011 | 1.116384 |
| Health_US | 48 | GPT4MTS | 1.655293 | 11.552011 | 3.000291 | 1.008402 |
| Health_US | 48 | MM-TSFlib | 1.700282 | 12.792823 | 3.163609 | 1.063450 |
| Traffic | 6 | iTransformer | 0.294487 | 0.225606 | 0.993351 | 1.000000 |
| Traffic | 6 | ADAPT-Z | 0.286870 | 0.216962 | 0.962726 | 0.969237 |
| Traffic | 6 | DSOF | 0.447685 | 0.330786 | 0.853399 | 0.859268 |
| Traffic | 6 | Proceed | 0.261502 | 0.205604 | 0.946256 | 0.952612 |
| Traffic | 6 | On-Adapter | 0.292480 | 0.219655 | 0.952079 | 0.958628 |
| Traffic | 6 | GPT4MTS | 0.285567 | 0.213504 | 0.967589 | 0.974171 |
| Traffic | 6 | MM-TSFlib | 0.287276 | 0.217879 | 0.981764 | 0.988375 |
| Traffic | 8 | iTransformer | 0.289265 | 0.232870 | 0.967029 | 1.000000 |
| Traffic | 8 | ADAPT-Z | 0.279319 | 0.223704 | 0.942648 | 0.974812 |
| Traffic | 8 | DSOF | 0.463189 | 0.347983 | 0.812777 | 0.840588 |
| Traffic | 8 | Proceed | 0.262428 | 0.217612 | 0.884590 | 0.914759 |
| Traffic | 8 | On-Adapter | 0.292238 | 0.232447 | 0.939178 | 0.971234 |
| Traffic | 8 | GPT4MTS | 0.280442 | 0.227198 | 0.951185 | 0.983518 |
| Traffic | 8 | MM-TSFlib | 0.281955 | 0.228864 | 0.964648 | 0.997541 |
| Traffic | 10 | iTransformer | 0.283455 | 0.238178 | 0.937951 | 1.000000 |
| Traffic | 10 | ADAPT-Z | 0.276697 | 0.230943 | 0.917444 | 0.978172 |
| Traffic | 10 | DSOF | 0.453792 | 0.339871 | 0.765165 | 0.815683 |
| Traffic | 10 | Proceed | 0.273062 | 0.231170 | 0.876850 | 0.934846 |
| Traffic | 10 | On-Adapter | 0.284193 | 0.236846 | 0.925163 | 0.986292 |
| Traffic | 10 | GPT4MTS | 0.282803 | 0.238492 | 0.934615 | 0.996546 |
| Traffic | 10 | MM-TSFlib | 0.279749 | 0.234124 | 0.928144 | 0.989582 |
| Traffic | 12 | iTransformer | 0.288808 | 0.247493 | 0.893295 | 1.000000 |
| Traffic | 12 | ADAPT-Z | 0.282145 | 0.241882 | 0.879673 | 0.984765 |
| Traffic | 12 | DSOF | 0.462373 | 0.355004 | 0.734332 | 0.822019 |
| Traffic | 12 | Proceed | 0.280491 | 0.239563 | 0.852319 | 0.954107 |
| Traffic | 12 | On-Adapter | 0.290874 | 0.246768 | 0.881254 | 0.986538 |
| Traffic | 12 | GPT4MTS | 0.281960 | 0.241307 | 0.883766 | 0.989313 |
| Traffic | 12 | MM-TSFlib | 0.280833 | 0.241416 | 0.882495 | 0.987922 |
| SocialGood | 6 | iTransformer | 0.417609 | 1.052778 | 2.385015 | 1.000000 |
| SocialGood | 6 | ADAPT-Z | 0.386598 | 0.922587 | 2.173615 | 0.911095 |
| SocialGood | 6 | DSOF | 0.420364 | 0.744785 | 1.752276 | 0.735498 |
| SocialGood | 6 | Proceed | 0.375575 | 1.022066 | 2.240237 | 0.939559 |
| SocialGood | 6 | On-Adapter | 0.399045 | 0.960738 | 2.247439 | 0.942348 |
| SocialGood | 6 | GPT4MTS | 0.370366 | 0.832240 | 1.981579 | 0.833440 |
| SocialGood | 6 | MM-TSFlib | 0.397738 | 0.959729 | 2.243341 | 0.942244 |
| SocialGood | 8 | iTransformer | 0.443317 | 1.078835 | 2.414642 | 1.000000 |
| SocialGood | 8 | ADAPT-Z | 0.418581 | 0.990035 | 2.266906 | 0.939327 |
| SocialGood | 8 | DSOF | 0.459856 | 0.824841 | 1.857999 | 0.769627 |
| SocialGood | 8 | Proceed | 0.415122 | 1.152905 | 2.443922 | 1.012573 |
| SocialGood | 8 | On-Adapter | 0.415101 | 0.978602 | 2.241651 | 0.929896 |
| SocialGood | 8 | GPT4MTS | 0.438871 | 1.052533 | 2.310353 | 0.957593 |
| SocialGood | 8 | MM-TSFlib | 0.460474 | 1.117021 | 2.464807 | 1.021472 |
| SocialGood | 10 | iTransformer | 0.470095 | 1.114410 | 2.434011 | 1.000000 |
| SocialGood | 10 | ADAPT-Z | 0.434676 | 0.998196 | 2.224108 | 0.914224 |
| SocialGood | 10 | DSOF | 0.489366 | 0.875013 | 1.895552 | 0.780180 |
| SocialGood | 10 | Proceed | 0.456066 | 1.280547 | 2.612471 | 1.074682 |
| SocialGood | 10 | On-Adapter | 0.422071 | 0.956083 | 2.115673 | 0.869240 |
| SocialGood | 10 | GPT4MTS | 0.465462 | 1.293418 | 2.545422 | 1.047014 |
| SocialGood | 10 | MM-TSFlib | 0.480943 | 1.161345 | 2.517389 | 1.035437 |
| SocialGood | 12 | iTransformer | 0.506984 | 1.191538 | 2.507962 | 1.000000 |
| SocialGood | 12 | ADAPT-Z | 0.482000 | 1.107808 | 2.358530 | 0.940876 |
| SocialGood | 12 | DSOF | 0.522580 | 0.935352 | 1.939675 | 0.773700 |
| SocialGood | 12 | Proceed | 0.475199 | 1.279949 | 2.567168 | 1.023641 |
| SocialGood | 12 | On-Adapter | 0.452470 | 1.009167 | 2.163458 | 0.863227 |
| SocialGood | 12 | GPT4MTS | 0.523350 | 1.363328 | 2.598905 | 1.035795 |
| SocialGood | 12 | MM-TSFlib | 0.511194 | 1.245428 | 2.579144 | 1.029118 |
| Security | 6 | iTransformer | 4.279760 | 83.461655 | 13.326072 | 1.000000 |
| Security | 6 | ADAPT-Z | 5.349132 | 109.028458 | 16.424321 | 1.232496 |
| Security | 6 | DSOF | 5.000982 | 94.743344 | 14.327313 | 1.075138 |
| Security | 6 | Proceed | 4.608320 | 86.759353 | 13.312790 | 0.999006 |
| Security | 6 | On-Adapter | 4.329349 | 83.721390 | 13.303313 | 0.998292 |
| Security | 6 | GPT4MTS | 4.282883 | 83.834947 | 13.485086 | 1.011943 |
| Security | 6 | MM-TSFlib | 4.316479 | 83.558296 | 13.309962 | 0.998791 |
| Security | 8 | iTransformer | 4.441206 | 88.516175 | 12.016533 | 1.000000 |
| Security | 8 | ADAPT-Z | 5.066922 | 97.268667 | 12.225714 | 1.017588 |
| Security | 8 | DSOF | 5.499513 | 103.264576 | 13.334086 | 1.109683 |
| Security | 8 | Proceed | 4.974820 | 96.327736 | 12.101393 | 1.007210 |
| Security | 8 | On-Adapter | 4.514960 | 89.391482 | 11.973020 | 0.996377 |
| Security | 8 | GPT4MTS | 4.723248 | 91.426945 | 12.292850 | 1.022957 |
| Security | 8 | MM-TSFlib | 4.410673 | 88.514607 | 12.051291 | 1.002916 |
| Security | 10 | iTransformer | 4.531327 | 93.303636 | 10.943071 | 1.000000 |
| Security | 10 | ADAPT-Z | 4.968226 | 98.378867 | 10.837143 | 0.990345 |
| Security | 10 | DSOF | 5.878933 | 111.110329 | 12.341853 | 1.127932 |
| Security | 10 | Proceed | 5.185724 | 102.524916 | 10.852371 | 0.991822 |
| Security | 10 | On-Adapter | 4.598132 | 94.020320 | 10.930890 | 0.998876 |
| Security | 10 | GPT4MTS | 4.722886 | 96.200812 | 10.922233 | 0.998412 |
| Security | 10 | MM-TSFlib | 4.547495 | 93.563994 | 10.932183 | 0.999125 |
| Security | 12 | iTransformer | 4.722228 | 99.231911 | 10.612488 | 1.000000 |
| Security | 12 | ADAPT-Z | 5.045878 | 102.379638 | 10.520155 | 0.991273 |
| Security | 12 | DSOF | 6.215266 | 118.739100 | 11.471519 | 1.080979 |
| Security | 12 | Proceed | 5.346645 | 108.960839 | 10.864823 | 1.023848 |
| Security | 12 | On-Adapter | 4.757821 | 99.587238 | 10.597172 | 0.998561 |
| Security | 12 | GPT4MTS | 4.951384 | 102.077921 | 10.910686 | 1.028130 |
| Security | 12 | MM-TSFlib | 4.747027 | 99.270438 | 10.526751 | 0.991952 |
| Economy | 6 | iTransformer | 0.387765 | 0.263188 | 0.797873 | 1.000000 |
| Economy | 6 | ADAPT-Z | 0.386577 | 0.261613 | 0.785045 | 0.983931 |
| Economy | 6 | DSOF | 0.518858 | 0.454376 | 0.922523 | 1.156098 |
| Economy | 6 | Proceed | 0.398881 | 0.281844 | 0.794905 | 0.996292 |
| Economy | 6 | On-Adapter | 0.388372 | 0.263635 | 0.795386 | 0.996891 |
| Economy | 6 | GPT4MTS | 0.390227 | 0.266763 | 0.771436 | 0.966846 |
| Economy | 6 | MM-TSFlib | 0.388219 | 0.264541 | 0.798234 | 1.000418 |
| Economy | 8 | iTransformer | 0.417856 | 0.305375 | 0.785143 | 1.000000 |
| Economy | 8 | ADAPT-Z | 0.417664 | 0.305352 | 0.774014 | 0.985873 |
| Economy | 8 | DSOF | 0.616399 | 0.629943 | 1.072320 | 1.366508 |
| Economy | 8 | Proceed | 0.434029 | 0.335308 | 0.794080 | 1.011256 |
| Economy | 8 | On-Adapter | 0.420107 | 0.308996 | 0.785196 | 1.000067 |
| Economy | 8 | GPT4MTS | 0.423583 | 0.317798 | 0.811594 | 1.034454 |
| Economy | 8 | MM-TSFlib | 0.419839 | 0.306506 | 0.779833 | 0.993298 |
| Economy | 10 | iTransformer | 0.447299 | 0.346497 | 0.759763 | 1.000000 |
| Economy | 10 | ADAPT-Z | 0.446441 | 0.346669 | 0.753343 | 0.991605 |
| Economy | 10 | DSOF | 0.625781 | 0.641657 | 1.125220 | 1.479495 |
| Economy | 10 | Proceed | 0.459087 | 0.372816 | 0.771099 | 1.015019 |
| Economy | 10 | On-Adapter | 0.448246 | 0.348591 | 0.760731 | 1.001247 |
| Economy | 10 | GPT4MTS | 0.449043 | 0.348774 | 0.758111 | 0.998113 |
| Economy | 10 | MM-TSFlib | 0.446391 | 0.347981 | 0.782653 | 1.030717 |
| Economy | 12 | iTransformer | 0.473060 | 0.394121 | 0.795862 | 1.000000 |
| Economy | 12 | ADAPT-Z | 0.473645 | 0.395865 | 0.787219 | 0.989177 |
| Economy | 12 | DSOF | 0.664092 | 0.739434 | 1.256641 | 1.579297 |
| Economy | 12 | Proceed | 0.487873 | 0.418398 | 0.810095 | 1.017621 |
| Economy | 12 | On-Adapter | 0.475227 | 0.397196 | 0.794728 | 0.998605 |
| Economy | 12 | GPT4MTS | 0.469197 | 0.388063 | 0.808555 | 1.016134 |
| Economy | 12 | MM-TSFlib | 0.473481 | 0.392073 | 0.791017 | 0.994303 |
| Climate | 6 | iTransformer | 0.891995 | 1.233073 | 1.499740 | 1.000000 |
| Climate | 6 | ADAPT-Z | 0.890995 | 1.230041 | 1.502473 | 1.001802 |
| Climate | 6 | DSOF | 0.812952 | 1.085206 | 1.321879 | 0.881479 |
| Climate | 6 | Proceed | 0.913420 | 1.288917 | 1.553180 | 1.035630 |
| Climate | 6 | On-Adapter | 0.893125 | 1.236130 | 1.503436 | 1.002456 |
| Climate | 6 | GPT4MTS | 0.906283 | 1.261678 | 1.522161 | 1.015199 |
| Climate | 6 | MM-TSFlib | 0.894842 | 1.236697 | 1.510447 | 1.007240 |
| Climate | 8 | iTransformer | 0.898365 | 1.247610 | 1.415163 | 1.000000 |
| Climate | 8 | ADAPT-Z | 0.897524 | 1.244544 | 1.419143 | 1.002830 |
| Climate | 8 | DSOF | 0.817012 | 1.093329 | 1.229371 | 0.868753 |
| Climate | 8 | Proceed | 0.913960 | 1.285347 | 1.445188 | 1.021242 |
| Climate | 8 | On-Adapter | 0.901310 | 1.257325 | 1.424230 | 1.006408 |
| Climate | 8 | GPT4MTS | 0.913188 | 1.271154 | 1.389175 | 0.981761 |
| Climate | 8 | MM-TSFlib | 0.899674 | 1.249827 | 1.418409 | 1.002332 |
| Climate | 10 | iTransformer | 0.905686 | 1.261735 | 1.337285 | 1.000000 |
| Climate | 10 | ADAPT-Z | 0.904128 | 1.256188 | 1.342974 | 1.004272 |
| Climate | 10 | DSOF | 0.823020 | 1.104057 | 1.188205 | 0.888594 |
| Climate | 10 | Proceed | 0.917054 | 1.283126 | 1.363164 | 1.019388 |
| Climate | 10 | On-Adapter | 0.906309 | 1.264013 | 1.343555 | 1.004703 |
| Climate | 10 | GPT4MTS | 0.939410 | 1.346297 | 1.337285 | 1.000027 |
| Climate | 10 | MM-TSFlib | 0.906089 | 1.262519 | 1.334872 | 0.998231 |
| Climate | 12 | iTransformer | 0.913657 | 1.276372 | 1.322917 | 1.000000 |
| Climate | 12 | ADAPT-Z | 0.912839 | 1.273618 | 1.323152 | 1.000175 |
| Climate | 12 | DSOF | 0.831034 | 1.118186 | 1.155580 | 0.873627 |
| Climate | 12 | Proceed | 0.920165 | 1.290537 | 1.343897 | 1.015830 |
| Climate | 12 | On-Adapter | 0.914908 | 1.278183 | 1.322444 | 0.999601 |
| Climate | 12 | GPT4MTS | 0.932780 | 1.326816 | 1.348389 | 1.019375 |
| Climate | 12 | MM-TSFlib | 0.914347 | 1.282262 | 1.328883 | 1.004563 |
| Agriculture | 6 | iTransformer | 0.661550 | 0.996441 | 1.493977 | 1.000000 |
| Agriculture | 6 | ADAPT-Z | 0.659097 | 0.987480 | 1.482679 | 0.992492 |
| Agriculture | 6 | DSOF | 0.910922 | 1.603755 | 1.841101 | 1.232786 |
| Agriculture | 6 | Proceed | 0.698036 | 1.060960 | 1.504267 | 1.006980 |
| Agriculture | 6 | On-Adapter | 0.664269 | 0.996066 | 1.488788 | 0.996546 |
| Agriculture | 6 | GPT4MTS | 0.655829 | 1.000332 | 1.511357 | 1.011649 |
| Agriculture | 6 | MM-TSFlib | 0.660532 | 0.993327 | 1.492895 | 0.999327 |
| Agriculture | 8 | iTransformer | 0.761821 | 1.268636 | 1.613790 | 1.000000 |
| Agriculture | 8 | ADAPT-Z | 0.757242 | 1.254093 | 1.601118 | 0.992158 |
| Agriculture | 8 | DSOF | 1.095888 | 2.158415 | 2.130101 | 1.319674 |
| Agriculture | 8 | Proceed | 0.783781 | 1.303876 | 1.607823 | 0.996303 |
| Agriculture | 8 | On-Adapter | 0.762109 | 1.267505 | 1.609557 | 0.997377 |
| Agriculture | 8 | GPT4MTS | 0.749833 | 1.252527 | 1.618762 | 1.003144 |
| Agriculture | 8 | MM-TSFlib | 0.759561 | 1.262785 | 1.611638 | 0.998736 |
| Agriculture | 10 | iTransformer | 0.834018 | 1.480750 | 1.678513 | 1.000000 |
| Agriculture | 10 | ADAPT-Z | 0.832086 | 1.474102 | 1.673321 | 0.996910 |
| Agriculture | 10 | DSOF | 1.149605 | 2.368550 | 2.182454 | 1.300296 |
| Agriculture | 10 | Proceed | 0.849084 | 1.500271 | 1.676195 | 0.998622 |
| Agriculture | 10 | On-Adapter | 0.835791 | 1.484239 | 1.682108 | 1.002144 |
| Agriculture | 10 | GPT4MTS | 0.825190 | 1.464533 | 1.662942 | 0.990707 |
| Agriculture | 10 | MM-TSFlib | 0.833109 | 1.476312 | 1.670849 | 0.995437 |
| Agriculture | 12 | iTransformer | 0.901866 | 1.686928 | 1.722786 | 1.000000 |
| Agriculture | 12 | ADAPT-Z | 0.899452 | 1.679770 | 1.716879 | 0.996577 |
| Agriculture | 12 | DSOF | 1.220227 | 2.638235 | 2.241029 | 1.300831 |
| Agriculture | 12 | Proceed | 0.913752 | 1.714884 | 1.733182 | 1.006047 |
| Agriculture | 12 | On-Adapter | 0.902074 | 1.686959 | 1.725889 | 1.001796 |
| Agriculture | 12 | GPT4MTS | 0.898689 | 1.681464 | 1.695208 | 0.984086 |
| Agriculture | 12 | MM-TSFlib | 0.901355 | 1.688615 | 1.729638 | 1.004006 |

## 6. Pre-declared exploratory groupings

The following exploratory groupings were declared before the protocol was frozen; they are not confirmatory conclusions.

| Method | Geometric mean relative CVaR90 over the six monthly domains |
|---|---|
| iTransformer | 1.000000 |
| ADAPT-Z | 0.989427 |
| DSOF | 1.015707 |
| Proceed | 0.998076 |
| On-Adapter | 0.978640 |
| GPT4MTS | 0.994964 |
| MM-TSFlib | 1.000387 |

| Method | daily | weekly | monthly | Frequency-equal-weight |
|---|---|---|---|---|
| iTransformer | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| ADAPT-Z | 1.043858 | 1.255960 | 0.989427 | 1.090604 |
| DSOF | 1.102200 | 1.136826 | 1.015707 | 1.083696 |
| Proceed | 1.185749 | 1.482114 | 0.998076 | 1.205996 |
| On-Adapter | 1.107786 | 1.516509 | 0.978640 | 1.180252 |
| GPT4MTS | 1.374133 | 1.015436 | 0.994964 | 1.115568 |
| MM-TSFlib | 1.006660 | 1.063141 | 1.000387 | 1.023012 |

## 7. Historical reference (existing 756 cells with V0 text + old batched encoding)

The table below is recomputed by **read-only reuse** of the saved forecasts of completed periodic-v3/v4 cells. It is **not a new main table**: its text methods use the original V0 text column and the batched encoding from before the fix, so its differences from the V1 main table combine the text repair and the encoding fix.

| Method | Geometric mean CVaR90 relative to backbone | Geometric mean relative to per-task best | Mean rank | Mean MAE | Mean MSE |
|---|---|---|---|---|---|
| iTransformer | 1.0000 | 1.0841 | 3.713 | 1.062820 | 11.978148 |
| MM-TSFlib | 1.0155 | 1.1006 | 3.917 | 1.069410 | 12.090394 |
| GPT4MTS | 1.0293 | 1.1152 | 3.880 | 1.087865 | 12.208640 |
| ADAPT-Z | 1.0495 | 1.1376 | 3.620 | 1.142148 | 13.389743 |
| DSOF | 1.0510 | 1.1389 | 3.824 | 1.285982 | 13.904060 |
| On-Adapter | 1.0930 | 1.1846 | 4.157 | 1.097628 | 12.222457 |
| Proceed | 1.1108 | 1.2040 | 4.889 | 1.175201 | 13.485760 |

**Direct comparison of V1 and history (same main-statistic definition)**

| Method | V1 GM relative to backbone | Historical GM relative to backbone | Relative difference | Mean rank V1 / history |
|---|---|---|---|---|
| iTransformer | 1.0000 | 1.0000 | +0.00% | 3.704 / 3.713 |
| MM-TSFlib | 1.0147 | 1.0155 | -0.08% | 3.843 / 3.917 |
| GPT4MTS | 1.0360 | 1.0293 | +0.65% | 3.944 / 3.880 |
| ADAPT-Z | 1.0495 | 1.0495 | +0.00% | 3.602 / 3.620 |
| DSOF | 1.0510 | 1.0510 | +0.00% | 3.815 / 3.824 |
| On-Adapter | 1.0936 | 1.0930 | +0.06% | 4.167 / 4.157 |
| Proceed | 1.1108 | 1.1108 | +0.00% | 4.926 / 4.889 |

Maximum relative difference 0.65%. **Neither the text repair nor the encoding fix materially changes the comparative structure among methods**: the ranking is unchanged, and On-Adapter's disadvantage on Energy (H24: 2.47 → 2.53) and its advantages on SocialGood / Traffic are reproduced almost bit for bit, task by task. This conclusion applies only to the range of entries repaired in this round; it cannot be extrapolated to "the old results were entirely caused by leakage", nor to "the remaining text has been proven clean".

The "three-axis equal-weight geometric ratio to the best" in the historical record is: backbone 1.0861, MM-TSFlib 1.0937, GPT4MTS 1.0953, ADAPT-Z 1.1112, On-Adapter 1.1408, Proceed 1.1477, DSOF 1.1622, with On in 5th place. The oracle ratio in this table is defined similarly but aggregated differently, so it serves only as a directional comparison.


## 8. Actual visibility of the text repair

| Domain | Individual entries | B (masked) | C (kept, disclosed) |
|---|---|---|---|
| Environment | 15014 | 238 | 126 |
| Energy | 1478 | 0 | 0 |
| Health_US | 1387 | 10 | 7 |
| Traffic | 2287 | 40 | 44 |
| SocialGood | 4513 | 103 | 20 |
| Security | 1288 | 4 | 0 |
| Economy | 1840 | 7 | 0 |
| Climate | 1866 | 9 | 5 |
| Agriculture | 1271 | 2 | 2 |

In most domains the 64-token window of On-Adapter does not contain the masked entries themselves; the 256-token window of MM-TSFlib keeps some of them. Actual token changes are in `../data_repair/repair_token_effects.csv`.

## 9. Limitations that must be kept

- This protocol was frozen after the old test set had been observed repeatedly; it is **not an independent confirmation or a preregistration**.
- Source decisions cover only candidate entries hit by screening; C/D entries are kept in the actual data, and V1 does not mean "every entry is certified leak-free".
- The numeric data still use the original `end_date` availability proxy and the retrospective target-mask origin grid; this does not prove full causality in the sense of real publication times and historical revisions.
- The 432 reused cells are historical results under the same protocol; their text independence is shown by both the code path and the numeric-equivalence rerun, but this is not the same as retraining them in this round.
- iTransformer is the shared backbone; ratios to it are relative quantities, not absolute verdicts of better or worse.
- The text methods in the historical reference use the old batched encoding, so their differences from the V1 main table are not a pure text-repair effect.
