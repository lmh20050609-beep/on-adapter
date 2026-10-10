# Nine-domain minimal text repair (V1): data and source report

2026-10-08. Goal: with numeric data, splits, targets, masks and feedback rules unchanged, extend the minimal repair of confirmed text timing mismatches to all 9 benchmark domains and freeze the full experimental protocol.

**V1 masks only individual entries confirmed to postdate the time they describe**; entries graded C/D by source are always kept and disclosed, and a screening miss does not mean certified. This report reads no model scores.

## 1. Text fields the models actually read

| Method | Reads `fact` | Input path |
|---|---|---|
| On-Adapter | yes | `online.ResidualOnline` via `data.prepared_text.pooled(origin)`, concatenation of visible rows (origin-1, origin), 64 tokens, mean pooling |
| GPT4MTS | yes | `multimodal.GPTForecast` via `text.gpt(origin, lookback)`, 512 tokens, cls |
| MM-TSFlib | yes | `multimodal.MMD` via `text.tokens(origin)`, 6 GPT2 layers, 256 tokens |
| iTransformer / ADAPT-Z / DSOF / Proceed | no | The code path does not access `data.texts` / `prepared_text` |

No method reads `preds`, `Final_Output` or `Final_Search_*`.

## 2. Source grading and masking

Rule file `source_evidence/decision_rules_all9.json`, SHA256 `85152a0e584b9bb5717895596c26b903f497e3a1285b0c02c78f51021f948689`; 56 frozen rules inherited from the 4-domain pilot (SHA256 `3d0109e3dd46e5f2a312dc59e8823d9f6d556437e7b06e6246531e5a11813eda`), plus 292 new rules from independent review of 5 domains.

| Domain | Individual entries | A | B (masked) | C (kept, disclosed) | D (kept, unverified) | Masked share of domain | Modified rows |
|---|---|---|---|---|---|---|---|
| Environment | 15014 | 0 | 238 | 126 | 14650 | 1.585% | 238 |
| Energy | 1478 | 1 | 0 | 0 | 1477 | 0.000% | 0 |
| Health_US | 1387 | 3 | 10 | 7 | 1367 | 0.721% | 10 |
| Traffic | 2287 | 0 | 40 | 44 | 2203 | 1.749% | 38 |
| SocialGood | 4513 | 0 | 103 | 20 | 4390 | 2.282% | 92 |
| Security | 1288 | 35 | 4 | 0 | 1249 | 0.311% | 4 |
| Economy | 1840 | 3 | 7 | 0 | 1830 | 0.380% | 7 |
| Climate | 1866 | 0 | 9 | 5 | 1852 | 0.482% | 9 |
| Agriculture | 1271 | 2 | 2 | 2 | 1265 | 0.157% | 2 |
| **Total** | 30944 | 44 | 413 | 204 | 30283 | 1.335% | 400 |

Masking counts entry occurrences; when the same raw text repeats across rows, it is masked separately in each row. No numeric row, forecast origin or scoring target was deleted.

## 3. Coverage per domain and split

| Domain | Split | Entries | A | B | C | D | Masked | Entry retention |
|---|---|---|---|---|---|---|---|---|
| Environment | train | 8914 | 0 | 126 | 63 | 8725 | 126 | 0.9859 |
| Environment | val | 3050 | 0 | 91 | 28 | 2931 | 91 | 0.9702 |
| Environment | test | 3050 | 0 | 21 | 35 | 2994 | 21 | 0.9931 |
| Energy | train | 886 | 0 | 0 | 0 | 886 | 0 | 1.0000 |
| Energy | val | 296 | 0 | 0 | 0 | 296 | 0 | 1.0000 |
| Energy | test | 296 | 1 | 0 | 0 | 295 | 0 | 1.0000 |
| Health_US | train | 831 | 2 | 8 | 4 | 817 | 8 | 0.9904 |
| Health_US | val | 278 | 0 | 1 | 0 | 277 | 1 | 0.9964 |
| Health_US | test | 278 | 1 | 1 | 3 | 273 | 1 | 0.9964 |
| Traffic | train | 1360 | 0 | 23 | 17 | 1320 | 23 | 0.9831 |
| Traffic | val | 461 | 0 | 10 | 13 | 438 | 10 | 0.9783 |
| Traffic | test | 466 | 0 | 7 | 14 | 445 | 7 | 0.9850 |
| SocialGood | train | 1459 | 0 | 44 | 8 | 1407 | 44 | 0.9698 |
| SocialGood | val | 1556 | 0 | 38 | 5 | 1513 | 38 | 0.9756 |
| SocialGood | test | 1498 | 0 | 21 | 7 | 1470 | 21 | 0.9860 |
| Security | train | 774 | 22 | 3 | 0 | 749 | 3 | 0.9961 |
| Security | val | 256 | 10 | 0 | 0 | 246 | 0 | 1.0000 |
| Security | test | 258 | 3 | 1 | 0 | 254 | 1 | 0.9961 |
| Economy | train | 1101 | 0 | 0 | 0 | 1101 | 0 | 1.0000 |
| Economy | val | 369 | 0 | 2 | 0 | 367 | 2 | 0.9946 |
| Economy | test | 370 | 3 | 5 | 0 | 362 | 5 | 0.9865 |
| Climate | train | 1001 | 0 | 3 | 2 | 996 | 3 | 0.9970 |
| Climate | val | 430 | 0 | 4 | 2 | 424 | 4 | 0.9907 |
| Climate | test | 435 | 0 | 2 | 1 | 432 | 2 | 0.9954 |
| Agriculture | train | 437 | 0 | 0 | 1 | 436 | 0 | 1.0000 |
| Agriculture | val | 401 | 1 | 1 | 0 | 399 | 1 | 0.9975 |
| Agriculture | test | 433 | 1 | 1 | 1 | 430 | 1 | 0.9977 |

## 4. Actual token changes

Using each method's frozen tokenizer, token limit and text mapping rule, the actual model input of every row is compared between V0 and V1.

| Domain | Method | Rows containing a masked entry | Of those, masked entry inside the token window | Rows with changed tokens | Test origins | Test origins with changed tokens |
|---|---|---|---|---|---|---|
| Environment | On-Adapter | 238 | 0 | 261 | 3003 | 23 |
| Environment | GPT4MTS | 238 | 0 | 231 | 3003 | 21 |
| Environment | MM-TSFlib | 238 | 224 | 262 | 3003 | 23 |
| Energy | On-Adapter | 0 | 0 | 0 | 285 | 0 |
| Energy | GPT4MTS | 0 | 0 | 0 | 285 | 0 |
| Energy | MM-TSFlib | 0 | 0 | 0 | 285 | 0 |
| Health_US | On-Adapter | 10 | 0 | 17 | 267 | 2 |
| Health_US | GPT4MTS | 10 | 0 | 10 | 267 | 1 |
| Health_US | MM-TSFlib | 10 | 10 | 20 | 267 | 2 |
| Traffic | On-Adapter | 38 | 0 | 27 | 102 | 4 |
| Traffic | GPT4MTS | 38 | 0 | 38 | 102 | 8 |
| Traffic | MM-TSFlib | 38 | 32 | 67 | 102 | 15 |
| SocialGood | On-Adapter | 92 | 0 | 51 | 172 | 10 |
| SocialGood | GPT4MTS | 92 | 0 | 53 | 172 | 10 |
| SocialGood | MM-TSFlib | 92 | 27 | 99 | 172 | 19 |
| Security | On-Adapter | 4 | 0 | 3 | 55 | 1 |
| Security | GPT4MTS | 4 | 0 | 4 | 55 | 1 |
| Security | MM-TSFlib | 4 | 3 | 7 | 55 | 2 |
| Economy | On-Adapter | 7 | 0 | 1 | 80 | 1 |
| Economy | GPT4MTS | 7 | 0 | 7 | 80 | 6 |
| Economy | MM-TSFlib | 7 | 5 | 13 | 80 | 12 |
| Climate | On-Adapter | 9 | 0 | 4 | 95 | 1 |
| Climate | GPT4MTS | 9 | 0 | 9 | 95 | 2 |
| Climate | MM-TSFlib | 9 | 6 | 16 | 95 | 4 |
| Agriculture | On-Adapter | 2 | 0 | 2 | 95 | 1 |
| Agriculture | GPT4MTS | 2 | 0 | 2 | 95 | 1 |
| Agriculture | MM-TSFlib | 2 | 2 | 4 | 95 | 2 |

In most domains the 64-token window of On-Adapter does not contain the masked entry itself; its token changes come mainly from the trailing 1200-character window of `clean_fact` shifting after the deletion. The 256-token window of MM-TSFlib keeps a sizeable share of the masked entries.

## 5. Identity and reproduction

| Domain | V1 file SHA256 | Non-fact columns identical |
|---|---|---|
| Environment | `8e49bd5a186f2a63f9547e7b6342d4f0…` | True |
| Energy | `d32f6dca6818cf8cb415c15af77a9d8e…` | True |
| Health_US | `d2a9e10eb61b3817221f7d8ca176275a…` | True |
| Traffic | `d1255c65f764f3ec07ef92129162347c…` | True |
| SocialGood | `9d31f381f459e0058a6808df1314756a…` | True |
| Security | `11df63ad31ef0a0d13448abbb25958d9…` | True |
| Economy | `1c30e12372d248e875691b45041180fb…` | True |
| Climate | `2b95bbc4cb16c02872422192bc9086a5…` | True |
| Agriculture | `eed9766319b022c240efcc1b028b6993…` | True |

V1 of Agriculture/Economy/Security is byte-identical to the frozen three-domain pilot files: True.

## 6. Limitations that must be kept

- Source review covers only candidate entries whose body mentions a year later than the described time; a miss does not mean there is no problem, nor that the entry is certified A.
- All C/D entries are kept in the actual data and only disclosed; V1 does not mean "every entry is certified".
- The numeric data still use the original `end_date` availability proxy and the retrospective target-mask origin grid; this does not prove full causality in the sense of real publication times and historical revisions.
- Source decisions read no model scores; masking rules are not adjusted based on results.
- This stage does not extend to other seeds or to a SocialGood main table; SocialGood V1 is included in the 9-domain data, but no separate extra pilot is run for it.
