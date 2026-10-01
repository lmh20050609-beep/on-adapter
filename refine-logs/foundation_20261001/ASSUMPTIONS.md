# C0 operational assumptions

Official source and HF metadata pins are recorded in EXPERIMENT_PLAN.md. Download transported via hf-mirror; direct official HF connectivity currently fails. Expected mirror digest is verified locally and disclosed, without claiming independent official LFS digest retrieval. Source is imported from pinned external/timesfm/src, no dependency upgrades.

Preflight is integration only, not formal numerical performance comparison. Model frozen, fp32, median forecast, batch1; no known future covariates. Prior first-party system checker is2.5-specific and does not certify3.0. C0 measures3.0 directly. Code-review batch2 quantile/finite/median check has been implemented. All wrapper/plan files are committed before launch; revision() dirty alone ignores untracked files and is not relied on for source inclusion.
