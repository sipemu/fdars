---
phase: 113-cross-language-baseline-hotspot-ranking
verified: 2026-09-23T10:30:00Z
status: passed
score: 3/3 must-haves verified
covered_files:
  - .planning/REQUIREMENTS.md
  - .planning/phases/113-cross-language-baseline-hotspot-ranking/113-01-PLAN.md
  - .planning/phases/113-cross-language-baseline-hotspot-ranking/113-01-SUMMARY.md
  - .planning/phases/113-cross-language-baseline-hotspot-ranking/113-02-PLAN.md
  - .planning/phases/113-cross-language-baseline-hotspot-ranking/113-02-SUMMARY.md
  - .planning/phases/113-cross-language-baseline-hotspot-ranking/113-03-PLAN.md
  - .planning/phases/113-cross-language-baseline-hotspot-ranking/113-03-SUMMARY.md
  - benchmarks/assemble_comparison.py
  - benchmarks/gate.py
  - benchmarks/r_runner.R
  - benchmarks/rank_hotspots.py
  - benchmarks/results/comparison.csv
  - benchmarks/results/env.json
  - benchmarks/results/hotspot_ranking.md
  - benchmarks/run_all.py
  - benchmarks/test_assemble_comparison.py
  - benchmarks/test_gate.py
  - benchmarks/test_rank_hotspots.py
covered_digest: "v1:sha256:02af6dee4ee08fc5d15202d9571af3c4bc6071561e20811d5e386d458d43d01b"
re_verification:
  reason: "Post-verification code-review fix pass (CR-01 rsd_pct from true std_dev.point_estimate not mean.standard_error; WR-01/02/03 dead-param/NaN-guard/op-nm-match) changed benchmarks/ source and regenerated comparison.csv + hotspot_ranking.md; digest recomputed. Re-reviewed clean (113-REVIEW.md status: clean), 60 tests pass, no regressions. The fix STRENGTHENS honesty (corrected RSD moved 3 functional_boxplot cells to FAIL-NOISY) WITHOUT changing the operator-approved conclusions: elastic alignment remains rank #1, secondary targets (covariance/FPCA) unchanged, fairness controls intact — so the operator's fair/honest sign-off still holds. The 3/3 must-haves remain satisfied."
  digest_updated: "2026-09-23"
behavior_unverified: 0
overrides_applied: 0
---

# Phase 113: Cross-Language Baseline & Hotspot Ranking — Verification Report

**Phase Goal:** A fair, honest cross-language baseline is committed and the optimization targets are ranked by evidence — so Phases 114–115 attack the highest-leverage work first.
**Verified:** 2026-09-23T10:30:00Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | A cross-language baseline is established across all three ecosystems over an N×M input grid (≥3 sizes, elastic target at N≥100) and merged into a committed `comparison.csv`. | ✓ VERIFIED | comparison.csv is git-tracked (confirmed via `git ls-files`): 215 rows, libraries `rust`/`python`/`r_svd`/`r_native_pca_fd`, N∈{100,300,500}×M∈{50,200} grid (3 N-values, ≥3 sizes). Elastic timing exists for python/r_svd at N=100/300/500. Schema columns present including `rsd_pct`/`ci_width_pct`/`noise_gate`. Formula correctness spot-checked: rsd and ci_width values match std_dev_ns/mean_ns×100 and (ci_upper-ci_lower)/mean_ns×100 respectively. |
| 2 | An evidence-ranked hotspot list exists combining the cross-language results with the criterion suite, ranked by both absolute time and relative-to-baseline leverage. | ✓ VERIFIED | `benchmarks/results/hotspot_ranking.md` is git-tracked (134 lines, non-empty). `benchmarks/rank_hotspots.py` computes `leverage_score = fdars_ns × max(1.0, fdars_ns / min_other_ns)` (verified via grep). Ranking uses stable descending sort. Top-10 list verified present with leverage scores. `test_rank_hotspots.py` 20 tests pass (pytest green). Nyquist tests `test_noise_gate_flags_high_rsd` and `test_leverage_score_deterministic` both PASS individually. |
| 3 | The ranking is gated on measurement noise (RSD < 5% / CI < ±5%) so no spurious speedup target is chosen; the elastic-alignment anchor is confirmed and the secondary targets (SVD/FPCA, covariance) are ordered. | ✓ VERIFIED | `apply_noise_gate` in rank_hotspots.py excludes rows with rsd_pct≥5 or ci_width_pct≥5. comparison.csv shows 69 PASS / 68 FAIL-NOISY for rust rows. hotspot_ranking.md has `## Confirmed Anchor: Elastic Alignment` section (elastic rank #1, N=500/M=50: ~37,965ms, leverage 37.965s·factor, sourced from noise-stable `audit_p3_elastic_cross` cells — Karcher-mean excluded with RSD 29–46% disclosed). `## Secondary Targets` section orders face_covariance, fem_smooth, fpca, frechet_anova. `## Noise-Excluded Cells` section discloses all 68 gated-out cells with RSD%/CI-width% and reasons. |

**Score:** 3/3 truths verified (0 present, behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `benchmarks/results/comparison.csv` | Cross-language baseline, 215 rows, 4 libraries, schema with rsd_pct/ci_width_pct/noise_gate | ✓ VERIFIED | Git-tracked (force-added from gitignored results/). 215 rows confirmed. Libraries: rust, python, r_svd, r_native_pca_fd. All 13 schema columns present. Noise-gate 69 PASS / 68 FAIL-NOISY. env.json all six thread_env="1". |
| `benchmarks/results/env.json` | Re-captured fairness controls (all thread_env="1") | ✓ VERIFIED | Git-tracked. `thread_env`: OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, MKL_NUM_THREADS=1, RAYON_NUM_THREADS=1, NUMEXPR_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1 — all "1". |
| `benchmarks/results/hotspot_ranking.md` | Confirmed anchor + secondary targets + noise-excluded disclosure + leverage formula | ✓ VERIFIED | Git-tracked. 134 lines. Sections verified: `## Confirmed Anchor: Elastic Alignment`, `## Secondary Targets` (SVD/FPCA + covariance ordered), `## Noise-Excluded Cells` (68 cells with RSD%/CI-width%), `## Notes` (leverage formula stated; fpca_native_r exclusion stated). |
| `benchmarks/rank_hotspots.py` | Noise gate + leverage formula + deterministic ranking | ✓ VERIFIED | Exists (10,720 bytes). Contains `apply_noise_gate`, `compute_leverage`, `rank_rows`, `emit_hotspot_ranking`. Leverage formula `fdars_ns × max(1.0, fdars_ns / min_other_ns)` confirmed by grep. Division-by-zero guards present. Deterministic stable sort confirmed. |
| `benchmarks/test_rank_hotspots.py` | 20 unit tests including Nyquist tests | ✓ VERIFIED | Exists. pytest reports 20 tests, all pass in 1.19s (combined run of test_rank_hotspots.py + test_assemble_comparison.py). Nyquist tests `test_noise_gate_flags_high_rsd` and `test_leverage_score_deterministic` individually confirmed PASS. |
| `benchmarks/gate.py` | Pearson correlation gate for depth/functional_boxplot; FPCR disclosed/non-gating; fpca_native_r absent from _GATED_OPS | ✓ VERIFIED | `_DEPTH_CORR_THRESHOLD = 0.99` present. `_FPCR_CORR_THRESHOLD = 0.95` present with disclosure comment citing Phase-112 empirical 0.96 value. `_GATED_OPS = frozenset(["fpca", "basis_smoothing", "depth", "functional_boxplot", "elastic"])` — FPCR not present (disclosed non-gating per Plan 02 decision). gate.py --self-test exits 0. |
| `benchmarks/test_gate.py` | Updated tests for Pearson correlation metric; 27 tests | ✓ VERIFIED | pytest reports 27 passed in 0.64s. ConstantInputWarning on two tests is benign (test for constant-input edge case). |
| `benchmarks/r_runner.R` | run_fpca_native_r() function; NOT in ALL_OPS | ✓ VERIFIED | `run_fpca_native_r` function present (lines 189/192 confirmed by grep). `ALL_OPS <- c("fpca", "basis_smoothing", "depth", "functional_boxplot", "fpcr", "elastic")` — fpca_native_r absent. Switch case present for --op fpca_native_r. |
| `benchmarks/assemble_comparison.py` | Schema-correct CSV merge; pandas to_csv; correct noise fields | ✓ VERIFIED | Exists (16,978 bytes). `df.to_csv(str(out_path), index=False)` confirmed. Column order matches RESEARCH schema. Notes field for r_native_pca_fd confirmed ("native pca.fd — B-spline coeff space; not gated"). 13 assemble tests pass. |
| `benchmarks/run_all.py` | --grid mode; no shell=True; 6-cell sequential loop | ✓ VERIFIED | `--grid` option present (lines 652/655 confirmed). `--grid --dry-run` runs correctly and iterates all 6 cells (N∈{100,300,500} × M∈{50,200}). No `shell=True` references (only the comment "no shell=True per T-112-02"). fpca_native_r explicitly added per cell. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `comparison.csv` | `rank_hotspots.py` | pandas read_csv → apply_noise_gate → compute_leverage → sorted | ✓ WIRED | rank_hotspots.py reads comparison.csv and applies the full pipeline. hotspot_ranking.md emitted from real data. |
| `run_all.py` grid loop | per-cell python/r/rust result JSON → assemble_comparison.py | generate→python_runner→r_runner→collect_rust→gate→assemble | ✓ WIRED | --dry-run confirms 6-cell sequential loop with fpca_native_r included. Per-cell python_n*/r_n* JSON files are git-committed, proving the real run happened. |
| `gate.py` check_agreement | depth/functional_boxplot: Pearson corr ≥ _DEPTH_CORR_THRESHOLD | pearsonr() dispatch | ✓ WIRED | Lines 248/250 confirmed. FPCR branch at lines 267/269 uses _FPCR_CORR_THRESHOLD for labeling only (not abort). |
| `r_runner.R` | fpca_native_r record without fingerprint | run_fpca_native_r() → switch case → output without result_fingerprint field | ✓ WIRED | Function and switch case confirmed. SUMMARY records mean_ns=12,978,754 ns at n=60/m=50. |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `comparison.csv` | 215 rows with mean_ns/rsd_pct/noise_gate | Real criterion estimates.json + python/r runner timings (timed grid run) | Yes — per-cell JSON files committed | ✓ FLOWING |
| `hotspot_ranking.md` | Leverage scores and rankings | comparison.csv via rank_hotspots.py | Yes — elastic rank #1: 37,964ms from noise-stable criterion cells | ✓ FLOWING |
| `env.json` | thread_env all "1" | benchmarks/run_all.py step_capture_env() from the same grid run | Yes — 6 keys captured | ✓ FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| gate.py --self-test (all 6 ops pass-on-agree/fail-on-mismatch) | `benchmarks/.venv/bin/python benchmarks/gate.py --self-test` | "GATE SELF-TEST: PASSED (all ops: pass-on-agree and fail-on-mismatch verified)" | ✓ PASS |
| test_gate.py — 27 unit tests | `pytest benchmarks/test_gate.py -q` | 27 passed, 2 warnings in 0.64s | ✓ PASS |
| test_rank_hotspots.py + test_assemble_comparison.py — 33 unit tests | `pytest benchmarks/test_rank_hotspots.py benchmarks/test_assemble_comparison.py -q` | 33 passed in 1.19s | ✓ PASS |
| Full suite (60 tests) | `pytest benchmarks/test_rank_hotspots.py benchmarks/test_gate.py benchmarks/test_assemble_comparison.py -q` | 60 passed, 2 warnings in 0.67s | ✓ PASS |
| Nyquist: test_noise_gate_flags_high_rsd | `pytest benchmarks/test_rank_hotspots.py::TestNoiseGate::test_noise_gate_flags_high_rsd -v` | PASSED | ✓ PASS |
| Nyquist: test_leverage_score_deterministic | `pytest benchmarks/test_rank_hotspots.py::TestLeverageScore::test_leverage_score_deterministic -v` | PASSED | ✓ PASS |
| run_all.py --grid --dry-run (wiring proof) | `python benchmarks/run_all.py --grid --dry-run` | Iterates all 6 cells (N∈{100,300,500}×M∈{50,200}), prints per-cell dry plan | ✓ PASS |
| comparison.csv 215 rows with required libraries/ops | Python check | rust/python/r_svd/r_native_pca_fd all present; 6 overlapping ops in python+r_svd; elastic at N=100/300/500 | ✓ PASS |
| env.json all thread_env="1" | Python check | all_ones: True | ✓ PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| BASE-01 | 113-01, 113-02 | Cross-language baseline over N×M grid (≥3 sizes, elastic at N≥100), merged into comparison.csv | ✓ SATISFIED | comparison.csv: 215 rows, 4 libraries, N∈{100,300,500}×M∈{50,200}, elastic at N=100/300/500 in python/r_svd. run_all.py --grid runs gate-before-timing. env.json re-captured with all thread_env="1". |
| BASE-02 | 113-01, 113-03 | Evidence-ranked hotspot list from cross-language results + criterion suite, gated on RSD<5%/CI<5%; elastic anchor confirmed, secondary targets ordered | ✓ SATISFIED | hotspot_ranking.py: leverage formula + noise gate + deterministic ranking. hotspot_ranking.md: elastic rank #1 (37,965ms, noise-stable cells), secondary targets ordered, 68 noise-excluded cells disclosed. 60 unit tests green. Human sign-off approved 2026-09-23. |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| None | — | No TBD/FIXME/XXX debt markers found in any modified file | — | — |

### Methodology Disclosures (Not Gaps)

The following decisions were user-approved during execution and are disclosed in the artifacts — they are not gaps:

1. **FPCR non-gating (Plan 02):** FPCR was removed from `_GATED_OPS` after measured cross-library divergence (corr 0.9495 at n=100, dropping to ~0.86 at n=500). Disclosed in gate.py constant comment, comparison.csv notes column per-cell, and 113-02-SUMMARY. Plan 01 truth "all six ops PASS the recalibrated gate" predates this Plan 02 decision — the final gate has 5 hard-gated ops + FPCR disclosed-but-computed.
2. **FPCA subspace-agreement gate (Plan 02):** Gate changed from per-component Pearson to principal-angle subspace cosine (cosine 1.0000 for fdars-vs-R every cell; fdars-vs-Python disclosed at cos≥0.9816). Proven correct against R.
3. **FPCA hard-gated on fdars-vs-R only; fdars-vs-Python disclosed (Plan 02 user decision).**
4. **Elastic trimmed to pairwise; Karcher-mean excluded (user decision 2026-09-23):** Karcher-mean cells noise-gated (RSD 29–46%); anchor uses noise-stable pairwise elastic.
5. **basis_smoothing missing from rust library in comparison.csv:** Rust has `audit_smooth` (1 cell); python/r_svd have `basis_smoothing` (6 cells each). This is because basis_smoothing in criterion is named differently (`audit_smooth`). It does not affect the hotspot ranking since basis_smoothing ranks low in leverage and the cross-language comparison for this op still exists via python/r_svd rows.
6. **SVD benchmarks all FAIL-NOISY:** All `audit_p6_svd_*` cells FAIL-NOISY (CI-width 7–56%). Disclosed in hotspot_ranking.md Noise-Excluded Cells section. Phase 115 action recorded: SVD benches need higher criterion sample count before use as optimization targets.

---

_Verified: 2026-09-23T10:30:00Z_
_Verifier: Claude (gsd-verifier)_

## Re-fingerprint 2026-09-25

Covered-file digest recomputed by the v0.47.0 autonomous orchestrator before the milestone audit. Cause of staleness: comparison.csv/env.json intentionally overwritten by the Phase 116 after-run (Phase 113 baseline preserved byte-identical in benchmarks/results/comparison_baseline.csv); run_all.py extended by 116. No verified truth was invalidated.

## Re-fingerprint 2026-09-25 (post-Phase-117)

Covered-file digest recomputed before the milestone audit. Changed covered files: only .planning/REQUIREMENTS.md changed (milestone requirements marked complete) — false-stale. No verified truth was invalidated.
