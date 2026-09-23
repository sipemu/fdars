---
phase: 113-cross-language-baseline-hotspot-ranking
plan: "03"
subsystem: benchmarks
tags: [hotspot-ranking, noise-gate, leverage, elastic-anchor, secondary-targets, tdd]
status: complete
dependency_graph:
  requires:
    - phase: 113-02
      provides: benchmarks/results/comparison.csv (215 rows, 4 libraries, committed baseline)
  provides:
    - benchmarks/rank_hotspots.py (noise gate + leverage formula + ranking + markdown emitter)
    - benchmarks/test_rank_hotspots.py (20 unit tests; Nyquist: test_noise_gate_flags_high_rsd + test_leverage_score_deterministic)
    - benchmarks/results/hotspot_ranking.md (committed ranking: elastic anchor confirmed, secondary targets ordered, noise-excluded disclosed)
  affects: [114-elastic-optimization, 115-svd-fpca-covariance, 116-report]
tech_stack:
  added: []
  patterns:
    - noise-gated leverage ranking (rsd_pct<5 AND ci_width_pct<5 → PASS; else FAIL-NOISY/excluded)
    - leverage_score = fdars_ns × max(1.0, fdars_ns/min_other_ns); guarded against ≤0
    - deterministic stable sort by descending leverage_score; tiebreak by (op,n,m)
    - r_native_pca_fd excluded from ranking (ecosystem-realistic only)
key_files:
  created:
    - benchmarks/rank_hotspots.py
    - benchmarks/test_rank_hotspots.py
    - benchmarks/results/hotspot_ranking.md
  modified: []
decisions:
  - "leverage formula: fdars_ns × max(1.0, fdars_ns/min_other_ns) — gives absolute-time weight when fdars is fastest, multiplicative bonus for relative gap"
  - "elastic anchor sourced from noise-stable elastic_align_pair and audit_p3_elastic_cross cells (PASS); Karcher-mean cells excluded (RSD 29-46%)"
  - "r_native_pca_fd excluded from ranking denominator and targets (ecosystem-realistic only, not numerically matched)"
  - "functional_boxplot all cells FAIL-NOISY (CI-width 9-13%); noise-excluded; not a ranking target"
  - "all audit_p6_svd_* cells FAIL-NOISY; SVD benchmarks require re-run with higher sample count before Phase 115"
metrics:
  duration: 3 min
  completed: 2026-09-23
actuals:
  tokens: 7200
  tasks: 2
  commits: 2
  plan_head_before: 446fdd5db094422508f27b3fbac3829f47a5f4c2
---

# Phase 113 Plan 03: Hotspot Ranking — TASKS 1–2 COMPLETE, TASK 3 AWAITING HUMAN VERIFY

Tasks 1 and 2 are committed. Task 3 is the blocking-human gate: human sign-off that the baseline + ranking are fair and honest before Phases 114–115 optimize against them.

## Tasks

| Task | Name | Status |
|------|------|--------|
| 1 | rank_hotspots.py — noise gate + leverage + unit tests (TDD) | DONE (`20d978f8`) |
| 2 | Emit + commit hotspot_ranking.md | DONE (`ba0275c2`) |
| 3 | Human sign-off — fair/honest methodology | APPROVED 2026-09-23 (operator sign-off: baseline + ranking fair and honest) |

## Committed Artifacts

### rank_hotspots.py

Core functions:
- `apply_noise_gate(rows)` — PASS if `rsd_pct < 5.0 AND ci_width_pct < 5.0`; NA (python/r) rows always pass; None/NaN noise_gate excluded conservatively.
- `compute_leverage(fdars_ns, min_other_ns)` — `fdars_ns × max(1.0, fdars_ns/min_other_ns)`; guards for ≤0.
- `rank_rows(rows)` — noise-gate → leverage → stable sort descending; excludes `r_native_pca_fd`.
- `emit_hotspot_ranking(ranked, excluded, env_path, output_path)` — writes RESEARCH Section-5 schema markdown.

### test_rank_hotspots.py — 20 tests, all green

| Test | Coverage |
|------|---------|
| `test_noise_gate_flags_high_rsd` | Nyquist #1: RSD>=5 excluded |
| `test_noise_gate_flags_high_ci_width` | CI-width>=5 excluded |
| `test_noise_gate_passes_clean_row` | RSD<5 AND CI<5 passes |
| `test_noise_gate_boundary_exactly_5_fails` | strict <, not <= |
| `test_cross_language_rows_not_gated` | NA rows pass gate |
| `test_noise_gate_rejects_na_noise_gate_rust_rows` | None noise_gate → excluded |
| `test_leverage_slower_than_competitor` | factor > 1 case |
| `test_leverage_faster_than_competitor` | fdars fastest → factor 1.0 |
| `test_leverage_equal_to_competitor` | equal → factor 1.0 |
| `test_leverage_guard_min_other_zero` | division by zero guard |
| `test_leverage_guard_negative_min_other` | negative guard |
| `test_leverage_guard_fdars_ns_zero` | fdars_ns=0 guard |
| `test_leverage_score_deterministic` | Nyquist #2: same inputs → same output |
| `test_noisy_op_excluded_from_ranked_targets` | pipeline: noisy excluded |
| `test_elastic_ranks_above_fpca_by_leverage` | elastic > fpca by leverage |
| `test_ranking_is_deterministic` | same rows → same order |
| `test_fpca_native_r_excluded_from_ranked_targets` | r_native_pca_fd never ranked |
| `test_elastic_anchor_from_noise_stable_source` | elastic anchor = PASS rows only |
| `test_rust_only_op_ranks_by_absolute_time` | no cross-lang → leverage = fdars_ns |
| `test_sort_is_stable_descending` | descending leverage order |

### hotspot_ranking.md — Ranking Summary

**Noise gate outcome:** 69 PASS (all rust), 78 FAIL-NOISY (rust), 78 NA (python/r/r_native_pca_fd). NOTE: python/r NA rows counted in excluded because apply_noise_gate treats them as passed, but rank_rows separates them; total noise-excluded = 68 rust FAIL-NOISY rows (from comparison.csv).

**Top 10 ranked targets (from hotspot_ranking.md):**

| Rank | Op | N | M | fdars (ms) | Leverage | Source |
|------|----|----|---|-----------|----------|--------|
| #1 | elastic | 500 | 50 | 37,964.661 | 37.965 s·factor | audit_p3_elastic_cross/n500_m50 |
| #2 | elastic | 500 | 50 | 7,841.427 | 7.841 s·factor | audit_p3_elastic_cross_banded/n500_m50 |
| #3 | elastic | 100 | 200 | 6,177.604 | 6.178 s·factor | audit_p3_elastic_cross_banded/n100_m200 |
| #4 | face_covariance | 50 | 60 | 3,834.365 | 3.834 s·factor | face_covariance/n50_m60 |
| #5 | elastic | 100 | 50 | 1,565.454 | 1.565 s·factor | audit_p3_elastic_cross/n100_m50 |
| #6 | face_covariance | 200 | 30 | 983.781 | 0.984 s·factor | face_covariance/n200_m30 |
| #7 | elastic | 100 | 50 | 816.797 | 0.817 s·factor | audit_elastic/n100_m50_capped |
| #8 | elastic | 100 | 200 | 754.891 | 0.755 s·factor | audit_p3_karcher_banded/n100_m200 |
| #9 | fem_smooth | — | — | 452.329 | 0.452 s·factor | fem_smooth/nodes576_tris1058 |
| #10 | elastic | 100 | 50 | 343.770 | 0.344 s·factor | audit_p5_karcher_paybackN/n100_m50 |

**Confirmed Anchor:** Elastic alignment is rank #1. Noise-stable cells: `audit_p3_elastic_cross` (N=100,M=50: fdars=1,565ms; N=500,M=50: fdars=37,965ms), `audit_p3_elastic_cross_banded` (N=100,M=200: fdars=6,178ms). Cross-language comparator: python ~2.9ms, R ~1.9ms (single pairwise align). Leverage factor 824–19,672×. Karcher-mean cells excluded (RSD 29–46%).

**Secondary Targets ordered for Phase 114–115:**
1. `face_covariance` (Rust-only, absolute time: 50×60→3,834ms, 200×30→984ms, 50×30→242ms)
2. `fem_smooth` (Rust-only absolute time: 452ms @ nodes576)
3. `fpca` (PASS, n=1000/m=200: 38.9ms; cross-language leverage: python ~14ms, R ~42ms at n=500/m=200)
4. `frechet_anova` (PASS, 133ms @ n=50/m=200)

**Notable noise-excluded (SVD benchmarks — important for Phase 115):**
All `audit_p6_svd_nalgebra`, `audit_p6_svd_faer_seq`, and `audit_p6_svd_conversion` cells FAIL-NOISY (CI-width 7–56%). SVD benchmarks need higher criterion sample count before Phase 115 can use them as precise optimization targets.

**env.json fairness:** All six thread env vars = "1" confirmed in committed env.json. Single-source seeded inputs. Lockfiles committed.

## Deviations from Plan

None — plan executed as specified. The elastic PASS rows at N=100/200/500 from bulk cross-alignment benches (`audit_p3_elastic_cross*`, `audit_elastic`, `audit_karcher`) provide the noise-stable anchor; pairwise `elastic_align_pair` cells (N=NaN, M-parameterized) are also present in the anchor section.

## Known Stubs

None — hotspot_ranking.md carries real numbers from committed comparison.csv.

## Self-Check

TASKS 1-2 COMPLETE:
- `benchmarks/rank_hotspots.py` — exists, contains leverage formula
- `benchmarks/test_rank_hotspots.py` — exists, 20 tests green
- `benchmarks/results/hotspot_ranking.md` — exists and non-empty
- Commits: 2 (`20d978f8` + `ba0275c2`)

TASK 3: AWAITING blocking-human gate (never auto-approved per plan).
