---
phase: 113-cross-language-baseline-hotspot-ranking
plan: "02"
subsystem: benchmarks
tags: [benchmark-harness, grid-baseline, comparison-csv, fpcr-disclosure, fpca-quadrature, subspace-agreement, elastic-pairwise]
status: complete
dependency_graph:
  requires:
    - phase: 113-01
      provides: recalibrated gate.py (depth/boxplot corr, FPCR threshold 0.95), fpca_native_r in r_runner.R, criterion output for fregre_lm + functional_boxplot
  provides:
    - benchmarks/run_all.py (--grid mode: N×M loop, per-cell gate-before-timing, per-cell result files)
    - benchmarks/gate.py (FPCA subspace-agreement metric; FPCR disclosed/non-gating; five ops hard-gated)
    - benchmarks/assemble_comparison.py + test_assemble_comparison.py (schema CSV merge with disclosure notes)
    - benchmarks/results/comparison.csv (committed cross-language baseline, 215 rows, 4 libraries)
    - benchmarks/results/env.json (re-captured from the committed run; threads pinned to 1)
affects: [113-03-hotspot-ranking, 114-elastic-optimization, 115-svd-fpca-covariance, 116-report]
---

# Phase 113 Plan 02: Cross-Language Grid Baseline + comparison.csv — COMPLETE

Committed cross-language before-baseline over the N×M grid, gate-clean, with four user-approved/orchestrator-directed methodology decisions applied and fully DISCLOSED in the CSV. The two gsd-executor stalls on the ~10-min timed grid run were recovered by finishing the grid run inline from the orchestrator main loop; the pipeline is fully reproducible via `run_all.py --grid` → `assemble_comparison.py --gate-passed`.

## Tasks

| Task | Name | Status |
|------|------|--------|
| 1 | `--grid` mode in run_all.py (per-cell sequential generate→run→gate) | DONE (`9f3f7cb5`) |
| 2 | assemble_comparison.py + TDD unit tests (13) | DONE (`f036bf8a`) |
| 3 | Committed grid baseline run + comparison.csv + env.json | DONE (baseline commit) |

## Methodology decisions (all disclosed in comparison.csv `notes`)

1. **FPCR — keep, disclose, do NOT hard-gate** (user 2026-09-22). scikit-fda FPCA→OLS vs R fda.usc `fregre.pc` diverge (corr 0.9495 @ n=100 → ~0.86 at n=500). Removed from `_GATED_OPS`; correlation computed + recorded per cell; never aborts. (`79146973`)
2. **FPCA gate = subspace agreement (principal angles), not per-component Pearson** (orchestrator-directed; correct metric for near-degenerate eigenspaces, hard-gated at 0.99). (`5480dd1c`)
3. **FPCA hard-gated on fdars-vs-R only; fdars-vs-Python DISCLOSED** (user 2026-09-23). fdars matches R exactly (subspace cos 1.0000 every cell — fdars proven correct); scikit-fda trapezoidal quadrature tilts trailing PCs (fdars-vs-python dips to 0.9816 @ n=500,m=50). Python cosine recorded per cell; FPCA also excluded from the redundant 6b Python-vs-R check.
4. **Elastic trimmed to pairwise; Karcher-mean excluded** (user 2026-09-23). Karcher (R fdasrvf time_warping, ~10 min/cell at m=200) is noise-gated out of ranking anyway (RSD 29–46%); anchor uses fast N-independent pairwise `elastic_align_pair`. Recorded as a disclosure note.

## Gate outcomes — all 6 cells PASS the five hard-gated ops

Grid N ∈ {100,300,500} × M ∈ {50,200}. Every cell: fdars-vs-R FPCA subspace = 1.0000 (hard-gate PASS); basis_smoothing / depth / functional_boxplot / pairwise-elastic pass. Disclosed (non-blocking):

| Cell | FPCA fdars-vs-python cos | FPCR Py-vs-R corr |
|------|-------------------------|-------------------|
| n=100,m=50 | 0.9992 | 0.9495 |
| n=100,m=200 | 0.9998 | — |
| n=300,m=50 | 0.9986 | 0.9236 |
| n=300,m=200 | 0.9998 | 0.9236 |
| n=500,m=50 | 0.9816 | 0.8585 |
| n=500,m=200 | 0.9994 | — |

## comparison.csv (committed before-baseline)

- **215 rows**; libraries `rust`, `python`, `r_svd`, `r_native_pca_fd`. Schema `op,library,n,m,mean_ns,ci_lower_ns,ci_upper_ns,std_dev_ns,rsd_pct,ci_width_pct,noise_gate,source,notes`.
- Rust rows span the full criterion suite (for BASE-02 ranking); python/r/r_native rows cover the six overlapping ops on the grid (for BASE-01 head-to-head).
- Rust noise-gate: 69 PASS / 68 FAIL-NOISY. `env.json` all six `*_NUM_THREADS = "1"`.

## Phase 116 report findings

- **FPCR cross-library divergence** (~0.95 → 0.86 as n grows) — disclosed, not gated.
- **FPCA quadrature divergence** (fdars/R Simpson's vs scikit-fda trapezoidal) — leading PCs agree, trailing near-degenerate PCs diverge at coarse grids (cos ≥0.9816); fdars validated correct vs R (1.0000).
- **Karcher-mean elastic** excluded (noise-gated); pairwise elastic is the anchor.
- **`r_native_pca_fd`** (fda::pca.fd, B-spline coeff space) timed separately + NOT gated — the ecosystem-realistic R FPCA number, distinct from SVD-matched `r_svd`.

## Deviations / operational notes

- Two gsd-executor 600s stream-watchdog stalls on the timed grid; recovered by backgrounding the grid from the orchestrator loop (no watchdog) and polling. Fully reproducible pipeline.
- Pipeline bug fixed mid-run: the grid loop persisted stale pre-gate `py_records`, dropping the FPCA disclosure from per-cell files → reload post-gate (`fix(113-02): persist post-gate py_records`).
- `benchmarks/results/*` is gitignored; committed baseline artifacts force-added (`git add -f`) intentionally as the reproducible reference for Phases 114–116.

## Next

Plan 03: noise-gated leverage ranking → committed `hotspot_ranking.md` (confirm elastic anchor, order SVD/FPCA + covariance) + blocking-human fair/honest sign-off.
