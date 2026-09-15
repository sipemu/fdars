---
phase: 106-top-level-complexity-hotspots
plan: "02"
subsystem: clustering_advanced
tags: [pmat, complexity-refactor, funfem, kcfc, clustering, fisher-em, kmeans-pp]

requires:
  - "106-01 (tracer: extract→test→pmat→commit loop proven)"
provides:
  - "funfem_cluster decomposed into 6 private helpers (validate, kmeans_pp_init, project_scores, gmm_disc_stats, e_step, subspace_update)"
  - "kcfc_cluster decomposed into 5 private helpers (validate, kmeans_pp_init, fit_cluster_fpca, reconstruction_errors, reassign)"
  - "Both target functions below pmat cognitive ≤25 / cyclomatic ≤30 thresholds"
  - "RNG draw order preserved exactly — seeded cluster assignments byte-identical"
affects: [106-03, 106-04]

actuals:
  tokens: 9164
  tasks: 2
  commits: 2
  plan_head_before: bb62a7f117b69e1be0091a4453e67879f67a88b0

tech-stack:
  added: []
  patterns:
    - "Extract-validate-and-call: 5-6 guard blocks extracted to private fn returning Result<(n,m), FdarError>"
    - "RNG-threading: k-means++ helpers take &mut StdRng and preserve exact draw sequence for numeric identity"
    - "Option-returning helpers: funfem_subspace_update returns Option<Vec<f64>>; kcfc_fit_cluster_fpca returns Option<FpcaResult> — parent rebinds cleanly without borrow aliasing"
    - "Owned-return helpers: kcfc_reconstruction_errors returns a new FdMatrix; funfem_e_step returns (Vec<f64>, f64) — eliminates double-borrow hazards"

key-files:
  created: []
  modified:
    - fdars-core/src/clustering_advanced.rs

key-decisions:
  - "funfem_subspace_update signature takes both n_k_new and _n_k_soft (latter unused but preserves future API room); the _ prefix silences clippy"
  - "kcfc_reassign returns bool (changed) and kcfc_cluster re-computes reconstruction_errors for the final Ok(KcfcResult) — avoids holding a reference to a prior-iteration value while the cluster vec is mutated"
  - "kcfc_kmeans_pp_init and funfem_kmeans_pp_init kept separate (RESEARCH.md Open Question 2): raw-curve D² with Simpson weights vs score-space Euclidean — distinct algorithms despite identical control flow"
  - "funfem_cluster cognitive score landed exactly at 25 (threshold) — no further split needed; all helpers are well below"

requirements-completed: [QUAL-01]

coverage:
  - id: D1
    description: "funfem_cluster cognitive complexity reduced from 167 to 25 (pmat 3.18.0) with 6 private helpers; test_funfem_deterministic confirms cluster assignments byte-identical"
    requirement: QUAL-01
    verification:
      - kind: unit
        ref: "cargo test -p fdars-core --features linalg,parallel --lib -- clustering_advanced::tests (32 tests, all pass)"
        status: pass
      - kind: other
        ref: "pmat analyze complexity --path fdars-core/src/clustering_advanced.rs: funfem_cluster cog=25 cyc=14"
        status: pass
    human_judgment: false
  - id: D2
    description: "kcfc_cluster cognitive complexity reduced from 93 to 11 (pmat 3.18.0) with 5 private helpers; test_kcfc_deterministic confirms cluster assignments byte-identical"
    requirement: QUAL-01
    verification:
      - kind: unit
        ref: "cargo test -p fdars-core --features linalg,parallel --lib -- clustering_advanced::tests (32 tests, all pass)"
        status: pass
      - kind: other
        ref: "pmat analyze complexity: kcfc_cluster cog=11 cyc=5"
        status: pass
    human_judgment: false
  - id: D3
    description: "Full test suite (2942 lib tests) green after both refactors; clippy --all-targets clean; cargo fmt clean"
    requirement: QUAL-01
    verification:
      - kind: unit
        ref: "cargo test -p fdars-core --features linalg,parallel: 2942 passed, 0 failed"
        status: pass
      - kind: other
        ref: "cargo clippy --all-targets --features linalg,parallel -- -D warnings: Finished (no errors)"
        status: pass
    human_judgment: false

duration: 15min
completed: "2026-09-14"
status: complete
---

# Phase 106 Plan 02: Top-Level Complexity Hotspots (Clustering) Summary

**Private-helper extraction of funfem_cluster (cog 167→25) and kcfc_cluster (cog 93→11) in clustering_advanced.rs — both the worst and second-worst clustering functions cleared, 11 new private helpers added, 2942 tests green**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-09-14
- **Completed:** 2026-09-14
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments

- `funfem_cluster` (cog 167 → 25, cyc 59 → 14): 6 private helpers extracted — `funfem_validate`, `funfem_kmeans_pp_init`, `funfem_project_scores`, `funfem_gmm_disc_stats`, `funfem_e_step`, `funfem_subspace_update`
- `kcfc_cluster` (cog 93 → 11, cyc 28 → 5): 5 private helpers extracted — `kcfc_validate`, `kcfc_kmeans_pp_init`, `kcfc_fit_cluster_fpca`, `kcfc_reconstruction_errors`, `kcfc_reassign`
- All 11 new helpers below cog 25 / cyc 30 (highest: `kcfc_reconstruction_errors` cog 21, `funfem_subspace_update` cog 23)
- RNG draw order preserved exactly in both k-means++ helpers — seeded cluster assignments are byte-identical (confirmed by `test_funfem_deterministic` and `test_kcfc_deterministic`)
- Full suite 2942 lib tests green; clippy --all-targets clean; cargo fmt clean
- No public API / signature / visibility change

## Task Commits

1. **Task 1: decompose funfem_cluster** — `931d2e87`
2. **Task 2: decompose kcfc_cluster + confirm file clean** — `6ef8d107`

## Files Created/Modified

- `fdars-core/src/clustering_advanced.rs` — 11 private helpers added; funfem_cluster and kcfc_cluster reduced to delegating coordinators; net +192 lines (helpers) / -115 lines (bodies inlined in parents)

## Decisions Made

- **kcfc_reassign returns bool, not a new cluster vec**: The original loop used a mutable `changed` flag. Extracting as `kcfc_reassign(&reconstruction_errors, &mut cluster, n, k) -> bool` preserves the in-place mutation pattern without borrow complications.
- **kcfc_reconstruction_errors returns FdMatrix by value**: Rather than passing `&mut FdMatrix` to a helper, the helper builds a fresh matrix. This avoids holding a mutable reference to the matrix while reading fpca_models, and matches the existing crate pattern of returning owned results from computation helpers.
- **funfem_e_step returns (Vec<f64>, f64)**: The original code initialized `resp` then filled it from `log_resp`. The helper returns a new `resp` Vec, which the parent rebinds — cleaner ownership than `&mut resp`.
- **funfem_subspace_update returns Option<Vec<f64>>**: On Cholesky failure or SVD u-unavailability, returns None; parent keeps previous disc_dirs. This was the A2 assumption in RESEARCH.md — confirmed: no borrow conflict.

## pmat Before → After

| Function | Baseline Cog | Final Cog | Baseline Cyc | Final Cyc |
|----------|-------------|----------|-------------|---------|
| `funfem_cluster` | **167** | **25** | **59** | 14 |
| `kcfc_cluster` | **93** | **11** | 28 | 5 |
| `funfem_validate` (new) | — | 16 | — | 7 |
| `funfem_kmeans_pp_init` (new) | — | 20 | — | 7 |
| `funfem_project_scores` (new) | — | 6 | — | 4 |
| `funfem_gmm_disc_stats` (new) | — | 22 | — | 11 |
| `funfem_e_step` (new) | — | 14 | — | 8 |
| `funfem_subspace_update` (new) | — | 23 | — | 13 |
| `kcfc_validate` (new) | — | 16 | — | 7 |
| `kcfc_kmeans_pp_init` (new) | — | 20 | — | 7 |
| `kcfc_fit_cluster_fpca` (new) | — | 7 | — | 5 |
| `kcfc_reconstruction_errors` (new) | — | 21 | — | 7 |
| `kcfc_reassign` (new) | — | 3 | — | 3 |

All functions at or below thresholds (cog ≤ 25, cyc ≤ 30). No new violations introduced.

## Deviations from Plan

None — plan executed exactly as written. The RESEARCH.md seams proved sufficient for both functions. No helper required further decomposition (unlike the tracer's 106-01 where `elastic_net_cd_sweep` needed an additional level). The `funfem_cluster` cognitive score landed at exactly 25 — on the boundary but within the allowed threshold.

## Known Stubs

None. All helper extractions are complete, production-quality private functions wired directly into their parents.

## Threat Flags

No new network endpoints, auth paths, file access patterns, or schema changes introduced. Pure internal refactor — no new security surface.

## Self-Check: PASSED

- File exists: `fdars-core/src/clustering_advanced.rs` — FOUND
- Commit 931d2e87 exists — FOUND (Task 1: funfem_cluster)
- Commit 6ef8d107 exists — FOUND (Task 2: kcfc_cluster + file clean)
- 2 commits measured from ledger (bb62a7f1..HEAD)
- pmat: funfem_cluster cog=25 ≤ 25, cyc=14 ≤ 30 — PASS
- pmat: kcfc_cluster cog=11 ≤ 25, cyc=5 ≤ 30 — PASS
- All 11 helpers below thresholds — PASS
- 2942 lib tests green — PASS
- clippy --all-targets clean — PASS
- cargo fmt clean — PASS

---
*Phase: 106-top-level-complexity-hotspots*
*Completed: 2026-09-14*
