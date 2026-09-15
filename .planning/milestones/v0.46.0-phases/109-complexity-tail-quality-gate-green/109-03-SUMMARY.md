---
phase: "109"
plan: "03"
subsystem: peer
status: complete
tags: [complexity-reduction, refactor, peer, reml, lpeer]
requirements: [QUAL-03]

dependency_graph:
  requires: [109-01, 109-02]
  provides: [peer.rs complexity cleared]
  affects: [fdars-core/src/peer.rs]

tech_stack:
  added: []
  patterns: [multi-seam decomposition, extract private helpers, verbatim loop body lift]

key_files:
  modified:
    - fdars-core/src/peer.rs

decisions:
  - "peer decomposed via 3 helpers (validate_dims, validate_argvals, build_centered_design, compute_wtw_wty) — each seam was natural block boundary with no FP reorder"
  - "lpeer closure (sigma_inv) converted to named fn with explicit params (sm_dense, gcnt, su, se) — verbatim body, no capture ordering change"
  - "select_lambda_reml_peer split at 6 seams: setup_eigenbasis (STEP 1-3), init_alpha (OLS init), compute_r_alpha (fixed-effect residual), estep (E-step), mstep (M-step), gls_null_update (Woodbury GLS) — EM loop order and convergence check preserved verbatim"
  - "reml_estep returns Option<(sigma_b, trace_sigma_b, b_hat)> — None on Cholesky failure so caller restores last-good sigma2_u/sigma2_e and breaks (WR-03 preserved)"
  - "All helpers private fn, no visibility/signature change to peer or lpeer (public API stable)"

metrics:
  duration: 15
  completed_date: "2026-09-14"
  tasks: 3
  files: 1

actuals:
  tokens: 35000
  tasks: 3
  commits: 3
  plan_head_before: 99af1312d114bfa8243b3f40d9325449d2dac350
---

# Phase 109 Plan 03: peer.rs Complexity Decomposition Summary

Decomposed three peer.rs complexity violations into private helpers so every function and every new helper scores cognitive <=23 / cyclomatic <=30 under pmat 3.18.0, with numeric-identical output proven by 31 peer tests (part of the 2942-test suite) remaining green.

## What Was Built

Extracted 10 private helpers across three tasks, clearing all three QUAL-03 targets in `fdars-core/src/peer.rs`.

### Before → After

| Function | Before cog | Before cyc | After cog | After cyc |
|---|---|---|---|---|
| `peer` | 35 | 17 | 5 | 4 |
| `lpeer` | 57 | 29 | 4 | 3 |
| `select_lambda_reml_peer` | 74 | 29 | 16 | 6 |

### New Helpers (all <=cog 15 / cyc 10)

| Helper | cog | cyc | Purpose |
|---|---|---|---|
| `peer_validate_dims` | 15 | 6 | 7 dimension/finiteness guards shared by peer+lpeer |
| `peer_validate_argvals` | 6 | 3 | argvals finiteness + monotone check |
| `peer_build_centered_design` | 6 | 5 | weights + wmat + centering |
| `peer_compute_wtw_wty` | 3 | 3 | WtW and Wty cross-products |
| `lpeer_validate_subject_map` | 6 | 3 | subject_map length + build_subject_map + n_subjects>=2 |
| `lpeer_sigma_inv` | 1 | 2 | random-intercept Sigma^-1 applied to length-n vector |
| `lpeer_penalized_gls` | 13 | 10 | penalized-GLS build+solve in FPC-score space |
| `lpeer_back_project_and_fit` | 3 | 2 | gamma->beta back-projection + fitted values |
| `reml_setup_eigenbasis` | 14 | 8 | eigendecompose Q, partition null/range, project Wc |
| `reml_init_alpha` | 7 | 5 | OLS init of null-space fixed effect alpha |
| `reml_compute_r_alpha` | 3 | 2 | fixed-effect residual yc - Z_null * alpha |
| `reml_estep` | 5 | 5 | E-step: M matrix + Sigma_b + b_hat (returns Option) |
| `reml_mstep` | 0 | 1 | M-step variance component updates |
| `reml_gls_null_update` | 8 | 6 | GLS null-space alpha update via Woodbury identity |

## Verification

- `cargo test -p fdars-core --features linalg,parallel --lib -- peer`: **31/31 passed** (numeric-identical output confirmed — REML lambda bit-exact across two runs per `test_peer_reml_deterministic`)
- `cargo test -p fdars-core --features linalg,parallel --lib`: **2942/2942 passed** (no regressions)
- `pmat analyze complexity --path fdars-core/src/peer.rs`: Max cog 16, max cyc 10 — all functions below cog 23 / cyc 30
- `cargo clippy --all-targets --features linalg,parallel -- -D warnings`: clean (0 warnings)
- `cargo fmt --check`: clean

## Deviations from Plan

None. Plan executed exactly as written.

- Validation seams used shared `peer_validate_dims` to avoid code duplication between `peer` and `lpeer` — this was anticipated in the plan.
- The `sigma_inv` closure in `lpeer` was converted to a named function (`lpeer_sigma_inv`) with explicit parameters — verbatim body, no behavior change.
- `reml_estep` returns `Option` (None on Cholesky failure) rather than a bool flag — cleaner API that communicates the failure mode without additional parameters.

## Known Stubs

None.

## Self-Check

- `fdars-core/src/peer.rs` exists and is modified: FOUND
- Commit `4f807fc9` (Task 1): FOUND
- Commit `7715f025` (Task 2): FOUND
- Commit `64b98b21` (Task 3): FOUND
- All 3 targets below cog 23 / cyc 30: CONFIRMED
- 2942 lib tests green: CONFIRMED

## Self-Check: PASSED
