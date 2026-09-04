---
phase: 66-core-peer-estimator-penalty-families
plan: "01"
subsystem: regression
tags: [peer, scalar-on-function, penalized-regression, cholesky, functional-data]

requires:
  - phase: none
    provides: shipped scalar_on_function/, function_on_scalar.rs (penalty_matrix), linalg.rs (Cholesky), helpers.rs (simpsons_weights)
provides:
  - "peer() structured-penalty scalar-on-function estimator (fixed λ)"
  - "PeerPenalty enum: Ridge, Difference{order}, Decree(Q)"
  - "PeerConfig / PeerResult public API"
affects: [67-automatic-lambda-selection, 68-longitudinal-peer-prediction-integration]

actuals:
  tokens: 5700
  tasks: 4
  commits: 1

tech-stack:
  added: []
  patterns: ["penalized normal equations (W_c'W_c + λQ)β = W_c'y_c via reused Cholesky", "penalty-family enum dispatch (build_q)"]

key-files:
  created: [fdars-core/src/peer.rs]
  modified: [fdars-core/src/lib.rs]

key-decisions:
  - "β(t) estimated pointwise on the argvals grid; design W[i,j] = data[(i,j)]·simpson_w[j]"
  - "Centered response + design; intercept = ȳ; effective_df = trace of hat matrix (A^{-1}WtW column-solve)"
  - "Difference restricted to order 2 (penalty_matrix builds order-2 only); other orders → InvalidParameter"
  - "Decree(Q) accepted as raw row-major symmetric matrix + dimension; wrong dim → InvalidDimension"
  - "Recovery test fixture uses spanning pseudo-random design (n=200 ≫ m=40) so W is full-rank and β(t) identifiable"

patterns-established:
  - "PeerPenalty enum + build_q dispatch: additive penalty families without changing the solver"

requirements-completed: [PER-01, PER-02]

coverage:
  - id: D1
    description: "peer() recovers a known β(t)=sin(π·t) within tolerance and returns a well-formed PeerResult (β, intercept, fitted, df, λ, penalty_type)"
    requirement: "PER-01"
    verification:
      - kind: unit
        ref: "fdars-core/src/peer.rs#test_peer_difference_beta_recovery, #test_peer_result_shape"
        status: pass
    human_judgment: false
  - id: D2
    description: "Three penalty families (Ridge, Difference{2}, Decree valid) each fit without error; unsupported Difference order → InvalidParameter"
    requirement: "PER-02"
    verification:
      - kind: unit
        ref: "fdars-core/src/peer.rs#test_peer_ridge_fits, #test_peer_decree_fits, #test_peer_difference_order_rejected"
        status: pass
    human_judgment: false
  - id: D3
    description: "Decree(Q) partition structure yields β(t) demonstrably distinct from plain roughness on the same data"
    requirement: "PER-02"
    verification:
      - kind: unit
        ref: "fdars-core/src/peer.rs#test_peer_decree_distinct_from_roughness"
        status: pass
    human_judgment: false
  - id: D4
    description: "Wrong-dim Q / mismatched argvals → descriptive FdarError (never panic); no NaN β(t) across all families"
    requirement: "PER-01"
    verification:
      - kind: unit
        ref: "fdars-core/src/peer.rs#test_peer_decree_wrong_dim, #test_peer_argvals_mismatch, #test_peer_no_nan_all_families"
        status: pass
    human_judgment: false

duration: 25min
completed: 2026-09-04
status: complete
---

# Phase 66 / Plan 01: Core PEER Estimator & Penalty Families — Summary

**Shipped a public `peer()` structured-penalty scalar-on-function regression estimator with three a-priori penalty families, recovering a known β(t) on synthetic data.**

## Performance

- **Duration:** ~25 min (executor dropped at wrap-up; finalized inline by orchestrator)
- **Completed:** 2026-09-04
- **Tasks:** 4 (all delivered in a single new file)
- **Files modified:** 2 (`fdars-core/src/peer.rs` new, `fdars-core/src/lib.rs` +1 line)

## Accomplishments

- New `fdars-core/src/peer.rs` (641 lines): `peer(data, y, argvals, &config) -> Result<PeerResult, FdarError>` fitting β(t) pointwise via penalized normal equations `(W_c'W_c + λQ)β = W_c'y_c` solved by the reused `linalg` Cholesky helpers.
- `PeerPenalty` enum (`Ridge`, `Difference{order}`, `Decree(Vec<f64>, usize)`) dispatched through a private `build_q`; roughness reuses `function_on_scalar::penalty_matrix`. `PeerConfig` / `PeerResult` follow crate conventions (`Debug, Clone, PartialEq`, `#[non_exhaustive]`, `#[must_use]`, conditional serde).
- Effective df via trace of the hat matrix (column-solve on `A = W'W + λQ`); intercept = ȳ; Simpson's-weighted functional design.
- Robust error/NaN surface: wrong-dim Q → `InvalidDimension`, unsupported Difference order → `InvalidParameter`, mismatched argvals → `InvalidDimension`, non-finite β → `ComputationFailed`; never panics.
- Registered `pub mod peer;` in `lib.rs`. Crate-root/prelude re-exports, doctest, `lpeer`, `predict` correctly DEFERRED to Phase 68.

## Task Commits

Executed across the plan's 4 tracer-first tasks; committed as one coherent feat commit after the executor subagent's connection dropped at wrap-up (before it committed) and the orchestrator finalized inline:

1. **Tasks 1–4 (peer.rs + lib.rs registration)** — `2d69c8bc` (feat)

## Files Created/Modified

- `fdars-core/src/peer.rs` — PEER estimator, penalty families, 9 inline tests.
- `fdars-core/src/lib.rs` — `pub mod peer;` (between `outliers` and `regression`).

## Decisions Made

- Kept the estimator formulation from the plan (design `W[i,j]=X_i(t_j)·w_j`, centered, Cholesky solve). The plan's RESEARCH-flagged assumptions A1/A2 are resolved: the β(t) recovery test passes with the correct formulation.

## Deviations from Plan

### Fixed: degenerate recovery-test fixture

- **Found during:** finalizing/verification (the tracer β(t) recovery test failed, error 0.685 then 0.162 ≥ 0.1).
- **Root cause:** the executor's fixture built curves `X_i(t_j)=sin(0.5·(0.3i+0.1j))`, a single-frequency phase-shifted sinusoid family that spans only a **2-D subspace** — so W had rank 2 and β(t) was unidentifiable in 38 directions (penalty bias dominated). Not an estimator bug.
- **Fix:** replaced with deterministic spanning pseudo-random design curves (stateless splitmix64 hash, no RNG dep) and raised `n` to 200 ≫ `m=40`, so `W` has full column rank, `W'W` is well-conditioned, and the fixed λ=1e-4 penalty bias falls well under the 0.1 recovery tolerance. β(t)=sin(π·t) now recovered. This corrects the fixture; the recovery tolerance and estimator were unchanged.
- Also fixed one clippy `doc_list_item_overindented` warning (doc comment reflow).

## Verification Evidence

- `cargo test -p fdars-core --lib --features linalg,parallel peer::` → **9 passed; 0 failed** (test_peer_difference_beta_recovery, test_peer_result_shape, test_peer_ridge_fits, test_peer_decree_fits, test_peer_difference_order_rejected, test_peer_decree_distinct_from_roughness, test_peer_decree_wrong_dim, test_peer_argvals_mismatch, test_peer_no_nan_all_families).
- `cargo clippy -p fdars-core --all-targets --features linalg,parallel` → clean (0 warnings).
- `cargo fmt -p fdars-core -- --check` → clean.

## Requirements Completed

- **PER-01** — public `peer()` estimator, β(t) via penalty null/range decomposition, `PeerResult` with β/intercept/fitted/df diagnostics.
- **PER-02** — three penalty families (Ridge, Difference{2} reusing `penalty_matrix`, caller-supplied Decree Q); structured Q yields partition-aware β(t) distinct from roughness.
