---
phase: 49-code-consolidation-dedup
plan: 02
subsystem: regression
tags: [refactor, dedup, consolidation, svd-sign, fpca, pace-fpca, golden-equivalence, CONS-01]
requires:
  - phase: 49
    plan: 01
    provides: tests/equivalence_phase49.rs shared golden harness (append target); bit-identical assert_eq! pattern under both feature configs
provides:
  - "src/regression.rs::dominant_sign_negative — the single pub(crate) SVD/eigendecomposition sign-DECISION core (max-abs-index + <0.0 test)"
  - "fix_svd_signs (FPCA rotation+scores two-matrix lockstep flip) gates from the shared core"
  - "pace_fpca::eigendecompose_cov (eigenfunction-only single-matrix flip) gates from the shared core"
  - CONS-01 SVD-sign-fix consolidation proven bit-identical (FPCA + pace_fpca) under both feature configs
affects: [49-03, 49-04]
actuals:
  tokens: 34000
  tasks: 2
  commits: 3
tech-stack:
  added: []
  patterns:
    - "shared sign-DECISION core, per-site flip policy: one pub(crate) dominant_sign_negative owns the max-abs-index + <0.0 rule; the two call sites keep their OWN flip arity (two-matrix vs single-matrix)"
    - "golden-equivalence capture-then-assert (assert_eq! bit-identical, NOT tolerance — code-motion)"
key-files:
  created: []
  modified:
    - fdars-core/src/regression.rs
    - fdars-core/src/pace_fpca.rs
    - fdars-core/tests/equivalence_phase49.rs
key-decisions:
  - "Sign-DECISION core lives in regression.rs (NOT distributions.rs — that module is numerical-tails-only per Open Question 2). New helper: pub(crate) fn dominant_sign_negative(col, k, nrows) -> bool."
  - "pace_fpca CANNOT call fix_svd_signs (different arity — no scores matrix exists at eigendecompose time; BLUP scores are computed later). It gates its OWN single-matrix eigenfunction flip from the shared decision core instead."
  - "fix_svd_signs kept as-is at the flip level (two-matrix rotation+scores lockstep); only its inline max-abs + <0.0 decision was replaced by a call to dominant_sign_negative."
  - "Bit-identical by construction: same max-abs tie-break (max_by on .abs().partial_cmp with unwrap_or(Equal), empty-range fallback index 0) and same <0.0 comparison, extracted verbatim. No public signature change; no new dependency; helper is pub(crate)."
  - "Goldens driven through the PUBLIC entry points (fdata_to_pc_1d, pace_fpca) reading public result fields (FpcaResult.rotation/scores, PaceFpcaResult.eigenfunctions) — no new #[doc(hidden)] forwarders needed for this plan."
requirements-completed: [CONS-01 (SVD-sign-fix target)]
coverage:
  - id: S1
    description: "FPCA rotation+scores signs bit-identical before and after the refactor (two-matrix lockstep flip), both feature configs"
    requirement: CONS-01
    verification:
      - kind: integration
        ref: "cargo test --test equivalence_phase49 svd_sign under --features linalg,parallel AND --no-default-features --features linalg => svd_sign_fpca_two_matrix_bit_identical pass both configs"
        status: pass
    human_judgment: false
  - id: S2
    description: "pace_fpca eigenfunction signs bit-identical before and after the refactor (single-matrix flip), both feature configs"
    requirement: CONS-01
    verification:
      - kind: integration
        ref: "svd_sign_pace_eigenfunctions_single_matrix_bit_identical pass both configs"
        status: pass
    human_judgment: false
  - id: S3
    description: "The sign rule lives in ONE pub(crate) core; fix_svd_signs and pace_fpca both gate from it"
    requirement: CONS-01
    verification:
      - kind: integration
        ref: "regression.rs::dominant_sign_negative called by fix_svd_signs and pace_fpca::eigendecompose_cov; inline decision loops removed at both sites"
        status: pass
    human_judgment: false
  - id: S4
    description: "Wave gate: full suite green both configs + clippy --all-targets clean; no public signature change; no new dependency"
    requirement: CONS-01
    verification:
      - kind: lint
        ref: "cargo clippy --all-targets --features linalg,parallel -- -D warnings => clean"
        status: pass
      - kind: integration
        ref: "cargo test both feature configs => 2583 lib each, 0 fail"
        status: pass
    human_judgment: false
status: complete
---

# Phase 49 Plan 02: SVD Sign-Fix Consolidation (CONS-01) Summary

Consolidated the SVD/eigendecomposition sign convention — "for each component k, make the
largest-absolute-value entry positive" — which previously lived in `regression.rs::fix_svd_signs`
(flips FPCA rotation AND scores in lockstep) and was MIRRORED inline in
`pace_fpca.rs::eigendecompose_cov` (flips eigenfunctions only). The two copies were a drift risk.
Per RESEARCH, `pace_fpca` cannot call `fix_svd_signs` directly — the arities differ: at
eigendecompose time there is NO scores matrix (BLUP scores are computed later), so pace_fpca does a
single-matrix flip while `fix_svd_signs` does a two-matrix lockstep flip.

The fix extracts the shared sign-DECISION core (the max-abs-index + `< 0.0` test) into ONE
`pub(crate)` helper in `regression.rs` — **not** `distributions.rs`, which is numerical-tails-only.
Each call site then keeps its OWN flip policy but gates that flip from the single decision. This is
bit-identical by construction (pure sign flips, identical comparison rule), verified by assert_eq!
goldens under both feature configs.

## What shipped

- **`src/regression.rs`**: new `pub(crate) fn dominant_sign_negative(col: &FdMatrix, k: usize,
  nrows: usize) -> bool` — the single home for the sign rule (max-abs-index via `max_by` on
  `.abs().partial_cmp()` with `unwrap_or(Equal)` and empty-range fallback `0`, then the `< 0.0`
  test — extracted verbatim). `fix_svd_signs` now delegates its decision to it and keeps only the
  two-matrix rotation+scores lockstep flip.
- **`src/pace_fpca.rs`**: the inline eigenfunction-sign block in `eigendecompose_cov` now gates from
  `crate::regression::dominant_sign_negative`, keeping the single-matrix flip (no scores matrix at
  that point). The mirrored decision loop is gone.
- **`tests/equivalence_phase49.rs`** (appended): two `svd_sign` goldens capturing the exact
  pre-refactor f64 bits — `svd_sign_fpca_two_matrix_bit_identical` (FPCA rotation 8×3 + scores 5×3,
  driven through the public `fdata_to_pc_1d`) and
  `svd_sign_pace_eigenfunctions_single_matrix_bit_identical` (pace_fpca eigenfunctions 21×2, driven
  through the public `pace_fpca`). Both assert bit-identically (assert_eq!) under both feature configs.

## Commit count

3 atomic commits:
- `d936fa67` test(49-02): capture SVD-sign goldens (FPCA two-matrix + pace_fpca single-matrix)
- `95afbe6a` refactor(49-02): extract pub(crate) SVD sign-decision core (CONS-01)
- (this) docs(49-02): complete SVD sign-fix consolidation plan

(Task 1 = the test commit; Task 2 = the refactor commit; plus this summary/docs commit.)

## Golden results (both feature configs)

| Config | `equivalence_phase49 svd_sign` | Full suite (lib) | clippy --all-targets |
|--------|--------------------------------|------------------|----------------------|
| `--features linalg,parallel` | 2/2 pass | 2583 pass, 0 fail | clean |
| `--no-default-features --features linalg` | 2/2 pass | 2583 pass, 0 fail | clean |

FPCA rotation+scores signs and pace_fpca eigenfunction signs are BIT-IDENTICAL before and after the
refactor under both configs. `fix_svd_signs` still flips two matrices in lockstep; `pace_fpca` still
flips a single matrix — both now gate that flip from the one `dominant_sign_negative` decision core.

## Deviations from Plan

None. The plan's design (shared decision core in regression.rs, per-site flip policy, no move to
distributions.rs) was followed exactly. No architectural change, no signature change, no new
dependency. Goldens were driven through the public entry points, so no new `#[doc(hidden)]`
forwarders were required (the plan allowed either; the public route was sufficient).

## Known Stubs

None. Both flip sites call the real shared decision core; no placeholders.

## Threat Flags

None. Internal deterministic sign-convention refactor — no new network endpoint, auth path, file
access, or schema at a trust boundary. Threat T-49-03 (SVD sign-convention tampering) was mitigated
exactly as the register prescribed: assert_eq! bit-identical FPCA + pace_fpca sign goldens under both
feature configs, with single-matrix vs two-matrix flip preserved.

## Self-Check: PASSED

- `fdars-core/src/regression.rs` (dominant_sign_negative) — FOUND
- `fdars-core/src/pace_fpca.rs` (gates from shared core) — FOUND
- `fdars-core/tests/equivalence_phase49.rs` (svd_sign goldens) — FOUND
- Commit `d936fa67` — FOUND
- Commit `95afbe6a` — FOUND
