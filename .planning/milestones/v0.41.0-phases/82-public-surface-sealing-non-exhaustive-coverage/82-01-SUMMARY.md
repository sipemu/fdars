---
schema: summary
plan: 82-01
phase: 82
requirements: [API-02]
status: complete
---

# Phase 82 / Plan 82-01 Summary — Seal accidental `pub` helpers (API-02)

## What changed

Two accidentally-public internal helpers were sealed to `pub(crate)`, per the
Phase 81 audit inventory (API-02). Pure visibility change — no rename, no
signature change, no behavior/numeric change.

| Item | Location | Change | AUD |
|------|----------|--------|-----|
| `sort_nan_safe` | `fdars-core/src/helpers.rs:10` | `pub fn` → `pub(crate) fn` | AUD-07 |
| `solve_gaussian_pub` | `fdars-core/src/smoothing.rs:336` | `pub fn` → `pub(crate) fn` | AUD-08 |

Neither is re-exported in `lib.rs`/`prelude.rs`. All in-crate callers
(including `concurrent_regression.rs:226` and `scalar_on_function/cv.rs:36`
for `solve_gaussian_pub`, and the many callers of `sort_nan_safe`) stay
visible under `pub(crate)` — zero caller edits required. The `_pub` suffix on
`solve_gaussian_pub` was intentionally left (rename is deferrable per CONTEXT;
the private `solve_gaussian` already owns the bare name).

## Files changed
- `fdars-core/src/helpers.rs`
- `fdars-core/src/smoothing.rs`

## Gate results (whole-crate, workspace root)
1. `cargo fmt` + `cargo fmt --check` — clean
2. `cargo clippy --all-targets --features linalg,parallel -- -D warnings` — clean, zero warnings
3. `cargo test` — main lib suite `test result: ok. 2850 passed; 0 failed; 0 ignored`. The only failures are the two pre-existing environmental golden flakes `golden_co_cluster_below_threshold` / `golden_co_cluster_parallel` in `tests/equivalence_phase48.rs`, confirmed base-failing (identical values) on a `git stash` of the source edits — NOT a regression of this plan.
4. `cargo build --features serde` — success
5. `cargo build --examples` — success (all 28 compile)

## Commit
- `681d84ff` — `refactor(82-01): seal accidental pub helpers to pub(crate) (API-02)`
