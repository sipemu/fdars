---
schema: verification
phase: 82
requirements: [API-02, API-03]
status: passed
score: 4/4
verified: 2026-09-07
verifier: orchestrator (goal-backward, independently grep-confirmed)
---

# Phase 82 Verification — Public-Surface Sealing & Non-Exhaustive Coverage

Goal-backward check of the 4 ROADMAP success criteria against the codebase (not SUMMARY narrative). All change sites independently confirmed via grep by the orchestrator.

## Criterion 1 — Accidental `pub` sealed per approved inventory (API-02): PASS
- `sort_nan_safe` (`src/helpers.rs:10`) → `pub(crate) fn` ✓ (AUD-07)
- `solve_gaussian_pub` (`src/smoothing.rs:336`) → `pub(crate) fn` ✓ (AUD-08; not renamed, per plan — private `solve_gaussian` owns that name)
- Both internal-only, not re-exported; all in-crate callers still resolve (crate compiles). No example/test/doctest usage of the public path existed, so nothing broke.

## Criterion 2 — `#[non_exhaustive]` coverage corrected per inventory (API-03): PASS
Grep confirms `#[non_exhaustive]` immediately precedes all 12 approved types:
- 10 enums: `PeerPenalty`, `LambdaChoice`, `LambdaMethod` (peer.rs); `DesignCriterion`, `OptimalityKind` (optimal_design.rs); `ExtrapolationPolicy`, `ImputationMethod` (helpers.rs); `SelectionCriterion` (scalar_on_function/mod.rs); `BasisType`, `BasisCriterion` (smooth_basis.rs) ✓ (AUD-10)
- 2 result structs: `OptimBandwidthResult`, `KnnCvResult` (smoothing.rs) ✓ (AUD-11)

## Criterion 3 — No numeric/behavioral change (visibility/attribute-only): PASS
Edits are visibility keyword + attribute additions only. **0** `_ =>` catch-all arms were required (crate + all targets compiled with no non-exhaustive-pattern errors), so no match behavior was altered and no future-variant warnings suppressed.

## Criterion 4 — Whole-crate gates green + 28 examples compile: PASS
- `cargo fmt --check` clean
- `cargo clippy --all-targets --features linalg,parallel -- -D warnings` clean (zero warnings)
- `cargo test` lib suite `test result: ok. 2850 passed; 0 failed`
- `cargo build --features serde` success
- `cargo build --examples` success (28/28)

**Note:** The full `cargo test` run showed the pre-existing environmental golden flake (`golden_co_cluster_*`) — confirmed NOT a Phase 82 regression (byte-identical failure on git-stashed pristine source; passes per-binary in isolation). Tracked separately for the Phase 85 gate / STAB-03 checklist. Not a gap for this phase (API-shape only, no numeric change).

## Scope discipline
Out-of-scope items correctly untouched: `wire` module still public (AUD-09/13), config-struct non_exhaustive deferred (AUD-12).

## Verdict
Phase goal ACHIEVED. status: passed (4/4). Ready for Phase 83.
