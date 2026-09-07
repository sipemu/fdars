---
schema: summary
plan: 82-02
phase: 82
requirements: [API-03]
status: complete
---

# Phase 82 / Plan 82-02 Summary — Add `#[non_exhaustive]` to 12 public types (API-03)

## What changed

Added `#[non_exhaustive]` (on its own line at the top of each attribute stack,
directly above the `#[derive(...)]` line, matching crate convention) to the 12
public types approved in the Phase 81 audit inventory (API-03). Attribute-only
change — no variants/fields/docs/derives touched, no behavior/numeric change.

### 10 enums (AUD-10)
| Type | Location |
|------|----------|
| `PeerPenalty` | `fdars-core/src/peer.rs:68` |
| `LambdaChoice` | `fdars-core/src/peer.rs:99` |
| `LambdaMethod` | `fdars-core/src/peer.rs:117` |
| `DesignCriterion` | `fdars-core/src/optimal_design.rs:92` |
| `OptimalityKind` | `fdars-core/src/optimal_design.rs:103` |
| `ExtrapolationPolicy` | `fdars-core/src/helpers.rs:884` |
| `ImputationMethod` | `fdars-core/src/helpers.rs:1015` |
| `SelectionCriterion` | `fdars-core/src/scalar_on_function/mod.rs:268` |
| `BasisType` | `fdars-core/src/smooth_basis.rs:22` |
| `BasisCriterion` | `fdars-core/src/smooth_basis.rs:1117` |

### 2 result structs (AUD-11)
| Type | Location |
|------|----------|
| `OptimBandwidthResult` | `fdars-core/src/smoothing.rs:544` |
| `KnnCvResult` | `fdars-core/src/smoothing.rs:767` |

## Catch-all `_ =>` arms
**None required.** After adding all 12 attributes the crate compiled with zero
errors and `clippy --all-targets` was clean — no in-crate `match` (including
the `Copy` enums `SelectionCriterion`/`BasisCriterion` and test code) needed a
catch-all arm. In-crate exhaustive matches remain exhaustive-by-intent, so no
future-variant warnings were suppressed.

## Files changed
- `fdars-core/src/peer.rs`
- `fdars-core/src/optimal_design.rs`
- `fdars-core/src/helpers.rs`
- `fdars-core/src/scalar_on_function/mod.rs`
- `fdars-core/src/smooth_basis.rs`
- `fdars-core/src/smoothing.rs`

## Gate results (whole-crate, workspace root)
1. `cargo fmt` + `cargo fmt --check` — clean
2. `cargo clippy --all-targets --features linalg,parallel -- -D warnings` — clean, zero warnings
3. `cargo test` — main lib suite `test result: ok. 2850 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out`. Only pre-existing environmental golden flakes `golden_co_cluster_below_threshold` / `golden_co_cluster_parallel` (`tests/equivalence_phase48.rs`) fail; confirmed base-failing before any Phase 82 edit — NOT a regression.
4. `cargo build --features serde` — success
5. `cargo build --examples` — success (all 28 compile)

## Commit
- `c435d863` — `refactor(82-02): add #[non_exhaustive] to 12 public types (API-03)`
