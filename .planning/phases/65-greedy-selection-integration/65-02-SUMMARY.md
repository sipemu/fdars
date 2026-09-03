# Phase 65 — Summary 65-02: Re-exports, Doctest + Criterion Benchmark

**Status:** complete
**Requirements:** FOD-05
**Commit (impl):** b925fac9

## Files
- **Modified** `fdars-core/src/lib.rs` — extended the Phase-64 `pub use optimal_design::{...}` line to the full six-symbol surface.
- **Modified** `fdars-core/src/prelude.rs` — new FOptDes re-export block after the k-Shape block.
- **Modified** `fdars-core/src/optimal_design.rs` — module-level `//!` end-to-end doctest.
- **Created** `fdars-core/benches/optimal_design.rs` — criterion 0.5 benchmark.
- **Modified** `fdars-core/Cargo.toml` — `[[bench]] name = "optimal_design"` stanza after the kshape stanza.

Additive/non-breaking: no existing signature changed, no new dependency, MSRV 1.81 preserved.

## Public API surface finalized
`lib.rs` crate root and `prelude` now both re-export:
```rust
pub use optimal_design::{
    design_criterion, optimal_design, DesignCriterion, OptDesConfig, OptDesResult, OptimalityKind,
};
```

## Implementation notes
- **Doctest** (module-level `//!`): builds a 6-curve sparse `IrregFdata::from_lists`, fits `pace_fpca` on a 21-point work grid (ncomp 2), then `optimal_design` with `candidate_grid = model.argvals.clone()`, budget 2, Trajectory; asserts `selected_indices.len() == 2`, `criterion_trace.len() == 2`, and reads `selected_argvals`. Imports through the crate-root surface (validates 65-02 Task-1 reachability). Uses the real fit path because `PaceFpcaResult` is `#[non_exhaustive]`.
- **Benchmark**: external benches crate → builds the model via `pace_fpca` (not a struct literal). One `"optimal_design"` group with four `black_box`-wrapped functions: `design_criterion` (Trajectory, Score(A)) on a fixed 5-point design, and `optimal_design` (Trajectory, Score(A)) at budget 5 over a 51-point grid. Small dataset keeps compile/bench fast.

## Divergences
- The doctest and bench closures needed an explicit `|&t: &f64|` type annotation on `t.sin()` (inference otherwise ambiguous). Doctest failed once on this (E0689), fixed with the annotation. No behavior change.
- `cargo fmt` reformatted the bench's `values_list` closure to a block body — cosmetic only.

## Gate tails (whole-crate, all clean)
- `cargo fmt -p fdars-core --check` → clean (exit 0).
- `cargo clippy -p fdars-core --all-targets --features linalg,parallel -- -D warnings` → Finished, no warnings (covers the new bench).
- `cargo build --benches -p fdars-core --features linalg,parallel` → Finished (bench compiles; not run).
- `cargo test -p fdars-core --features linalg,parallel` → lib **2685 passed, 0 failed**; full integration + doctest suites 0 failed.
- `cargo test -p fdars-core --doc --features linalg,parallel` → **196 passed, 0 failed, 4 ignored** (includes the new `optimal_design` module doctest).

`--features serde` NOT run (pre-existing unrelated ClassifFit break, not a Phase 65 regression). `cargo bench` NOT run in CI (too slow) — compile-check only.

## Result
FOD-05 complete: full crate-root + prelude FOptDes surface, a passing end-to-end module doctest, and a registered criterion benchmark. 28 examples + WASM + R bindings unaffected.
