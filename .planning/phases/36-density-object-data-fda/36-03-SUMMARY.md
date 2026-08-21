# Plan 36-03 Summary — LQD-FPCA + phase gate

**Requirement:** DENS-01 · **Wave:** 3 · **Status:** complete

## What was built

Added to `fdars-core/src/density_fda.rs`:
- `LqdFpcaResult { fpca: FpcaResult, fve: Vec<f64> }` — `#[non_exhaustive]`, derives Debug/Clone/PartialEq + conditional serde; embeds the reused `FpcaResult`.
- `lqd_fpca(density_matrix, argvals, ncomp) -> Result<LqdFpcaResult, FdarError>` — transforms each density row to LQD space on the common t-grid → assembles the LQD `FdMatrix` → delegates to `regression::fdata_to_pc_1d` → FVE = cumsum(sv²)/sum(sv²). Density-space modes deferred (rustdoc documents the manual `inverse_lqd` recipe).
- Crate-root re-export of `lqd_fpca`, `LqdFpcaResult`.

## Verification (phase-wide gate)

- **Full lib suite:** 2379 passed, 0 failed (`cargo test -p fdars-core --lib --features linalg,parallel`).
- **density_fda module:** 16/16 tests pass (round-trip, inverse-normalized, FVE monotone non-decreasing + reaches 1 at full rank, single-mode leading-PC capture, barycenter reduction, normalization integral-to-1, error paths).
- **Clippy:** `cargo clippy --all-targets --features linalg,parallel -- -D warnings` — clean (2 lints fixed inline: doc over-indentation, `manual_range_contains`).
- **rustfmt:** `cargo fmt --check` clean.
- **Cargo.toml:** `git diff --exit-code` clean — no new dependency.

## Notes / deviations

- The originating executor implemented all three waves' code in one pass, then dropped its connection before committing or writing summaries. The orchestrator recovered the uncommitted module: added the missing crate-root re-exports (`lqd_fpca`, `wasserstein_barycenter`, `LqdFpcaResult`), resolved the round-trip tolerance to an empirically-honest 1.5e-2 with a documented rustdoc divergence, fixed 2 clippy lints, ran the full gate green, and committed the work (`0f204417` feat + `2d81a217` fmt normalization of phase 34/35 source).
- **FVE convention** `cumsum(sv²)/sum(sv²)` matches the milestone FPCA convention.
