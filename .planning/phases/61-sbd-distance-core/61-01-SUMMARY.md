# Phase 61 — Summary 61-01: SBD Distance Core

**Status:** complete
**Requirements:** KSH-01, KSH-02
**Commit (impl):** 3f7bc35b

## Files
- **Created** `fdars-core/src/metric/sbd.rs` — SBD primitive + distance matrix + 8 inline tests + doctest.
- **Modified** `fdars-core/src/metric/mod.rs` — `pub mod sbd;` + `pub use sbd::{sbd, sbd_distance_matrix, SbdResult};`.
- **Created** `.planning/phases/61-sbd-distance-core/61-01-PLAN.md`.

No crate-root flat re-exports (deferred to Phase 63). No version bump (stays 0.33.0). No new dependency.

## Public API added
```rust
pub struct SbdResult { pub distance: f64, pub shift: isize }   // Debug/Clone/PartialEq, serde-gated, #[non_exhaustive]

#[must_use] pub fn sbd(x: &[f64], y: &[f64]) -> Result<SbdResult, FdarError>;
#[must_use] pub fn sbd_distance_matrix(data: &FdMatrix) -> Result<FdMatrix, FdarError>;
```
Reachable as `fdars_core::metric::{sbd, sbd_distance_matrix, SbdResult}`.

## Implementation notes
- Z-normalizes both inputs via `shapelet::z_normalize_window` (never trusts caller).
- `fft_len = next_power_of_two(2*max(x.len(),y.len()) − 1)`; zero-pad; `CC = Re(IFFT(FFT(x_z)·conj(FFT(y_z))))`.
- IFFT divided by `fft_len` (rustfft unnormalized) folded into `scale = 1/(fft_len·‖x_z‖·‖y_z‖)`.
- Scans only the 2m−1 meaningful lags; converts raw index → signed lag (`+k` for `0..m`, `k−fft_len` for the tail). `distance = 1 − clamp(max NCCc, −1, 1)` ∈ [0,2].
- Constant-series guard: `denom ≤ 1e-12` → `SbdResult { distance: 1.0, shift: 0 }` (never NaN).
- One `FftPlanner` per `sbd` call (it is `!Send`); forward plan reused for both transforms + one inverse plan.
- `sbd_distance_matrix`: upper triangle via `iter_maybe_parallel!` (each rayon task calls `sbd`, which builds its own planner), mirrored to lower; `row_to_buf` each row; zero diagonal by construction.

## Tests + results (all pass)
`test_sbd_self_zero`, `test_sbd_symmetric`, `test_sbd_shifted_copy`, `test_sbd_offset_scale_invariant`, `test_sbd_ncc_bounds`, `test_sbd_constant_series`, `test_sbd_matrix_symmetric_zero_diag`, `test_sbd_matrix_parallel_matches` — **8/8 pass** (default features and `linalg`). Doctest on `sbd` passes.

## Divergences
- **`test_sbd_shifted_copy`**: SBD uses *linear* (zero-padded) cross-correlation, so a right-shifted-with-zero-fill copy is not bit-exactly distance 0 — the k/n non-overlapping tail contributes a small residual (measured ≈0.0275 for k/n=5/128). The test's **primary** gate is the signed-shift correctness (`|shift| == k`, not a `fft_len − k` wrap); the distance is asserted `< 0.05`. Exact-zero shape identity is covered separately by `test_sbd_self_zero` and the offset/scale-invariance test. This is inherent to the Paparrizos linear-NCCc definition, not a bug.
- Clamp added on `max NCCc` to `[−1, 1]` to absorb finite-precision overshoot and keep `distance ∈ [0, 2]`.

## Gate tails
- `cargo fmt --check` → clean (exit 0).
- `cargo clippy --all-targets --features linalg,parallel -- -D warnings` → Finished, no warnings.
- `cargo test -p fdars-core --features linalg sbd` → 8 passed; `cargo test -p fdars-core sbd` (default) → 8 passed; `--doc sbd` → 1 passed.

## Seams for Phases 62/63
- **`(distance, shift)` contract:** `SbdResult.shift` is the SIGNED cyclic lag `w*` (in `−(m−1)..=(m−1)`) by which `y` aligns to `x`. Phase 62 stores this to shift-align cluster members before shape extraction. Sign convention: for `y[i] = x[i−k]` (y is x delayed by k), `sbd(x, y).shift` has magnitude k (negative-index tail lag).
- `sbd_distance_matrix` is Phase 63's input to `kmedoids_from_distances` (symmetric, zero-diagonal `FdMatrix`).
