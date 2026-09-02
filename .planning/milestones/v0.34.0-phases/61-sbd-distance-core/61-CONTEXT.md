# Phase 61: SBD Distance Core - Context

**Gathered:** 2026-09-02
**Status:** Ready for planning
**Mode:** Smart-discuss (autonomous) — resolved from `.planning/research/` (SUMMARY/FEATURES/ARCHITECTURE/PITFALLS). No open user decisions.

<domain>
## Phase Boundary

Deliver the Shape-Based Distance (SBD) primitive that k-Shape (Phase 62) and SBD-k-medoids (Phase 63) build on. New `src/metric/sbd.rs` (peer of `gak.rs`/`soft_dtw.rs`). Additive/non-breaking, no new dependency.

In scope (KSH-01/02):
- **`sbd(x, y)`** — FFT normalized cross-correlation → `(distance, optimal_shift)`.
- **`sbd_distance_matrix`** — public n×n SBD matrix over a curve set, symmetric, zero diagonal, parallel.

Out of scope: k-Shape clustering (Phase 62), SBD-k-medoids convenience (Phase 63), crate-root re-exports/bench (Phase 63).
</domain>

<decisions>
## Implementation Decisions

### Resolved (from research — treat as fixed)

1. **Module:** new `src/metric/sbd.rs`; add `pub mod sbd;` + re-exports to `src/metric/mod.rs`. (Crate-root flat re-exports are finalized in Phase 63; reachable via `fdars_core::metric::sbd::...` meanwhile is fine, but do add the `metric/mod.rs` `pub use`.)

2. **SBD / NCCc formula (KSH-01):** `SBD(x,y) = 1 − max_w NCCc_w(x,y)`, where the coefficient-normalized cross-correlation is `NCCc_w = CC_w(x,y) / (‖x_z‖·‖y_z‖)` with `x_z,y_z` the z-normalized inputs. Compute `CC` via FFT: zero-pad to `fft_len = next_power_of_two(2·m − 1)` (m = series length; for unequal lengths use `2·max(len)−1`), `CC = Re(IFFT(FFT(x_z) · conj(FFT(y_z))))`. **rustfft's IFFT is UNNORMALIZED — divide the IFFT output by `fft_len` explicitly** (mirror the treatment near `seasonal/mod.rs:350`). Rearrange the `2m−1` lags so the zero-lag is centered; return `distance = 1 − max` and the `optimal_shift: isize` (signed lag, NOT the raw `fft_len − k` index). Range: distance ∈ [0, 2].

3. **Return type:** `pub struct SbdResult { pub distance: f64, pub shift: isize }` (Debug/Clone/PartialEq, serde-gated). `pub fn sbd(x: &[f64], y: &[f64]) -> Result<SbdResult, FdarError>` (`#[must_use]`). Validation: non-empty; a series shorter than 1 → `FdarError::InvalidDimension`.

4. **z-normalization:** reuse `shapelet::z_normalize_window` / `z_normalize_into` (v0.33.0). **Constant-series guard:** if either input has std ≈ 0 (z-norm → zero vector), the NCCc denominator is 0 → define `SbdResult { distance: 1.0, shift: 0 }` (max NCC = 0 for a flat series), never NaN. Document this convention.

5. **FftPlanner (`!Send`):** each call (and each rayon task in the matrix builder) constructs its OWN `FftPlanner` — it cannot be shared across threads. Mirror the `FftPlanner::<f64>::new()` + `plan_fft_forward`/`plan_fft_inverse` + `.process(&mut buf)` idiom from `fts/spectral.rs` (L142/307). Build the forward plan once per `sbd` call and reuse it for both transforms of the same `fft_len`.

6. **`sbd_distance_matrix` (KSH-02):** `pub fn sbd_distance_matrix(data: &FdMatrix) -> Result<FdMatrix, FdarError>` (`#[must_use]`) → n×n symmetric matrix, `D[i][i] = 0`, `D[i][j] = sbd(row_i, row_j).distance`. Compute the upper triangle + mirror (symmetric by assignment). Parallelize the outer loop with `iter_maybe_parallel!` (each task builds its own `FftPlanner`). `row_to_buf` each row contiguously. All rows share length m.

7. **Determinism:** SBD is deterministic (no RNG); sequential and `parallel` builds must be bit-identical.
</decisions>

<code_context>
## Existing Code Insights
- `src/shapelet/distance.rs`: `z_normalize_window(&[f64]) -> Vec<f64>` (L114), `z_normalize_into(&[f64], &mut [f64])` (L57) — reuse for z-norm.
- `src/fts/spectral.rs`: the rustfft idiom — `FftPlanner::<f64>::new()` (L142), `plan_fft_forward(n)` (L143), `plan_fft_inverse(n)` (L308), `.process(&mut buf)` with `buf: Vec<Complex<f64>>`. `num_complex::Complex` is available.
- `src/seasonal/mod.rs:350`: `let fft_len = (2 * n).next_power_of_two();` — the padding + IFFT-scaling pattern to mirror (note: this milestone needs `2·m − 1`, then next_pow2).
- `src/matrix.rs`: column-major `FdMatrix`, `row_to_buf(i, &mut buf)`, constructors.
- `src/parallel.rs`: `iter_maybe_parallel!`; `src/error.rs`: `FdarError`.
- Conventions: `#[must_use]`, `Debug,Clone,PartialEq` + serde-gated, `Result<_,FdarError>`, doc examples.
</code_context>

<specifics>
## Specific Ideas (verification hooks — from PITFALLS.md; the FFT/NCC silent-correctness gates)
Tests the plan must include (inline `#[cfg(test)] mod tests`):
- `test_sbd_self_zero`: `sbd(x, x).distance ≈ 0.0` (catches FFT padding + IFFT scaling + NCC normalization all at once) with `shift == 0`.
- `test_sbd_symmetric`: `sbd(x,y).distance ≈ sbd(y,x).distance`.
- `test_sbd_shifted_copy`: for `y` = x circularly/linearly shifted by k, `sbd(x,y).distance ≈ 0` and the returned `shift` equals the correct SIGNED lag (±k), not `fft_len − k`.
- `test_sbd_offset_scale_invariant`: `sbd(x, x + c).distance ≈ 0` and `sbd(x, a·x).distance ≈ 0` for a>0 (z-norm invariance).
- `test_sbd_ncc_bounds`: distance ∈ [0, 2]; NCCc within [−1, 1].
- `test_sbd_constant_series`: a constant series → `distance = 1.0`, `shift = 0`, no NaN.
- `test_sbd_matrix_symmetric_zero_diag`: `sbd_distance_matrix` symmetric, zero diagonal.
- `test_sbd_matrix_parallel_matches`: matrix bit-identical (determinism / seq==parallel).
- Doctest on `sbd`.
</specifics>

<deferred>
## Deferred Ideas
- k-Shape clustering (shape extraction, n_init, predict) → Phase 62 (needs `sbd`'s `(distance, shift)`).
- `sbd_kmedoids` convenience + crate-root re-exports + bench → Phase 63.
- Cross-length SBD / multivariate → future (KSH-BREADTH).
</deferred>
