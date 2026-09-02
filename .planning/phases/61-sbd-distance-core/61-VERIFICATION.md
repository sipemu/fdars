---
phase: 61
title: SBD Distance Core
status: passed
requirements: [KSH-01, KSH-02]
verified: 2026-09-02
---

# Phase 61 Verification — SBD Distance Core

All four ROADMAP success criteria verified against `fdars-core/src/metric/sbd.rs`
(commit 3f7bc35b). Gates: `cargo fmt --check` clean, `cargo clippy --all-targets
--features linalg,parallel -- -D warnings` clean, `cargo test -p fdars-core
[--features linalg] sbd` = 8/8 pass, doctest 1/1 pass.

## Criterion 1 — `sbd(x, y)` correctness path — **PASS**
> z-normalize internally; FFT zero-padded to `next_power_of_two(2m−1)`; IFFT divided by `fft_len`; NCC coefficient-normalized by `‖x‖·‖y‖`; `distance = 1 − max NCCc ∈ [0,2]`; optimal lag returned.

Evidence: `sbd` (sbd.rs:104) z-normalizes both inputs via `z_normalize_window`
(never the caller); `fft_len = (2*m − 1).next_power_of_two()` (sbd.rs:132);
IFFT scale folded as `scale = 1/(fft_len · denom)` where `denom = ‖x_z‖·‖y_z‖`
(sbd.rs:161); `distance = 1 − clamp(max_ncc, −1, 1)` returned with signed
`shift`. `test_sbd_ncc_bounds` confirms `distance ∈ [0,2]`. Signature
`Result<SbdResult, FdarError>`, `#[must_use]`.

## Criterion 2 — FFT/NCC correctness gates — **PASS**
> `sbd(x,x) ≈ 0`; `sbd(x,y) == sbd(y,x)` within 1e-10; right-shifted copy → sbd ≈ 0 at correct SIGNED shift (not `fft_len − k`); every NCC in `[−1,1]`.

Evidence:
- `test_sbd_self_zero`: `sbd(x,x).distance < 1e-10`, `shift == 0` — PASS (exercises IFFT-scale + coefficient-normalization together).
- `test_sbd_symmetric`: `|sbd(x,y) − sbd(y,x)| < 1e-10` — PASS.
- `test_sbd_shifted_copy`: recovered `|shift| == k` (signed lag, `< n`, not a wrap-around), `distance < 0.05` — PASS. (Distance is not bit-exact 0 because SBD uses *linear* zero-padded cross-correlation; the small non-overlap residual is inherent to the Paparrizos definition. Signed-shift extraction — the actual gate for Pitfall 1/4 — is exact.)
- `test_sbd_ncc_bounds`: `NCCc = 1 − distance ∈ [−1,1]` across shape / anti-shape / step pairs — PASS.

## Criterion 3 — shape invariance + constant-series guard — **PASS**
> `sbd(x, x+c) ≈ 0`, `sbd(x, a·x) ≈ 0` (a>0) within 1e-10; constant series → distance 1.0, shift 0, no NaN.

Evidence: `test_sbd_offset_scale_invariant`: `sbd(x, x+100).distance < 1e-10`
and `sbd(x, 50·x).distance < 1e-10` — PASS. `test_sbd_constant_series`:
constant `c` → `distance == 1.0`, `shift == 0`, `!is_nan()`; both-constant
case also `1.0`/`0` — PASS. Guard at sbd.rs:121 (`denom ≤ NORM_EPS = 1e-12`).

## Criterion 4 — `sbd_distance_matrix` — **PASS**
> public n×n symmetric, zero diagonal, parallelized via `iter_maybe_parallel!` (each rayon task builds its own `FftPlanner`), output equals independent pairwise `sbd`.

Evidence: `sbd_distance_matrix` (sbd.rs:206) `#[must_use]`, returns
`Result<FdMatrix, FdarError>`, computes upper triangle via
`iter_maybe_parallel!(0..n)` (each task calls `sbd`, which constructs its own
`!Send` `FftPlanner`) and mirrors; diagonal left zero. `test_sbd_matrix_
symmetric_zero_diag`: symmetric, zero diagonal, entries equal independent
pairwise `sbd` — PASS. `test_sbd_matrix_parallel_matches`: two builds
byte-identical (`to_bits()`) and bit-identical to independent pairwise `sbd` —
PASS (determinism / seq==parallel).

## Verdict
All 4 criteria **PASS**. No gaps.
