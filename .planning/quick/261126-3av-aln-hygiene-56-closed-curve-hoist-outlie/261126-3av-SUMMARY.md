---
quick_id: 261126-3av
issue: 56
backlog_id: ALN-HYGIENE
status: complete
---
# Summary — ALN-HYGIENE (GH #56)

All four items delivered; additive and non-breaking, every default bit-identical.

## Commits
- f6322fd0 perf: hoist SRSF/Simpson weights out of closed-curve rotation search + parallel candidates (refs #56)
- e93fbc27 feat: OutlierThreshold option wires elastic outlier alpha; parallel distance loops (refs #56)
- a0a04d65 perf: parallelise remaining elastic-FPCA per-curve loops, bit-identical (refs #56)
- 68326ec7 feat: opt-in SRSF pre-smoothing (refs #56)
- ffd80f88 docs: CHANGELOG + clippy-clean tests (closes #56)

## Public API added
- `alignment::OutlierThreshold { TukeyFence { k }, Quantile }` (`#[non_exhaustive]`, default `TukeyFence { k: 1.5 }`)
- `ElasticOutlierConfig::threshold_method` (new field; sealed config, default = old behaviour)
- `alignment::SrsfSmoothing { None (default), BoxFilter { passes }, SmoothingSpline }`
- `alignment::srsf_transform_smoothed(data, argvals, &SrsfSmoothing) -> Result<FdMatrix>`
- `alignment::srsf_presmooth(data, argvals, &SrsfSmoothing) -> Result<FdMatrix>`
- all re-exported from `alignment` and the crate root.

## Details
1. **Closed curves** (`alignment/closed.rs`): `q1` and Simpson weights computed once; candidate
   rotations (coarse + fine) evaluated through `iter_maybe_parallel!` and scanned in order with
   the same strict `<` (first minimum kept). Bench `elastic_align_pair_closed` (`--quick`):
   m=50 37.7 → 17.1 ms, m=100 170.7 → 35.0 ms. The hoist alone measured within noise (−0.6 %/+0.1 %,
   the DP dominates); the speedup is from candidate parallelism.
2. **Outlier alpha** (`alignment/outlier.rs`): wiring alpha into the default would change results, so
   the explicit enum was added; `Quantile` uses alpha ((1−alpha) empirical quantile). Validation of
   k / alpha. The three per-curve distance loops are now parallel maps.
3. **SRSF pre-smoothing** (`alignment/srsf_smoothing.rs`): BoxFilter = fdasrvf `smooth.data`
   (golden vs R fdasrvf 2.5.0, 1e-14). SmoothingSpline = semantics of fdasrvf 2.5.1
   `f_to_srvf(smooth=TRUE)` (`smooth.spline` + derivative; source read from
   github jdtuck/fdasrvf_R master since only 2.5.0 is installed): per-curve GCV over a
   unit-invariant log-λ grid with `smooth_basis` (cubic B-spline, ∫(f'')² penalty, R's
   `.nknots.smspl` knot count), then the standard numerical SRSF derivative. On R's noisy-sine
   reference case: SRSF L2 error 0.0270 (fdars) vs 0.0236 (R smooth.spline), raw 0.41.
   Not wired into karcher/pairwise entry points (opt-in function only).
4. **Elastic FPCA** (`elastic_fpca.rs`): the three loops named in PERF-PAR-ELFPCA were already
   parallel (earlier work); the remaining sequential ones — `warps_to_normalized_psi`,
   `mean_tangent_vector`, `build_symmetric_covariance`, `project_onto_eigenvectors` — now go through
   `map_indexed` (parallel above ~200k flops, sequential below, same output), with in-order
   reductions; covariance now uses contiguous centred columns. Joint FPCA balance search evaluates
   its two probe SVDs concurrently. `audit_p4_elastic_fpca` (`--quick`): vert_fpca 561 → 369 µs,
   joint_fpca(Some(1)) 3.16 → 3.04 ms. (A first version without the work threshold regressed small
   sizes by ~50 %; fixed before commit.)
- `dp_prune` default untouched.

## Validation
- `alignment/hygiene_tests.rs`: FNV-1a hashes of f64 bit patterns recorded on the pre-change code
  (ce810d5f) for closed pair/Karcher (λ=0, 0.2), outlier detection (median/mean) and
  vert/horiz/joint FPCA; pass with default features AND `--no-default-features` (sequential ==
  parallel == pre-change).
- elastic_fpca helper tests compare the parallel path (sizes above threshold) bit-for-bit with
  verbatim copies of the old loops.
- Outlier: hand-computed Tukey/quantile cutoffs, end-to-end quantile, alpha-independence of the
  default, validation.
- SRSF smoothing: R golden (box filter), R-reference accuracy (spline), linear preservation,
  time-rescaling invariance, None bit-identity. Script: `golden_srsf_smoothing.R` (+ `.out`).

## Gates
- `cargo fmt --all` clean.
- `cargo clippy -p fdars-core --all-targets --features linalg,parallel -- -D warnings` clean.
- `cargo test --lib alignment::` 376 pass; `--lib elastic_fpca` 24 pass; `--doc alignment` 38 pass;
  `--no-default-features --lib hygiene` 3 pass; `cargo check --no-default-features --lib` ok.
- No full-suite run (per runbook).

## Deviations / not done
- Closed-curve candidate parallelism goes beyond the literal "hoist" request (needed for a
  measurable speedup; result-preserving).
- SmoothingSpline matches fdasrvf 2.5.1 in accuracy, not bit-for-bit (uniform knots, grid λ search,
  numerical derivative). fdasrvf 2.5.1 itself is not installed (CRAN has 2.5.0); the reference
  number comes from reproducing its `spline_derivative` in R 2.5.0's environment.
