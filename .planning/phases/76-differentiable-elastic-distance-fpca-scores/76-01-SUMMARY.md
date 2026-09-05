# 76-01 SUMMARY — DIF-02: Differentiable soft-DTW + fixed-warp amplitude distance

**Status:** COMPLETE
**Commit:** `cc8a44fc` — `feat(autodiff): differentiable soft-DTW + fixed-warp amplitude distance (DIF-02)`

## Symbols added

- `fdars_core::metric::soft_dtw::soft_dtw_distance_generic<S: Scalar>` (pub, also re-exported at `metric::` level)
- `softmin3_generic<S: Scalar>` (private helper)
- `soft_dtw_distance_inner<S: Scalar>` (private shared kernel; `soft_dtw_distance` now delegates to it at `f64`)
- `fdars-core/src/alignment/differentiable.rs` (new module, registered in `alignment/mod.rs`)
  - `amplitude_distance_at_warp_generic<S: Scalar>` (pub, re-exported)
  - `generic_linear_interp<S>`, `generic_srsf_central_diff<S>`, `generic_l2_srsf_distance<S>` (private helpers)

## Soft-DTW: DELEGATION (not companion)

`soft_dtw_distance` now delegates to `soft_dtw_distance_inner::<f64>`. The f64-parity gate (SC #2)
passed **bit-identically** (`assert_eq!`, exact equality) across gamma ∈ {0.1, 1.0, 10.0} and a
20-point series, so delegation was adopted (no separate companion needed). The only f64-path change
is `a.min(b)` → `PartialOrd` `if/else`, which agrees for all NaN-free inputs.

## Gradient-vs-oracle result (SC #1)

The Dual gradient matches an **independent soft-DTW gradient oracle** (forward + backward + accumulate)
to ≤1e-9 for all 5 coordinates.

### Deviation / notable finding (backlog item)

The shipped hand-written oracle `soft_dtw_accumulate_gradient` (via `soft_dtw_backward`) has a
**pre-existing boundary bug** (present in HEAD commit `18b38179`, untouched by this phase): the reverse
double-loop visits the endpoint `(n, m)` and overwrites the `E[n][m] = 1.0` seed with `a+b+c = 0`,
zeroing the entire backward pass — so the shipped oracle returns an **all-zero gradient** for any input.
This is masked in production because `soft_dtw_barycenter` then descends on a zero gradient (effectively
a no-op optimizer) and the barycenter tests only assert loose closeness.

**Scope decision:** production code left byte-for-byte unchanged (additive-only phase constraint; a
one-line fix cascades into `soft_dtw_barycenter` numerics and broke `test_soft_dtw_barycenter_identical`,
which is out of scope to re-validate here). SC #1 instead validates the Dual gradient against a
**corrected oracle computed inline in the test** (`corrected_oracle_gradient`, which preserves the
endpoint), giving a genuine independent check distinct from finite differences.

**BACKLOG (GSD-ready):** *Fix `soft_dtw_backward` endpoint-overwrite bug so `soft_dtw_accumulate_gradient`
returns the correct gradient, and re-validate `soft_dtw_barycenter` convergence.* (Candidate requirement
for a future soft-DTW-correctness phase.)

## Amplitude-at-warp result (SC #4, SC #7)

- SC #7 (f64 parity): `amplitude_distance_at_warp_generic::<f64>` == hand-composed
  interp→central-diff-SRSF→weighted-L2 reference to ≤1e-10. PASS.
- SC #4 (FD gradient): Dual gradient w.r.t. `curve2[j]` matches central FD to ≤1e-5 for all 20 points.
  PASS. Grid `[0.1, 0.9]` avoids the sin/cos(2πt) derivative zeros at t=0.25/0.75 (SRSF `sqrt(|f'|)`
  singularity).
- Warp-SEARCHED `elastic_distance` documented as deferred (discrete DP argmin, non-differentiable) in the
  module doc comment of `differentiable.rs`.

### softmin3_generic infinity handling (implementation note)

Beyond the `>= S::infinity()` early-return guard, a **per-term guard** was required: computing
`(inf - min_val) * neg_inv_gamma` on a sentinel input produces `inf * 0 = NaN` in the `Dual` tangent
(product rule), even though the primal is fine. Sentinel `+inf` terms are substituted with exact
`S::zero()` (identical f64 numerics, `exp(-inf)=0`), keeping tangents NaN-free.

## Gate results

- Targeted (`soft_dtw`): lib tests incl. `dual_gradient_vs_oracle`, `f64_parity`, `dual_gradient_vs_fd` — **all PASS**
- Targeted (`differentiable`): `amplitude_f64_parity`, `amplitude_gradient_vs_fd` — **PASS**
- Whole crate (`--features linalg,parallel`): **2853 lib passed, 0 failed** (+ all integration/doc suites 0 failed)
- clippy `--all-targets --features linalg,parallel -- -D warnings`: **clean**
- `cargo fmt --check`: **clean**
- `git diff --stat fdars-core/Cargo.toml`: **empty** (no new dependency); `grep num-traits` = 0
- MSRV 1.81 preserved (only `S::from_f64` conversions + existing `Scalar` ops used)

## Constraints honored

No existing public f64 signature changed (`soft_dtw_distance` signature + doc unchanged; delegation only).
`FdMatrix` stays `Vec<f64>`. No new crate dependency.
