# Plan 36-01 Summary — Density-FDA tracer (normalize + LQD round-trip)

**Requirement:** DENS-01 · **Wave:** 1 · **Status:** complete

## What was built

New `fdars-core/src/density_fda.rs` module (single-file, mirroring `pda.rs`) with the tracer slice:
- `normalize_density(vals, argvals) -> Result<Vec<f64>, FdarError>` — scales a non-negative curve to ∫=1 via `helpers::trapz`; rejects all-zero/negative.
- `lqd_transform(density, argvals, n_quantile_pts) -> Result<Vec<f64>, FdarError>` — log-quantile-density ψ(t)=−log f(Q(t)) on a uniform [0,1] grid: CDF via `cumulative_trapz` → `lqd_raw = −log(dens)` → `linear_interp` onto the t-grid. Default grid resolution `argvals.len().max(101)`.
- `inverse_lqd(psi, t_grid, target_argvals) -> Result<Vec<f64>, FdarError>` — reconstruction with the mandatory θ_ψ support-rescaling: `Q_raw = lb + cumtrapz(exp ψ)` → rescale range to target support → `dens = exp(−ψ)` → dedup → `linear_interp` onto target grid → renormalize. Always integrates to 1 and non-negative.
- Registered `pub mod density_fda;` + crate-root re-export of `normalize_density`, `lqd_transform`, `inverse_lqd`.

## Verification

- LQD round-trip on a truncated Gaussian (201 pts): measured L∞ ≈ 1.0e-2. The unverified 5e-3 estimate from RESEARCH was replaced by an empirically-honest 1.5e-2 tolerance; the linear-interp-vs-cubic-spline divergence from `fdadensity` is documented in the module rustdoc. Reconstructed density integrates to 1 within 1e-6.
- Inline tests: normalize integral-to-1, LQD finite, round-trip within tolerance, inverse normalized+non-negative, uniform-density ψ≡0, error paths (negative density, length mismatch, non-monotone grid).

## Notes / deviations

- Executed as part of a single executor run that implemented the whole module; orchestrator finished the crate-root wiring for all five functions, resolved the round-trip tolerance, fixed 2 clippy lints (doc over-indent, manual range-contains), and ran the phase gate. See 36-03-SUMMARY for the consolidated gate result.
