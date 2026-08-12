# Phase 14: Shift Registration - Context

**Gathered:** 2026-08-12
**Status:** Ready for planning

<domain>
## Phase Boundary

Deliver the two registration-parity capabilities of milestone v0.17.0, both additive/non-breaking within `fdars-core/src/alignment/`:

- **FEAT-06** — `least_squares_shift_registration`: register a set of curves by a per-curve rigid horizontal shift `δᵢ` that minimizes each curve's L2 distance to the cross-sectional sample mean. Fills the "simplest registration method" gap (fdars currently jumps from landmark shifts straight to full elastic SRSF warping).
- **FEAT-07** — three registration-quality validation scores added to `alignment/quality.rs`: `least_squares_score`, `pairwise_correlation_score`, `sobolev_least_squares_score` (scikit-fda's `LeastSquares` / `PairwiseCorrelation` / `SobolevLeastSquares`).

Out of scope: elastic/SRSF changes, iterative multi-template registration, regularized FPCA (v2), and any change to existing `alignment/` signatures.
</domain>

<decisions>
## Implementation Decisions

### Area 1 — Shift Registration API (FEAT-06)
- **Return type:** dedicated `ShiftRegistrationResult { registered_data: FdMatrix, shifts: Vec<f64> }`, mirroring the existing `AlignmentSetResult { gammas, aligned_data, distances }` pattern. Derive `Debug, Clone, PartialEq`; `#[cfg_attr(feature = "serde", ...)]` per convention.
- **Reference target:** single-pass alignment to the cross-sectional sample mean via `fdata::mean_1d`. Deterministic; matches the backlog phrasing "align each curve to the sample mean." (Iterative template update was considered and deferred — not needed for the parity gap.)
- **Out-of-domain evaluation** when computing `fᵢ(t − δ)`: reuse the v0.16.0 `ExtrapolationPolicy` enum, defaulting to `Boundary` (clamp to endpoints). This composes with the existing interpolation infrastructure rather than inventing new boundary logic.
- **Optimizer:** golden-section search over `δ ∈ [−max_shift, +max_shift]`, minimizing `‖fᵢ(t − δ) − mean(t)‖²` (Simpson-weighted L2), with the shifted curve resampled via `helpers::linear_interp`. `max_shift` default = 0.25 × domain range; expose as a parameter.

### Area 2 — Registration-Quality Scores (FEAT-07)
- **Return type:** `Result<f64, FdarError>` for all three scores — honors the milestone success criterion (all new public functions return `Result`) and enables dimension validation. This intentionally differs from the older raw-`f64` neighbors (`warp_complexity`, `alignment_quality`) in the same file; note the deviation in rustdoc.
- **Signatures (match backlog):** `least_squares_score(registered: &FdMatrix, argvals: &[f64])`, `pairwise_correlation_score(registered: &FdMatrix, argvals: &[f64])`, `sobolev_least_squares_score(registered: &FdMatrix, argvals: &[f64], lambda: f64)`.
- **`least_squares_score`** = (1/n) Σᵢ ∫‖registeredᵢ − mean‖² dt, mean = cross-sectional `mean_1d`, Simpson-weighted integral.
- **`pairwise_correlation_score`** = mean Pearson correlation over all n(n−1)/2 curve pairs via the Simpson inner product; O(n²·m), documented (no sampling cap — matches scikit-fda semantics).
- **`sobolev_least_squares_score`** = first-derivative Sobolev (W¹,²): LS term + λ·(1/n) Σᵢ ∫(fᵢ′ − mean′)² dt, derivative via the existing uniform-gradient helper used by `warp_smoothness`.

### Area 3 — Integration & Conventions
- **Crate-root re-export:** re-export `least_squares_shift_registration`, `ShiftRegistrationResult`, and the three score functions at the crate root (milestone SC), alongside the existing `alignment/` re-exports.
- **Tests:** inline `#[cfg(test)]` in the respective module files — (a) already-aligned set → `δᵢ ≈ 0`, curves ~unchanged; (b) identical curves with injected constant offsets → recovered `δᵢ` match within tolerance; (c) on synthetic shifted-bumps, `least_squares_score` drops and `pairwise_correlation_score` rises after registration.
- **Non-breaking:** purely additive; no existing `alignment/` signature is modified; no new dependencies.

### Claude's Discretion
- Exact module placement of `least_squares_shift_registration` within `alignment/` (new file e.g. `shift.rs` vs an existing module), tolerance constants in tests, golden-section iteration count / convergence tolerance, and whether `max_shift` / `ExtrapolationPolicy` are positional params or bundled in a small config — all at Claude's discretion, guided by codebase conventions.
</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `fdata::mean_1d(&FdMatrix) -> Vec<f64>` — cross-sectional sample mean.
- `helpers::linear_interp(x, y, t) -> f64` — pointwise resampling for the shifted curve.
- `helpers::simpsons_weights(argvals)` / `helpers::trapz` — integration weights already used throughout `alignment/quality.rs`.
- `ExtrapolationPolicy` (v0.16.0, FEAT-04) — boundary/exception/fill/periodic handling for off-domain evaluation.
- `gradient_uniform` (used by `warp_smoothness` in `quality.rs`) — derivative approximation for the Sobolev term.
- `iter_maybe_parallel!(0..n)` — per-curve parallelism, already used by `align_to_target` in `alignment/set.rs`.

### Established Patterns
- Registration result struct pattern: `AlignmentSetResult { gammas, aligned_data, distances }` (`alignment/set.rs`) — model `ShiftRegistrationResult` on it.
- `align_to_target(data, target, argvals, lambda) -> AlignmentSetResult` is the closest existing "align each curve to a target" entry point — same shape, but elastic; shift registration is the rigid analogue.
- Quality functions live in `alignment/quality.rs` and currently return raw values; new scores go beside them but return `Result`.

### Integration Points
- New public functions re-exported at the crate root (via `alignment/mod.rs` → `lib.rs`), consistent with existing `alignment` re-exports.
- `ShiftRegistrationResult` output feeds naturally into the three new quality scores (register → score the `registered_data`).
</code_context>

<specifics>
## Specific Ideas

- scikit-fda references: `LeastSquaresShiftRegistration` (shift estimator), and the `preprocessing.registration.validation` scores `LeastSquares`, `PairwiseCorrelation`, `SobolevLeastSquares`. Match their semantics but keep fdars conventions (column-major `FdMatrix`, `Result` returns, Simpson integration).
- Backlog anchors: PREP-04 (rank 11, P1/M) and PREP-05 (rank 13, P2/S) in `.planning/research/BACKLOG.md` carry the exact formulas and signatures — reuse, do not re-derive.
</specifics>

<deferred>
## Deferred Ideas

- Iterative multi-pass shift registration with template re-estimation (scikit-fda's iterative mode) — single-pass to the sample mean suffices for this parity gap.
- PREP-06 (LDO-regularized FPCA) and ACC-VALIDATE (numerical validation vs scikit-fda) — explicitly v2 per REQUIREMENTS.md.
</deferred>
