# Phase 74: Elastic Conformal Anomaly Detection - Context

**Gathered:** 2026-09-05
**Status:** Ready for planning

<domain>
## Phase Boundary

Add an inductive conformal anomaly detector for functional data that flags both
magnitude and shape outliers by scoring curves against a reference template using
elastic distances (requirements ECA-01, ECA-02, ECA-03). Independent of the VEE
group (Phases 72–73).

Additive and non-breaking: the existing `conformal_prediction_band` path stays
unchanged, no new crate dependency, reuse-first over `tolerance/conformal.rs`
(`NonConformityScore`, `conformal_prediction_band`) and `alignment/pairwise.rs`
elastic distances + `karcher_mean`.

</domain>

<decisions>
## Implementation Decisions

### Elastic nonconformity scores (ECA-01)
- **Extend the existing `#[non_exhaustive] NonConformityScore`** in place with three
  elastic variants: `AmplitudeElastic`, `PhaseElastic`, `CombinedElastic`.
  `conformal_prediction_band` returns `FdarError::InvalidParameter` for these
  template-based variants (it has no reference template) — the band path stays
  otherwise unchanged.
- Score function: `elastic_nonconformity(curve, template, argvals, lambda, variant) -> f64`
  — non-negative, and **zero for a curve identical to the reference** (a make-or-break gate).
- Expose a `lambda` (warp penalty) parameter, default 0.0.
- Distance reuse: call `amplitude_distance` / `phase_distance_pair` / `elastic_distance`
  (`alignment/pairwise.rs`) verbatim; do NOT re-derive.

### Inductive conformal anomaly detector (ECA-02)
- Reference template: **compute the calibration Karcher mean by default**; allow the
  caller to supply an explicit template.
- p-value: standard inductive conformal formula `(1 + #{calib_score >= test_score}) / (n_calib + 1)`.
- Flag rule: flag a curve when `p <= alpha` (equivalently, its score exceeds the
  (1-alpha) quantile of the calibration scores → the calibrated threshold).
- API shape: `elastic_conformal_anomaly(calibration, test, config)` taking a
  `ConformalAnomalyConfig` (variant / alpha / lambda / optional template) — builder-style
  config per crate convention.
- Behavioral gates: on exchangeable clean data the flag rate ≈ alpha (marginal validity);
  injected **magnitude** AND **shape** outliers are flagged.

### Result type + integration (ECA-03)
- `ConformalAnomalyResult { p_values, scores, flags, threshold }` (per-curve p-values +
  nonconformity scores + boolean flags + the calibrated threshold).
- Module placement: new additive `tolerance/conformal_anomaly.rs`; leave `conformal.rs`
  band path unchanged.
- Full **crate-root (`lib.rs`) + prelude re-exports** for all new public items.
- A running **module doctest** (calibrate → flag) under `cargo test --doc`.

### Claude's Discretion
- Exact `ConformalAnomalyConfig` field layout + defaults, internal helper factoring,
  the home of the elastic-nonconformity dispatch (in `conformal_anomaly.rs`), and doctest
  fixture construction are at Claude's discretion within the above and crate conventions
  (column-major `FdMatrix`, `Result<T, FdarError>`, `#[non_exhaustive]`, `#[must_use]`,
  conditional serde, `Debug/Clone/PartialEq`).

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `fdars-core/src/tolerance/conformal.rs`: `NonConformityScore` enum (currently
  `{SupNorm, L2}`, `#[non_exhaustive]`), `conformal_prediction_band` (must stay
  unchanged). The band function matches on `NonConformityScore` — the new variants'
  arms there return `InvalidParameter`.
- `fdars-core/src/alignment/pairwise.rs`: `amplitude_distance` (line ~384),
  `phase_distance_pair` (line ~389), `elastic_distance` (line ~103) — all
  `(f1, f2, argvals, lambda)`; `elastic_distance` for combined.
- `alignment/` `karcher_mean` (for the calibration Karcher-mean template).
- `tolerance/` submodule layout (`conformal.rs`, `types.rs`, `helpers.rs`, `mod.rs`, ...)
  — add `conformal_anomaly.rs` as a sibling and wire via `tolerance/mod.rs`.

### Established Patterns
- Column-major `FdMatrix` (rows = curves); `Result<T, FdarError>`; dimension checks at entry.
- Config structs (builder style): `ConformalConfig`, `ClassifCvConfig`, etc.
- Result structs derive `Debug, Clone, PartialEq`; conditional serde; `#[non_exhaustive]`;
  `#[must_use]` on expensive computations.
- Crate-root re-exports in `src/lib.rs`; prelude re-exports in `src/prelude.rs`.

### Integration Points
- New public items (`elastic_nonconformity`, `elastic_conformal_anomaly`,
  `ConformalAnomalyConfig`, `ConformalAnomalyResult`, new `NonConformityScore` variants)
  re-exported from `lib.rs` + `prelude.rs` and `tolerance/mod.rs`.

</code_context>

<specifics>
## Specific Ideas

- Nonconformity zero-for-identical is a known-answer gate (a curve scored against itself
  as template → 0).
- Marginal validity: on exchangeable clean data, flag rate ≈ alpha — test with a
  moderate calibration/test split and a tolerance band around alpha.
- Injected outliers: a magnitude outlier (scaled/shifted curve) AND a shape outlier
  (warped/phase-distorted curve) must both be flagged — use the appropriate elastic
  variant (Combined catches both; Amplitude catches magnitude; Phase catches shape).
- Confirm at plan time the exact argument order/lambda semantics of
  `amplitude_distance`/`phase_distance_pair`/`elastic_distance` so template scoring matches.

</specifics>

<deferred>
## Deferred Ideas

- Full conditional / Mondrian conformal anomaly detection (class-conditional validity),
  ECA-F1 → future milestone; v1 covers the inductive marginal case only.
- R/WASM binding exposure of the anomaly surface → future milestone (issue fdars-j75).

</deferred>
