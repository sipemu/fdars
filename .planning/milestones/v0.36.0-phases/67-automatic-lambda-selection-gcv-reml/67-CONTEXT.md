# Phase 67: Automatic λ Selection — GCV + REML - Context

**Gathered:** 2026-09-04
**Status:** Ready for planning

<domain>
## Phase Boundary

Add automatic smoothing-parameter (λ) selection to the Phase 66 `peer()` estimator:
GCV grid search or REML/mixed-model estimation, selectable via config, with an
explicit λ honored verbatim when supplied. Wraps the Phase 66 core penalized fit.

In scope (PER-03):
- GCV grid search: pick the λ minimizing a GCV score over a fixed log-spaced grid,
  deterministically.
- REML: estimate λ via the PEER-as-mixed-model equivalence (eigendecomposition of Q →
  null-space fixed + range-space random effect; λ = σ²_e/σ²_u), deterministically.
- Explicit λ: used verbatim, no search, recorded unchanged in the result.
- Selection diagnostics recorded in `PeerResult`.

Out of scope (later phases):
- Longitudinal `lpeer`, out-of-sample `predict`, crate-root/prelude exports, end-to-end
  module doctest — Phase 68.

</domain>

<decisions>
## Implementation Decisions

### λ-selection config & API shape
- Change `PeerConfig.lambda: f64` → `lambda: LambdaChoice` enum with variants
  `Fixed(f64)`, `Gcv`, `Reml`. One field both pins and selects. Serde behind `serde`
  feature; derives per crate convention. `Default` = `Gcv`.
- Default selection method is **GCV** (deterministic, robust workhorse). refund's
  `peer` defaults to REML — noted, but GCV chosen here for reproducibility; REML is a
  first-class alternative, not second-class.
- `Fixed(λ)` is used verbatim: no search runs, and the λ appears unchanged in
  `PeerResult` diagnostics.
- Evolve `PeerConfig` now (crate is unpublished until the v0.36.0 tag after Phase 68).
  Mechanically update the Phase 66 tests that construct `PeerConfig { lambda: 1e-4 }`
  to `LambdaChoice::Fixed(1e-4)`. `PeerConfig::default()` stays valid.

### GCV grid search
- Fixed internal log-spaced grid over ~[1e-6, 1e4], ~40 points (not user-tunable this
  phase).
- GCV score reuses the crate's penalized-regression pattern:
  `GCV = n · RSS / (n − tr(H))²`, where `tr(H)` reuses the existing
  `compute_peer_trace_hat` / `compute_trace_hat` column-solve. RSS from the penalized
  fit's residuals.
- Selection = argmin GCV over the grid, fully deterministic (no RNG). Ties broken by
  the smaller grid index (documented) so two runs pick the same λ.

### REML approach (resolves the STATE `famm` reuse concern)
- Map PEER to a linear mixed model via the **eigendecomposition of the penalty
  operator Q** (nalgebra `symmetric_eigen`, already used in `fpca_variants.rs`):
  null-space of Q → fixed (unpenalized) columns; range-space → random effect
  `b ~ N(0, σ²_u I)`; residual `ε ~ N(0, σ²_e I)`. This is the "partially empirical
  eigenvector" decomposition that names PEER.
- Estimate σ²_u, σ²_e via a **self-contained REML EM** in `peer.rs`, reusing `famm`'s
  EM *pattern* / variance-component helpers where they fit. Do NOT drive
  `famm::fit_scalar_mixed_model` directly — it is subject-grouped (per-subject random
  intercept via `subject_map`), not the shared smoothing random effect PEER needs. A
  thin PEER-specific REML EM is the additive, non-breaking path.
- λ = σ²_e / σ²_u.
- REML-vs-GCV agreement is asserted within a **documented tolerance** (β(t) agree
  within a stated relative tolerance on synthetic SNR data); both selectors recover the
  known β(t).

### Result diagnostics & determinism
- `PeerResult.lambda` holds the **selected** λ (the value actually used). Add
  `gcv: Option<f64>` (GCV score at the selected λ; `None` when GCV did not run) and a
  `lambda_method` marker recording which path ran (`Fixed`/`Gcv`/`Reml`).
- Both selectors are fully deterministic: fixed grid for GCV; fixed EM initialization
  and iteration count for REML; no RNG anywhere. Two runs on the same data pick the
  same λ.
- Scope stays λ-selection only; prediction/exports remain Phase 68.

### Claude's Discretion
- Exact grid point count and REML EM iteration cap / convergence tolerance.
- Whether `lambda_method` is a new small enum or reuses `LambdaChoice`'s discriminant.
- Internal factoring (helper fns) within `peer.rs`.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `function_on_scalar.rs`: `select_lambda_gcv` (~line 370, GCV grid pattern),
  `compute_fosr_gcv` (~171, `GCV = (1/nm)Σ(r/(1−tr(H)/n))²`), `compute_trace_hat`
  (~401), `penalized_solve` (~121). The `FosrResult` carries `lambda` + `gcv` fields —
  a shape template.
- `peer.rs` (Phase 66): `peer()`, `build_q`, `compute_peer_trace_hat`,
  `PeerConfig`/`PeerResult`/`PeerPenalty`. β solved via `(W_c'W_c + λQ)β = W_c'y_c`.
- `famm.rs`: `fit_scalar_mixed_model` (pub(crate), ~438) — REML EM for subject-grouped
  random effects; variance-component helpers (`estimate_variance_components`,
  `shrinkage_weights`, EM loop). Reuse the pattern, not the subject-grouped entry point.
- `fpca_variants.rs:484` — nalgebra `symmetric_eigen` usage precedent for the
  null/range eigendecomposition of Q.
- `linalg.rs`: Cholesky helpers (already used by peer).

### Established Patterns
- Column-major `FdMatrix`; all public fns return `Result<T, FdarError>`.
- Deterministic numerics; no logging; inline `#[cfg(test)] mod tests`.
- GCV convention already in-crate (`compute_fosr_gcv`) — match it.

### Integration Points
- All changes land in `fdars-core/src/peer.rs` (extend Phase 66). No new files
  expected. No crate-root/prelude re-exports yet (Phase 68).

</code_context>

<specifics>
## Specific Ideas

- Reference baseline: refund `peer`/`pfr` λ selection (mgcv gam, REML/GCV).
- Known-answer gates (ROADMAP): GCV picks the GCV-minimizing λ deterministically; REML
  picks a sensible positive λ and REML/GCV β(t) agree within tolerance; explicit λ used
  verbatim; both selectors land a non-degenerate λ recovering the known β(t) on
  synthetic SNR data.
- STATE concern explicitly resolved: `famm::fit_scalar_mixed_model` is subject-grouped;
  Phase 67 uses a self-contained smoothing-REML EM (additive), not that entry point.

</specifics>

<deferred>
## Deferred Ideas

- Longitudinal `lpeer` (which DOES use subject-grouped random effects and may reuse
  `famm::fit_scalar_mixed_model`), out-of-sample `predict`, crate-root/prelude exports,
  and the end-to-end module doctest — Phase 68.
- User-configurable GCV grid bounds/count — future enhancement if needed.

</deferred>
