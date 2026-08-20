# Phase 30: Interval Testing Procedure Family - Context

**Gathered:** 2026-08-20
**Status:** Ready for planning

<domain>
## Phase Boundary

Add the Interval Testing Procedure (ITP) family that `fdatest` provides — a **one-population**
interval-wise test, a **two-population** interval-wise test, and an **interval-wise FLM coefficient**
test — over B-spline and Fourier bases with **domain-selective adjusted p-values** (the ITP
interval-wise closure adjustment), via a new `inference/itp.rs`, reusing the shipped INF-01
permutation infrastructure and the `basis/` projection, **without changing any existing inference or
basis code or public signatures**.

**In scope:** three `Result`-returning entry points; per-component/per-domain adjusted p-values;
crate-root re-exports; inline `#[cfg(test)]` tests. **Out of scope:** any plotting/rendering of ITP
p-value surfaces; random-projection ANOVA/MANOVA (`fdANOVA`, deferred); new crate dependency;
changes to existing `inference/` (INF-01/INF-02) or `basis/` signatures.
</domain>

<decisions>
## Implementation Decisions

### API Shape & Basis Choice
- Three `Result<ItpResult, FdarError>`-returning public fns in `inference/itp.rs`: a one-population
  interval-wise test, a two-population interval-wise test, and an interval-wise FLM coefficient test.
- Basis selection via a basis-choice parameter (B-spline `{nbasis, order}` / Fourier `{nbasis}`),
  reusing the existing `basis/` projection — `ProjectionBasisType { Bspline, Fourier }` (basis/projection.rs)
  and `bspline_basis` / `fourier_basis` — rather than a per-basis function or reimplemented projection.
- `ItpResult { adjusted_pvalues: Vec<f64>, raw_pvalues: Vec<f64>, basis metadata (type + nbasis), n_perm }`
  with the standard derive/attribute stack (`Debug, Clone, PartialEq`, `#[non_exhaustive]`,
  conditional serde). Per-component / per-domain outputs (not a single global p-value).
- New file `inference/itp.rs`; crate-root re-exported (`lib.rs` inference re-export block). Existing
  inference entry points untouched.

### Permutation Reuse & Seeding
- Build the interval-wise permutation null from the shipped INF-01 permutation infrastructure
  (`t_perm_test` / `f_perm_test` in `inference/permutation.rs`, or their underlying pooling/relabel
  machinery) — **no new permutation engine**.
- `DEFAULT_N_PERM = 999` (reuse the existing constant) as the configurable default.
- Seeded for reproducibility using the per-thread `StdRng::seed_from_u64(seed + k)` convention.

### Closure Adjustment
- ITP interval-wise closure: the adjusted p-value for basis component `k` is the **max** over all
  contiguous intervals `I` containing `k` of that interval's joint permutation p-value — so a user
  can identify which sub-intervals of the domain drive a significant result.
- Enumerate **all contiguous intervals** of basis components (O(p²)); document the complexity.
- Pin the exact `fdatest` ITP statistic (Pini & Vantini interval-wise closure) during research;
  document any divergence from the R baseline in rustdoc (prior-milestone practice).

### Testing
- Synthetic fixtures: a localized between-group difference confined to a known sub-interval (adjusted
  p small on the true differing interval, non-significant elsewhere) **and** a null case (no
  difference → non-significant everywhere), within a documented tolerance.
- Tolerance-based assertions on the adjusted p-values with seeded determinism.
- Error-path coverage per entry point: empty / mismatched group sizes / incompatible basis parameters
  → `FdarError`, never panic.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/inference/permutation.rs` — `t_perm_test`, `f_perm_test`, `DEFAULT_N_PERM` (999) — the INF-01
  permutation infrastructure (pooling + relabel null) to reuse for the interval-wise tests.
- `src/inference/mod.rs` — `TestResult` struct; re-export patterns.
- `src/inference/flm.rs` — `flm_f_test`, `flm_gof_test` (FLM testing analogs for the interval-wise
  FLM coefficient test).
- `src/basis/projection.rs` — `ProjectionBasisType { Bspline, Fourier }`; `src/basis/bspline.rs`
  (`bspline_basis`, `construct_bspline_knots`, `bspline_basis_from_knots`), `src/basis/fourier.rs`
  (`fourier_basis`) — the projection to reuse.
- `src/matrix.rs` — column-major `FdMatrix`; `src/error.rs` — `FdarError`; `src/parallel.rs` —
  `iter_maybe_parallel!` + per-thread RNG seeding.

### Established Patterns
- Column-major storage, `Result<T, FdarError>` public API, inline `#[cfg(test)] mod tests`,
  `#[non_exhaustive]` result structs with conditional serde, per-file module split with `pub use` in
  `inference/mod.rs` and crate-root re-export in `lib.rs`.
- Permutation-test seeding + `DEFAULT_N_PERM` reproducibility convention (INF-01).

### Integration Points
- New `src/inference/itp.rs` — the three entry points + `ItpResult`.
- `src/inference/mod.rs` — add `mod itp;` + `pub use itp::{...}`.
- `src/lib.rs` — extend the inference re-export block.

</code_context>

<specifics>
## Specific Ideas

- The interval-wise closure adjustment (max over containing intervals) is the defining feature —
  pin the exact statistic and adjustment against `fdatest` (Pini & Vantini) during research, and
  reconcile the B-spline vs Fourier projection paths with the INF-01 permutation infra.
- Match the R baseline by **capability**, not `fdatest`'s exact signatures.

</specifics>

<deferred>
## Deferred Ideas

- Any plotting/rendering of ITP p-value surfaces / heatmaps (numeric outputs only).
- Random-projection ANOVA/MANOVA (`fdANOVA`) — explicitly deferred this milestone.

</deferred>
