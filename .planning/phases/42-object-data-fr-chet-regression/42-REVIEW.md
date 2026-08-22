---
phase: 42-object-data-fr-chet-regression
reviewed: 2026-08-23T00:00:00Z
resolution_status: resolved
resolution: all findings addressed — CR-01 Power(Inf) guard fix + Inf/NaN tests, WR-01 anova doc fix, IN-01 SpdMetric #[non_exhaustive]
depth: deep
files_reviewed: 8
files_reviewed_list:
  - fdars-core/src/frechet/spaces/spd.rs
  - fdars-core/src/frechet/spaces/correlation.rs
  - fdars-core/src/frechet/spaces/spherical.rs
  - fdars-core/src/frechet/spaces/network.rs
  - fdars-core/src/frechet/spaces/point_process.rs
  - fdars-core/src/frechet/spaces/mod.rs
  - fdars-core/src/frechet/regression.rs
  - fdars-core/src/frechet/anova.rs
findings:
  critical: 1
  warning: 1
  info: 1
  total: 3
status: issues_found
---

# Phase 42: Code Review Report

**Reviewed:** 2026-08-23
**Depth:** deep
**Files Reviewed:** 8
**Status:** issues_found

## Summary

Phase 42 adds five `MetricSpace` backends (SPD/Frobenius, SPD/Power-α, SPD/Log-Cholesky, Correlation,
Spherical, Network, Point-Process), extracts the shared Petersen-Müller weight helpers
`compute_global_weights` / `compute_local_weights`, and adds generic entry points
`frechet_global_reg_space`, `frechet_local_reg_space`, `frechet_anova_space`.

**Numerical correctness:** Log-Cholesky coordinate round-trip is correct (row-major L, column-major
output, `i.min(k)+1` loop produces symmetric `LLᵀ`). Power-α eigenvector reconstruction is correct
(column-major outer product `V[:,k] ⊗ V[:,k]` accumulates into `result[i + j*d]`). Power-α = 1
reproduces the Frobenius distance analytically. Correlation unit-diagonal renormalization uses
correct column-major diagonal indexing (`m[i + i*d]`). Spherical Karcher descent correctly
normalizes iterates and guards the antipodal case. Network and point-process weighted averages are
correct for non-negative weights.

**Refactor fidelity:** The extraction of `compute_global_weights` / `compute_local_weights` is
byte-identical to the original inline code in `frechet_global_reg` / `frechet_local_reg`; the
covariance loop (`sigma[a*p+b]`) and ridge term (`sigma[j*p+j] += 1e-6`) are preserved exactly.
`compute_tn_generic` is a faithful generalization of the old `compute_tn` — only the monomorphic
`WassersteinDensitySpace`/`Vec<f64>` types are replaced with `S: MetricSpace` / `S::Object`. The
existing `frechet_anova` call site correctly switches from `compute_tn` to `compute_tn_generic`.

**One correctness bug found:** `SpdMetric::Power(f64::INFINITY)` bypasses the alpha validation
guard and silently produces NaN distances and wrong means, as described in CR-01 below.

---

## Critical Issues

### CR-01: `SpdMetric::Power(f64::INFINITY)` passes validation and produces NaN

**File:** `fdars-core/src/frechet/spaces/spd.rs:66`

**Issue:** The `SpdMatrixSpace::new` guard rejects `alpha <= 0.0` and `alpha.is_nan()` but does
**not** reject `alpha = f64::INFINITY`. Infinity is positive and not NaN, so it passes. In
`spd_power`, any eigenvalue > 1.0 becomes `eigenvalue.powf(f64::INFINITY) = f64::INFINITY`,
filling the output matrix with `inf` entries. Then:

- `distance` computes `frobenius_norm_diff(&inf_vec, &inf_vec) / f64::INFINITY = NaN / f64::INFINITY = NaN`.
- `weighted_frechet_mean` calls `spd_power` with `1.0 / f64::INFINITY = 0.0` on the averaged
  `inf`-filled matrix: `eigenvalue.max(0.0).powf(0.0) = 1.0` for all eigenvalues, silently
  returning an identity matrix regardless of the input data.

Both results are silently wrong — no error is returned, which violates the function contract. Any
SPD matrix with an eigenvalue > 1 (e.g., any matrix not dominated by eigenvalues ≤ 1) triggers this.

**Fix:** Add `|| !alpha.is_finite()` to the guard:

```rust
if let SpdMetric::Power(alpha) = metric {
    if alpha <= 0.0 || !alpha.is_finite() {
        return Err(FdarError::InvalidParameter {
            parameter: "alpha",
            message: "power-metric exponent must be positive and finite".to_string(),
        });
    }
}
```

The separate `.is_nan()` check is then redundant (NaN is not finite) and can be dropped, but
including it does no harm. The error message should be updated from "must be > 0" to "must be
positive and finite" to reflect the new constraint.

---

## Warnings

### WR-01: `frechet_anova` doc comment says `seed + k` (wrong variable) — stale after Phase 42 edit

**File:** `fdars-core/src/frechet/anova.rs:111`

**Issue:** The doc comment for `frechet_anova` reads:

```
/// per-iteration seeded RNG (`StdRng::seed_from_u64(seed + k)`), so it is
```

The variable `k` is the **number of groups** (defined at line 145), not the permutation index.
The permutation loop (line 171-172) uses `perm` as the iteration variable:
```rust
for perm in 0..n_perm {
    let mut rng = StdRng::seed_from_u64(seed.wrapping_add(perm as u64));
```
This stale comment predates Phase 42 but was not corrected when the function body was modified by
this phase (renaming `compute_tn` to `compute_tn_generic`). The sibling `frechet_anova_space`
(added in Phase 42) correctly documents `seed.wrapping_add(perm)`. The inconsistency is now
more visible and likely to mislead callers trying to predict the RNG sequence.

**Fix:**

```rust
/// per-iteration seeded RNG (`StdRng::seed_from_u64(seed.wrapping_add(perm))`), so it is
```

---

## Info

### IN-01: `SpdMetric` is not `#[non_exhaustive]` but all other result structs are

**File:** `fdars-core/src/frechet/spaces/spd.rs:31`

**Issue:** Project convention (CLAUDE.md) specifies `#[non_exhaustive]` on public result structs
for forward compatibility. `SpdMetric` is a public enum with three variants (`Frobenius`, `Power`,
`LogCholesky`). If a new metric variant is added in a future milestone (e.g., `AffineInvariant`),
external crates exhaustively matching `SpdMetric` would break without a semver bump. The five new
space types (`CorrelationMatrixSpace`, `NetworkSpace`, `PointProcessSpace`, `SphericalSpace`,
`SpdMatrixSpace`) are structs and the `new()` constructor enforces all invariants — those are fine.
Only `SpdMetric` as a matchable enum is affected.

**Fix:**

```rust
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SpdMetric {
    Frobenius,
    Power(f64),
    LogCholesky,
}
```

Note: adding `#[non_exhaustive]` is a breaking change if any downstream crate is already
exhaustively matching on `SpdMetric`. Since this is a new type added in this phase (no prior
consumers), the window to add it is now, before it appears on crates.io.

---

_Reviewed: 2026-08-23_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
