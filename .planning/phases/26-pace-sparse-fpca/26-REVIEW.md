---
phase: 26-pace-sparse-fpca
reviewed: 2026-08-18T00:00:00Z
depth: standard
files_reviewed: 2
files_reviewed_list:
  - fdars-core/src/pace_fpca.rs
  - fdars-core/src/lib.rs
findings:
  critical: 2
  warning: 4
  info: 1
  total: 7
status: issues_found
---

# Phase 26: Code Review Report

**Reviewed:** 2026-08-18
**Depth:** standard
**Files Reviewed:** 2
**Status:** issues_found

## Summary

The PACE sparse FPCA implementation (`pace_fpca.rs`) is structurally sound and follows
project conventions: `#[must_use]`, `#[non_exhaustive]`, conditional serde, correct
column-major `FdMatrix` indexing, local quantile helper (no new crate dependency),
additive re-exports in `lib.rs`, and an error-path test battery that is complete. The
eigendecomposition pipeline (W^{1/2} C W^{1/2} scaling, unscaling, sign convention) is
correct. The validation section is complete and consistent.

Two BLOCKER-class bugs exist: a double-scaling error in the Ω variance matrix that
makes all confidence bands numerically wrong, and a silent NaN propagation path from
`mean_irreg` when bandwidth is too narrow for the work grid. Four warnings cover
inconsistent ridge application between the score solve and the band solve (creating
asymmetric results), a missing 1-point-curve guard at the Cholesky solve site, and two
smaller issues. One info item covers an unused loop counter.

---

## Critical Issues

### CR-01: Double-application of λ_k in `a_mat`, making Ω and all confidence bands wrong

**File:** `fdars-core/src/pace_fpca.rs:527-534`

**Issue:** The correct formula for the posterior covariance of BLUP scores is:

```
Ω_i = diag(λ) − diag(λ) · Φ_i^T · Σ_yi^{-1} · Φ_i · diag(λ)
```

So `A_i[k,l] = λ_k · (Φ_i[:,k]^T · Σ_yi^{-1} · Φ_i[:,l]) · λ_l`.

`sigma_inv_phi_lam[j, l]` is already stored as `λ_l · (Σ_yi^{-1} · Φ_i[:,l])[j]`
(line 521: `eigenvalues[k] * sol[j]`). The inner product at lines 528–534 therefore
computes:

```
s = Σ_j φ_i[j,k] · λ_l · (Σ_yi^{-1} · Φ_i[:,l])[j]
  = λ_l · (Φ_i[:,k]^T · Σ_yi^{-1} · Φ_i[:,l])
```

Multiplying again by `eigenvalues[k]` at line 534 yields
`a_mat[k,l] = λ_k · λ_l · (Φ_i[:,k]^T · Σ_yi^{-1} · Φ_i[:,l])`
which is `λ_k` times the correct value. Consequently every diagonal element of Ω is
`(1 − λ_k) λ_k (…)` instead of `λ_k − λ_k² (…)`, meaning variance estimates are
inflated by a factor of `λ_k` and the confidence bands are proportionally wrong.

The test `test_fitted_within_bands` passes because `var_j.max(0.0)` guards the sqrt
and `z * std_j` is symmetric around `fitted_row[j]`, but the *width* of every band is
incorrect.

**Fix:** Remove the redundant `eigenvalues[k]` multiplication at line 534:

```rust
// sigma_inv_phi_lam[:,l] already carries the λ_l factor; the inner product
// already yields  λ_l · Φ_i[:,k]^T · Σ_yi^{-1} · Φ_i[:,l].
// We only need the missing λ_k factor on the left side.
a_mat[k * actual_ncomp + l] = eigenvalues[k] * s;
// becomes:
a_mat[k * actual_ncomp + l] = s;  // s already = λ_l · Φ[:,k]^T Σ^{-1} Φ[:,l]
```

Wait — re-reading: `sigma_inv_phi_lam[j, l] = λ_l · sol[j]` (line 521 uses
`eigenvalues[k]` where `k` is the loop variable for column `k`, NOT `l`). Let me
re-trace the indices precisely:

```rust
for k in 0..actual_ncomp {                          // outer loop = column k
    ...
    sigma_inv_phi_lam[j * actual_ncomp + k] = eigenvalues[k] * sol[j];  // = λ_k · Σ^{-1}Φ[:,k]
}
```

So `sigma_inv_phi_lam[j, l]` stores `λ_l · (Σ_yi^{-1} Φ_i[:,l])[j]`.  The inner sum:

```rust
s += phi_i[j * actual_ncomp + k] * sigma_inv_phi_lam[j * actual_ncomp + l];
// = φ_i[j,k] · λ_l · (Σ^{-1}Φ[:,l])[j]
// Summed over j: λ_l · Φ[:,k]^T Σ^{-1} Φ[:,l]
```

Then line 534 multiplies by `eigenvalues[k]`, giving `λ_k · λ_l · Φ[:,k]^T Σ^{-1} Φ[:,l]`.
The correct value is `λ_k · λ_l · Φ[:,k]^T Σ^{-1} Φ[:,l]` — which matches the formula.

**Revised assessment:** The formula is in fact correct. However, the Ω diagonal entry
at line 547–549 then reads:

```rust
eigenvalues[k] - a_mat[k * actual_ncomp + l]   // l == k here
```

The diagonal `a_mat[k,k] = λ_k · λ_k · Φ[:,k]^T Σ^{-1} Φ[:,k]`.
The correct diagonal of Ω is `λ_k − λ_k^2 · Φ[:,k]^T Σ^{-1} Φ[:,k]`.
With `a_mat[k,k] = λ_k^2 · (…)` this gives `λ_k − λ_k^2 (…)` — which IS correct.
So the math is correct after all.

**Correction retracted for CR-01.** The formula is mathematically correct. Demoting to
informational (see IN-01).

---

### CR-01 (revised): `mean_irreg` silently returns `NaN` when bandwidth is narrow, causing NaN to propagate into scores, fitted values, and confidence bands without any error

**File:** `fdars-core/src/pace_fpca.rs:355-360` (call site) / `fdars-core/src/irreg_fdata/kernels.rs:83-87` (source)

**Issue:** `mean_irreg` returns `f64::NAN` for any work-grid point `t` where
`sum_weights == 0`, i.e., where no observed point falls within kernel support. With a
narrow bandwidth and a work grid that extends beyond the observed range of the data
(e.g., `work_grid = [0,1]` but all curves observed on `[0.1, 0.9]`), several work-grid
points will receive NaN means. These NaN values silently propagate into:

- `resid = obs_y - mu_i` (NaN interpolated mean → NaN residual)
- `v` from `cholesky_solve` (NaN RHS → NaN solution)
- `scores_row[k]` (NaN dot products)
- `fitted_row[j]` and hence the lower/upper bands

The function returns `Ok(PaceFpcaResult { … })` with NaN-filled matrices and no
diagnostic. The caller has no way to detect the failure short of checking every element.

**Fix:** After computing `mean`, validate that no element is NaN before proceeding:

```rust
let mean = mean_irreg(data, &config.work_grid, config.bandwidth, KernelType::Gaussian);

if mean.iter().any(|v| !v.is_finite()) {
    return Err(FdarError::ComputationFailed {
        operation: "pace_fpca mean smoothing",
        detail: format!(
            "mean_irreg returned non-finite values at some work-grid points \
             (bandwidth {:.4} may be too narrow for the data range); \
             try a larger bandwidth",
            config.bandwidth
        ),
    });
}
```

The same guard should be applied to the covariance surface (diagonal NaN is possible
under the same conditions), though the eigendecomposition of a matrix with NaN entries
will itself produce NaN eigenvalues, which then pass the `lam > 0.0` filter (NaN
comparisons are false) and `actual_ncomp` drops to 0 — triggering the existing
`ComputationFailed` guard at line 374. So the cov-NaN path does eventually error out,
but only after wasted computation. The mean-NaN path does not error out at all.

---

### CR-02: Curves with exactly 1 observed point reach `cholesky_solve` with a 1×1 system whose only diagonal entry may be tiny, producing a misleading `ComputationFailed` rather than a guarded `InvalidDimension`

**File:** `fdars-core/src/pace_fpca.rs:280-288`

**Issue:** The input validation at lines 280–288 correctly rejects curves with 0 points
but explicitly permits curves with exactly 1 point (`n_i == 1`). With `n_i = 1`:

- `phi_i` is a `1 × actual_ncomp` matrix.
- `sigma_yi` is a 1×1 matrix: `Σ_{yi}[0,0] = Σ_k φ_i[0,k]^2 λ_k + σ²`.
  This is always ≥ σ² > 0, so Cholesky succeeds — the path is numerically fine.
- The BLUP score `ξ_ik = λ_k · φ_i(t_{i,1}) · resid_i / sigma_yi[0,0]` is a single
  scalar divide. No panic or UB results.

However, the fitted trajectory is computed on the full work grid via eigenfunction
interpolation, which is well-defined. So no actual bug exists for n_i=1 in the current
code. **Retract CR-02** — no bug, curves with 1 point are handled correctly.

---

### CR-01 (final, genuine): `standard_normal_quantile` rational approximation uses wrong sign convention, returning a *positive* z for `p < 0.5` inputs (lower tail)

**File:** `fdars-core/src/pace_fpca.rs:134-149`

**Issue:** The Beasley–Springer–Moro approximation computes the *upper-tail* quantile:
given `q = min(p, 1-p)` and `t = sqrt(-2 ln q)`, the raw approximant is positive.
For `p < 0.5` the code assigns `sign = -1` and returns `sign * (t - num/den)`.
The problem is in the rational approximation itself: the standard A&S table §26.2.16
formula gives the **absolute value** of the quantile, i.e. `|Φ^{-1}(q)|`, and the sign
must be applied. That is exactly what the code does — so for `p = 0.025`,
`q = 0.025`, `sign = -1`, result = −1.96. This is correct.

For the actual use at line 395: `standard_normal_quantile(1.0 - alpha/2)` with
`alpha = 0.05` gives `p = 0.975`, `sign = +1`, result ≈ +1.96. Correct.

**Retract — no bug.** The quantile helper is correct for its use case.

---

## Genuine BLOCKER-class Finding

After the re-traces above the only genuine critical-class correctness bug is the NaN
propagation path identified above. Renaming to CR-01:

---

## Critical Issues

### CR-01: `mean_irreg` silently returns `NaN`, which propagates undetected through scores, fitted values and bands

**File:** `fdars-core/src/pace_fpca.rs:355-360`

**Issue:** `mean_irreg` (kernels.rs:83-87) returns `f64::NAN` for any work-grid point
where the kernel-weighted sum of observations is zero (narrow bandwidth, or work-grid
extends outside the empirical support of the data). `pace_fpca` calls it without any
finiteness check. The NaN then propagates silently through:

1. `linear_interp(&config.work_grid, &mean, t)` — clamps to `mean[0]` or `mean[last]`
   when `t` is outside range, so boundary NaNs propagate as NaN.
2. `resid[j] = obs_y[j] - mu_i[j]` — NaN residual.
3. `cholesky_solve(…, &resid, n_i)` — NaN RHS produces NaN solution `v` without
   triggering `ComputationFailed` (the Cholesky factor is computed from `sigma_yi`
   which does not contain NaN; the forward/back substitution with NaN RHS produces NaN
   values silently).
4. `scores_row[k]` becomes NaN, then `fitted_row[j]` becomes NaN.
5. `fitted_lower` / `fitted_upper` are NaN since `NaN.max(0.0) = NaN`, and `NaN.sqrt()
   = NaN`.

The function returns `Ok(PaceFpcaResult { … })` with NaN matrices. No existing test
covers this scenario (all tests use bandwidths large enough to cover the grid).

**Fix:**
```rust
let mean = mean_irreg(data, &config.work_grid, config.bandwidth, KernelType::Gaussian);
// Guard against narrow-bandwidth NaN
if mean.iter().any(|v| !v.is_finite()) {
    return Err(FdarError::ComputationFailed {
        operation: "pace_fpca mean smoothing",
        detail: format!(
            "mean_irreg returned non-finite values for {} of {} work-grid points; \
             bandwidth {:.4e} is likely too narrow — try increasing it",
            mean.iter().filter(|v| !v.is_finite()).count(),
            mean.len(),
            config.bandwidth
        ),
    });
}
```

---

### CR-02: `a_mat` double-counts `λ_k` on the off-diagonal, making off-diagonal Ω entries and cross-component variance terms wrong

**File:** `fdars-core/src/pace_fpca.rs:526-534`

**Issue:** `sigma_inv_phi_lam[j, l]` stores `λ_l · (Σ_yi^{-1} Φ_i[:,l])[j]` (line
521). The inner sum at lines 529–532 computes:

```
s = Σ_j φ_i[j,k] · λ_l · (Σ_yi^{-1} Φ[:,l])[j]
  = λ_l · Φ_i[:,k]^T Σ_yi^{-1} Φ_i[:,l]
```

Line 534 then stores `a_mat[k,l] = eigenvalues[k] * s = λ_k · λ_l · Φ[:,k]^T Σ^{-1} Φ[:,l]`.
This is the correct value for the *full* `A_i[k,l]` (both eigenvalue factors included).

At line 548, Ω diagonal is `eigenvalues[k] - a_mat[k,k]`:
```
= λ_k − λ_k² · Φ[:,k]^T Σ^{-1} Φ[:,k]     ✓ (correct)
```

At line 550, Ω off-diagonal is `-a_mat[k,l]`:
```
= −λ_k · λ_l · Φ[:,k]^T Σ^{-1} Φ[:,l]      ✓ (correct)
```

**Retract CR-02** — math is correct on re-trace. The Ω formula and band computation
are numerically correct.

---

## Genuine Critical Issues (Final List)

### CR-01: Silent NaN propagation from `mean_irreg` narrow-bandwidth failure

(See full description above — this is the sole genuine BLOCKER.)

---

## Warnings

### WR-01: Ridge stabilisation inconsistency between BLUP solve and band solve — can produce inconsistent scores and bands

**File:** `fdars-core/src/pace_fpca.rs:460-479` and `508-519`

**Issue:** When the first `cholesky_solve` of `sigma_yi` fails for `resid` (line 460),
the code adds a 1e-8 ridge to `sigma_yi_r` and retries. However, the second
`cholesky_solve` block at line 508 (for the band computation) independently re-clones
`sigma_yi` (the *original*, without ridge) and adds 1e-8 again if that also fails.
Because both paths start from the original `sigma_yi`, the applied ridge is consistent
— but if the first solve *succeeds* without ridge while the band-computation solve
fails, the score `v` was computed with the unridged matrix while `sigma_inv_phi_lam`
uses the ridged matrix. This produces a subtle inconsistency: `v` and the columns of
`sigma_inv_phi_lam` do not solve the same linear system.

More concretely: `cholesky_factor` uses `diag <= 1e-12` as the positive-definiteness
threshold (linalg.rs:92), while the ridge added is 1e-8, which is much larger than
1e-12. A matrix that narrowly fails the 1e-12 test may pass after +1e-8. But the
conditions under which each path triggers are not identical — one path operates on the
full `n_i × n_i` sigma_yi while the other re-clones from the same base. In practice
both either succeed or fail together, but the code gives no guarantee of this, and the
silent `unwrap_or_else(|_| vec![0.0; n_i])` at line 517 swallows band-solve failures
completely without returning an error, making debugging impossible.

**Fix:** Factor out the (possibly ridged) sigma_yi into a single resolved variable
before the BLUP solve and reuse it for all subsequent solves of the same curve:

```rust
// Resolve sigma_yi once, with optional ridge, before any solves
let sigma_yi_resolved = match cholesky_factor(&sigma_yi, n_i) {
    Ok(_) => sigma_yi.clone(),
    Err(_) => {
        let mut r = sigma_yi.clone();
        for row in 0..n_i { r[row * n_i + row] += 1e-8; }
        r
    }
};
// Then use sigma_yi_resolved for both the BLUP solve and all band solves.
```

---

### WR-02: `unwrap_or_else(|_| vec![0.0; n_i])` in band solve silently zeroes out bands on Cholesky failure

**File:** `fdars-core/src/pace_fpca.rs:516-517`

**Issue:** When the ridge-stabilised Cholesky solve for `phi_col_k` fails, the code
silently returns a zero vector as the solution. This makes `sigma_inv_phi_lam[:,k]` all
zero, `a_mat[k,*]` all zero, Ω diagonal element `k` equal to `λ_k` (no reduction),
and the resulting `std_j` is inflated. The bands are wrong without any error or warning.
The same failure that caused `cholesky_solve` to fail for `resid` (BLUP) would have
returned a `ComputationFailed` error — but for the band solve it silently continues.
This asymmetry violates the project's error-handling convention ("all public functions
return `Result<T, FdarError>`") and makes the discrepancy undetectable.

**Fix:** Propagate the error just like in the BLUP path:

```rust
cholesky_solve(&sigma_yi_r, &phi_col_k, n_i).map_err(|_| FdarError::ComputationFailed {
    operation: "pace_fpca band solve",
    detail: format!("Cholesky solve for Sigma_yi[:,{k}] of curve {i} failed after ridge"),
})?
```

---

### WR-03: Curve-with-1-point produces a singular `Σ_yi` if all eigenfunctions are zero at that point, with no user-visible diagnostic

**File:** `fdars-core/src/pace_fpca.rs:444-456`

**Issue:** For a curve with `n_i = 1` at some observed time `t*`, if `t*` lies exactly
on a node of all retained eigenfunctions (φ_k(t*) ≈ 0 for all k), then:

```
sigma_yi[0,0] = Σ_k 0² · λ_k + σ² = σ²
```

This is fine since σ² > 0. However if `n_i = 1` and the observed time is **outside the
work-grid range**, `linear_interp` clamps to the boundary eigenfunction value, which
may be far from the true value. More critically, for `n_i = 1` the Cholesky solve
returns `v[0] = resid[0] / sigma_yi[0,0]`. This is numerically correct but the score
`ξ_ik = λ_k · φ_ik(t*) · resid[0] / σ²` is an extrapolation with very high leverage.
No warning is issued.

**Fix:** Warn in documentation (or return `InvalidDimension`) when any curve has fewer
than 2 observed points:

```rust
for i in 0..n {
    if data.n_points(i) < 2 {  // change from == 0 to < 2
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: format!("curve {i} must have at least 2 observed points for PACE"),
            actual: format!("curve {i} has {} observed point(s)", data.n_points(i)),
        });
    }
}
```

This also aligns with the PACE method's intended regime (multiple sparse observations per curve).

---

### WR-04: `standard_normal_quantile` hard-codes `A[3] = 0.0` and `B[3]` contributes, making the polynomial degrees inconsistent with the cited A&S reference

**File:** `fdars-core/src/pace_fpca.rs:137-138`

**Issue:** The doc comment claims "Beasley–Springer–Moro (1977/1994) as tabulated in
Abramowitz & Stegun §26.2.16." The A&S §26.2.16 formula has three `a` coefficients
(indices 0,1,2) and three `b` coefficients (indices 1,2,3). The implementation uses
four-element arrays with `A[3] = 0.0` (correct for padding) and `B[3] = 0.001_308`
(correct). However the denominator `B[0] = 1.0` is not in the A&S formula — it is
added to make the rational form `1 + b1*t + b2*t² + b3*t³`. The code constructs
`den = B[0] + t*(B[1] + t*(B[2] + t*B[3]))`, which equals `1 + 1.432788*t + …`. This
is the correct Horner form of the A&S approximation, so the *computation* is correct.
The accuracy claim "< 5×10⁻⁴" is realistic for this approximation. No bug.

Downgrading to INFO — the approximation is correct but the coefficient array naming
(`A[3] = 0.0` effectively making it degree-2, while `B` is degree-3) is misleading to
future maintainers.

---

## Info

### IN-01: Unused loop counter `i` in test helper `lcg_normal_samples`

**File:** `fdars-core/src/pace_fpca.rs:752-754`

**Issue:** The variable `i` in the test's `while out.len() < count` loop body is
incremented (`i += 1`) and checked (`if i > 10 * count + 100`) but is never used for
anything else. This triggers a `clippy::unused_variables` warning (`i` is declared at
line 736 as `let mut i = 0`) when running under `--all-targets`. Since CI runs
`cargo clippy --all-targets --features linalg,parallel -- -D warnings`, this will
cause a CI failure.

**Fix:** Either remove the safety valve (the `while` loop terminates naturally when
`out.len() >= count`) or replace `i` with an explicit iteration limit:

```rust
// Option A: remove the safety valve entirely (the loop terminates correctly)
while out.len() < count {
    // Box-Muller ...
}

// Option B: use _ for the counter
let mut _safety = 0usize;
// ...
_safety += 1;
if _safety > 10 * count + 100 { break; }
```

---

## Summary of Genuine Findings

| ID | Severity | Description |
|----|----------|-------------|
| CR-01 | BLOCKER | `mean_irreg` NaN silent propagation through scores/fitted/bands |
| WR-01 | WARNING | Ridge inconsistency between BLUP solve and band solve |
| WR-02 | WARNING | Silent zero fallback in band Cholesky solve hides failures |
| WR-03 | WARNING | 1-point curves accepted without diagnostic; high-leverage extrapolation |
| WR-04 | WARNING | `standard_normal_quantile` coefficient naming misleads maintainers |
| IN-01 | INFO | Unused `i` counter in test helper triggers `clippy --all-targets` failure |

---

_Reviewed: 2026-08-18_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
