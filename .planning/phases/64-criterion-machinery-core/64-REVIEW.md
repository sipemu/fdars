---
phase: 64-criterion-machinery-core
reviewed: 2026-09-02T23:30:00Z
depth: deep
files_reviewed: 2
files_reviewed_list:
  - fdars-core/src/optimal_design.rs
  - fdars-core/src/lib.rs
findings:
  critical: 0
  warning: 4
  info: 1
  total: 5
status: issues_found
---

# Phase 64: Code Review Report

**Reviewed:** 2026-09-02T23:30:00Z
**Depth:** deep
**Files Reviewed:** 2
**Status:** issues_found

## Summary

Reviewed `fdars-core/src/optimal_design.rs` (new, 525 lines including tests) and the 4
additive lines in `fdars-core/src/lib.rs`. All 14 tests pass; clippy `--all-targets
--features linalg,parallel -- -D warnings` is clean.

The core numerical machinery is **mathematically correct**: `build_sigma_design` assembles
the right p×p matrix with `σ²I_p` added exactly once per diagonal entry; the
cross-covariance p-vector for the trajectory criterion is correctly computed as
`Σ_k λ_k φ_k(t_j) φ_k(argvals[selected[i]])` (not a K-vector); the score criterion's
`A_mat[k,l] = λ_k λ_l Φ_d[:,k]ᵀ Σ_d⁻¹ Φ_d[:,l]` matches the CONTEXT.md spec and the
`pace_fpca.rs:547–558` reference pattern verbatim; column-major indexing into
`eigenfunctions` is correct throughout; D-opt returns the unnegated log-det. No blocker
bugs were found.

Four warning-level issues were identified: a misleading doc claim about D-opt sign, a
missing argvals-length guard mentioned as required in the research doc, asymmetric
ridge-retry robustness for the D-opt branch, and a test that claims to exercise the
ridge-retry path but actually never triggers it (the retry code path has no effective
test coverage).

---

## Warnings

### WR-01: D-opt doc claim "NEGATIVE for informative design" is incorrect when λ_k > 1

**File:** `fdars-core/src/optimal_design.rs:52` (also line 307)

**Issue:** The `OptimalityKind::D` doc says "The value is NEGATIVE for an informative
design (posterior eigenvalues ≤ prior λ_k)" and the inline comment at line 307 repeats
"NEGATIVE for an informative design. Do NOT negate." Both are false when any `λ_k > 1`.

The correct invariant is: `log det(Cov(ξ|Y_S)) ≤ log det(Λ) = Σ_k log λ_k`. When the
prior eigenvalues are `[2.0, 1.0]`, the empty-set D-opt value is `ln(2) + ln(1) ≈ 0.693`
(positive), which the passing `test_score_d_empty_set` confirms. The code itself is
correct (it returns the unnegated log-det); only the doc is wrong.

The "do not negate" instruction is still valid and important — the sign error described in
RESEARCH.md Pitfall 4 would produce a positive value only when all λ_k < 1. The doc
misleads implementers of Phase 65 who will read it for the sign convention.

**Fix:**
```rust
/// D-optimality: log-determinant of the posterior score covariance.
/// Returned un-negated; adding design points monotonically DECREASES this value
/// (fewer observations → D-opt returns Σ_k log λ_k ≥ log det Cov(ξ|Y_S) for any S).
/// Empty design returns `Σ_k log λ_k` (the prior log-det, which may be positive or
/// negative depending on the eigenvalue scale).
D,
```

And at line 307:
```rust
// log det(Cov) via Cholesky. Returned un-negated; monotone non-increasing.
// Do NOT negate — the criterion is minimized, so smaller (more negative) = better design.
```

---

### WR-02: Missing m ≥ 2 guard contradicts RESEARCH.md security analysis

**File:** `fdars-core/src/optimal_design.rs:83–114`

**Issue:** RESEARCH.md (Security Domain → Known Threat Patterns) states "entry guard
requires `m >= 2` via argvals length check" as the mitigation for
`simpsons_weights`-related edge cases. No such guard exists in the implementation.

If `m = 1`, the trajectory criterion computes a single-point "integral" with weight 1.0
(the `n < 2` fallback in `simpsons_weights`), producing a result that is
physically meaningless but numerically quiet — no error, no panic, wrong value silently.
`PaceFpcaResult` from `pace_fpca.rs` always has `m ≥ 2` in practice, but the public
`design_criterion` accepts any `&PaceFpcaResult` and the test harness constructs models
directly (see `synthetic_model_params`).

**Fix:** Add after the `sigma2 <= 0.0` guard:
```rust
if m < 2 {
    return Err(FdarError::InvalidParameter {
        parameter: "model.argvals",
        message: format!(
            "argvals must have length >= 2 for Simpson quadrature; got {m}"
        ),
    });
}
```

---

### WR-03: Posterior covariance Cholesky has no ridge-retry in D-opt branch

**File:** `fdars-core/src/optimal_design.rs:308–311`

**Issue:** The trajectory and score criteria both protect their `Σ_d` Cholesky via
`factor_sigma_design_with_retry` (1e-8 ridge). However, in the D-opt branch, the
posterior covariance `Cov = Λ − A_mat` undergoes `cholesky_factor(&cov, ncomp)` with no
retry:

```rust
let l_cov = cholesky_factor(&cov, ncomp).map_err(|_| FdarError::ComputationFailed { ... })?;
```

The Schur complement `Cov` is theoretically positive-definite, but when `Σ_d` itself was
ridge-adjusted (sigma2 very small), the resulting `A_mat` values are inflated, and `Cov`
can develop near-zero or slightly negative diagonal elements due to floating-point
cancellation in `(λ_k : 0) − A_mat[k,k]`. This causes hard `ComputationFailed` from
D-opt in the same near-singular regime where the trajectory and A-opt branches succeed
gracefully.

The asymmetry is subtle: a caller who switches from `OptimalityKind::A` to
`OptimalityKind::D` on the same (valid, ridge-retried) model would get an error from D
but not from A. The Phase 65 greedy sweep iterates all three criteria; this failure
breaks that loop unexpectedly.

**Fix:** Wrap the posterior covariance factorization with a small ridge:
```rust
let l_cov = {
    let mut cov_copy = cov.clone();
    match cholesky_factor(&cov_copy, ncomp) {
        Ok(l) => l,
        Err(_) => {
            for i in 0..ncomp { cov_copy[i * ncomp + i] += 1e-10; }
            cholesky_factor(&cov_copy, ncomp).map_err(|_| FdarError::ComputationFailed {
                operation: "optimal_design D-optimality log-det",
                detail: "posterior covariance Cholesky failed after ridge; \
                         model may be near-degenerate".into(),
            })?
        }
    }
};
```

---

### WR-04: test_ridge_retry does not exercise the ridge-retry code path

**File:** `fdars-core/src/optimal_design.rs:428–433`

**Issue:** The test uses `sigma2 = 1e-12` expecting the first Cholesky to fail and the
retry to rescue it. However, `cholesky_factor` fails only when a diagonal element of
the running L-factor drops to ≤ `1e-12`. With design points `[10, 20, 30]` from the
cosine synthetic model, the `Σ_d` diagonal is dominated by the eigenfunction outer
products:

```
sigma_d[i,i] = Σ_k λ_k · φ_k(t_selected[i])² + 1e-12
             ≈ 2·cos²(π·t) + 1·cos²(2π·t) ≈ 0.9–2.0  ≫  1e-12 threshold
```

The first `cholesky_factor` call succeeds trivially (no retry needed), so the branch
at `optimal_design.rs:157–164` — the only code that calls the ridged `cholesky_factor`
again — is never reached. The retry mechanism has zero test coverage.

**Fix:** Construct a provably near-singular `Σ_d` by using coincident design points
(which produce a rank-deficient outer product) while keeping `sigma2` just below the
singularity threshold:

```rust
#[test]
fn test_ridge_retry_actually_triggered() {
    // Use a single-component model with sigma2=0 (forbidden at entry) — instead,
    // use a model where Sigma_d is nearly rank-1 by using two identical indices.
    // With selected=[0, 0] and sigma2=1e-13, Sigma_d off-diagonals ≈ diagonals;
    // the Schur step in Cholesky: diag -= L[1,0]^2 → near zero → triggers retry.
    let model = synthetic_model_params(51, vec![2.0, 1.0], 1e-13);
    let res = design_criterion(&model, &[0, 0], DesignCriterion::Trajectory);
    // If retry fires, Ok; if it doesn't, the test is wrong.
    // Also verify the retry counter by checking the Ok branch.
    assert!(res.is_ok(), "ridge-retry should rescue near-singular Sigma_d: {res:?}");
}
```

Alternatively, inject a custom model with a near-zero sigma2 AND eigenfunctions
whose values at the selected points are nearly collinear (e.g., all design points
clustered near a node of the second eigenfunction).

---

## Info

### IN-01: `cholesky_solve` not imported — inconsistency with RESEARCH.md guidance

**File:** `fdars-core/src/optimal_design.rs:26`

**Issue:** The RESEARCH.md "Don't Hand-Roll" and "Anti-Patterns" tables recommend using
`cholesky_solve` as the public consolidated API. The implementation instead imports only
`{cholesky_factor, cholesky_forward_back, log_det_from_cholesky}` and uses the lower-level
factor-once-then-solve-many pattern. This is functionally correct and actually more
efficient (O(p³) factorization amortized over m grid points), but diverges from the
documented recommendation.

The choice is technically superior for the trajectory criterion's inner loop, but a
reviewer arriving fresh at this code would expect `cholesky_solve` to appear given the
research documentation.

**Fix:** No code change required — the implementation is correct and more efficient. Add
a comment explaining the deliberate choice:
```rust
// Import the factor/forward-back pair directly rather than cholesky_solve:
// trajectory_criterion factors Σ_d once (O(p³)) and then solves m right-hand-sides
// via cholesky_forward_back (O(p²) each), avoiding O(m·p³) re-factorizations.
use crate::linalg::{cholesky_factor, cholesky_forward_back, log_det_from_cholesky};
```

---

_Reviewed: 2026-09-02T23:30:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_

---

## Resolution

**Fix commit:** `2fdaa9bb760afa69be8ce2e038ca26fa07c88345`
`fix(64): correct D-opt sign doc, add m>=2 guard, ridge-retry posterior Cholesky, real ridge test (code review)`

All 4 warnings + 1 info resolved in `fdars-core/src/optimal_design.rs` (single-file change, 103 insertions / 12 deletions).

- **WR-01 — RESOLVED.** Rewrote the `OptimalityKind::D` doc and the inline D-opt
  comment. They no longer claim the value is "NEGATIVE for an informative design."
  New text states D-opt returns `Σ_k log(posterior eigenvalues)`, is monotone
  NON-INCREASING as points are added, and its SIGN depends on the eigenvalue scale
  (with the `λ = [2, 1] → +0.693` example). Code unchanged (still un-negated).

- **WR-02 — RESOLVED.** Added an `m < 2` entry-point guard (alongside the
  sigma2 / ncomp / index-range guards) returning
  `InvalidParameter { parameter: "model.argvals", .. }`. New test
  `test_validation_grid_too_small` builds a model with `m = 1` (constructed directly,
  since `synthetic_model_params` divides by `m-1`) and asserts the exact variant.

- **WR-03 — RESOLVED.** Added helper `factor_posterior_cov_with_retry`, mirroring
  `factor_sigma_design_with_retry`: on a failed posterior-covariance Cholesky it adds a
  `1e-8 · scale` diagonal ridge (scale = max |diag|, ≥ 1) and retries once, else returns
  `ComputationFailed` — never panics. D-opt now succeeds wherever A-opt does. The D-opt
  branch calls this helper instead of a bare `cholesky_factor`.

- **WR-04 — RESOLVED.** `test_ridge_retry` now forces Σ_d genuinely non-PD by
  duplicating a design index (`selected = [10, 10]`, tolerated per docs) with
  `sigma2 = 1e-13`. The two identical rows make Σ_d rank-1 + σ²I; its second Cholesky
  pivot ≈ 2·σ² ≈ 2e-13 ≤ the 1e-12 threshold, so the FIRST factorization fails and only
  the 1e-8 ridge rescues it. The test asserts `Ok(..)` AND asserts the raw
  `cholesky_factor(&Σ_d, 2)` is `Err(..)` as a precondition — so it now fails if the
  retry branch is removed.

- **IN-01 — RESOLVED (comment only, no behavior change).** Added a comment at the
  `use crate::linalg::{...}` import explaining the deliberate factor-once /
  forward-back-many choice over `cholesky_solve` (amortizes the O(p³) factorization
  across the m trajectory solves instead of re-factoring O(m) times).

**Gates (all green):**
- `cargo fmt -p fdars-core` — clean
- `cargo test -p fdars-core --features linalg optimal_design` — 15 passed, 0 failed
- `cargo clippy -p fdars-core --all-targets --features linalg,parallel -- -D warnings` — clean
- `cargo test -p fdars-core --features linalg,parallel` — 2672 (lib) + integration all passed, 0 failed

No findings were declined.
