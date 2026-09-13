---
phase: 101-in-crate-l-bfgs-barycenter-optimizer
reviewed: 2026-09-13T00:00:00Z
depth: deep
files_reviewed: 1
files_reviewed_list:
  - fdars-core/src/metric/soft_dtw.rs
findings:
  critical: 1
  warning: 4
  info: 1
  total: 6
status: resolved
disposition:
  CR-01: fixed — commit 7d1aad18; non-breaking gamma guard returns mean barycenter with converged=false
  WR-01: deferred — soft_dtw_backward is CORR-01 code, out of scope for Phase 101; cancelling terms numerically correct and oracle-verified
  WR-02: fixed — commit 7d1aad18; rho_i stored in Vec, reused in both L-BFGS loops
  WR-03: fixed — commit 7d1aad18; #[must_use] added to soft_dtw_barycenter
  WR-04: deferred — strict lbfgs_cost < mm_cost assertion is the in-crate Phase 101 SC#1 verifier; deterministic f64 on this dataset; leave as-is
  IN-01: fixed — commit 7d1aad18; clarifying comment added to cost_monotone test
---

# Phase 101: Code Review Report

**Reviewed:** 2026-09-13
**Depth:** deep (full file read + numerical trace of L-BFGS algorithm vs. Nocedal & Wright)
**Files Reviewed:** 1
**Status:** issues_found

## Summary

Phase 101 replaces the removed MM (inverse-curvature diagonal-Newton) barycenter
with a hand-rolled L-BFGS optimizer: two-loop recursion, backtracking Armijo line
search (c1=1e-4, β=0.5, ≤20 backtracks), curvature safeguard (sᵀy > 1e-20),
non-descent direction reset, and initial Hessian scaling γ_k = (sᵀy)/(yᵀy).

The two-loop recursion, curvature safeguard, history eviction, Hessian scaling,
and descent-direction check are all correct. The objective / gradient assembly and
`soft_dtw_backward` are unchanged and correct.

One **BLOCKER** is present: the public `soft_dtw_barycenter` function accepts
`gamma ≤ 0` silently, producing NaN output — violating the project's mandatory
`InvalidParameter` guard convention and corrupting every downstream caller that
trusts the return value.

Four **WARNINGs** cover: an algebraically dead double-cancellation in the backward
pass that survived from an intermediate draft and produces dead arithmetic on every
backward call; a `#[must_use]` omission on an O(max_iter·n·m) function; a `rho_i`
double-computation maintenance trap in the two-loop recursion; and a fragile SC#1b
strict-inequality test assertion that is not guaranteed by the algorithm on all
platforms.

---

## Critical Issues

### CR-01: Missing `gamma <= 0` input validation in `soft_dtw_barycenter`

**File:** `fdars-core/src/metric/soft_dtw.rs:575`

**Issue:** `soft_dtw_barycenter` accepts `gamma: f64` without guarding against
`gamma <= 0`. The function's doc says `(> 0)` but no runtime check is performed.
When `gamma = 0.0`, `softmin3` computes `neg_inv_gamma = -1.0 / 0.0 = -inf`,
causing all exponent terms to produce NaN throughout the DP recurrence. The
returned `SoftDtwBarycenterResult::barycenter` becomes a NaN-filled vector with
`converged: true` (the gradient ∞-norm fold of NaN values yields 0.0 via
`f64::max`, which satisfies `< tol` on the first iteration — false convergence).

This is a silent data-corruption path: the caller receives a `converged: true`
result with an all-NaN barycenter. The existing `n == 0 || m == 0` early-return
pattern on line 582 shows the intent to validate inputs; `gamma` was simply missed.

By project convention (`.claude/CLAUDE.md`), all public functions must return
`Result<T, FdarError>` with `InvalidParameter` for out-of-range inputs. This
function currently returns the struct directly (not a `Result`), so the immediate
fix is a debug-assertion guard and/or converting to `Result` in a future API bump.
As a minimum non-breaking fix within Phase 101's `-> SoftDtwBarycenterResult`
signature, clamp or assert:

**Fix (non-breaking, defense-in-depth):**
```rust
pub fn soft_dtw_barycenter(
    data: &FdMatrix,
    gamma: f64,
    max_iter: usize,
    tol: f64,
) -> SoftDtwBarycenterResult {
    // Guard: gamma must be strictly positive; NaN propagates silently otherwise.
    assert!(gamma > 0.0 && gamma.is_finite(), "gamma must be finite and > 0, got {gamma}");
    let (n, m) = data.shape();
    // ...
```

Longer term, convert to `Result<SoftDtwBarycenterResult, FdarError>` per project
conventions and return:
```rust
return Err(FdarError::InvalidParameter {
    parameter: "gamma",
    message: "must be strictly positive".to_string(),
});
```

---

## Warnings

### WR-01: Dead double-cancellation arithmetic in `soft_dtw_backward` exponent terms

**File:** `fdars-core/src/metric/soft_dtw.rs:322, 331, 340-342`

**Issue:** The exponent expression for each backward contribution has an added
then immediately subtracted `r[i+k][j+k]` term that cancels algebraically to zero:

```rust
// Line 322 — "up" contribution:
-(r[i][j] - r[i+1][j] + r[i+1][j] - softmin3_val(r, i+1, j, gamma)) / gamma
//           ^^^^^^^^^^   ^^^^^^^^^^  these cancel
```

After cancellation this equals `-(r[i][j] - softmin3_val(r, i+1, j, gamma)) / gamma`,
which is exactly the formula in the `corrected_oracle_gradient` test helper at
lines 782, 787, 793. The same pattern appears in the "right" arm (line 331) and
"diagonal" arm (lines 340-342).

Functionally the result is correct (the test at line 857 passes), but the spurious
arithmetic performs two extra floating-point additions per E-matrix cell on every
backward pass — once per barycenter iteration per input series. More critically it
creates a maintenance trap: a future reader seeing `r[i+1][j] + r[i+1][j]` may
mistakenly "fix" it and introduce an actual sign error, or may not notice that the
simpler form in the oracle does not match the production code.

**Fix:** Remove the cancelling terms so the expression matches the oracle form:
```rust
// Contribution from (i+1, j): "up" move
let a = if i < n {
    e[i + 1][j]
        * (-(r[i][j] - softmin3_val(r, i + 1, j, gamma)) / gamma).exp()
} else {
    0.0
};
// Contribution from (i, j+1): "right" move
let b = if j < m {
    e[i][j + 1]
        * (-(r[i][j] - softmin3_val(r, i, j + 1, gamma)) / gamma).exp()
} else {
    0.0
};
// Contribution from (i+1, j+1): "diagonal" move
let c = if i < n && j < m {
    e[i + 1][j + 1]
        * (-(r[i][j] - softmin3_val(r, i + 1, j + 1, gamma)) / gamma).exp()
} else {
    0.0
};
```

---

### WR-02: `rho_i` recomputed twice in `two_loop_lbfgs`

**File:** `fdars-core/src/metric/soft_dtw.rs:468, 483`

**Issue:** `rho_i = 1.0 / lbfgs_dot(y, s)` is computed independently in Loop 1
(line 468) and Loop 2 (line 483). Because `lbfgs_dot` iterates the full history
vector each time, this is correct numerically but duplicates O(m) work per history
entry. More importantly it creates a maintenance risk: if a future change modifies
the curvature denominator in one loop (e.g., switching to the Oren-Luenberger
scaled inverse), the other loop becomes silently inconsistent.

The standard Nocedal & Wright Algorithm 7.4 stores `rho_i` values in the first
loop and reuses them in the second. The existing `alphas` array already follows
this pattern — `rho_i` should do the same:

**Fix:** Precompute `rho` values alongside `alphas`:
```rust
let mut alphas = vec![0.0_f64; m_actual];
let mut rhos   = vec![0.0_f64; m_actual];

// Loop 1: newest to oldest
for i in (0..m_actual).rev() {
    let s = &s_hist[i];
    let y = &y_hist[i];
    rhos[i]   = 1.0 / lbfgs_dot(y, s);
    alphas[i] = rhos[i] * lbfgs_dot(s, q_buf);
    lbfgs_axpy(-alphas[i], y, q_buf);
}

// Loop 2: oldest to newest
for i in 0..m_actual {
    let s    = &s_hist[i];
    let y    = &y_hist[i];
    let beta = rhos[i] * lbfgs_dot(y, d);
    lbfgs_axpy(alphas[i] - beta, s, d);
}
```

---

### WR-03: `#[must_use]` missing on `soft_dtw_barycenter`

**File:** `fdars-core/src/metric/soft_dtw.rs:575`

**Issue:** Project convention requires `#[must_use]` on expensive computation
functions (`.claude/CLAUDE.md`: "74+ functions marked"). `soft_dtw_barycenter`
runs up to `max_iter` iterations each invoking O(n × m) forward + backward DP
passes — it is clearly expensive. Without `#[must_use]`, a caller that accidentally
drops the return value (e.g., `soft_dtw_barycenter(&data, gamma, 100, tol);`)
compiles silently with no warning.

**Fix:**
```rust
#[must_use]
pub fn soft_dtw_barycenter(
    data: &FdMatrix,
    gamma: f64,
    max_iter: usize,
    tol: f64,
) -> SoftDtwBarycenterResult {
```

---

### WR-04: `lbfgs_cost_le_mm` SC#1b strict-inequality assertion is not algorithm-guaranteed

**File:** `fdars-core/src/metric/soft_dtw.rs:1175`

**Issue:** The test at line 1175 asserts `lbfgs_cost < mm_cost` (strictly lower,
not ≤). The assertion comment says "L-BFGS uses curvature information; MM is a
first-order diagonal step". This is true on average but is NOT guaranteed for all
datasets, all gamma values, or all platform floating-point implementations. The
soft-DTW barycenter objective is non-convex; L-BFGS can stall or take larger
initial steps that overshoot, while MM's per-coordinate majorization step can
sometimes take a better-conditioned path on a specific sinusoidal dataset.

If this assertion fails on any CI platform (e.g., Windows x86_64 with different
`f64` rounding), the test becomes a flaky blocker. SC#1a (`lbfgs_cost <= mm_cost +
tiny_eps`) is the correct, algorithm-backed assertion. SC#1b is speculative.

**Fix:** Replace the strict-lower assertion with the weaker but correct bound, or
gate SC#1b with an explicit `// empirical only — may fail under unusual FP environments`
comment and skip it from CI gates:
```rust
// SC#1b (empirical): L-BFGS typically finds a lower cost via curvature info.
// This holds on the sinusoidal test case but is not theoretically guaranteed.
// If this fails on a platform, it is acceptable to relax to SC#1a only.
if lbfgs_cost >= mm_cost {
    eprintln!(
        "WARN: L-BFGS cost ({lbfgs_cost:.8}) not strictly lower than MM ({mm_cost:.8}) — \
         L-BFGS still satisfied SC#1a (≤ MM + eps); SC#1b is empirical only"
    );
}
```

Or simply remove SC#1b and rely on SC#1a:
```rust
// SC#1a: L-BFGS cost must be no worse than MM
assert!(
    lbfgs_cost <= mm_cost + tiny_eps,
    "L-BFGS cost ({lbfgs_cost:.8}) > MM cost ({mm_cost:.8}) + eps ({tiny_eps:.2e})"
);
```

---

## Info

### IN-01: `cost_monotone` test measures cross-run monotonicity, not intra-run

**File:** `fdars-core/src/metric/soft_dtw.rs:977`

**Issue:** The `cost_monotone` test runs `soft_dtw_barycenter` independently for
budgets 1, 2, …, 10, each starting from the same pointwise-mean initialization,
then checks that the cost is non-increasing across these independent runs. This
works because the optimizer is deterministic: the run with `budget=k+1` reproduces
all k steps of the `budget=k` run and then takes one more Armijo-guaranteed descent
step. The test is valid, but it validates "determinism + one extra descent step" not
"the Armijo condition is satisfied within the loop". A comment clarifying this
reasoning would prevent future maintainers from misreading the test as proving
something about a single run's trajectory.

**Fix:** Add clarifying comment at line 981:
```rust
// Each call is independent (restarts from the same pointwise-mean init).
// Because the optimizer is deterministic, budget=k+1 reproduces all k steps of
// budget=k and then takes one additional Armijo-guaranteed descent step.
// So costs across runs are non-increasing iff (a) determinism holds and (b)
// Armijo is satisfied — both properties we want to verify.
```

---

_Reviewed: 2026-09-13_
_Reviewer: Claude Sonnet 4.6 (gsd-code-reviewer)_
_Depth: deep_
