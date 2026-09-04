---
phase: 66-core-peer-estimator-penalty-families
reviewed: 2026-09-04T00:00:00Z
depth: deep
files_reviewed: 2
files_reviewed_list:
  - fdars-core/src/peer.rs
  - fdars-core/src/lib.rs
findings:
  critical: 1
  warning: 3
  info: 2
  total: 6
status: findings
---

# Phase 66: Code Review Report

**Reviewed:** 2026-09-04
**Depth:** deep (cross-file analysis including helper contracts)
**Files Reviewed:** 2 (peer.rs NEW, lib.rs one-line change)
**Status:** findings

## Summary

`peer.rs` is a well-structured PEER estimator implementation. The normal-equations
assembly, Cholesky reuse, penalty-family dispatch, and validation layering are all
sound for the mainline path. One critical design gap affects out-of-sample usability
of `PeerResult`: the stored `intercept` is not sufficient to reconstruct fitted values
for new observations using the result struct's public fields. Three warnings cover
missing input-validity guards (unsorted argvals, non-finite inputs, n==1 degenerate
case) that will silently produce wrong results rather than returning errors. Two info
items cover minor quality issues.

Reference reads confirmed:
- `cholesky_factor`/`cholesky_forward_back`/`cholesky_solve` in `linalg.rs` are
  `pub(crate)` without a feature gate — import in `peer.rs` is valid.
- `penalty_matrix` in `function_on_scalar.rs` returns an m×m row-major D'D matrix,
  and returns all-zeros for m < 3 — peer's `m >= 3` guard makes this dead.
- `simpsons_weights` in `helpers.rs` returns weights without validating that the grid
  is sorted or finite — no pre-validation in `peer.rs` before calling it.

---

## Critical Issues

### CR-01: `PeerResult` does not store `w_bar` — out-of-sample prediction is impossible from the result struct

**File:** `fdars-core/src/peer.rs:88-108` (struct definition), `95-99` (intercept field),
`234-237` (fitted values)

**Issue:** The fitted values are computed as:

```rust
let fitted_values: Vec<f64> = (0..n)
    .map(|i| y_bar + (0..m).map(|j| wc[(i, j)] * beta[j]).sum::<f64>())
    .collect();
```

Here `wc[i,j] = data[i,j] * w[j] - w_bar[j]`, where `w_bar[j]` is the column mean
of the weighted design (training-set-specific). This is mathematically correct for the
*training* fitted values, but `w_bar` is **not stored** in `PeerResult`.

The `intercept` field is documented as "= ȳ, the mean of the response vector" (line 99),
but the true prediction intercept for out-of-sample observations is:

```
ŷ_new = ȳ + Σ_j (data_new[i,j] * w[j] - w_bar[j]) * β[j]
       = (ȳ - Σ_j w_bar[j] * β[j]) + Σ_j data_new[i,j] * w[j] * β[j]
```

The actual intercept for prediction from raw (uncentered) inputs is
`ȳ - Σ_j w_bar[j] * β[j]`, which differs from the stored `intercept = ȳ` whenever
`w_bar · β ≠ 0`. A caller who reconstructs predictions as
`intercept + Σ_j data_new[i,j] * w[j] * β[j]` will get systematically biased values.
Since `w_bar` is not stored, the result struct is incomplete for inference.

This will be especially misleading once Phase 68 adds a `predict()` function — if it
relies on the stored `intercept` alone, all out-of-sample predictions will be wrong.

**Fix:** Store `w_bar` (the column means of the weighted design) in `PeerResult`,
and document the prediction formula explicitly:

```rust
pub struct PeerResult {
    pub beta: Vec<f64>,
    pub intercept: f64,        // ȳ — mean of training response
    pub w_bar: Vec<f64>,       // column means of W_c before centering (length m)
    pub fitted_values: Vec<f64>,
    pub effective_df: f64,
    pub lambda: f64,
    pub penalty_type: PeerPenalty,
}
```

And update the doc comment to clarify that the prediction formula for new data is:
`ŷ_new = intercept + Σ_j (wmat_new[i,j] - w_bar[j]) * β[j]`

or equivalently (pre-compute the adjustment once):
`ŷ_new = (intercept - w_bar · β) + Σ_j data_new[i,j] * w[j] * β[j]`

---

## Warnings

### WR-01: No validation that `argvals` is monotonically increasing — negative/zero weights silently corrupt the solve

**File:** `fdars-core/src/peer.rs:173-174`

**Issue:** `simpsons_weights(argvals)` is called without checking that `argvals` is
sorted. `simpsons_weights` in `helpers.rs` computes `h = argvals[k+1] - argvals[k]`
and uses those as interval widths. A non-monotone grid (e.g., reversed or scrambled)
produces negative or zero weights. These propagate silently into `wmat`, `WtW`, and
the normal equations, yielding a mathematically invalid system that may still produce
a finite β through the Cholesky solve — no error is raised.

**Fix:** Add a monotonicity guard before line 174:

```rust
if argvals.windows(2).any(|w| w[1] <= w[0]) {
    return Err(FdarError::InvalidParameter {
        parameter: "argvals",
        message: "evaluation grid must be strictly monotone increasing".to_string(),
    });
}
```

---

### WR-02: No finite-value guard on `argvals` or `y` — non-finite inputs bypass the NaN guard on β

**File:** `fdars-core/src/peer.rs:143-171` (validation block)

**Issue:** The validation block checks lengths and n/m bounds but does not check that
`argvals` or `y` contain finite values. Non-finite values in `argvals` cause
`simpsons_weights` to return `NaN` or infinite weights; non-finite values in `y`
produce a `NaN` RHS `wty`. The NaN guard at line 224 checks only `beta`, but if
the contamination enters via weights rather than the RHS, the Cholesky system may
still converge to a finite (but mathematically meaningless) β with no error returned.

**Fix:** Add finite-value checks after the length checks:

```rust
if y.iter().any(|v| !v.is_finite()) {
    return Err(FdarError::InvalidParameter {
        parameter: "y",
        message: "response vector contains non-finite values (NaN or Inf)".to_string(),
    });
}
if argvals.iter().any(|v| !v.is_finite()) {
    return Err(FdarError::InvalidParameter {
        parameter: "argvals",
        message: "evaluation grid contains non-finite values".to_string(),
    });
}
```

---

### WR-03: `n == 1` silently produces a degenerate zero-β solution with no warning

**File:** `fdars-core/src/peer.rs:185-196` (centering block)

**Issue:** When `n == 1`, centering produces `yc = [0.0]` and every column of `wc`
is `[0.0]` (since the single row equals the column mean). Consequently `WtW = 0` and
`wty = 0`, so the penalized system `(λQ)β = 0` yields `β = 0` for any λ > 0. The
Cholesky succeeds (Ridge: A = λI), `beta = 0` passes the finite check, and
`PeerResult` is returned with all-zero β and `fitted_values = [ȳ]`. This is
arithmetically correct but scientifically meaningless — a silent degenerate case that
gives no indication to the caller.

The validation at line 144 only requires `n >= 1`. Adding a minimum of `n >= 2` (or
at minimum `n > m` for the system to be identifiable) would surface this early.

**Fix:**

```rust
if n < 2 {
    return Err(FdarError::InvalidDimension {
        parameter: "data",
        expected: "at least 2 observations for centering".to_string(),
        actual: "1 row".to_string(),
    });
}
```

---

## Info

### IN-01: `effective_df` has no lower-bound clamp — can silently return a negative value

**File:** `fdars-core/src/peer.rs:303`

**Issue:** `compute_peer_trace_hat` returns `trace.min(n as f64)` (upper clamp) but
does not clamp below 0.0. In pathological cases (near-rank-deficient WtW, very high λ)
the column-solve accumulation can yield a small negative trace due to floating-point
cancellation. The test at line 405 asserts `effective_df > 0.0`, so tests would catch
this on the fixture but a caller relying on the field semantically (e.g., to compute
GCV = RSS / (n - edf)²) would get a nonsensical negative denominator.

**Fix:** Change the clamp to:

```rust
trace.clamp(0.0, n as f64)
```

---

### IN-02: Doc-comment in `compute_peer_trace_hat` says "Falls back to `m as f64`" but this path is unreachable

**File:** `fdars-core/src/peer.rs:284-295`

**Issue:** The fallback on Cholesky failure at line 294 (`return m as f64`) documents
a defensive path, but if `cholesky_solve` at line 221 succeeded (which it must have
for the function to reach this point), then the *identical* matrix `A` re-built at
lines 289-292 with the *same* inputs will also succeed `cholesky_factor`. The fallback
is dead code. It is not harmful, but the doc comment ("Falls back to `m as f64` on
Cholesky failure") creates a false impression that `effective_df == m` is a real
observable outcome, which could confuse future callers checking for this sentinel.

**Fix:** Either remove the fallback and propagate the error, or add a comment
explaining why this path is theoretically unreachable and keep it as a silent
safety net:

```rust
// Unreachable in practice: A was already factored successfully in the main solve.
// Kept as a last-resort guard against future code paths.
let Ok(l) = cholesky_factor(&a, m) else {
    return m as f64;
};
```

---

## Scope Confirmation

No Phase 68 items (crate-root re-exports, `prelude` additions, `lpeer`, `predict`,
running doctests) were found to have leaked into this diff. The `pub mod peer;` in
`lib.rs` is the sole change outside `peer.rs` and is correct. Confirmed clean scope.

---

_Reviewed: 2026-09-04_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_

---

## Resolution (orchestrator, autonomous 3c.5)

All findings addressed in `fdars-core/src/peer.rs` (follow-up commit):

- **CR-01 (critical) — RESOLVED.** Added `pub w_bar: Vec<f64>` to `PeerResult` (Simpson-weighted design column means) so out-of-sample prediction can reconstruct the intercept exactly (`ȳ − w_bar·β`). `intercept` semantics unchanged (= ȳ), so existing tests hold; `#[non_exhaustive]` makes it additive. New test `test_peer_stores_w_bar_for_prediction` proves the prediction reconstruction reproduces training fitted values within 1e-9.
- **WR-01 — RESOLVED.** `argvals` must be strictly increasing → `InvalidParameter` (guards against negative Simpson weights). Test `test_peer_rejects_non_monotonic_argvals`.
- **WR-02 — RESOLVED.** Non-finite `y`/`argvals` → `InvalidParameter` (before they can bypass the β NaN guard). Test `test_peer_rejects_non_finite_y`.
- **WR-03 — RESOLVED.** `n < 2` → `InvalidDimension` (avoids the silent centering-collapse to β=0). Test `test_peer_rejects_single_observation`.

Verification after fixes: `cargo test -p fdars-core --lib --features linalg,parallel peer::` → 13 passed, 0 failed; clippy `--all-targets` clean; fmt clean; peer doctest compiles.

**Status:** all findings resolved.
