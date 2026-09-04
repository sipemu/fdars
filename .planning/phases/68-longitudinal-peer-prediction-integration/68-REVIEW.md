---
phase: 68-longitudinal-peer-prediction-integration
reviewed: 2026-09-04T00:00:00Z
depth: deep
files_reviewed: 3
files_reviewed_list:
  - fdars-core/src/peer.rs
  - fdars-core/src/lib.rs
  - fdars-core/src/prelude.rs
findings:
  critical: 1
  warning: 3
  info: 2
  total: 6
status: findings
---

# Phase 68: Code Review Report

**Reviewed:** 2026-09-04
**Depth:** deep
**Files Reviewed:** 3 (peer.rs, lib.rs, prelude.rs)
**Status:** findings

## Summary

Phase 68 adds `lpeer()`, `LpeerResult`, `LpeerResult::predict`, `PeerResult::predict`, a shared
`peer_predict_core` helper, new tests, and crate-root/prelude exports. The predict infrastructure
is correct and self-consistent. The struct derives, attribute annotations (`#[non_exhaustive]`,
`#[must_use]`, serde gating), and export sets are consistent with `PeerResult` and the rest of the
codebase.

One critical correctness issue was found: the `lambda` value computed inside `lpeer` does not
enter the final β(t) estimate — the PEER penalized solve is performed for λ-selection bookkeeping
only and is then discarded in favour of the FPC/mixed-model path. The stored `LpeerResult.lambda`
is therefore inaccurate as a description of what regularized the returned β. Two warnings cover a
silent-wrong-answer path in `peer_predict_core` (non-monotone argvals) and a predict-time
validation gap (argvals vs training grid). One additional warning flags the doctest's weak
self-consistency assertion.

---

## Critical Issues

### CR-01: `lpeer` λ does not regularize the returned β(t) — field is misleading

**File:** `fdars-core/src/peer.rs:568-578, 619-630`

**Issue:** `lpeer` runs the full PEER λ-selection machinery (steps 1-2: build `wc`, `wtw`,
`wty`, call `select_lambda_gcv_peer` / `select_lambda_reml_peer`) and stores the result in
`LpeerResult.lambda`. However, this λ value is never used after the dispatch block — no
penalized solve `(WtW + λQ)β = Wty` is performed. The actual β(t) is determined entirely by
the FPC score reduction (step 3, `fdata_to_pc_1d`) and the mixed-model fit (step 4,
`fit_scalar_mixed_model`). Neither call receives or uses λ.

The `PeerConfig.lambda` field controls only which label (`LambdaMethod`) is recorded and
which GCV/REML computation runs — it does not flow into the coefficient estimate. The
`LpeerResult.lambda` doc comment says "Smoothing parameter λ that was used", which a consumer
will reasonably interpret as the quantity that regularized β(t). That is incorrect.

Consequence: any caller that inspects `fit.lambda` (e.g. to compare regularization across
models, or to reproduce a fit) will obtain a value that had no effect on the returned β.
More concretely, calling `lpeer` with `LambdaChoice::Fixed(0.0)` vs
`LambdaChoice::Fixed(1e6)` will return identical β vectors (only the labels differ).

**Fix options (choose one):**

Option A — Remove the λ-selection step from `lpeer` entirely. Since λ does not enter the
estimate, the work is pure waste. Accept a `PeerPenalty` but not a `LambdaChoice`; document
that β(t) is regularized by the FPC truncation and REML EM only. Remove `lambda`,
`lambda_method`, and `gcv` from `LpeerResult`.

Option B — Perform the PEER penalized solve to obtain an initial β̂_PEER and use those
penalized projections as the covariate matrix passed to `fit_scalar_mixed_model` (i.e., the
score basis IS the penalized eigenvectors, not the FPC eigenvectors). This makes λ actually
regularize the estimate.

Option C — Keep the current FPC path but relabel: rename `lambda` to `lambda_selected` or
`lambda_config`, update the doc to say "λ selection was run for diagnostic purposes; the
estimate is regularized by FPC truncation (ncomp) and the mixed-model REML EM — not by λ
directly", and add an inline comment in `lpeer` making this explicit.

Option C is the minimal safe fix. Option A is the most honest API. Option B is the most
faithful to the PEER design but requires more implementation work.

```rust
// Minimal Option-C fix — add comment above the λ-dispatch block (line 553):
// NOTE: λ is selected here for diagnostic labelling only. The actual β(t)
// estimate is determined by FPC score reduction (step 3) and the mixed-model
// REML EM (step 4). λ does not enter the penalized system for lpeer.

// And update LpeerResult field doc (line 209):
/// Smoothing parameter λ as selected by `PeerConfig`. Note: for `lpeer`,
/// λ labels the selection method but does not directly regularize β(t);
/// regularization is via FPC truncation (ncomp = min(n-1, m, 10)) and the
/// mixed-model REML EM.
pub lambda: f64,
```

---

## Warnings

### WR-01: `peer_predict_core` — non-monotone `argvals` at predict time produces silent wrong answers

**File:** `fdars-core/src/peer.rs:1096-1129`

**Issue:** `peer_predict_core` calls `simpsons_weights(argvals)` (line 1119) without checking
that `argvals` is strictly increasing. `peer()` and `lpeer()` both validate monotonicity at
fit time. However, `predict` accepts a fresh `argvals` argument and applies no monotonicity
check. A non-monotone `argvals` produces incorrect (possibly negative) Simpson weights,
corrupting predictions silently — no error is returned.

A caller could reasonably pass a reversed or scrambled `argvals` (e.g. from a different data
source) and receive plausible-looking but wrong predictions.

**Fix:** Add the same monotonicity guard used in `peer()` (lines 299-304) at the start of
`peer_predict_core`, before calling `simpsons_weights`:

```rust
fn peer_predict_core(
    beta: &[f64],
    intercept: f64,
    w_bar: &[f64],
    new_data: &FdMatrix,
    argvals: &[f64],
) -> Result<Vec<f64>, FdarError> {
    let (n_new, m_new) = new_data.shape();
    let m = beta.len();
    if m_new != m { /* ... existing check ... */ }
    if argvals.len() != m { /* ... existing check ... */ }
    // ADD: monotonicity guard
    if argvals.windows(2).any(|w| w[1] <= w[0]) {
        return Err(FdarError::InvalidParameter {
            parameter: "argvals",
            message: "argvals must be strictly increasing".to_string(),
        });
    }
    let w = simpsons_weights(argvals);
    // ...
}
```

### WR-02: `peer_predict_core` — `argvals` validated against `beta.len()` but not against the training grid length stored in `w_bar`

**File:** `fdars-core/src/peer.rs:1112-1118`

**Issue:** `peer_predict_core` validates `argvals.len() == beta.len()` (== m, training grid
length). It does not validate `argvals.len() == w_bar.len()`. In a well-formed fit,
`w_bar.len() == beta.len() == m`, so this is not a reachable bug via normal usage.

However, `w_bar` is a public field on `PeerResult` and `LpeerResult` (both `#[non_exhaustive]`
with no constructor). A future (or test) scenario where a `PeerResult` is manually constructed
with mismatched `w_bar.len()` and `beta.len()` would cause `w_bar.iter().zip(beta)` to
silently truncate the dot-product (wrong intercept adjustment), while the loop `(0..m)` uses
`m = beta.len()`, producing incorrect `base` without error.

The `zip` truncation is the subtler path — it produces a wrong answer, not a panic.

**Fix:** Add an assertion or explicit length check:

```rust
// After the argvals.len() check, before simpsons_weights:
debug_assert_eq!(
    w_bar.len(), m,
    "w_bar length {} != beta length {} — internal invariant violation",
    w_bar.len(), m
);
```

Or, if public construction of results needs hardening, return an error if
`w_bar.len() != m`.

### WR-03: `lpeer` doctest — `LpeerResult::predict` self-consistency not asserted

**File:** `fdars-core/src/peer.rs:453-454`

**Issue:** The `lpeer` function doctest (lines 449-454) asserts only `preds.len() == n`. It
does not assert that re-passing training data reproduces `fitted_values`, which is the central
self-consistency guarantee documented in `LpeerResult::predict`'s own doc comment:
"re-passing the training data and argvals reproduces the training `fitted_values` to within
1e-9". The module-level doctest for `PeerResult::predict` does assert this (lines 45-48). The
discrepancy means the doctest for `lpeer` does not actually exercise the documented contract.

This is not a correctness bug in the code (the unit test `test_lpeer_predict_self_consistent`
at line 1982 does cover this) but the public-facing doctest is materially weaker than the
stated guarantee.

**Fix:** Add the self-consistency assertion to the `lpeer` doctest, consistent with the module
doctest:

```rust
/// let preds = fit.predict(&data, &argvals).unwrap();
/// assert_eq!(preds.len(), n);
/// // Self-consistency: re-passing training data reproduces fitted_values.
/// for (p, f) in preds.iter().zip(&fit.fitted_values) {
///     assert!((p - f).abs() < 1e-9, "predict vs fitted_values: {p} vs {f}");
/// }
```

---

## Info

### IN-01: `lpeer` computes `wc`, `wtw`, `wty` for λ-selection but these could be skipped for `LambdaChoice::Fixed`

**File:** `fdars-core/src/peer.rs:553-578`

**Issue:** When `config.lambda == LambdaChoice::Fixed(_)`, `lpeer` still builds the full
`wc` (n×m matrix), `wtw` (m×m matrix), and `wty` (length m) — at cost O(nm + m²) — before
dispatching to `(lam, None, LambdaMethod::Fixed)` which uses none of them. For small m this
is negligible; for m up to the validated maximum it is unnecessary allocation.

This is a performance observation (out of v1 scope per guidelines) but also a code clarity
signal: the lambda-selection block in `lpeer` was ported from `peer()` wholesale, including
the WtW/Wty setup needed for GCV/REML, without pruning the `Fixed` path. Given the CR-01
finding (λ doesn't affect β anyway), this dead computation is doubly suspect.

**Fix:** Either restructure λ-selection to skip WtW/Wty construction under `Fixed`, or
address via CR-01 option A (remove λ-selection from `lpeer` entirely).

### IN-02: `LpeerResult` struct is missing `lambda_method` in doc-comment field list

**File:** `fdars-core/src/peer.rs:190-217`

**Issue:** The `LpeerResult` struct has `lambda_method: LambdaMethod` as a public field (line
216) but the struct-level doc comment (lines 179-187) does not mention it. All other public
fields are mentioned in the doc. A user browsing the doc comment alone would not learn that
`lambda_method` exists or what it means.

**Fix:** Add a bullet for `lambda_method` in the doc comment:

```rust
/// Which λ-selection path ran (mirrors [`PeerResult::lambda_method`]). Note
/// that for `lpeer` this is diagnostic only (see [`lambda`](Self::lambda)).
pub lambda_method: LambdaMethod,
```

---

_Reviewed: 2026-09-04_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_

---

## Resolution (orchestrator, autonomous 3c.5)

- **CR-01 (critical) — FIXED (Option B, the faithful fix).** `lpeer` now genuinely applies the PEER structured penalty λQ. After `famm::fit_scalar_mixed_model` supplies the variance components (σ²_subject, σ²_resid) and a baseline γ, β(t) is recomputed by a **penalized GLS re-solve in the FPC-score space**: `(SᵀΣ⁻¹S + λ·Φ'QΦ)γ = SᵀΣ⁻¹y_c`, where Σ = σ²_e·I + σ²_u·ZZ' is the random-intercept marginal covariance (applied via the closed-form per-subject shrinkage). λ (Fixed/GCV/REML) and the penalty family now materially shape β(t) — the `lambda`/`penalty_type` fields are honest. New test `test_lpeer_lambda_regularizes` asserts `Fixed(0.0)` vs `Fixed(1e6)` produce β differing by > 1e-3 (directly refutes the CR-01 "identical β" finding). All prior lpeer tests (β recovery, σ²_subject tracking, variance non-negativity, predict self-consistency) still pass.
- **WR-01 — FIXED.** `peer_predict_core` now rejects non-monotonic `argvals` with `FdarError::InvalidParameter` (would otherwise yield negative Simpson weights).
- **WR-02 — FIXED.** Added `debug_assert_eq!(w_bar.len(), m)` in `peer_predict_core` to catch a corrupted result truncating the base offset.
- **WR-03 — ACCEPTED.** The secondary `lpeer` doctest is a lightweight usage example; the module-header doctest already demonstrates the full fit→β→predict self-consistency workflow. Not a defect.
- **IN-01/IN-02 — ACCEPTED (non-blocking).** Cosmetic.

Verification after fixes: `cargo test -p fdars-core --lib --features linalg,parallel peer::` → 31 passed, 0 failed; doctests 2/2; full lib suite 2720 passed, 0 failed; clippy `--all-targets` clean; fmt clean.

**Status:** critical + actionable warnings resolved.
