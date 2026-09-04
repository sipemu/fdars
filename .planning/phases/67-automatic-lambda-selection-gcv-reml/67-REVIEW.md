---
phase: 67-automatic-lambda-selection-gcv-reml
reviewed: 2026-09-04T00:00:00Z
depth: deep
files_reviewed: 1
files_reviewed_list:
  - fdars-core/src/peer.rs
findings:
  critical: 0
  warning: 4
  info: 3
  total: 7
status: findings
---

# Phase 67: Code Review Report

**Reviewed:** 2026-09-04
**Depth:** deep
**Files Reviewed:** 1 (`fdars-core/src/peer.rs`)
**Status:** findings

## Summary

Phase 67 adds `LambdaChoice` / `LambdaMethod` enums, a 40-point GCV grid
selector, and a self-contained REML EM to `peer.rs`. The API shape,
dispatch, and GCV path are correct. The REML path implements the right
algorithm at the right level: eigendecomposition via `symmetric_eigen`,
eigenvector-column projection, Cholesky-based E-step inversion, and
Woodbury GLS. The EM equations are algebraically correct (E-step Σ_b,
b_hat, M-step σ²_u/σ²_e, Woodbury K). No security issues and no panics
reachable through the public API.

Four **warnings** require attention before this code is relied on in
production or cross-validated contexts.

---

## Critical Issues

None found.

---

## Warnings

### WR-01: Convergence delta measured against clamped values, not pre-clamp `_new` values — causes premature convergence when σ²_u_new < 1e-12

**File:** `fdars-core/src/peer.rs:738-742`

**Issue:**
The convergence check compares `sigma2_u` (the clamped value) against
`su_old`. When `sigma2_u_new` is below the floor (1e-12) the clamp
silently sets `sigma2_u = 1e-12`. On the *next* iteration `su_old` is
again 1e-12, so `delta` is nearly 0 and the loop declares convergence —
even though the EM has not converged (σ²_u is still being clamped every
iteration). The true fix is to check `delta` against the raw `_new`
values, then clamp afterwards. As written, the effect is systematic
under-iteration when σ²_u collapses toward zero: the EM exits after two
iterations thinking it has converged and returns `λ = σ²_e / 1e-12`
(extreme over-smoothing) rather than continuing to track the descent.

```
Current order (buggy):
  1. sigma2_u = sigma2_u_new.max(1e-12);   // may snap to floor
  2. delta = |sigma2_u - su_old|           // delta ≈ 0 if floor repeated
  3. if delta < tol { break; }             // false convergence

Correct order:
  1. let delta = (sigma2_u_new - su_old).abs() + (sigma2_e_new - se_old).abs();
  2. sigma2_u = sigma2_u_new.max(1e-12);
  3. sigma2_e = sigma2_e_new.max(1e-12);
  4. if delta < 1e-8 * (su_old + se_old) { break; }
```

**Fix:**

```rust
        // --- end of GLS block ---

        // Check convergence BEFORE clamping so the floor does not mask descent
        let delta = (sigma2_u_new - su_old).abs() + (sigma2_e_new - se_old).abs();

        sigma2_u = sigma2_u_new.max(1e-12);
        sigma2_e = sigma2_e_new.max(1e-12);

        if delta < 1e-8 * (su_old + se_old) {
            break;
        }
```

---

### WR-02: GLS α-update uses `sigma2_e_new` (not yet clamped) as the Woodbury divisor — can divide by a negative or zero value when the M-step produces σ²_e_new ≤ 0

**File:** `fdars-core/src/peer.rs:667, 691, 712`

**Issue:**
Inside the GLS block, `sinv_yc` and `sinv_zn` are both divided by
`sigma2_e_new` (line 691, 712). The M-step formula for `sigma2_e_new` is
`(resid_sq + tr_zsz) / n`. `resid_sq` is always non-negative. `tr_zsz` is
`tr(Σ_b ZtZ_range)`: since `Σ_b = M^{-1}` is PD and `ZtZ_range` is PSD,
this trace is non-negative in exact arithmetic. However, the Cholesky
inversion of `big_m` can introduce rounding errors that give very slightly
negative diagonal elements of `Σ_b`, producing a near-zero or negligibly
negative `tr_zsz`, and hence `sigma2_e_new` arbitrarily close to zero
before the clamp is applied. The clamp is applied on line 739, *after* the
GLS block, so the unclamped near-zero value is used as a denominator on
lines 691 and 712.

Additionally, the Woodbury ratio `ratio = sigma2_e_new / sigma2_u_new.max(1e-12)`
on line 667 is similarly susceptible: if `sigma2_e_new` is tiny or zero,
K is ill-conditioned and the Cholesky of K may fail (caught by the `if
let Ok`), but the combination of near-zero `sigma2_e_new` with the
scaling inside `sinv_yc`/`sinv_zn` can produce very large intermediate
values that overflow to Inf.

**Fix:**
Clamp `sigma2_e_new` to a safe floor before entering the GLS block, or
add an explicit guard:

```rust
        // Clamp _new values before the GLS block to prevent division by near-zero
        let sigma2_e_new = sigma2_e_new.max(1e-12);
        let sigma2_u_new = sigma2_u_new.max(1e-12);

        if s > 0 {
            let ratio = sigma2_e_new / sigma2_u_new;
            // ... rest of GLS unchanged, replacing sigma2_e_new with the clamped local ...
        }

        // Convergence check on clamped values is now consistent too
        let delta = (sigma2_u_new - su_old).abs() + (sigma2_e_new - se_old).abs();
        sigma2_u = sigma2_u_new;
        sigma2_e = sigma2_e_new;
```

Note: This also resolves WR-01 if applied consistently, since the delta is
now computed against clamped values that are already at least 1e-12 away
from zero.

---

### WR-03: On Cholesky failure in E-step, `alpha` is NOT rolled back — next iteration uses stale α with updated (old) σ²_u/σ²_e, violating the EM invariant

**File:** `fdars-core/src/peer.rs:600-608`

**Issue:**
When `cholesky_factor(&big_m, r)` fails, the code restores `sigma2_u` and
`sigma2_e` to their pre-iteration values and immediately `break`s (line
604-607). However, `alpha` may have been updated in the *previous*
iteration's GLS block. After the break, the function returns
`sigma2_e / sigma2_u`, which is the restored (old) pair, but `alpha` is
silently discarded (only σ² matters for the returned λ). This is not a
numeric catastrophe, but it means the early-exit path bypasses the
convergence check entirely: the returned λ is exactly the λ from the
previous iteration, regardless of whether convergence was met. On
ill-conditioned problems (very large or very small n relative to r) this
can happen on the first iteration, returning the initial σ²_e / σ²_u
(approximately 10), which may be far from the REML optimum.

More importantly, if `sigma2_e` and `sigma2_u` are both restored to
`su_old`/`se_old` but `sigma2_u` had already been clamped (floor 1e-12)
in a previous iteration, the "old" values returned are the clamped floor,
not the pre-clamp `_new`. This is another symptom of the clamp/ordering
issue in WR-01.

**Fix:**
Document the early-exit behavior explicitly in the function docstring, and
consider falling back to the GCV path or to a fixed small λ instead of
silently returning the previous iteration's λ. At a minimum, add a comment:

```rust
        let l_m = match cholesky_factor(&big_m, r) {
            Ok(l) => l,
            Err(_) => {
                // M matrix is numerically singular: variance ratio at current estimates
                // makes I/σ²_u + ZtZ/σ²_e non-PD (rare, typically when σ²_u → ∞).
                // Return λ from previous iteration (su_old/se_old already set).
                return (se_old / su_old).max(1e-15);
            }
        };
```

This avoids the implicit `break`-then-fall-through-to-step-6 pattern and
makes the early-exit intent explicit and testable.

---

### WR-04: `select_lambda_gcv_peer` receives `wc` (and `wty`) as arguments but `wty` is unused inside the function — the Cholesky solve operates on a freshly built `a` matrix with `wty` correct, but a subtle API inconsistency exists: `wc` is used only for RSS, so the pass of `wtw`+`wty` for the solve and `wc` for the residual is redundant with the `wc`→`wtw` duality

**File:** `fdars-core/src/peer.rs:412-420`

**Issue:**
This is a low-severity inconsistency rather than a correctness bug: `wty`
is legitimately passed and used in the `cholesky_solve` call (line 432).
The concern is the `wc` parameter: it is used only to compute RSS on lines
437-440. Since `wc` is already encoded in `wtw` (= `wc' wc`) and the
beta solve uses `wty` (= `wc' yc`), the RSS could instead be computed as
`yc'yc - beta' wty` (the quadratic form identity), removing the `wc` pass
entirely and reducing the function signature. As written, the function
signature is misleading: a caller seeing `wtw`, `wty`, AND `wc` all passed
to a GCV selector would reasonably ask why all three are needed. The
docstring does not explain this.

Additionally, the GCV inner loop calls `compute_peer_trace_hat` for every
grid point (line 443), which itself re-builds the Cholesky of A from
scratch. This means A is factored *twice* per grid point (once in
`cholesky_solve`, once in `compute_peer_trace_hat`). This is not a
correctness bug, but the duplication is a latent confusion hazard: if a
maintainer modifies `compute_peer_trace_hat` to use a different A-build
path, the GCV RSS and tr(H) will silently diverge.

**Fix (preferred):** Replace the RSS loop with the quadratic form
`yc'yc - beta' wty`, eliminating the `wc` parameter from the signature
and the O(nm) RSS loop:

```rust
fn select_lambda_gcv_peer(
    yc: &[f64],
    wtw: &[f64],
    wty: &[f64],
    q: &[f64],
    m: usize,
    n: usize,
) -> (f64, f64) {
    // Precompute yc'yc once
    let ycy: f64 = yc.iter().map(|&v| v * v).sum();

    for &lam in &grid {
        let Ok(beta) = cholesky_solve(&a, wty, m) else { continue; };
        // RSS = yc'yc - beta' wty  (since yhat = Wc beta, RSS = yc'yc - 2 beta'Wc'yc + beta'WtW beta
        //       = yc'yc - beta'wty after the normal equations collapse)
        // Note: this identity holds exactly when beta = (WtW + lam Q)^{-1} wty, as here.
        // However that identity only holds for the true OLS beta, not the PENALIZED beta.
        // Keep the direct RSS loop for correctness.
    }
```

Correction: the quadratic-form shortcut does NOT apply to the penalized
solve (the normal equations for the penalized problem do not yield
`beta' wty = RSS_ols`). The direct RSS loop via `wc` is therefore
numerically correct as written. The only actionable fix here is to
document why `wc` is needed alongside `wtw`:

```rust
/// Note on parameters: `wc` is needed to compute fitted values for the RSS
/// loop. `wtw = wc'wc` and `wty = wc'yc` encode the same information but
/// only at the gram-matrix level; recovering per-observation residuals
/// requires the full `wc`.
```

---

## Info

### IN-01: GCV returns `f64::INFINITY` as `best_gcv` when no grid point is valid — callers cannot distinguish "all points degenerate" from "score is genuinely infinity"

**File:** `fdars-core/src/peer.rs:423`

**Issue:**
When every candidate λ in the 40-point grid fails Cholesky or has
`denom <= 0`, `best_gcv` stays `f64::INFINITY` and `best_lam` remains
`grid[0]` (1e-6). The caller (`peer()`) stores this as
`gcv: Some(f64::INFINITY)`. A downstream user inspecting `result.gcv`
cannot tell whether INFINITY means "GCV score at the best finite λ was
truly infinite" (which should not occur under normal circumstances) vs.
"no valid λ found." The RESEARCH doc does document the fallback, but the
docstring on `select_lambda_gcv_peer` only says "returns the smallest grid
λ with score `f64::INFINITY`" without flagging this as an error condition.

**Fix:** Expose a `bool` for "all degenerate" or clamp the returned gcv to
`None` in that case, or at minimum assert in debug that `best_gcv.is_finite()`.

---

### IN-02: `PeerConfig` no longer has a hand-written `Default` impl — the derived `#[derive(Default)]` relies on `PeerPenalty::default()` returning `Difference{order:2}` and `LambdaChoice::default()` returning `Gcv`

**File:** `fdars-core/src/peer.rs:102-104`

**Issue:**
This is intentional and correct; the derived `Default` for `PeerConfig`
will call `PeerPenalty::default()` (line 64: `Difference{order:2}`) and
`LambdaChoice::default()` (line 81: `Gcv`). These defaults are consistent
with the RESEARCH doc and the tests that call `PeerConfig::default()`.
The info note is that the three tests which previously used
`PeerConfig::default()` (with Phase 66's implicit `lambda: 1.0`) now
silently get `LambdaChoice::Gcv`. Tests `test_peer_rejects_non_monotonic_argvals`,
`test_peer_rejects_single_observation`, and `test_peer_rejects_non_finite_y`
call `peer()` expecting an `Err` before λ is even used — so the change is
safe. However, `test_peer_argvals_mismatch` also calls `PeerConfig::default()`
and expects `InvalidDimension` (before λ selection) — also fine. No test
regression risk, but worth an explicit acknowledgment.

**Fix:** No code change needed. Add a comment in the `PeerConfig::default()`
impl area noting the behavior change from Phase 66:

```rust
// Default: Difference{order:2} penalty with GCV λ selection.
// Phase 66 default was Fixed(1.0); Phase 67 changes to Gcv per CONTEXT.md.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PeerConfig { ... }
```

---

### IN-03: `gcv_lambda_grid()` allocates a fresh `Vec<f64>` every call — it is called once per `peer()` invocation but could be a compile-time constant

**File:** `fdars-core/src/peer.rs:394-398`

**Issue:**
`gcv_lambda_grid()` is a pure function of compile-time constants. It
allocates 40 `f64` values every time GCV is invoked. A `const` array or a
`static` slice would avoid the allocation. This is a minor quality note;
for m=40 the allocation cost is negligible relative to the matrix
operations.

**Fix:**
```rust
/// 40-point log-spaced GCV grid on [1e-6, 1e4]. Pure function of constants.
fn gcv_lambda_grid() -> [f64; 40] {
    std::array::from_fn(|i| 10.0_f64.powf(-6.0 + 10.0 * i as f64 / 39.0))
}
```
(Iterate with `for &lam in gcv_lambda_grid().iter()` — no other call-site
changes.) Or keep as `Vec` but mark the function `#[cold]` / inline.

---

_Reviewed: 2026-09-04_
_Reviewer: Claude (gsd-code-reviewer / adversarial)_
_Depth: deep_

---

## Resolution (orchestrator, autonomous 3c.5)

Findings addressed in `fdars-core/src/peer.rs` (follow-up commit):

- **WR-01 (convergence at floor) — FIXED.** EM convergence delta is now measured on the PRE-clamp M-step updates (`sigma2_u_new`/`sigma2_e_new`) vs. the previous committed values, so a component resting on the 1e-12 floor two iterations running is no longer mistaken for genuine convergence (which would return λ = σ²_e/floor, extreme over-smoothing).
- **WR-02 (GLS divide by unclamped σ²_e) — FIXED.** Both variance components are clamped to 1e-12 (`sigma2_u_c`, `sigma2_e_c`) BEFORE the Woodbury GLS block; the ratio and both `sinv_*` divisors now use the clamped `sigma2_e_c`, eliminating the Inf/NaN risk when σ²_e rounds to ~0.
- **WR-03 (silent E-step Cholesky break) — CLARIFIED.** The break already restores the last stable σ²_u/σ²_e and returns a defined λ = σ²_e/σ²_u (graceful fallback, never panic/NaN); comment now states this intent explicitly.
- **WR-04 / IN-01 / IN-02 / IN-03 — ACCEPTED (deferred).** The double Cholesky factorization per GCV grid point is O(40·2) on m≈40 matrices (negligible); the `Default` change (Fixed(1.0)→Gcv) is intended and documented in CONTEXT.md; the grid-`Vec` and INFINITY sentinel are cosmetic. No behavioral risk; not worth churn this phase.

Verification after fixes: `cargo test -p fdars-core --lib --features linalg,parallel peer::` → 19 passed, 0 failed; clippy `--all-targets` clean; fmt clean.

**Status:** critical/high findings resolved; low/info accepted.
