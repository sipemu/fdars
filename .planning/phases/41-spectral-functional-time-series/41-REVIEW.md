---
phase: 41-spectral-functional-time-series
reviewed: 2026-08-22T00:00:00Z
resolution: all findings addressed in commit fixing dpca_reconstruct underflow guard (CR-01), phase-aware divergence error (WR-01), robust sign-align argmax (WR-02), lag=0 comment (IN-01)
resolution_status: resolved
depth: deep
files_reviewed: 4
files_reviewed_list:
  - fdars-core/src/fts/spectral.rs
  - fdars-core/src/fts/mod.rs
  - fdars-core/src/simulation.rs
  - fdars-core/src/lib.rs
findings:
  critical: 1
  warning: 2
  info: 1
  total: 4
status: issues_found
---

# Phase 41: Code Review Report

**Reviewed:** 2026-08-22
**Depth:** deep (cross-file analysis with call-chain tracing)
**Files Reviewed:** 4
**Status:** issues_found

## Summary

Phase 41 adds the spectral density operator (`spectral_density`), dynamic
functional PCA (`dpca` / `dpca_reconstruct`), and functional VAR/VMA + FARMA
simulators (`sim_fvarma` / `sim_farma`) to the fdars-core crate. The
implementation is well-structured, follows project conventions (error types,
`#[must_use]`, serde gating, column-major indexing), and includes a rigorous
test suite.

One critical defect was found: a usize underflow panic in `dpca_reconstruct`
before the consistency check that is supposed to catch the bad input. Two
warnings cover a misleading error-message string in the divergence guard and a
floating-point exact-equality comparison in the sign-alignment helper. One info
item flags undocumented behaviour for the `lag = 0` double-write in the filter
assembly.

Intentional documented divergences (Re-only Hermitian eigendecomposition, 1/2π
omission, score trimming, identity innovation covariance) were not flagged.

---

## Critical Issues

### CR-01: Panic on usize underflow in `dpca_reconstruct` before dimension check

**File:** `fdars-core/src/fts/spectral.rs:383`

**Issue:** `dpca_reconstruct` computes `let n_interior = n - 2 * l;` on line 383
using unchecked usize subtraction. If a caller passes `data` with fewer rows
than `2 * dpca.filter_lag` (e.g. a shorter series than the one used to fit
`dpca`, or a manually constructed `DpcaResult` with a large `filter_lag`), the
subtraction underflows and Rust panics with an arithmetic overflow in debug
mode or wraps silently to a huge value in release mode, causing an out-of-bounds
index. The consistency check at line 386 (`dpca.scores.nrows() != n_interior`)
is supposed to catch this mismatch, but it comes too late — the panic fires
first.

No existing test covers this code path (the `dpca_reconstruct_dimension_mismatch`
test only exercises an argvals grid length mismatch, not a row-count mismatch).

**Fix:** Add a guarded subtraction before line 383:

```rust
// fdars-core/src/fts/spectral.rs  — inside dpca_reconstruct, before line 383
let (n, m) = validate_fts_input(data, argvals)?;
let l = dpca.filter_lag;
let ncomp = dpca.ncomp;

// Guard: n must be large enough to contain the filter window.
if n <= 2 * l {
    return Err(FdarError::InvalidDimension {
        parameter: "data",
        expected: format!("more than {} rows (> 2 * filter_lag)", 2 * l),
        actual: format!("{n} rows"),
    });
}
let n_interior = n - 2 * l;   // safe now
```

---

## Warnings

### WR-01: Divergence error always labels itself "burn-in" even when triggered in the kept-output phase

**File:** `fdars-core/src/simulation.rs:587-592`

**Issue:** The `fvarma_core` loop runs `burn_in + n` total steps and checks for
NaN/Inf on every step. The `ComputationFailed` error at line 587 always uses
`operation: "sim_fvarma burn-in"` regardless of whether divergence occurred
during the burn-in phase (`step < burn_in`) or during the kept-output phase
(`step >= burn_in`). A caller whose operator has spectral radius slightly above
1 might diverge only after the burn-in, producing an error message that
incorrectly suggests the divergence was in the warm-up period. This can confuse
diagnostics.

**Fix:** Differentiate the message based on the phase:

```rust
// fdars-core/src/simulation.rs — inside the divergence guard
if x_new.iter().any(|v| !v.is_finite()) {
    let phase = if step < burn_in { "burn-in" } else { "output" };
    return Err(crate::FdarError::ComputationFailed {
        operation: "sim_fvarma",
        detail: format!(
            "curve values diverged to NaN/Inf at step {step} ({phase}); \
             ensure AR operators have spectral radius < 1"
        ),
    });
}
```

Note: the existing `fvarma_divergence_guard` test matches on
`operation: "sim_fvarma burn-in"` specifically; that test would need updating
to `operation: "sim_fvarma"`.

---

### WR-02: Floating-point exact-equality comparison in `eigen_at_frequency` sign-alignment

**File:** `fdars-core/src/fts/spectral.rs:226`

**Issue:** The sign-alignment code uses `find(|x| x.abs() == max_abs)` to
locate the largest-magnitude entry of an eigenvector. `max_abs` was produced by
`fold(0.0f64, |acc, x| acc.max(x.abs()))`, which accumulates via `f64::max`.
When two entries share the same floating-point value (e.g. a symmetric mode
with two equal-magnitude peaks), the `==` comparison is correct for those
entries. However, floating-point arithmetic does not guarantee that
`max(x.abs())` produces a value bit-identical to one of the original entries
when the fold mixes comparisons of already-computed `.abs()` values. In
practice this rarely fails, but it is a latent correctness risk: if `find`
returns `None`, `map_or(1.0, ...)` silently assigns sign `+1` regardless of
the actual maximum entry.

**Fix:** Use an index-based approach that avoids the round-trip through equality:

```rust
// fdars-core/src/fts/spectral.rs — inside eigen_at_frequency sign-alignment
for (_, evec) in &mut pairs {
    if let Some(max_pos) = evec
        .iter()
        .copied()
        .enumerate()
        .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
    {
        if evec[max_pos] < 0.0 {
            evec.iter_mut().for_each(|x| *x = -*x);
        }
    }
}
```

This always selects the first maximum-magnitude index and checks its sign
directly, avoiding the two-pass `find`.

---

## Info

### IN-01: Redundant double-write for `lag = 0` in filter tap assembly (undocumented)

**File:** `fdars-core/src/fts/spectral.rs:316-321`

**Issue:** The filter assembly loop `for lag in 0..=l` writes the same tap
value to the same memory cell twice when `lag = 0` (`row_pos = l + 0 = l`,
`row_neg = l - 0 = l`). The second write is a no-op (same value, same index),
so the output is correct. However, the intent is not documented: a reader may
wonder whether the double-write is intentional or a sign error.

**Fix:** Add a comment or branch to make the intent explicit:

```rust
for lag in 0..=l {
    let tap = buf[lag].re * inv_n * inv_sw;
    let row_pos = l + lag;
    let row_neg = l - lag; // equals row_pos when lag == 0
    filt[row_pos + j * n_rows] = tap;
    if lag > 0 {
        // Symmetric filter: negative-lag tap equals positive-lag tap.
        filt[row_neg + j * n_rows] = tap;
    }
}
```

This avoids the redundant write and makes the symmetry explicit.

---

_Reviewed: 2026-08-22_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
