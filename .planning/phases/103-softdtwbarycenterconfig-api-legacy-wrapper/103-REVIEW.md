---
phase: 103-softdtwbarycenterconfig-api-legacy-wrapper
reviewed: 2026-09-13T00:00:00Z
depth: standard
files_reviewed: 4
files_reviewed_list:
  - fdars-core/src/metric/soft_dtw.rs
  - fdars-core/src/metric/mod.rs
  - fdars-core/src/lib.rs
  - fdars-core/src/prelude.rs
findings:
  critical: 1
  warning: 2
  info: 1
  total: 4
status: resolved
---

# Phase 103: Code Review Report

**Reviewed:** 2026-09-13
**Depth:** standard
**Files Reviewed:** 4
**Status:** issues_found

## Summary

Phase 103 adds `SoftDtwBarycenterConfig`, `soft_dtw_barycenter_with_config`, and associated
re-exports across three sites. The implementation is structurally sound: the config struct is
correctly sealed with `#[non_exhaustive]`, the manual `Default` impl delivers the documented
non-zero values, `#[must_use]` is applied, and the parity test passes at runtime for normal inputs.

One critical correctness gap was found: the documented parity guarantee ("bit-identical to
`n_restarts=1`") breaks silently under NaN-valued input data because the two code paths diverge
on how they handle a cost total of NaN. Two warnings cover a doc/code inconsistency on wrapping
arithmetic and an undocumented silent clamp of `n_restarts=0`. One info item flags a confusing
field-level doc phrasing.

All tests pass (`cargo test` and `--doc` both green). `cargo clippy --all-targets --features
linalg,parallel -- -D warnings` is clean.

## Structural Findings (fallow)

No structural pre-pass was provided for this review.

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-01: Parity Guarantee Silently Broken for NaN-Valued Input

**File:** `fdars-core/src/metric/soft_dtw.rs:863`

**Issue:** The public documentation on `soft_dtw_barycenter_with_config` states a "bit-identical"
parity guarantee with `soft_dtw_barycenter` when `n_restarts=1`:

> "For callers who do not need multiple restarts, the 4-argument legacy entry point
> `soft_dtw_barycenter` remains available and is bit-identical to this function with
> `config.n_restarts = 1` (SDTW-03 non-breaking guarantee)."

This guarantee fails for NaN-valued input data. When `FdMatrix` contains NaN values (which
the constructor does not validate), the L-BFGS optimizer returns a NaN-valued barycenter. The
two paths then diverge:

- **Legacy path** (`soft_dtw_barycenter`): calls `lbfgs_refine_from_init` and returns its result
  directly — the NaN barycenter is returned.

- **Config path** (`soft_dtw_barycenter_with_config` with `n_restarts=1`): calls
  `soft_dtw_barycenter_multirestart`, which after `lbfgs_refine_from_init` evaluates the total
  cost `Σᵢ soft_dtw_distance(result.barycenter, row_i, gamma)`. If the barycenter is NaN, this
  sum is NaN. The comparison `NaN < f64::INFINITY` is `false`, so `best_result` stays `None`.
  The function then returns the `unwrap_or` fallback: the unperturbed pointwise mean with
  `n_iter=0, converged=false` (line 863–867).

The two paths thus return **different values** for the same NaN input. The parity test
`legacy_equals_config_single_restart` does not cover NaN inputs and therefore does not catch
this divergence.

`FdMatrix` does not guard against NaN at construction time (noted in the inline comment at
line 858), so this is a reachable code path. A user who calls `soft_dtw_barycenter_with_config`
after constructing a matrix with NaN sentinel values for missing data will receive the mean
(not the optimizer's NaN output), while a user of the legacy API would receive the NaN — an
observable behavioral difference.

**Fix:** Either:

1. Weaken the documented guarantee to match the actual behavior (mention NaN inputs are
   excluded from the parity guarantee), or

2. Make the legacy path delegate through `soft_dtw_barycenter_multirestart` as well, so
   both paths share exactly the same execution. This is cleaner and would look like:

```rust
pub fn soft_dtw_barycenter(
    data: &FdMatrix,
    gamma: f64,
    max_iter: usize,
    tol: f64,
) -> SoftDtwBarycenterResult {
    soft_dtw_barycenter_multirestart(data, gamma, max_iter, tol, 1, 0)
}
```

   Under this approach the legacy function would inherit the gamma guard and the NaN-cost
   fallback from the shared driver, and parity is guaranteed by construction — both paths
   call exactly the same code.

---

## Warnings

### WR-01: Doc/Code Mismatch on Restart Seeding Arithmetic

**File:** `fdars-core/src/metric/soft_dtw.rs:790`

**Issue:** The doc comment for `soft_dtw_barycenter_multirestart` describes the seeding formula
as `StdRng::seed_from_u64(seed + k)` (plain addition), while the actual implementation in
`perturbed_init` uses `seed.wrapping_add(k)` (wrapping addition). The field-level doc on
`SoftDtwBarycenterConfig::seed` correctly says `seed.wrapping_add(k)`, but the function doc
says `seed + k`. A user relying on the function doc to predict seeds when `seed` is near
`u64::MAX` would compute the wrong value.

```rust
// Doc says:         seed + k           (overflows for large seed)
// Code executes:    seed.wrapping_add(k)  (defined, wraps to 0)
```

**Fix:** Update line ~790 in the function doc to read `seed.wrapping_add(k)`:

```text
- * - Restart k ≥ 1: mean + additive Gaussian noise scaled to per-evaluation-point std,
- *   seeded with `StdRng::seed_from_u64(seed + k)`.
+ * - Restart k ≥ 1: mean + additive Gaussian noise scaled to per-evaluation-point std,
+ *   seeded with `StdRng::seed_from_u64(seed.wrapping_add(k))`.
```

---

### WR-02: Silent Clamp of `n_restarts = 0` Not Documented

**File:** `fdars-core/src/metric/soft_dtw.rs:830`

**Issue:** The multirestart driver clamps `n_restarts` to at least 1 via `n_restarts.max(1)`:

```rust
let n_runs = n_restarts.max(1);
```

A user who sets `config.n_restarts = 0` expecting "no restarts" (perhaps intending to skip
optimization entirely) will instead get a single restart (the unperturbed mean). This silent
clamping is not mentioned anywhere in the `SoftDtwBarycenterConfig::n_restarts` field doc.
Per project convention (CLAUDE.md: "Parameter range validation with contextual error messages"),
invalid inputs should either be rejected with `InvalidParameter` or explicitly documented as
accepted with a specific clamping behavior.

Since `soft_dtw_barycenter_with_config` does not return `Result`, the cleanest fix is to
document the clamping behavior:

```rust
/// Number of random restarts (default: `10`).
///
/// Restart 0 uses the unperturbed pointwise mean; restarts 1..n use
/// Gaussian-perturbed inits. The barycenter with the lowest total
/// soft-DTW cost over all restarts is returned.
///
/// A value of `0` is treated as `1` (the unperturbed mean restart only).
pub n_restarts: usize,
```

---

## Info

### IN-01: `n_restarts` Field Doc Ambiguous About Total Count vs. Perturbed Count

**File:** `fdars-core/src/metric/soft_dtw.rs:67–72`

**Issue:** The `n_restarts` field doc reads:

> "Restart 0 uses the unperturbed pointwise mean; restarts 1..n use Gaussian-perturbed inits."

A reader might parse "1..n" as `n` additional perturbed restarts on top of the mean, expecting
`n_restarts + 1` total runs. The actual behavior is `n_restarts` total runs (0 through
`n_restarts-1`), so `n_restarts=10` means 1 unperturbed + 9 perturbed, not 10 perturbed + 1.
This mismatch could lead a caller to set `n_restarts = 11` when they want 10 perturbed restarts.

**Fix:** Reword to make the total count explicit:

```rust
/// Number of restarts (default: `10`).
///
/// Total runs = `n_restarts` (clamped to 1 if set to 0). Restart 0 always
/// uses the unperturbed pointwise mean. Restarts 1 through `n_restarts - 1`
/// use Gaussian-perturbed initializations. The run with the lowest total
/// soft-DTW cost is returned.
```

---

_Reviewed: 2026-09-13_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
