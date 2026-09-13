---
phase: 102-deterministic-multi-restart-wrapper
reviewed: 2026-09-13T00:00:00Z
depth: deep
files_reviewed: 1
files_reviewed_list:
  - fdars-core/src/metric/soft_dtw.rs
findings:
  critical: 0
  warning: 3
  info: 2
  total: 5
status: resolved
---

# Phase 102: Code Review Report

**Reviewed:** 2026-09-13
**Depth:** deep
**Files Reviewed:** 1 (`fdars-core/src/metric/soft_dtw.rs`, diff `f70e1230..HEAD`)
**Status:** issues_found

## Summary

Phase 102 extracts `lbfgs_refine_from_init` from the old monolithic `soft_dtw_barycenter`, adds `per_col_std`, `perturbed_init`, and `soft_dtw_barycenter_multirestart`, and rewires the public 4-arg entry to delegate. The refactor is structurally sound: no `HashMap` iteration, no `thread_rng`, no float-sum-order nondeterminism. The perturbation RNG is sequential (`Iterator::zip` + `.map()`) and seeded per project convention. The winner-selection comparison is strict-less-than, preserving restart-0 priority. The `lbfgs_refine_from_init` extraction is a mechanical lift with byte-identical output; objective and gradient code are untouched.

Three correctness/safety defects are flagged, all in the new private driver and helpers. None break the current test suite (which uses small, well-formed inputs), but two become BLOCKER-level when Phase 103 exposes `soft_dtw_barycenter_multirestart` publicly.

---

## Warnings

### WR-01: `best_result.unwrap()` panics when all restart costs are NaN

**File:** `fdars-core/src/metric/soft_dtw.rs:797`

**Issue:** `best_result` starts as `None`. The loop only calls `best_result = Some(result)` when `total < best_total`. Because `f64::NAN < f64::INFINITY` evaluates to `false`, if any restart's total cost is NaN the replacement never fires and `best_result` stays `None`. If *all* restarts produce NaN cost — possible when the input `FdMatrix` contains NaN values — the final `.unwrap()` panics. The SAFETY comment at line 796 only argues `n_runs >= 1`, which is correct but insufficient: it silently ignores the NaN-propagation path.

`FdMatrix` does not validate NaN at construction, so a caller can reach this function with NaN data. The gamma guard at line 759 screens only gamma, not the data. The function is currently private and Phase 103 will make it public-facing.

**Fix:**

```rust
// Replace the bare unwrap with a NaN-safe fallback:
best_result.unwrap_or_else(|| {
    // All restarts produced NaN cost (NaN input data). Return
    // the unperturbed mean as the least-bad fallback.
    SoftDtwBarycenterResult {
        barycenter: mean,
        n_iter: 0,
        converged: false,
    }
})
```

Alternatively, add a data NaN check before the loop and return early with `ComputationFailed` (preferred once Phase 103 switches to `Result<T, FdarError>`).

---

### WR-02: Integer overflow in `seed + k` inside `perturbed_init`

**File:** `fdars-core/src/metric/soft_dtw.rs:717`

**Issue:** `StdRng::seed_from_u64(seed + k)` performs unchecked `u64` addition. In Rust debug builds this panics on overflow; in release builds it silently wraps to zero. If a caller passes `seed = u64::MAX` and `n_restarts >= 2`, restart k=1 computes `u64::MAX + 1 = 0` (release) or panics (debug). Wrapping means restart k=1 and any later restart `k` where `seed + k ≡ seed_other + k_other (mod 2^64)` receive the same RNG state, producing identical perturbations — breaking the uniqueness and "lower cost is strictly better" invariants the driver relies on.

The function is private and current tests use `seed = 42` or `seed = 99`, so no test exercises this path. It becomes user-visible when Phase 103 exposes a `seed` parameter.

**Fix:**

```rust
let mut rng = StdRng::seed_from_u64(seed.wrapping_add(k));
```

Use `wrapping_add` consistently so behavior is identical in debug and release, and document the wrap-around property in the seed parameter's doc comment.

---

### WR-03: `soft_dtw_barycenter` recomputes `init_barycenter_mean` twice on the hot path

**File:** `fdars-core/src/metric/soft_dtw.rs:847-856`

**Issue:** The restructured `soft_dtw_barycenter` computes `init_barycenter_mean(&rows)` in two separate branches: once (line 847) inside the `!gamma.is_finite() || gamma <= 0.0` early-return path, and once (line 856) as the `init` argument to `lbfgs_refine_from_init`. On the normal (valid-gamma) path, which is the only path users hit in production, the mean is computed once and immediately consumed by `lbfgs_refine_from_init`. This is not a bug in isolation, but the structure obscures that the fallback branch has already consumed one heap allocation — and the valid-gamma path now allocates the mean only to pass it to the helper (no longer holding it for reuse as in Phase 101). The code is correct but brittle: if future code between the gamma check and the delegate call ever needs the mean, it will recompute it a second time without noticing.

This was a deliberate consequence of the refactor, but the old code was cleaner here — it held a single `bary` binding that served both the fallback and the optimizer. Consider:

**Fix:**

```rust
let mean = init_barycenter_mean(&rows);
if !gamma.is_finite() || gamma <= 0.0 {
    return SoftDtwBarycenterResult { barycenter: mean, n_iter: 0, converged: false };
}
lbfgs_refine_from_init(&rows, mean, gamma, max_iter, tol)
```

This is one allocation on all paths and eliminates the redundancy.

---

## Info

### IN-01: `soft_dtw_barycenter_multirestart` missing `#[must_use]`

**File:** `fdars-core/src/metric/soft_dtw.rs:739`

**Issue:** Project convention (`.claude/CLAUDE.md`) mandates `#[must_use]` on all expensive computations (74+ functions marked). `soft_dtw_barycenter_multirestart` runs up to `n_restarts` full L-BFGS descents — it is at least as expensive as `soft_dtw_barycenter`, which is already marked `#[must_use]` (line 822). The attribute is currently suppressed by `#[allow(dead_code)]` (the function is private), but it should be added now so it is in place when Phase 103 makes it public. Omitting it now means callers can accidentally discard the result without a compiler warning.

**Fix:**

```rust
#[must_use]
#[allow(dead_code)]
fn soft_dtw_barycenter_multirestart(...) -> SoftDtwBarycenterResult {
```

---

### IN-02: `SoftDtwBarycenterResult` missing `PartialEq` derive (pre-existing, worsened by Phase 102)

**File:** `fdars-core/src/metric/soft_dtw.rs:20`

**Issue:** Project convention requires all public result types to derive `#[derive(Debug, Clone, PartialEq)]` and carry a `#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]`. `SoftDtwBarycenterResult` derives only `Debug, Clone` — both `PartialEq` and the serde gate are missing. This was introduced in Phase 101 and not corrected here. Phase 102 adds four tests that compare barycenters element-by-element using `.to_bits()` rather than the idiomatic `assert_eq!(result1, result2)` — a workaround symptom of the missing `PartialEq`. The tests work around it correctly (bit-level comparison is appropriate for determinism tests), but the struct should still conform to crate conventions before Phase 103 publishes it in the API.

**Fix:**

```rust
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct SoftDtwBarycenterResult {
```

---

## Determinism Audit (all clear)

- No `HashMap`, `HashSet`, or `BTreeMap` iteration in new code.
- No `thread_rng` or `OsRng`; all randomness routes through `StdRng::seed_from_u64(seed + k)`.
- `perturbed_init` draws noise via `Iterator::zip(...).map(...)` — sequential left-to-right; order is fixed by the `mean`/`col_std` slice order.
- `soft_dtw_barycenter_multirestart` restart loop is sequential (`for k in 0..n_runs`), not parallel.
- Float sum order in `total` cost (`rows.iter().map(...).sum()`) is deterministic because it uses a sequential iterator — no rayon dispatch in this path.
- `seed + k` arithmetic is the only determinism risk (WR-02 above): wrapping in release vs. panic in debug on extreme seeds.

## Refactor Correctness (all clear)

- `lbfgs_refine_from_init` is a verbatim extraction of the inner body of the old `soft_dtw_barycenter`. The three `&rows` → `rows` reference adjustments in the diff are trivially correct (ownership transfer from local to parameter). All L-BFGS hyperparameters, safeguards (curvature guard, non-descent fallback, Armijo), and early-exit logic are preserved byte-for-byte.
- `soft_dtw_barycenter` delegates via `lbfgs_refine_from_init(&rows, init_barycenter_mean(&rows), gamma, max_iter, tol)`. The init is the same `init_barycenter_mean` the old body computed. The public signature is unchanged. `#[must_use]` is retained.
- `SoftDtwBarycenterResult` fields and `#[non_exhaustive]` attribute are unchanged — no API break.

---

## Fix Disposition (2026-09-13, commit ecdad836)

- **WR-02** (seed overflow): Fixed — `seed + k` → `seed.wrapping_add(k)` in `perturbed_init`.
- **WR-01** (all-NaN unwrap panic): Fixed — `best_result.unwrap()` → `unwrap_or(SoftDtwBarycenterResult { barycenter: mean, n_iter: 0, converged: false })`; SAFETY comment updated.
- **WR-03** (double `init_barycenter_mean`): Fixed — single `let mean = init_barycenter_mean(&rows)` binding in `soft_dtw_barycenter` reused for fallback and L-BFGS init.
- **IN-01** (`#[must_use]`): Fixed — attribute added to `soft_dtw_barycenter_multirestart`; `#[allow(dead_code)]` retained while function is private.
- **IN-02** (`SoftDtwBarycenterResult` missing `PartialEq`/serde derives): **DEFERRED to Phase 103** — pre-existing from Phase 101; result-type API derives and serde re-exports belong to the Phase 103 (SDTW-03) API phase.

Gates run in main checkout: `cargo fmt` clean, `cargo clippy --all-targets --features linalg,parallel -D warnings` clean, 28/28 `soft_dtw` unit + integration tests pass, `cargo check --target wasm32-unknown-unknown` clean.

---

_Reviewed: 2026-09-13_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
