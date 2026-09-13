---
phase: 104-quality-evidence-escapes-local-minimum-f64-parity
reviewed: 2026-09-13T00:00:00Z
depth: deep
files_reviewed: 1
files_reviewed_list:
  - fdars-core/src/metric/soft_dtw.rs
findings:
  critical: 1
  warning: 2
  info: 1
  total: 4
status: resolved
resolution_note: CR-01/WR-01/WR-02 fixed in commit b5e57bff; IN-01 deferred (trivial duplication, intentional module isolation).
---

# Phase 104: Code Review Report

**Reviewed:** 2026-09-13
**Depth:** deep (evidence integrity focus — test-only phase)
**Files Reviewed:** 1
**Status:** issues_found

## Summary

Reviewed the new `#[cfg(test)] mod barycenter_escape_evidence_tests` block added in commit
f03ee900, which introduces two tests: `escape_from_local_minimum` and `legacy_bounded_convex_hull`.

The comparison in `escape_from_local_minimum` uses the real `soft_dtw_barycenter` (confirmed to
delegate to `soft_dtw_barycenter_multirestart` with `n_restarts=1`) and the real
`soft_dtw_barycenter_with_config` (confirmed to delegate with the same `gamma`, `max_iter`, and
`tol`). The baseline is not artificially crippled. The RNG is deterministic (fixed `seed: 7`,
`StdRng::seed_from_u64`, no `thread_rng`). The barycenter optimization loop is purely sequential
(no rayon), so `RAYON_NUM_THREADS` cannot affect the result.

One BLOCKER: the convex hull assertion in `legacy_bounded_convex_hull` tests a property that the
soft-DTW optimizer does not guarantee and cannot guarantee in general. Two WARNINGs: the compute
budget asymmetry between the two compared calls is undisclosed in the evidence claim, and the
asserted margin (0.1%) is weaker than the documented claim (>1%). One INFO: minor doc inconsistency
in the margin threshold.

---

## Critical Issues

### CR-01: `legacy_bounded_convex_hull` asserts a property the soft-DTW optimizer does not guarantee

**File:** `fdars-core/src/metric/soft_dtw.rs:2105-2123`

**Issue:** The test asserts that every coordinate `j` of the returned barycenter lies within
`[col_min[j] - eps, col_max[j] + eps]` — the per-column pointwise convex hull of the input rows.
This property is **not guaranteed by soft-DTW barycenters**.

Soft-DTW optimizes an elastic alignment objective via unconstrained L-BFGS. The optimizer has no
boundary constraint, and the gradient of the soft-DTW cost can point the barycenter outside the
input data range, particularly when the elastic warping assigns a time-shifted contribution. The
convex hull containment property holds for pointwise (Fréchet) means under a squared Euclidean
cost, but NOT for DTW or soft-DTW barycenters.

The test passes on this specific sinusoidal dataset with `gamma=1.0` (a large gamma drives the
result toward the pointwise mean, which is within the hull by construction), but the epsilon
`1e-6 * range` provides essentially no slack. The property does not hold in general, and passing
here is a dataset/gamma coincidence, not a correctness proof.

This makes the evidence claim ("proving no divergence in the f64 path") misleading: it proves
only that _this specific configuration on sinusoidal data with gamma=1.0_ does not diverge, not
that the optimizer is bounded. A future caller who uses a different dataset or a lower gamma can
legitimately get a barycenter outside the pointwise convex hull without any bug being present.

**Fix:** Replace the convex hull assertion with a divergence check that the optimizer actually
guarantees — specifically, finiteness (assertion 2) plus a cost monotonicity check (the
barycenter's total soft-DTW cost must not exceed the cost of the pointwise mean initialization).
The cost-does-not-exceed-init assertion follows from Armijo backtracking and is a true guarantee
of the L-BFGS implementation. Alternatively, document explicitly that the convex hull containment
is an empirical observation on this dataset, not a logical guarantee, and relabel the assertion
accordingly:

```rust
// Assertion 3 (empirical, dataset-specific): on this sinusoidal dataset with gamma=1.0
// the result stays within the pointwise convex hull. This is NOT a general guarantee.
// It holds here because large gamma drives the result close to the pointwise mean.
// Do NOT generalize this assertion to other datasets or smaller gamma values.
```

Or replace with a cost-does-not-regress check:

```rust
// Compute the cost of the initialization (pointwise mean) for comparison.
let mean_init: Vec<f64> = (0..m)
    .map(|j| rows.iter().map(|r| r[j]).sum::<f64>() / rows.len() as f64)
    .collect();
let init_cost: f64 = rows.iter().map(|r| soft_dtw_distance(&mean_init, r, gamma)).sum();
let result_cost: f64 = rows.iter().map(|r| soft_dtw_distance(&result.barycenter[..], r, gamma)).sum();
assert!(
    result_cost <= init_cost * (1.0 + 1e-9),
    "barycenter cost {result_cost:.6} exceeds initialization cost {init_cost:.6} \
     — L-BFGS descent property violated"
);
```

---

## Warnings

### WR-01: `escape_from_local_minimum` compares unequal total compute budgets without disclosure

**File:** `fdars-core/src/metric/soft_dtw.rs:2007-2040`

**Issue:** The single-init baseline runs `1 restart × 300 iterations`. The multi-restart call
runs `16 restarts × 300 iterations` = 4800 L-BFGS iterations total. The test comment says
"multi-restart escapes the local minimum that stalls the single-init baseline," but the
comparison conflates restart diversity with raw compute budget — a single-init optimizer given
16× the iteration budget (e.g., `max_iter=4800`) might also escape the local minimum via the
Armijo line-search taking large steps.

This is not a code bug — the multi-restart implementation is the production feature being tested
— but it is an evidence integrity gap: the test does not demonstrate that it is the _qualitative
property_ of multi-restart (diverse initializations) rather than the _quantitative property_ of
16× budget that achieves the cost reduction. The SDTW-04 requirement should be verified against a
single-init baseline with the same 16× iteration budget, or the evidence description must
acknowledge the budget asymmetry explicitly.

**Fix:** Either:
1. Add a second baseline comparison: `soft_dtw_barycenter(&data, gamma, 16 * max_iter, tol)`
   and verify multi-restart beats it too (proves qualitative benefit of diverse inits, not just
   budget), OR
2. Add a comment acknowledging: "Note: the multi-restart call uses 16× more total compute than
   the single-init baseline. The test demonstrates that restart diversity + budget together
   improve over single-init + equal per-restart budget; it does not isolate the contribution of
   restart diversity alone."

### WR-02: Doc comment claims ">1% improvement" but assertion enforces only 0.1%

**File:** `fdars-core/src/metric/soft_dtw.rs:1965` (doc) vs `2033` (code)

**Issue:** The function-level doc comment at line 1965 states:

```
/// Tuned parameters verified to achieve a clear gap (> 1% improvement):
```

The actual assertion at line 2033 uses `required_margin = 1e-3 * single_init_total.abs().max(1.0)`,
which is a **0.1% relative margin**. These are inconsistent by a factor of 10. If the actual
observed improvement is between 0.1% and 1%, the test passes but the documentation's "clear gap
(> 1%)" claim is false. A reviewer reading the doc comment would expect a 10× stronger guarantee
than the code actually enforces.

**Fix:** Align the doc comment with the code:

```rust
/// Tuned parameters verified to achieve a clear gap (> 0.1% improvement):
///   m = 50 (series length for resolution), σ = 0.04 (tight clusters),
///   cluster centers at 0.12 and 0.88, γ = 0.01 (near hard-DTW),
///   n_restarts = 16, seed = 7, max_iter = 300, tol = 1e-9.
```

Or tighten the assertion to 1%:

```rust
let required_margin = 1e-2 * single_init_total.abs().max(1.0);
```

---

## Info

### IN-01: `mat_from_rows` helper duplicated across two test modules

**File:** `fdars-core/src/metric/soft_dtw.rs:1914-1924` (new module) vs `barycenter_config_tests`

**Issue:** The `mat_from_rows` function is defined identically in both
`barycenter_config_tests` and the new `barycenter_escape_evidence_tests` module.
This is test-only duplication — not a production code issue — but consolidating
to a shared helper (e.g., a private module `#[cfg(test)] mod test_helpers`) would
reduce maintenance surface.

**Fix:** Extract into a shared `#[cfg(test)]` helper module within `soft_dtw.rs`, or
mark with a comment that the duplication is intentional given module isolation.

---

_Reviewed: 2026-09-13_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep — evidence integrity focus_
