---
phase: 65-greedy-selection-integration
reviewed: 2026-09-03T22:00:00Z
depth: deep
files_reviewed: 5
files_reviewed_list:
  - fdars-core/src/optimal_design.rs
  - fdars-core/src/lib.rs
  - fdars-core/src/prelude.rs
  - fdars-core/benches/optimal_design.rs
  - fdars-core/Cargo.toml
findings:
  critical: 1
  warning: 1
  info: 2
  total: 4
status: issues_found
---

# Phase 65: Code Review Report

**Reviewed:** 2026-09-03T22:00:00Z
**Depth:** deep
**Files Reviewed:** 5
**Status:** issues_found

## Summary

Phase 65 adds the greedy `optimal_design` wrapper, `OptDesConfig`/`OptDesResult` types, additive
crate-root and prelude re-exports, a module-level doctest, and a Criterion benchmark. The
determinism story is sound: `iter_maybe_parallel!(remaining)` consumes a `Vec<usize>`, whose
`into_par_iter()` implements `IndexedParallelIterator`, so the parallel `collect` into `Vec` is
order-preserving. The subsequent sequential fold-based argmin is correct and no rayon `min_by` is
used. The `#[must_use = "..."]` with a string message avoids the `double_must_use` clippy lint.
Serde-gating, `#[non_exhaustive]` on `OptDesResult` only, derives, and re-export placement all
follow established codebase conventions.

One blocker was found: a `panic` reachable through a publicly documented code path when
`candidate_grid` contains duplicate values (or near-duplicate values within the 1e-9 FP tolerance)
that map to the same `argvals` index. The input validation counts `candidate_grid.len()` to bound
`budget`, but the uniqueness of the mapped argvals indices is never checked; a caller supplying
`candidate_grid = [0.0, 0.0]` with `budget = 2` passes all validation guards yet reaches an
`.expect()` panic when `remaining` is exhausted at the second greedy step.

## Critical Issues

### CR-01: Panic on duplicate `candidate_grid` values with `budget == candidate_grid.len()`

**File:** `fdars-core/src/optimal_design.rs:385`

**Issue:** The `.expect()` on the greedy argmin fold states its non-emptiness invariant is
"guaranteed by `budget <= candidate count`", but `candidate count` is `candidate_grid.len()` (the
raw slice length, counted in the validation guard at line 317), whereas the actual pool of
selectable argvals indices is `unique(candidate_indices)`. When `candidate_grid` contains duplicate
values — or two values both within `1e-9` of the same `argvals` point — `map_candidates_to_indices`
produces a `candidate_indices` vec with repeated usize values. After those indices have been
selected, the `.filter(|idx| !selected.contains(idx))` gate drains `remaining` to empty while
`_step` still has iterations remaining, and the fold returns `None`, causing `.expect()` to panic.

**Concrete trigger:**
```rust
let model = /* PaceFpcaResult with argvals = [0.0, 0.02, ..., 1.0] */;
let config = OptDesConfig {
    candidate_grid: vec![0.0, 0.0, 0.5],  // duplicate 0.0 maps to argvals index 0 twice
    budget: 3,                              // 3 <= candidate_grid.len() = 3 — passes validation
    criterion: DesignCriterion::Trajectory,
};
let _ = optimal_design(&model, &config); // PANICS at step 3: remaining is empty
```

Step-by-step:
- `candidate_indices = [0, 0, 25]` (two copies of index 0).
- Step 1: `remaining = [0, 0, 25]`, select index 0, `selected = [0]`.
- Step 2: `remaining = [25]`, select index 25, `selected = [0, 25]`.
- Step 3: `remaining = []` (both 0 entries and 25 are all filtered). Fold returns `None`.
  `.expect()` panics.

The field documentation for `candidate_grid` says values must be present in `model.argvals`, but
does not require them to be distinct, so a caller has no warning that duplicates are forbidden.

**Fix — Option A (preferred): add uniqueness validation before the loop.**
After `map_candidates_to_indices` succeeds, check that the returned indices are distinct:

```rust
// After line 341:
let candidate_indices = map_candidates_to_indices(&config.candidate_grid, &model.argvals)?;

// Guard: the unique argvals-index count must be at least budget.
{
    let mut seen = std::collections::HashSet::with_capacity(candidate_indices.len());
    for &idx in &candidate_indices {
        seen.insert(idx);
    }
    if seen.len() < config.budget {
        return Err(FdarError::InvalidParameter {
            parameter: "config.candidate_grid",
            message: format!(
                "candidate_grid contains duplicate argvals indices; only {} unique grid \
                 point(s) found but budget is {}",
                seen.len(),
                config.budget
            ),
        });
    }
}
```

Also update the `candidate_grid` field doc to say "must be distinct (no two values may map to
the same `model.argvals` index within the `1e-9` tolerance)".

**Fix — Option B (lighter): replace `.expect()` with a checked `ok_or_else` and propagate the
error:**

```rust
let (best_idx, best_val) = scores
    .into_iter()
    .fold(None::<(usize, f64)>, |acc, (idx, val)| {
        Some(match acc {
            None => (idx, val),
            Some((bi, bv)) => if val < bv { (idx, val) } else { (bi, bv) },
        })
    })
    .ok_or_else(|| FdarError::InvalidParameter {
        parameter: "config.candidate_grid",
        message: "candidate_grid contains duplicate argvals indices; \
                  remaining candidates exhausted before budget was reached".into(),
    })?;
```

Option B converts the panic to a recoverable error without an upfront `HashSet` scan; Option A
gives a better diagnostic at validation time before any greedy work starts.

## Warnings

### WR-01: Tie-break documentation says "smallest-index" but means "first-in-`candidate_grid`-order"

**File:** `fdars-core/src/optimal_design.rs:288,369`

**Issue:** Two doc comments assert a "smallest-index tie-break":

- Line 288 (function doc): "smallest-index tie-break (never rayon `min_by`, which is not stable
  under ties)"
- Line 369 (inline comment): "keeps the FIRST minimum → smallest-index tie-break"

The actual tie-break rule is *first position in `candidate_grid` order*, which is the same as
smallest argvals index only when `candidate_grid` is sorted by ascending argvals index. If a caller
provides `candidate_grid` in a different order (e.g. `[0.5, 0.0, 1.0]`), ties are broken by
`candidate_grid` position (0.5 wins over 0.0 on a tie) — the opposite of what "smallest-index"
implies. The sequential-fold determinism guarantee is correctly stated; only the described
tie-break rule is wrong.

This is a semantic specification error: a caller who needs smallest-argvals-index tie-breaking for
reproducibility across different `candidate_grid` orderings would get incorrect results.

**Fix:** Update both doc comments to accurately describe the rule:

```rust
// was: "smallest-index tie-break"
// fix:
/// … with a *first-in-`candidate_grid`-order* tie-break (equivalent to smallest argvals
/// index when `candidate_grid` is in ascending argvals-index order, which is the typical
/// and recommended usage). Never uses rayon `min_by`, which is not stable under ties.
```

And at line 369:
```rust
// keeps the FIRST minimum → first-in-candidate_grid-order tie-break (rayon `min_by` is NOT
// stable under ties, so it must not be used here).
```

## Info

### IN-01: `optimal_design` doc's `# Errors` section omits the `m < 2` propagated error

**File:** `fdars-core/src/optimal_design.rs:299–304`

**Issue:** The `# Errors` section for `optimal_design` lists five `InvalidParameter` cases
(`budget == 0`, `budget > grid`, off-grid candidate, `ncomp == 0`, `sigma2 <= 0`) but omits that
`design_criterion` also returns `InvalidParameter` when `model.argvals.len() < 2`. This error is
correctly propagated via `?` at line 366 but is not mentioned in the public API contract, so a
caller with a 1-point model would see an undocumented error.

**Fix:** Add to the `# Errors` section:

```text
/// Returns [`FdarError::InvalidParameter`] if `model.argvals.len() < 2` (a trajectory
/// integral requires at least two evaluation points for Simpson quadrature), propagated
/// from [`design_criterion`].
```

### IN-02: D-optimality not covered by the benchmark

**File:** `fdars-core/benches/optimal_design.rs`

**Issue:** The benchmark covers `DesignCriterion::Trajectory` and `Score(OptimalityKind::A)` but
not `Score(OptimalityKind::D)`. D-optimality takes a different code path
(`factor_posterior_cov_with_retry` + `log_det_from_cholesky`) and is exercised by unit tests but
not by a benchmark, so its performance cannot be tracked over time.

**Fix (optional):** Add two benchmark cases:

```rust
let cfg_score_d = OptDesConfig {
    candidate_grid: candidate_grid.clone(),
    budget: 5,
    criterion: DesignCriterion::Score(OptimalityKind::D),
};
group.bench_function("design_criterion_score_d_p5_m51", |b| {
    b.iter(|| {
        design_criterion(
            black_box(&model),
            black_box(&fixed_design),
            black_box(DesignCriterion::Score(OptimalityKind::D)),
        )
    });
});
group.bench_function("optimal_design_score_d_budget5_m51", |b| {
    b.iter(|| optimal_design(black_box(&model), black_box(&cfg_score_d)));
});
```

---

_Reviewed: 2026-09-03T22:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
