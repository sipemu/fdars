---
phase: 15-elastic-fpca-performance
reviewed: 2026-08-12T00:00:00Z
depth: deep
files_reviewed: 1
files_reviewed_list:
  - fdars-core/src/elastic_fpca.rs
findings:
  critical: 0
  warning: 0
  info: 1
  total: 1
status: clean
---

# Phase 15: Code Review Report

**Reviewed:** 2026-08-12
**Depth:** deep (single-file parallelization refactor, cross-referenced macro + reference pattern)
**Files Reviewed:** 1 (`fdars-core/src/elastic_fpca.rs`)
**Status:** clean

## Summary

Phase 15 is a pure parallelization refactor of three per-curve loops in `elastic_fpca.rs`
(`shooting_vectors_from_psis`, `build_augmented_srsfs`, and the score-fill in
`svd_scores_and_eigenvalues`), each now dispatched via `iter_maybe_parallel!`. The review
verified data-race safety, column-major indexing correctness, threshold-guard correctness,
and numerical equivalence.

**No correctness, safety, or quality defects were found.** The refactor is sound:

- **Data-race safety (confirmed).** All three loops use the collect-then-assign pattern.
  The parallel `map` produces OWNED per-index values (`Vec<f64>` for the row loops, `f64`
  for the score column), collected into a `Vec`, then a SEQUENTIAL pass assigns into the
  shared `FdMatrix`. There is no parallel write into any shared `FdMatrix`/buffer, no
  captured `&mut`, and no `unsafe`. The parallel closures read only through shared `&`
  references (`inv_exp_map_sphere(mu_psi, &psis[i], time)`, `qn[(i,j)]`,
  `aligned_data[(i,id)]`, `u[(i,k)]`). This mirrors the established
  `alignment/set.rs::align_to_target` idiom (verified: `set.rs:59` uses the same
  `iter_maybe_parallel!(0..n).map(...).collect()` shape).

- **Column-major indexing / order preservation (confirmed).** `iter_maybe_parallel!(0..n)`
  expands to `into_par_iter()` on a `Range<usize>` (`parallel.rs:48`), which is an
  `IndexedParallelIterator`; rayon guarantees `collect()` into a `Vec` preserves index
  order. Thus `rows.into_iter().enumerate()` yields `(i, value_for_i)`, and the assignments
  `shooting[(i,j)] = v[j]`, `q_aug[(i,j)] = row[j]`, and `scores[(i,k)] = val` map to the
  correct `(row, col)`. All indexing goes through the `FdMatrix[(i,j)]` operator (which
  encapsulates the `i + j*nrows` arithmetic); no manual index math and no transposition was
  introduced. `build_augmented_srsfs` correctly preserves the augmented column: `row[m]` /
  `q_aug[(i,m)]` carries `sign(f_id)*sqrt(|f_id|)` exactly as the original sequential code.

- **`:796` threshold guard (confirmed).** `SCORES_PARALLEL_THRESHOLD = 50`. The
  `if n >= SCORES_PARALLEL_THRESHOLD` branches ONCE (outer-if), not per-component. Both
  arms compute the identical body `scores[(i,k)] = u[(i,k)] * sv`. The parallel arm
  collects a per-component `col: Vec<f64>` then assigns in order (order-preserving as
  above). Boundary is correct (n=50 → parallel, n<50 → sequential); the `test_scores_threshold`
  test exercises both n=10 and n=51 and asserts bit-exact equality to `u[(i,k)]*sv`.

- **Equivalence preservation (confirmed).** All three loops are pure disjoint per-index
  writes with NO cross-iteration reduction or accumulation, so each output element is the
  result of exactly one independent expression — floating-point output is identical
  regardless of execution order. The equivalence tests use `assert_eq!` on `f64` (bit-exact,
  not approximate) and PASS under both `--features parallel` (default) and
  `--no-default-features` (sequential). Verified locally: 20/20 elastic_fpca tests pass in
  both configurations.

- **Non-breaking (confirmed).** No signature changes (`shooting_vectors_from_psis` and
  `build_augmented_srsfs` remain `pub(crate)`; `svd_scores_and_eigenvalues` remains
  private). No new dependencies (rayon already gated by the pre-existing `parallel`
  feature). `iter_maybe_parallel!` is imported at line 13; the `ParallelIterator` trait
  import (line 18) is correctly `#[cfg(feature = "parallel")]`-gated (required for
  `.map()/.collect()` on the parallel iterator, unused in the sequential build). Compiles
  and behaves correctly with `parallel` off.

- **Convention (confirmed).** The named constant `SCORES_PARALLEL_THRESHOLD` is documented
  with a rationale referencing the PERF-04-C payback analysis. Clippy is clean under
  `--all-targets` for this file.

## Info

### IN-01: Two loops are unconditionally parallelized while the third is threshold-guarded

**File:** `fdars-core/src/elastic_fpca.rs:715` and `:738` (vs. `:796`)
**Issue:** `shooting_vectors_from_psis` and `build_augmented_srsfs` parallelize
unconditionally (no N-threshold), whereas `svd_scores_and_eigenvalues` added an `N >= 50`
guard. This asymmetry is intentional and correct — the per-item work in the first two
(`inv_exp_map_sphere` on the Hilbert sphere; an m-length SRSF row copy) is far heavier than
the single multiply in the score fill, so dispatch overhead pays back at much smaller N —
and the code comments document the distinction. Recorded only for future maintainer
awareness, not a defect. No change required.
**Fix:** None needed. If a future audit finds the row loops also underperform at tiny N,
the same `SCORES_PARALLEL_THRESHOLD`-style guard could be applied; the current form is
correct as-is.

---

_Reviewed: 2026-08-12_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
