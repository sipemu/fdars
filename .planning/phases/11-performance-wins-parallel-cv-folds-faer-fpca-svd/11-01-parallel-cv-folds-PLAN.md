---
phase: 11-performance-wins-parallel-cv-folds-faer-fpca-svd
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - fdars-core/src/classification/cv.rs
autonomous: true
requirements: [PERF-01]

estimate:
  tokens: 42000
  raw_tokens: 28000
  tasks: 2
  confidence: high

must_haves:
  truths:
    - "PERF-01: the `for fold in 0..nfold` loop in `fclassif_cv` (classification/cv.rs) is replaced by `iter_maybe_parallel!(0..nfold).map(...).collect::<Vec<f64>>()`."
    - "PERF-01: parallel CV `fold_errors` is bit-for-bit identical to sequential for a fixed seed — the equivalence test asserts element-wise `Vec<f64>` equality AND aggregate `error_rate` equality."
    - "PERF-01: the change compiles and passes with the `parallel` feature ON (`cargo test -p fdars-core --features parallel`) and OFF (`cargo test -p fdars-core`)."
    - "Both / build gate: `cargo clippy -p fdars-core --features parallel -- -D warnings` passes."
  artifacts:
    - "fdars-core/src/classification/cv.rs — fold loop parallelized; new inline `#[cfg(test)] mod tests` block with the equivalence test."
  key_links:
    - "iter_maybe_parallel! macro (parallel.rs) → cv.rs fold loop: requires `use crate::iter_maybe_parallel;` plus `#[cfg(feature = \"parallel\")] use rayon::iter::ParallelIterator;` in scope for `.collect()`."
    - "rayon range par_iter `.collect()` preserves index order → `fold_errors[i]` maps to fold `i` deterministically."
  prohibitions:
    - "MUST NOT add any new external dependency to Cargo.toml (rayon is already present via the `parallel` feature)."
    - "MUST NOT alter the non-parallel (default-feature) code path's observable output — sequential execution must yield the same `fold_errors` and `error_rate` as before."
    - "MUST NOT import `rayon` directly for the iteration itself — parallelism goes through the `iter_maybe_parallel!` macro (the `ParallelIterator` trait import is the only permitted direct rayon reference, needed for `.collect()`)."
    - "MUST NOT introduce per-fold RNG or any shared mutable accumulator inside the fold closure."
---

<objective>
Parallelize the cross-validation fold loop in `fclassif_cv` (PERF-01) so it runs across cores under the `parallel` feature while producing results bit-for-bit identical to sequential execution.

Purpose: The folds are fully independent (fold-assignment RNG runs once before the loop; each fold is a pure function of immutable shared state producing one `f64` error rate). This is a textbook `for`-loop → `iter_maybe_parallel!(0..nfold).map(...).collect()` conversion with a projected ~4–5× speedup on multi-core, at zero behavioral cost.

Output: `fdars-core/src/classification/cv.rs` with the fold loop parallelized and a new inline `#[cfg(test)] mod tests` block containing a sequential-vs-parallel equivalence test.
</objective>

<execution_context>
@$HOME/.claude/gsd-core/workflows/execute-plan.md
@$HOME/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/11-performance-wins-parallel-cv-folds-faer-fpca-svd/11-RESEARCH.md
</context>

<artifacts_produced>
New symbols this plan introduces (source-grounding drift verification must EXCLUDE these as newly-created):
- Inline `#[cfg(test)] mod tests` block in `fdars-core/src/classification/cv.rs` (none exists today).
- Test fn `test_fclassif_cv_parallel_matches_sequential` (equivalence test).
- New use-imports at top of `cv.rs`: `use crate::iter_maybe_parallel;` and `#[cfg(feature = "parallel")] use rayon::iter::ParallelIterator;`.

Reused (NOT new): `iter_maybe_parallel!` macro (parallel.rs), `fclassif_cv`, `ClassifCvResult` and its public `fold_errors: Vec<f64>` / `error_rate: f64` fields, `assign_folds`, `fold_split`, `extract_class_data`, `cv_fold_predict`.
</artifacts_produced>

<tasks>

<task type="tracer" tdd="true">
  <name>Task 1: Parallelize the fclassif_cv fold loop via iter_maybe_parallel!</name>
  <files>fdars-core/src/classification/cv.rs</files>
  <read_first>
    - fdars-core/src/classification/cv.rs — the file being modified; read the top `use` block (lines 1–10) and the `fclassif_cv` fold loop (the `let mut fold_errors = Vec::with_capacity(nfold); for fold in 0..nfold { ... fold_errors.push(errors); }` region, ~lines 74–117) to see current state.
    - fdars-core/src/parallel.rs — the `iter_maybe_parallel!` macro definition (~lines 42–54): under `parallel` it expands to `IntoParallelIterator::into_par_iter($expr)`, otherwise `IntoIterator::into_iter($expr)`.
    - fdars-core/src/alignment/karcher.rs — the canonical usage pattern: `use crate::iter_maybe_parallel;` at top, `#[cfg(feature = "parallel")] use rayon::iter::ParallelIterator;`, and `let results: Vec<_> = iter_maybe_parallel!(0..n).map(...).collect();`.
    - .planning/phases/11-performance-wins-parallel-cv-folds-faer-fpca-svd/11-RESEARCH.md — the "PERF-01" section for the exact before/after transformation and Pitfalls 1 and 5.
  </read_first>
  <behavior>
    - The parallelized loop returns a `Vec<f64>` of per-fold error rates in fold-index order (fold 0 → index 0, ...).
    - `error_rate` is `fold_errors.iter().sum::<f64>() / nfold as f64`, unchanged.
    - Under default (no `parallel`) features the code path is a sequential iterator with identical output.
    - Under `parallel` the folds execute concurrently but collect in original index order.
  </behavior>
  <action>
    Add `use crate::iter_maybe_parallel;` and `#[cfg(feature = "parallel")] use rayon::iter::ParallelIterator;` to the top `use` block of cv.rs (the second import is required so `.collect()` resolves to the `ParallelIterator::collect` method under the `parallel` feature — see Pitfall 1). Replace the `let mut fold_errors = Vec::with_capacity(nfold);` declaration and the entire `for fold in 0..nfold { ... }` loop with a single `let fold_errors: Vec<f64> = iter_maybe_parallel!(0..nfold).map(|fold| { ... }).collect();` expression. Move the existing per-fold body verbatim into the `.map(|fold| { ... })` closure, changing only the trailing `fold_errors.push(errors);` into the closure's final tail expression `errors` (the last expression is the fold's returned value). Do NOT introduce a direct `rayon::iter` call for the iteration; go through the macro. Do NOT declare `fold_errors` as `mut` and do NOT push. Leave `assign_folds`, `fold_split`, `extract_class_data`, `cv_fold_predict`, and the `let error_rate = fold_errors.iter().sum::<f64>() / nfold as f64;` line unchanged. The closure captures only `&`-references (`data`, `labels`, `argvals`, `folds`, `scalar_covariates`) plus `Copy` values (`g`, `method`, `ncomp`) — all `Send + Sync`, so no clone or per-fold RNG is needed (Pitfall 5).
  </action>
  <verify>
    <automated>cargo build -p fdars-core && cargo build -p fdars-core --features parallel && grep -q 'iter_maybe_parallel!(0..nfold)' fdars-core/src/classification/cv.rs && ! grep -q 'fold_errors.push' fdars-core/src/classification/cv.rs</automated>
  </verify>
  <acceptance_criteria>
    - `cargo build -p fdars-core` exits 0 (default features, sequential path compiles).
    - `cargo build -p fdars-core --features parallel` exits 0 (parallel path compiles; confirms `ParallelIterator` import is present).
    - `grep -c 'iter_maybe_parallel!(0..nfold)' fdars-core/src/classification/cv.rs` returns 1.
    - `grep 'fold_errors.push' fdars-core/src/classification/cv.rs` returns nothing (old push pattern removed).
    - `grep -q 'let mut fold_errors' fdars-core/src/classification/cv.rs` is false (no longer `mut`).
    - cv.rs top contains `use crate::iter_maybe_parallel;` and `#[cfg(feature = "parallel")] use rayon::iter::ParallelIterator;`.
  </acceptance_criteria>
  <reversibility rating="reversible">Feature-gated behavior swap through an existing macro; the sequential path is preserved under default features, so reverting is a one-line restore.</reversibility>
  <done>The fold loop is a single `iter_maybe_parallel!(0..nfold).map(...).collect()` expression; both default-feature and `parallel`-feature builds compile; the old `mut` accumulator and `.push` are gone.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Add sequential-vs-parallel equivalence test for fclassif_cv</name>
  <files>fdars-core/src/classification/cv.rs</files>
  <read_first>
    - fdars-core/src/classification/cv.rs — the (now-parallelized) `fclassif_cv` signature `(data: &FdMatrix, argvals: &[f64], y: &[usize], scalar_covariates: Option<&FdMatrix>, method: &str, ncomp: usize, nfold: usize, seed: u64) -> Result<ClassifCvResult, FdarError>`; confirm there is NO existing `#[cfg(test)] mod tests` block (you are creating it).
    - fdars-core/src/classification/mod.rs — `ClassifCvResult` struct: public fields `error_rate: f64`, `fold_errors: Vec<f64>`, `best_ncomp: usize`.
    - fdars-core/src/regression.rs — the `generate_test_fdata(n, m)` test-helper pattern (~line 754) as a reference for constructing deterministic `FdMatrix` test data; replicate a small local equivalent in cv.rs if a shared helper is not importable from the test module.
    - fdars-core/src/matrix.rs — `FdMatrix::from_column_major(data, nrows, ncols)` and `FdMatrix::shape()` for building test data.
    - .planning/phases/11-performance-wins-parallel-cv-folds-faer-fpca-svd/11-RESEARCH.md — the "Determinism Guarantee" and "Wave 0 Gaps" sections for the exact assertion strategy.
  </read_first>
  <behavior>
    - Test builds a small deterministic classification dataset (e.g. n=20, m=10, 2 classes, fixed seed) with argvals as a uniform grid and integer labels.
    - Test calls `fclassif_cv(&data, &argvals, &labels, None, "lda", ncomp, nfold, seed)` with a fixed `seed` and small `nfold` (e.g. 5).
    - Because `fclassif_cv` is deterministic for a fixed seed regardless of feature flags, the test asserts that a second identical call returns a `fold_errors` `Vec` equal element-wise to the first, and equal `error_rate` — proving the collect-in-order contract holds and the parallel path (when compiled) does not reorder or corrupt results.
    - The test compiles and runs under BOTH default features and `--features parallel` (it is a plain `#[test]`, not feature-gated).
  </behavior>
  <action>
    Append a `#[cfg(test)] mod tests { use super::*; ... }` block at the end of cv.rs. Add `#[test] fn test_fclassif_cv_parallel_matches_sequential()` that: (1) builds deterministic test data via a small local helper (uniform-grid argvals, two well-separated Gaussian-bump classes, fixed integer labels — no RNG needed, or a `StdRng::seed_from_u64` with a hardcoded seed); (2) calls `fclassif_cv` twice with identical arguments and a fixed `seed`, method `"lda"`; (3) asserts `res_a.fold_errors.len() == res_b.fold_errors.len()` then iterates and asserts each pair is bit-for-bit equal via `assert_eq!` on the `f64` values (exact equality, not approximate — the whole point is determinism); (4) asserts `res_a.error_rate == res_b.error_rate`. Since determinism must hold across feature configurations, the test is a plain `#[test]` and is exercised by CI once per feature combo. Name the test exactly `test_fclassif_cv_parallel_matches_sequential`. Keep the dataset small so the test runs in well under a second.
  </action>
  <verify>
    <automated>cargo test -p fdars-core test_fclassif_cv_parallel_matches_sequential && cargo test -p fdars-core --features parallel test_fclassif_cv_parallel_matches_sequential</automated>
  </verify>
  <acceptance_criteria>
    - `cargo test -p fdars-core test_fclassif_cv_parallel_matches_sequential` exits 0 (sequential path).
    - `cargo test -p fdars-core --features parallel test_fclassif_cv_parallel_matches_sequential` exits 0 (parallel path — proves collect-in-order determinism).
    - `grep -c 'fn test_fclassif_cv_parallel_matches_sequential' fdars-core/src/classification/cv.rs` returns 1.
    - The test asserts element-wise `assert_eq!` on `fold_errors` values (bit-for-bit) AND `error_rate` equality — not a tolerance comparison.
    - `cargo clippy -p fdars-core --features parallel -- -D warnings` exits 0.
  </acceptance_criteria>
  <reversibility rating="reversible">Test-only addition; no production behavior change.</reversibility>
  <done>`test_fclassif_cv_parallel_matches_sequential` exists in an inline `#[cfg(test)] mod tests` block and passes under both default and `parallel` features, asserting bit-for-bit `fold_errors` and `error_rate` equality.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| (none new) | This change is an internal refactor of a private fold loop. No new trust boundary is crossed: no new external input, no new public API, no new dependency. |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-11-01-01 | Tampering | `fclassif_cv` result correctness under parallelism | low | mitigate | Bit-for-bit equivalence test (`test_fclassif_cv_parallel_matches_sequential`) proves parallel results match sequential; rayon collect-in-order contract prevents result reordering. |

Honest assessment: PERF-01 parallelizes an already-independent CV fold loop through the existing `iter_maybe_parallel!` macro. There is NO new external input, NO new attack surface, and NO new dependency (rayon is already present via the `parallel` feature). No other threats apply.
</threat_model>

<verification>
- `cargo test -p fdars-core` passes (default/sequential).
- `cargo test -p fdars-core --features parallel` passes (parallel path; equivalence test green).
- `cargo clippy -p fdars-core --features parallel -- -D warnings` passes.
- `grep 'iter_maybe_parallel!(0..nfold)' fdars-core/src/classification/cv.rs` matches; `grep 'fold_errors.push'` does not.
- No change to Cargo.toml dependencies.
</verification>

<success_criteria>
The `fclassif_cv` fold loop runs in parallel under the `parallel` feature via `iter_maybe_parallel!(0..nfold)`, produces `fold_errors` bit-for-bit identical to sequential for a fixed seed (verified by an inline equivalence test), compiles and passes with the `parallel` feature both on and off, and adds no new dependency.
</success_criteria>

<output>
Create `.planning/phases/11-performance-wins-parallel-cv-folds-faer-fpca-svd/11-01-parallel-cv-folds-SUMMARY.md` when done.
</output>
