---
phase: 77-gradient-api-composition-demo-integration
reviewed: 2026-09-06T00:00:00Z
depth: deep
files_reviewed: 4
files_reviewed_list:
  - fdars-core/src/autodiff.rs
  - fdars-core/src/lib.rs
  - fdars-core/src/prelude.rs
  - fdars-core/tests/autodiff_reexports.rs
findings:
  critical: 0
  warning: 1
  info: 2
  total: 3
status: findings
---

# Phase 77: Code Review Report

**Reviewed:** 2026-09-06
**Depth:** deep
**Files Reviewed:** 4
**Status:** issues_found (1 warning, 2 info — no blockers)

## Summary

Reviewed the public gradient API (`grad` / `jacobian` / `directional_derivative`), the composed-objective FD cross-check test, the composition module doctest, and the crate-root + prelude re-exports for the final integration phase of milestone v0.39.0 (forward-mode AD).

**Verdict on the two load-bearing checks:**

- **`grad` correctness: CORRECT.** `grad<F: Fn(&[Dual]) -> Dual>(f, x) -> (f64, Vec<f64>)` seeds input `k` with tangent 1.0 (`Dual::seed`) and all others with tangent 0.0 (`Dual::constant`), runs one forward pass per input, and collects `gradient[k] = tangent of the k-th run`. `value` is captured on pass `k==0` only, which is sound because the primal is identical across passes (only tangents differ). The `m == 0` edge returns `(f(&[]).value, Vec::new())` without indexing `x` — no panic. Verified against the closed-form `Sum(xi^2) -> [2,4,6]` inline test (<=1e-12) and the empty-input test. `jacobian` and `directional_derivative` are analogous and correct (verified by `jacobian_known_answer` and `directional_derivative_projects_gradient`).

- **Composition FD test: LEGITIMATE, not rigged.** `grad_composed_objective_matches_finite_diff` is a genuine cross-check:
  - Uses a **spanning full-rank** training set: `n=40 >> m=24` seeded random combos (`StdRng::seed_from_u64(20260906)`) of `sin(pi t)`, `cos(2 pi t)`, `sin(3 pi t)` with independent coefficients per curve — not low-rank phase-shifted sinusoids.
  - Grid is on `[0.1, 0.9]` (avoids SRSF derivative-zero / degenerate endpoints), matching the milestone convention.
  - The objective genuinely wires **two** differentiable ops: `soft_dtw_distance_generic` (a nonlinear soft-min DP recurrence over squared costs, `gamma=0.1`) **plus** `lambda * sum(project_scores_generic(...)^2)` (FPCA-score projection, quadratic). Both route the input `c` through — the objective is neither linear nor constant.
  - The FD comparison perturbs **every** component `j` by `±h` (`h=1e-6`) and asserts `|gradient[j] - (f(x+h)-f(x-h))/2h| < 1e-6` — a real component-by-component non-vacuous comparison (AD vs central FD, not zeros-vs-zeros). It also asserts f64 composition parity `|value - f64_obj(curve)| < 1e-12`.

  I ran the suite: all 30 autodiff unit tests pass (incl. `grad_composed_objective_matches_finite_diff`), both reexports integration tests pass, and all 6 autodiff doctests pass (incl. the composed module doctest at line 48).

**Re-export completeness:** confirmed. `project_scores_generic` appears exactly once in `lib.rs` (line 582, pre-existing) — not duplicated. `soft_dtw_distance_generic` (line 640), `amplitude_distance_at_warp_generic` (line 272), and `pub use autodiff::{diff, directional_derivative, grad, jacobian, Dual, Scalar}` (line 585) are all present. The prelude adds the same surface. `tests/autodiff_reexports.rs` exercises BOTH crate-root and `prelude::*` paths with live calls to every symbol (no dead code, no false positives).

**Additive/non-breaking:** confirmed. `git show f3b4064b -- fdars-core/Cargo.toml` produces no diff — `[dependencies]` untouched. No `pub fn` / `pub struct` / `pub trait` line was removed; the only `-` lines in the diff are rustfmt reflows of existing re-export lists (identical identifier sets). `diff`, `Dual`, `Scalar`, and the three generic ops keep their signatures.

No blocking or security issues. One robustness warning and two informational notes below.

## Warnings

### WR-01: `directional_derivative` silently truncates on `direction`/`x` length mismatch in release builds

**File:** `fdars-core/src/autodiff.rs:589-607`
**Issue:** The length equality is enforced only by `debug_assert_eq!`, which is a no-op in release (`cargo build --release`, `--profile release` tests). The body then does `x.iter().zip(direction.iter())`, and `zip` stops at the **shorter** iterator. If a caller passes `direction.len() < x.len()`, the built `duals` vector is silently shorter than `x`, so the user closure `f` receives fewer inputs than expected. Depending on `f`, this either panics on out-of-bounds indexing inside `f` (a confusing, misattributed panic) or — worse — returns a wrong directional derivative with no error. This diverges from the crate's stated convention (CLAUDE.md: "public functions never panic on input validation; return `Result` / validate dimensions at entry"). `grad` and `jacobian` are not affected (they size everything off `x.len()` internally).

**Fix:** Either promote the check to a hard runtime guard, or (matching the crate's `Result`-based validation convention) surface it. Minimal hard-guard fix:
```rust
pub fn directional_derivative<F: Fn(&[Dual]) -> (f64, f64)>(/* ... */) -> (f64, f64) {
    assert_eq!(
        direction.len(),
        x.len(),
        "direction length ({}) must match input length ({})",
        direction.len(),
        x.len()
    );
    // ... unchanged
}
```
(Using `assert_eq!` keeps the signature additive and non-breaking while guaranteeing the failure is loud and correctly attributed in all build profiles. A `Result`-returning variant would be more idiomatic but changes the return type.)

## Info

### IN-01: `grad` / `jacobian` re-seed the full input vector every pass (m² allocations)

**File:** `fdars-core/src/autodiff.rs:509-528` (`grad`), `558-587` (`jacobian`)
**Issue:** Each of the `m` passes rebuilds the entire length-`m` `Vec<Dual>` from scratch (`(0..m).map(...).collect()`), so `grad` performs `m` allocations of size `m` (O(m²) `Dual` writes) even though only one element's tangent changes between passes. Performance is explicitly out of v1 review scope and this is correct, but it is a natural future optimization: allocate the `duals` buffer once as all-constant, then on pass `k` set `duals[k].tangent = 1.0`, run `f`, and reset `duals[k].tangent = 0.0`. Noting only; no action required for this phase.

### IN-02: `jacobian` empty-input branch shape depends on `f(&[])` output arity

**File:** `fdars-core/src/autodiff.rs:551-556`
**Issue:** For `m == 0`, `jacobian` returns `(values, vec![Vec::new(); rows])` where `rows = f(&[]).len()`. This is internally consistent (an `n × 0` Jacobian: `n` output rows, each with zero columns). It is correct, but the doc comment ("An empty input yields `(values, empty rows)`") is slightly terse — a caller iterating `j[i][k]` on the empty-input result would index into empty rows. Behavior is fine; a one-line doc clarification that the rows are length-0 (not length-`n`) would remove any ambiguity. Cosmetic only.

---

_Reviewed: 2026-09-06_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
