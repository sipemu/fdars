---
phase: 14-shift-registration
reviewed: 2026-08-12T12:37:28Z
depth: deep
files_reviewed: 4
files_reviewed_list:
  - fdars-core/src/alignment/shift.rs
  - fdars-core/src/alignment/quality.rs
  - fdars-core/src/alignment/mod.rs
  - fdars-core/src/lib.rs
findings:
  critical: 1
  warning: 3
  info: 3
  total: 7
status: issues_found
---

# Phase 14: Shift Registration - Code Review Report

**Reviewed:** 2026-08-12T12:37:28Z
**Depth:** deep (cross-file + call-chain + CI-config cross-check)
**Files Reviewed:** 4
**Status:** issues_found

## Summary

Reviewed the Phase-14 diff (`eb06c5e7..HEAD`) for `least_squares_shift_registration` /
`ShiftRegistrationResult` (`shift.rs`) and the three registration-quality scores
(`quality.rs`), plus the additive re-exports in `mod.rs` / `lib.rs`.

**Correctness of the numerical core is sound.** Golden-section search is the standard
textbook form with a correct bracket `[−max_shift, +max_shift]`, a convergence guard
(`(hi−lo) < GS_TOL`) AND an iteration cap (`GS_MAX_ITER = 100`) — so there is no
infinite loop and no premature stop; endpoint minima converge safely to the boundary.
The L2 objective is Simpson-weighted and evaluated consistently against `mean_1d`
via `linear_interp`, whose boundary-clamping is non-panicking (verified at
`helpers.rs:172-191`). Input validation is thorough and every public function returns
`Result<T, FdarError>` without a panic path. The score formulas correctly implement the
CONTEXT-locked **standalone-energy** forms — they were **not** silently reverted to
scikit-fda's ratio-based scores. The `λ=0` Sobolev path returns the LS term exactly
(early return), and the pairwise n<2 guard is present.

**One BLOCKER:** two leftover test-only warnings (an unused variable and a dead
function) in `shift.rs` will fail CI, which runs `cargo clippy --all-targets ... -D warnings`.

The remaining findings are documentation/consistency issues: the pairwise score is
documented as "Pearson correlation" but implements cosine similarity (uncentered), and
two validation-consistency gaps between the shift function and the score functions.

## Critical Issues

### CR-01: Dead-code / unused-variable warnings in `shift.rs` tests will fail CI (`-D warnings`)

**File:** `fdars-core/src/alignment/shift.rs:277` and `fdars-core/src/alignment/shift.rs:411`
**Issue:**
CI enforces `cargo clippy --all-targets --features linalg,parallel,serde -- -D warnings ...`
(`.github/workflows/rust-ci.yml:77`). Because `--all-targets` compiles test code, the
following two warnings are promoted to hard errors and will break the clippy job:

- `shift.rs:277` — `fn make_shifted_bumps(...)` is defined in the test module but never
  called (`warning: function make_shifted_bumps is never used`). The quality-tests copy of
  this helper *is* used; the shift-tests copy is not.
- `shift.rs:411` — `let argvals = uniform_grid(m);` in `test_shift_registration_argvals_mismatch`
  is never read (the test passes `wrong_argvals` / `short_argvals`), producing
  `warning: unused variable: argvals`.

Both were reproduced locally under `cargo clippy -p fdars-core --all-targets`.

**Fix:**
Remove the unused helper and drop the dead binding (or prefix with `_`):

```rust
// shift.rs:277 — delete the unused test helper entirely:
// fn make_shifted_bumps(n: usize, m: usize, delta: f64) -> (FdMatrix, Vec<f64>) { ... }

// shift.rs:411 — remove the unused `argvals` binding in test_shift_registration_argvals_mismatch:
    fn test_shift_registration_argvals_mismatch() {
        let m = 5;
        let data = FdMatrix::zeros(2, m);          // <- drop `let argvals = uniform_grid(m);`
        let wrong_argvals = uniform_grid(m + 1);
        ...
```

## Warnings

### WR-01: `pairwise_correlation_score` documented as "Pearson correlation" but implements cosine similarity

**File:** `fdars-core/src/alignment/quality.rs` (rustdoc at the `pairwise_correlation_score`
doc block; implementation `inner / (norms[i]*norms[k])`)
**Issue:**
The rustdoc and the CONTEXT/RESEARCH spec both say "mean **Pearson** correlation", but the
implementation computes the **uncentered** normalized L2 inner product
`⟨fᵢ,fₖ⟩ / (‖fᵢ‖·‖fₖ‖)` — i.e. cosine similarity in L2 space. True Pearson correlation
subtracts each curve's (weighted) mean before taking the inner product. For non-zero-mean
curves (the Gaussian-bump fixtures are strictly positive, so mean ≫ 0) these differ
materially: cosine similarity of two positive bumps is inflated toward 1 regardless of shift,
which weakens the discriminative power the "rises after registration" test relies on.
RESEARCH assumption A1 already flags this as a deliberate divergence, but the public rustdoc
still calls it "functional **Pearson** correlation", which will mislead callers comparing
against scikit-fda or textbook Pearson.

**Fix:**
Either (a) correct the terminology in the rustdoc to "cosine similarity (uncentered L2
inner product)", explicitly noting it equals Pearson only for zero-mean curves; or (b)
center each row by its Simpson-weighted mean before the inner product to make it a true
Pearson correlation. Minimal doc fix:

```rust
/// **Formula:** `mean over (i<k) of [⟨fᵢ, f_k⟩_L2 / (‖fᵢ‖_L2 · ‖f_k‖_L2)]`
/// (uncentered normalized L2 inner product, i.e. cosine similarity — equals
/// Pearson correlation only for zero-mean curves).
```

### WR-02: `least_squares_score` / `sobolev_least_squares_score` accept `m == 1`, unlike the shift function

**File:** `fdars-core/src/alignment/quality.rs` (validation blocks of `least_squares_score`
and `sobolev_least_squares_score`)
**Issue:**
`least_squares_shift_registration` explicitly rejects `argvals.len() < 2`
(`shift.rs:200-205`), but the two score functions only check `n==0 || m==0` and the
argvals-length match — they will happily accept a single-point grid (`m == 1`). At `m == 1`,
`simpsons_weights` returns `vec![1.0]` (its `n < 2` fallback), so the "integral" is a bare
point-mass with no domain length — a meaningless quantity for a quality *integral*. For
`sobolev_least_squares_score` with `m == 1` and `lambda > 0`, `h = (argvals[0]-argvals[0])/0
= NaN`; this happens to be masked only because `gradient_uniform` early-returns for `n < 2`
without using `h`. This is a latent inconsistency that could surface as NaN if the guard in
`gradient_uniform` ever changes.

**Fix:**
Add the same `argvals.len() < 2` guard to both score functions for consistency and to keep
the integral well-defined:

```rust
if argvals.len() < 2 {
    return Err(FdarError::InvalidParameter {
        parameter: "argvals",
        message: "must have at least 2 evaluation points".to_string(),
    });
}
```

### WR-03: `sobolev_least_squares_score` silently ignores non-uniform grids (no validation, no `NaN` guard)

**File:** `fdars-core/src/alignment/quality.rs` (Sobolev derivative branch:
`let h = (argvals[m-1] - argvals[0]) / (m-1) as f64;`)
**Issue:**
The derivative term computes a single uniform spacing `h` and feeds `gradient_uniform`, but
nothing verifies the grid is actually uniform. On a genuinely non-uniform grid the derivative
(and hence the whole Sobolev penalty) is silently wrong — the caller gets a plausible-looking
number with no error. The rustdoc mentions the assumption, but per project convention
(dimension/parameter checks at entry, "no silent truncation"), a silent numerically-incorrect
result on valid-shaped input is a robustness defect. This mirrors the pre-existing
`warp_smoothness` behavior, so it is a WARNING rather than a BLOCKER, but the new `Result`
return type gives a natural channel to reject it.

**Fix:**
When `lambda > 0`, either validate uniformity and return `Err(FdarError::InvalidParameter{...})`
on a non-uniform grid, or use the already-available `gradient_nonuniform` helper referenced in
the rustdoc so the score is correct on any grid. Minimal guard:

```rust
let h = (argvals[m - 1] - argvals[0]) / (m - 1) as f64;
let uniform = argvals.windows(2).all(|w| ((w[1]-w[0]) - h).abs() < 1e-9 * h.abs());
if !uniform {
    return Err(FdarError::InvalidParameter {
        parameter: "argvals",
        message: "sobolev_least_squares_score with lambda>0 requires a uniform grid".to_string(),
    });
}
```

## Info

### IN-01: `l2_shift_objective` and the score loops duplicate the Simpson-weighted squared-diff sum

**File:** `fdars-core/src/alignment/shift.rs:103-120`, `quality.rs` (LS loop in
`least_squares_score` and the identical LS loop inside `sobolev_least_squares_score`)
**Issue:**
The `(a-b)*(a-b)*w` Simpson-weighted squared-distance sum is written out three times
(`least_squares_score`, the LS term of `sobolev_least_squares_score`, and conceptually in
`l2_shift_objective`). `sobolev_least_squares_score` could call `least_squares_score` for its
LS term instead of re-deriving mean/weights and re-summing.
**Fix:** Factor a small private `weighted_sq_dist(row, mean, weights) -> f64` helper, or have
`sobolev_least_squares_score` delegate the LS term to `least_squares_score(registered, argvals)?`.

### IN-02: `least_squares_score` recomputes `mean`/`weights` that `sobolev` also computes

**File:** `fdars-core/src/alignment/quality.rs` (both score bodies)
**Issue:** Minor duplication of `simpsons_weights(argvals)` + `mean_1d(registered)` across the
two energy scores; not a correctness issue, noted for maintainability.
**Fix:** Optional — share a private helper as in IN-01.

### IN-03: `max_shift` default is documentation-only; callers can pass a shift larger than the domain

**File:** `fdars-core/src/alignment/shift.rs:178-211`
**Issue:** `max_shift` is validated only as `> 0.0`. A caller may pass `max_shift` far larger
than the domain range; golden-section then searches shifts that push every evaluation off-grid,
where `linear_interp` clamps to the endpoints. This is non-panicking and well-defined (per the
Boundary policy), so it is Info, not a defect — but the objective becomes flat/degenerate for
huge shifts and the returned δ can be arbitrary within the flat region.
**Fix:** Optional — document that `max_shift` should not exceed the domain range, or soft-clamp
`max_shift` to `argvals.last() - argvals.first()`.

---

_Reviewed: 2026-08-12T12:37:28Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
