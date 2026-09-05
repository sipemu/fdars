---
phase: "72-jfpca-fit-transform-seam"
reviewed: "2026-09-05T14:30:00Z"
depth: standard
files_reviewed: 4
files_reviewed_list:
  - fdars-core/src/jfpca_model.rs
  - fdars-core/src/elastic_fpca.rs
  - fdars-core/src/lib.rs
  - fdars-core/src/prelude.rs
findings:
  critical: 0
  warning: 3
  info: 2
  total: 5
status: issues_found
---

# Phase 72: Code Review Report

**Reviewed:** 2026-09-05T14:30:00Z
**Depth:** standard (per-file + cross-file Rust checks)
**Files Reviewed:** 4
**Status:** issues_found

## Summary

`jfpca_model.rs` is a well-structured addition. The core deliverable — `JfpcaModel`, `JfpcaTransform`, `jfpca_fit`, `transform()`, and `score_training()` — is correct at the algorithmic level. The score formula (`project_joint`) exactly matches RESEARCH §9; the amplitude and phase dimension bounds (`0..m_aug` / `0..m`) are correct; no off-by-one errors. The `build_combined_representation` visibility promotion in `elastic_fpca.rs` and the additive `lib.rs` / `prelude.rs` re-exports are non-breaking.

Three quality/robustness concerns require attention before this surface is exposed further (e.g., Phase 73 VEESA). None constitute silent data corruption — all produce panics or wrong-but-detectable results — but two of the three (`score_training` validation gap, missing `max_iter` guard) can be silently triggered by plausible caller mistakes.

---

## Narrative Findings (AI reviewer)

## Warnings

### WR-01: `score_training` validates `training_aligned` but not `training_gammas` — silent wrong results or panic

**File:** `fdars-core/src/jfpca_model.rs:331-363`

**Issue:** `score_training()` checks `training_aligned.ncols() == m` but makes no equivalent check on `training_gammas`. All fields of `JfpcaModel` are `pub`, so a caller who serializes a model (serde feature), modifies `training_gammas` externally, and then calls `score_training()` will feed a shape-mismatched matrix into `warps_to_normalized_psi`. Inside that function, `m` is taken from `gammas.shape().1`, not from `argvals`'s length, so the computation proceeds with the wrong dimensionality. If `training_gammas.ncols() > argvals.len()`, the `argvals[m-1]` access at `elastic_fpca.rs:648` uses gammas' `m`, which exceeds `argvals`'s bounds and panics (Vec bounds panic in release). If `training_gammas.ncols() < argvals.len()`, it produces silently wrong psis.

This is a correctness gap for the public surface: the error doc promises `InvalidDimension` returns, but a `training_gammas` mismatch is entirely silent until panic or wrong output.

**Fix:** Add a shape check on `training_gammas` at the top of `score_training`, alongside the existing check for `training_aligned`:

```rust
pub fn score_training(&self) -> Result<JfpcaTransform, FdarError> {
    let m = self.argvals.len();
    let (n_tr, m_tr) = self.training_aligned.shape();
    if m_tr != m {
        return Err(FdarError::InvalidDimension {
            parameter: "training_aligned columns",
            expected: format!("== {} (argvals length)", m),
            actual: format!("{}", m_tr),
        });
    }
    // ADD: validate training_gammas shape before it is passed to warps_to_normalized_psi.
    let (n_gam, m_gam) = self.training_gammas.shape();
    if n_gam != n_tr || m_gam != m {
        return Err(FdarError::InvalidDimension {
            parameter: "training_gammas",
            expected: format!("({}, {})", n_tr, m),
            actual: format!("({}, {})", n_gam, m_gam),
        });
    }
    // ... rest unchanged
```

---

### WR-02: `jfpca_fit` does not validate `max_iter >= 1` — silent degenerate model

**File:** `fdars-core/src/jfpca_model.rs:182-199`

**Issue:** The validation guard at line 184 checks `n < 2 || m < 2 || ncomp < 1 || argvals.len() != m` but does not check `max_iter < 1`. When `max_iter == 0`, `karcher_mean_impl` runs zero iterations for both its coarse and fine loops (both compute to `0` via the `max_iter / 2` split). The function returns an unaligned Karcher result using only the initial mean. `jfpca_fit` then succeeds and returns a `JfpcaModel` with a degenerate alignment base. Downstream use of this model produces numerically incorrect scores without any error signal.

A caller passing `max_iter: 0` by accident (e.g., from a misconfigured loop or a default struct field) gets a silently wrong model.

**Fix:** Add `max_iter < 1` to the entry guard:

```rust
if n < 2 || m < 2 || ncomp < 1 || argvals.len() != m || max_iter < 1 {
    return Err(FdarError::InvalidDimension {
        parameter: "data/argvals/ncomp/max_iter",
        expected: "n >= 2, m >= 2, ncomp >= 1, argvals.len() == m, max_iter >= 1".to_string(),
        actual: format!(
            "n={}, m={}, ncomp={}, argvals.len()={}, max_iter={}",
            n, m, ncomp, argvals.len(), max_iter
        ),
    });
}
```

Alternatively, use `FdarError::InvalidParameter` for `max_iter` since it is a scalar range constraint, not a dimension mismatch. Either variant is acceptable; the key is that the error is returned rather than silently producing wrong results.

---

### WR-03: `transform()` and `score_training()` are missing `#[must_use]`

**File:** `fdars-core/src/jfpca_model.rs:265, 331`

**Issue:** `jfpca_fit` carries `#[must_use]` (line 174), but neither `transform()` nor `score_training()` does. Both methods are expensive (alignment + SRSF computation + projection) and return a `Result<JfpcaTransform, FdarError>` that is meaningless if discarded. The project convention (`CLAUDE.md`: "`#[must_use]` on expensive computations") explicitly requires this annotation; omitting it is inconsistent with the 74+ other annotated functions and with the pattern in comparable methods (`beta_surface`, `predict_fosr_2d`, etc.).

**Fix:**

```rust
// Before JfpcaModel::transform:
#[must_use = "expensive computation: transform result (scores, aligned, warping) should not be discarded"]
pub fn transform(&self, new_curves: &FdMatrix) -> Result<JfpcaTransform, FdarError> {

// Before JfpcaModel::score_training:
#[must_use = "expensive computation: score_training result should not be discarded"]
pub fn score_training(&self) -> Result<JfpcaTransform, FdarError> {
```

---

## Info

### IN-01: `build_combined_representation` promoted to `pub(crate)` but unused outside `elastic_fpca.rs`

**File:** `fdars-core/src/elastic_fpca.rs:914`

**Issue:** `build_combined_representation` was promoted from private `fn` to `pub(crate) fn` in commit `ee5eab2f`. However, `jfpca_model.rs` does not import or call it — the scoring is correctly inlined in `project_joint` as a direct dot-product loop. The promotion adds unnecessary crate-level visibility to a function that has no external consumer. `pub(crate)` visibility does not trigger Rust's `dead_code` lint, so this will not cause a compile warning, but it enlarges the internal API surface without benefit.

**Fix:** Revert to private `fn`:

```rust
// elastic_fpca.rs:914
fn build_combined_representation(   // revert from pub(crate)
```

If Phase 73 or a later phase needs `build_combined_representation` directly, the promotion can be reinstated then.

---

### IN-02: `horiz_fpca` is called twice for the same `karcher` result — double computation

**File:** `fdars-core/src/jfpca_model.rs:203-208`

**Issue:** `jfpca_fit` calls `joint_fpca(&karcher, ...)` at line 203, which internally calls `horiz_fpca` (see `elastic_fpca.rs:308`). Then `jfpca_fit` calls `horiz_fpca(&karcher, ...)` again at line 208 solely to capture `mean_psi`. This doubles the cost of horizontal FPCA (sphere Karcher mean on psis, SVD of shooting vectors) for every `jfpca_fit` call. Both calls are deterministic and produce the same `mean_psi`, so correctness is unaffected.

The fix would require either promoting `HorizFpcaResult` out of `joint_fpca` or restructuring `joint_fpca` to return it alongside `JointFpcaResult`. This is a non-trivial API change; the current solution is a pragmatic workaround that avoids modifying `JointFpcaResult`.

**Fix (deferred):** Accept as-is for Phase 72. File a backlog item: expose `horiz_fpca`'s result from `joint_fpca` (e.g., return a tuple or extend `JointFpcaResult`) to avoid the duplicate computation in Phase 73+ when `jfpca_fit` usage grows. No action required before shipping Phase 72.

---

_Reviewed: 2026-09-05T14:30:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
