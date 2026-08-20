---
phase: 33-model-based-density-functional-clustering
fixed_at: 2026-08-20T00:00:00Z
review_path: .planning/phases/33-model-based-density-functional-clustering/33-REVIEW.md
iteration: 1
findings_in_scope: 6
fixed: 6
skipped: 0
status: all_fixed
---

# Phase 33: Code Review Fix Report

**Fixed at:** 2026-08-20
**Source review:** `.planning/phases/33-model-based-density-functional-clustering/33-REVIEW.md`
**Iteration:** 1

**Summary:**
- Findings in scope: 6 (CR-01, CR-02, WR-01, WR-02, WR-03, IN-04)
- Fixed: 6
- Skipped: 0

## Fixed Issues

### CR-01: Clamp `complement_sq` to 0 in `log_density_subspace` E-step

**Files modified:** `fdars-core/src/gmm/subspace.rs`
**Commit:** 32260c7b
**Applied fix:** Changed `let complement_sq = diff_sq - z_sq;` to `let complement_sq = (diff_sq - z_sq).max(0.0);` with an explanatory comment. Floating-point arithmetic can make z_sq slightly exceed diff_sq when a curve lies almost entirely within the subspace, producing a negative complement that inflates the log-density. The M-step already clamped this value; now the E-step does too.

---

### WR-03: Use `d.max(1)` for degenerate subspace fallback shape

**Files modified:** `fdars-core/src/gmm/subspace.rs`
**Commit:** f225ade0
**Applied fix:** Changed `FdMatrix::zeros(m, 1)` to `FdMatrix::zeros(m, d.max(1))` in the degenerate-cluster conversion path. This preserves shape consistency between `subspaces[c].ncols()` and `within_vars[c].len()` per the `FunHddcResult` struct contract.

---

### WR-01: Reject `ncomp == 0` in `kcfc_cluster` at entry

**Files modified:** `fdars-core/src/clustering_advanced.rs`
**Commit:** e8a11d79
**Applied fix:** Added `if config.ncomp == 0 { return Err(FdarError::InvalidParameter { parameter: "ncomp", ... }) }` immediately after the `config.k > n` guard in `kcfc_cluster`. Previously a zero ncomp caused `fdata_to_pc_1d` to return Err which was silently swallowed by the degenerate-cluster arm, assigning all curves to cluster 0 with no signal to the caller. Also added a regression test `test_kcfc_ncomp_zero_returns_err` that verifies both the Err result and the correct parameter name.

---

### WR-02: Initialize `sigma_k` to zero in `update_gmm_params_from_hard`

**Files modified:** `fdars-core/src/clustering_advanced.rs`
**Commit:** 457e731f
**Applied fix:** Changed `sigma_k[ki] = vec![1.0; d];` to `sigma_k[ki] = vec![0.0; d];` so that the subsequent accumulation `sigma_k[ki][j] += diff * diff` yields the correct sum of squared deviations instead of `1.0 + Σ(x-μ)²`.

---

### CR-02: Prevent premature convergence after empty-cluster reinit in `align_cluster_fd`

**Files modified:** `fdars-core/src/clustering_advanced.rs`
**Commit:** 9c533342
**Applied fix:** Introduced a `template_changed: bool` flag initialized to `false` before the template-update loop. The flag is set to `true` only inside the empty-cluster reinit branch (`if !non_members.is_empty()`). The convergence check was changed from `if !changed` to `if !changed && !template_changed`, ensuring that after an empty-cluster reinit the loop runs at least one more iteration to test the new template. The outer `for _iter in 0..config.max_iter` loop still guarantees termination.
**Status:** fixed: requires human verification (logic fix — convergence semantics change)

---

### IN-04: Remove fragile `1 - gt0_cluster` arithmetic in `test_kcfc_errors_ordering`

**Files modified:** `fdars-core/src/clustering_advanced.rs`
**Commit:** a2ac9019
**Applied fix:** Replaced `let gt1_cluster = 1 - gt0_cluster;` with `let gt1_cluster = result.cluster[n_per];` (explicit lookup of the first curve in ground-truth group 1). Also replaced `1 - expected_cluster` in the error comparison with a `find`-based scan: `let other_cluster = (0..2).find(|&c| c != expected_cluster).unwrap_or(0);`. Eliminates the latent `usize` wrap-around panic that would occur if k > 2.

---

## Skipped Issues

None — all 6 in-scope findings were fixed.

---

_Fixed: 2026-08-20_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
_Verification: Tier 1 (re-read) + Tier 2 (`cargo build --lib` clean) for all fixes._
_Tests ran in: main checkout (workflow.use_worktrees=false — sequential mode)._
_Test result: 32 clustering_advanced tests + 8 gmm::subspace tests = 40 tests, all passing._
