---
phase: 33-model-based-density-functional-clustering
reviewed: 2026-08-20T00:00:00Z
depth: deep
files_reviewed: 3
files_reviewed_list:
  - fdars-core/src/gmm/subspace.rs
  - fdars-core/src/clustering_advanced.rs
  - fdars-core/src/test_helpers.rs
findings:
  critical: 2
  warning: 3
  info: 4
  total: 9
status: issues_found
---

# Phase 33: Code Review Report

**Reviewed:** 2026-08-20
**Depth:** deep
**Files Reviewed:** 3 (+ cross-referenced `gmm/mod.rs`, `gmm/covariance.rs`, `gmm/init.rs`, `gmm/em.rs`, `linalg.rs`, `matrix.rs`, `regression.rs`, `alignment/karcher.rs`, `lib.rs`)
**Status:** issues_found

## Summary

Phase 33 adds five functional clusterers across two new files (`gmm/subspace.rs`, `clustering_advanced.rs`) plus a test helper. The code compiles cleanly, follows project conventions, introduces no new crate dependencies, and is strictly additive — no existing public signature was modified.

Two correctness defects were found: an unguarded negative `complement_sq` in the funHDDC log-density function that systematically inflates E-step likelihoods, and a convergence-before-reinit bug in `align_cluster_fd` that can silently terminate the algorithm one iteration early after an empty-cluster reinitialization. Three warnings cover a missing validation guard in kCFC, a dead-but-incorrect variance accumulation in funFEM initialization, and a misleading shape inconsistency in the funHDDC degenerate-subspace result. Four info items cover dead code and minor quality issues.

Convention compliance is high: `#[must_use]`, `#[non_exhaustive]`, serde gating, column-major layout, and per-init RNG seeding all follow project patterns. The ARI helper and inline tests are well-structured.

---

## Critical Issues

### CR-01: Unguarded negative `complement_sq` in `log_density_subspace` inflates E-step log-likelihood

**File:** `fdars-core/src/gmm/subspace.rs:148-157`

**Issue:** The complement squared norm is computed as `complement_sq = diff_sq - z_sq` but is never clamped to zero before use. Due to floating-point arithmetic, when `diff` lies almost entirely within the subspace, `z_sq` can exceed `diff_sq` by a small epsilon, producing a negative `complement_sq`. The term `complement_sq / b_k` then adds a positive value to `ll` instead of contributing zero, inflating the log-density for observations that are well-explained by the subspace.

This biases the E-step: clusters with tight subspace fit are assigned artificially higher likelihood, distorting responsibilities away from the noise-variance term. The effect grows with large `(m - d_k_eff)` — i.e., high-dimensional data with small subspace dimension — exactly the regime where funHDDC is most useful. The M-step correctly clamps `complement_var` at zero (lines 359 and 470), but the E-step does not.

**Fix:**
```rust
// After computing z_sq:
let complement_sq = (diff_sq - z_sq).max(0.0); // clamp to avoid fp underflow bias
```

---

### CR-02: `align_cluster_fd` declares convergence after empty-cluster reinit, skipping one iteration

**File:** `fdars-core/src/clustering_advanced.rs:1421-1460`

**Issue:** After the reassignment step sets `changed = false` (no curve moved), the template-update step still runs. If any cluster is empty, it is reinitialized with a random non-member curve (lines 1429-1432), changing `templates[ki]` without setting `changed`. The convergence check at line 1458 then evaluates `!changed` as `true` and sets `converged = true`, breaking without running any further iteration. The new template is never used for a subsequent reassignment.

In practice, an empty cluster triggers reinit precisely because curves drifted away from it; the new random template may represent a region of the data space that would attract some curves on the next pass. Silently declaring convergence after reinit gives a result that is neither fully converged nor correctly reinitialized.

**Fix:** Track template changes separately and include them in the convergence check:

```rust
let mut template_changed = false;

// Inside the empty-cluster reinit block:
if !non_members.is_empty() {
    let rand_idx = rng.gen_range(0..non_members.len());
    templates[ki] = data.row(non_members[rand_idx]);
    template_changed = true;
}
// ...

// Replace the convergence check with:
if !changed && !template_changed {
    converged = true;
    break;
}
```

---

## Warnings

### WR-01: `kcfc_cluster` missing validation for `config.ncomp == 0`

**File:** `fdars-core/src/clustering_advanced.rs:371-572`

**Issue:** The `KcfcConfig` doc comment says "`ncomp` is clamped internally", but there is no validation that `config.ncomp >= 1`. When `config.ncomp == 0` is passed, `fdata_to_pc_1d` returns `FdarError::InvalidParameter`, which is silently swallowed at line 505-508 (`Err(_) => { // Degenerate cluster; keep prior model }`). All `fpca_models` remain `None`, all reconstruction errors become `f64::INFINITY`, and all n curves are assigned to cluster 0. No error is returned to the caller. This is silent wrong behavior — the caller has no way to distinguish this from a legitimate all-cluster-0 result.

The sibling function `funfem_cluster` correctly validates `config.ncomp == 0` and returns `InvalidParameter` (line 729-733).

**Fix:** Add entry-point validation in `kcfc_cluster`, immediately after the existing `config.k > n` check:

```rust
if config.ncomp == 0 {
    return Err(FdarError::InvalidParameter {
        parameter: "ncomp",
        message: "ncomp must be >= 1".to_string(),
    });
}
```

---

### WR-02: `update_gmm_params_from_hard` accumulates variance incorrectly due to wrong initial value

**File:** `fdars-core/src/clustering_advanced.rs:1107-1140`

**Issue:** `sigma_k[ki]` is initialized to `vec![1.0; d]` at line 1110, and then `diff * diff` is added to it at line 1127. The result is `1.0 + Σ (x_ij - mu_kj)²` instead of `Σ (x_ij - mu_kj)²`. After dividing by `counts[ki]`, the variance estimate is inflated by `1.0 / counts[ki]` per dimension. The correct initialization is zero.

While `sigma_k` is not currently used in any downstream computation within `funfem_cluster` (it is passed to `update_gmm_params_from_soft` which overwrites it completely), the field is computed during initialization and appears to represent a valid variance estimate. Any future reader or maintainer who reads `mu_k`/`sigma_k` from this function will get an incorrect value. The bug is also inconsistent with `update_gmm_params_from_soft` which correctly initializes `sigma_k[ki] = vec![1.0; d]` as a placeholder but immediately overwrites it from `var_j` (line 1184), never mixing addition with the initial value.

**Fix:**
```rust
// Line 1110 — change initialization to zero:
sigma_k[ki] = vec![0.0; d];
// (The accumulation at line 1127 and normalization at line 1133 are then correct.)
```

---

### WR-03: `FunHddcResult::subspaces[c]` has shape `(m, 1)` when `within_vars[c].len() == 0` for degenerate cluster

**File:** `fdars-core/src/gmm/subspace.rs:683-695`

**Issue:** At lines 688-693, when `d == av.len() == 0` or `flat.is_empty()`, the code returns `FdMatrix::zeros(m, 1)`. But the caller's contract (documented in the struct: "each is m × d_k_eff") implies the ncols of `subspaces[c]` should equal `within_vars[c].len()`. When the empty-cluster fallback fires (line 426-430 in `run_one_em`), `within_vars[c]` has length `d_k_req.min(m)` and `subspaces[c]` has length `m * d_k_req.min(m)`. At that point both are consistent. But the conversion at line 689 intercepts the `d == 0` case and substitutes an m×1 matrix, creating a `subspaces[c].ncols() == 1` vs `within_vars[c].len() == d_k_req.min(m)` mismatch.

In practice, `d == 0` can only occur if `within_vars[c]` is empty, which cannot happen after the fallback (it always gets `d_k_req.min(m) >= 1` elements because `d_k >= 1` is validated). But the code as written is fragile: if future changes allow `d == 0`, the returned result will have inconsistent shapes between the two related fields.

**Fix:** Either assert consistency or use the correct fallback size:
```rust
FdMatrix::zeros(m, d.max(1))  // preserve the dimension even for degenerate clusters
```

---

## Info

### IN-01: `#![allow(non_snake_case)]` placed after opening doc comment

**File:** `fdars-core/src/gmm/subspace.rs:1-2`

**Issue:** The inner attribute `#![allow(non_snake_case)]` on line 2 appears after the `//!` module doc comment on line 1. While Rust accepts this ordering (both are inner attributes), the project convention is to place lint attributes before documentation. It also makes the function name `funhddC_cluster` permanently non-snake-case in the public API; a more conventional approach would be to name the function `funhddc_cluster` and document the capitalization choice explicitly.

**Fix:** Either reorder so `#![allow(non_snake_case)]` is on line 1, or consider renaming the function to `funhddc_cluster` and noting the deviation in the doc comment.

---

### IN-02: `sigma_k` is computed but never read in `funfem_cluster`

**File:** `fdars-core/src/clustering_advanced.rs:815, 825, 960-968`

**Issue:** `sigma_k` (per-cluster diagonal variance in score space) is initialized at line 815, updated via `update_gmm_params_from_hard` at line 825, and updated each iteration via `update_gmm_params_from_soft` at line 960-968. It is never read in the E-step, scatter computation, or result construction. This is dead work. It also causes confusion because the incorrectly computed value in `update_gmm_params_from_hard` (WR-02) could mislead readers who expect `sigma_k` to hold meaningful data.

**Fix:** Either remove `sigma_k` from `update_gmm_params_from_hard` / `update_gmm_params_from_soft` signatures and eliminate the computation, or add a comment making clear that it is maintained for potential future use and is NOT a correct variance estimate until the first call to `update_gmm_params_from_soft`.

---

### IN-03: DBSCAN BFS uses `Vec::contains` (O(queue.len()) per lookup) to prevent duplicate enqueuing

**File:** `fdars-core/src/clustering_advanced.rs:235`

**Issue:** `if !queue.contains(&nb)` at line 235 is O(current queue length) per neighbor. For a large dense cluster (queue grows to O(n)), this produces O(n²) work inside the BFS for a single cluster. The `visited` array at line 227 handles the standard DBSCAN termination; the `queue.contains` check is only needed to avoid re-queuing a point that is already in the queue but not yet visited. This is a correctness concern only at scale — for small n (< a few hundred curves) it is not observable — but for large functional datasets it degrades noticeably.

**Fix:** Replace the linear scan with a boolean `in_queue` array:
```rust
let mut in_queue: Vec<bool> = vec![false; n];
// Mark initial neighbors:
for &nb in &neighbors { in_queue[nb] = true; }
// ...
if !in_queue[nb] {
    in_queue[nb] = true;
    queue.push(nb);
}
```

---

### IN-04: `test_kcfc_errors_ordering` assumes `k == 2` by using `1 - gt0_cluster`

**File:** `fdars-core/src/clustering_advanced.rs:1759-1784`

**Issue:** The test computes `gt1_cluster = 1 - gt0_cluster`, which only works for exactly two clusters. The test hardcodes k=2 and the assertion, so this is not currently broken. But if the test is refactored with k > 2, the `1 - gt0_cluster` arithmetic would produce an incorrect cluster index (potentially wrapping to `usize::MAX`). The test should use explicit label mapping rather than arithmetic inversion.

**Fix:** Replace `1 - gt0_cluster` with an explicit search for the cluster containing the first curve in ground truth group 1:
```rust
let gt1_cluster = result.cluster[n_per]; // first curve in ground-truth group 1
```

---

_Reviewed: 2026-08-20_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
