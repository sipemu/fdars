---
phase: 47-hot-path-allocation-performance
reviewed: 2026-08-31T00:00:00Z
depth: deep
files_reviewed: 4
files_reviewed_list:
  - fdars-core/src/fts/spectral.rs
  - fdars-core/src/fpca_variants.rs
  - fdars-core/src/fts/acf.rs
  - fdars-core/src/irreg_fdata/smoothing.rs
  - fdars-core/src/fem_smoothing.rs
findings:
  critical: 0
  warning: 0
  info: 0
  total: 0
status: clean
---

# Phase 47: Code Review Report

**Reviewed:** 2026-08-31T00:00:00Z
**Depth:** deep (cross-file, call-chain tracing, index arithmetic verification)
**Files Reviewed:** 5 (4 source + `fem_smoothing.rs`)
**Status:** clean

## Summary

Five behavior-preserving optimizations (OPT-A through OPT-F) were reviewed across four source
files changed in commits `14db185f`–`47ef4796`. Every change was traced for algebraic equivalence
to the prior code. Build (`cargo build --features linalg,parallel`) and clippy
(`cargo clippy --all-targets --features linalg,parallel -- -D warnings`) are both clean.

No correctness bugs, security issues, or quality defects were found.

---

## Detailed Verification

### OPT-A — `eigen_at_frequency` index-sort (`fts/spectral.rs`)

**Claim:** Replace `Vec<(f64, Vec<f64>)>` pair-sort with an index-sort over `idx: Vec<usize>`,
materializing only the retained `ncomp` eigenvectors.

**Verified:**
- `idx` starts as `(0..m).collect()` — identical ascending original-index order to the old
  `pairs` built by `zip(eigenvalues.iter(), eigenvectors.column_iter())`.
- Both old and new use `sort_by` (Rust stable sort); tie-breaking preserves ascending original
  index in both paths. ✓
- Old code applied sign-alignment to ALL `m` vectors and then `.take(ncomp)`. New code applies
  sign-alignment only to the `ncomp` vectors actually retained. Each eigenvector's sign flip is
  independent, so the output for the retained set is identical. ✓
- Caller (`dpca`) validates `ncomp >= 1 && ncomp <= m` before calling `eigen_at_frequency`;
  `take = ncomp.min(m)` is a defensive guard that never fires in practice and is not a behavior
  change. ✓

### OPT-B — `fsvd` Gram matrix `from_fn` (`fpca_variants.rs`)

**Claim:** Replace flat `gram` staging Vec + `DMatrix::from_column_slice` with
`DMatrix::from_fn`.

**Verified (gram_on_right branch):**
- Old: `gram[a + b * q] = Σ_s cw[(s,a)]·cw[(s,b)]` (column-major; row=a, col=b at flat index
  `a + b*q`).
- New: `DMatrix::from_fn(q, q, |a, b| Σ_s cw[(s,a)]·cw[(s,b)])` — closure `(row=a, col=b)`
  element equals old flat layout exactly. ✓

**Verified (gram_on_left branch):**
- Old: `gram[a + b * p] = Σ_t cw[(a,t)]·cw[(b,t)]`.
- New: `DMatrix::from_fn(p, p, |a, b| Σ_t cw[(a,t)]·cw[(b,t)])`. Identical. ✓

FP summation order for the inner `Σ_s` / `Σ_t` is unchanged per element. ✓

### OPT-C — `ssvd` scaled covariance `from_fn` (`fpca_variants.rs`)

**Claim:** Replace `c_scaled` staging Vec + `from_column_slice` with `from_fn`.

**Verified:**
- Old: `c_scaled[row + col * m] = sqrt_w[row] * smooth_cov[(row, col)] * sqrt_w[col]`.
  `DMatrix::from_column_slice(m, m, &c_scaled)` → `mat[(row, col)] = c_scaled[row + col*m]`.
- New: `DMatrix::from_fn(m, m, |row, col| sqrt_w[row] * smooth_cov[(row, col)] * sqrt_w[col])`.
  Element at `(row, col)` is the same expression. ✓

### OPT-D — `functional_acf` `c0_scaled` `from_fn` (`fts/acf.rs`)

**Claim:** Replace `c0_scaled` Vec + `from_column_slice` with `from_fn`, precomputing `sqrt_w`
once.

**Verified:**
- Old inner loop: `c0_scaled[j1 + j2 * m] = c0[j1 + j2 * m] * sqrt_w_j1 * weights[j2].sqrt()`.
  Consumed by `DMatrix::from_column_slice(m, m, &c0_scaled)` → `mat[(j1,j2)] = c0_scaled[j1+j2*m]`.
- New: `DMatrix::from_fn(m, m, |j1, j2| c0[j1 + j2 * m] * sqrt_w[j1] * sqrt_w[j2])`.
  Indexing `c0[j1 + j2 * m]` is the correct column-major access for the existing `c0` flat Vec
  (filled earlier with the same convention). `sqrt_w[j1]` and `sqrt_w[j2]` are the precomputed
  square roots, equivalent to `weights[j1].sqrt()` and `weights[j2].sqrt()`. ✓

### OPT-E — `cov_irreg` kernel-weight precompute (`irreg_fdata/smoothing.rs`)

**Claim:** Move `kernel_gaussian` calls out of the `(si, ti)` grid loop by precomputing
`w_s[obs_idx + si*total_obs]` and `w_t[obs_idx + ti*total_obs]`.

**Key correctness question:** Does the global-index `j1 in start..end` in the new code correctly
index the same data as the old local-index `j1 in 0..obs_t.len()` in `accumulate_cov_at_point`?

**Verified:**
- `IrregFdata`: `argvals` and `values` are flat concatenated Vecs; `offsets[i]..offsets[i+1]`
  gives the global range for observation i.
- Old: `obs_t = &argvals[start..end]`; `obs_t[j1_local]` = `argvals[start + j1_local]`.
  `obs_c[j1_local]` = `centered[start + j1_local]`.
- New precompute: `for (obs_idx, &ot) in ifd.argvals.iter().enumerate()` iterates all global
  indices; `w_s[obs_idx + si*total_obs]` = `kernel_gaussian((argvals[obs_idx] - s) / bw)`.
- New accumulation: `j1 in start..end` — global index. `w_s[j1 + si*total_obs]` =
  `kernel_gaussian((argvals[j1] - s) / bw)` = `kernel_gaussian((obs_t[j1-start] - s) / bw)`. ✓
  `centered[j1]` = `centered[start + j1_local]` = `obs_c[j1_local]`. ✓

**Bounds safety:**
- `w_s` has length `total_obs * ns`; max index accessed = `(end-1) + (ns-1)*total_obs`
  = `total_obs - 1 + (ns-1)*total_obs` = `ns * total_obs - 1` = `w_s.len() - 1`. ✓
- `centered` has length `ifd.values.len()` = `total_obs`; `j1 < end <= total_obs`. ✓

**Accumulation order:** Within each `(si, ti)` cell, the `i → j1 → j2` loop nesting and FP
summation order are identical to the old `accumulate_cov_at_point`. ✓

`sum_weights > 0.0` guard is preserved. ✓

### OPT-F — `fem_smooth` single-pass Φ'Φ / `a_mat` (`fem_smoothing.rs`)

**Claim:** Build `phi_t_phi` (pure Φ'Φ) and `a_mat` simultaneously in a single loop, removing the
`phi_t_phi.clone()` that preceded the regularization additions.

**Verified:**
- Both `phi_t_phi` and `a_mat` are zero-initialized and then filled with the same assembly loop
  using identical index arithmetic (`a * big_n + b` and `b * big_n + a`).
- After the loop, `a_mat` receives `+ lambda * k_global[ab]` and `+ 1e-10` diagonal ridge —
  additions that in the old code were applied to the clone only.
- `phi_t_phi` is never modified after the assembly loop and is read at line 596 for the GCV
  trace `edf = tr(A^{-1} · Φ'Φ)`. It remains pure Φ'Φ throughout. ✓
- No aliasing: `phi_t_phi` and `a_mat` are separate allocations. ✓

---

## Build and Lint Results

```
cargo build -p fdars-core --features linalg,parallel
  → Finished `dev` profile [unoptimized + debuginfo]  (0 errors, 0 warnings)

cargo clippy -p fdars-core --all-targets --features linalg,parallel -- -D warnings
  → Finished `dev` profile  (0 errors, 0 warnings)
```

---

_Reviewed: 2026-08-31T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
