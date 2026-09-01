# Phase 47: Hot-Path & Allocation Performance — Research

**Researched:** 2026-08-30
**Domain:** Rust performance optimization — allocation reduction, constant-factor compute wins, FEM linear algebra, functional time series spectral estimation
**Confidence:** HIGH (source files read directly this session)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Optimize the highest-leverage subset with clear safe wins, not all top-10:
  - `fts::dpca` — allocation hotspot #1 (42 MB / 8.6 MB peak / 17,739 blocks @ n200_m50, `src/fts/spectral.rs:203`). PERF-02 primary.
  - `irreg_fdata::face_covariance` — compute hotspot #1 (984 ms @ n200_m30, `src/irreg_fdata/face.rs:128`).
  - `fem_smoothing::fem_smooth` — compute #2 (452 ms @ 576 nodes, `src/fem_smoothing.rs:475`).
  - Opportunistic FdMatrix↔DMatrix copy removals surfaced by PROF-01 (fsvd `:488`, ssvd `:740`, long_run_covariance `acf.rs:337`) where mechanical and safe.
- "Measurable improvement" bar: ≥15% wall-time (non-overlapping criterion CIs) OR ≥25% allocation reduction.
- A target with no safe behavior-preserving win is documented + deferred — do not force a risky rewrite.
- Attack order: allocation-reduction first (mechanical, low-risk), then compute paths (algorithmic, higher-risk).
- Tolerance: exact for counting/integer paths; relative ≤1e-10 for float SVD/eigen paths, documented per change.
- Add permanent `#[test]` equivalence/golden tests.
- The existing full suite must stay green at every commit.
- PERF-02 allocation proof: re-add a committed feature-gated `dhat-heap` alloc-audit test (mirror `tests/alloc_audit_fpca.rs`) showing before→after fewer/smaller allocations.
- Register the PERF-proof benches permanently now (`[[bench]]`) for the optimized paths.
- Record before/after numbers in the phase SUMMARY + a `PERF-RESULTS.md`.
- Capture governor + `RAYON_NUM_THREADS` for every before/after; pin the `performance` governor if permitted, else note the `powersave` LOW-CONFIDENCE caveat (v0.14.0).
- No public signature changes; keep `linalg`/non-`linalg` branches producing equivalent results (SC3).

### Claude's Discretion
- None specified — all targets and approach are locked.

### Deferred Ideas (OUT OF SCOPE)
- Parallelism (feature-gated rayon) for these hot paths → Phase 48 (PERF-03).
- Documenting the benches as formal regression guards with a before/after table → Phase 51 (BENCH-02).
- Any target with no safe behavior-preserving win → deferred with a documented rationale.
- Breaking/asymptotic rewrites of inherently O(n·m²) paths → out of scope (behavior-preserving only).
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| PERF-01 | Each top-ranked hot path from PROF-01 is optimized behavior-preservingly, proven by a before/after criterion benchmark showing measurable improvement (≥15% wall-time), with existing tests green (numeric outputs unchanged or provably-equivalent within documented tolerance). | face_covariance and fem_smooth optimizations below; cov_irreg inner loop is the specific target; fem_smooth GCV A⁻¹ column loop is deferrable if no safe win. |
| PERF-02 | Allocation hotspots identified by PROF-01 (unnecessary FdMatrix↔DMatrix copies, per-iteration allocations in hot loops) are reduced, verified by an allocation profile (feature-gated `dhat-heap`) and equivalence tests. | dpca spectral.rs:203 allocation analysis below; fsvd/ssvd/long_run_covariance copy removal analysis below. |
</phase_requirements>

---

## Summary

This research covers the five concrete optimization targets from PROF-01 for Phase 47. The source files have been read directly to produce specific anchor-level proposals rather than general advice.

The clearest win is **`fts::dpca` allocation reduction** (OPT-A): `eigen_at_frequency` at `src/fts/spectral.rs:203` is called once per Fourier frequency (`n_freq = n` times, typically 200). Each call allocates a fresh `DMatrix::from_column_slice(m, m, &scaled)` of size `m² × 8` bytes, then `SymmetricEigen` allocates intermediate eigenvector/value storage internally. This explains the 17,739-block count: approximately `n` calls × ~89 allocs per eigen call at m=50. The fix is mechanical: hoist the `DMatrix` construction out of the per-frequency loop by reusing a pre-allocated buffer, mutating it in-place before each eigendecomposition.

The **FdMatrix↔DMatrix copy removal** targets (OPT-B, OPT-C, OPT-D) are all smaller wins (6–275 blocks) but are also mechanical and low-risk. The pattern is identical in each case: a `Vec<f64>` is filled, then `DMatrix::from_column_slice` copies it into a nalgebra heap allocation. In several cases the `Vec` intermediate can be dropped.

**`irreg_fdata::face_covariance`** (OPT-E) is dominated by `cov_irreg` (`src/irreg_fdata/smoothing.rs:111`) which iterates over an `(ns × nt)` grid and for each point runs a nested loop over all `n` observations' observation-pairs. The kernel weight for each observation point-pair is computed redundantly across `ns × nt` evaluations. Precomputing per-observation kernel weights once per grid point is the clear win: instead of computing `kernel_gaussian((obs_t[j1] - s) / bandwidth)` inside the `(si, ti)` loop, precompute an `n_total_obs × ns` weight matrix and an `n_total_obs × nt` weight matrix once each. This is a constant-factor improvement of `O(ns*nt × n_total_obs_pairs) → O((ns + nt) × n_total_obs_pairs + ns*nt × n_total_obs_pairs)` — i.e. it eliminates duplicate `exp()` evaluations across grid columns. Given `exp()` is the dominant FLOPs, this is the primary constant-factor win. The architecture of `face_covariance` → `cov_irreg` → `gaussian_smooth_cov` → `psd_project` can be profiled without restructuring any public API.

**`fem_smoothing::fem_smooth`** (OPT-F) is dominated by two factors: (1) `assemble_fem_matrices` + `fem_basis_eval` are called once and produce dense N×N matrices — no redundant recomputation exists; (2) the `A⁻¹` column-by-column computation for GCV (`big_n` Cholesky solves, each O(N²)) is inherently O(N³) and is the structural bottleneck at large node counts. The only safe constant-factor win is to eliminate the `phi_t_phi.clone()` call that copies the entire N×N matrix to construct `a_mat` — instead building `a_mat` directly. This is a small memory win (saves one N×N allocation) but does not meaningfully reduce the O(N³) solve cost. The primary cost is the linear algebra, not allocation, so this target is a **partial defer** — allocations can be trimmed trivially, but the computational bottleneck requires accepting the O(N³) cost or deferring to a sparse solver (out of scope).

**Primary recommendation:** Attack in order OPT-A (dpca alloc churn) → OPT-B/C/D (FdMatrix copy removals) → OPT-E (face_covariance kernel precompute) → OPT-F (fem_smooth clone removal, accept O(N³) limitation and DEFER further compute work).

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Allocation reduction (`fts::dpca`) | `src/fts/spectral.rs` — `eigen_at_frequency` fn | `dpca` fn that calls it | All 17,739 blocks originate in this helper; caller loop is the fix site |
| FdMatrix↔DMatrix copy (`fsvd`) | `src/fpca_variants.rs:488` | `cross_covariance` helper | `DMatrix::from_column_slice(g_dim, g_dim, &gram)` — gram already exists as Vec |
| FdMatrix↔DMatrix copy (`ssvd`) | `src/fpca_variants.rs:740` | `gaussian_smooth_cov` (already FdMatrix) | `DMatrix::from_column_slice(m, m, &c_scaled)` — c_scaled is a Vec |
| FdMatrix↔DMatrix copy (`long_run_cov`) | `src/fts/acf.rs:337` | `functional_acf` entry | `DMatrix::from_column_slice(m, m, &c0_scaled)` — c0_scaled is a Vec |
| Compute reduction (`face_covariance`) | `src/irreg_fdata/smoothing.rs:111` — `cov_irreg` | `face_covariance` entry, `gaussian_smooth_cov` | Inner `accumulate_cov_at_point` loop computes `exp()` redundantly |
| Compute reduction (`fem_smooth`) | `src/fem_smoothing.rs:540-541` — `phi_t_phi.clone()` | Cholesky GCV loop `:568-578` | Clone is the mechanical win; GCV loop is O(N³) structural cost |

---

## Optimization Catalogue

Each optimization is specified with: anchor, before-number, proposed change, expected win, risk level, proof method.

---

### OPT-A: `fts::dpca` — Hoist DMatrix construction out of per-frequency loop

**Anchor:** `src/fts/spectral.rs:203` — `eigen_at_frequency` function body
**Before-number:** 17,739 blocks / 42,084,568 bytes total / 8,637,712 bytes peak @ n200_m50
**PROF-01 rank:** Allocation #1 (dwarfs others by ~70× total bytes)

**Root cause (read this session):** `eigen_at_frequency` is called in a `for k in 0..n_freq` loop inside `dpca` (`spectral.rs:299`). `n_freq = n = 200` at the measurement cell. Each call:
1. Fills a local `scaled: Vec<f64>` (m×m = 2,500 f64 = 20 KB) — stack-like, but heap-allocated.
2. Calls `DMatrix::from_column_slice(m, m, &scaled)` at line 203 — copies 20 KB into a new nalgebra heap allocation.
3. Calls `nalgebra::SymmetricEigen::new(mat)` — internally allocates eigenvector matrix + work buffers.
4. Collects `eig.eigenvalues.iter()` and `eig.eigenvectors.column_iter()` into new `Vec<(f64, Vec<f64>)>` — another set of allocations.

At n=200 (200 calls), even 5 heap allocations per call × 200 = 1,000 allocations. The measured 17,739 = ~89 per call — consistent with nalgebra's internal LAPACK-style eigen work buffers at m=50.

**Proposed change (mechanical, safe):**

The `scaled` Vec and the `DMatrix` wrapper can be pre-allocated once and reused:

```rust
// In dpca(), before the k loop:
let mut scaled_buf = vec![0.0f64; m * m];
let mut pairs_buf: Vec<(f64, Vec<f64>)> = Vec::with_capacity(m);

for k in 0..n_freq {
    let (vals, vecs) = eigen_at_frequency_inplace(&sd.re[k], m, ncomp, &sqrt_w,
                                                   &mut scaled_buf, &mut pairs_buf);
    ...
}
```

Concretely: add a `scaled_buf: &mut Vec<f64>` parameter to `eigen_at_frequency`, fill it in-place (no new allocation), then construct the `DMatrix` from the mutable slice. The `DMatrix` itself still allocates on each call (nalgebra does not provide a way to give it a buffer), but eliminating the intermediate `Vec<f64>` halves one allocation site.

The more impactful change is to eliminate the `Vec<(f64, Vec<f64>)>` allocation by sorting a pre-allocated index array instead:

```rust
// Instead of:
let mut pairs: Vec<(f64, Vec<f64>)> = eig.eigenvalues.iter()
    .zip(eig.eigenvectors.column_iter())
    .map(|(&val, col)| (val, col.iter().copied().collect()))
    .collect();
pairs.sort_by(...);

// Use:
let mut idx: Vec<usize> = (0..m).collect(); // pre-allocated once
idx.sort_by(|&a, &b| eig.eigenvalues[b].partial_cmp(&eig.eigenvalues[a])...);
// Read eigenvectors directly by index without copying to Vec<f64>
```

The `col.iter().copied().collect()` in the map creates `m` Vec<f64> allocations (one per eigenvalue) — at m=50 and 200 calls: 200 × 50 = 10,000 allocations, which closely matches the observed count.

**Full fix:**
1. Pre-allocate `idx: Vec<usize>` once in `dpca()` and pass to a refactored `eigen_at_frequency`.
2. Sort `idx` in-place each call (no allocation).
3. Read eigenvectors from `eig.eigenvectors` by column index without collecting to `Vec<Vec<f64>>`.
4. The `pairs_buf` intermediate disappears entirely.

This eliminates the O(m × n_freq) Vec allocations that dominate the block count.

**Expected win:** The 10,000+ `Vec<f64>` eigenvector copy allocations (dominant contribution to 17,739 blocks) are eliminated. Target: ≥25% allocation reduction (likely much more — down to ~200 blocks for the DMatrix per call). Wall-time improvement from fewer allocator pressure events.
**Risk:** LOW — mechanical refactor, no numeric change. The eigenvalues/vectors from `SymmetricEigen` are read in the same order as before; only the intermediate copy is removed.
**Equivalence proof:** Golden test that captures `dpca` output at a fixed seed before and after; asserts `max_abs_diff < 1e-12` on filters, scores, eigenvalues.
**Proof of allocation reduction:** `dhat-heap` test in `tests/alloc_audit_dpca.rs` (mirror of `tests/alloc_audit_fpca.rs`), records before/after block count.

[VERIFIED: src/fts/spectral.rs:194-237] — `eigen_at_frequency` body; the `col.iter().copied().collect()` at line 215 is the dominant allocation site. Verbatim context:
```rust
// line 212–217 (verbatim):
    let mut pairs: Vec<(f64, Vec<f64>)> = eig
        .eigenvalues
        .iter()
        .zip(eig.eigenvectors.column_iter())
        .map(|(&val, col)| (val, col.iter().copied().collect()))
        .collect();
```

---

### OPT-B: `fpca_variants::fsvd` — Eliminate `DMatrix::from_column_slice` copy of gram

**Anchor:** `src/fpca_variants.rs:488` — line `let eigen = DMatrix::from_column_slice(g_dim, g_dim, &gram).symmetric_eigen();`
**Before-number:** 275 blocks / 600,049 bytes total / 410,880 bytes peak @ n200_m50
**PROF-01 rank:** Allocation #2

**Root cause (read this session):** `fsvd` builds `gram: Vec<f64>` (g_dim×g_dim, where g_dim = min(p,q)) in a nested loop at lines 474–487. Then it immediately calls `DMatrix::from_column_slice(g_dim, g_dim, &gram)` which **copies** the entire Vec into a DMatrix heap allocation. The Vec `gram` is then dropped — it was only a staging buffer for the DMatrix.

[VERIFIED: src/fpca_variants.rs:471-488] Verbatim:
```rust
    let mut gram = vec![0.0_f64; g_dim * g_dim];
    if gram_on_right {
        for a in 0..q {
            for b in 0..q {
                gram[a + b * q] = (0..p).map(|s| cw[(s, a)] * cw[(s, b)]).sum();
            }
        }
    } else {
        for a in 0..p {
            for b in 0..p {
                gram[a + b * p] = (0..q).map(|t| cw[(a, t)] * cw[(b, t)]).sum();
            }
        }
    }
    let eigen = DMatrix::from_column_slice(g_dim, g_dim, &gram).symmetric_eigen();
```

**Proposed change:** Replace `vec![0.0_f64; g_dim * g_dim]` + fill + `from_column_slice` with constructing the `DMatrix` directly via `DMatrix::from_fn(g_dim, g_dim, |a, b| ...)`. This eliminates the intermediate Vec allocation:

```rust
let gram_mat = if gram_on_right {
    DMatrix::from_fn(q, q, |a, b| (0..p).map(|s| cw[(s, a)] * cw[(s, b)]).sum())
} else {
    DMatrix::from_fn(p, p, |a, b| (0..q).map(|t| cw[(a, t)] * cw[(b, t)]).sum())
};
let eigen = gram_mat.symmetric_eigen();
```

`DMatrix::from_fn` constructs the matrix in-place — no intermediate Vec. [ASSUMED] — nalgebra's `from_fn` constructs directly, but internal double-buffering cannot be confirmed without reading nalgebra source. Risk: the optimization saves the intermediate Vec; the DMatrix allocation itself remains.

**Expected win:** Eliminate the staging `gram` Vec allocation (~g_dim²×8 bytes). At g_dim=50 that is 20 KB per call. Not a large percentage of the 600 KB total, but the change is also the safest possible.
**Risk:** LOW — algebraically identical, just removes a copy.
**Equivalence proof:** The existing `fsvd` tests in `fpca_variants.rs` cover numerical correctness. Add a golden test with a fixed input (n=20, p=15, q=10) asserting `singular_values`, `left_functions`, `right_functions` match within 1e-12.

---

### OPT-C: `fpca_variants::ssvd` — Eliminate `DMatrix::from_column_slice` copy of c_scaled

**Anchor:** `src/fpca_variants.rs:740` — `let eigen = DMatrix::from_column_slice(m, m, &c_scaled).symmetric_eigen();`
**Before-number:** 22 blocks / 314,416 bytes total / 182,384 bytes peak @ n200_m50
**PROF-01 rank:** Allocation #3

**Root cause (read this session):** `ssvd` fills `c_scaled: Vec<f64>` (m×m) at lines 734–738, then calls `DMatrix::from_column_slice(m, m, &c_scaled)` — again a copy into DMatrix.

[VERIFIED: src/fpca_variants.rs:733-740] Verbatim:
```rust
    let mut c_scaled = vec![0.0_f64; m * m];
    for col in 0..m {
        for row in 0..m {
            c_scaled[row + col * m] = sqrt_w[row] * smooth_cov[(row, col)] * sqrt_w[col];
        }
    }
    let eigen = DMatrix::from_column_slice(m, m, &c_scaled).symmetric_eigen();
```

**Proposed change:** Same pattern as OPT-B:

```rust
let eigen = DMatrix::from_fn(m, m, |row, col| {
    sqrt_w[row] * smooth_cov[(row, col)] * sqrt_w[col]
}).symmetric_eigen();
```

This is even more straightforward than OPT-B because `smooth_cov` is already an `FdMatrix` (indexable directly).

**Expected win:** Eliminate the `c_scaled` Vec (m×m×8 bytes, ~20 KB at m=50). Small block count, but the pattern elimination improves code clarity.
**Risk:** LOW — identical numerics, just removes copy.
**Equivalence proof:** Existing `ssvd` tests + golden test on fixed input.

---

### OPT-D: `fts::functional_acf` / `long_run_covariance` — Eliminate `DMatrix::from_column_slice` of c0_scaled

**Anchor:** `src/fts/acf.rs:337` — `let mut c0_mat = DMatrix::from_column_slice(m, m, &c0_scaled);`
**Before-number:** 6 blocks / 100,400 bytes total / 40,400 bytes peak @ n200_m50
**PROF-01 rank:** Allocation #4

**Root cause (read this session):** `functional_acf` fills `c0_scaled: Vec<f64>` (m×m) at lines 329–334, then at line 337 calls `DMatrix::from_column_slice` — another staging Vec → DMatrix copy.

[VERIFIED: src/fts/acf.rs:329-337] Verbatim:
```rust
    let mut c0_scaled = vec![0.0f64; m * m];
    for j1 in 0..m {
        let sw1 = weights[j1].sqrt();
        for j2 in 0..m {
            c0_scaled[j1 + j2 * m] = c0[j1 + j2 * m] * sw1 * weights[j2].sqrt();
        }
    }
    // Symmetrise defensively (should already be symmetric up to fp noise).
    let mut c0_mat = DMatrix::from_column_slice(m, m, &c0_scaled);
```

**Note:** `weights[j2].sqrt()` is recomputed on every (j1, j2) pair. Precomputing `sqrt_w: Vec<f64>` saves m×m sqrt calls.

**Proposed change:**

```rust
let sqrt_w: Vec<f64> = weights.iter().map(|w| w.sqrt()).collect();
let mut c0_mat = DMatrix::from_fn(m, m, |j1, j2| c0[j1 + j2 * m] * sqrt_w[j1] * sqrt_w[j2]);
// Symmetrise:
for j1 in 0..m {
    for j2 in (j1 + 1)..m {
        let avg = 0.5 * (c0_mat[(j1, j2)] + c0_mat[(j2, j1)]);
        c0_mat[(j1, j2)] = avg;
        c0_mat[(j2, j1)] = avg;
    }
}
```

This eliminates the `c0_scaled` Vec AND avoids m² sqrt calls (saves m×m - m sqrt evaluations).
**Note:** `long_run_covariance` itself (`acf.rs:676–730`) does not call `from_column_slice` — it accumulates into a plain `Vec<f64>`. The 6-block allocation count for `long_run_covariance` measurement is the `autocovariance_matrix` helper's per-lag Vec allocations (each returns a new `Vec<f64>`). Those are load-bearing (the caller accumulates into `acc`). No change needed there.

**Expected win:** Eliminate `c0_scaled` Vec + save m²-m sqrt calls. Small total (6 blocks, 100 KB), but the sqrt precompute is a correctness-quality improvement too.
**Risk:** LOW — algebraically identical, sqrt precompute is a strict constant-factor improvement.
**Equivalence proof:** Existing `functional_acf` tests cover numerical correctness. The symmetrization result is unchanged.

---

### OPT-E: `irreg_fdata::face_covariance` — Precompute kernel weights in `cov_irreg`

**Anchor:** `src/irreg_fdata/smoothing.rs:111` — `cov_irreg` and `accumulate_cov_at_point`
**Before-number:** 984 ms @ n200_m30, 242 ms @ n50_m30 (wall-time, PROF-01 rank #1)
**PROF-01 rank:** Compute #1

**Root cause (read this session):** `cov_irreg` calls `accumulate_cov_at_point` for every `(si, ti)` pair in an `(ns × nt)` grid (ns×nt = 30×30 = 900 calls at the m30 cell). Each call runs nested loops over all observations' observation-point pairs, computing:

```rust
let w1 = kernel_gaussian((obs_t[j1] - s) / bandwidth);
let w2 = kernel_gaussian((obs_t[j2] - t) / bandwidth);
```

[VERIFIED: src/irreg_fdata/smoothing.rs:159-166] Verbatim:
```rust
        for j1 in 0..obs_t.len() {
            for j2 in 0..obs_t.len() {
                let w1 = kernel_gaussian((obs_t[j1] - s) / bandwidth);
                let w2 = kernel_gaussian((obs_t[j2] - t) / bandwidth);
                let w = w1 * w2;
                sum_weights += w;
                sum_products += w * obs_c[j1] * obs_c[j2];
            }
        }
```

`kernel_gaussian` calls `(-0.5 * u * u).exp()` which is an expensive transcendental operation. The key inefficiency: `w1 = kernel_gaussian((obs_t[j1] - s) / bandwidth)` depends on `s` (the column of the grid) but not on `t` (the row). Conversely `w2` depends on `t` but not on `s`. So across the `ns × nt` grid, `w1` for a given `(i, j1, si)` combination is recomputed `nt` times (once per `ti`), and `w2` for `(i, j2, ti)` is recomputed `ns` times.

**Proposed change — precompute kernel weight arrays:**

For each grid column `si` and each observation point `(i, j1)`, precompute `w1[i][j1] = kernel_gaussian(...)` once. Similarly for `ti` and `(i, j2)`. Then the inner loop becomes a multiply:

```rust
// Precompute: w_s[total_obs_idx][si] = kernel_gaussian((obs_t[obs_idx] - s_grid[si]) / bw)
// Precompute: w_t[total_obs_idx][ti] = kernel_gaussian((obs_t[obs_idx] - t_grid[ti]) / bw)
// Shape: w_s is (total_obs_points × ns), w_t is (total_obs_points × nt)
// These are computed once, then the (si, ti) loop uses table lookups instead of exp().

let total_obs = ifd.argvals.len(); // total observation points across all curves
let mut w_s = vec![0.0f64; total_obs * ns];
let mut w_t = vec![0.0f64; total_obs * nt];
for (obs_idx, &obs_t_val) in ifd.argvals.iter().enumerate() {
    for (si, &s) in s_grid.iter().enumerate() {
        w_s[obs_idx + si * total_obs] = kernel_gaussian((obs_t_val - s) / bandwidth);
    }
    for (ti, &t) in t_grid.iter().enumerate() {
        w_t[obs_idx + ti * total_obs] = kernel_gaussian((obs_t_val - t) / bandwidth);
    }
}
```

The `(si, ti)` loop then becomes:

```rust
for (si, _) in s_grid.iter().enumerate() {
    for (ti, _) in t_grid.iter().enumerate() {
        let mut sum_w = 0.0;
        let mut sum_p = 0.0;
        for i in 0..n {
            let (ps, pe) = (offsets[i], offsets[i + 1]);
            for j1 in ps..pe {
                let w1 = w_s[j1 + si * total_obs];
                if w1 < 1e-300 { continue; } // early exit if kernel is effectively zero
                for j2 in ps..pe {
                    let w2 = w_t[j2 + ti * total_obs];
                    let w = w1 * w2;
                    sum_w += w;
                    sum_p += w * centered[j1] * centered[j2];
                }
            }
        }
        cov[si + ti * ns] = if sum_w > 0.0 { sum_p / sum_w } else { 0.0 };
    }
}
```

**Expected win:** The number of `exp()` calls drops from `ns × nt × total_obs_points_per_curve × n_obs` (all four dimensions) to `(ns + nt) × total_obs_points` (two precompute passes). At n=200, m=30, with ~3 obs points per curve → total_obs = 600, the reduction is:
- Before: 900 grid cells × 200 curves × ~9 obs-pairs = ~1,620,000 `exp()` calls
- After precompute: 600 × (30 + 30) = 36,000 `exp()` calls
- Savings: ~98% reduction in `exp()` evaluations
- Wall-time gain: the computation is dominated by `exp()`; a 40-80% wall-time reduction is plausible, targeting ≥15%.

**Risk:** MEDIUM — restructuring `cov_irreg` internals, but the computation is algebraically identical (w1*w2 factoring is exact). The public API of `cov_irreg` and `face_covariance` is unchanged; only internal loop order changes.

**Allocation:** Introduces two pre-allocated `Vec<f64>` at function entry (not per-call allocations). Net effect on allocation count is slightly negative (slightly more total memory, but far fewer calls to `exp()` which drives compute, not allocation).

**Equivalence proof:** Golden test: call `face_covariance` with the same `IrregFdata` + grid before and after; assert all elements match within 1e-12. The existing `test_face_covariance_dense_limit` test in `face.rs` also provides a statistical correctness guard.

**Implementation note:** The `accumulate_cov_at_point` private function must be replaced with this new restructured form. Since it is `fn` (not `pub`), this is internal only — no API change.

---

### OPT-F: `fem_smoothing::fem_smooth` — Eliminate `phi_t_phi.clone()` (partial win; accept O(N³) cost)

**Anchor:** `src/fem_smoothing.rs:541` — `let mut a_mat = phi_t_phi.clone();`
**Before-number:** 452 ms @ 576 nodes (PROF-01 rank #2); no allocation measurement taken
**PROF-01 rank:** Compute #2

**Root cause (read this session):** `fem_smooth` builds `phi_t_phi` (N×N flat Vec) at lines 523–538, then at line 541 clones it to create `a_mat`. Then adds `lambda * k_global + epsilon*I` in-place to `a_mat`.

[VERIFIED: src/fem_smoothing.rs:540-541] Verbatim:
```rust
    let mut a_mat = phi_t_phi.clone();
    for ab in 0..(big_n * big_n) {
        a_mat[ab] += lambda * k_global[ab];
    }
```

The clone is needed because `phi_t_phi` is later used in the GCV trace computation at lines 581–584:
```rust
    for a in 0..big_n {
        for b in 0..big_n {
            edf += a_inv[a * big_n + b] * phi_t_phi[b * big_n + a];
        }
    }
```

So the clone IS load-bearing (both `phi_t_phi` and `a_mat` are needed). However, we can **build `a_mat` directly** without going through `phi_t_phi` as an intermediate, if we build `phi_t_phi` into `a_mat` from the start and keep a second copy only for the GCV trace. Actually the cleanest approach is: build `a_mat = phi_t_phi + lambda*K + eps*I` directly in one pass, and separately keep `phi_t_phi` for the GCV trace. This saves one copy of the N×N matrix (saving ~N²×8 bytes, ~2.6 MB at N=576).

**Proposed change:**
```rust
// Build phi_t_phi AND a_mat in the SAME pass:
let mut phi_t_phi = vec![0.0_f64; big_n * big_n];
let mut a_mat     = vec![0.0_f64; big_n * big_n]; // = phi_t_phi + lambda*K + eps*I
for i in 0..n_obs {
    for a in 0..big_n {
        let phi_ia = phi[i * big_n + a];
        if phi_ia == 0.0 { continue; }
        for b in a..big_n {
            let val = phi_ia * phi[i * big_n + b];
            phi_t_phi[a * big_n + b] += val;
            a_mat[a * big_n + b]     += val;
            if a != b {
                phi_t_phi[b * big_n + a] += val;
                a_mat[b * big_n + a]     += val;
            }
        }
    }
}
// Add lambda*K and eps*I to a_mat (but NOT phi_t_phi):
for ab in 0..(big_n * big_n) {
    a_mat[ab] += lambda * k_global[ab];
}
for a in 0..big_n {
    a_mat[a * big_n + a] += 1e-10;
}
```

This builds both in one pass instead of building `phi_t_phi` then cloning. The loop cost is identical (same number of FLOPs); the savings is one O(N²) allocation + memcopy.

**Structural bottleneck — DEFER:** The 452 ms wall time is dominated by:
1. `cholesky_factor(&a_mat, big_n)` — O(N³) — unavoidable with a dense solver.
2. The `A⁻¹` column-by-column computation (`big_n` `cholesky_forward_back` calls, each O(N²)) — O(N³) total.

These are inherently O(N³) with dense matrices. No behavior-preserving constant-factor win exists here without:
- Sparse assembly (requires a new crate dependency or significant hand-rolled sparse structure — out of scope).
- Skipping the GCV computation entirely (would change the returned `edf`/`gcv` fields — API change, out of scope).

**Verdict:** The `phi_t_phi.clone()` elimination is the only safe constant-factor win in `fem_smooth`. The primary O(N³) cost must be **deferred** with documented rationale.

**Expected win from clone removal:** Saves one N×N allocation (~2.6 MB at N=576) + one O(N²) copy. Wall-time improvement is negligible vs the O(N³) solve (~0.1% of 452 ms). The win is meaningful only for memory pressure, not compute.

**Risk:** LOW — algebraically identical.
**Equivalence proof:** Existing `fem_smooth` tests (assertions on `node_values`, `fitted_obs`, `edf`, `gcv`, `rss`) cover correctness. Add a golden test on the 576-node input asserting all outputs match within 1e-12.

**DEFER note:** The O(N³) compute bottleneck in `fem_smooth` (Cholesky + GCV column solve) has no safe constant-factor win without introducing sparse solvers or skipping GCV computation. Document and defer per CONTEXT.md policy.

---

## Ranked Implementation Sequence

Ranked by: allocation-first, low-risk-first, then compute wins.

| Priority | Opt ID | Target | Type | Risk | Expected Win | Proof |
|----------|--------|--------|------|------|--------------|-------|
| 1 | OPT-A | `fts::dpca` eigen_at_frequency | Allocation | LOW | ≥25% alloc reduction (eliminates ~10,000 Vec allocs per dpca call) | dhat alloc-audit test + golden |
| 2 | OPT-C | `fpca_variants::ssvd` c_scaled | Allocation | LOW | Eliminate 1 Vec copy (m×m) per call | Golden test |
| 3 | OPT-D | `fts::functional_acf` c0_scaled + sqrt precompute | Allocation + micro | LOW | Eliminate 1 Vec copy + save m² sqrt calls | Golden test |
| 4 | OPT-B | `fpca_variants::fsvd` gram Vec | Allocation | LOW | Eliminate 1 Vec copy (g_dim×g_dim) per call | Golden test |
| 5 | OPT-E | `irreg_fdata::cov_irreg` kernel precompute | Compute | MEDIUM | ≥15% wall-time (up to 80% `exp()` reduction) | Criterion bench before/after + golden |
| 6 | OPT-F | `fem_smoothing::fem_smooth` clone removal | Allocation | LOW | Memory only (negligible wall-time) | Golden test |
| DEFER | — | `fem_smooth` O(N³) Cholesky/GCV | Compute | HIGH (sparse) | — | Documented rationale |

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Eigendecomposition of symmetric matrix | Custom Jacobi/Lanczos | `nalgebra::SymmetricEigen` | Edge cases (near-degenerate, complex eigenvalues from rounding) |
| Cholesky factorization | Custom Cholesky | `crate::linalg::cholesky_factor` (already exists) | Already validated, reused across codebase |
| Index sorting without data copy | Custom in-place sort with data movement | `sort_by` on an index array, read data by index | Avoids O(m) allocation per eigenvalue while maintaining sort |
| Pre-allocated DMatrix | DMatrix::from_fn with closure that re-executes computation | DMatrix::from_fn (already allocates once) | Clean API, no intermediate Vec needed |

---

## Common Pitfalls

### Pitfall 1: Mutating `phi_t_phi` and using it as `a_mat`
**What goes wrong:** If you build `a_mat` by mutating `phi_t_phi` in-place (adding K + eps), then `phi_t_phi` no longer holds the pure `Φ'Φ` needed for the GCV trace.
**Why it happens:** The two matrices share the same values before adding the regularization terms.
**How to avoid:** Build both from the start (OPT-F proposal above), or clone at the GCV step (reverting the optimization). The double-pass approach in OPT-F is the clean solution.
**Warning signs:** `edf` diverges from the reference implementation at non-zero lambda.

### Pitfall 2: Index sort on `eig.eigenvalues` vs eigenvector alignment
**What goes wrong:** After sorting an index array `idx` by descending eigenvalue, you must read `eig.eigenvectors.column(idx[c])` — not `eig.eigenvectors.column(c)`. Mixing the two produces wrong eigenvectors for the wrong eigenvalues.
**Why it happens:** nalgebra returns eigenvalues in arbitrary order; sorting creates a mapping that must be applied consistently.
**How to avoid:** Write the index-sort refactor as a single coherent block with the mapping applied uniformly. The sign-align loop must also use `idx[c]` for the eigenvector column.
**Warning signs:** The golden test fails with large discrepancies on specific eigenvalue modes (first or last component wrong).

### Pitfall 3: `from_fn` vs `from_column_slice` index convention
**What goes wrong:** `DMatrix::from_fn(m, m, |row, col| ...)` fills in (row, col) order, which is what nalgebra expects. But `DMatrix::from_column_slice(m, m, &slice)` expects column-major: `slice[row + col*m]`. If you write a closure using `slice[row + col*m]` it is correct; if you write `slice[col + row*m]` (row-major) the matrix is transposed.
**Why it happens:** `FdMatrix` is column-major (`data[i + j * nrows]`); nalgebra `DMatrix` is also column-major — they agree. But vanilla `Vec<f64>` used as a staging buffer may have been filled row-major in some paths.
**How to avoid:** Verify index convention in the existing fill loop before switching to `from_fn`. For OPT-C (`c_scaled`), the fill is `c_scaled[row + col * m]` (column-major) — consistent with nalgebra.
**Warning signs:** Eigenvectors come out transposed; numerical test fails with relative error near 1.0.

### Pitfall 4: `cpupower` requires sudo for governor pinning
**What goes wrong:** `cpupower frequency-set -g performance` silently fails without sudo, and the governor stays at `powersave`. Benchmark numbers appear unstable and lower than they should be.
**Why it happens:** Writing to `/sys/devices/system/cpu/cpuX/cpufreq/scaling_governor` requires root.
**How to avoid:** Run `sudo cpupower frequency-set -g performance` before benchmarking, restore with `sudo cpupower frequency-set -g powersave` after. If sudo is unavailable, note `powersave` caveat in PERF-RESULTS.md as a LOW-CONFIDENCE qualifier on the delta.
**Warning signs:** Criterion reports overlapping confidence intervals that should be clearly separated; high variance across samples.

### Pitfall 5: dhat requires `--test-threads=1` and a separate process
**What goes wrong:** Running `cargo test --features dhat-heap` without `-- --test-threads=1 --nocapture` allows multiple test processes sharing the global allocator, producing wrong counts.
**Why it happens:** dhat sets a global allocator; multiple tests in one binary contaminate each other.
**How to avoid:** Always run the dhat audit test as its own binary with `-- <test_name> --nocapture --test-threads=1`. This is documented in `tests/alloc_audit_fpca.rs` but must be applied to the new `tests/alloc_audit_dpca.rs` file too.
**Warning signs:** Block counts vary across runs; counts appear far too high (counting allocations from other tests).

---

## Code Examples

### Example 1: Index-sort eigendecomposition without Vec<(f64, Vec<f64>)> allocation

```rust
// Source: derived from spectral.rs eigen_at_frequency; eliminates per-eigenvalue Vec copies
fn eigen_at_frequency_v2(
    spec_real: &[f64],
    m: usize,
    ncomp: usize,
    sqrt_w: &[f64],
    idx_buf: &mut Vec<usize>,   // pre-allocated, length m
) -> (Vec<f64>, Vec<Vec<f64>>) {
    // Build scaled matrix directly via from_fn (no intermediate Vec)
    let mut mat = DMatrix::from_fn(m, m, |j1, j2| {
        spec_real[j1 + j2 * m] * sqrt_w[j1] * sqrt_w[j2]
    });
    // Symmetrize defensively
    for j1 in 0..m {
        for j2 in (j1 + 1)..m {
            let avg = 0.5 * (mat[(j1, j2)] + mat[(j2, j1)]);
            mat[(j1, j2)] = avg;
            mat[(j2, j1)] = avg;
        }
    }
    let eig = nalgebra::SymmetricEigen::new(mat);

    // Sort by descending eigenvalue using a pre-allocated index buffer
    idx_buf.clear();
    idx_buf.extend(0..m);
    idx_buf.sort_by(|&a, &b| eig.eigenvalues[b]
        .partial_cmp(&eig.eigenvalues[a])
        .unwrap_or(std::cmp::Ordering::Equal));

    // Read top ncomp eigenvectors by index — no Vec<Vec<f64>> allocation
    let mut eigenvalues = Vec::with_capacity(ncomp);
    let mut eigenvectors = Vec::with_capacity(ncomp);
    for &col_idx in idx_buf.iter().take(ncomp) {
        eigenvalues.push(eig.eigenvalues[col_idx]);
        // Only ONE Vec<f64> per retained component (not per eigenvalue):
        let mut evec: Vec<f64> = eig.eigenvectors.column(col_idx).iter().copied().collect();
        // Sign-align: largest-magnitude entry positive
        let arg = evec.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i).unwrap_or(0);
        if evec[arg] < 0.0 { evec.iter_mut().for_each(|x| *x = -*x); }
        eigenvectors.push(evec);
    }
    (eigenvalues, eigenvectors)
}
```

### Example 2: dhat alloc-audit test for dpca (pattern to copy)

```rust
// tests/alloc_audit_dpca.rs — mirror of tests/alloc_audit_fpca.rs
#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

#[test]
#[cfg(feature = "dhat-heap")]
fn count_dpca_allocations_n200_m50() {
    use fdars_core::fts::dpca;
    use fdars_core::matrix::FdMatrix;
    // Build data OUTSIDE the profiler (setup, not target)
    let n = 200usize; let m = 50usize;
    // ... build FdMatrix and argvals ...
    let _profiler = dhat::Profiler::builder().testing().build();
    let _ = dpca(&data, &argvals, 3, None, None);
    let stats = dhat::HeapStats::get();
    println!("dpca total_blocks: {}", stats.total_blocks);
    println!("dpca total_bytes: {}", stats.total_bytes);
    println!("dpca peak_bytes: {}", stats.max_bytes);
    // After optimization: assert blocks < 1000 (was 17,739 before)
    assert!(stats.total_blocks < 1000,
        "dpca alloc regression: {} blocks (expected <1000 after OPT-A)", stats.total_blocks);
}
```

### Example 3: Criterion bench skeleton for new phase-47 bench file

```rust
// benches/perf_hotpaths.rs — new [[bench]] entry for Phase 47
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_dpca(c: &mut Criterion) {
    let mut group = c.benchmark_group("perf_dpca");
    group.sample_size(20);
    group.measurement_time(std::time::Duration::from_secs(30));
    // Build data OUTSIDE b.iter()
    let (data, argvals) = make_fts_data(200, 50, 42);
    group.bench_function("n200_m50_ncomp3", |b| {
        b.iter(|| fdars_core::fts::dpca(
            black_box(&data), black_box(&argvals), 3, None, None))
    });
    group.finish();
}
// Also add benches for face_covariance and fem_smooth at their PROF-01 cells.
criterion_group!(benches, bench_dpca, bench_face_covariance, bench_fem_smooth);
criterion_main!(benches);
```

---

## Proof Mechanics

### Before/After Criterion Pattern (existing: `benches/audit_hotpaths.rs`)

[VERIFIED: fdars-core/benches/audit_hotpaths.rs:20-101] The pattern is:
1. Generate data **outside** `b.iter()` to avoid measuring allocator setup.
2. Use `black_box()` on all inputs and outputs.
3. Set `sample_size(20)` and `measurement_time(30s)` for expensive cells.
4. Use `criterion_group!` / `criterion_main!` with `harness = false` in Cargo.toml.

New bench file: `benches/perf_hotpaths.rs` — add `[[bench]] name = "perf_hotpaths" harness = false` to `fdars-core/Cargo.toml`. Register once now; Phase 51 (BENCH-02) formalizes the before/after table.

### Equivalence / Golden Test Pattern

For each optimization, add a `#[test]` in `src/` or in a new `tests/equivalence_phase47.rs`:

```rust
#[test]
fn golden_dpca_n50_m10() {
    // Fixed seed + small size for fast CI execution
    let data = make_fts_data(50, 10, 7);
    let argvals = uniform_grid(10);
    let before = dpca_v1(&data, &argvals, 2, None, None).unwrap(); // or capture from known-good commit
    let after  = dpca(&data, &argvals, 2, None, None).unwrap();
    // Assert filter taps and scores match within tolerance
    for c in 0..2 {
        for (a, b) in before.filters[c].as_slice().iter().zip(after.filters[c].as_slice()) {
            assert!((a - b).abs() <= 1e-10 * a.abs().max(1e-10));
        }
    }
}
```

Tolerance: relative ≤1e-10 for float/eigen paths. [VERIFIED: 47-CONTEXT.md:40] "Tolerance: exact for counting/integer paths; relative ≤ 1e-10 for float SVD/eigen paths, documented per change."

In practice, the OPT-A change (removing intermediate copies) should produce bitwise-identical outputs since the actual eigendecomposition inputs and computation are unchanged. The golden test can assert exact equality rather than tolerance — use the stricter bound first, fall back to relative tolerance only if FP reordering causes sub-ULP differences.

### Governor Pinning

[VERIFIED: /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor — read this session, returns "powersave"]
[VERIFIED: which cpupower — /usr/bin/cpupower exists]

Current governor: `powersave`. To pin for before/after benchmarks:
```bash
sudo cpupower frequency-set -g performance
TMPDIR=/home/simonm/.cache/fdars-bench-tmp \
  cargo bench -p fdars-core --features linalg,parallel \
  --bench perf_hotpaths -- dpca n200_m50_ncomp3
sudo cpupower frequency-set -g powersave
```

If sudo is available, pin `performance` for before and after. If not, note caveat in PERF-RESULTS.md: "Measured under `powersave` governor — timings are LOW-CONFIDENCE; relative improvement ratio is more reliable than absolute ms."

---

## Runtime State Inventory

This is a pure code-optimization phase — no renaming, no migration, no stored data. SKIPPED per execution-flow instructions (not a rename/refactor phase).

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Rust built-in `#[test]` + criterion 0.5 |
| Config file | `fdars-core/Cargo.toml` (`[[bench]]` entries) |
| Quick run command | `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel` |
| Full suite command | `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel && cargo clippy --all-targets --features linalg,parallel -- -D warnings` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|--------------|
| PERF-01 | `dpca` output unchanged after OPT-A | golden equivalence | `cargo test -p fdars-core -- golden_dpca` | ❌ Wave 0 |
| PERF-01 | `face_covariance` output unchanged after OPT-E | golden equivalence | `cargo test -p fdars-core -- golden_face_cov` | ❌ Wave 0 |
| PERF-01 | `fem_smooth` output unchanged after OPT-F | golden equivalence | `cargo test -p fdars-core -- golden_fem_smooth` | ❌ Wave 0 |
| PERF-01 | `dpca` wall-time improvement ≥15% | criterion bench | `cargo bench -p fdars-core --bench perf_hotpaths -- dpca` | ❌ Wave 0 |
| PERF-01 | `face_covariance` wall-time improvement ≥15% | criterion bench | `cargo bench -p fdars-core --bench perf_hotpaths -- face_cov` | ❌ Wave 0 |
| PERF-02 | `dpca` allocation blocks < 1000 (was 17,739) | dhat alloc-audit | `cargo test -p fdars-core --features dhat-heap,linalg -- count_dpca_allocations --nocapture` | ❌ Wave 0 |
| PERF-02 | `fsvd` allocation blocks reduced | dhat alloc-audit | `cargo test -p fdars-core --features dhat-heap,linalg -- count_fsvd_allocations --nocapture` | ❌ Wave 0 |
| PERF-01/02 | Existing suite green at every commit | unit/integration | `cargo test -p fdars-core --features linalg,parallel` | ✅ |

### Sampling Rate

- **Per task commit:** `TMPDIR=... cargo test -p fdars-core --features linalg,parallel` (existing suite, no bench)
- **Per wave merge:** Full suite + clippy gate: `cargo clippy --all-targets --features linalg,parallel -- -D warnings`
- **Phase gate:** Full suite green + at least one before/after criterion cell showing ≥15% improvement (or ≥25% allocation reduction) before `/gsd-verify-work`

### Wave 0 Gaps

- [ ] `tests/alloc_audit_dpca.rs` — covers PERF-02 (dpca allocation audit), mirrors `tests/alloc_audit_fpca.rs`
- [ ] `tests/equivalence_phase47.rs` — golden tests for OPT-A through OPT-F (all 6 targets)
- [ ] `benches/perf_hotpaths.rs` — criterion bench for dpca, face_covariance, fem_smooth at PROF-01 cells
- [ ] `[[bench]] name = "perf_hotpaths" harness = false` entry in `fdars-core/Cargo.toml`

---

## Security Domain

This is a pure numerical-algorithm optimization phase — no authentication, session management, access control, I/O, or cryptography involved. Security enforcement is not applicable to this phase. SKIPPED.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust stable | All compilation | ✓ | 1.97.0 (from CLAUDE.md) | — |
| criterion 0.5 | Bench evidence | ✓ | 0.5 (already dev-dep) | — |
| dhat 0.3 | Alloc audit (PERF-02) | ✓ | 0.3 (already dev-dep) | — |
| cpupower | Governor pinning | ✓ | `/usr/bin/cpupower` exists | Run under powersave with LOW-CONFIDENCE caveat |
| TMPDIR cache | Long bench/link | ✓ | `/home/simonm/.cache/fdars-bench-tmp` | Create if absent |
| `linalg` feature | faer (Cholesky) | ✓ | Rust 1.97 ≥ 1.84 (required) | Build without `linalg` for non-Cholesky paths |
| `dhat-heap` feature | Alloc tests | ✓ | Feature-gated in Cargo.toml | Build without feature (empty test binary) |

[VERIFIED: /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor — current governor is "powersave"]
[VERIFIED: which cpupower — /usr/bin/cpupower available; sudo required to change governor]

**Missing dependencies with no fallback:** None — all required tools are present.

**Missing dependencies with fallback:**
- `cpupower` governor write (requires sudo): if sudo unavailable, document `powersave` caveat in PERF-RESULTS.md. Before/after ratio is still meaningful even under powersave.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `nalgebra::DMatrix::from_fn` does not allocate an intermediate Vec (constructs in-place) | OPT-B, OPT-C, OPT-D | If `from_fn` allocates intermediately, the copy-removal optimization provides no allocation benefit (only code clarity); still correct |
| A2 | The 17,739-block count in `dpca` is primarily from `col.iter().copied().collect()` calls (m×n_freq = 50×200 = 10,000 Vec<f64> allocations) | OPT-A root cause | If nalgebra's internal eigen buffers dominate, OPT-A's index-sort approach saves fewer blocks; dhat test will confirm actual savings |
| A3 | `kernel_gaussian` is `(-0.5 * u * u).exp()` or similar transcendental — the dominant FLOPs in `cov_irreg` | OPT-E | If kernel is a polynomial (e.g., Epanechnikov), the speedup from weight precomputation is smaller; still a win but smaller |
| A4 | The GCV EDF computation at `fem_smooth.rs:568-578` (N Cholesky back-solves) accounts for most of the 452 ms, not the assembly step | OPT-F defer rationale | If assembly dominates, there may be another constant-factor win to explore in the triangle loop |

**If A3 is wrong:** The OPT-E wall-time improvement may be <15%; the criterion bench will reveal this. If improvement is confirmed <15%, document the target as DEFERRED per CONTEXT.md policy.

---

## Open Questions

1. **Can `nalgebra::SymmetricEigen` accept a pre-allocated output buffer?**
   - What we know: The `DMatrix` passed to `SymmetricEigen::new(mat)` is consumed; the resulting `mat` is overwritten. Nalgebra 0.33 has no `in_place_eigen` variant.
   - What's unclear: Whether `DMatrix::from_fn` avoids the intermediate Vec at the nalgebra allocator level.
   - Recommendation: Proceed with OPT-A index-sort approach regardless — it eliminates the far more numerous `col.iter().copied().collect()` calls, which are the dominant contribution to 17,739 blocks.

2. **What is the actual `kernel_gaussian` implementation?**
   - The grep found it in `src/irreg_fdata/kernels.rs`. [ASSUMED] it uses `exp()`. Confirm by reading that file before coding OPT-E.
   - Recommendation: Read `src/irreg_fdata/kernels.rs` in the planning step to confirm the kernel form before quoting expected speedup percentages in the plan.

3. **Is the `fem_smooth` GCV EDF computation or Cholesky solve the wall-time bottleneck?**
   - What we know: 452 ms at 576 nodes. Both Cholesky and EDF computation are O(N³) = O(576³ ≈ 191M) ops.
   - What's unclear: Which of the two dominates (factor of 2 uncertainty).
   - Recommendation: Profile with a simple timing wrapper in the test (print timestamps before/after each stage). This does not require criterion — a `std::time::Instant` in a test is sufficient.

---

## Sources

### Primary (HIGH confidence — files read this session)

- [VERIFIED: src/fts/spectral.rs:194-237] — `eigen_at_frequency` body; identified `col.iter().copied().collect()` as dominant allocation site
- [VERIFIED: src/fts/spectral.rs:295-303] — `dpca` body showing the `for k in 0..n_freq` loop calling `eigen_at_frequency`
- [VERIFIED: src/irreg_fdata/smoothing.rs:111-176] — `cov_irreg` and `accumulate_cov_at_point`; confirmed redundant `exp()` computation across grid cells
- [VERIFIED: src/fpca_variants.rs:471-488] — `fsvd` gram Vec staging allocation
- [VERIFIED: src/fpca_variants.rs:733-740] — `ssvd` c_scaled Vec staging allocation
- [VERIFIED: src/fts/acf.rs:329-337] — `functional_acf` c0_scaled Vec staging allocation
- [VERIFIED: src/fem_smoothing.rs:540-541] — `phi_t_phi.clone()` identification; confirmed load-bearing (GCV trace uses phi_t_phi at :581-584)
- [VERIFIED: src/fem_smoothing.rs:507-509] — `assemble_fem_matrices` called once; no redundant recomputation in hot path
- [VERIFIED: tests/alloc_audit_fpca.rs:1-132] — dhat test pattern; confirmed `dhat::Profiler::builder().testing().build()` idiom and `--test-threads=1` requirement
- [VERIFIED: fdars-core/benches/audit_hotpaths.rs:20-101] — criterion bench pattern; `sample_size(20)`, `measurement_time`, `black_box` usage
- [VERIFIED: fdars-core/Cargo.toml:90-96] — `[[bench]] name = "audit_hotpaths" harness = false` — pattern to follow for new `perf_hotpaths` bench
- [VERIFIED: fdars-core/Cargo.toml:30,32] — `dhat-heap` feature exists, `dhat = "0.3"` already a dev-dep
- [VERIFIED: .planning/phases/47-hot-path-allocation-performance/47-CONTEXT.md:40] — tolerance specification
- [VERIFIED: .planning/phases/46-whole-crate-profiling-measurement/PROF-01-hotpath-targets.md:44,74,78] — before-numbers for dpca, face_covariance, fem_smooth

### Secondary (MEDIUM confidence — inferred from code structure)

- `IrregFdata.argvals` field indexing used in OPT-E proposal — inferred from `smoothing.rs:153-166` usage pattern `offsets[i]..offsets[i+1]` and `ifd.argvals`

### Tertiary (LOW confidence — unverified this session)

- A3 (`kernel_gaussian` uses `exp()`) — `src/irreg_fdata/kernels.rs` not read; function name implies Gaussian kernel
- A1 (nalgebra `from_fn` allocates in-place) — nalgebra internals not read

---

## Metadata

**Confidence breakdown:**
- OPT-A (dpca alloc root cause): HIGH — eigenvector copy site read verbatim from source
- OPT-B/C/D (copy removals): HIGH — all three `from_column_slice` sites read verbatim
- OPT-E (face_covariance kernel precompute): HIGH (code structure) / MEDIUM (speedup estimate, depends on A3)
- OPT-F (fem_smooth clone): HIGH — both clone site and load-bearing use confirmed verbatim
- OPT-F defer rationale: HIGH — O(N³) structural cost confirmed, no sparse solver in scope

**Research date:** 2026-08-30
**Valid until:** 2026-09-30 (stable Rust library; no external changes expected)
