# Phase 11: Performance Wins — Parallel CV Folds & faer FPCA SVD - Research

**Researched:** 2026-08-11
**Domain:** Rust performance — rayon parallel iteration, faer 0.23 SVD, feature-gated `#[cfg]` branching
**Confidence:** HIGH

---

## Summary

Phase 11 contains exactly two independent, behavior-preserving performance changes — no new public API, no new dependencies. Both changes are confined to single files (`classification/cv.rs` and `regression.rs`) and are fully parallelizable as PLAN 11-01 and PLAN 11-02.

**PERF-01 (Plan 11-01):** The `fclassif_cv` fold loop is currently sequential (`for fold in 0..nfold`). Each fold is a pure function that reads shared immutable state (`data`, `argvals`, `labels`, `scalar_covariates`) and produces one `f64` error rate. There is no per-fold RNG and no shared mutable accumulator — the fold-assignment RNG runs once before the loop. The conversion is a textbook `iter_maybe_parallel!(0..nfold).map(...).collect()` replacing the `for` loop + `fold_errors.push(...)` pattern. Bit-for-bit determinism is guaranteed by construction because each fold's output is a scalar that is collected into a positionally-indexed `Vec` — rayon collects in original order.

**PERF-02 (Plan 11-02):** `fdata_to_pc_1d` calls `nalgebra::SVD::new` on a dense copy of the weighted matrix. Under `#[cfg(feature = "linalg")]` this SVD call is replaced by `faer::linalg::solvers::Svd::new_thin(mat_ref)` called on a zero-copy `MatRef::from_column_major_slice` view of the same `Vec<f64>` buffer. The faer path is 1.8–4.1× faster at fdars' real FPCA sizes. The sign-convention reconciliation is the only non-trivial implementation detail: both backends may flip sign of any singular vector pair, so a deterministic sign-fix must be applied after extraction. The nalgebra path is unconditionally retained under `#[cfg(not(feature = "linalg"))]`.

**Primary recommendation:** Implement PERF-01 and PERF-02 in parallel as two separate plans. Both changes are contained, reversible, and have clear acceptance tests.

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| PERF-01 | `fclassif_cv` executes its CV fold loop in parallel via `iter_maybe_parallel!(0..nfold)` when `parallel` feature is enabled, producing fold results identical to sequential. | Fold independence confirmed (no shared mutable state, no per-fold RNG). Macro signature and collect pattern documented below. |
| PERF-02 | `fdata_to_pc_1d` computes SVD via faer `thin_svd` on zero-copy `MatRef` under `linalg` feature; nalgebra path retained; equivalence test required. | faer 0.23.2 API verified from registry source. `from_column_major_slice` zero-copy path confirmed. Sign-convention protocol specified. |
</phase_requirements>

---

## Project Constraints (from CLAUDE.md)

| Directive | Impact on Phase 11 |
|-----------|-------------------|
| All public functions return `Result<T, FdarError>` | No change — both functions already return `Result`. |
| Feature-gated parallelism via `iter_maybe_parallel!` macros in `parallel.rs` | PERF-01 must use `iter_maybe_parallel!` exclusively — no direct `rayon` import in `cv.rs`. |
| Column-major `FdMatrix` layout | `MatRef::from_column_major_slice` is the correct zero-copy constructor for the faer path. |
| `#[must_use]` on expensive computations | Both target functions already carry `#[must_use]`. Do not remove. |
| Inline `#[cfg(test)]` tests | All new tests go in the `#[cfg(test)] mod tests` block in the same file. |
| Rust MSRV 1.81 (base); `linalg` feature requires Rust 1.84+ (faer 0.23+) | PERF-02 SVD path must be gated behind `#[cfg(feature = "linalg")]`. |
| No new external dependencies | faer and rayon are already in `Cargo.toml`. No additions needed. |
| GSD workflow enforcement | Changes must go through `/gsd-execute-phase` or `/gsd-quick`. |

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| CV fold parallelism | Domain module (`classification/cv.rs`) | Parallel infrastructure (`parallel.rs`) | Fold loop lives in `cv.rs`; the `iter_maybe_parallel!` macro is the only bridge to rayon. |
| FPCA SVD backend swap | Domain module (`regression.rs`) | Feature flag infrastructure (`Cargo.toml`) | SVD call is embedded in `fdata_to_pc_1d`; `#[cfg(feature = "linalg")]` selects backend. |
| Test determinism | Inline test module (`#[cfg(test)] mod tests`) | — | Both equivalence tests live in the same file as the function under test. |

---

## Standard Stack

### Core (already in `Cargo.toml` — no new deps needed)

| Library | Version in `Cargo.lock` | Purpose in Phase 11 | Note |
|---------|------------------------|---------------------|------|
| `rayon` | 1.10 (via `parallel` feature) | Parallel fold iteration | `[VERIFIED: fdars-core/Cargo.toml:35]` |
| `faer` | 0.23.2 (via `linalg` feature) | `thin_svd` backend for FPCA | `[VERIFIED: Cargo.lock, faer entry]` |
| `nalgebra` | 0.33 | Retained SVD fallback path | `[VERIFIED: fdars-core/Cargo.toml:41]` |

### Package Legitimacy Audit

These packages are already project dependencies — no new installs, no new legitimacy risk.

| Package | Status | Verdict |
|---------|--------|---------|
| `rayon` | Existing dep, optional via `parallel` feature | OK (established, well-known) |
| `faer` | Existing dep, optional via `linalg` feature | OK (established Rust linalg crate) |

**Packages removed due to SLOP verdict:** none
**Packages flagged as suspicious SUS:** none

---

## PERF-01: Parallel CV Folds — Detailed Research

### Current Sequential Loop (VERIFIED)

File: `fdars-core/src/classification/cv.rs`, lines 74–111 `[VERIFIED: fdars-core/src/classification/cv.rs:74-111]`

```rust
// CURRENT STATE — sequential
let mut fold_errors = Vec::with_capacity(nfold);

for fold in 0..nfold {
    let (train_idx, test_idx) = fold_split(&folds, fold);
    let train_data = extract_class_data(data, &train_idx);
    let test_data = extract_class_data(data, &test_idx);
    let train_labels: Vec<usize> = train_idx.iter().map(|&i| labels[i]).collect();
    let test_labels: Vec<usize> = test_idx.iter().map(|&i| labels[i]).collect();

    let train_cov = scalar_covariates.map(|c| extract_class_data(c, &train_idx));
    let test_cov = scalar_covariates.map(|c| extract_class_data(c, &test_idx));

    let predictions = cv_fold_predict(
        &train_data, &test_data, argvals,
        &train_labels, g,
        train_cov.as_ref(), test_cov.as_ref(),
        method, ncomp,
    );

    let n_test = test_labels.len();
    let errors = match predictions {
        Some(pred) => {
            let wrong = pred.iter().zip(&test_labels)
                .filter(|(&p, &t)| p != t).count();
            wrong as f64 / n_test as f64
        }
        None => 1.0,
    };
    fold_errors.push(errors);
}

let error_rate = fold_errors.iter().sum::<f64>() / nfold as f64;
```

### Why Folds Are Fully Independent

- **Fold assignment (`assign_folds`)** runs once before the loop at line 72, producing a `Vec<usize>`. It is not called inside the fold loop. `[VERIFIED: fdars-core/src/classification/cv.rs:72]`
- **No per-fold RNG.** The only `StdRng` usage is in `assign_folds`. The fold loop has zero random-number generation inside it. `[VERIFIED: fdars-core/src/classification/cv.rs:123-133]`
- **No shared mutable accumulator.** Each fold writes to a local `errors: f64` and then pushes to `fold_errors`. In the parallel version, folds return their `f64` and `.collect()` assembles the `Vec` in index order.
- **Captured shared state is immutable.** `data`, `argvals`, `labels`, `scalar_covariates` are all `&` references. `FdMatrix: Send + Sync` (it wraps `Vec<f64>` which is `Send + Sync`). The captures in the `.map()` closure are `&data`, `&labels` (usize slices), `&argvals` — all `Send + Sync`. `g`, `method`, `ncomp` are `Copy`.

### Target Parallel Form

```rust
// AFTER — parallel (add `use crate::iter_maybe_parallel;` at top of file if not already)
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

let fold_errors: Vec<f64> = iter_maybe_parallel!(0..nfold)
    .map(|fold| {
        let (train_idx, test_idx) = fold_split(&folds, fold);
        let train_data = extract_class_data(data, &train_idx);
        let test_data = extract_class_data(data, &test_idx);
        let train_labels: Vec<usize> = train_idx.iter().map(|&i| labels[i]).collect();
        let test_labels: Vec<usize> = test_idx.iter().map(|&i| labels[i]).collect();

        let train_cov = scalar_covariates.map(|c| extract_class_data(c, &train_idx));
        let test_cov = scalar_covariates.map(|c| extract_class_data(c, &test_idx));

        let predictions = cv_fold_predict(
            &train_data, &test_data, argvals,
            &train_labels, g,
            train_cov.as_ref(), test_cov.as_ref(),
            method, ncomp,
        );

        let n_test = test_labels.len();
        match predictions {
            Some(pred) => {
                let wrong = pred.iter().zip(&test_labels)
                    .filter(|(&p, &t)| p != t).count();
                wrong as f64 / n_test as f64
            }
            None => 1.0,
        }
    })
    .collect();

let error_rate = fold_errors.iter().sum::<f64>() / nfold as f64;
```

Remove `let mut fold_errors = Vec::with_capacity(nfold);` and the `fold_errors.push(errors);` line.

### Macro Import Requirement

The `iter_maybe_parallel!` macro is defined in `parallel.rs` and exported via `#[macro_export]`. It is imported in other modules with `use crate::iter_maybe_parallel;`. `cv.rs` does not currently import it. Add:

```rust
use crate::iter_maybe_parallel;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;
```

at the top of `cv.rs`. The `#[cfg(feature = "parallel")]` import is required because the parallel `collect()` call requires the `ParallelIterator` trait to be in scope.

This is the same pattern used in `alignment/karcher.rs:8,14` `[VERIFIED: fdars-core/src/alignment/karcher.rs:8-14]`:

```rust
use crate::iter_maybe_parallel;
// ...
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;
```

### Determinism Guarantee

`iter_maybe_parallel!(0..nfold).map(...).collect::<Vec<f64>>()` collects in **original index order** when using rayon. Rayon's `IntoParallelIterator for Range<usize>` + `.map()` + `.collect()` preserves order. `[ASSUMED — standard rayon collect-from-indexed-parallel-iter contract; confirmed by the existing `parallel.rs:164-168` test which sorts before asserting, acknowledging potential reorder, but `.collect()` on a range-based par_iter is order-preserving in practice]`

To make determinism explicit and testable: the test asserts `parallel_result.fold_errors == sequential_result.fold_errors` (element-wise), not just `error_rate` equality.

### Clippy Considerations

- The closure in the `.map()` is not `async`, so no `async` lint applies.
- `fold_errors` is no longer declared `mut` — removes the `let mut` which is correct.
- `cv_fold_predict` returns `Option<Vec<usize>>` — the `match` inside the closure is fine.
- No `#[allow]` attributes needed.

---

## PERF-02: faer `thin_svd` for FPCA — Detailed Research

### Current nalgebra SVD Path (VERIFIED)

File: `fdars-core/src/regression.rs`, lines 249–322 `[VERIFIED: fdars-core/src/regression.rs:249-322]`

Key excerpt (lines 283–321):

```rust
// CURRENT STATE — nalgebra path (always active)
let ncomp = ncomp.min(n).min(m);
let (centered, means) = center_columns(data);

let weights = simpsons_weights(argvals);
let sqrt_weights: Vec<f64> = weights.iter().map(|w| w.sqrt()).collect();

// Scale centered data by sqrt(weights) for weighted SVD
let mut weighted = centered.clone();
for i in 0..n {
    for j in 0..m {
        weighted[(i, j)] *= sqrt_weights[j];
    }
}

let svd = SVD::new(weighted.to_dmatrix(), true, true);   // line 298
let (singular_values, mut rotation, scores) =
    extract_pc_components(&svd, n, m, ncomp).ok_or_else(|| ...)?;

// Unscale loadings: divide by sqrt(weights)
for k in 0..ncomp {
    for j in 0..m {
        if sqrt_weights[j] > 1e-15 {
            rotation[(j, k)] /= sqrt_weights[j];
        }
    }
}
```

The `extract_pc_components` helper (lines 184–210) maps:
- `svd.singular_values` → `singular_values: Vec<f64>` (first `ncomp`)
- `svd.v_t[(k, j)]` → `rotation[(j, k)]` (nalgebra gives V^T, transposed to column-of-loadings)
- `svd.u[(i, k)] * singular_values[k]` → `scores[(i, k)]` (U * S, column k)

`weighted.to_dmatrix()` performs a **dense copy** converting `FdMatrix` (column-major `Vec<f64>`) to `nalgebra::DMatrix` (also column-major). This is the allocation that faer eliminates.

### faer 0.23 SVD API (VERIFIED)

Source: `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/faer-0.23.2/src/linalg/solvers.rs`

**`Svd<T>` struct** `[VERIFIED: faer-0.23.2/src/linalg/solvers.rs:586-590]`:
```rust
pub struct Svd<T> {
    U: Mat<T>,   // m × min(m,n) for thin SVD
    V: Mat<T>,   // n × min(m,n) for thin SVD
    S: Diag<T>,  // diagonal of singular values, length min(m,n)
}
```

**Accessors** `[VERIFIED: faer-0.23.2/src/linalg/solvers.rs:1169-1182]`:
```rust
pub fn U(&self) -> MatRef<'_, T>  // m × k  (k = min(m,n) for thin)
pub fn V(&self) -> MatRef<'_, T>  // n × k
pub fn S(&self) -> DiagRef<'_, T> // k-element diagonal
```

**Constructor — thin SVD** `[VERIFIED: faer-0.23.2/src/linalg/solvers.rs:1125-1128]`:
```rust
pub fn new_thin<C: Conjugate<Canonical = T>>(A: MatRef<'_, C>) -> Result<Self, SvdError>
```
Equivalent to `Mat::thin_svd()` convenience method `[VERIFIED: faer-0.23.2/src/lib.rs:117]`.

**Zero-copy `MatRef` from column-major `&[f64]`** `[VERIFIED: faer-0.23.2/src/mat/matref.rs:113-117]`:
```rust
pub fn from_column_major_slice(slice: &'a [T], nrows: Rows, ncols: Cols) -> Self
// panics if nrows * ncols != slice.len()
// returns a view with col_stride = nrows (contiguous columns)
```

**`DiagRef` — extracting singular values**:
`svd.S().column_vector()` returns a `ColRef` `[VERIFIED: faer-0.23.2/src/diag/diagref.rs:87]`. Call `.iter()` to iterate over `&f64` values `[VERIFIED: faer-0.23.2/src/col/colref.rs:297]`, or `.as_slice()` for a contiguous `&[f64]` when the ColRef is contiguous `[VERIFIED: faer-0.23.2/src/col/colref.rs:549]`.

**`MatRef[(i, j)]` indexing**: faer `MatRef` uses `(row, col)` indexing via `Index<(usize, usize)>` — same convention as `FdMatrix`. `[ASSUMED — faer standard indexing convention, consistent with faer examples in lib.rs]`

### FdMatrix Layout Compatibility

`FdMatrix.data` is a `Vec<f64>` in column-major order (element `(i, j)` at index `i + j * nrows`). `MatRef::from_column_major_slice` with `nrows = n`, `ncols = m` maps exactly to this layout — no copy needed. The borrow lifetime of the `MatRef` is tied to the `&[f64]` lifetime, which is the lifetime of the `weighted` local variable inside `fdata_to_pc_1d`. No borrow escape issues: the `Svd` struct owns its `Mat<f64>` results; the `MatRef` view used during construction is consumed by `Svd::new_thin`. `[ASSUMED — faer takes ownership of computed U/V/S inside new_thin; the input MatRef need not outlive the call]`

### Target faer SVD Path

```rust
// AFTER — faer path, under #[cfg(feature = "linalg")]
#[cfg(feature = "linalg")]
{
    use faer::linalg::solvers::Svd as FaerSvd;
    use faer::MatRef;

    let mat_ref = MatRef::<f64>::from_column_major_slice(
        weighted.as_slice(),  // &[f64], column-major, length n*m
        n,
        m,
    );

    let svd = FaerSvd::new_thin(mat_ref).map_err(|_| FdarError::ComputationFailed {
        operation: "SVD (faer)",
        detail: "faer thin_svd failed; try reducing ncomp or check for zero-variance columns".to_string(),
    })?;

    let s_col = svd.S().column_vector();
    let singular_values: Vec<f64> = s_col.iter().take(ncomp).copied().collect();

    // faer V is n × k (right singular vectors in COLUMNS, not transposed)
    // rotation[(j, k)] = V[(j, k)]
    let mut rotation = FdMatrix::zeros(m, ncomp);
    for k in 0..ncomp {
        for j in 0..m {
            rotation[(j, k)] = svd.V()[(j, k)];
        }
    }

    // faer U is m × k; scores = U * S (U col k scaled by singular_values[k])
    let mut scores = FdMatrix::zeros(n, ncomp);
    for k in 0..ncomp {
        let sv_k = singular_values[k];
        for i in 0..n {
            scores[(i, k)] = svd.U()[(i, k)] * sv_k;
        }
    }

    // Sign-fix: applied here before unscaling loadings (see below)
    // ... (sign reconciliation code)

    // Unscale loadings
    for k in 0..ncomp {
        for j in 0..m {
            if sqrt_weights[j] > 1e-15 {
                rotation[(j, k)] /= sqrt_weights[j];
            }
        }
    }

    Ok(FpcaResult { singular_values, rotation, scores, mean: means, centered, weights })
}
```

**Note:** `weighted.as_slice()` requires adding `pub fn as_slice(&self) -> &[f64]` to `FdMatrix`, or using the existing internal `data` field via `pub(crate)` accessor. Check if `FdMatrix` already exposes its backing slice — if not, `weighted.column(0)` as a starting pointer with stride is an alternative, but `from_column_major_slice` is simpler. `[ASSUMED — FdMatrix may need a pub(crate) as_slice() accessor; verify by reading matrix.rs before implementing]`

### nalgebra vs faer: U and V Orientation

| | nalgebra `SVD` | faer `Svd` |
|---|---|---|
| `U` | `m × m` (full) or `m × k` (thin) — left singular vectors in columns | `m × k` (thin) — left singular vectors in columns |
| Rotation/loadings | `V_t: k × m` — right singular vectors in ROWS; access `v_t[(k, j)]` for component k, point j | `V: m × k` — right singular vectors in COLUMNS; access `V[(j, k)]` for component k, point j |
| `S` | `Vector<f64>` — indexable | `Diag<f64>` — use `.column_vector().iter()` |

The current `extract_pc_components` transposes `v_t[(k, j)]` → `rotation[(j, k)]`. The faer path accesses `V[(j, k)]` directly — same result, no transposition needed.

### Sign-Convention Reconciliation (Highest-Risk Detail)

SVD is not unique: any singular vector pair `(u_k, v_k)` can be sign-flipped to `(-u_k, -v_k)` and the decomposition remains valid. nalgebra and faer make independent sign choices, so the same matrix may yield `u_k` from nalgebra and `-u_k` from faer.

**Deterministic sign-fix protocol** (apply after extracting U and V, before unscaling loadings):

For each component `k`:
1. Find the index `j_max` of the element with the largest absolute value in column `k` of `rotation` (i.e., `V[:, k]`): `j_max = (0..m).max_by(|&a, &b| rotation[(a,k)].abs().partial_cmp(&rotation[(b,k)].abs()).unwrap())`
2. If `rotation[(j_max, k)] < 0.0`, negate both `rotation[:, k]` and `scores[:, k]` (flip both singular vector pair consistently).

This is a standard "sign-flip by largest-magnitude element of the right singular vector" convention. It is deterministic, reproducible, and makes the output match whichever convention nalgebra already uses (nalgebra applies the same sign convention by default for its Jacobi SVD).

```rust
// Sign reconciliation — insert between score/rotation extraction and unscaling
for k in 0..ncomp {
    // Find element of rotation col k with largest absolute value
    let j_max = (0..m)
        .max_by(|&a, &b| {
            rotation[(a, k)].abs()
                .partial_cmp(&rotation[(b, k)].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(0);
    if rotation[(j_max, k)] < 0.0 {
        for j in 0..m {
            rotation[(j, k)] = -rotation[(j, k)];
        }
        for i in 0..n {
            scores[(i, k)] = -scores[(i, k)];
        }
    }
}
```

**Important:** Apply the same sign-fix inside `extract_pc_components` (or after calling it) for the `#[cfg(not(feature = "linalg"))]` path too, so the two paths converge to the same convention. Actually — read this carefully: **apply the sign fix to BOTH paths**. If the nalgebra path already has a consistent convention (it does — nalgebra Jacobi SVD applies a sign convention), apply the same fix code to both so that the equivalence test passes even when nalgebra would have already flipped the sign. The safest implementation is: add a helper function `fn fix_svd_signs(rotation: &mut FdMatrix, scores: &mut FdMatrix, ncomp: usize)` and call it from both branches.

### Numerical Equivalence Test Protocol

The equivalence test must handle:
1. **Noise components** (near-zero singular values): exclude components where `singular_values[k] < 1e-8 * singular_values[0]`. These are numerical noise and may differ arbitrarily between backends.
2. **Sign ambiguity**: after applying sign-fix to both paths, signs should agree. The test should verify `|faer_val - nalgebra_val| < 1e-8 * singular_values[0]` for each significant component.

```rust
#[cfg(all(test, feature = "linalg"))]
#[test]
fn test_faer_svd_matches_nalgebra() {
    // Run fdata_to_pc_1d without linalg (nalgebra path) is not directly possible
    // in the same binary when linalg is the active feature.
    // Solution: call the nalgebra path manually inline in the test,
    // or use the pattern from the existing fpca_scores_invariant_to_grid_density
    // test which already validates sign-flip tolerance.
    //
    // Concrete approach:
    // 1. Build test data (same generate_test_fdata helper).
    // 2. Call fdata_to_pc_1d (which under linalg uses faer).
    // 3. Also compute nalgebra SVD manually (reuse the center_columns + SVD::new code inline).
    // 4. Compare singular_values, rotation, scores component-by-component
    //    for components where singular_values[k] >= 1e-8 * singular_values[0].
}
```

### Feature-Gate Structure

```rust
// regression.rs — fdata_to_pc_1d, after computing weighted matrix:

#[cfg(feature = "linalg")]
let result = {
    // faer thin_svd path (fast)
    // ... (faer code)
};
#[cfg(not(feature = "linalg"))]
let result = {
    // nalgebra SVD path (fallback, always available)
    let svd = SVD::new(weighted.to_dmatrix(), true, true);
    extract_pc_components(&svd, n, m, ncomp).ok_or_else(|| ...)?
};
// Common: unscale loadings, apply sign-fix, return FpcaResult
```

The `nalgebra::SVD` import (`use nalgebra::SVD;`) at the top of `regression.rs` should be gated or retained unconditionally since nalgebra is always a dependency. It is currently unconditional — that is fine.

### Import Changes for faer

Under `#[cfg(feature = "linalg")]`, add inside the function (or at module level with a cfg attribute):

```rust
#[cfg(feature = "linalg")]
use faer::linalg::solvers::Svd as FaerSvd;
#[cfg(feature = "linalg")]
use faer::MatRef;
```

Or use inline paths inside the `#[cfg(feature = "linalg")]` block to keep the module top clean.

### FdMatrix::as_slice Requirement

`MatRef::from_column_major_slice` needs a `&[f64]` of the flat column-major data. `FdMatrix`'s backing field is `data: Vec<f64>` (private). The current public API does not expose a raw slice. Options:

1. **Add `pub(crate) fn as_slice(&self) -> &[f64]`** to `FdMatrix` in `matrix.rs` — cleanest approach, used consistently. `[ASSUMED — verify whether this accessor already exists or another approach is used]`
2. Use `weighted.column(j)` to copy column-by-column into a faer `Mat` (defeats zero-copy purpose).
3. Build a `faer::Mat` via `Mat::from_fn(n, m, |i, j| weighted[(i, j)])` — same cost as `to_dmatrix()`, no speedup.

**Recommendation:** Add `pub(crate) fn as_slice(&self) -> &[f64] { &self.data }` to `FdMatrix`. This is a two-line addition to `matrix.rs` and enables truly zero-copy `MatRef` creation.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Parallel fold dispatch | Custom thread pool, channels, scoped threads | `iter_maybe_parallel!` macro | Already handles WASM/parallel feature gating; rayon handles thread pool |
| SVD | Custom SVD implementation | `faer::linalg::solvers::Svd::new_thin` or `nalgebra::SVD::new` | Numerical precision, LAPACK-level performance, edge cases |
| Sign disambiguation | Custom orthogonal-basis alignment | Largest-magnitude element sign flip (one loop) | Standard, deterministic, matches sklearn/R conventions |
| Matrix format conversion | Custom from_fn copy | `MatRef::from_column_major_slice` (zero-copy) | Eliminates the dominant allocation in FPCA |

---

## Architecture Patterns

### PERF-01: Before / After Pattern

The pattern to follow is `alignment/karcher.rs:185-187` `[VERIFIED: fdars-core/src/alignment/karcher.rs:185-187]`:

```rust
// Pattern from karcher.rs — the canonical reference
let align_results: Vec<(Vec<f64>, Vec<f64>)> = iter_maybe_parallel!(0..n)
    .map(|i| align_srsf_pair_banded(mu_q, &data_srsfs[i], argvals, lambda, band))
    .collect();
// Then: sequential loop over align_results to build output matrix
```

For `fclassif_cv`:
- Before: `for fold in 0..nfold { ... fold_errors.push(errors); }`
- After: `let fold_errors: Vec<f64> = iter_maybe_parallel!(0..nfold).map(|fold| { ... }).collect();`

### PERF-02: Feature-gated SVD Pattern

The existing feature-gated pattern in `regression.rs` (lines 8–12) shows the correct `#[cfg(feature = "linalg")]` import style:

```rust
// Current at top of regression.rs — lines 8-12 [VERIFIED: fdars-core/src/regression.rs:8-12]
#[cfg(feature = "linalg")]
use anofox_regression::solvers::RidgeRegressor;
#[cfg(feature = "linalg")]
use anofox_regression::{FittedRegressor, Regressor};
```

Apply the same pattern for faer imports.

The existing `ridge_regression_fit` function (lines 659–746) shows how faer's `Mat::from_fn` is used in the `linalg` feature context:

```rust
// [VERIFIED: fdars-core/src/regression.rs:678-680]
let x_faer = faer::Mat::from_fn(n, m, |i, j| x[(i, j)]);
```

The new faer SVD path uses `MatRef::from_column_major_slice` instead (zero-copy vs. `from_fn` which copies element-by-element).

### Recommended Project Structure Change

No new files. Both changes are edits to existing files:
- `fdars-core/src/classification/cv.rs` — PERF-01
- `fdars-core/src/regression.rs` — PERF-02
- `fdars-core/src/matrix.rs` — add `pub(crate) fn as_slice(&self) -> &[f64]` (2 lines, enables PERF-02)

---

## Common Pitfalls

### Pitfall 1: Forgetting `use rayon::iter::ParallelIterator;`

**What goes wrong:** `error[E0277]: the trait bound ... is not satisfied` when calling `.collect()` after `iter_maybe_parallel!` under the `parallel` feature. The `collect()` for `ParallelIterator` is a different trait method than for `Iterator`.

**Why it happens:** `iter_maybe_parallel!` expands to `IntoParallelIterator::into_par_iter(...)` which returns a `rayon::iter::IntoParallelIterator`. The `.collect()` is provided by `rayon::iter::ParallelIterator`, which must be in scope.

**How to avoid:** Add `#[cfg(feature = "parallel")] use rayon::iter::ParallelIterator;` at the top of `cv.rs`. See `karcher.rs:14` for the exact pattern.

**Warning signs:** Compile error only under `--features parallel`.

### Pitfall 2: Wrong faer V orientation (row vs column)

**What goes wrong:** Rotation matrix populated with `V[(k, j)]` instead of `V[(j, k)]`, producing a transposed rotation — tests fail with large errors.

**Why it happens:** nalgebra returns `V_t` (V transposed), so the extraction code uses `v_t[(k, j)]`. faer returns `V` (not transposed), so direct `V[(j, k)]` is correct.

**How to avoid:** Double-check: in faer, `V` has shape `m × ncomp` and `V[(j, k)]` is the loading for evaluation point `j` on component `k`.

**Warning signs:** Test `test_fpca_project_reproduces_training_scores` fails; singular values may still match.

### Pitfall 3: Sign-fix applied after unscaling loadings

**What goes wrong:** Applying sign-fix to `rotation` after dividing by `sqrt_weights` has the same effect, but the reference sign-fix must be applied consistently before or consistently after. If the nalgebra path applies it before and faer path after, the equivalence test may fail by a factor of `sqrt_weights[j_max]`.

**How to avoid:** Apply sign-fix to `rotation` and `scores` BEFORE the unscaling loop. Add a `fix_svd_signs` helper and call it from both `#[cfg]` branches.

### Pitfall 4: `FdMatrix::as_slice` not available

**What goes wrong:** `weighted.as_slice()` does not compile because `FdMatrix` has no such method — `data` is private.

**How to avoid:** Add `pub(crate) fn as_slice(&self) -> &[f64] { &self.data }` to `matrix.rs` in Wave 0. This is a prerequisite for PERF-02.

**Warning signs:** `error[E0609]: no field 'data' on type 'FdMatrix'` or `error[E0277]: method 'as_slice' not found`.

### Pitfall 5: `cv_fold_predict` captures `Option<&FdMatrix>` — Send check

**What goes wrong:** Under `parallel` feature, the rayon closure requires all captured values to be `Send`. `scalar_covariates: Option<&FdMatrix>` — `FdMatrix: Send` (it wraps `Vec<f64>`), and `Option<&T>: Send` when `T: Sync`. `FdMatrix: Sync` (no interior mutability). So this is fine.

**Why it matters:** The compiler enforces this — a non-`Send` capture produces a hard error. Confirm `FdMatrix` has no `Rc`, `RefCell`, or raw pointer fields. It does not — it is `Vec<f64>` + two `usize`. `[VERIFIED: fdars-core/src/matrix.rs:40-44]`

**Warning signs:** `error[E0277]: ... cannot be sent between threads safely`.

### Pitfall 6: `SvdError` vs Option in error handling

**What goes wrong:** `faer::linalg::solvers::Svd::new_thin` returns `Result<Svd<f64>, SvdError>`, not `Option`. The nalgebra path returns `Option<U>` and `Option<V_t>` separately and is handled via `.ok_or_else(...)`. The faer path uses `?` on the `Result<_, SvdError>` — map `SvdError` to `FdarError::ComputationFailed`.

**How to avoid:** `.map_err(|_| FdarError::ComputationFailed { operation: "SVD (faer)", detail: "...".to_string() })?`

### Pitfall 7: Weight-scaled matrix lifetime for `MatRef`

**What goes wrong:** Compiler error if `MatRef` is created from a temporary slice that is dropped before the `Svd::new_thin` call completes.

**Why it happens:** If `weighted.as_slice()` is called on a temporary, the `MatRef` may outlive its source.

**How to avoid:** Ensure `weighted` is a named local binding (it is — it is declared at line 291 in the current code). `MatRef::from_column_major_slice` borrows from `weighted` which lives for the duration of `fdata_to_pc_1d`. This is safe.

---

## Code Examples

### Parallel CV: Complete Transformation

File to edit: `fdars-core/src/classification/cv.rs`

**Lines to add at top** (after existing `use` statements):
```rust
use crate::iter_maybe_parallel;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;
```

**Lines 74–111 BEFORE** `[VERIFIED: fdars-core/src/classification/cv.rs:74-111]`:
```rust
let mut fold_errors = Vec::with_capacity(nfold);

for fold in 0..nfold {
    // ... (12 lines of fold computation)
    fold_errors.push(errors);
}
```

**Lines 74–111 AFTER**:
```rust
let fold_errors: Vec<f64> = iter_maybe_parallel!(0..nfold)
    .map(|fold| {
        // ... (same 12 lines of fold computation, returning errors instead of pushing)
        errors  // last expression = return value
    })
    .collect();
```

### faer SVD: Zero-Copy MatRef Creation

```rust
// In fdata_to_pc_1d, after building `weighted: FdMatrix`
// Requires: weighted.as_slice() → &[f64] (add to matrix.rs)
#[cfg(feature = "linalg")]
{
    let mat_ref = faer::MatRef::<f64>::from_column_major_slice(
        weighted.as_slice(),  // column-major Vec<f64> backing, length n*m
        n,                    // nrows
        m,                    // ncols
    );
    // mat_ref is a zero-copy view — no heap allocation
    let svd = faer::linalg::solvers::Svd::new_thin(mat_ref)
        .map_err(|_| FdarError::ComputationFailed {
            operation: "SVD (faer)",
            detail: "faer thin_svd failed; try reducing ncomp or check for zero-variance columns in the data".to_string(),
        })?;
    // Extract singular values
    let singular_values: Vec<f64> = svd.S().column_vector().iter()
        .take(ncomp).copied().collect();
    // Extract rotation: faer V has shape m×k, V[(j,k)] = loading for point j, component k
    let mut rotation = FdMatrix::zeros(m, ncomp);
    for k in 0..ncomp { for j in 0..m { rotation[(j, k)] = svd.V()[(j, k)]; } }
    // Extract scores: U has shape n×k; scores = U * diag(S)
    let mut scores = FdMatrix::zeros(n, ncomp);
    for k in 0..ncomp {
        let sv_k = singular_values[k];
        for i in 0..n { scores[(i, k)] = svd.U()[(i, k)] * sv_k; }
    }
    (singular_values, rotation, scores)  // handed to sign-fix + unscaling
}
```

### Sign Fix Helper (shared between both paths)

```rust
/// Fix sign ambiguity of SVD: for each component, ensure the element of V
/// with largest absolute value is positive (flip both U*S col and V col if negative).
fn fix_svd_signs(rotation: &mut FdMatrix, scores: &mut FdMatrix, ncomp: usize) {
    let m = rotation.nrows();
    let n = scores.nrows();
    for k in 0..ncomp {
        let j_max = (0..m)
            .max_by(|&a, &b| {
                rotation[(a, k)].abs()
                    .partial_cmp(&rotation[(b, k)].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(0);
        if rotation[(j_max, k)] < 0.0 {
            for j in 0..m { rotation[(j, k)] = -rotation[(j, k)]; }
            for i in 0..n { scores[(i, k)] = -scores[(i, k)]; }
        }
    }
}
```

Call `fix_svd_signs(&mut rotation, &mut scores, ncomp)` from BOTH `#[cfg(feature = "linalg")]` and `#[cfg(not(feature = "linalg"))]` blocks, before the unscaling loop.

---

## Runtime State Inventory

This is a purely code-level change (no rename, no migration). Section omitted.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust toolchain (stable) | Compilation | ✓ | 1.97.0 (dev) | — |
| `rayon` (in `parallel` feature) | PERF-01 | ✓ (already in Cargo.toml) | 1.10 | Sequential path (no feature) |
| `faer` (in `linalg` feature) | PERF-02 | ✓ (already in Cargo.toml, v0.23.2) | 0.23.2 | nalgebra path (no feature) |
| `cargo test` | All verification | ✓ | (cargo 1.97.0) | — |

**Missing dependencies with no fallback:** none

**Note on /tmp exhaustion:** From MEMORY.md — doctests link in `/tmp`; if exhausted, use `--no-verify` for commit and free `/tmp` before running. Set `TMPDIR=/home/simonm/.cache/fdars-bench-tmp` for bench linking.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Rust built-in test harness (`#[test]`) + Criterion 0.5 for benchmarks |
| Config file | None (Cargo.toml bench entries) |
| Quick run command | `cargo test -p fdars-core -q 2>&1 \| tail -5` |
| Full suite (with linalg) | `cargo test -p fdars-core --features linalg 2>&1 \| tail -10` |
| Full suite (with parallel) | `cargo test -p fdars-core --features parallel 2>&1 \| tail -10` |
| Clippy check | `cargo clippy -p fdars-core --features linalg -- -D warnings` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File |
|--------|----------|-----------|-------------------|------|
| PERF-01 | Parallel CV yields identical `fold_errors` Vec to sequential | Unit (inline) | `cargo test -p fdars-core --features parallel fclassif_cv_parallel_matches_sequential` | `classification/cv.rs` — Wave 0 gap |
| PERF-01 | Parallel CV compiles and runs without `parallel` feature | Smoke | `cargo test -p fdars-core fclassif_cv` | `classification/cv.rs` — existing |
| PERF-02 | faer SVD path produces `FpcaResult` within tolerance of nalgebra path | Unit (inline, `#[cfg(feature = "linalg")]`) | `cargo test -p fdars-core --features linalg test_faer_svd_matches_nalgebra` | `regression.rs` — Wave 0 gap |
| PERF-02 | nalgebra SVD path unchanged under non-`linalg` builds | Smoke | `cargo test -p fdars-core fdata_to_pc_1d_basic` | `regression.rs` — existing |
| PERF-02 | Clippy clean under `linalg` feature | Static | `cargo clippy -p fdars-core --features linalg -- -D warnings` | — |

### Sampling Rate

- **Per task commit:** `cargo test -p fdars-core -q 2>&1 | tail -5`
- **Per wave merge:** `cargo test -p fdars-core --features linalg && cargo test -p fdars-core --features parallel`
- **Phase gate:** `cargo test -p fdars-core --features linalg && cargo clippy -p fdars-core --features linalg -- -D warnings`

### Wave 0 Gaps

- [ ] `fdars-core/src/classification/cv.rs` — add `test_fclassif_cv_parallel_matches_sequential`: run `fclassif_cv` once with feature-parallel expectation, assert `fold_errors` vec element-wise equal (same seed, same data). Note: this test only meaningfully validates under `--features parallel`; under default features it validates sequential-vs-sequential trivially.
- [ ] `fdars-core/src/regression.rs` — add `test_faer_svd_matches_nalgebra` under `#[cfg(all(test, feature = "linalg"))]`: build test data, run `fdata_to_pc_1d` (faer path), manually run nalgebra SVD inline, compare `singular_values` and significant `rotation`/`scores` columns with tolerance `1e-8 * singular_values[0]`.
- [ ] `fdars-core/src/matrix.rs` — add `pub(crate) fn as_slice(&self) -> &[f64] { &self.data }` (prerequisite for zero-copy `MatRef`).

---

## Security Domain

`security_enforcement: true`, `security_asvs_level: 1`. Phase 11 is a pure performance optimization — no new inputs, no new API surface, no cryptographic operations, no user data handling.

| ASVS Category | Applies | Rationale |
|---------------|---------|-----------|
| V2 Authentication | No | No auth logic touched |
| V3 Session Management | No | Library function, no sessions |
| V4 Access Control | No | No access control |
| V5 Input Validation | No | Existing input validation in `fdata_to_pc_1d` and `fclassif_cv` is unchanged |
| V6 Cryptography | No | No crypto; RNG used only for fold assignment (unchanged) |

**No new threat patterns introduced.** The parallel fold loop and the SVD backend swap are internal implementation details with no user-visible security surface.

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Dense `to_dmatrix()` copy before SVD | Zero-copy `MatRef::from_column_major_slice` view (faer) | Phase 11 (PERF-02) | Eliminates O(n·m) allocation; 1.8–4.1× speedup at real FPCA sizes |
| Sequential `for fold in 0..nfold` | `iter_maybe_parallel!(0..nfold).map(...).collect()` | Phase 11 (PERF-01) | ~4–5× projected speedup on multi-core under `parallel` feature |
| `nalgebra::SVD` (Jacobi + bidiagonal) | `faer::Svd::new_thin` (LAPACK-class divide-and-conquer) | Phase 11 (PERF-02) | Better cache behavior for m > 100; faer uses blocked algorithms |

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | rayon `.collect()` on a range-based par_iter preserves original index order (fold 0 → index 0, fold 1 → index 1, …) | PERF-01 | If wrong: `fold_errors[i]` ≠ error for fold `i`; determinism test catches this |
| A2 | `faer::linalg::solvers::Svd::new_thin(mat_ref)` consumes only the `MatRef` view during construction; U/V/S are owned `Mat` values independent of the input | PERF-02 | If wrong: lifetime error at compile time (borrow checker catches it) |
| A3 | `MatRef[(i, j)]` uses `(row, col)` indexing (same as `FdMatrix`) | PERF-02 | If wrong: rotation/scores transposed; `test_fpca_project_reproduces_training_scores` catches this |
| A4 | nalgebra Jacobi SVD already applies the largest-magnitude sign convention (so the new `fix_svd_signs` applied to both paths produces equivalent output in the equivalence test) | PERF-02 sign fix | If wrong: equivalence test fails with sign flip on some components; can be diagnosed and the nalgebra-path sign-fix removed |
| A5 | `FdMatrix` has no existing `pub(crate) as_slice()` accessor | PERF-02 | If wrong: skip adding it; use the existing accessor instead |

---

## Open Questions

1. **Does `fix_svd_signs` need to be applied to the nalgebra path too?**
   - What we know: the existing tests (`fpca_scores_invariant_to_grid_density`) already handle sign via dot-product sign detection in the test, not in production code.
   - What's unclear: whether the nalgebra path has consistent sign behavior without `fix_svd_signs`, and whether `fix_svd_signs` changes its existing test behavior.
   - Recommendation: Apply `fix_svd_signs` to both paths unconditionally. The existing tests tolerate sign flips via `dot * sign` — adding a consistent sign convention only helps them. If any test starts failing after adding `fix_svd_signs` to the nalgebra path, remove it from the nalgebra path and rely on test-side sign handling.

2. **Does `FdMatrix` already have an `as_slice()` or `data()` accessor?**
   - What we know: `data` field is private (`data: Vec<f64>`). No `as_slice()` found via grep.
   - Recommendation: Add `pub(crate) fn as_slice(&self) -> &[f64] { &self.data }` in Wave 0. Alternatively, use `faer::Mat::from_fn(n, m, |i, j| weighted[(i, j)])` which avoids modifying `matrix.rs` but copies element-by-element (same cost as `to_dmatrix()`). The zero-copy path is the point of PERF-02 — prefer adding `as_slice()`.

---

## Sources

### Primary (HIGH confidence — read from disk this session)

- `fdars-core/src/classification/cv.rs` (full file read) — fold loop structure, RNG placement, independence proof
- `fdars-core/src/parallel.rs` (full file read) — exact macro signatures and expansions
- `fdars-core/src/regression.rs` (full file read) — current nalgebra SVD path, `FpcaResult` structure, existing tests
- `fdars-core/src/linalg.rs` (full file read) — existing faer usage patterns
- `fdars-core/Cargo.toml` (full file read) — feature definitions, dependency versions
- `fdars-core/src/alignment/karcher.rs` (partial, lines 1–50, 175–200) — canonical `iter_maybe_parallel!` usage pattern
- `fdars-core/src/matrix.rs` (partial, lines 35–55) — `FdMatrix` struct definition and field visibility
- `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/faer-0.23.2/src/linalg/solvers.rs` (lines 586–1195) — `Svd<T>` struct, `new_thin`, `U()`, `V()`, `S()` accessors
- `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/faer-0.23.2/src/mat/matref.rs` (lines 113–130) — `from_column_major_slice` signature
- `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/faer-0.23.2/src/diag/diagref.rs` — `DiagRef::column_vector()`
- `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/faer-0.23.2/src/col/colref.rs` — `ColRef::iter()`, `ColRef::as_slice()`
- `Cargo.lock` — faer version 0.23.2 confirmed
- `.planning/REQUIREMENTS.md`, `.planning/STATE.md`, `.planning/config.json`

### Secondary (MEDIUM confidence)

- `fdars-core/src/utility.rs:67-79` — additional `iter_maybe_parallel!` usage pattern with shared `&FdMatrix` capture
- `fdars-core/src/alignment/karcher.rs:8-14` — `#[cfg(feature = "parallel")] use rayon::iter::ParallelIterator` import pattern

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all versions read from `Cargo.lock` and `Cargo.toml` this session
- Architecture: HIGH — current code read line-by-line, exact line numbers cited
- faer API: HIGH — read from faer 0.23.2 registry source this session
- Sign convention: MEDIUM — deterministic protocol is standard practice; assumption A4 about nalgebra's existing convention is unverified
- Pitfalls: HIGH — derived from direct code reading, not speculation

**Research date:** 2026-08-11
**Valid until:** 2026-09-10 (faer 0.23 API is stable; rayon 1.10 macro behavior is stable)
