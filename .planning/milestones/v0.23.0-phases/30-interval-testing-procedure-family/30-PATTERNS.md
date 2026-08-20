# Phase 30: Interval Testing Procedure Family - Pattern Map

**Mapped:** 2026-08-20
**Files analyzed:** 3 (1 new, 2 edits)
**Analogs found:** 3 / 3

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `src/inference/itp.rs` | service / inference module | request-response (FdMatrix → ItpResult) | `src/inference/permutation.rs` | exact (same permutation + seeding pattern) |
| `src/inference/itp.rs::ItpResult` | result struct | — | `src/inference/mod.rs::TestResult` | exact (same derive stack) |
| `src/inference/mod.rs` (edit) | module barrel | — | `src/inference/mod.rs` existing pattern | exact |
| `src/lib.rs` (edit) | crate root re-export | — | `src/lib.rs` lines 225-229 | exact |

---

## Pattern Assignments

### `src/inference/itp.rs` — entry points (`itp_one_pop`, `itp_two_pop`, `itp_flm`)

**Analog:** `src/inference/permutation.rs`

**Imports pattern** (lines 1-16):
```rust
use super::TestResult;
use crate::error::FdarError;
use crate::function_on_scalar::integrated_f_statistic;
use crate::helpers::simpsons_weights;
use crate::matrix::FdMatrix;
use rand::rngs::StdRng;
use rand::SeedableRng;
```

For `itp.rs`, adapt to:
```rust
use crate::basis::projection::{fdata_to_basis, ProjectionBasisType};
use crate::error::FdarError;
use crate::matrix::FdMatrix;
use rand::rngs::StdRng;
use rand::SeedableRng;
```

**Input validation pattern — `InvalidDimension` / `InvalidParameter`** (lines 24-59, `validate_two_samples`):
```rust
fn validate_two_samples(
    data_a: &FdMatrix,
    data_b: &FdMatrix,
    argvals: &[f64],
) -> Result<(usize, usize, usize), FdarError> {
    let (n_a, m_a) = data_a.shape();
    let (n_b, m_b) = data_b.shape();
    if m_a == 0 || m_b == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "at least 1 column (grid points)".to_string(),
            actual: format!("data_a has {m_a} columns, data_b has {m_b} columns"),
        });
    }
    if m_a != m_b {
        return Err(FdarError::InvalidDimension {
            parameter: "data_b",
            expected: format!("{m_a} columns (matching data_a)"),
            actual: format!("{m_b} columns"),
        });
    }
    if argvals.len() != m_a {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("{m_a} elements (matching data columns)"),
            actual: format!("{} elements", argvals.len()),
        });
    }
    if n_a < 2 || n_b < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "at least 2 rows per sample".to_string(),
            actual: format!("data_a has {n_a} rows, data_b has {n_b} rows"),
        });
    }
    Ok((n_a, n_b, m_a))
}
```

For `n_perm == 0` check (lines 160-164):
```rust
if n_perm == 0 {
    return Err(FdarError::InvalidParameter {
        parameter: "n_perm",
        message: "must be >= 1".to_string(),
    });
}
```

**Seeded RNG + permutation loop pattern** (lines 173-188, `t_perm_test`):
```rust
let mut rng = StdRng::seed_from_u64(seed);
let mut n_ge = 0usize;
for _ in 0..n_perm {
    shuffle_labels(&mut labels, &mut rng);
    let perm_stat = integrated_l2_mean_diff(&pooled, &labels, n_a, m, &weights);
    if perm_stat >= observed {
        n_ge += 1;
    }
}
let p_value = (n_ge as f64 + 1.0) / (n_perm as f64 + 1.0);
```

ITP replaces the scalar `perm_stat` with a `Vec<f64>` row per permutation — the loop structure is otherwise identical.

**Fisher–Yates shuffle helper** (lines 119-127, private `fn shuffle_labels`):
```rust
fn shuffle_labels(v: &mut [usize], rng: &mut StdRng) {
    use rand::Rng;
    let n = v.len();
    for i in (1..n).rev() {
        let j = rng.gen_range(0..=i);
        v.swap(i, j);
    }
}
```

CRITICAL: `shuffle_labels` is `fn` (not `pub fn`) — it cannot be called from `itp.rs`. Copy these 7 lines as `fn shuffle_itp` inside `itp.rs`. Same for `pool_two_samples` (lines 63-81), which is also private.

**Pool helper** (lines 63-81, private `fn pool_two_samples`):
```rust
fn pool_two_samples(
    data_a: &FdMatrix,
    data_b: &FdMatrix,
    n_a: usize,
    n_b: usize,
    m: usize,
) -> FdMatrix {
    let mut pooled = FdMatrix::zeros(n_a + n_b, m);
    for j in 0..m {
        for i in 0..n_a {
            pooled[(i, j)] = data_a[(i, j)];
        }
        for i in 0..n_b {
            pooled[(n_a + i, j)] = data_b[(i, j)];
        }
    }
    pooled
}
```

For ITP, this pools coefficient rows (FdMatrix, shape (n, p)) rather than raw curve rows (shape (n, m)) — same pattern, different matrix dimensions.

**p-value formula** (line 183):
```rust
let p_value = (n_ge as f64 + 1.0) / (n_perm as f64 + 1.0);
```

Apply the same `(n_ge + 1) / (n_perm + 1)` convention per component k for `raw_pvalues[k]`.

**Test structure** (lines 254-417):
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::uniform_grid;

    fn make_sample(n: usize, argvals: &[f64], shift: f64, seed: u64) -> FdMatrix { ... }

    #[test]
    fn t_perm_separated_small_p() { ... assert!(res.p_value < 0.05, ...); }

    #[test]
    fn t_perm_null_large_p() { ... assert!(res.p_value > 0.1, ...); }

    #[test]
    fn t_perm_deterministic() {
        let r1 = t_perm_test(&a, &b, &argvals, 99, 123).unwrap();
        let r2 = t_perm_test(&a, &b, &argvals, 99, 123).unwrap();
        assert_eq!(r1, r2, "same seed must give bit-identical result");
    }

    #[test]
    fn t_perm_invalid_input() {
        assert!(matches!(
            t_perm_test(&a, &b, &argvals, 99, 1),
            Err(FdarError::InvalidDimension { .. })
        ));
        assert!(matches!(
            t_perm_test(&a, &b2, &argvals, 0, 1),
            Err(FdarError::InvalidParameter { .. })
        ));
    }
}
```

Copy this pattern for ITP tests: localized-difference fixture, null fixture, deterministic (same-seed) fixture, error-path fixture.

---

### `src/inference/itp.rs::ItpResult` — result struct

**Analog:** `src/inference/mod.rs::TestResult` (lines 42-57)

**Derive / attribute stack** (lines 47-57):
```rust
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct TestResult {
    /// Observed test statistic.
    pub statistic: f64,
    /// P-value for the null hypothesis of equal group means.
    pub p_value: f64,
    /// Number of permutations used (0 for non-permutation paths).
    pub n_perm: usize,
}
```

Apply identically to `ItpResult`:
```rust
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct ItpResult {
    pub adjusted_pvalues: Vec<f64>,
    pub raw_pvalues: Vec<f64>,
    pub basis_type: ProjectionBasisType,
    pub n_basis: usize,
    pub n_perm: usize,
}
```

Note: `ProjectionBasisType` already derives `Debug, Clone, Copy, PartialEq` (see `src/basis/projection.rs` lines 17-24) so it is valid inside a `PartialEq`-derived struct. It does NOT derive serde — add `#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]` to `ProjectionBasisType` if needed, or store the type as a string/integer in `ItpResult`. Confirm with crate's existing pattern before adding.

---

### `src/inference/itp.rs` — basis projection call site

**Analog:** `src/basis/projection.rs`, `fdata_to_basis` (lines 98-150)

**Projection call + Option → Result bridge pattern:**
```rust
// fdata_to_basis returns Option<BasisProjectionResult> — bridge to Result
let proj = fdata_to_basis(data, argvals, nbasis, basis_type)
    .ok_or_else(|| FdarError::InvalidParameter {
        parameter: "nbasis",
        message: format!("basis projection failed (nbasis={nbasis}, m={m})"),
    })?;
let coeff = proj.coefficients;   // FdMatrix, shape (n, p)
let p = proj.n_basis;            // actual nbasis (may differ from requested for B-spline)
```

CRITICAL: Always use `proj.n_basis` (not the caller-supplied `nbasis`) as `p` for all subsequent loops, because B-spline construction clamps: `nbasis.saturating_sub(4).max(2)` (line 70 of projection.rs).

**`iter_maybe_parallel!` usage pattern** (lines 124-137 of projection.rs):
```rust
let rows: Vec<Vec<f64>> = iter_maybe_parallel!(0..n)
    .map(|i| {
        let curve: Vec<f64> = (0..m).map(|j| data[(i, j)]).collect();
        (0..actual_nbasis)
            .map(|k| { ... })
            .collect::<Vec<_>>()
    })
    .collect();
```

Use the same `iter_maybe_parallel!(0..p)` form for parallelizing the per-component rank-transform and the O(p²) interval-p-value matrix build. The sequential permutation loop (single RNG state) must remain a plain `for _ in 0..n_perm` loop.

---

### `src/inference/mod.rs` (edit) — module declaration + pub use

**Analog:** `src/inference/mod.rs` lines 29-40 (exact pattern to extend)

**Existing pattern:**
```rust
mod anova;
mod dist;
mod flm;
mod hotelling;
mod permutation;
mod scb;

pub use anova::oneway_anova_vstat;
pub use flm::{flm_f_test, flm_gof_test};
pub use hotelling::two_sample_mean_test;
pub use permutation::{f_perm_test, t_perm_test, DEFAULT_N_PERM};
pub use scb::{mean_scb, scb_two_sample_test};
```

**Add after `mod scb;`:**
```rust
mod itp;
pub use itp::{itp_flm, itp_one_pop, itp_two_pop, ItpResult};
```

---

### `src/lib.rs` (edit) — inference re-export block

**Analog:** `src/lib.rs` lines 225-229 (exact block to extend)

**Existing block:**
```rust
// Re-export functional inference types and two-sample tests
pub use inference::{
    f_perm_test, flm_f_test, flm_gof_test, mean_scb, oneway_anova_vstat, scb_two_sample_test,
    t_perm_test, two_sample_mean_test, TestResult, DEFAULT_N_PERM,
};
```

**Replace with:**
```rust
// Re-export functional inference types and two-sample tests
pub use inference::{
    f_perm_test, flm_f_test, flm_gof_test, itp_flm, itp_one_pop, itp_two_pop,
    mean_scb, oneway_anova_vstat, scb_two_sample_test,
    t_perm_test, two_sample_mean_test, ItpResult, TestResult, DEFAULT_N_PERM,
};
```

---

## Shared Patterns

### Error Handling (InvalidDimension / InvalidParameter)
**Source:** `src/inference/permutation.rs` lines 24-59, 160-164
**Apply to:** All three entry points in `itp.rs`

Validation order per entry point:
1. Grid/column check (`m == 0`, `m_a != m_b`)
2. `argvals.len() != m` → `InvalidDimension { parameter: "argvals", ... }`
3. Minimum rows (`n < 2` or `n_a < 2 || n_b < 2`) → `InvalidDimension { parameter: "data", ... }`
4. `nbasis < 2` → `InvalidParameter { parameter: "nbasis", ... }`
5. `n_perm == 0` → `InvalidParameter { parameter: "n_perm", ... }`
6. For `itp_flm`: `y.len() != n` → `InvalidDimension { parameter: "y", ... }`

### Conditional Serde
**Source:** `src/inference/mod.rs` line 48
**Apply to:** `ItpResult` struct
```rust
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
```

### RNG Seeding Convention
**Source:** `src/inference/permutation.rs` line 173
**Apply to:** All three entry points

```rust
let mut rng = StdRng::seed_from_u64(seed);
```

Note: ITP uses a single sequential loop (not per-thread), so no `seed + k` offset is needed. The `use rand::Rng;` trait import must be inside the function scope (same as `shuffle_labels` at line 121 of permutation.rs).

### `iter_maybe_parallel!` usage
**Source:** `src/basis/projection.rs` lines 124-137; `src/parallel.rs` lines 41-55
**Apply to:** rank-transform step, O(p²) interval p-value matrix loop in `itp.rs`

```rust
// Parallelize over components k (or interval rows) where the computation is independent
let results: Vec<_> = iter_maybe_parallel!(0..p)
    .map(|k| { ... })
    .collect();
```

Sequential permutation loop must NOT use `iter_maybe_parallel!` (single shared `rng` state).

### Column-Major FdMatrix Access
**Source:** Throughout `src/inference/permutation.rs`, `src/matrix.rs`
**Apply to:** All coefficient matrix access in `itp.rs`

```rust
coeff[(i, k)]   // row i = curve, col k = basis component
```

---

## No Analog Found

None — all patterns have exact or role-match analogs in the codebase.

---

## Pitfall Reminders (for planner)

| Pitfall | Analog Location | Guard |
|---------|----------------|-------|
| `shuffle_labels` is `fn`, not `pub fn` | permutation.rs line 120 | Duplicate 7-line helper as `fn shuffle_itp` in itp.rs |
| `pool_two_samples` is `fn`, not `pub fn` | permutation.rs line 64 | Duplicate as `fn pool_coefficients_itp` in itp.rs (operates on FdMatrix coefficients) |
| `fdata_to_basis` returns `Option`, not `Result` | projection.rs line 103 | `.ok_or_else(|| FdarError::InvalidParameter { ... })?` at call site |
| `proj.n_basis` may differ from requested `nbasis` | projection.rs line 70 | Use `proj.n_basis` as `p` throughout; store in `ItpResult.n_basis` |
| Fisher combination requires log-safe p-values | RESEARCH.md Pitfall 4 | `v.max(1e-300).ln()` in all Fisher combination calls |
| `pval_correct` reverses output at end | RESEARCH.md Pitfall 3 | `corrected.reverse()` — verified by localized-difference test |

## Metadata

**Analog search scope:** `src/inference/`, `src/basis/`, `src/parallel.rs`, `src/lib.rs`
**Files read:** 5 source files (permutation.rs, mod.rs, flm.rs lines 1-80, projection.rs, parallel.rs lines 1-60, lib.rs lines 220-229)
**Pattern extraction date:** 2026-08-20
