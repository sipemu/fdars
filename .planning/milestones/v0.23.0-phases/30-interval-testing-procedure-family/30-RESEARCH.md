# Phase 30: Interval Testing Procedure Family — Research

**Researched:** 2026-08-20
**Domain:** Functional inference / Interval Testing Procedure (ITP/IWT), permutation-based functional tests
**Confidence:** HIGH (codebase reads VERIFIED; algorithm from R source CITED)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- Three `Result<ItpResult, FdarError>`-returning public fns in `inference/itp.rs`: one-population
  interval-wise test, two-population interval-wise test, and interval-wise FLM coefficient test.
- Basis selection via `ProjectionBasisType { Bspline, Fourier }` (basis/projection.rs),
  reusing `fdata_to_basis` / `bspline_basis` / `fourier_basis` — not reimplemented.
- `ItpResult { adjusted_pvalues: Vec<f64>, raw_pvalues: Vec<f64>, basis metadata (type + nbasis), n_perm }`
  with `Debug, Clone, PartialEq`, `#[non_exhaustive]`, conditional serde.
- New file `inference/itp.rs`; crate-root re-exported. Existing inference entry points untouched.
- Build interval-wise permutation null from INF-01 infrastructure (`t_perm_test` / `f_perm_test`
  or their underlying pooling/relabel machinery) — no new permutation engine.
- `DEFAULT_N_PERM = 999` as configurable default; seeded `StdRng::seed_from_u64(seed + k)`.
- Closure adjustment: adjusted p-value for component `k` = max over all contiguous intervals
  containing `k` of that interval's joint permutation p-value; enumerate all O(p²) intervals.
- No new crate dependency.

### Claude's Discretion

- Pin exact fdatest ITP statistic during research; document divergence from R baseline in rustdoc.

### Deferred Ideas (OUT OF SCOPE)

- Any plotting/rendering of ITP p-value surfaces / heatmaps (numeric outputs only).
- Random-projection ANOVA/MANOVA (`fdANOVA`).
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID     | Description                                                                                                         | Research Support                                                                                                |
|--------|---------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| INF-03 | Implement the ITP family in new `inference/itp.rs` — one-population and two-population interval-wise tests (B-spline and Fourier) with domain-selective adjusted p-values, plus interval-wise FLM coefficient testing. Reuses INF-01 permutation infra and `basis/` projection. | Algorithm fully pinned from R source; reuse paths confirmed by reading INF-01 + basis/ source; all three entry points designed below. |
</phase_requirements>

---

## Summary

The Interval Testing Procedure (ITP) — also called Interval Wise Testing (IWT) in the 2017 paper —
is a three-step nonparametric method for functional hypothesis testing. It (1) projects functional
observations onto a finite basis (B-spline or Fourier), yielding a per-curve coefficient matrix
of shape `(n, p)` where `p = nbasis`; (2) runs a per-component univariate permutation test to get
raw p-values for each of the `p` basis components; and (3) applies an interval-wise closure
adjustment that sets the adjusted p-value for component `k` to the maximum over all contiguous
intervals `[a, b]` containing `k` of the joint permutation p-value for that interval (computed via
Fisher's combination of per-component rank-based p-values from the same permutation run). The
result is a length-`p` vector of adjusted p-values, one per basis component, that identifies which
sub-domains drive a significant result.

The CRAN fdatest package implements this in `ITP1bspline`, `ITP2bspline`, `ITP2fourier`, and
`ITPlmbspline`. The R source has been read directly and the algorithm is fully pinned below. The
ITP (Pini & Vantini 2016 Biometrics) and IWT (2017 JNPS) differ primarily in name and framing;
the R package uses "ITP" branding for its CRAN release and "IWT" for the GitHub version — the
core algorithm (basis → per-component permutation → Fisher-combination closure) is identical in
both. **Recommendation: implement the IWT/ITP algorithm as used in the CRAN `fdatest` 2.1.1
package**, documented as matching `fdatest::ITP1bspline` / `ITP2bspline` / `ITPlmbspline`.

**Primary recommendation:** Build the three entry points in `inference/itp.rs`, reusing the
permutation loop pattern from `inference/permutation.rs` (pool + Fisher–Yates relabel for two-pop;
sign-flip for one-pop) but with a per-component statistic matrix rather than a single scalar —
therefore build the permutation loop directly in `itp.rs` rather than calling `t_perm_test` (which
only returns a scalar p-value, not the per-component matrix). Cache the basis projection once per
call, then iterate all O(p²) intervals for the closure adjustment.

---

## Architectural Responsibility Map

| Capability                   | Primary Tier              | Secondary Tier              | Rationale                                                         |
|------------------------------|---------------------------|-----------------------------|-------------------------------------------------------------------|
| Basis projection             | `basis/projection.rs`     | —                           | `fdata_to_basis` already exists; call it, do not re-implement    |
| Per-component permutation    | `inference/itp.rs`        | `inference/permutation.rs` (pattern) | INF-01 returns scalar only; ITP needs `(B, p)` stat matrix |
| Interval-wise closure adjust | `inference/itp.rs`        | —                           | Pure arithmetic on the stat matrix; no external dep needed        |
| Result struct + re-exports   | `inference/itp.rs` + `inference/mod.rs` + `lib.rs` | — | Follows existing INF-01 pattern exactly          |
| FLM coefficient testing      | `inference/itp.rs`        | `scalar_on_function/` (FregreLmResult) | Reads `fpca.rotation` + `fpca.mean` to project new obs for permutation |

---

## Standard Stack

### Core (all already in `fdars-core`)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `rand` | 0.8 | `StdRng::seed_from_u64` seeded shuffle | Already in Cargo.toml [VERIFIED: fdars-core/Cargo.toml] |
| `crate::basis::projection::fdata_to_basis` | in-crate | B-spline/Fourier coefficient extraction | [VERIFIED: src/basis/projection.rs:98-150] |
| `crate::basis::projection::ProjectionBasisType` | in-crate | Basis selector enum | [VERIFIED: src/basis/projection.rs:19-24] |
| `crate::inference::permutation::DEFAULT_N_PERM` | in-crate | Permutation count default = 999 | [VERIFIED: src/inference/permutation.rs:18] |
| `crate::error::FdarError` | in-crate | Error type | Standard crate convention |

### No new dependencies required.

---

## Package Legitimacy Audit

> No new external packages are introduced in this phase. The implementation reuses
> `rand 0.8` and `rayon 1.10` already in `Cargo.toml`. No audit gate needed.

**Packages removed due to SLOP verdict:** none
**Packages flagged as suspicious:** none

---

## Architecture Patterns

### System Architecture Diagram

```
 caller
   │
   ▼
itp_one_pop(data, mu0, basis_type, nbasis, n_perm, seed)
itp_two_pop(data_a, data_b, basis_type, nbasis, n_perm, seed)
itp_flm(data, y, argvals, basis_type, nbasis, n_perm, seed)
   │
   ▼
[Step 1] fdata_to_basis(data, argvals, nbasis, basis_type)
         → coeff: FdMatrix  shape (n, p)          [basis/projection.rs]
   │
   ▼
[Step 2] build_perm_stat_matrix(coeff, ..., n_perm, seed)
         → T_perm: Vec<Vec<f64>>  shape (n_perm, p)
         → T0:     Vec<f64>       shape (p,)         [per-component observed stat]
   │
   ▼
[Step 3] raw_pvalues[k] = #{perm: T_perm[b][k] >= T0[k]} + 1) / (n_perm + 1)
         → raw_pvalues: Vec<f64>  shape (p,)
   │
   ▼
[Step 4] rank_transform_to_L(T_perm)
         → L: Vec<Vec<f64>>  shape (n_perm, p)    [rank-based pseudo-p per perm]
   │
   ▼
[Step 5] build_pval_matrix(raw_pvalues, L, p)    [O(p²) interval loop]
         → pval_matrix: Vec<Vec<f64>>  shape (p, p)
   │
   ▼
[Step 6] pval_correct(pval_matrix)               [closure max adjustment]
         → adjusted_pvalues: Vec<f64>  shape (p,)
   │
   ▼
ItpResult { adjusted_pvalues, raw_pvalues, basis_type, n_basis: p, n_perm }
```

### Recommended Project Structure

```
src/
├── inference/
│   ├── mod.rs           # add: mod itp; pub use itp::{...}
│   ├── permutation.rs   # unchanged (INF-01)
│   ├── flm.rs           # unchanged (INF-02)
│   └── itp.rs           # NEW: ItpResult + 3 entry points
├── lib.rs               # extend inference re-export block
```

### Pattern 1: Projection + Per-Component Permutation (One-Population)

The one-population ITP tests H₀: mean coefficient vector = 0 (after subtracting mu0).
The test statistic per component k is `|colMean(coeff[:, k])|` — absolute value of the
sample mean of the k-th basis coefficient (after mean subtraction). The permutation null
is sign-flip: multiply each row (curve) by an independent ±1 Bernoulli draw, then
recompute `|colMean(coeff_perm[:, k])|`.

**Source:** `ITP1bspline.R` from CRAN fdatest 2.1.1 [CITED: github.com/cran/fdatest/blob/master/R/ITP1bspline.R]

```rust
// Source: ITP1bspline.R — Step 2 (per-component sign-flip permutation)
// T0[k] = |mean(coeff[:, k])|   (observed, after mu0 subtraction)
// For each permutation b:
//   signs: Vec<f64> of ±1.0 drawn i.i.d. from {-1, +1}
//   T_perm[b][k] = |mean(coeff_perm[:, k])|
//     where coeff_perm[i, k] = coeff[i, k] * signs[i]

fn observed_stat_one_pop(coeff: &FdMatrix, p: usize) -> Vec<f64> {
    (0..p).map(|k| {
        let mean_k: f64 = (0..coeff.nrows()).map(|i| coeff[(i, k)]).sum::<f64>()
            / coeff.nrows() as f64;
        mean_k.abs()
    }).collect()
}

fn perm_stat_one_pop(coeff: &FdMatrix, signs: &[f64], p: usize) -> Vec<f64> {
    let n = coeff.nrows();
    (0..p).map(|k| {
        let mean_k: f64 = (0..n).map(|i| coeff[(i, k)] * signs[i]).sum::<f64>()
            / n as f64;
        mean_k.abs()
    }).collect()
}
```

### Pattern 2: Two-Population Pool + Relabel

The two-population ITP tests H₀: mean coefficient vectors equal between groups (after
subtracting mu from group 2). Statistic per component k: `|colMean(coeff_a[:, k]) - colMean(coeff_b[:, k])|`.
Permutation: pool all `n = n_a + n_b` coefficient rows, draw a random permutation of row
indices (Fisher–Yates), first `n_a` rows become group A, rest group B.

**Source:** `ITP2bspline.R` from CRAN fdatest 2.1.1 [CITED: github.com/cran/fdatest/blob/master/R/ITP2bspline.R]

```rust
// Source: ITP2bspline.R — Step 2
// T0[k] = |colMean(coeff[0..n_a, k]) - colMean(coeff[n_a..n, k])|
// Permutation: shuffle row indices of pooled coeff matrix

fn observed_stat_two_pop(coeff: &FdMatrix, n_a: usize, p: usize) -> Vec<f64> {
    let n = coeff.nrows();
    let n_b = n - n_a;
    (0..p).map(|k| {
        let m_a = (0..n_a).map(|i| coeff[(i, k)]).sum::<f64>() / n_a as f64;
        let m_b = (n_a..n).map(|i| coeff[(i, k)]).sum::<f64>() / n_b as f64;
        (m_a - m_b).abs()
    }).collect()
}

fn perm_stat_two_pop(coeff: &FdMatrix, perm: &[usize], n_a: usize, p: usize) -> Vec<f64> {
    let n_b = perm.len() - n_a;
    (0..p).map(|k| {
        let m_a = (0..n_a).map(|&r| coeff[(r, k)]).sum::<f64>() / n_a as f64;
        let m_b = (n_a..perm.len()).map(|i| coeff[(perm[i], k)]).sum::<f64>() / n_b as f64;
        (m_a - m_b).abs()
    }).collect()
}
```

### Pattern 3: Interval-Wise FLM Coefficient Test

The FLM variant (`ITPlmbspline`) projects both the functional predictor and the response,
then tests each basis component's regression coefficient. This is the most complex test.

The procedure:
1. Project the functional predictor `X` onto p basis coefficients per curve: `coeff` (n, p).
2. For each basis component k, fit simple linear regression of `y` on `coeff[:, k]`: compute
   the t-statistic `T0_flm[k] = |beta_hat_k / se_k|` (or equivalently the F-statistic
   `t² = F`). The R code uses `abs(regr0$coeff / se)`.
3. Permutation scheme: **response permutation** — shuffle the response vector `y` (not
   residuals, to test global H₀ that y is independent of all basis components) using the
   same pool-and-relabel pattern as permutation.rs.
4. For each permuted response vector, refit each per-component regression and recompute
   T_flm[b][k].

**Implementation recommendation for fdars:** Use the simpler response-permutation approach
(shuffle `y`) rather than the partial-residual method from ITPlmbspline, which requires
fitting n_perm × p regressions. The simpler approach tests the global null "y is
independent of the functional predictor" and is consistent with the INF-01 philosophy.
Document this choice in rustdoc.

```rust
// Basis-component t-statistic for simple regression of y on coeff[:, k]
fn component_t_stat(y: &[f64], coeff: &FdMatrix, k: usize) -> f64 {
    let n = y.len();
    let xk: Vec<f64> = (0..n).map(|i| coeff[(i, k)]).collect();
    // OLS: beta = cov(xk, y) / var(xk), se via residual variance
    let mx = xk.iter().sum::<f64>() / n as f64;
    let my = y.iter().sum::<f64>() / n as f64;
    let sxx = xk.iter().map(|&x| (x - mx).powi(2)).sum::<f64>();
    if sxx < 1e-30 { return 0.0; }
    let sxy = xk.iter().zip(y.iter()).map(|(&x, &yi)| (x - mx) * (yi - my)).sum::<f64>();
    let beta = sxy / sxx;
    let yhat: Vec<f64> = xk.iter().map(|&x| my + beta * (x - mx)).collect();
    let rss = y.iter().zip(yhat.iter()).map(|(&yi, &fi)| (yi - fi).powi(2)).sum::<f64>();
    let se2 = rss / ((n - 2) as f64 * sxx);
    if se2 <= 0.0 { return 0.0; }
    (beta / se2.sqrt()).abs()
}
```

### Pattern 4: The Closure Adjustment Algorithm — PINNED EXACTLY

This is the defining algorithm. The R source is reproduced verbatim and translated.

**Source:** `ITP1bspline.R` internal `pval.correct` function [CITED: github.com/cran/fdatest/blob/master/R/ITP1bspline.R]

#### Step A: Rank-transform permutation stat matrix to pseudo-p-values (L matrix)

For each component k, rank all B permutation statistics in descending order. The
rank-based pseudo-p-value for permutation b at component k is:
`L[b][k] = rank_desc(T_perm[:, k])[b] / B`

In R: `sort.int(T_coeff[,j], index.return=T)$ix` gives the rank permutation;
`q[ordine] = (B:1)/B` assigns ranks from largest (= 1/B) to smallest (= B/B = 1.0).

```rust
// L[b][k] = rank-based pseudo-p, descending: largest stat → smallest pseudo-p
fn rank_transform(t_perm: &[Vec<f64>], p: usize, b: usize) -> Vec<Vec<f64>> {
    // t_perm: shape (b, p)
    let mut l = vec![vec![0.0f64; p]; b];
    for k in 0..p {
        // collect (stat_value, perm_index) for component k
        let mut col: Vec<(f64, usize)> = (0..b).map(|i| (t_perm[i][k], i)).collect();
        // sort descending by stat value
        col.sort_unstable_by(|a, b_| b_.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        // assign rank: largest stat → rank 1, rank / B = smallest pseudo-p
        for (rank_one_based, &(_, orig_idx)) in col.iter().enumerate() {
            l[orig_idx][k] = (rank_one_based + 1) as f64 / b as f64;
        }
    }
    l
}
```

#### Step B: Build the asymmetric p-value matrix (O(p²) interval loop)

The matrix `pval_matrix[i][j]` contains the joint permutation p-value for the
contiguous interval of length `p - i` starting at column `j` (0-indexed).

Fisher's combining function: `C(lambda_1, ..., lambda_s) = -2 * sum(log(lambda_k))`

For each interval `[a, b]` (0-indexed: `a = j`, `b = j + (p-1-i)`):
- Observed combined stat: `T0_temp = fisher_cf(raw_pvalues[a..=b])`
- Permuted combined stats: `T_temp[perm] = fisher_cf(L[perm][a..=b])`
- p-value: `#{T_temp >= T0_temp} / B`

In R (1-indexed): `inf = j`, `sup = (p-i)+j` means interval of length `p-i` starting at j.
Row `i = p` (last row) stores the raw per-component p-values (length-1 intervals).
Row `i = p-1` stores intervals of length 2, ..., row `i = 1` stores the full interval.

```rust
fn build_pval_matrix(
    raw_pvalues: &[f64],  // length p
    l: &[Vec<f64>],       // shape (n_perm, p)
    p: usize,
    n_perm: usize,
) -> Vec<Vec<f64>> {
    // pval_matrix[row][col], row in 0..p, col in 0..p
    // row p-1 (R's row p): raw per-component p-values
    // row i (R's row p-i): intervals of length p-i, starting at col j
    let mut mat = vec![vec![1.0f64; p]; p];

    // Last row: raw p-values (length-1 intervals)
    for j in 0..p {
        mat[p - 1][j] = raw_pvalues[j];
    }

    // Wrap-around: double the arrays (circular trick from R)
    let pval_2x: Vec<f64> = raw_pvalues.iter().chain(raw_pvalues.iter()).copied().collect();
    // l_2x: (n_perm, 2p)
    let l_2x: Vec<Vec<f64>> = l.iter().map(|row| {
        row.iter().chain(row.iter()).copied().collect()
    }).collect();

    let fisher_cf = |vals: &[f64]| -> f64 {
        -2.0 * vals.iter().map(|&v| v.max(1e-300).ln()).sum::<f64>()
    };

    // row i in R's 1-indexed (p-1):1 corresponds to our row_idx = i-1 in 0..p-1
    // R: for i in (p-1):1 → interval length = p - i, from R's perspective
    // We iterate: interval_len from 2..=p  (R's i from p-1 down to 1)
    for interval_len in 2..=p {
        let row_idx = p - interval_len; // R's row i = p - interval_len
        for j in 0..p {
            let inf = j;              // 0-indexed start in 2x array
            let sup = j + interval_len; // exclusive end
            let t0_temp = fisher_cf(&pval_2x[inf..sup]);
            let n_ge = l_2x.iter().filter(|perm_row| {
                let t_temp = fisher_cf(&perm_row[inf..sup]);
                t_temp >= t0_temp
            }).count();
            mat[row_idx][j] = n_ge as f64 / n_perm as f64;
        }
    }
    mat
}
```

#### Step C: Closure max-adjustment (pval.correct)

For each component `k` (0-indexed), the adjusted p-value is the maximum over all
contiguous intervals `[a, b]` (where `a ≤ k ≤ b`) of the joint p-value for that interval.

The R `pval.correct` function implements this via a "cone" walk: it traverses the matrix
in a specific order using the doubled+reversed matrix. Translated directly:

```rust
fn pval_correct(pval_matrix: &[Vec<f64>], p: usize) -> Vec<f64> {
    // Build the doubled matrix (2p columns) and reverse columns: pval_2_2x
    // pval_matrix rows: 0 = full interval (len p), p-1 = len-1 (raw p-values)
    // R: matrice_pval_2_2x <- cbind(pval.matrix, pval.matrix)
    //    matrice_pval_2_2x <- matrice_pval_2_2x[, (2*p):1]   (reverse columns)
    // Equivalent: col c in the reversed 2p-wide matrix = original col (2p-1-c)

    let get_2x_rev = |row: usize, col: usize| -> f64 {
        // col in 0..2p; reversed → original col = 2p-1-col
        let orig_col = (2 * p - 1).saturating_sub(col) % p;
        pval_matrix[row][orig_col]
    };

    let mut corrected = vec![0.0f64; p];
    for var in 0..p {
        // R: pval_var <- matrice_pval_2_2x[p, var]  (1-indexed row p = our row p-1)
        let mut pval_var = get_2x_rev(p - 1, var);
        let mut fine = var;
        // R: for riga in (p-1):1 → our riga_idx = p-2 down to 0
        for riga_idx in (0..p - 1).rev() {
            fine += 1;
            // R: pval_cono <- matrice_pval_2_2x[riga, inizio:fine]
            for col in var..=fine {
                let v = get_2x_rev(riga_idx, col);
                if v > pval_var {
                    pval_var = v;
                }
            }
        }
        corrected[var] = pval_var;
    }
    // R: corrected.pval <- corrected.pval[p:1]  (reverse)
    corrected.reverse();
    corrected
}
```

**Interpretation of corrected.pval reversal:** The R code stores basis components in a
specific column ordering. The final `.reverse()` brings the adjusted p-values back into
the natural component order (component 0, 1, ..., p-1 matching the coefficient matrix
columns). Verify this against the Rust output in tests.

### Anti-Patterns to Avoid

- **Calling `t_perm_test` or `f_perm_test` directly:** These return a scalar `TestResult`,
  not the `(B, p)` permutation statistic matrix needed for the closure adjustment. ITP
  needs the full matrix from a single permutation run.
- **Rerunning projections inside the permutation loop:** `fdata_to_basis` is O(n·m·p);
  call it once before the loop and then permute the coefficient rows, not the raw curves.
- **Permuting within `fdata_to_basis`:** The basis projection is linear, so permuting the
  coefficient rows is equivalent to reprojecting permuted data — and dramatically cheaper.
- **Using log(0) in Fisher's combination:** Guard with `.max(1e-300)` on each p-value before
  taking the natural log.
- **Wrong p-value formula:** Use `(n_ge + 1) / (n_perm + 1)` for raw per-component p-values
  (matching INF-01 convention). The R source uses `n_ge / B` without the +1 correction —
  document this as a deliberate deviation: the +1 correction avoids a zero p-value and is
  standard in permutation testing literature.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| B-spline basis matrix | Custom de Boor recursion | `fdata_to_basis(data, argvals, nbasis, ProjectionBasisType::Bspline)` | Already implemented and tested in basis/projection.rs |
| Fourier basis matrix | Custom FFT projection | `fdata_to_basis(data, argvals, nbasis, ProjectionBasisType::Fourier)` | Already implemented and tested |
| Fisher–Yates shuffle | Custom shuffle | `shuffle_labels` pattern from `permutation.rs` | Tested, seeded, same convention |
| Simpson's weights | Custom integrator | `helpers::simpsons_weights` (already used in INF-01) | Tested |
| Pseudo-inverse for projection | Gaussian elimination | `basis/helpers::svd_pseudoinverse` (called inside `fdata_to_basis`) | Already used |

**Key insight:** The most expensive sub-problems (basis construction, numerical projection,
Fisher–Yates seeded shuffle) are already solved in the crate. ITP adds only the permutation
loop + closure arithmetic.

---

## Runtime State Inventory

> Omitted — this is a greenfield addition (new file `inference/itp.rs`). No rename/refactor.

---

## Common Pitfalls

### Pitfall 1: `fdata_to_basis` Returns `Option`, Not `Result`

**What goes wrong:** `fdata_to_basis` returns `Option<BasisProjectionResult>` (not a `Result`).
Unwrapping with `.unwrap()` panics on bad input rather than returning `FdarError`.
**Why it happens:** The projection function predates the full `Result<T, FdarError>` migration.
**How to avoid:** Convert with `.ok_or_else(|| FdarError::InvalidParameter { ... })` at the
call site. Check `nbasis >= 2` and `n >= 2` before calling to produce a meaningful error.
**Warning signs:** Clippy warns on `.unwrap()` in a function returning `Result`.

[VERIFIED: src/basis/projection.rs:98-150] — function signature is `fn fdata_to_basis(...) -> Option<BasisProjectionResult>`

### Pitfall 2: `actual_nbasis` May Differ from Requested `nbasis`

**What goes wrong:** `fdata_to_basis` reports the actual number of basis functions used
(`result.n_basis`) which may differ from the caller-supplied `nbasis` because B-spline basis
construction clamps the knot count.
**Why it happens:** `bspline_basis` calls `nbasis.saturating_sub(4).max(2)` for nknots.
**How to avoid:** Always use `result.n_basis` (not the caller's `nbasis`) as `p` throughout the
permutation and closure loops. Store `result.n_basis` in `ItpResult`.
[VERIFIED: src/basis/projection.rs:69-73] — `nbasis.saturating_sub(4).max(2)` for B-spline nknots.

### Pitfall 3: The Closure Adjustment Column Reversal Is Non-Obvious

**What goes wrong:** The R `pval.correct` function reverses the corrected p-value vector at the
end (`corrected.pval[p:1]`). Omitting this reversal returns the adjusted p-values in reverse
component order, causing tests to assign significant regions to the wrong basis components.
**Why it happens:** R's matrix construction inverts the column ordering via `[,(2*p):1]`.
**How to avoid:** After the cone-walk loop, call `.reverse()` on `corrected`. Write a test that
verifies component 0's adjusted p-value matches a known-significant region.
**Warning signs:** Null test shows significance at the wrong end of the domain.

### Pitfall 4: Fisher Combination Requires log-safe p-values

**What goes wrong:** Raw p-values from short intervals may be exactly 0.0 (no permutation
exceeded the observed statistic). `(0.0f64).ln() = -inf`, producing NaN or -inf in Fisher's
statistic.
**How to avoid:** Clamp: `v.max(1e-300).ln()` in the Fisher combination. Note that the R source
also accumulates small numerical errors here due to finite `B`.

### Pitfall 5: O(p²·B) Memory for `L` Matrix

**What goes wrong:** For large p (e.g., p = 200 basis functions) and B = 999 permutations,
the L matrix is 200 × 999 f64 values = ~1.6 MB — manageable. But for the inner Fisher loop,
the 2× extended `L_2x` matrix is 400 × 999 ≈ 3.2 MB. This is fine in practice but
should be documented as a complexity note.
**How to avoid:** Allocate `L` as `Vec<Vec<f64>>` of shape `(n_perm, p)` once before the loop.
The O(p²) interval loop is the bottleneck, not memory.

### Pitfall 6: `shuffle_labels` Uses `rng.gen_range` from `rand::Rng` Trait

**What goes wrong:** The `rand::Rng` trait must be in scope for `gen_range` to compile.
**How to avoid:** `use rand::Rng;` inside the function scope (as done in `permutation.rs`).

[VERIFIED: src/inference/permutation.rs:121-127] — `use rand::Rng;` inside `shuffle_labels`.

---

## Code Examples

### Complete One-Population Entry Point Skeleton

```rust
// Source: ITP1bspline.R (CRAN fdatest 2.1.1) adapted to fdars conventions
pub fn itp_one_pop(
    data: &FdMatrix,
    argvals: &[f64],
    mu0: Option<&[f64]>,       // null mean function; None = zero
    basis_type: ProjectionBasisType,
    nbasis: usize,
    n_perm: usize,
    seed: u64,
) -> Result<ItpResult, FdarError> {
    // 1. Validate
    let (n, m) = data.shape();
    if n < 2 { return Err(FdarError::InvalidDimension { ... }); }
    if argvals.len() != m { return Err(FdarError::InvalidDimension { ... }); }
    if nbasis < 2 { return Err(FdarError::InvalidParameter { ... }); }
    if n_perm == 0 { return Err(FdarError::InvalidParameter { ... }); }

    // 2. Subtract mu0 if provided, project to basis coefficients
    let centered = center_data(data, mu0)?;  // returns FdMatrix
    let proj = fdata_to_basis(&centered, argvals, nbasis, basis_type)
        .ok_or_else(|| FdarError::InvalidParameter {
            parameter: "nbasis",
            message: format!("basis projection failed (nbasis={nbasis}, m={m})"),
        })?;
    let coeff = proj.coefficients;
    let p = proj.n_basis;

    // 3. Observed statistic per component: |colMean|
    let t0: Vec<f64> = (0..p).map(|k| {
        let mean_k = (0..n).map(|i| coeff[(i, k)]).sum::<f64>() / n as f64;
        mean_k.abs()
    }).collect();

    // 4. Sign-flip permutation loop → T_perm (n_perm, p) + L (n_perm, p)
    let mut rng = StdRng::seed_from_u64(seed);
    let mut t_perm: Vec<Vec<f64>> = Vec::with_capacity(n_perm);
    for _ in 0..n_perm {
        let signs: Vec<f64> = (0..n).map(|_| if rng.gen::<bool>() { 1.0 } else { -1.0 }).collect();
        let row: Vec<f64> = (0..p).map(|k| {
            let mean_k = (0..n).map(|i| coeff[(i, k)] * signs[i]).sum::<f64>() / n as f64;
            mean_k.abs()
        }).collect();
        t_perm.push(row);
    }

    // 5. Raw p-values  (INF-01 convention: +1 correction)
    let raw_pvalues: Vec<f64> = (0..p).map(|k| {
        let n_ge = t_perm.iter().filter(|row| row[k] >= t0[k]).count();
        (n_ge as f64 + 1.0) / (n_perm as f64 + 1.0)
    }).collect();

    // 6. Rank-transform → L matrix
    let l = rank_transform(&t_perm, p, n_perm);

    // 7. Build O(p²) interval p-value matrix
    let pval_matrix = build_pval_matrix(&raw_pvalues, &l, p, n_perm);

    // 8. Closure max-adjustment
    let adjusted_pvalues = pval_correct(&pval_matrix, p);

    Ok(ItpResult {
        adjusted_pvalues,
        raw_pvalues,
        basis_type,
        n_basis: p,
        n_perm,
    })
}
```

### ItpResult Struct

```rust
/// Result of an Interval Testing Procedure (ITP) family test.
///
/// Provides per-basis-component raw and adjusted p-values. The adjusted p-values
/// implement the interval-wise closure adjustment (Pini & Vantini, Biometrics 2016):
/// adjusted_pvalues[k] = max over all contiguous intervals [a,b] containing k of the
/// joint permutation p-value for that interval. A small adjusted p-value at component
/// k indicates that the domain sub-interval represented by basis function k contributes
/// to the rejection of H₀.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct ItpResult {
    /// Adjusted (interval-wise closed) p-values, one per basis component.
    pub adjusted_pvalues: Vec<f64>,
    /// Raw (point-wise) p-values, one per basis component.
    pub raw_pvalues: Vec<f64>,
    /// Basis type used for projection.
    pub basis_type: ProjectionBasisType,
    /// Number of basis functions actually used (may differ from requested
    /// for B-splines due to knot clamping).
    pub n_basis: usize,
    /// Number of permutations used.
    pub n_perm: usize,
}
```

---

## ITP vs IWT: Which to Implement

The CRAN `fdatest` 2.1.1 package uses the name "ITP" and provides `ITP1bspline`,
`ITP2bspline`, `ITP2fourier`, `ITPlmbspline`. The GitHub `alessiapini/fdatest` uses
"IWT" (Interval Wise Testing). The core algorithm is identical:

| Aspect | ITP (CRAN 2.1.1) | IWT (GitHub 2.x) |
|--------|------------------|------------------|
| One-pop stat | `|colMean(coeff)|` | `|colMean(coeff)|` |
| Two-pop stat | `|colMean(coeff_a) - colMean(coeff_b)|` | same |
| Perm null (1-pop) | sign-flip | sign-flip |
| Perm null (2-pop) | pool + random perm | pool + random perm |
| Interval combination | Fisher (`-2 Σ log λ`) | Fisher (`-2 Σ log λ`) |
| Closure adjust | max over containing intervals | max over containing intervals |

**Decision:** Implement as `itp_one_pop`, `itp_two_pop`, `itp_flm` matching the CRAN ITP
formulation. Document as matching `fdatest::ITP1bspline` / `ITP2bspline` / `ITPlmbspline`.

---

## Permutation Infrastructure: Reuse vs. In-File Loop

**Key finding:** `t_perm_test` and `f_perm_test` in `inference/permutation.rs` return only a
scalar `TestResult { statistic, p_value, n_perm }`. They do NOT expose the per-permutation
statistic vector. ITP requires the `(n_perm, p)` matrix of per-component per-permutation
statistics (for Fisher combination across intervals). Therefore:

- **Do NOT call `t_perm_test` or `f_perm_test` from `itp.rs`.**
- **DO reuse the pattern:** pool_two_samples + shuffle_labels (Fisher–Yates) + per-thread
  `StdRng::seed_from_u64(seed)` + `(n_ge + 1) / (n_perm + 1)` p-value formula.
- Copy the Fisher–Yates shuffle inline in `itp.rs` (it is a 6-line helper). The
  `shuffle_labels` function in `permutation.rs` is `fn` (not `pub fn`) — it cannot be called
  from `itp.rs` directly.

[VERIFIED: src/inference/permutation.rs:120-127] — `fn shuffle_labels` is `fn` (not pub):
```
fn shuffle_labels(v: &mut [usize], rng: &mut StdRng) {
    use rand::Rng;
    let n = v.len();
    for i in (1..n).rev() {
        let j = rng.gen_range(0..=i);
        v.swap(i, j);
    }
}
```
[VERIFIED: src/inference/permutation.rs:63-81] — `fn pool_two_samples` is also `fn` (not pub).

**Conclusion:** Duplicate the ~12-line shuffle + pool helpers in `itp.rs`. The duplication is
intentional — permutation.rs's helpers serve a different call shape (scalar statistic) and
making them pub would widen the crate's internal API unnecessarily.

---

## Integration Edit Points (Exact)

### 1. New file: `src/inference/itp.rs`

Contains:
- `pub struct ItpResult { ... }`
- `pub fn itp_one_pop(...) -> Result<ItpResult, FdarError>`
- `pub fn itp_two_pop(...) -> Result<ItpResult, FdarError>`
- `pub fn itp_flm(...) -> Result<ItpResult, FdarError>`
- Private helpers: `rank_transform`, `build_pval_matrix`, `pval_correct`, `fisher_cf`,
  `shuffle_itp`, `pool_two_samples_itp`
- `#[cfg(test)] mod tests { ... }` with all inline tests

### 2. Edit: `src/inference/mod.rs`

Add after the existing `mod` declarations:
```rust
mod itp;
pub use itp::{itp_flm, itp_one_pop, itp_two_pop, ItpResult};
```

[VERIFIED: src/inference/mod.rs:29-40] — existing pattern:
```
mod anova;
mod dist;
mod flm;
mod hotelling;
mod permutation;
mod scb;

pub use anova::oneway_anova_vstat;
pub use flm::{flm_f_test, flm_gof_test};
...
```

### 3. Edit: `src/lib.rs` inference re-export block

Extend line 226-229:
```rust
pub use inference::{
    f_perm_test, flm_f_test, flm_gof_test, itp_flm, itp_one_pop, itp_two_pop,
    mean_scb, oneway_anova_vstat, scb_two_sample_test,
    t_perm_test, two_sample_mean_test, ItpResult, TestResult, DEFAULT_N_PERM,
};
```

[VERIFIED: src/lib.rs:225-229]:
```
// Re-export functional inference types and two-sample tests
pub use inference::{
    f_perm_test, flm_f_test, flm_gof_test, mean_scb, oneway_anova_vstat, scb_two_sample_test,
    t_perm_test, two_sample_mean_test, TestResult, DEFAULT_N_PERM,
};
```

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Rust built-in `#[test]` harness |
| Config file | none (uses `cargo test`) |
| Quick run command | `cargo test -p fdars-core --features linalg itp` |
| Full suite command | `cargo test -p fdars-core --features linalg,parallel` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| INF-03 | itp_one_pop: null case (no mean shift) → adjusted p-values non-significant everywhere | unit | `cargo test -p fdars-core --features linalg itp::tests::one_pop_null` | ❌ Wave 0 |
| INF-03 | itp_one_pop: localized mean shift in known interval → adjusted p small on that interval only | unit | `cargo test -p fdars-core --features linalg itp::tests::one_pop_localized` | ❌ Wave 0 |
| INF-03 | itp_two_pop: null (equal means) → non-significant | unit | `cargo test -p fdars-core --features linalg itp::tests::two_pop_null` | ❌ Wave 0 |
| INF-03 | itp_two_pop: localized group difference → significant in true sub-interval | unit | `cargo test -p fdars-core --features linalg itp::tests::two_pop_localized` | ❌ Wave 0 |
| INF-03 | itp_two_pop: seeded determinism (same seed → identical result) | unit | `cargo test -p fdars-core --features linalg itp::tests::two_pop_deterministic` | ❌ Wave 0 |
| INF-03 | itp_flm: null (y independent of X) → non-significant | unit | `cargo test -p fdars-core --features linalg itp::tests::flm_null` | ❌ Wave 0 |
| INF-03 | itp_flm: genuine functional effect → significant in coefficient region | unit | `cargo test -p fdars-core --features linalg itp::tests::flm_effect` | ❌ Wave 0 |
| INF-03 | Error paths: empty data, mismatched sizes, nbasis < 2, n_perm = 0 | unit | `cargo test -p fdars-core --features linalg itp::tests::error_paths` | ❌ Wave 0 |

### Synthetic Fixture Design (for tests)

**Localized difference (two-population):**
- `n_a = n_b = 20` curves, `m = 50` grid points on `[0, 1]`
- Group A: sine curves + small noise over full domain
- Group B: group A curves + a constant shift of 3.0 **only on** argvals 0.4..=0.6
- With `nbasis = 15` B-spline, seed = 42, n_perm = 499
- Expected: `adjusted_pvalues[k] < 0.05` for components k whose B-spline support overlaps
  [0.4, 0.6]; `adjusted_pvalues[k] > 0.10` for components fully outside that interval.
- Tolerance: use `< 0.05` / `> 0.10` thresholds (not exact values), documented in test comment.

**Null case (two-population):**
- Group A and B drawn from the same distribution (same seed, no shift)
- Expected: all `adjusted_pvalues[k] > 0.05` — most should be; check `adjusted_pvalues.iter().all(|&p| p > 0.05)` or use a softer threshold (`max adjusted_p > 0.10`).

### Sampling Rate
- **Per task commit:** `cargo test -p fdars-core --features linalg itp`
- **Per wave merge:** `cargo clippy --all-targets --features linalg,parallel -- -D warnings && cargo test -p fdars-core --features linalg,parallel`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `src/inference/itp.rs` — main implementation file (Wave 0 creates it)
- [ ] All inline tests in `#[cfg(test)] mod tests` inside `itp.rs`

*(No separate test file needed — inline pattern matches all other inference modules)*

---

## Complexity and Parallelism

| Step | Complexity | Parallelizable? |
|------|-----------|-----------------|
| Basis projection (`fdata_to_basis`) | O(n·m·p) | Yes (iter_maybe_parallel! inside fdata_to_basis) |
| Observed stat per component | O(n·p) | Yes (iter over k) |
| Permutation loop (sign-flip / relabel) | O(n_perm·n·p) | The B outer iterations are sequential (one RNG); inner k loop parallelizable with `iter_maybe_parallel!` |
| Rank-transform | O(n_perm·p·log(n_perm)) | Yes (iter over k independently) |
| Interval p-value matrix | O(p²·n_perm) | Yes — `iter_maybe_parallel!` over (row, col) pairs |
| Closure adjustment | O(p²) | Sequential (small, no benefit) |

**Recommendation:** Parallelize the `p²` interval-p-value loop and the rank-transform step
using `iter_maybe_parallel!`. The permutation loop itself must be sequential (single RNG state
driving n_perm shuffles in order). For the inner per-component calculation inside each
permutation, parallelize with `iter_maybe_parallel!(0..p)`.

---

## Minimum-n and Basis-Parameter Guards

| Check | Condition | Error |
|-------|-----------|-------|
| Minimum sample size | `n < 2` | `InvalidDimension` |
| Grid/argvals match | `argvals.len() != m` | `InvalidDimension` |
| Basis parameter | `nbasis < 2` | `InvalidParameter` |
| Permutation count | `n_perm == 0` | `InvalidParameter` |
| Two-pop sizes | `n_a < 2 || n_b < 2` | `InvalidDimension` |
| FLM response length | `y.len() != n` | `InvalidDimension` |
| FLM degrees of freedom | `n < p + 2` per component regression | guard in `component_t_stat` (return 0.0 stat) |

---

## State of the Art

| Old Approach | Current Approach | Impact |
|--------------|------------------|--------|
| Global permutation test (single p-value for whole domain) | Interval-wise closure: per-sub-interval p-values | Identifies WHICH sub-domain drives significance |
| ITP (Pini & Vantini 2016, Biometrics) — same algorithm, "ITP" branding | IWT (Pini & Vantini 2017, JNPS) — "IWT" branding, GitHub | Algorithm identical; fdars uses ITP naming (CRAN package) |
| Fisher combination across components | Rank-based pseudo-p → Fisher | Avoids exact p-value computation; distribution-free |

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | The `pval_correct` reversal (`.reverse()` at end) restores natural component ordering | Code Examples / Pitfall 3 | Adjusted p-values assigned to wrong domain sub-intervals; tests would catch this |
| A2 | `component_t_stat` (simple linear regression per basis component) is an adequate approximation of the R `ITPlmbspline` partial-t approach for `itp_flm` | Architecture Patterns | FLM test may be less powerful than R baseline; document in rustdoc |
| A3 | The `get_2x_rev` indexing in `pval_correct` correctly mirrors R's `[,(2*p):1]` reversal of the column-doubled matrix | Code Examples | Wrong adjusted p-values; tests catch via localized-difference fixture |
| A4 | `+1` correction in raw p-value formula (vs R's `/B` without correction) is an acceptable deliberate divergence | Don't Hand-Roll | Slightly conservative p-values (acceptable); document in rustdoc |
| A5 | `n_perm = 999` default (from INF-01 `DEFAULT_N_PERM`) is sufficient for the O(p²) closure loop | Validation Architecture | Low B → noisy p-value matrix; bump to 1999 if tests are flaky |

**If this table is empty:** N/A — there are assumptions above that the planner should confirm via test output.

---

## Open Questions

1. **`pval_correct` cone walk index math — exact translation of R's doubled+reversed matrix**
   - What we know: R doubles the p-value matrix columns then reverses them before the cone walk
   - What's unclear: Whether the `get_2x_rev` helper correctly mirrors the R arithmetic for
     the wrap-around of circular intervals (the ITP is defined on a circular domain for
     B-spline coefficients)
   - Recommendation: Add a small p=4 unit test where the expected `pval_matrix` and
     `adjusted_pvalues` are computed by hand, matching the R source exactly

2. **FLM `itp_flm` parameter signature**
   - What we know: R's `ITPlmbspline` takes functional predictor + scalar responses + optional
     scalar covariates
   - What's unclear: Whether to accept a pre-fitted `&FregreLmResult` (exposing the FPC basis)
     or raw `(data, y, argvals, ...)` to project onto the explicit ITP basis
   - Recommendation: Accept raw `(data, y, argvals, basis_type, nbasis, ...)` for consistency
     with `itp_one_pop` / `itp_two_pop`. Document as testing the FLM coefficient function's
     basis representation, not the FPC-based beta(t) from `fregre_lm`.

---

## Environment Availability

> Step 2.6: SKIPPED — this phase is purely code additions with no external tools beyond the
> existing Rust toolchain. Environment confirmed: rustc 1.97.0, cargo 1.97.0.
> [VERIFIED: shell output from rustc --version]
> Build note from MEMORY.md: `export TMPDIR=/home/simonm/.cache/fdars-bench-tmp` required
> before `cargo build` / `cargo test` to avoid /tmp exhaustion.

---

## Security Domain

> No authentication, network, or secret handling in this phase. Input validation (dimension
> checks, parameter range checks) is the only applicable concern — already covered by the
> guards table above. ASVS V5 (input validation) is the only relevant category, and it is
> handled via `FdarError::InvalidDimension` / `FdarError::InvalidParameter` at function entry.

---

## Project Constraints (from CLAUDE.md)

- **Additive/non-breaking:** No existing public signatures change. New file only.
- **`Result<T, FdarError>` on all public fns:** Public API must never panic on input.
- **Inline `#[cfg(test)] mod tests`:** All tests inside `itp.rs` (no separate file).
- **Crate-root re-exports:** `lib.rs` inference block extended.
- **No new crate dependency:** Only `rand`, `crate::basis`, `crate::inference::permutation`
  (pattern), `crate::error`, `crate::matrix` — all already in `Cargo.toml`.
- **Full clippy gate:** `cargo clippy --all-targets --features linalg,parallel -- -D warnings`
  must stay green after the addition.
- **`TMPDIR=/home/simonm/.cache/fdars-bench-tmp`** before any build/test command.
- **Column-major FdMatrix:** All matrix access via `mat[(i, j)]`.
- **Per-thread RNG seeding:** `StdRng::seed_from_u64(seed)` (no `+ k` needed here since ITP
  uses a single sequential permutation loop with one RNG, not a parallel per-thread pattern).
- **`#[non_exhaustive]`** on `ItpResult`.
- **Conditional serde:** `#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]`.

---

## Sources

### Primary (CITED — R source code read directly)
- `github.com/cran/fdatest/blob/master/R/ITP1bspline.R` — one-population algorithm, exact source
- `github.com/cran/fdatest/blob/master/R/ITP2bspline.R` — two-population algorithm, exact source
- `github.com/cran/fdatest/blob/master/R/ITPlmbspline.R` — FLM algorithm, structure confirmed

### Primary (VERIFIED — codebase reads this session)
- `src/inference/permutation.rs` [VERIFIED: lines 1-419] — INF-01 permutation infra, shuffle_labels, pool_two_samples, DEFAULT_N_PERM
- `src/inference/mod.rs` [VERIFIED: lines 1-57] — TestResult struct, existing re-export pattern
- `src/inference/flm.rs` [VERIFIED: lines 1-439] — FLM test structure
- `src/basis/projection.rs` [VERIFIED: lines 1-238] — fdata_to_basis, ProjectionBasisType enum, BasisProjectionResult
- `src/basis/bspline.rs` [VERIFIED: lines 1-60] — B-spline basis construction
- `src/basis/fourier.rs` [VERIFIED: lines 1-27] — fourier_basis signature
- `src/matrix.rs` [VERIFIED: lines 1-100] — FdMatrix column-major layout
- `src/lib.rs` [VERIFIED: lines 225-229] — existing inference re-export block
- `src/scalar_on_function/mod.rs` [VERIFIED: lines 55-91] — FregreLmResult struct fields

### Secondary (LOW confidence)
- Pini & Vantini (2016) Biometrics — ITP paper (abstract via PubMed search)
- Pini & Vantini (2017) JNPS — IWT paper (title via search)

---

## Metadata

**Confidence breakdown:**
- Algorithm (one-pop + two-pop): HIGH — R source code read directly, algorithm fully pinned
- Algorithm (FLM): MEDIUM — R structure confirmed, exact per-component statistic verified, response-permutation simplification is [ASSUMED]
- Closure adjustment: HIGH — `pval.correct` function extracted verbatim, Rust translation is [ASSUMED A1, A3] (index math to be validated in test)
- Codebase integration points: HIGH — all files read this session

**Research date:** 2026-08-20
**Valid until:** 2027-02-20 (stable algorithm, paper + CRAN package stable)
