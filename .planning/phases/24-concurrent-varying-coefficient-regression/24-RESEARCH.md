# Phase 24: Concurrent / Varying-Coefficient Regression - Research

**Researched:** 2026-08-17
**Domain:** Functional concurrent regression, pointwise OLS, kernel smoothing (Rust/fdars-core)
**Confidence:** HIGH (codebase reads) / MEDIUM (R-ecosystem conventions)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **DENSE variant only**; sparse/PACE path is DEFERRED (no FPCA-01 dependency).
- β(t) estimated by **pointwise OLS per grid column, then smoothing the β(t) sequence** via `smoothing.rs` kernels (two-step pointwise-then-smooth).
- Roughness penalty = **kernel bandwidth** fed to the `smoothing.rs` smoother (larger bandwidth → smoother β(t)).
- Include a **time-varying intercept β₀(t)**; default kernel **"gaussian"**.
- Public fn name **`concurrent_regression`**; predictors passed as **`predictors: &[FdMatrix]`**; flat params `(response, predictors, argvals: Option<&[f64]>, bandwidth, kernel)`; uniform 0..1 grid if argvals None.
- Result: **`FdMatrix beta_curve`** (rows = predictor coeffs, cols = grid points) + separate **`intercept: Vec<f64>`** + **`fitted`** (FdMatrix n×m) + **`residuals`** (FdMatrix n×m) + **`argvals`**. No R² diagnostics.

### Claude's Discretion

- (none stated beyond the locked decisions)

### Deferred Ideas (OUT OF SCOPE)

- Sparse/PACE kernel-weighted concurrent-regression variant (needs FPCA-01 PACE infra).
- R²(t) / overall R² and other GLM-style diagnostics on the result struct.
- λ difference-penalty (basis-penalty / refund-style) estimation alternative.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| REG-01 | Dense functional concurrent (varying-coefficient) regression via new public `concurrent_regression` entry point in `fdars-core/src/concurrent_regression.rs`. β(t) estimated by pointwise/local-linear LS over shared dense grid with roughness (smoothing) penalty, reusing `smoothing.rs` kernels. Result carries `{ beta_curve, fitted, residuals }`, is `Result`-returning, re-exported at crate root. Existing regression APIs untouched. | Full coverage: estimation algorithm, struct layout, reuse targets, validation tests, and integration steps all documented below. |
</phase_requirements>

---

## Summary

Concurrent (varying-coefficient) regression models the relationship y_i(t) = β₀(t) + β₁(t)x_{i1}(t) + … + βₚ(t)x_{ip}(t) + ε_i(t) where all curves share the same dense grid of m evaluation points. The fdaconcur R package (authoritative reference) implements this in two clean steps: (1) at each grid column j, fit an ordinary (unweighted) OLS with a (p+1)-column design matrix [1 | x_1[·,j] | … | x_p[·,j]] and extract the intercept β₀[j] and slopes β₁[j]…βₚ[j]; (2) smooth each resulting discrete sequence over j using local-linear kernel regression.

In fdars-core this maps exactly onto the existing `smoothing.rs::local_linear` function (or `nadaraya_watson` as fallback). The pointwise OLS systems are (p+1)×(p+1) — small enough for Gaussian elimination, matching the `solve_gaussian_pub` already in `smoothing.rs`. Step 2 calls `local_linear(argvals, raw_beta_k, argvals, bandwidth, kernel)` independently for each of the p+1 coefficient sequences.

The result struct mirrors `FofResult` from `fof_regression.rs`: a `beta_curve: FdMatrix` (p×m, rows = predictor indices, cols = grid points), a separate `intercept: Vec<f64>` (length m, the smoothed β₀(t)), `fitted` and `residuals` (both FdMatrix n×m), and `argvals: Vec<f64>`. No external crate dependency is added.

**Primary recommendation:** Implement as a single new file `src/concurrent_regression.rs` with one public function `concurrent_regression` and one public result struct `ConcurrentRegrResult`. Re-export both from `src/lib.rs` alongside `fof_regression`. The inner loop over grid columns can use `iter_maybe_parallel!` for the parallelism gate.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Pointwise OLS at each grid column | `concurrent_regression.rs` (new) | `smoothing.rs::solve_gaussian_pub` (reused) | Each column is an independent small linear system; no FPCA, no SVD needed |
| Coefficient smoothing (β₀, β₁…βₚ over t) | `smoothing.rs::local_linear` (reused directly) | — | The locked decision explicitly mandates reusing `smoothing.rs` kernels |
| Input validation & error returning | `concurrent_regression.rs` entry point | `FdarError` variants | All public fns return `Result<T, FdarError>`; dimension checks at entry |
| Fitted/residuals computation | `concurrent_regression.rs` | `FdMatrix` arithmetic | Direct pointwise: fitted[i,j] = β₀[j] + Σₖ βₖ[j]·xₖ[i,j] |
| Feature-gated parallelism | `iter_maybe_parallel!` macro | `parallel.rs` | Consistent with all other modules; the grid-column loop is embarrassingly parallel |
| Crate-root re-export | `src/lib.rs` | — | Additive `pub mod` + `pub use` following `fof_regression` precedent |

---

## Standard Stack

### Core (no new dependencies)

| Library | Version in Cargo.toml | Purpose | Why Standard |
|---------|----------------------|---------|--------------|
| fdars-core internal | — | `FdMatrix`, `FdarError`, `smoothing.rs`, `parallel.rs` | All reused; zero new crate deps per milestone mandate |

No new external crate dependencies are introduced. This phase is reuse-only.

**Installation:** None required. `cargo build -p fdars-core` picks up the new module automatically once registered in `lib.rs`.

---

## Package Legitimacy Audit

> No external packages are added in this phase (pure internal reuse). This section is intentionally empty.

**Packages removed due to SLOP verdict:** none
**Packages flagged as suspicious SUS:** none

---

## Architecture Patterns

### System Architecture Diagram

```
caller
  │
  ▼
concurrent_regression(response, predictors, argvals?, bandwidth, kernel)
  │
  ├─ [validate] shape checks, n≥2, m≥1, all predictors same (n,m), bandwidth>0
  │
  ├─ [Step 1: pointwise OLS] for each grid column j ∈ 0..m  ← iter_maybe_parallel!
  │     Build X_j : n×(1+p)  col 0 = ones, col k+1 = predictor_k[:,j]
  │     Solve X_j' X_j β_j = X_j' y_j  via solve_gaussian_pub (size (1+p)²)
  │     raw_intercept[j] = β_j[0]
  │     raw_beta[k,j]    = β_j[k+1]  for k in 0..p
  │
  ├─ [Step 2: smooth] for each coefficient sequence  (p+1 calls to local_linear)
  │     intercept[j]   = local_linear(argvals, raw_intercept, argvals, bw, kernel)
  │     beta_k[j]      = local_linear(argvals, raw_beta[k,:], argvals, bw, kernel)
  │
  ├─ [Step 3: fitted & residuals]
  │     fitted[i,j] = intercept[j] + Σₖ beta_k[j] * predictor_k[i,j]
  │     residuals[i,j] = response[i,j] - fitted[i,j]
  │
  └─ return Ok(ConcurrentRegrResult { beta_curve, intercept, fitted, residuals, argvals })

ConcurrentRegrResult
  ├── beta_curve : FdMatrix (p × m)    — smoothed predictor coefficients
  ├── intercept  : Vec<f64> (length m) — smoothed time-varying intercept
  ├── fitted     : FdMatrix (n × m)    — reconstructed response
  ├── residuals  : FdMatrix (n × m)    — response − fitted
  └── argvals    : Vec<f64> (length m) — grid used
```

### Recommended Project Structure

```
fdars-core/src/
├── concurrent_regression.rs   # NEW — this phase's deliverable
├── smoothing.rs               # REUSED — local_linear, solve_gaussian_pub
├── matrix.rs                  # REUSED — FdMatrix API
├── error.rs                   # REUSED — FdarError
├── lib.rs                     # MODIFIED — add pub mod + pub use
└── ...
```

### Pattern 1: Pointwise OLS per Grid Column

**What:** For each grid column j (0..m), extract a response vector y_j (length n) and a design matrix X_j (n × (p+1)), then solve the normal equations (X_j'X_j) coef = X_j'y_j.

**When to use:** Dense shared-grid concurrent regression where all curves are measured at the same m points.

**Why this design:** The (p+1)×(p+1) normal equations are tiny — p is typically 1–5. Gaussian elimination (`solve_gaussian_pub` from `smoothing.rs`) is already present and sufficient; no nalgebra SVD needed at this step. This keeps the inner loop allocation-light.

**Example (pseudocode matching codebase style):**

```rust
// Source: derived from smoothing.rs::local_polynomial accumulate_weighted_normal_equations pattern
// and fdaconcur ptFCReg pointwise OLS convention [CITED: rdrr.io/cran/fdaconcur/man/ptFCReg.html]
let p = predictors.len();
let q = p + 1; // intercept + p slopes

// For grid column j:
let mut xtx = vec![0.0_f64; q * q];
let mut xty = vec![0.0_f64; q];
for i in 0..n {
    // design row: [1, pred_0[i,j], pred_1[i,j], ...]
    let mut row = vec![1.0_f64; q];
    for (k, pred) in predictors.iter().enumerate() {
        row[k + 1] = pred[(i, j)];
    }
    for a in 0..q {
        for b in 0..q {
            xtx[a * q + b] += row[a] * row[b]; // row-major XtX
        }
        xty[a] += row[a] * response[(i, j)];
    }
}
// Optional ridge stabilizer for near-singular columns
let eps = 1e-10 * (xtx[0] + 1.0); // xtx[0] = n (intercept*intercept sum)
for d in 0..q { xtx[d * q + d] += eps; }

let coef = smoothing::solve_gaussian_pub(&mut xtx, &mut xty, q);
// coef[0] = raw_intercept[j], coef[1..q] = raw_beta[k,j]
```

**Note on `solve_gaussian_pub`:** The function is already `pub` in `smoothing.rs` [VERIFIED: fdars-core/src/smoothing.rs:336-338]:

```rust
pub fn solve_gaussian_pub(a: &mut [f64], b: &mut [f64], p: usize) -> Vec<f64> {
    solve_gaussian(a, b, p)
}
```

The XtX matrix stored here is row-major (the convention used by `smoothing.rs::accumulate_weighted_normal_equations`), which is what `solve_gaussian_pub` expects.

### Pattern 2: Coefficient Smoothing via `local_linear`

**What:** After collecting raw_intercept (length m) and raw_beta[k] (length m for each k), smooth each sequence with `local_linear(argvals, sequence, argvals, bandwidth, kernel)`.

**When to use:** Always — the bandwidth is the only roughness knob (locked decision).

**Example:**

```rust
// Source: smoothing.rs::local_linear — same argvals as both x and x_new
// [VERIFIED: fdars-core/src/smoothing.rs:160-230]
use crate::smoothing::local_linear;

let smooth_intercept = local_linear(&argvals, &raw_intercept, &argvals, bandwidth, kernel)?;
for k in 0..p {
    let raw_k: Vec<f64> = (0..m).map(|j| raw_beta[k * m + j]).collect();
    let smooth_k = local_linear(&argvals, &raw_k, &argvals, bandwidth, kernel)?;
    for j in 0..m {
        beta_curve[(k, j)] = smooth_k[j];
    }
}
```

### Pattern 3: Fitted Curves and Residuals

**What:** Reconstruct fitted[i,j] = intercept[j] + Σₖ beta_curve[k,j] * predictors[k][i,j]; residuals[i,j] = response[i,j] − fitted[i,j].

```rust
let mut fitted = FdMatrix::zeros(n, m);
let mut residuals = FdMatrix::zeros(n, m);
for j in 0..m {
    for i in 0..n {
        let mut val = intercept[j];
        for (k, pred) in predictors.iter().enumerate() {
            val += beta_curve[(k, j)] * pred[(i, j)];
        }
        fitted[(i, j)] = val;
        residuals[(i, j)] = response[(i, j)] - val;
    }
}
```

### Pattern 4: Parallel Gate for Column Loop

**What:** Wrap the outer grid-column loop with `iter_maybe_parallel!` so results are collected into a `Vec<(Vec<f64>, Vec<f64>)>` (raw_intercept_j, raw_beta_j).

**Constraint:** The parallelism gate is mandatory for consistency with all other modules. Because `solve_gaussian_pub` takes `&mut` slices, each column's XtX/Xty must be allocated inside the closure (not shared) — this is already the pattern in `local_polynomial`.

```rust
// Consistent with smoothing.rs::local_polynomial pattern [VERIFIED: fdars-core/src/smoothing.rs:404-411]
let raw_cols: Vec<_> = iter_maybe_parallel!((0..m))
    .map(|j| {
        // ... allocate local xtx, xty, solve, return (intercept_j, beta_slice_j)
        compute_column_ols(j, n, p, response, predictors)
    })
    .collect();
```

### Anti-Patterns to Avoid

- **Materializing row vectors for each observation inside the inner loop:** use `pred[(i, j)]` direct index access (O(1) in column-major), not `pred.row(i)` (O(m) allocation) [VERIFIED: fdars-core/src/matrix.rs:146-150]
- **Using `nalgebra::SVD` for the tiny (p+1)×(p+1) system:** overkill and requires `to_dmatrix()` conversion; `solve_gaussian_pub` is already present and sufficient
- **Sharing mutable XtX/Xty buffers across threads:** each grid column must have its own local allocation inside the `iter_maybe_parallel!` closure
- **Forgetting `#[must_use]` on the public function:** the project convention requires it on all expensive computations [VERIFIED: fdars-core/src/fof_regression.rs:112]
- **Using `nalgebra` for the smoothing step:** `local_linear` already handles this correctly; do not bypass it

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Gaussian elimination for (p+1)×(p+1) system | Custom solver | `smoothing::solve_gaussian_pub` | Already present, tested, handles singularity gracefully |
| Kernel-weighted local-linear smooth of β(t) | Custom smoother | `smoothing::local_linear` | Already handles bandwidth validation, edge behavior, parallel gate |
| Kernel selection ("gaussian"/"epanechnikov"/"tricube") | Enum or string match | `smoothing::get_kernel` (internal) / pass `kernel: &str` directly to `local_linear` | `local_linear` already dispatches via `get_kernel` internally |
| Uniform 0..1 grid | Manual `(0..m).map(...)` | Follow `scalar_on_function::fregre_lm` precedent: `(0..m).map(\|j\| j as f64 / (m-1).max(1) as f64)` | Project-wide convention for `argvals: None` default |

**Key insight:** Both computational building blocks (Gaussian elimination for the small normal equations and local-linear smoothing) already exist and are tested. The only new code is the two-step orchestration, the result struct, and the validation guards.

---

## Runtime State Inventory

> SKIPPED — this is a greenfield addition (new file, no rename/refactor). No existing state to inventory.

---

## Common Pitfalls

### Pitfall 1: Near-Singular Column at Grid Point j

**What goes wrong:** If all observations have nearly identical predictor values at grid column j (e.g., a predictor that is nearly flat), the XtX matrix is near-singular. Gaussian elimination returns zeros, producing a spurious β=0 at that point.

**Why it happens:** The pointwise design matrix X_j has rank < p+1 when predictors are collinear at that time point (common in slowly varying functional data).

**How to avoid:** Add a small diagonal ridge `eps * I` to XtX before solving. Use `eps = 1e-10 * (xtx[0] + 1.0)` (xtx[0] = n, the intercept-squared sum). This is exactly the pattern used in `fof_regression.rs` [VERIFIED: fdars-core/src/fof_regression.rs:183-187]:
```rust
let ridge = 1e-8 * (0..ncomp_x).map(|k| xtx[k * ncomp_x + k]).sum::<f64>() / ncomp_x as f64;
for k in 0..ncomp_x { xtx[k * ncomp_x + k] += ridge.max(1e-12); }
```
Adapt the same logic for the (p+1)×(p+1) case.

**Warning signs:** `beta_curve` has isolated zero columns; recovered β(t) has sharp unexplained dips.

---

### Pitfall 2: `solve_gaussian_pub` Row-Major vs Column-Major XtX

**What goes wrong:** If XtX is built in the wrong memory order, solve_gaussian_pub returns garbage coefficients without erroring.

**Why it happens:** `solve_gaussian_pub` expects a row-major flat representation (element `[a*q + b]` = row a, col b). This is the convention used by `accumulate_weighted_normal_equations` in `smoothing.rs` [VERIFIED: fdars-core/src/smoothing.rs:246-258]:
```rust
for j in 0..p {
    ...
    for k in 0..p {
        xtx[j * p + k] += w_dj * d.powi(k as i32);
    }
```

**How to avoid:** Build XtX with `xtx[a * q + b] += row[a] * row[b]`. Confirm: `xtx[0]` = sum of squares of intercept column = n (all-ones column). Verify in a unit test with known β.

**Warning signs:** Recovery test fails to recover constant β(t) even on noise-free data.

---

### Pitfall 3: Edge Behavior of `local_linear` Smoother

**What goes wrong:** The raw_intercept / raw_beta sequences are length m, smoothed on the same m argvals grid. At boundary points (j=0, j=m-1), the local-linear kernel has less data on one side, producing edge effects (bias toward the interior trend).

**Why it happens:** Standard behavior of local polynomial estimators at domain boundaries; not a bug.

**How to avoid:** Document in rustdoc that the smoother is not boundary-corrected. In tests, check interior points only (e.g., indices 5..m-5) for tight tolerance, and boundary points more loosely.

**Warning signs:** Recovery test fails at j=0 or j=m-1 but passes in interior.

---

### Pitfall 4: Parallelism + Mutable Buffer

**What goes wrong:** Attempting to write to a shared `raw_beta` buffer from `iter_maybe_parallel!` causes a borrow-check error (or, if using unsafe, a data race).

**Why it happens:** rayon parallel iterators do not allow shared mutable access.

**How to avoid:** Return per-column results from the closure as a `Vec<(f64, Vec<f64>)>` (intercept_j, beta_j), then serialize into the output buffers after `.collect()`. Exactly mirrors the pattern in `smoothing.rs::local_polynomial` where each x0 closure returns a single f64.

---

### Pitfall 5: FdMatrix Column-Major Index in Inner Loop

**What goes wrong:** Using `pred.row(i)[j]` instead of `pred[(i, j)]` inside the tight inner loop allocates a new Vec per row per column per observation — O(n*m) allocations.

**Why it happens:** `FdMatrix::row` materializes a Vec [VERIFIED: fdars-core/src/matrix.rs:146-150]. `FdMatrix::index (i,j)` is O(1) [VERIFIED: fdars-core/src/matrix.rs:122-130 via IndexMut derivation].

**How to avoid:** Always use `mat[(i, j)]` inside the OLS accumulation loop. See `fof_regression.rs` for the canonical pattern:
```rust
// [VERIFIED: fdars-core/src/fof_regression.rs:195-199]
for i in 0..n {
    s += x_scores[(i, k)] * y_scores[(i, l_col)];
}
```

---

## Code Examples

### Result Struct (following FofResult convention)

```rust
// Pattern: fdars-core/src/fof_regression.rs:28-53 [VERIFIED]
/// Result of concurrent (varying-coefficient) functional regression.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ConcurrentRegrResult {
    /// Time-varying predictor coefficients: rows = predictor index, cols = grid points (p × m).
    pub beta_curve: FdMatrix,
    /// Time-varying intercept β₀(t) (length m).
    pub intercept: Vec<f64>,
    /// Fitted response curves (n × m).
    pub fitted: FdMatrix,
    /// Residuals: response − fitted (n × m).
    pub residuals: FdMatrix,
    /// Argument values (shared grid, length m).
    pub argvals: Vec<f64>,
}
```

### Function Signature

```rust
// Pattern: fof_regression, flat-params convention per CONTEXT.md
/// Concurrent (varying-coefficient) functional regression.
///
/// Fits the model y_i(t) = β₀(t) + Σₖ βₖ(t)·xₖᵢ(t) + εᵢ(t) for
/// a dense shared grid of m evaluation points and p ≥ 1 functional predictors.
///
/// # Estimation
/// Two-step: (1) pointwise OLS at each grid column; (2) smooth the resulting
/// discrete β sequences via local-linear kernel regression (`smoothing::local_linear`).
/// The `bandwidth` parameter controls roughness — larger values produce smoother β(t).
/// This matches the fdaconcur `ptFCReg` → `smPtFCRegCoef` convention
/// (local-linear smoothing of pointwise OLS estimates).
///
/// # Arguments
/// * `response` - Functional response (n × m)
/// * `predictors` - Slice of functional predictor matrices, each (n × m)
/// * `argvals` - Shared grid (length m); if `None`, uniform 0..1 grid is used
/// * `bandwidth` - Kernel bandwidth for β(t) smoothing (must be positive)
/// * `kernel` - Kernel type: "gaussian" (default), "epanechnikov", "tricube"
#[must_use = "expensive computation whose result should not be discarded"]
pub fn concurrent_regression(
    response: &FdMatrix,
    predictors: &[FdMatrix],
    argvals: Option<&[f64]>,
    bandwidth: f64,
    kernel: &str,
) -> Result<ConcurrentRegrResult, FdarError>
```

### lib.rs Registration (additive)

```rust
// Insert alongside fof_regression in src/lib.rs — two lines each
// [VERIFIED pattern: fdars-core/src/lib.rs:87, 233]
pub mod concurrent_regression;
pub use concurrent_regression::{concurrent_regression, ConcurrentRegrResult};
```

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Rust built-in test harness (`#[test]`, `#[cfg(test)]`) |
| Config file | None — uses Cargo |
| Quick run command | `cargo test -p fdars-core concurrent_regression` |
| Full suite command | `cargo test -p fdars-core --features linalg,parallel && cargo clippy --all-targets --features linalg,parallel -- -D warnings` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | Notes |
|--------|----------|-----------|-------------------|-------|
| REG-01 SC1 | `concurrent_regression` is a public entry point in `concurrent_regression.rs`, re-exported | compile+smoke | `cargo build -p fdars-core` | Verified by successful build |
| REG-01 SC2 | Recovered `beta_curve` reproduces known β(t) within tolerance | unit | `cargo test -p fdars-core concurrent_regression::tests::test_recovery_known_beta` | Interior points, tolerance 0.15 with n=50, m=50, low noise |
| REG-01 SC3 | Larger bandwidth → demonstrably smoother `beta_curve` | unit | `cargo test -p fdars-core concurrent_regression::tests::test_monotone_roughness` | Roughness = sum of squared second differences |
| REG-01 SC4a | `residuals == response − fitted` pointwise | unit | `cargo test -p fdars-core concurrent_regression::tests::test_residuals_consistency` | Tolerance 1e-10 |
| REG-01 SC4b | Invalid inputs → `FdarError`, no panic | unit | `cargo test -p fdars-core concurrent_regression::tests::test_invalid_inputs` | Mismatched n, empty predictors, zero bandwidth |
| REG-01 SC5 | Existing regression APIs unchanged | regression | `cargo test -p fdars-core --features linalg,parallel` | Full suite green |

### Wave 0 Gaps

- [ ] `fdars-core/src/concurrent_regression.rs` — new file (the entire phase deliverable)
- [ ] Inline `#[cfg(test)] mod tests` within `concurrent_regression.rs` — covers all 5 SC test cases
- [ ] `lib.rs` modification: `pub mod concurrent_regression;` and `pub use concurrent_regression::...;`

### Sampling Rate

- **Per task commit:** `cargo test -p fdars-core concurrent_regression`
- **Per wave merge:** `cargo test -p fdars-core --features linalg,parallel`
- **Phase gate:** Full suite + clippy green before `/gsd-verify-work`

---

## Test Design Details

### Recovery Test (SC2): Known β(t)

```
n = 50 observations, m = 50 grid points, p = 1 predictor
true_beta(t) = sin(2πt)            — smooth, known
true_beta0(t) = 0.5               — constant intercept
x[i,j] = sin(2π(i/n) + argvals[j]) — predictor curves
y[i,j] = true_beta0 + true_beta(argvals[j]) * x[i,j] + noise
noise scale: 0.05 * x.max()

Expected: |recovered beta_curve[0,j] - sin(2π*argvals[j])| < 0.15 for j in 5..45
bandwidth: 0.15  kernel: "gaussian"
```

**Design note:** Use LCG-deterministic noise (same pattern as `smoothing.rs` tests) or a fixed seed. Do NOT use `rand` — no randomness in test code is the project convention. Use a small LCG multiplier inline.

### Monotone Roughness Test (SC3)

```
roughness(beta_k) = Σ_{j=1}^{m-2} (beta[j+1] - 2*beta[j] + beta[j-1])²

Run concurrent_regression with bandwidths [0.05, 0.15, 0.35] on synthetic data.
Assert: roughness(bw=0.05) > roughness(bw=0.15) > roughness(bw=0.35)
```

This is a discrete approximation to the integrated squared second derivative — standard roughness metric for smooth curves [CITED: math literature on roughness penalties].

### Residuals Consistency Test (SC4a)

```
For all (i, j): assert (residuals[i,j] - (response[i,j] - fitted[i,j])).abs() < 1e-10
```

### Input Guard Tests (SC4b)

| Scenario | Expected Error |
|----------|---------------|
| `predictors` is empty slice | `FdarError::InvalidDimension { parameter: "predictors", ... }` |
| predictor[0].nrows() != response.nrows() | `FdarError::InvalidDimension { parameter: "predictors[0]", ... }` |
| predictor[0].ncols() != response.ncols() | `FdarError::InvalidDimension { parameter: "predictors[0]", ... }` |
| `bandwidth <= 0.0` | `FdarError::InvalidParameter { parameter: "bandwidth", ... }` |
| `response.nrows() < 2` | `FdarError::InvalidDimension { parameter: "response", ... }` |
| `argvals.len() != m` (if Some) | `FdarError::InvalidDimension { parameter: "argvals", ... }` |

---

## Input Validation Sequence

This must happen at the top of `concurrent_regression` before any computation:

```
1. predictors.is_empty() → InvalidDimension("predictors", "at least 1", "0")
2. let (n, m) = response.shape();
   n < 2 → InvalidDimension("response", "at least 2 rows", n)
   m == 0 → InvalidDimension("response", "non-zero columns", 0)
3. For each (k, pred) in predictors.iter().enumerate():
   pred.nrows() != n → InvalidDimension("predictors[k]", "n rows", pred.nrows())
   pred.ncols() != m → InvalidDimension("predictors[k]", "m cols", pred.ncols())
4. bandwidth <= 0.0 → InvalidParameter("bandwidth", "must be positive, got {bandwidth}")
5. if let Some(av) = argvals { av.len() != m → InvalidDimension("argvals", "m elements", av.len()) }
6. Compute owned argvals: argvals.map(|v| v.to_vec()).unwrap_or_else(|| uniform_0_1(m))
```

`uniform_0_1(m)` = `(0..m).map(|j| j as f64 / (m-1).max(1) as f64).collect()` — the convention used by `fregre_lm` [VERIFIED: fdars-core/src/scalar_on_function/fregre_lm.rs:74].

---

## R Convention Pinned

The fdaconcur `ptFCReg` → `smPtFCRegCoef` two-step convention is the match target [CITED: rdrr.io/cran/fdaconcur/man/ptFCReg.html, rdrr.io/cran/fdaconcur/man/smPtFCRegCoef.html]:

- `ptFCReg`: pointwise OLS at each tGrid column — returns `beta0` (intercept vector) and `beta` (p × m matrix)
- `smPtFCRegCoef`: smooths using `Lwls1D` (local linear 1D smoother) with bandwidth `bw`; modifies `beta0` and `beta` in-place

fdars matches this via `local_linear` from `smoothing.rs`. The fdars implementation does NOT match `refund::pffr` (which uses basis-penalty / penalized splines) or `fdapace::FCReg` (which uses 2D smoothing). Those are deferred alternatives.

---

## State of the Art

| Old Approach | Current Approach | Status in fdars |
|--------------|------------------|-----------------|
| Basis-penalty (penalized splines, refund/pffr) | Pointwise OLS + kernel smooth (fdaconcur) | fdars implements the fdaconcur two-step |
| 2D smoothing of (s,t) surface (fdapace::FCReg) | 1D smoothing per coefficient | 2D is DEFERRED |
| Sparse/PACE kernel-weighted variant | Dense-only (shared grid assumed) | Sparse path DEFERRED |

---

## Security Domain

`security_enforcement: true` in config.json. For this phase:

| ASVS Category | Applies | Control |
|---------------|---------|---------|
| V5 Input Validation | yes | All inputs validated at entry — no silent truncation |
| V2 Authentication | no | Pure numeric library, no auth surface |
| V3 Session Management | no | Stateless computation |
| V4 Access Control | no | No access control surface |
| V6 Cryptography | no | No cryptographic operations |

**Threat patterns relevant to numeric libraries:**

| Pattern | STRIDE | Mitigation |
|---------|--------|------------|
| Malformed input dimensions (n=0, m=0) | Denial of service (panic) | Validated at entry; `n < 2`, `m == 0`, empty predictors all → `FdarError` |
| NaN/Inf propagation from inputs | Tampering | Not guarded (consistent with existing modules — NaN propagates, no silent NaN injection) |
| Integer overflow in index arithmetic | Tampering | n*m for typical FDA data (n≤10k, m≤1000) well within usize; no explicit overflow check needed |

---

## Environment Availability

> SKIPPED — purely internal code change; no external tools, services, or CLIs required beyond the existing Rust toolchain already confirmed present.

---

## Open Questions

1. **`n = 1` edge case:** With a single observation, OLS is underdetermined at each column (the design is rank-1 with p+1 > 1 columns). Options: (a) error if `n < p+2`, (b) require `n ≥ 2` as a minimum. Given project convention (`fof_regression` requires `n ≥ 3`), **recommendation:** require `n ≥ 2` and rely on the ridge stabilizer for the degenerate case. Document that for `n < p+2` results may be unreliable.

   - What we know: `n ≥ 2` is used as the minimum in most smoothing functions.
   - What's unclear: Should the guard be `n ≥ p+2` (full rank guarantee) or just `n ≥ 2`?
   - Recommendation: Use `n ≥ 2` for the error guard, add a doc note that sufficient sample size for stable estimates is `n > p+1`.

2. **Parallel gate placement:** The outer column loop (j=0..m) is the natural parallel dimension. The inner OLS (p+1)×(p+1) solve is sequential. The smoothing step (p+1 calls to `local_linear`) can optionally be parallelized over coefficient index. Given `local_linear` already uses `slice_maybe_parallel!` internally, running p+1 calls sequentially is safe; running them in parallel would require a second `iter_maybe_parallel!` — probably not worth it for small p.
   - Recommendation: Parallelize the column loop, keep smoothing calls sequential.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `solve_gaussian_pub` expects XtX in row-major order (index `a*q+b` = row a, col b) | Pointwise OLS pattern | Regression coefficients would be wrong; recovery test would fail immediately |
| A2 | Roughness = sum of squared second differences is sufficient to demonstrate monotone smoothing | Monotone roughness test | Test might not detect pathological cases where kernel overshoot increases local roughness |
| A3 | `n ≥ 2` is the right minimum guard (not `n ≥ p+2`) | Input validation | Could allow ill-conditioned fits for small n; ridge stabilizer mitigates in practice |

**A1 is verifiable by reading `solve_gaussian_pub` source** — confirmed [VERIFIED: fdars-core/src/smoothing.rs:326-338] that it calls `solve_gaussian(a, b, p)` which uses `a[j * p + col]` indexing (row j, col `col`), i.e., row-major.

---

## Project Constraints (from CLAUDE.md)

All of the following directives from `.claude/CLAUDE.md` must be respected in the plan and execution:

| Directive | Impact on This Phase |
|-----------|---------------------|
| Column-major `FdMatrix` storage | Inner loop accesses `mat[(i, j)]` = `data[i + j*n]`; never `mat.row(i)[j]` in hot paths |
| All public fns return `Result<T, FdarError>` | `concurrent_regression` returns `Result<ConcurrentRegrResult, FdarError>` |
| Feature-gated rayon parallelism via `iter_maybe_parallel!` macros | Column loop uses `iter_maybe_parallel!((0..m))` |
| MSRV 1.81.0 (1.84 for `linalg` feature) | No new `linalg`-feature-gated dependencies introduced |
| `#[derive(Debug, Clone, PartialEq)]` on all public types | `ConcurrentRegrResult` must derive all three |
| `#[non_exhaustive]` on public result structs | `ConcurrentRegrResult` must be `#[non_exhaustive]` |
| `#[cfg_attr(feature = "serde", derive(...))]` for serde | Add conditional serde derive to `ConcurrentRegrResult` |
| `#[must_use]` on expensive computations | `concurrent_regression` function must be annotated |
| Inline `#[cfg(test)] mod tests` per module | All tests inside `concurrent_regression.rs` |
| `uniform_grid(n)` from `test_helpers.rs` in tests | Use `use crate::test_helpers::uniform_grid;` in test module |
| Full clippy gate: `--all-targets --features linalg,parallel -- -D warnings` | Plan must include this check step |
| `TMPDIR=/home/simonm/.cache/fdars-bench-tmp` may be needed | Pre-commit doctest linking; set if /tmp is full |
| Additive/non-breaking — zero changes to existing public signatures | No existing function or struct may be modified |
| Re-export at crate root via `pub mod` + `pub use` in `src/lib.rs` | Two lines added to lib.rs |
| No new external crate dependencies this milestone | Confirmed: only internal reuse |

---

## Sources

### Primary (HIGH confidence — codebase reads this session)

- `fdars-core/src/smoothing.rs:160-230` — `local_linear` function signature and implementation [VERIFIED]
- `fdars-core/src/smoothing.rs:326-338` — `solve_gaussian_pub` public wrapper, row-major convention [VERIFIED]
- `fdars-core/src/smoothing.rs:404-411` — `local_polynomial` parallel gate pattern [VERIFIED]
- `fdars-core/src/fof_regression.rs:28-53` — `FofResult` struct (structural analog for `ConcurrentRegrResult`) [VERIFIED]
- `fdars-core/src/fof_regression.rs:112` — `#[must_use]` annotation pattern [VERIFIED]
- `fdars-core/src/fof_regression.rs:183-187` — ridge stabilizer for XtX [VERIFIED]
- `fdars-core/src/matrix.rs:122-150` — `FdMatrix::column`, `FdMatrix::row`, column-major index [VERIFIED]
- `fdars-core/src/error.rs:1-25` — `FdarError` enum variants (verbatim) [VERIFIED]
- `fdars-core/src/test_helpers.rs:6-8` — `uniform_grid(n)` [VERIFIED]
- `fdars-core/src/lib.rs:87,233` — `fof_regression` pub mod + pub use pattern [VERIFIED]
- `fdars-core/src/scalar_on_function/fregre_lm.rs:74` — `argvals` uniform-0-1 default pattern [VERIFIED]

### Secondary (MEDIUM confidence — official R package documentation)

- [rdrr.io/cran/fdaconcur/man/ptFCReg.html](https://rdrr.io/cran/fdaconcur/man/ptFCReg.html) — ptFCReg return value fields (beta0, beta, tGrid, R2, Ldf) and pointwise OLS algorithm description [CITED]
- [rdrr.io/cran/fdaconcur/man/smPtFCRegCoef.html](https://rdrr.io/cran/fdaconcur/man/smPtFCRegCoef.html) — smPtFCRegCoef: local-linear smoothing via Lwls1D, bandwidth parameter bw scaled as `2.5/(nGrid-1)` [CITED]

### Tertiary (LOW confidence — general statistical literature)

- Roughness metric as sum of squared second differences: standard finite-difference approximation to ∫(β''(t))² dt — well-established in FDA literature [ASSUMED from statistical training knowledge]

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all reuse targets read and verified this session
- Algorithm (pointwise OLS + smoothing): HIGH (codebase) / MEDIUM (R convention confirmation)
- Pitfalls: HIGH — derived from verified codebase conventions and column-major layout
- Test designs: MEDIUM — roughness metric formula is standard but not verified against R output

**Research date:** 2026-08-17
**Valid until:** 2026-09-17 (stable library; no fast-moving dependencies)
