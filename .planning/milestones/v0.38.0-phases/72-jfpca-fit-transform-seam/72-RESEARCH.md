# Phase 72: jfPCA Fit/Transform Seam — Research

**Researched:** 2026-09-05
**Domain:** Elastic functional PCA — fit/transform seam over existing `elastic_fpca.rs` and `alignment/` machinery
**Confidence:** HIGH (all claims verified by direct source reads this session)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Expose the trained transformer as a **new `JfpcaModel` struct** (does NOT mutate `JointFpcaResult`).
- `fit` consumes raw curves + `argvals` + `ncomp` + `balance_c` — mirrors `joint_fpca` input surface and runs Karcher-mean alignment internally.
- Model stores: trained Karcher-mean template, `mean_psi`, `vert_component` / `horiz_component`, `balance_c`, `argvals`, eigenvalues, plus an embedded `JointFpcaResult` so training scores are retained.
- Naming: constructor `jfpca_fit()` → `JfpcaModel`; projection method `.transform()`.
- Transform is a method on the model: `model.transform(&new_curves)`.
- Returns `JfpcaTransform { scores, aligned, warping }` (not bare scores).
- New curves aligned to the **trained Karcher-mean template** (same convention as fit) via existing private `project_onto_eigenvectors`. Do NOT re-run a fresh Karcher mean on new curves.
- Grid handling: **require identical `argvals`**; return `FdarError::InvalidDimension` on mismatch (no silent resampling).
- Crate-root + prelude re-exports for `JfpcaModel`, `JfpcaTransform`, `jfpca_fit`.
- Keep `project_onto_eigenvectors` **private** — drive it through the new seam.
- Numerical gates: `jfpca_fit` training scores reproduce `joint_fpca` scores within **1e-8**; fit→transform round-trip on training curves within tolerance.
- Add a running module doctest under `cargo test --doc`.

### Claude's Discretion
- Exact struct field visibility, internal helper factoring, doctest curve construction, and test-fixture design are at Claude's discretion within the above constraints and the crate's conventions (column-major `FdMatrix`, `Result<T, FdarError>`, `#[must_use]` on expensive computations, `Debug/Clone/PartialEq` derives, conditional serde).

### Deferred Ideas (OUT OF SCOPE)
- Model-agnostic PFI, principal-direction reconstruction, end-to-end pipeline → Phase 73.
- Elastic conformal anomaly detection → Phase 74.
- R/WASM binding exposure of the new surface → future milestone (issue fdars-j75).
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| VEE-01 | Public jfPCA fit step — reusable transformer storing Karcher-mean template, `mean_psi`, `vert_component`/`horiz_component`, `balance_c`, `argvals`; training scores reproduce `joint_fpca` within 1e-8. | Full read of `elastic_fpca.rs`: `joint_fpca` path, `horiz_fpca` (provides `mean_psi`+`shooting_vectors`), `svd_scores_and_eigenvalues`. The exact computation to reproduce is documented below. |
| VEE-02 | Public out-of-sample transform — aligns new curves to trained Karcher-mean template, projects onto trained joint-FPCA basis; fit→transform round-trip on training curves within tolerance. Reuses private `project_onto_eigenvectors`. | `align_to_target` (set.rs:51) / `elastic_align_pair` (pairwise.rs:40) for single-curve alignment; exact `project_onto_eigenvectors` signature documented below; alignment convention confirmed. |
</phase_requirements>

---

## Summary

Phase 72 exposes a public fit→transform seam over the already-complete `elastic_fpca.rs` jfPCA machinery. All numerical algorithms exist; the task is threading them into a reusable `JfpcaModel` struct and a `.transform()` method that correctly applies the trained basis to out-of-sample curves.

The make-or-break is alignment convention parity. At training time `joint_fpca` calls `horiz_fpca` which calls `sphere_karcher_mean` on the ψ (sphere-space) representation of the warping functions to compute `mean_psi`, and then shoots each curve onto the tangent space to form `shooting_vectors`. For an out-of-sample curve, the transform must replicate the ψ-space alignment — specifically: (1) align the raw new curve to the **trained Karcher-mean function curve** `karcher.mean` via `elastic_align_pair`, (2) convert the resulting warping γ to ψ via `warps_to_normalized_psi`, (3) compute the shooting vector `inv_exp_map_sphere(mean_psi, psi_new, time)`, (4) build the augmented-SRSF vector using the trained `mean_q`, and (5) project the combined `[q_aug_centered | c * shooting]` vector onto the trained `v_t` eigenvectors via `project_onto_eigenvectors`.

**Primary recommendation:** Add `src/jfpca_model.rs` (new file), wire into `src/lib.rs` and `src/prelude.rs` additively. No existing signature changes. `JfpcaModel` owns the trained basis; `.transform()` calls the existing private helpers through `pub(crate)` visibility.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| jfPCA fit (Karcher alignment + joint FPCA) | `elastic_fpca` module | `alignment/` module | `joint_fpca` already performs alignment + FPCA; `jfpca_fit` is a thin wrapper that captures the trained state |
| Out-of-sample alignment to template | `alignment/` module | — | `elastic_align_pair` / `align_to_target` are the alignment primitives; the transform drives them against the fixed trained template |
| Projection onto trained basis | `elastic_fpca` module | — | `project_onto_eigenvectors` (line 865–884) is the projection primitive; it must stay private but be called through the seam |
| Score return and result packaging | `jfpca_model` module (new) | — | New `JfpcaModel` / `JfpcaTransform` types own fit results and transform output |
| Public API surface | `src/lib.rs` + `src/prelude.rs` | — | Crate-root `pub use` and prelude re-export per established crate convention |

---

## Standard Stack

No new dependencies. All required machinery already exists in `fdars-core`.

### Existing Assets Reused

| Asset | Location | Role |
|-------|----------|------|
| `joint_fpca` | `elastic_fpca.rs:286` | Training path to reproduce exactly |
| `horiz_fpca` | `elastic_fpca.rs:194` | Computes `mean_psi` + `shooting_vectors` (critical for transform) |
| `sphere_karcher_mean` | `elastic_fpca.rs:674` (`pub(crate)`) | ψ-space Karcher mean — available inside the crate |
| `warps_to_normalized_psi` | `elastic_fpca.rs:645` (`pub(crate)`) | γ → ψ conversion for new-curve warps |
| `shooting_vectors_from_psis` | `elastic_fpca.rs:705` (`pub(crate)`) | Computes shooting vector for transform step |
| `inv_exp_map_sphere` | `warping` module (used by above) | Log map on sphere — available through `pub(crate)` helpers |
| `build_augmented_srsfs` | `elastic_fpca.rs:728` (`pub(crate)`) | Builds `q_aug` for new curves |
| `center_matrix` | `elastic_fpca.rs:759` (`pub(crate)`) | Centers augmented SRSF using trained mean — available inside the crate |
| `project_onto_eigenvectors` | `elastic_fpca.rs:865` (`fn`, private) | **THE** projection primitive; must stay private but be driven from `jfpca_model.rs` within the same crate by making it `pub(crate)` or moving the model into the same file |
| `build_combined_representation` | `elastic_fpca.rs:914` (private) | Concatenates [q_centered | c * shooting] — needed for transform |
| `karcher_mean` | `alignment/karcher.rs:293` (`pub`) | Run at fit time to compute Karcher mean; result stored in model |
| `align_to_target` | `alignment/set.rs:51` (`pub`) | Align N new curves to a fixed target; drives the per-curve step in `.transform()` |
| `elastic_align_pair` | `alignment/pairwise.rs:40` (`pub`) | Single-pair alignment; alternative for single-curve transform |
| `srsf_transform` | `alignment/srsf.rs` (re-exported via `alignment/mod.rs:104`) | Computes SRSFs from aligned curves |
| `FdMatrix` | `src/matrix.rs` | Column-major storage (rows=curves, cols=eval-points) |
| `FdarError` | `src/error.rs` | `InvalidDimension`, `InvalidParameter`, `ComputationFailed` |

**Installation:** none required.

---

## Architecture Patterns

### System Architecture Diagram

```
  Raw training curves (FdMatrix, n×m)
           │
           ▼
  jfpca_fit(data, argvals, ncomp, balance_c)
           │
           ├──► karcher_mean(data, argvals, …)
           │         → KarcherMeanResult { mean, mean_srsf, gammas, aligned_data }
           │
           ├──► joint_fpca(&karcher, argvals, ncomp, balance_c)
           │         → JointFpcaResult { scores, eigenvalues, balance_c,
           │                             vert_component, horiz_component }
           │
           ├──► horiz_fpca(&karcher, argvals, ncomp)  [internal, for mean_psi]
           │         → HorizFpcaResult { mean_psi, shooting_vectors, … }
           │
           ▼
  JfpcaModel {
    karcher_mean: Vec<f64>,        // the trained template curve
    mean_q: Vec<f64>,              // trained augmented SRSF mean (length m+1)
    mean_psi: Vec<f64>,            // trained ψ Karcher mean (length m)
    vert_component: FdMatrix,      // (ncomp × (m+1)) amplitude eigenvectors
    horiz_component: FdMatrix,     // (ncomp × m) phase eigenvectors
    balance_c: f64,
    argvals: Vec<f64>,
    eigenvalues: Vec<f64>,
    joint_result: JointFpcaResult, // training scores retained here
    ncomp: usize,
  }

  Out-of-sample: model.transform(&new_curves)
           │
           ├──► align_to_target(&new_curves, &model.karcher_mean, &model.argvals, 0.0)
           │         → AlignmentSetResult { gammas, aligned_data }
           │
           ├──► warps_to_normalized_psi(&gammas, &model.argvals)   → psis
           │
           ├──► shooting vectors: inv_exp_map_sphere(mean_psi, psi_i, time) for each i
           │
           ├──► build_augmented_srsfs(&aligned_srsfs, &aligned_data, n, m) → q_aug
           │
           ├──► center using model.mean_q  (do NOT recompute mean from new data!)
           │
           ├──► build_combined_representation(&q_centered, &shooting, model.balance_c, …)
           │
           └──► project_onto_eigenvectors equivalent  → JfpcaTransform { scores, aligned, warping }
```

### Recommended Project Structure

```
fdars-core/src/
├── jfpca_model.rs          # NEW: JfpcaModel, JfpcaTransform, jfpca_fit()
├── elastic_fpca.rs         # EXISTING: promote project_onto_eigenvectors,
│                           #   build_combined_representation, center_matrix,
│                           #   build_augmented_srsfs to pub(crate)
├── lib.rs                  # +pub use jfpca_model::{jfpca_fit, JfpcaModel, JfpcaTransform};
└── prelude.rs              # +pub use crate::{JfpcaModel, JfpcaTransform, jfpca_fit};
```

---

## Critical Implementation Details (verified from source)

### 1. `project_onto_eigenvectors` — exact signature

[VERIFIED: fdars-core/src/elastic_fpca.rs:865-884]

```rust
fn project_onto_eigenvectors(
    data: &FdMatrix,       // the full (NOT yet centered) augmented-SRSF matrix (n × d)
    mean: &[f64],          // column mean vector (length d) — centering applied INSIDE
    u_cov: &nalgebra::DMatrix<f64>,  // eigenvector matrix: column k is eigenvector k
    n: usize,
    d: usize,
    ncomp: usize,
) -> FdMatrix              // scores (n × ncomp)
```

The body:
```rust
// verbatim from elastic_fpca.rs:873-883
let mut scores = FdMatrix::zeros(n, ncomp);
for k in 0..ncomp {
    for i in 0..n {
        let mut s = 0.0;
        for j in 0..d {
            s += (data[(i, j)] - mean[j]) * u_cov[(j, k)];
        }
        scores[(i, k)] = s;
    }
}
```

**Note:** This function is used in `vert_fpca` only (amplitude scores on the covariance-SVD `u_cov`). The `joint_fpca` path uses `svd_scores_and_eigenvalues` (line 783), not `project_onto_eigenvectors`. See section 3 below.

### 2. `joint_fpca` score computation — the path to reproduce

[VERIFIED: fdars-core/src/elastic_fpca.rs:286-358]

`joint_fpca` does NOT call `project_onto_eigenvectors`. It calls `svd_scores_and_eigenvalues` on the SVD of the full combined matrix. The steps are:

1. `vert_fpca(karcher, argvals, ncomp)` — runs (result discarded; only side effect is ncomp clamping)
2. `horiz_fpca(karcher, argvals, ncomp)` — runs; captures `horiz.shooting_vectors` and `horiz.mean_psi`
3. Build `qn`: either from `karcher.aligned_srsfs` (if `Some`) or via `srsf_transform(&karcher.aligned_data, argvals)`
4. `q_aug = build_augmented_srsfs(&qn, &karcher.aligned_data, n, m)` — shape (n × (m+1))
5. `(q_centered, _mean_q) = center_matrix(&q_aug, n, m_aug)` — `mean_q` is the training mean (must be stored!)
6. `c = balance_c` or optimized via golden section
7. `combined = build_combined_representation(&q_centered, &shooting, c, n, m_aug, m)` — shape (n × (m+1+m))
8. `svd = SVD::new(combined.to_dmatrix(), true, true)`
9. `(scores, eigenvalues) = svd_scores_and_eigenvalues(&svd, ncomp, n)` — scores = U * S (each col k: `u[(i,k)] * sv[k]`)
10. `split_joint_eigenvectors(v_t, ncomp, m_aug, m)` → `(vert_component, horiz_component)`

**Key insight for out-of-sample transform:** The training scores are `U * S` from SVD of `combined`. For a new curve, the equivalent operation is: project the new combined vector `[q_aug_centered | c * shooting]` onto the right singular vectors `V^T`, which equals `combined_new @ V` (where V's columns are the right singular vectors, i.e., rows of `v_t`). This is equivalent to `combined_new_row · v_t[k, :]` for score k.

Therefore the out-of-sample score for a new curve is:
```
score_i_k = sum_j (combined_new[(i,j)] * v_t[(k,j)])
```
where `combined_new` is built with the **trained** `mean_q` (not recomputed from new data) and `v_t` rows are the `horiz_component` concatenated with `vert_component` (in the reverse order from `split_joint_eigenvectors`).

`split_joint_eigenvectors` [VERIFIED: elastic_fpca.rs:818-835]:
```rust
// verbatim
for k in 0..ncomp {
    for j in 0..m_aug {
        vert_component[(k, j)] = v_t[(k, j)];        // first m+1 cols
    }
    for j in 0..m {
        horiz_component[(k, j)] = v_t[(k, m_aug + j)]; // next m cols
    }
}
```

So `v_t` row k = `[vert_component[k, ..] | horiz_component[k, ..]]` and reconstruction:
```
combined_new_col = [q_aug_centered_col (m+1 dims) | c * shooting_col (m dims)]
score_k = dot(combined_new_col, v_t_row_k)
        = dot(q_aug_centered_col, vert_component_row_k)
          + c * dot(shooting_col, horiz_component_row_k)
```

**This is the exact out-of-sample projection formula.** `project_onto_eigenvectors` from `vert_fpca` is NOT the right primitive to call for joint FPCA; instead the combined dot product above must be computed.

### 3. `mean_psi` location

[VERIFIED: fdars-core/src/elastic_fpca.rs:50-68]

`mean_psi` is a field of `HorizFpcaResult`, NOT `JointFpcaResult` or `KarcherMeanResult`.

```rust
// verbatim HorizFpcaResult fields (lines 52-68)
pub struct HorizFpcaResult {
    pub scores: FdMatrix,
    pub eigenfunctions_psi: FdMatrix,
    pub eigenfunctions_gam: FdMatrix,
    pub eigenvalues: Vec<f64>,
    pub cumulative_variance: Vec<f64>,
    pub mean_psi: Vec<f64>,            // ← the ψ Karcher mean on the sphere
    pub shooting_vectors: FdMatrix,    // ← training shooting vectors (n × m)
}
```

`JointFpcaResult` fields [VERIFIED: elastic_fpca.rs:72-86]:
```rust
pub struct JointFpcaResult {
    pub scores: FdMatrix,
    pub eigenvalues: Vec<f64>,
    pub cumulative_variance: Vec<f64>,
    pub balance_c: f64,
    pub vert_component: FdMatrix,    // ncomp × (m+1)
    pub horiz_component: FdMatrix,   // ncomp × m
}
```

`mean_psi` and `mean_q` are NOT in `JointFpcaResult` — they must be captured separately during `jfpca_fit` and stored in `JfpcaModel`.

### 4. `mean_q` (augmented-SRSF training mean) — also NOT in JointFpcaResult

[VERIFIED: fdars-core/src/elastic_fpca.rs:318]

```rust
let (q_centered, _mean_q) = center_matrix(&q_aug, n, m_aug);
```

`_mean_q` is discarded in `joint_fpca` (prefixed with `_` → unused). The fit step must capture and store this value (shape: `Vec<f64>`, length m+1). Without it, out-of-sample centering cannot reproduce the training-time centering.

### 5. `KarcherMeanResult` fields

[VERIFIED: fdars-core/src/alignment/mod.rs:149-167]

```rust
pub struct KarcherMeanResult {
    pub mean: Vec<f64>,            // Karcher mean curve (length m) — THE template
    pub mean_srsf: Vec<f64>,       // SRSF of the Karcher mean (length m)
    pub gammas: FdMatrix,          // Final warping functions (n × m)
    pub aligned_data: FdMatrix,    // Curves aligned to the mean (n × m)
    pub n_iter: usize,
    pub converged: bool,
    pub aligned_srsfs: Option<FdMatrix>,  // Pre-computed SRSFs of aligned curves
}
```

`karcher.mean` is the trained template curve used for out-of-sample alignment.

### 6. Out-of-sample alignment primitive

[VERIFIED: fdars-core/src/alignment/set.rs:51-83]

`align_to_target(&data, target, argvals, lambda)` aligns all N rows of `data` to the scalar target curve and returns `AlignmentSetResult { gammas, aligned_data, distances }`.

Signature:
```rust
pub fn align_to_target(
    data: &FdMatrix,      // new curves (n_new × m)
    target: &[f64],       // the trained Karcher mean template (length m)
    argvals: &[f64],
    lambda: f64,
) -> AlignmentSetResult
```

This is the correct out-of-sample alignment call. Use `lambda = 0.0` to match the default in `karcher_mean` (which also uses `lambda = 0.0` unless overridden).

**CRITICAL alignment convention match:** The training-time Karcher mean alignment uses `lambda = 0.0` (default call in tests: `karcher_mean(&data, &t, 10, 1e-4, 0.0)`). The transform step must use the same lambda value. Store the training `lambda` in `JfpcaModel` so the caller can match it, or default to `0.0` (safe default matching the common case).

### 7. ψ-space alignment steps for out-of-sample shooting vectors

[VERIFIED: fdars-core/src/elastic_fpca.rs:645-670, 704-725]

`warps_to_normalized_psi(gammas: &FdMatrix, argvals: &[f64]) -> Vec<Vec<f64>>` converts warp matrix → ψ vectors (all on unit sphere). Already `pub(crate)`.

`shooting_vectors_from_psis(psis, mu_psi, time) -> FdMatrix` computes `inv_exp_map_sphere(mu_psi, psi_i, time)` for each curve. Already `pub(crate)`.

For out-of-sample transform:
```
time = (0..m).map(|i| i as f64 / (m-1) as f64)  // unit grid [0,1]
psis = warps_to_normalized_psi(&new_gammas, &model.argvals)
shooting_new = shooting_vectors_from_psis(&psis, &model.mean_psi, &time)
```

### 8. `build_augmented_srsfs` and centering for out-of-sample curves

[VERIFIED: fdars-core/src/elastic_fpca.rs:728-755, 759-774]

`build_augmented_srsfs(qn, aligned_data, n, m) -> FdMatrix` — already `pub(crate)`. Augments with `sign(f(id)) * sqrt(|f(id)|)` at column `id = m / 2`.

For centering: do NOT call `center_matrix` on new curves (that would recompute the mean from new data). Instead subtract the **trained** `mean_q` manually:
```rust
let mut q_aug_centered = q_aug.clone();
for i in 0..n_new {
    for j in 0..(m+1) {
        q_aug_centered[(i, j)] -= model.mean_q[j];
    }
}
```

### 9. Scoring formula recap

[VERIFIED from analysis of elastic_fpca.rs:326-348]

For out-of-sample score of curve i, component k:
```
combined_i = [q_aug_centered_i[0..m+1] | c * shooting_i[0..m]]
score_i_k  = sum_{j=0}^{m} (q_aug_centered_i[j] * vert_component[k, j])
           + c * sum_{j=0}^{m-1} (shooting_i[j] * horiz_component[k, j])
```

where `vert_component` and `horiz_component` are rows of `v_t` stored in the `JointFpcaResult` embedded in `JfpcaModel`.

### 10. Visibility promotion needed in `elastic_fpca.rs`

Current visibility of needed private functions [VERIFIED: elastic_fpca.rs]:

| Function | Current | Required |
|----------|---------|----------|
| `project_onto_eigenvectors` (line 865) | `fn` (private) | NOT called directly by transform — joint scores use a different formula (see §2) |
| `build_augmented_srsfs` (line 728) | `pub(crate)` | Already accessible |
| `center_matrix` (line 759) | `pub(crate)` | Already accessible |
| `warps_to_normalized_psi` (line 645) | `pub(crate)` | Already accessible |
| `shooting_vectors_from_psis` (line 705) | `pub(crate)` | Already accessible |
| `build_combined_representation` (line 914) | `fn` (private) | Promote to `pub(crate)` OR inline equivalent in `jfpca_model.rs` |
| `sphere_karcher_mean` (line 674) | `pub(crate)` | Already accessible |

**Minimum change required:** promote `build_combined_representation` to `pub(crate)`. The scoring can then be an inner product of the combined vector with `v_t` rows, which is trivial to inline.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Curve-to-template alignment | Custom DP warp | `align_to_target` / `elastic_align_pair` | Already handles SRSF computation, DP, sqrt(γ') adjustment |
| ψ conversion from warps | Manual sqrt-gradient | `warps_to_normalized_psi` | Handles domain normalization, clamping, sphere normalization |
| Shooting vector computation | Manual `inv_exp_map` loop | `shooting_vectors_from_psis` | Already parallel-dispatched |
| Augmented SRSF matrix | Custom augmentation | `build_augmented_srsfs` | Handles sign convention at `id = m/2` |
| Joint FPCA (training) | New FPCA implementation | `joint_fpca` call, then capture state | All algorithms exist; just retain intermediate results |
| Score formula | Re-derive from scratch | See §9 exact formula (vert_component / horiz_component dot products) | Direct dot product against stored eigenvectors — no SVD needed at transform time |

**Key insight:** At transform time, NO SVD is required. Scores are a simple inner product of the new combined representation with the stored eigenvectors. Computational cost is O(n_new × ncomp × (m+1+m)).

---

## Common Pitfalls

### Pitfall 1: Calling `project_onto_eigenvectors` for joint scores
**What goes wrong:** `project_onto_eigenvectors` computes `(data[i,j] - mean[j]) * u_cov[(j,k)]`, which is correct for `vert_fpca` (covariance SVD, eigenvectors in U). But `joint_fpca` uses SVD of the data matrix directly and scores are `u * s` (left singular vectors scaled by singular values). These are numerically different.
**Why it happens:** The function name sounds generic.
**How to avoid:** For joint FPCA scores, use the `v_t`-based dot product formula (§9 above), NOT `project_onto_eigenvectors`.
**Warning signs:** Round-trip test fails at high absolute error (>1e-4).

### Pitfall 2: Recomputing `mean_q` from new curves
**What goes wrong:** Centering new curves' augmented SRSFs by subtracting their own mean instead of the trained `mean_q` breaks coordinate system alignment.
**Why it happens:** Calling `center_matrix(&new_q_aug, ...)` recomputes the mean from new data.
**How to avoid:** Store `mean_q` (length m+1, `Vec<f64>`) in `JfpcaModel`; subtract it elementwise during transform.
**Warning signs:** Round-trip test passes but out-of-sample scores are systematically offset.

### Pitfall 3: Fresh Karcher mean on new curves
**What goes wrong:** Running `karcher_mean` on the new curves computes a new template; alignment to that template is in a different coordinate system than the training alignment.
**Why it happens:** Re-using the training workflow naively.
**How to avoid:** Use `align_to_target(&new_curves, &model.karcher_mean, ...)` — aligns to the FIXED trained template, not a new mean.
**Warning signs:** Round-trip test fails because training curves realigned to themselves give different gammas than stored training gammas.

### Pitfall 4: Lambda mismatch between fit and transform
**What goes wrong:** Training Karcher mean uses `lambda = 0.0`; transform uses non-zero lambda → different alignment geodesic → different shooting vectors → score mismatch.
**How to avoid:** Store `lambda` in `JfpcaModel` or document the default (0.0).

### Pitfall 5: `mean_psi` not captured
**What goes wrong:** `joint_fpca` discards `horiz.mean_psi` — it is not returned in `JointFpcaResult`. If not captured in `jfpca_fit`, out-of-sample shooting vectors cannot be computed.
**How to avoid:** In `jfpca_fit`, run `horiz_fpca` explicitly and store `.mean_psi`.

### Pitfall 6: ncomp clamping discrepancy
**What goes wrong:** `joint_fpca` clamps `ncomp = ncomp.min(n - 1)` (line 311), which can differ from the user-supplied value. If `JfpcaModel.ncomp` stores the user-supplied value rather than the clamped value, projection loops over wrong range.
**How to avoid:** Store clamped `ncomp` = `joint_result.eigenvalues.len()` in `JfpcaModel`.

### Pitfall 7: clippy `--all-targets` catches test code
**What goes wrong:** Clippy in CI uses `--all-targets --features linalg,parallel -- -D warnings`; dead-code or missing `#[allow]` in test module fails the gate.
**How to avoid:** Run full clippy gate before committing.

### Pitfall 8: `cargo fmt --check` drift
**What goes wrong:** `--no-verify` commits to dodge slow hook leave fmt drift; CI fails.
**How to avoid:** Run `cargo fmt` before every commit.

---

## Code Examples

### JfpcaModel struct skeleton (conventions-compliant)

```rust
// Source: derived from verified codebase conventions (elastic_fpca.rs:72-86, alignment/mod.rs:149-167)
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct JfpcaModel {
    /// Trained Karcher-mean template curve (length m).
    pub karcher_mean: Vec<f64>,
    /// Mean of augmented SRSF matrix (length m+1). Required for centering new curves.
    pub mean_q: Vec<f64>,
    /// Trained ψ Karcher mean on the Hilbert sphere (length m). Required for shooting vectors.
    pub mean_psi: Vec<f64>,
    /// Vertical (amplitude) eigenvector component (ncomp × (m+1)).
    pub vert_component: FdMatrix,
    /// Horizontal (phase) eigenvector component (ncomp × m).
    pub horiz_component: FdMatrix,
    /// Phase-vs-amplitude balance weight used at training time.
    pub balance_c: f64,
    /// Evaluation grid (length m).
    pub argvals: Vec<f64>,
    /// Eigenvalues (length ncomp, clamped).
    pub eigenvalues: Vec<f64>,
    /// Number of principal components (clamped to n-1 at training time).
    pub ncomp: usize,
    /// Full joint FPCA result from training (contains training scores).
    pub joint_result: JointFpcaResult,
    /// Warp penalty weight used at training time (for matching during transform).
    pub lambda: f64,
}
```

### JfpcaTransform struct skeleton

```rust
// Source: derived from CONTEXT.md decisions + codebase conventions
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct JfpcaTransform {
    /// PC scores in the trained coordinate system (n_new × ncomp).
    pub scores: FdMatrix,
    /// New curves aligned to the trained Karcher-mean template (n_new × m).
    pub aligned: FdMatrix,
    /// Warping functions mapping new curves to the template (n_new × m).
    pub warping: FdMatrix,
}
```

### `jfpca_fit` skeleton

```rust
// Source: derived from verified joint_fpca path (elastic_fpca.rs:286-358)
#[must_use = "expensive computation whose result should not be discarded"]
pub fn jfpca_fit(
    data: &FdMatrix,
    argvals: &[f64],
    ncomp: usize,
    balance_c: Option<f64>,
) -> Result<JfpcaModel, FdarError> {
    // 1. Run Karcher mean alignment
    let karcher = karcher_mean(data, argvals, 20, 1e-4, 0.0);

    // 2. Run joint FPCA (reproduces training scores)
    let joint_result = joint_fpca(&karcher, argvals, ncomp, balance_c)?;

    // 3. Run horiz_fpca to capture mean_psi (joint_fpca discards it)
    let horiz = horiz_fpca(&karcher, argvals, ncomp)?;

    // 4. Compute mean_q (center_matrix on augmented SRSFs)
    let (n, m) = karcher.aligned_data.shape();
    let qn = srsf_transform(&karcher.aligned_data, argvals);
    let q_aug = build_augmented_srsfs(&qn, &karcher.aligned_data, n, m);
    let (_, mean_q) = center_matrix(&q_aug, n, m + 1);

    let ncomp_actual = joint_result.eigenvalues.len();
    Ok(JfpcaModel {
        karcher_mean: karcher.mean.clone(),
        mean_q,
        mean_psi: horiz.mean_psi,
        vert_component: joint_result.vert_component.clone(),
        horiz_component: joint_result.horiz_component.clone(),
        balance_c: joint_result.balance_c,
        argvals: argvals.to_vec(),
        eigenvalues: joint_result.eigenvalues.clone(),
        ncomp: ncomp_actual,
        joint_result,
        lambda: 0.0,
    })
}
```

### `.transform()` score formula (verified)

```rust
// Source: derived from elastic_fpca.rs:326-348 (build_combined_representation +
//         svd_scores_and_eigenvalues analysis) — out-of-sample projection
fn project_joint(
    q_aug_centered: &FdMatrix,   // n_new × (m+1), already mean-subtracted with model.mean_q
    shooting: &FdMatrix,         // n_new × m
    balance_c: f64,
    vert_component: &FdMatrix,   // ncomp × (m+1)
    horiz_component: &FdMatrix,  // ncomp × m
    n: usize,
    ncomp: usize,
    m: usize,
) -> FdMatrix {
    let mut scores = FdMatrix::zeros(n, ncomp);
    for k in 0..ncomp {
        for i in 0..n {
            let mut s = 0.0;
            // amplitude part (m+1 dims)
            for j in 0..(m + 1) {
                s += q_aug_centered[(i, j)] * vert_component[(k, j)];
            }
            // phase part (m dims)
            for j in 0..m {
                s += balance_c * shooting[(i, j)] * horiz_component[(k, j)];
            }
            scores[(i, k)] = s;
        }
    }
    scores
}
```

### lib.rs re-export pattern (existing convention)

```rust
// Source: verified from fdars-core/src/lib.rs:501-505
// EXISTING:
pub use elastic_fpca::{
    horiz_fpca, horiz_fpca_from_alignment, joint_fpca, joint_fpca_from_alignment, vert_fpca,
    vert_fpca_from_alignment, HorizFpcaResult, JointFpcaResult, VertFpcaResult,
};

// ADD (new module):
pub use jfpca_model::{jfpca_fit, JfpcaModel, JfpcaTransform};
```

### prelude.rs re-export pattern

```rust
// Source: verified from fdars-core/src/prelude.rs:71
// EXISTING:
pub use crate::elastic_fpca::{HorizFpcaResult, JointFpcaResult, VertFpcaResult};

// ADD:
pub use crate::{JfpcaModel, JfpcaTransform, jfpca_fit};
```

---

## Validation Architecture

`workflow.nyquist_validation` is `true` in `.planning/config.json` — this section is required.

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Rust built-in test harness (`#[test]`, `#[cfg(test)]`) |
| Config file | none (uses cargo) |
| Quick run command | `cargo test -p fdars-core --lib jfpca_model -- --nocapture` |
| Full suite command | `cargo test -p fdars-core --features linalg,parallel` |
| Doctest command | `cargo test -p fdars-core --doc` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| VEE-01a | `jfpca_fit` training scores reproduce `joint_fpca` scores within 1e-8 | unit (known-answer) | `cargo test -p fdars-core --lib jfpca_model::tests::test_fit_scores_match_joint_fpca` | ❌ Wave 0 |
| VEE-01b | `JfpcaModel` stores `mean_psi`, `mean_q`, `vert_component`, `horiz_component` correctly | unit | `cargo test -p fdars-core --lib jfpca_model::tests::test_model_fields_populated` | ❌ Wave 0 |
| VEE-02a | fit→transform round-trip on training curves reproduces training scores | unit (known-answer) | `cargo test -p fdars-core --lib jfpca_model::tests::test_roundtrip_training_curves` | ❌ Wave 0 |
| VEE-02b | grid-mismatch returns `FdarError::InvalidDimension` | unit (error path) | `cargo test -p fdars-core --lib jfpca_model::tests::test_transform_grid_mismatch_error` | ❌ Wave 0 |
| VEE-01+02 | Module doctest `jfpca_fit` → `.transform()` compiles and runs | doctest | `cargo test -p fdars-core --doc` | ❌ Wave 0 |
| Integration | All existing `elastic_fpca` tests still pass (non-breaking) | regression | `cargo test -p fdars-core --lib elastic_fpca` | ✅ (existing) |

### Key Tolerance Values

- Training-score reproduction: **1e-8** (VEE-01)
- Round-trip tolerance: **1e-8** (matching round-trip; training curves fed through `.transform()`)
- Use sinusoidal test data matching existing test helper `generate_test_data(n, m)` in `elastic_fpca.rs:1007-1018`

### Wave 0 Gaps

- [ ] `src/jfpca_model.rs` — the entire new module
- [ ] Test functions listed in the table above (to be placed in `#[cfg(test)] mod tests` inside `jfpca_model.rs`)
- [ ] `lib.rs` additive `pub use` line
- [ ] `prelude.rs` additive `pub use` line
- [ ] `elastic_fpca.rs` visibility changes: `build_combined_representation` → `pub(crate)`

*(No framework install needed — existing `#[test]` harness covers all cases.)*

### Sampling Rate

- Per task commit: `cargo test -p fdars-core --lib jfpca_model`
- Per wave merge: `cargo test -p fdars-core --features linalg,parallel`
- Phase gate: full suite green (`cargo test -p fdars-core --features linalg,parallel`) + clippy gate + `cargo fmt --check` + `cargo test --doc`

---

## Security Domain

`security_enforcement` is `true` in `.planning/config.json`, ASVS level 1.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | library only; no auth |
| V3 Session Management | no | stateless computation |
| V4 Access Control | no | library only |
| V5 Input Validation | yes | dimension checks at function entry — pattern already established throughout codebase |
| V6 Cryptography | no | no cryptographic operations |

### Threat Patterns

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Out-of-bounds matrix access via wrong `m` or `ncomp` | Tampering / DoS | Dimension check at `jfpca_fit` and `.transform()` entry using `FdarError::InvalidDimension` |
| Grid mismatch (new curves have different `m`) | Tampering | Explicit `argvals.len() != model.argvals.len()` check → `InvalidDimension` |
| NaN/Inf propagation from degenerate input | DoS | Existing karcher / SVD error paths surface `ComputationFailed`; no new handling needed |

No cryptographic concerns. All input validation follows the crate's established `Result<T, FdarError>` pattern.

---

## Runtime State Inventory

SKIP — this is a greenfield additive module (no rename/refactor).

---

## Environment Availability

SKIP — no external dependencies beyond the existing Rust toolchain and `fdars-core` crate.

Available toolchain [ASSUMED — known from CLAUDE.md]:
- Rust stable 1.97.0 (exceeds MSRV 1.81.0 and linalg-feature requirement 1.84.0)
- `cargo`, `clippy`, `rustfmt` available
- `cargo test --features linalg,parallel` is the standard gate

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Training lambda defaults to `0.0` in `jfpca_fit` (matching the test-established default `karcher_mean(&data, &t, 10, 1e-4, 0.0)`) | Implementation Details §6 | Round-trip gate fails if caller used non-zero lambda during training but transform uses 0.0; mitigated by storing `lambda` in `JfpcaModel` |
| A2 | `cargo test --doc` is the correct command for running doctests in this crate | Validation Architecture | Low risk — standard Cargo behavior |

**If this table is short:** All other claims were verified by direct source read this session.

---

## Open Questions

1. **`karcher_mean` iteration count for `jfpca_fit`**
   - What we know: tests use `max_iter = 10`; the `karcher_mean` doc example uses `max_iter = 20`.
   - What's unclear: what default `max_iter` should `jfpca_fit` use? The function could accept it as a parameter or hard-code 20.
   - Recommendation: accept it as a parameter with a default-friendly constant (e.g., 20), matching the doc example.

2. **Whether to expose `lambda` as a `jfpca_fit` parameter**
   - What we know: `karcher_mean` accepts `lambda`; tests use 0.0.
   - What's unclear: should `jfpca_fit` expose `lambda` for power users?
   - Recommendation: yes, expose it with default 0.0; store it in `JfpcaModel` so `.transform()` can match.

---

## Sources

### Primary (HIGH confidence — direct source reads this session)

- `fdars-core/src/elastic_fpca.rs:1-1499` — full read: all function signatures, score computation paths, private helper visibility
- `fdars-core/src/alignment/mod.rs:1-230` — `KarcherMeanResult` struct, `AlignmentOutput` trait, module re-exports
- `fdars-core/src/alignment/karcher.rs:1-489` — `karcher_mean` implementation
- `fdars-core/src/alignment/set.rs:1-124` — `align_to_target`, `apply_stored_warps`
- `fdars-core/src/alignment/pairwise.rs:1-80` — `elastic_align_pair`, `elastic_align_pair_banded`
- `fdars-core/src/lib.rs` (grep) — confirmed existing `elastic_fpca` re-exports at lines 501-505
- `fdars-core/src/prelude.rs` (grep) — confirmed existing prelude re-exports at line 71
- `.planning/phases/72-jfpca-fit-transform-seam/72-CONTEXT.md` — locked decisions
- `.planning/REQUIREMENTS.md` — VEE-01, VEE-02 definitions
- `.planning/STATE.md` — milestone decisions and hazard log
- `.planning/config.json` — `nyquist_validation: true`, `security_enforcement: true`

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new dependencies; all reused assets directly verified
- Architecture: HIGH — `joint_fpca` path fully read; score formula derived from verified source
- Pitfalls: HIGH — `mean_psi` absence in `JointFpcaResult`, `mean_q` discard, ncomp clamping all confirmed by source reads
- Validation: HIGH — test framework and fixture patterns verified from existing tests in `elastic_fpca.rs:1002-1498`

**Research date:** 2026-09-05
**Valid until:** 2026-11-05 (stable codebase; no external deps; re-verify if `elastic_fpca.rs` changes)
