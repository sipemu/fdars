# Phase 72: jfPCA Fit/Transform Seam - Context

**Gathered:** 2026-09-05
**Status:** Ready for planning

<domain>
## Phase Boundary

Deliver a public, reusable **fit → transform** seam for joint-FPCA (jfPCA) elastic
shape analysis (requirements VEE-01, VEE-02). A `fit` step trains a transformer on
a set of curves; a `transform` step projects *new, out-of-sample* curves onto the
*trained* joint-FPCA basis in the trained coordinate system. Foundational for
Phase 73 (VEESA explainability), which consumes this seam.

Additive and non-breaking: no changes to existing public signatures (protects R +
WASM bindings + 28 examples), no new crate dependency, reuse-first over the shipped
`elastic_fpca.rs` and `alignment/` machinery.

</domain>

<decisions>
## Implementation Decisions

### Fit Transformer API Shape
- Expose the trained transformer as a **new `JfpcaModel` struct** that owns the
  trained basis + template (clean seam; does NOT mutate `JointFpcaResult`).
- `fit` consumes **raw curves + `argvals` + `ncomp` + `balance_c`** — mirrors the
  `joint_fpca` input surface and runs Karcher-mean alignment internally.
- The model stores everything needed to project new curves: the trained
  **Karcher-mean template**, `mean_psi`, joint eigenvector components
  (`vert_component` / `horiz_component`), `balance_c`, `argvals`, eigenvalues,
  plus an embedded `JointFpcaResult` so training scores are retained.
- Naming: constructor `jfpca_fit()` → `JfpcaModel`; projection method `.transform()`.

### Out-of-sample Transform
- Transform is a **method on the model**: `model.transform(&new_curves)`.
- Returns a struct `JfpcaTransform { scores, aligned, warping }` (not bare scores)
  so downstream consumers (Phase 73) get aligned functions + warping too.
- New curves are aligned to the **trained Karcher-mean template** (same alignment
  convention as fit), then projected via the existing private
  `project_onto_eigenvectors`. Do NOT re-run a fresh Karcher mean on new curves —
  that would break the round-trip gate.
- Grid handling: **require identical `argvals`**; return `FdarError::InvalidDimension`
  on mismatch (no silent resampling).

### Integration & Numerical Gates
- **Crate-root + prelude re-exports** for `JfpcaModel`, `JfpcaTransform`, `jfpca_fit`.
- Keep `project_onto_eigenvectors` **private** — drive it through the new seam; do
  not add it to the public API.
- Numerical gates (known-answer tests, the make-or-break of this phase):
  - `jfpca_fit` training scores reproduce `joint_fpca` scores within **1e-8**.
  - **fit→transform round-trip**: transforming the original training curves
    reproduces the training scores within tolerance.
- Add a **running module doctest** exercising fit → transform under `cargo test --doc`.

### Claude's Discretion
- Exact struct field visibility, internal helper factoring, doctest curve
  construction, and test-fixture design are at Claude's discretion within the
  above constraints and the crate's conventions (column-major `FdMatrix`,
  `Result<T, FdarError>`, `#[must_use]` on expensive computations, `Debug/Clone/PartialEq`
  derives, conditional serde).

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `fdars-core/src/elastic_fpca.rs` — `joint_fpca`, `vert_fpca`, `horiz_fpca`,
  `*_from_alignment` variants; `JointFpcaResult` (fields: `scores`, `eigenvalues`,
  `cumulative_variance`, `balance_c`, `vert_component`, `horiz_component`);
  private `fn project_onto_eigenvectors` (line ~865) — the projection primitive to
  drive for out-of-sample curves.
- `alignment/` — Karcher-mean alignment (`karcher_mean`, `KarcherMeanResult` with
  `aligned_data` + `gammas`), elastic distances. `vert_fpca`/`horiz_fpca` already
  take a pre-computed `KarcherMeanResult`.

### Established Patterns
- Column-major `FdMatrix` (rows = curves/observations, cols = eval points).
- All public fns return `Result<T, FdarError>`; dimension checks at entry.
- Result structs derive `Debug, Clone, PartialEq`; conditional serde via
  `#[cfg_attr(feature = "serde", derive(...))]`; `#[non_exhaustive]` on public
  result structs; `#[must_use]` on expensive computations.
- Crate-root re-exports in `src/lib.rs`; convenience re-exports in `src/prelude.rs`.

### Integration Points
- New `JfpcaModel` / `JfpcaTransform` / `jfpca_fit` re-exported from `lib.rs` + `prelude.rs`.
- Phase 73 (VEESA explainability) consumes `JfpcaModel.transform()` output.

</code_context>

<specifics>
## Specific Ideas

- Alignment of new curves to the *trained* Karcher mean must match the training-time
  alignment convention exactly, or the round-trip gate fails — confirm the exact
  convention used inside `joint_fpca` / `*_from_alignment` at plan time.
- The private `project_onto_eigenvectors` must be exposed *internally* (crate-visible)
  and driven correctly: align → project onto trained basis → scores in trained coords.

</specifics>

<deferred>
## Deferred Ideas

- Model-agnostic PFI, principal-direction reconstruction, end-to-end pipeline, doctest
  of the full VEESA path → Phase 73 (VEE-03/04/05).
- Elastic conformal anomaly detection → Phase 74.
- R/WASM binding exposure of the new surface → future milestone (issue fdars-j75).

</deferred>
