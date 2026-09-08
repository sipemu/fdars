# Phase 88: Large Suffix Batch - Research

**Researched:** 2026-09-08
**Domain:** Rust API refactoring — crate-wide `_1d`/`_2d` suffix unification onto `Dim` dispatch
**Confidence:** HIGH

## Summary

Phase 88 is the terminal naming-unification pass for `fdars-core` v0.42.0. It consolidates all remaining public `_1d`/`_2d` suffix functions onto `Dim`-dispatched signatures. The complete enumeration identifies **46 `pub fn *_1d` functions** (excluding `_1d_seeded` variants) and **3 `pub fn *_2d` functions**. Of the 46 `_1d` functions, 5 are true exclusions that must not be changed: `from_1d` (a method on `FdCurveSet`, not a dimensionality selector), `basis_to_fdata_1d`, `fdata_to_basis_1d` (legacy `i32` shims for R interop), and 2 inherently-1D functions (`fdata_to_pc_1d`, `fdata_to_pls_1d`) flagged as grey areas. The remaining 41 are in-scope for consolidation.

The pattern is fully established: v0.41.0 created 5 dispatchers (`mean`, `fraiman_muniz`, `modal`, `random_projection`, `random_tukey`) all of which still publicly expose their `_1d` sibling alongside the dispatcher. Phase 88 does the v0.42.0 **removal** step: downgrade those `_1d` bodies to `pub(crate) fn ..._impl` (or keep as `pub(crate)` with the old name), drop them from all re-exports, and update every external call site (examples, tests, benches). Internal (in-crate) callers of a function downgraded to `pub(crate)` do NOT need rewriting — they remain valid. The external call-site migration is the true blast radius.

The 3 lone `_2d` functions split cleanly: `fosr_2d` and `predict_fosr_2d` stay unchanged (naming collision with existing `fosr`/`predict_fosr`); `simpsons_weights_2d` stays unchanged (it is a `pub` helper used internally but has no `_1d` sibling, and `simpsons_weights` already exists as the 1D counterpart).

**Primary recommendation:** Execute in 7 grouped tasks by module, sequenced lowest-risk first (Cat 1 downgrades before Cat 2 new-dispatcher adds). Use `cargo build` as the per-task compile gate; full suite (`cargo test --features linalg,parallel`, `cargo build --examples`, `cargo build --benches`, `cargo test --doc`) at the wave boundary.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
The v0.41.0 `Dim`-dispatch pattern is the canonical pattern. Both dispatcher arms forward to the `_1d` primitive (now private): `match dim { Dim::One | Dim::Two => name_impl(..) }`. Public `_1d`/`_2d` names are REMOVED (v0.42.0 breaking pass). Byte-identical `_impl` bodies — code-review gate confirms zero numeric drift.

### Claude's Discretion
Pure infrastructure/refactor — all mechanics at Claude's discretion, following the established convention.

### Deferred Ideas (OUT OF SCOPE)
None — this is the terminal naming-unification phase. Release prep (REL-01) is Phase 89.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| NAME-04 | The large remaining batch of lone-`_1d`/`_2d` suffix functions across the crate is consolidated onto `Dim` dispatch, and all 28 examples + docs updated to the new surface | Complete enumeration below; per-function category and new signatures specified; re-export inventory complete; external call-site counts by file |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| _impl body preservation | fdars-core src/ | — | Rename to `pub(crate)` or private; byte-identical, no logic change |
| Dispatcher addition | fdars-core src/ | — | New `pub fn name(.., dim: Dim)` per Cat 2 function |
| Re-export cleanup | src/lib.rs, src/prelude.rs, module mod.rs files | — | Drop old `_1d` names, add bare dispatcher names |
| External call-site update | examples/, tests/, benches/ | src/ internal callers | Compile-time gate proves completeness; internal callers of `pub(crate)` need no change |

---

## 1. Complete Enumeration of Public `_1d`/`_2d` Functions

**Methodology:** `grep -rnE 'pub fn [a-z_]+_(1d|2d)\b' fdars-core/src/` [VERIFIED: grep this session]

### 1.1 All `pub fn *_1d` (46 total, excluding `_seeded` variants)

[VERIFIED: grep this session — exhaustive, single authoritative pass]

| File | Line | Function Name | Category |
|------|------|---------------|----------|
| `src/basis/auto_select.rs` | 274 | `select_basis_auto_1d` | Cat 2 (new dispatcher needed) |
| `src/basis/fourier_fit.rs` | 42 | `fourier_fit_1d` | Cat 2 (new dispatcher needed) |
| `src/basis/projection.rs` | 160 | `fdata_to_basis_1d` | Cat 5 EXCLUSION (load-bearing R interop shim) |
| `src/basis/projection.rs` | 226 | `basis_to_fdata_1d` | Cat 5 EXCLUSION (load-bearing R interop shim) |
| `src/basis/pspline.rs` | 66 | `pspline_fit_1d` | Cat 2 (new dispatcher needed) |
| `src/depth/band.rs` | 30 | `band_1d` | Cat 2 (new dispatcher needed) |
| `src/depth/band.rs` | 43 | `modified_band_1d` | Cat 2 (new dispatcher needed) |
| `src/depth/band.rs` | 57 | `modified_epigraph_index_1d` | Cat 2 (new dispatcher needed) |
| `src/depth/erl.rs` | 60 | `extreme_rank_length_depth_1d` | Cat 2 (new dispatcher needed) |
| `src/depth/extremal.rs` | 59 | `extremal_depth_1d` | Cat 2 (new dispatcher needed) |
| `src/depth/fraiman_muniz.rs` | 33 | `fraiman_muniz_1d` | Cat 1 (dispatcher `fraiman_muniz` exists) |
| `src/depth/half_region.rs` | 40 | `half_region_depth_1d` | Cat 2 (new dispatcher needed) |
| `src/depth/half_region.rs` | 115 | `modified_half_region_depth_1d` | Cat 2 (new dispatcher needed) |
| `src/depth/hypo_epi.rs` | 35 | `hypograph_index_1d` | Cat 2 (new dispatcher needed) |
| `src/depth/hypo_epi.rs` | 94 | `epigraph_index_1d` | Cat 2 (new dispatcher needed) |
| `src/depth/hypo_epi.rs` | 152 | `modified_hypograph_index_1d` | Cat 2 (new dispatcher needed) |
| `src/depth/linf.rs` | 33 | `linfinity_depth_1d` | Cat 2 (new dispatcher needed) |
| `src/depth/modal.rs` | 18 | `modal_1d` | Cat 1 (dispatcher `modal` exists) |
| `src/depth/random_projection.rs` | 33 | `random_projection_1d` | Cat 1 (dispatcher `random_projection` exists) |
| `src/depth/random_tukey.rs` | 12 | `random_tukey_1d` | Cat 1 (dispatcher `random_tukey` exists) |
| `src/depth/rpd.rs` | 38 | `rpd_depth_1d` | Cat 2 (new dispatcher needed) |
| `src/depth/tvd.rs` | 87 | `total_variation_depth_1d` | Cat 2 (new dispatcher needed) |
| `src/fdata.rs` | 168 | `mean_1d` | Cat 1 (dispatcher `mean` exists) |
| `src/fdata.rs` | 222 | `center_1d` | Cat 2 (new dispatcher needed) |
| `src/fdata.rs` | 759 | `norm_lp_1d` | Cat 2 (new dispatcher needed) |
| `src/matrix.rs` | 495 | `from_1d` (method on `FdCurveSet`) | Cat 5 EXCLUSION (suffix not dimensionality selector; means "from a 1D FdMatrix") |
| `src/metric/basis_coef.rs` | 62 | `basis_coef_self_1d` | Cat 3 (`_self`/`_cross` family) |
| `src/metric/basis_coef.rs` | 114 | `basis_coef_cross_1d` | Cat 3 (`_self`/`_cross` family) |
| `src/metric/deriv.rs` | 100 | `deriv_self_1d` | Cat 3 (`_self`/`_cross` family) |
| `src/metric/deriv.rs` | 159 | `deriv_cross_1d` | Cat 3 (`_self`/`_cross` family) |
| `src/metric/dtw.rs` | 71 | `dtw_self_1d` | Cat 3 (`_self`/`_cross` family) |
| `src/metric/dtw.rs` | 81 | `dtw_cross_1d` | Cat 3 (`_self`/`_cross` family) |
| `src/metric/fourier.rs` | 26 | `fourier_self_1d` | Cat 3 (`_self`/`_cross` family) |
| `src/metric/fourier.rs` | 49 | `fourier_cross_1d` | Cat 3 (`_self`/`_cross` family) |
| `src/metric/hshift.rs` | 45 | `hshift_self_1d` | Cat 3 (`_self`/`_cross` family) |
| `src/metric/hshift.rs` | 59 | `hshift_cross_1d` | Cat 3 (`_self`/`_cross` family) |
| `src/metric/kl.rs` | 136 | `kl_self_1d` | Cat 3 (`_self`/`_cross` family) |
| `src/metric/kl.rs` | 190 | `kl_cross_1d` | Cat 3 (`_self`/`_cross` family) |
| `src/metric/pca.rs` | 98 | `pca_self_1d` | Cat 3 (`_self`/`_cross` family) |
| `src/metric/pca.rs` | 179 | `pca_cross_1d` | Cat 3 (`_self`/`_cross` family) |
| `src/metric/soft_dtw.rs` | 176 | `soft_dtw_self_1d` | Cat 3 (`_self`/`_cross` family) |
| `src/metric/soft_dtw.rs` | 191 | `soft_dtw_cross_1d` | Cat 3 (`_self`/`_cross` family) |
| `src/metric/soft_dtw.rs` | 205 | `soft_dtw_div_self_1d` | Cat 3 (`_self`/`_cross` family) |
| `src/metric/soft_dtw.rs` | 222 | `soft_dtw_div_cross_1d` | Cat 3 (`_self`/`_cross` family) |
| `src/regression.rs` | 387 | `fdata_to_pc_1d` | Cat 5 GREY AREA (see §5) |
| `src/regression.rs` | 714 | `fdata_to_pls_1d` | Cat 5 GREY AREA (see §5) |

**Additional `pub fn *_1d_seeded` (3 total — EXCLUSIONS, suffix is load-bearing):**
[VERIFIED: grep this session]
- `src/depth/random_projection.rs:39` — `random_projection_1d_seeded` (Cat 5: `_1d` here is part of the parent name, `_seeded` is the differentiator; the dispatcher `random_projection` does NOT have a seeded form)
- `src/depth/random_tukey.rs:18` — `random_tukey_1d_seeded` (same logic)
- `src/depth/rpd.rs:53` — `rpd_depth_1d_seeded` (same logic)

### 1.2 All `pub fn *_2d` (3 total)

[VERIFIED: grep this session]

| File | Line | Function Name | Category |
|------|------|---------------|----------|
| `src/function_on_scalar_2d.rs` | 468 | `fosr_2d` | Cat 4 → STAY AS-IS (collision) |
| `src/function_on_scalar_2d.rs` | 625 | `predict_fosr_2d` | Cat 4 → STAY AS-IS (collision) |
| `src/helpers.rs` | 173 | `simpsons_weights_2d` | Cat 4 → STAY AS-IS (paired with `simpsons_weights`, no dimensionality ambiguity) |

---

## 2. Per-Function Classification

### Category 1: Lone `_1d` WITH Existing Bare `Dim` Dispatcher (5 functions)

[VERIFIED: grep this session — dispatchers read from source]

Action: Downgrade `_1d` body to `pub(crate) fn ..._impl` (rename, keep body verbatim). Remove the old `_1d` name from all `pub use` re-exports. The dispatcher already IS the public API.

| `_1d` Function | Existing Dispatcher | Dispatcher Body (verified) |
|---------------|--------------------|-----------------------------|
| `fraiman_muniz_1d` (fraiman_muniz.rs:33) | `fraiman_muniz` (fraiman_muniz.rs:54) | `match dim { Dim::One | Dim::Two => fraiman_muniz_1d(data_obj, data_ori, scale) }` |
| `modal_1d` (modal.rs:18) | `modal` (modal.rs:55) | `match dim { Dim::One | Dim::Two => modal_1d(data_obj, data_ori, h) }` |
| `random_projection_1d` (random_projection.rs:33) | `random_projection` (random_projection.rs:70) | `match dim { Dim::One | Dim::Two => random_projection_1d(data_obj, data_ori, nproj) }` |
| `random_tukey_1d` (random_tukey.rs:12) | `random_tukey` (random_tukey.rs:49) | `match dim { Dim::One | Dim::Two => random_tukey_1d(data_obj, data_ori, nproj) }` |
| `mean_1d` (fdata.rs:168) | `mean` (fdata.rs:193) | `match dim { Dim::One | Dim::Two => mean_1d(data) }` |

**Mechanical steps for each Cat 1:**
1. In the definition file: change `pub fn name_1d` → `pub(crate) fn name_impl` (or keep as `pub(crate) fn name_1d` — either works since the name is only internal now)
2. Update the dispatcher `match` arm to call `name_impl` (or `name_1d` if kept as-is internally)
3. In `depth/mod.rs`/`fdata.rs` re-export block: remove `pub use ...::{name_1d}` (keep `name` dispatcher)
4. In `src/lib.rs`: remove `name_1d` from re-export
5. In `src/prelude.rs`: remove `name_1d` from re-export (if present)

**Critical: `dispatch.rs` internal caller.** `src/depth/dispatch.rs` imports and calls `fraiman_muniz_1d`, `band_1d`, `modified_band_1d`, and `random_projection_1d_seeded` directly. Since these become `pub(crate)`, `dispatch.rs` (in the same crate) continues to work without change. Only the public re-exports need updating.

**Critical: `equivalence_phase50.rs` test.** This external test calls `modal_1d`, `fraiman_muniz_1d`, and `mean_1d` directly as public API to assert bit-identity with the dispatcher. Once these become `pub(crate)`, the test MUST be updated to use only the dispatcher. [VERIFIED: fdars-core/tests/equivalence_phase50.rs:131-167]

### Category 2: Lone `_1d` WITHOUT a Dispatcher (17 functions)

Action: Add `pub fn name(.., dim: Dim) { match dim { Dim::One | Dim::Two => name_impl(..) } }`, rename `pub fn name_1d` → `pub(crate) fn name_impl` (body verbatim), update re-exports.

All Cat 2 functions are lone `_1d` with no `_2d` sibling — both `Dim::One` and `Dim::Two` arms forward to the same `_impl` (the `_2d` path never diverged for these functions). [VERIFIED: no `_2d` siblings found in grep this session]

**Exact new public signatures:**

```rust
// --- depth/band.rs ---
pub fn band(data_obj: &FdMatrix, data_ori: &FdMatrix, dim: Dim) -> Vec<f64>
pub fn modified_band(data_obj: &FdMatrix, data_ori: &FdMatrix, dim: Dim) -> Vec<f64>
pub fn modified_epigraph_index(data_obj: &FdMatrix, data_ori: &FdMatrix, dim: Dim) -> Vec<f64>

// --- depth/erl.rs ---
pub fn extreme_rank_length_depth(data_obj: &FdMatrix, data_ori: &FdMatrix, dim: Dim) -> Result<Vec<f64>, FdarError>

// --- depth/extremal.rs ---
pub fn extremal_depth(data_obj: &FdMatrix, data_ori: &FdMatrix, dim: Dim) -> Result<Vec<f64>, FdarError>

// --- depth/half_region.rs ---
pub fn half_region_depth(data_obj: &FdMatrix, data_ori: &FdMatrix, dim: Dim) -> Result<Vec<f64>, FdarError>
pub fn modified_half_region_depth(data_obj: &FdMatrix, data_ori: &FdMatrix, dim: Dim) -> Result<Vec<f64>, FdarError>

// --- depth/hypo_epi.rs ---
pub fn hypograph_index(data_obj: &FdMatrix, data_ori: &FdMatrix, dim: Dim) -> Result<Vec<f64>, FdarError>
pub fn epigraph_index(data_obj: &FdMatrix, data_ori: &FdMatrix, dim: Dim) -> Result<Vec<f64>, FdarError>
pub fn modified_hypograph_index(data_obj: &FdMatrix, data_ori: &FdMatrix, dim: Dim) -> Result<Vec<f64>, FdarError>

// --- depth/linf.rs ---
pub fn linfinity_depth(data_obj: &FdMatrix, data_ori: &FdMatrix, dim: Dim) -> Result<Vec<f64>, FdarError>

// --- depth/rpd.rs ---
pub fn rpd_depth(data_obj: &FdMatrix, data_ori: &FdMatrix, argvals: &[f64], nproj: usize, nderiv: usize, dim: Dim) -> Vec<f64>

// --- depth/tvd.rs ---
pub fn total_variation_depth(data_obj: &FdMatrix, data_ori: &FdMatrix, dim: Dim) -> Result<TvdMssResult, FdarError>

// --- fdata.rs ---
pub fn center(data: &FdMatrix, dim: Dim) -> FdMatrix
pub fn norm_lp(data: &FdMatrix, argvals: &[f64], p: f64, dim: Dim) -> Vec<f64>

// --- basis/auto_select.rs ---
pub fn select_basis_auto(data: &FdMatrix, argvals: &[f64], criterion: i32, nbasis_min: usize, nbasis_max: usize, lambda_pspline: f64, use_seasonal_hint: bool, dim: Dim) -> BasisAutoSelectionResult

// --- basis/fourier_fit.rs ---
pub fn fourier_fit(data: &FdMatrix, argvals: &[f64], nbasis: usize, dim: Dim) -> Result<FourierFitResult, FdarError>

// --- basis/pspline.rs ---
pub fn pspline_fit(data: &FdMatrix, argvals: &[f64], nbasis: usize, lambda: f64, order: usize, dim: Dim) -> Option<PsplineFitResult>
```

Note: `#[must_use]` attribute must be preserved on dispatchers where the underlying `_1d` had it.

**`dispatch.rs` impact:** `src/depth/dispatch.rs` calls `band_1d`, `modified_band_1d`, `epigraph_index_1d`, `extremal_depth_1d`, `extreme_rank_length_depth_1d`, `fraiman_muniz_1d`, `half_region_depth_1d`, `hypograph_index_1d`, `linfinity_depth_1d`, `modified_half_region_depth_1d`, `modified_hypograph_index_1d`, `total_variation_depth_1d`, and `random_projection_1d_seeded`. [VERIFIED: fdars-core/src/depth/dispatch.rs:13-17] All of these become `pub(crate)` after Cat 1/2 conversion. Since `dispatch.rs` is inside `fdars-core`, all these calls remain valid — **no changes needed in `dispatch.rs`** unless the planner chooses to update the internal calls to `name_impl` for consistency (not required).

### Category 3: `_self`/`_cross` Metric Families (12 functions → 10 dispatchers)

Action: Each `_self`/`_cross` member gets its own `Dim` dispatcher. The `_1d` body becomes `pub(crate) fn name_impl`.

All metric `_self_1d`/`_cross_1d` functions have no `_2d` sibling — `Dim::One | Dim::Two` both forward to the same impl. [VERIFIED: grep this session — no `_self_2d`/`_cross_2d` found in metric/]

**Families and new public signatures:**

```rust
// --- metric/basis_coef.rs ---
pub fn basis_coef_self(data: &FdMatrix, argvals: &[f64], nbasis: usize, basis_type: ProjectionBasisType, dim: Dim) -> FdMatrix
pub fn basis_coef_cross(data1: &FdMatrix, data2: &FdMatrix, argvals: &[f64], nbasis: usize, basis_type: ProjectionBasisType, dim: Dim) -> FdMatrix

// --- metric/deriv.rs ---
pub fn deriv_self(data: &FdMatrix, argvals: &[f64], nderiv: usize, user_weights: &[f64], dim: Dim) -> FdMatrix
pub fn deriv_cross(data1: &FdMatrix, data2: &FdMatrix, argvals: &[f64], nderiv: usize, user_weights: &[f64], dim: Dim) -> FdMatrix

// --- metric/dtw.rs ---
pub fn dtw_self(data: &FdMatrix, p: f64, w: usize, dim: Dim) -> FdMatrix
pub fn dtw_cross(data1: &FdMatrix, data2: &FdMatrix, p: f64, w: usize, dim: Dim) -> FdMatrix

// --- metric/fourier.rs ---
pub fn fourier_self(data: &FdMatrix, nfreq: usize, dim: Dim) -> FdMatrix
pub fn fourier_cross(data1: &FdMatrix, data2: &FdMatrix, nfreq: usize, dim: Dim) -> FdMatrix

// --- metric/hshift.rs ---
pub fn hshift_self(data: &FdMatrix, argvals: &[f64], max_shift: usize, dim: Dim) -> FdMatrix
pub fn hshift_cross(data1: &FdMatrix, data2: &FdMatrix, argvals: &[f64], max_shift: usize, dim: Dim) -> FdMatrix

// --- metric/kl.rs ---
pub fn kl_self(data: &FdMatrix, argvals: &[f64], epsilon: f64, dim: Dim) -> FdMatrix
pub fn kl_cross(data1: &FdMatrix, data2: &FdMatrix, argvals: &[f64], epsilon: f64, dim: Dim) -> FdMatrix

// --- metric/pca.rs ---
pub fn pca_self(data: &FdMatrix, ncomp: usize, dim: Dim) -> Result<FdMatrix, FdarError>
pub fn pca_cross(data1: &FdMatrix, data2: &FdMatrix, ncomp: usize, argvals: &[f64], dim: Dim) -> Result<FdMatrix, FdarError>

// --- metric/soft_dtw.rs ---
pub fn soft_dtw_self(data: &FdMatrix, gamma: f64, dim: Dim) -> FdMatrix
pub fn soft_dtw_cross(data1: &FdMatrix, data2: &FdMatrix, gamma: f64, dim: Dim) -> FdMatrix
pub fn soft_dtw_div_self(data: &FdMatrix, gamma: f64, dim: Dim) -> FdMatrix
pub fn soft_dtw_div_cross(data1: &FdMatrix, data2: &FdMatrix, gamma: f64, dim: Dim) -> FdMatrix
```

Note: The `pca_cross_1d` signature needs verification — check the exact args [VERIFIED: fdars-core/src/metric/pca.rs:179]:
```rust
pub fn pca_cross_1d(
    data1: &FdMatrix,
    data2: &FdMatrix,
    ncomp: usize,
    argvals: &[f64],
) -> Result<FdMatrix, FdarError>
```
Dispatcher: `pub fn pca_cross(data1: &FdMatrix, data2: &FdMatrix, ncomp: usize, argvals: &[f64], dim: Dim) -> Result<FdMatrix, FdarError>`

And `deriv_cross_1d` [VERIFIED: fdars-core/src/metric/deriv.rs:159]:
```rust
pub fn deriv_cross_1d(
    data1: &FdMatrix,
    data2: &FdMatrix,
    argvals: &[f64],
    nderiv: usize,
    user_weights: &[f64],
) -> FdMatrix
```
Note: `nderiv` comes before `user_weights` in `deriv_cross_1d`, and `deriv_self_1d` signature [VERIFIED: fdars-core/src/metric/deriv.rs:100] is `(data, argvals, nderiv, user_weights)`.

### Category 4: Lone `_2d` with No `_1d` Sibling (3 functions)

[VERIFIED: grep this session]

All three STAY AS-IS. Decisions:

**`fosr_2d` (function_on_scalar_2d.rs:468):**
- Decision: STAY AS-IS (no rename, no `Dim` param)
- Justification: `fosr` (without suffix) already exists in `function_on_scalar.rs` and is the 1D function-on-scalar regression. Renaming `fosr_2d` → `fosr` would create a name collision at the crate root re-export level. The `_2d` suffix here is NOT a dimensionality selector in the Dim-dispatch sense — it genuinely identifies a different algorithm (2D surface response) vs. `fosr` (1D curve response). The two algorithms are architecturally separate: different modules, different result types (`Fosr2dResult` vs `FosrResult`). [VERIFIED: function_on_scalar.rs:275 and function_on_scalar_2d.rs:468]

**`predict_fosr_2d` (function_on_scalar_2d.rs:625):**
- Decision: STAY AS-IS (same reasoning as `fosr_2d`; `predict_fosr` already exists in function_on_scalar.rs:654)

**`simpsons_weights_2d` (helpers.rs:173):**
- Decision: STAY AS-IS
- Justification: This is a public helper with no external callers (no hits in examples/, tests/, benches/). [VERIFIED: grep this session] It is used internally by `fdata.rs` (geometric_median_2d_impl), `metric/lp.rs` (lp_cross_2d/lp_self_2d path). The `_2d` suffix is load-bearing — it distinguishes a 2D grid weight function from `simpsons_weights` (1D). There is no `Dim` ambiguity because `simpsons_weights` already exists as the 1D counterpart — they serve completely different argument lists (`argvals_s, argvals_t` vs `argvals`). A `Dim` dispatcher would require a superset signature that adds no value and just obscures intent.

### Category 5: Exclusions — NOT Changed

[VERIFIED: grep this session]

| Function | File:Line | Reason |
|---------|-----------|--------|
| `FdCurveSet::from_1d` | matrix.rs:495 | Method on `FdCurveSet`, NOT a free function. Suffix means "from a 1D `FdMatrix`" — a constructor semantics selector, not a dimensionality selector. Changing to `from_dim(data, Dim::One)` would be nonsensical. |
| `fdata_to_basis_1d` | basis/projection.rs:160 | Legacy `i32` shim for R interop — delegates to `fdata_to_basis` (which uses `ProjectionBasisType`). The `_1d` is vestigial naming from a predecessor API. Changing this breaks R bindings. The modern API is already `fdata_to_basis`. |
| `basis_to_fdata_1d` | basis/projection.rs:226 | Same: legacy `i32` shim delegating to `basis_to_fdata`. |
| `random_projection_1d_seeded` | depth/random_projection.rs:39 | `_1d_seeded` is a compound suffix; `_seeded` is the differentiator (adds `seed: Option<u64>`). The base is `random_projection_1d`, not a standalone name family. Keeping `_1d_seeded` preserves the naming relationship with the base. A `random_projection_seeded(…, dim)` dispatcher would need to be added IF we want parity, but that is a new seeded dispatcher question, not a suffix-removal question. See grey area §5.3. |
| `random_tukey_1d_seeded` | depth/random_tukey.rs:18 | Same. |
| `rpd_depth_1d_seeded` | depth/rpd.rs:53 | Same. |
| `fdata_to_pc_1d` | regression.rs:387 | GREY AREA — see §5.1 |
| `fdata_to_pls_1d` | regression.rs:714 | GREY AREA — see §5.2 |

---

## 3. Re-Export Inventory

Every `pub use` line that names a changing function. [VERIFIED: fdars-core/src/lib.rs and src/prelude.rs and module mod.rs files read this session]

### `src/lib.rs`

```rust
// CURRENT (lines 632-638) — metric block:
pub use metric::{
    dtw_cross_1d, dtw_distance, dtw_self_1d, fourier_cross_1d, fourier_self_1d, gak,
    gak_gram_matrix, gak_gram_predict, gak_gram_train, hausdorff_3d, hausdorff_cross,
    hausdorff_self, hshift_cross_1d, hshift_self_1d, lp_cross, lp_self, sbd, sbd_distance_matrix,
    sigma_gak, soft_dtw_barycenter, soft_dtw_cross_1d, soft_dtw_distance,
    soft_dtw_distance_generic, soft_dtw_div_cross_1d, soft_dtw_div_self_1d, soft_dtw_divergence,
    soft_dtw_self_1d, GakConfig, GakGramTrain, LpDomain, SbdResult, SoftDtwBarycenterResult,
};
// REMOVE: dtw_cross_1d, dtw_self_1d, fourier_cross_1d, fourier_self_1d,
//          hshift_cross_1d, hshift_self_1d, soft_dtw_cross_1d, soft_dtw_div_cross_1d,
//          soft_dtw_div_self_1d, soft_dtw_self_1d
// ADD: dtw_cross, dtw_self, fourier_cross, fourier_self, hshift_cross, hshift_self,
//      soft_dtw_cross, soft_dtw_div_cross, soft_dtw_div_self, soft_dtw_self
// Also ADD (currently NOT in lib.rs): basis_coef_self, basis_coef_cross, deriv_self, deriv_cross,
//      kl_self, kl_cross, pca_self, pca_cross

// CURRENT (lines 644-653) — depth block:
pub use depth::{
    band_1d, epigraph_index_1d, extremal_depth_1d, extreme_rank_length_depth_1d, fraiman_muniz,
    fraiman_muniz_1d, functional_boxplot, functional_depth, functional_spatial,
    half_region_depth_1d, hypograph_index_1d, kernel_functional_spatial, linfinity_depth_1d, modal,
    modal_1d, modified_band_1d, modified_epigraph_index_1d, modified_half_region_depth_1d,
    modified_hypograph_index_1d, random_projection, random_projection_1d,
    random_projection_1d_seeded, random_tukey, random_tukey_1d, random_tukey_1d_seeded,
    total_variation_depth_1d, DepthMethod, FunctionalBoxplotResult, TvdMssResult,
};
// REMOVE: band_1d, epigraph_index_1d, extremal_depth_1d, extreme_rank_length_depth_1d,
//          fraiman_muniz_1d, half_region_depth_1d, hypograph_index_1d, linfinity_depth_1d,
//          modal_1d, modified_band_1d, modified_epigraph_index_1d, modified_half_region_depth_1d,
//          modified_hypograph_index_1d, random_projection_1d, random_tukey_1d,
//          total_variation_depth_1d
// ADD: band, epigraph_index, extremal_depth, extreme_rank_length_depth, half_region_depth,
//      hypograph_index, linfinity_depth, modified_band, modified_epigraph_index,
//      modified_half_region_depth, modified_hypograph_index, rpd_depth, total_variation_depth
// KEEP: fraiman_muniz, functional_boxplot, functional_depth, functional_spatial,
//        kernel_functional_spatial, modal, random_projection, random_projection_1d_seeded,
//        random_tukey, random_tukey_1d_seeded, DepthMethod, FunctionalBoxplotResult, TvdMssResult
// NOTE: rpd_depth_1d currently NOT in lib.rs re-export — add rpd_depth (dispatcher)

// CURRENT (line 578-580) — regression block:
pub use regression::{
    fdata_to_pc_1d, fdata_to_pls_1d, project_scores_generic, FpcaResult, PlsResult,
};
// GREY AREA — see §5.1/5.2 for decision

// CURRENT (lines 671-676) — fdata block:
pub use fdata::{
    center_1d, depth_based_median, deriv, functional_covariance, functional_std,
    functional_variance, geometric_median, mean, mean_1d, norm_lp_1d, normalize,
    normalize_with_argvals, trim_mean, Deriv2DResult, DerivDomain, DerivResult,
    NormalizationMethod,
};
// REMOVE: center_1d, mean_1d, norm_lp_1d
// ADD: center, norm_lp

// CURRENT (lines 679-687) — basis block:
pub use basis::{
    basis_to_fdata, basis_to_fdata_1d, bspline_basis, bspline_basis_from_knots, constant_basis,
    construct_bspline_knots, difference_matrix, exponential_basis, fdata_to_basis,
    fdata_to_basis_1d, fourier_basis, fourier_basis_with_period, fourier_fit_1d, monomial_basis,
    polygonal_basis, power_basis, pspline_evaluate, pspline_fit_1d, pspline_fit_gcv,
    select_basis_auto_1d, select_fourier_nbasis_gcv, BasisAutoSelectionResult,
    BasisProjectionResult, BasisSystem, FourierFitResult, ProjectionBasisType, PsplineFitResult,
    SingleCurveSelection,
};
// REMOVE: fourier_fit_1d, pspline_fit_1d, select_basis_auto_1d
// ADD: fourier_fit, pspline_fit, select_basis_auto
// KEEP: basis_to_fdata_1d, fdata_to_basis_1d (Cat 5 EXCLUSIONS — R interop shims)
```

### `src/prelude.rs`

```rust
// CURRENT (lines 40-43) — depth block:
pub use crate::depth::{
    band_1d, fraiman_muniz_1d, functional_spatial, modal_1d, modified_band_1d,
    random_projection_1d, random_tukey_1d, rpd_depth_1d,
};
// REMOVE: band_1d, fraiman_muniz_1d, modal_1d, modified_band_1d, random_projection_1d,
//          random_tukey_1d, rpd_depth_1d
// ADD: band, fraiman_muniz, modal, modified_band, random_projection, random_tukey, rpd_depth
// (functional_spatial stays — already has its dispatcher)
```

### `src/depth/mod.rs`

```rust
// CURRENT (lines 30-43):
pub use band::{band_1d, modified_band_1d, modified_epigraph_index_1d};
pub use dispatch::{functional_boxplot, functional_depth, DepthMethod, FunctionalBoxplotResult};
pub use erl::extreme_rank_length_depth_1d;
pub use extremal::extremal_depth_1d;
pub use fraiman_muniz::{fraiman_muniz, fraiman_muniz_1d};
pub use half_region::{half_region_depth_1d, modified_half_region_depth_1d};
pub use hypo_epi::{epigraph_index_1d, hypograph_index_1d, modified_hypograph_index_1d};
pub use linf::linfinity_depth_1d;
pub use modal::{modal, modal_1d};
pub use random_projection::{random_projection, random_projection_1d, random_projection_1d_seeded};
pub use random_tukey::{random_tukey, random_tukey_1d, random_tukey_1d_seeded};
pub use rpd::{rpd_depth_1d, rpd_depth_1d_seeded};
pub use spatial::{functional_spatial, kernel_functional_spatial};
pub use tvd::{total_variation_depth_1d, TvdMssResult};

// AFTER — remove all _1d public names, add bare dispatcher names:
pub use band::{band, modified_band, modified_epigraph_index};  // dispatchers (new)
pub use dispatch::{functional_boxplot, functional_depth, DepthMethod, FunctionalBoxplotResult};
pub use erl::extreme_rank_length_depth;  // dispatcher (new)
pub use extremal::extremal_depth;  // dispatcher (new)
pub use fraiman_muniz::fraiman_muniz;  // drop fraiman_muniz_1d
pub use half_region::{half_region_depth, modified_half_region_depth};  // dispatchers (new)
pub use hypo_epi::{epigraph_index, hypograph_index, modified_hypograph_index};  // dispatchers (new)
pub use linf::linfinity_depth;  // dispatcher (new)
pub use modal::modal;  // drop modal_1d
pub use random_projection::{random_projection, random_projection_1d_seeded};  // drop random_projection_1d
pub use random_tukey::{random_tukey, random_tukey_1d_seeded};  // drop random_tukey_1d
pub use rpd::{rpd_depth, rpd_depth_1d_seeded};  // drop rpd_depth_1d
pub use spatial::{functional_spatial, kernel_functional_spatial};
pub use tvd::{total_variation_depth, TvdMssResult};  // dispatcher (new)
```

### `src/metric/mod.rs`

```rust
// CURRENT (lines 138-154):
pub use basis_coef::{basis_coef_cross_1d, basis_coef_self_1d};
pub use deriv::{deriv_cross_1d, deriv_self_1d};
pub use dtw::{dtw_cross_1d, dtw_distance, dtw_self_1d};
pub use fourier::{fourier_cross_1d, fourier_self_1d};
// ... hausdorff already consolidated ...
pub use hshift::{hshift_cross_1d, hshift_self_1d};
pub use kl::{kl_cross_1d, kl_self_1d};
pub use pca::{pca_cross_1d, pca_self_1d};
pub use soft_dtw:{ ..., soft_dtw_cross_1d, ..., soft_dtw_div_cross_1d, soft_dtw_div_self_1d, ..., soft_dtw_self_1d, ...};

// AFTER — swap _1d names for bare dispatcher names:
pub use basis_coef::{basis_coef_cross, basis_coef_self};
pub use deriv::{deriv_cross, deriv_self};
pub use dtw::{dtw_cross, dtw_distance, dtw_self};
pub use fourier::{fourier_cross, fourier_self};
pub use hshift::{hshift_cross, hshift_self};
pub use kl::{kl_cross, kl_self};
pub use pca::{pca_cross, pca_self};
pub use soft_dtw:{ ..., soft_dtw_cross, ..., soft_dtw_div_cross, soft_dtw_div_self, ..., soft_dtw_self, ...};
```

### `src/basis/mod.rs`

```rust
// CURRENT (line 27): pub use auto_select::{select_basis_auto_1d, BasisAutoSelectionResult, SingleCurveSelection};
// CURRENT (line 33): pub use fourier_fit::{fourier_fit_1d, select_fourier_nbasis_gcv, FourierFitResult};
// CURRENT (line 37-40): pub use projection:{ basis_to_fdata, basis_to_fdata_1d, fdata_to_basis, fdata_to_basis_1d, ...};
// CURRENT (line 41-44): pub use pspline:{ difference_matrix, pspline_evaluate, pspline_fit_1d, pspline_fit_gcv, PsplineFitResult};

// AFTER:
pub use auto_select::{select_basis_auto, BasisAutoSelectionResult, SingleCurveSelection};  // drop select_basis_auto_1d
pub use fourier_fit::{fourier_fit, select_fourier_nbasis_gcv, FourierFitResult};  // drop fourier_fit_1d
pub use projection:{ basis_to_fdata, basis_to_fdata_1d, fdata_to_basis, fdata_to_basis_1d, ...};  // KEEP _1d shims (Cat 5)
pub use pspline:{ difference_matrix, pspline_evaluate, pspline_fit, pspline_fit_gcv, PsplineFitResult};  // drop pspline_fit_1d
```

---

## 4. External Call-Site Scope

[VERIFIED: grep this session — exhaustive search of examples/, tests/, benches/]

### 4.1 Affected Examples (10 of 28 examples)

| Example | `_1d`/`_2d` call sites (approx) | Functions affected |
|---------|--------|-------------------|
| `02_functional_operations` | 7 | `mean_1d` (×3), `center_1d` (×1), `norm_lp_1d` (×3) |
| `04_basis_representation` | 4 | `fourier_fit_1d` (×3), `pspline_fit_1d` (×1) |
| `05_depth_measures` | 9 | `band_1d`, `fraiman_muniz_1d`, `modal_1d`, `modified_band_1d`, `modified_epigraph_index_1d`, `random_projection_1d`, `random_tukey_1d` |
| `06_distances_and_metrics` | 7 | `dtw_self_1d`, `fourier_self_1d`, `hshift_self_1d`, `soft_dtw_self_1d`, `soft_dtw_div_self_1d` |
| `08_regression` | 3 | `fdata_to_pc_1d`, `fdata_to_pls_1d` (grey area — see §5) |
| `09_outlier_detection` | 2 | `fraiman_muniz_1d` |
| `12_streaming_depth` | 4 | `band_1d`, `fraiman_muniz_1d`, `modified_band_1d` |
| `14_complete_pipeline` | 6 | `fdata_to_pc_1d`, `modified_band_1d`, `pspline_fit_1d` (and grey area) |
| `17_equivalence_test` | 2 | `mean_1d` |
| `22_gmm_clustering` | 2 | `fdata_to_pc_1d` (grey area) |

**18 examples are unaffected** (no `_1d`/`_2d` names in scope).

### 4.2 Affected Test Files

| File | Key `_1d`/`_2d` functions referenced |
|------|---------------------------------------|
| `tests/validate_against_r.rs` | `band_1d`, `center_1d`, `dtw_cross_1d`, `dtw_self_1d`, `fdata_to_basis_1d` (shim-kept), `fdata_to_pc_1d`, `fdata_to_pls_1d`, `fourier_cross_1d`, `fourier_fit_1d`, `fourier_self_1d`, `fraiman_muniz_1d`, `hshift_cross_1d`, `hshift_self_1d`, `mean_1d`, `modal_1d`, `modified_band_1d`, `modified_epigraph_index_1d`, `norm_lp_1d`, `pspline_fit_1d`, `random_projection_1d`, `random_projection_1d_seeded` (keep), `random_tukey_1d`, `select_basis_auto_1d`, `soft_dtw_div_self_1d`, `soft_dtw_self_1d` |
| `tests/equivalence_phase50.rs` | `fraiman_muniz_1d`, `mean_1d`, `modal_1d` (must be updated to use dispatcher + remove bit-identity check using `_1d`, OR the test restructured to compare with `name(…, Dim::One)`) |
| `tests/equivalence_phase49.rs` | `fdata_to_pc_1d` (grey area) |
| `tests/alloc_audit_fpca.rs` | `fdata_to_pc_1d` (grey area) |

### 4.3 Affected Bench Files

| File | `_1d`/`_2d` functions referenced |
|------|----------------------------------|
| `benches/audit_hotpaths.rs` | `fraiman_muniz_1d`, `fdata_to_pc_1d` (grey area) |
| `benches/basis_benchmarks.rs` | `basis_to_fdata_1d` (kept), `fdata_to_basis_1d` (kept), — bench string labels also updated |
| `benches/depth_benchmarks.rs` | `band_1d`, `dtw_self_1d`, `fourier_self_1d`, `fraiman_muniz_1d`, `modified_band_1d`, `norm_lp_1d`, `random_projection_1d` |
| `benches/regression_benchmarks.rs` | `fdata_to_pc_1d` (grey area) |

---

## 5. Grey Areas and Unresolved Decisions

### 5.1 GREY AREA: `fdata_to_pc_1d` — Dispatch on `Dim` is Arguably Awkward

**What it is:** `fdata_to_pc_1d` computes functional PCA from 1D functional data. It is inherently 1D in the sense that it expects a 1D grid `argvals: &[f64]` and computes Simpson's weights from it. There is no `fdata_to_pc_2d` sibling.

**The problem:** A `Dim`-dispatched `pub fn fdata_to_pc(data, ncomp, argvals, dim: Dim) -> Result<FpcaResult, FdarError>` where both arms do the same thing (there is no 2D FPCA path) is technically consistent with the pattern but arguably adds a `dim` parameter that serves no purpose — 1D FPCA is always 1D FPCA. The alternative is to simply rename to `fdata_to_pc` without a `Dim` parameter (a pure rename, not a dispatch consolidation).

**Callers:** `fdata_to_pc_1d` is called by ~120 internal `src/` locations and 6+ external locations (examples, tests, benches). It is the most heavily used `_1d` function in the codebase.

**Options:**
- **Option A (rename only):** `pub fn fdata_to_pc(data, ncomp, argvals) -> Result<FpcaResult, FdarError>` — no `Dim` param. Simple rename, no behavioral change. Consistent with dropping the dimensional suffix where there's only ever been one path.
- **Option B (Dim dispatch):** `pub fn fdata_to_pc(data, ncomp, argvals, dim: Dim) -> Result<FpcaResult, FdarError>` — adds `Dim::One | Dim::Two => same body`. Pattern-consistent but noisy.

**Recommendation (Claude's discretion):** Option A — rename to `fdata_to_pc` without `Dim`. The `_1d` suffix is vestigial (there is no 2D FPCA in this crate), and forcing a `Dim` parameter where no 2D path exists contradicts the spirit of "caller intent explicit." The compiler documentation in `dim.rs` says "makes caller intent explicit" — but if there is only one valid intent, the parameter is noise.

**Risk:** High call-count (120+ internal, 6+ external). Every `src/` caller must be updated. This is the largest single-function migration in the phase.

**User decision required:** Confirm Option A (plain rename, no `Dim`) is acceptable for `fdata_to_pc_1d`.

### 5.2 GREY AREA: `fdata_to_pls_1d` — Same Analysis as `fdata_to_pc_1d`

**What it is:** PLS regression from 1D functional data. No `_2d` sibling exists.

**Recommendation (Claude's discretion):** Option A — rename to `fdata_to_pls` without `Dim` param. Same reasoning as `fdata_to_pc_1d`. Lower call count (few external callers), but same conceptual issue.

**User decision required:** Confirm Option A (plain rename, no `Dim`) is acceptable for `fdata_to_pls_1d`.

### 5.3 GREY AREA: `_1d_seeded` Variants — Should Matching `_seeded` Dispatchers Be Added?

**What they are:** `random_projection_1d_seeded`, `random_tukey_1d_seeded`, `rpd_depth_1d_seeded`. These are Cat 5 exclusions (suffix-compound names). Their base forms `random_projection_1d`, `random_tukey_1d`, `rpd_depth_1d` are being removed (consolidated into `random_projection`, `random_tukey`, `rpd_depth` dispatchers). But no `_seeded` dispatcher exists.

**The gap:** After Phase 88, callers who want a seeded, Dim-parameterized depth function have no single entry point — they must use `random_projection_1d_seeded(data, data, nproj, seed)` (which is still `pub`) alongside `random_projection(data, data, nproj, Dim::One)` (unseeded). This is slightly inconsistent.

**Recommendation (Claude's discretion):** Leave the `_seeded` variants as public `_1d_seeded` names for this phase. Adding seeded dispatchers (`pub fn random_projection_seeded(…, seed: Option<u64>, dim: Dim)`) is a new API surface addition, not a suffix removal, and would require additional decisions about naming. Defer to a later phase or accept the minor inconsistency. The `_seeded` names are already used by `dispatch.rs` internally and by `validate_against_r.rs` externally — removing them would require changing those callers too.

**User decision required:** Confirm `_seeded` variants are left public for now and not consolidated in this phase.

### 5.4 GREY AREA: `select_basis_auto_1d`, `fourier_fit_1d`, `pspline_fit_1d` — Dim Parameter Utility

**Analysis:** These basis fitting functions operate on 1D functional data. No `_2d` siblings exist. A `Dim` dispatcher follows the pattern mechanically, but like `fdata_to_pc_1d`, the `Dim` parameter is a no-op (both arms do the same thing). The 87-RESEARCH noted "the `_2d` path never diverged from the `_1d` one" — so this is pattern-consistent.

**Recommendation (Claude's discretion):** Follow the same decision as for `fdata_to_pc_1d`:
- If the user accepts Option A (rename without `Dim`) for `fdata_to_pc_1d`, apply the same to the basis functions.
- If the user prefers `Dim` dispatch everywhere for consistency, add `Dim` to the basis functions too.

This is NOT a blocker — it is a stylistic question that follows from the `fdata_to_pc_1d` decision.

---

## 6. Recommended Task Decomposition

Group by module, lowest-blast-radius first, each verified by `cargo build`.

### Task 1: Cat 1 Downgrades — Depth (fraiman_muniz, modal, random_projection, random_tukey)

**Scope:** `src/depth/fraiman_muniz.rs`, `src/depth/modal.rs`, `src/depth/random_projection.rs`, `src/depth/random_tukey.rs`, `src/depth/mod.rs`

**Actions:**
1. In each definition file: rename `pub fn name_1d` → `pub(crate) fn name_impl` (body verbatim); update the dispatcher's `match` arm to call `name_impl`.
2. Update `src/depth/mod.rs`: remove `fraiman_muniz_1d`, `modal_1d`, `random_projection_1d`, `random_tukey_1d` from `pub use` lines (keep dispatchers, keep `_seeded` variants).
3. Update `src/lib.rs` depth block: remove same names.
4. Update `src/prelude.rs` depth block: remove same names.
5. Update `tests/equivalence_phase50.rs`: replace `fraiman_muniz_1d`/`modal_1d`/`mean_1d` uses with `fraiman_muniz(…, Dim::One)`/`modal(…, Dim::One)` — the bit-identity tests remain valid using the dispatcher.
6. Update all external example/test/bench callers of `fraiman_muniz_1d`, `modal_1d`, `random_projection_1d`, `random_tukey_1d`.

**Gate:** `cargo build -p fdars-core --features linalg,parallel`

### Task 2: Cat 1 Downgrade — fdata (mean_1d)

**Scope:** `src/fdata.rs`, `src/lib.rs`, `src/prelude.rs`

**Actions:**
1. `src/fdata.rs`: rename `pub fn mean_1d` → `pub(crate) fn mean_impl`; update `mean` dispatcher to call `mean_impl`.
2. Update `src/lib.rs` fdata block: remove `mean_1d`.
3. Update `src/prelude.rs`: no change (mean_1d not in prelude).
4. Update all external callers: `examples/02_functional_operations/main.rs`, `examples/17_equivalence_test/main.rs`, `tests/validate_against_r.rs` (3 hits), `tests/equivalence_phase50.rs`.
5. Update internal `src/` callers (18 locations — all `pub(crate)` or `src/` callers are now fine since `mean_impl` is still accessible).

**Gate:** `cargo build -p fdars-core --features linalg,parallel`

### Task 3: Cat 2 — Remaining Depth Functions (band, hypo_epi, erl, extremal, half_region, linf, rpd, tvd)

**Scope:** `src/depth/band.rs`, `src/depth/hypo_epi.rs`, `src/depth/erl.rs`, `src/depth/extremal.rs`, `src/depth/half_region.rs`, `src/depth/linf.rs`, `src/depth/rpd.rs`, `src/depth/tvd.rs`, `src/depth/mod.rs`

**Actions (per file):**
1. Add `pub fn name(data_obj, data_ori, [params,] dim: Dim) -> ReturnType { match dim { Dim::One | Dim::Two => name_impl(data_obj, data_ori, [params]) } }`
2. Rename `pub fn name_1d` → `pub(crate) fn name_impl`
3. Update `src/depth/mod.rs`: swap `name_1d` for `name` in `pub use` blocks
4. Update `src/lib.rs` depth block: swap `name_1d` for `name`
5. Update external callers (examples/tests/benches — relatively few; mostly internal)

**Note:** `dispatch.rs` calls most of these. Since they become `pub(crate)`, no change needed in `dispatch.rs`.

**Gate:** `cargo build -p fdars-core --features linalg,parallel`

### Task 4: Cat 2 — fdata (center_1d, norm_lp_1d)

**Scope:** `src/fdata.rs`, `src/lib.rs`

**Actions:**
1. Add `pub fn center(data, dim: Dim) -> FdMatrix { match dim { Dim::One | Dim::Two => center_impl(data) } }`
2. Rename `pub fn center_1d` → `pub(crate) fn center_impl`
3. Same for `norm_lp`/`norm_lp_1d`
4. Update `src/lib.rs` fdata block: remove `center_1d`, `norm_lp_1d`; add `center`, `norm_lp`
5. Update external callers: `examples/02_functional_operations/main.rs`, `tests/validate_against_r.rs`

**Internal callers:** `fpca_variants.rs` (4 hits calling `center_1d`) and `tolerance/equivalence.rs` (3 hits) — these become `pub(crate)` calls, no change needed.

**Gate:** `cargo build -p fdars-core --features linalg,parallel`

### Task 5: Cat 3 — Metric `_self`/`_cross` Families (dtw, fourier, hshift, kl, pca, basis_coef, deriv, soft_dtw)

**Scope:** All files in `src/metric/` (except hausdorff — already done), `src/metric/mod.rs`, `src/lib.rs`

**Actions (per family):**
1. Add `pub fn name_self(data, [params,] dim: Dim)` and `pub fn name_cross(data1, data2, [params,] dim: Dim)` dispatchers
2. Rename `pub fn name_self_1d`/`name_cross_1d` → `pub(crate) fn name_self_impl`/`name_cross_impl`
3. Update `src/metric/mod.rs`: swap all `_1d` names for bare names
4. Update `src/lib.rs` metric block: swap all `_1d` names; add `basis_coef_self`, `basis_coef_cross`, `deriv_self`, `deriv_cross`, `kl_self`, `kl_cross`, `pca_self`, `pca_cross` (currently not in lib.rs, only in metric/mod.rs)
5. Update external callers in examples, tests, benches

**Gate:** `cargo build -p fdars-core --features linalg,parallel`

### Task 6: Cat 2 — Basis Functions (select_basis_auto, fourier_fit, pspline_fit)

**Scope:** `src/basis/auto_select.rs`, `src/basis/fourier_fit.rs`, `src/basis/pspline.rs`, `src/basis/mod.rs`, `src/lib.rs`

**Actions (pending user decision on Dim vs. plain rename):**
- If Option A (plain rename): rename `fourier_fit_1d` → `fourier_fit`, `pspline_fit_1d` → `pspline_fit`, `select_basis_auto_1d` → `select_basis_auto` (no `Dim` param, just a visibility rename). No `pub(crate)` needed — just drop the `_1d` suffix from the public name.
- If Option B (Dim dispatch): add dispatchers, rename to `pub(crate) fn ..._impl`.
1. Update `src/basis/mod.rs`: swap `_1d` names for bare names
2. Update `src/lib.rs` basis block: swap `_1d` names for bare names (keep `fdata_to_basis_1d`, `basis_to_fdata_1d`)
3. Update external callers: `examples/04_basis_representation/main.rs`, `examples/14_complete_pipeline/main.rs`, `tests/validate_against_r.rs`

**Note:** `src/seasonal/peak.rs` calls `crate::basis::fourier_fit_1d` internally — must be updated.

**Gate:** `cargo build -p fdars-core --features linalg,parallel`

### Task 7: Cat 2 — Regression (fdata_to_pc_1d, fdata_to_pls_1d) [PENDING USER DECISION]

**Scope:** `src/regression.rs`, `src/lib.rs`, + 120+ internal callers in `src/`, + 6+ external callers

**Actions (pending §5.1/5.2 decision):**
- If plain rename (Option A): rename `fdata_to_pc_1d` → `fdata_to_pc` in definition + all `src/` callers + all external callers. This is the largest single migration.
- Update `src/lib.rs` regression block: swap name.
- Update all 120+ internal callers (mass sed/grep pass).
- Update external: `examples/08_regression`, `examples/14_complete_pipeline`, `examples/22_gmm_clustering`, `tests/validate_against_r.rs`, `tests/equivalence_phase49.rs`, `tests/alloc_audit_fpca.rs`, `benches/audit_hotpaths.rs`, `benches/regression_benchmarks.rs`.

**Gate:** `cargo build -p fdars-core --features linalg,parallel` + `cargo build --examples` + `cargo build --benches`

---

## Architecture Patterns

### Established Dispatch Pattern (Cat 2 template)

```rust
// Source: fdars-core/src/depth/fraiman_muniz.rs:54 (verified example)

// Step 1: rename body
pub(crate) fn band_impl(data_obj: &FdMatrix, data_ori: &FdMatrix) -> Vec<f64> {
    // ... verbatim body of old band_1d ...
}

// Step 2: public dispatcher
#[must_use = "expensive computation whose result should not be discarded"]
pub fn band(data_obj: &FdMatrix, data_ori: &FdMatrix, dim: Dim) -> Vec<f64> {
    match dim {
        Dim::One | Dim::Two => band_impl(data_obj, data_ori),
    }
}
```

### Cat 1 Pattern (dispatcher already exists, _1d just goes `pub(crate)`)

```rust
// BEFORE
pub fn fraiman_muniz_1d(data_obj: &FdMatrix, data_ori: &FdMatrix, scale: bool) -> Vec<f64> {
    // ... body ...
}
pub fn fraiman_muniz(data_obj: &FdMatrix, data_ori: &FdMatrix, scale: bool, dim: Dim) -> Vec<f64> {
    match dim {
        Dim::One | Dim::Two => fraiman_muniz_1d(data_obj, data_ori, scale),
    }
}

// AFTER
pub(crate) fn fraiman_muniz_impl(data_obj: &FdMatrix, data_ori: &FdMatrix, scale: bool) -> Vec<f64> {
    // ... verbatim body ...
}
pub fn fraiman_muniz(data_obj: &FdMatrix, data_ori: &FdMatrix, scale: bool, dim: Dim) -> Vec<f64> {
    match dim {
        Dim::One | Dim::Two => fraiman_muniz_impl(data_obj, data_ori, scale),
    }
}
```

### dead_code Warning Avoidance

`pub(crate)` functions in Rust are NOT subject to `dead_code` warnings as long as they are called within the crate. Since every `name_impl` function is called by its dispatcher, and every Cat 1/2 `_impl` is called by internal code (via `dispatch.rs` for depth functions, via the dispatcher for others), no `dead_code` warnings will appear. [ASSUMED — based on Rust's dead_code lint rules; the rule is that `pub(crate)` functions visible to internal callers are not flagged unless unused within the crate]

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| New dispatch mechanism | trait objects, enum-data patterns | plain `match dim { Dim::One | Dim::Two => impl_fn(...) }` | Consistent with all 5 existing codebase precedents; `Dim` is data-free by design |
| Mass call-site rename | manual per-file edits | `sed -i` or `cargo-fix` patterns; compiler errors as a guide | 120+ callers of `fdata_to_pc_1d` — a compile-then-fix workflow is reliable |
| Data-carrying `Dim` | `Dim::Two(m1, m2)` | `Option<&[f64]>` superset signature | `Dim` is `#[non_exhaustive]` data-free; adding data is a breaking `Dim` change |

---

## Common Pitfalls

### Pitfall 1: `dispatch.rs` imports all the `_1d` depth functions directly
**What goes wrong:** After renaming `fraiman_muniz_1d` → `pub(crate) fraiman_muniz_impl`, the import in `dispatch.rs` (`use crate::depth::fraiman_muniz_1d`) fails to compile.
**How to avoid:** The downgrade to `pub(crate)` alone is fine — `dispatch.rs` is inside the crate and can access `pub(crate)`. But if the function is ALSO renamed (to `_impl`), the import path in `dispatch.rs` must be updated. Resolve: either keep the old `_1d` name as `pub(crate)` (no rename), or update `dispatch.rs` imports. The planner must pick one and be consistent.
**Recommendation:** For functions called by `dispatch.rs`, keep the internal name as `pub(crate) fn name_1d` (no rename to `_impl`) to minimize diff size in `dispatch.rs`.

### Pitfall 2: `equivalence_phase50.rs` breaks on Cat 1 downgrade
**What goes wrong:** `tests/equivalence_phase50.rs` calls `fraiman_muniz_1d`, `modal_1d`, `mean_1d` directly as public API. Once these become `pub(crate)`, the test (external crate) gets a compile error.
**How to avoid:** Update `equivalence_phase50.rs` as part of Task 1/2: replace `fraiman_muniz_1d(data, data, scale)` with `fraiman_muniz(data, data, scale, Dim::One)`. The bit-identity goldens still hold since the dispatcher forwards to the same impl.

### Pitfall 3: `seasonal/peak.rs` internal caller of `fourier_fit_1d`
**What goes wrong:** `src/seasonal/peak.rs:21` calls `crate::basis::fourier_fit_1d`. After renaming to `fourier_fit` or `fourier_fit_impl`, this internal call breaks.
**How to avoid:** Update `src/seasonal/peak.rs` as part of Task 6.

### Pitfall 4: `lib.rs` doc example uses old names
**What goes wrong:** The crate `lib.rs` module doc at line 26 uses `fdata_to_pc_1d` in a `use` example. Doctests compile this as external code.
**How to avoid:** Update the doc example in `lib.rs` header (line 26 and 34) when renaming `fdata_to_pc_1d`.

### Pitfall 5: Benchmark group name strings are not compile errors
**What goes wrong:** Bench files like `depth_benchmarks.rs` have group name strings like `"fraiman_muniz_1d"`. These are string literals — they won't cause a compile error but will name benchmarks with stale `_1d` names.
**How to avoid:** Update bench group name strings alongside the function call updates. Use grep to find all group name strings referencing `_1d`.

### Pitfall 6: `basis_coef_self_1d` and related functions not currently re-exported from `lib.rs`
**What goes wrong:** `basis_coef_self_1d`, `basis_coef_cross_1d`, `deriv_self_1d`, `deriv_cross_1d`, `kl_self_1d`, `kl_cross_1d`, `pca_self_1d`, `pca_cross_1d` ARE in `metric/mod.rs` re-exports but are NOT currently individually listed in `src/lib.rs` — they are accessible via `fdars_core::metric::basis_coef_self_1d` but NOT as `fdars_core::basis_coef_self_1d`. When adding their dispatchers, add the new bare names to `lib.rs`.

---

## Validation Architecture

### Compile-time gates (primary correctness proof)

No behavioral change — byte-identical `_impl` bodies. The primary gate is compilation. The existing test suite proves numeric correctness.

```bash
# Per-task gate (run after each task):
cargo build -p fdars-core --features linalg,parallel

# Wave/phase gate:
cargo fmt --check
cargo clippy --all-targets --features linalg,parallel -- -D warnings
cargo test -p fdars-core --features linalg,parallel
cargo build --features serde
cargo build --examples
cargo build --benches
cargo test --doc
```

### Phase Verification Strategy

| Check | What It Proves |
|-------|----------------|
| `cargo build -p fdars-core` | All symbol renames consistent; no broken internal references |
| `cargo build --examples` | All 28 examples updated to new API surface |
| `cargo build --benches` | All benches updated |
| `cargo test --features linalg,parallel` | Byte-identical `_impl` bodies — no numeric drift |
| `cargo test --doc` | Doctests (including lib.rs doc examples) updated |
| `cargo clippy --all-targets -D warnings` | No `dead_code` on `pub(crate)` `_impl` functions |
| Code-review gate (human) | `_impl` bodies compared to removed `_1d` bodies — confirms byte-identical |
| `cargo build --features serde` | serde derives still compile after renames |

### Wave 0 Gaps

None — no new test files needed. All coverage comes from the existing suite. The compile-time gate is the primary proof; the existing 1,654+ tests prove numeric correctness.

---

## Environment Availability

Step 2.6: SKIPPED — this phase is purely code/config changes within an existing Rust crate with no new external dependencies.

---

## Security Domain

Not applicable — this is a rename/API-shape phase with no new input-handling, cryptography, authentication, or network paths.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `pub(crate)` functions called within the crate do not trigger `dead_code` clippy warnings | §Architecture Patterns | If wrong, clippy gate fails; fix: add `#[allow(dead_code)]` or ensure each `pub(crate)` fn is reachable from a call site |
| A2 | `fdata_to_pc_1d` has 120+ internal callers (approximate grep count) | §5.1 | If count is higher/lower, affects Task 7 effort estimate only; compile errors reveal all callers |
| A3 | No example outside examples/ uses `fosr_2d` or `predict_fosr_2d` | §4 Cat 4 | If wrong, those callers must be updated too; confirmed via grep |
| A4 | `basis_coef_self_1d`, `deriv_self_1d`, `kl_self_1d`, `pca_self_1d` and cross variants have zero external callers in examples/tests/benches | §4.2 | If wrong, external callers need updating; confirmed via grep returning empty |
| A5 | `simpsons_weights_2d` has zero external callers in examples/tests/benches | §Cat 4 | If wrong, callers would need updating; confirmed via grep returning empty |

---

## Project Constraints (from CLAUDE.md)

| Directive | Constraint |
|-----------|------------|
| Gate set | `cargo fmt --check`, `cargo clippy --all-targets --features linalg,parallel -- -D warnings`, `cargo test --features linalg,parallel`, `cargo build --features serde`, `cargo build --examples`, `cargo build --benches`, `cargo test --doc` |
| Scope | `fdars-core` only — no changes to `fdars-r` or external packages |
| Behavior | API-shape-only — `_impl` bodies byte-identical, code-review gate confirms zero numeric drift |
| Breaking | v0.42.0 breaking pass is intentional — old `_1d`/`_2d` public names are removed |
| Pattern | v0.41.0 Dim-dispatch pattern from `src/dim.rs`; no data-carrying `Dim` variants |
| MSRV | Rust 1.81.0 minimum; `linalg` feature requires 1.84.0 |

---

## Sources

### Primary (HIGH confidence)
- [VERIFIED: fdars-core/src/lib.rs:632-687] — complete re-export inventory for metric, depth, fdata, basis, regression blocks
- [VERIFIED: fdars-core/src/prelude.rs:40-43] — prelude depth re-exports
- [VERIFIED: fdars-core/src/depth/mod.rs:30-43] — depth module re-exports
- [VERIFIED: fdars-core/src/metric/mod.rs:138-154] — metric module re-exports
- [VERIFIED: fdars-core/src/basis/mod.rs:27-44] — basis module re-exports
- [VERIFIED: fdars-core/src/dim.rs:19-26] — `enum Dim` definition
- [VERIFIED: fdars-core/src/depth/dispatch.rs:13-18] — dispatch.rs internal import list
- [VERIFIED: fdars-core/src/function_on_scalar.rs:275, function_on_scalar_2d.rs:468] — fosr naming collision
- [VERIFIED: fdars-core/src/matrix.rs:495] — `FdCurveSet::from_1d` method
- [VERIFIED: fdars-core/src/basis/projection.rs:160-241] — `fdata_to_basis_1d`/`basis_to_fdata_1d` are i32 shims
- [VERIFIED: fdars-core/tests/equivalence_phase50.rs:131-167] — tests that call `_1d` functions directly
- [VERIFIED: grep this session] — all external caller enumeration (examples/, tests/, benches/)
- [VERIFIED: grep this session] — all internal caller counts for key functions

## Metadata

**Confidence breakdown:**
- Function enumeration: HIGH — exhaustive grep from source
- Category assignments: HIGH — read each file's source
- Re-export inventory: HIGH — read lib.rs, prelude.rs, mod.rs files directly
- External caller counts: HIGH — grep confirmed
- Grey area analysis: HIGH — reasoning from verified source facts

**Research date:** 2026-09-08
**Valid until:** Until any enumerated source file changes (stable codebase, no time-sensitivity)
