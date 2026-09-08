---
phase: 88-large-suffix-batch
verified: 2026-09-08T21:05:46Z
status: passed
score: 3/3 must-haves verified
behavior_unverified: 0
overrides_applied: 0
re_verification: null
---

# Phase 88: Large Suffix Batch Verification Report

**Phase Goal:** The remaining crate-wide lone-`_1d`/`_2d` suffix sprawl is unified onto `Dim` dispatch and the entire example/doc surface is migrated to the new signatures.
**Verified:** 2026-09-08T21:05:46Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| #  | Truth | Status | Evidence |
|----|-------|--------|----------|
| 1  | Every consolidated lone-`_1d`/`_2d` family (NAME-04) is callable through a single Dim-dispatched (or plain-renamed) public signature routing to byte-identical bodies (SC-1) | ✓ VERIFIED | See artifact checks below |
| 2  | All 28 examples and all docs/doctests compile against the new surface (SC-2) | ✓ VERIFIED | Gates confirmed by executor; examples use `Dim::One` form; `cargo build --examples` + `cargo test --doc` (208 tests) green |
| 3  | The whole crate compiles with no lingering public references to removed lone-suffix names; the existing suite proves zero numeric drift (SC-3) | ✓ VERIFIED | Negative grep clean in examples/tests/benches; src/ internal `_1d` calls are all to `pub(crate)` bodies (valid); 2857 lib tests pass |

**Score:** 3/3 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `fdars-core/src/depth/band.rs` | `pub fn band(.., dim: Dim)` dispatcher; `band_1d` is `pub(crate)` | ✓ VERIFIED | line 32: `pub(crate) fn band_1d`; line 98: `pub fn band(…, dim: Dim) -> Vec<f64>` |
| `fdars-core/src/metric/dtw.rs` | `pub fn dtw_self/dtw_cross(.., dim: Dim)` dispatchers; `_1d` bodies `pub(crate)` | ✓ VERIFIED | lines 73/83: `pub(crate) fn dtw_{self,cross}_1d`; lines 97/106: `pub fn dtw_{self,cross}(…, dim: Dim)` |
| `fdars-core/src/fdata.rs` | `pub fn center(.., dim: Dim)` and `pub fn norm_lp(.., dim: Dim)` dispatchers | ✓ VERIFIED | lines 224/777: `pub(crate) fn center_1d/norm_lp_1d`; lines 260/830: `pub fn center/norm_lp(…, dim: Dim)` |
| `fdars-core/src/depth/fraiman_muniz.rs` | Cat 1: `fraiman_muniz_1d` is `pub(crate)`, dispatcher `fraiman_muniz` exists | ✓ VERIFIED | line 34: `pub(crate) fn fraiman_muniz_1d`; line 55: `pub fn fraiman_muniz(…, dim: Dim)` |
| `fdars-core/src/depth/modal.rs` | Cat 1: `modal_1d` is `pub(crate)`, dispatcher `modal` exists | ✓ VERIFIED | line 18: `pub(crate) fn modal_1d`; line 55: `pub fn modal(…, dim: Dim)` |
| `fdars-core/src/fdata.rs` (mean) | Cat 1: `mean_1d` is `pub(crate)`, dispatcher `mean` exists | ✓ VERIFIED | line 169: `pub(crate) fn mean_1d`; line 194: `pub fn mean(…, dim: Dim)` |
| `fdars-core/src/regression.rs` | Plain rename: `pub fn fdata_to_pc` and `pub fn fdata_to_pls` (no `_1d`) | ✓ VERIFIED | lines 387/714: `pub fn fdata_to_pc` / `pub fn fdata_to_pls` (bare names, no `Dim` param) |
| `fdars-core/src/basis/fourier_fit.rs` | Plain rename: `pub fn fourier_fit` (no `_1d`) | ✓ VERIFIED | line 42: `pub fn fourier_fit` |
| `fdars-core/src/basis/pspline.rs` | Plain rename: `pub fn pspline_fit` (no `_1d`) | ✓ VERIFIED | line 66: `pub fn pspline_fit` |
| `fdars-core/src/basis/auto_select.rs` | Plain rename: `pub fn select_basis_auto` (no `_1d`) | ✓ VERIFIED | line 274: `pub fn select_basis_auto` |
| `fdars-core/src/metric/pca.rs` | `pca_cross(data1, data2, ncomp, dim: Dim)` — NO `argvals` param (executor correction of RESEARCH error) | ✓ VERIFIED | `pub(crate) fn pca_cross_1d(data1, data2, ncomp)` — body has no `argvals`; dispatcher matches: `pub fn pca_cross(data1, data2, ncomp, dim: Dim)` |
| `fdars-core/src/depth/mod.rs` | Re-exports swapped: bare dispatcher names, `_1d` removed | ✓ VERIFIED | `pub use band::{band, modified_band, modified_epigraph_index}`; `pub use fraiman_muniz::fraiman_muniz`; `pub use rpd::{rpd_depth, rpd_depth_1d_seeded}`; no non-seeded `_1d` present |
| `fdars-core/src/metric/mod.rs` | Re-exports swapped to bare names; `_1d` behind `#[cfg(test)] pub(crate)` only | ✓ VERIFIED | Lines 138–156: all bare names (`basis_coef_cross`, `deriv_cross`, `dtw_cross`, etc.); `_1d` forms gated by `#[cfg(test)]` at lines 161–176 |
| `fdars-core/src/basis/mod.rs` | Bare names (`fourier_fit`, `pspline_fit`, `select_basis_auto`) + R-interop shims kept (`fdata_to_basis_1d`, `basis_to_fdata_1d`) | ✓ VERIFIED | lines 27/33/42: bare names; line 38: `basis_to_fdata_1d, fdata_to_basis_1d` retained |
| `fdars-core/src/lib.rs` | Re-exports swapped; new bare metric names added; no removed `_1d` in `pub use` blocks; regression block uses `fdata_to_pc`/`fdata_to_pls`; doctests updated | ✓ VERIFIED | metric block (630–638): all bare names; depth block (644–651): `band`, `fraiman_muniz`, `rpd_depth`, etc., no non-seeded `_1d`; fdata block (669–673): `center`, `norm_lp`; regression block (578): `fdata_to_pc`, `fdata_to_pls` |
| `fdars-core/src/prelude.rs` | Depth re-exports swapped to bare names | ✓ VERIFIED | line 41: `band, fraiman_muniz, …, modal, modified_band, random_projection, random_tukey` |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `pub fn band(…, dim: Dim)` in `depth/band.rs` | `pub(crate) fn band_1d(…)` | `match dim { Dim::One \| Dim::Two => band_1d(..) }` | ✓ WIRED | body byte-identical, visibility flip only |
| `pub fn dtw_self/dtw_cross(…, dim: Dim)` in `metric/dtw.rs` | `pub(crate) fn dtw_{self,cross}_1d(…)` | `match dim { … => dtw_*_1d(..) }` | ✓ WIRED | confirmed in source |
| `pub fn center/norm_lp(…, dim: Dim)` in `fdata.rs` | `pub(crate) fn center_1d/norm_lp_1d(…)` | `match dim { … => center_1d(..) }` | ✓ WIRED | confirmed in source |
| `pub fn pca_cross(data1, data2, ncomp, dim: Dim)` in `metric/pca.rs` | `pub(crate) fn pca_cross_1d(data1, data2, ncomp)` | `match dim { Dim::One \| Dim::Two => pca_cross_1d(data1, data2, ncomp) }` | ✓ WIRED | executor corrected RESEARCH signature error (no `argvals`); dispatcher matches body exactly |
| `pub fn fourier_fit(…)` in `basis/fourier_fit.rs` | `seasonal/peak.rs` internal caller | `crate::basis::fourier_fit(data, argvals, nbasis)` | ✓ WIRED | line 21 of `seasonal/peak.rs` confirmed |
| `pub fn fdata_to_pc(…)` in `regression.rs` | examples/benches external callers | direct call (plain rename) | ✓ WIRED | `examples/08_regression/main.rs` line 10/124; `benches/audit_hotpaths.rs` line 30 |
| Internal `src/` callers of `pub(crate)` `_1d` bodies | `pub(crate) fn X_1d` | crate-internal access (valid) | ✓ WIRED | `fpca_variants.rs` calls `fdata::center_1d`; `explain/helpers/kernel.rs` calls `depth::fraiman_muniz_1d`, `depth::modified_band_1d`; `alignment/phase_boxplot.rs` calls `depth::modified_band_1d` — all `pub(crate)`, valid |

### Behavioral Spot-Checks

Step 7b: SKIPPED (this is an API-shape-only rename/consolidation phase; bodies are byte-identical, no behavioral change. The executor-confirmed gate set — 2857 lib tests + 208 doctests + full integration suite — is the numeric non-regression proof. Re-running the full suite is out-of-scope per verification instructions.)

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|---------|
| NAME-04 | 88-01-PLAN.md | Consolidate large remaining batch of lone-`_1d`/`_2d` suffix functions onto `Dim` dispatch; update all 28 examples + docs | ✓ SATISFIED | Dispatchers verified in depth/fdata/metric; plain renames verified in regression/basis; re-exports swapped in lib.rs/prelude.rs/mod.rs files; external callers migrated in examples/tests/benches |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `metric/mod.rs` | 157–176 | `#[cfg(test)] pub(crate)` re-exports of `_1d` primitives for bit-identity tests | ℹ️ Info | Intentional and documented; not externally visible; no public surface leak |

No TBD/FIXME/XXX markers found. No stubs. No hardcoded empty returns in dispatch paths.

**Exclusions verified unchanged:**

| Symbol | Status |
|--------|--------|
| `fosr_2d`, `predict_fosr_2d` | ✓ Still public in `function_on_scalar_2d.rs` |
| `simpsons_weights_2d` | ✓ Still public in `helpers.rs` |
| `random_projection_1d_seeded`, `random_tukey_1d_seeded` | ✓ Still public; retained in `depth/mod.rs` and `lib.rs` |
| `rpd_depth_1d_seeded` | ✓ Still public in `depth/rpd.rs` and `depth/mod.rs` (not in `lib.rs` root — pre-existing state, unchanged by phase 88) |
| `FdCurveSet::from_1d` | ✓ Still public method in `matrix.rs` line 495 |
| `fdata_to_basis_1d`, `basis_to_fdata_1d` | ✓ R-interop shims still public in `basis/projection.rs`; re-exported from `basis/mod.rs` and `lib.rs` |

**Private internal helpers confirmed not in-scope:**

`log_gaussian_1d` in `coclustering.rs` (private `fn`, not public API), `penalty_matrix_2d` / `compute_beta_se_2d` in `function_on_scalar_2d.rs` (private `fn`), `uniform_grid_1d` in `function_on_scalar_2d.rs` (private test helper), `find_peaks_1d` in `seasonal/peak.rs` (private fn, not public surface) — none of these are in-scope for NAME-04 and all are correctly private.

### Human Verification Required

None. All truths are verifiable programmatically from source. The executor-confirmed gate set (fmt --check, clippy --all-targets -D warnings, 2857 tests, serde build, examples build, benches build, 208 doctests) provides complete numeric non-regression evidence for this API-shape-only change.

### Gaps Summary

No gaps. All three success criteria are met:

- **SC-1:** Every in-scope function has a Dim-dispatched public entry or bare plain-renamed public entry routing to a byte-identical `pub(crate)` body (Cat 1/2/3) or renamed definition (Cat 4).
- **SC-2:** All 28 examples and 208 doctests compile (executor gate + spot-checks of examples 06/08 confirming `Dim::One` and plain-rename forms).
- **SC-3:** The negative grep over examples/tests/benches/src shows zero non-excluded lone `_1d`/`_2d` public call sites. Internal `pub(crate)` `_1d` calls in `fpca_variants.rs`, `explain/helpers/kernel.rs`, and `alignment/phase_boxplot.rs` are valid crate-internal uses of correctly-demoted bodies, not API leaks.

---

_Verified: 2026-09-08T21:05:46Z_
_Verifier: Claude (gsd-verifier)_
