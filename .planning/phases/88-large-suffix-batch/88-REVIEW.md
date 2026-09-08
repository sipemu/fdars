---
phase: 88-large-suffix-batch
reviewed: 2026-09-08T21:07:22Z
depth: deep
files_reviewed: 41
files_reviewed_list:
  - fdars-core/src/andrews.rs
  - fdars-core/src/autodiff.rs
  - fdars-core/src/basis/auto_select.rs
  - fdars-core/src/basis/fourier_fit.rs
  - fdars-core/src/basis/mod.rs
  - fdars-core/src/basis/pspline.rs
  - fdars-core/src/basis/tests.rs
  - fdars-core/src/boosting_regression/bayesian.rs
  - fdars-core/src/boosting_regression/boost_fofr.rs
  - fdars-core/src/boosting_regression/mod.rs
  - fdars-core/src/classification/cv.rs
  - fdars-core/src/classification/mod.rs
  - fdars-core/src/classification/tests.rs
  - fdars-core/src/clustering_advanced.rs
  - fdars-core/src/coclustering.rs
  - fdars-core/src/density_fda.rs
  - fdars-core/src/depth/band.rs
  - fdars-core/src/depth/erl.rs
  - fdars-core/src/depth/extremal.rs
  - fdars-core/src/depth/fraiman_muniz.rs
  - fdars-core/src/depth/half_region.rs
  - fdars-core/src/depth/hypo_epi.rs
  - fdars-core/src/depth/linf.rs
  - fdars-core/src/depth/mod.rs
  - fdars-core/src/depth/modal.rs
  - fdars-core/src/depth/random_projection.rs
  - fdars-core/src/depth/random_tukey.rs
  - fdars-core/src/depth/rpd.rs
  - fdars-core/src/depth/tvd.rs
  - fdars-core/src/fdata.rs
  - fdars-core/src/lib.rs
  - fdars-core/src/metric/basis_coef.rs
  - fdars-core/src/metric/deriv.rs
  - fdars-core/src/metric/dtw.rs
  - fdars-core/src/metric/fourier.rs
  - fdars-core/src/metric/hshift.rs
  - fdars-core/src/metric/kl.rs
  - fdars-core/src/metric/mod.rs
  - fdars-core/src/metric/pca.rs
  - fdars-core/src/metric/soft_dtw.rs
  - fdars-core/src/regression.rs
findings:
  critical: 0
  warning: 0
  info: 0
  total: 0
status: clean
---

# Phase 88: Code Review Report

**Reviewed:** 2026-09-08T21:07:22Z
**Depth:** deep
**Files Reviewed:** 41
**Status:** clean

## Summary

Phase 88 is a pure API-shape refactor with two categories of change, and no
numeric/behavioral modifications:

- **Cat 1/2/3 (Dim dispatch):** ~35 lone `_1d` functions across `depth/`,
  `fdata.rs`, and `metric/` had their visibility flipped from `pub fn X_1d` to
  `pub(crate) fn X_1d`, and new `pub fn X(.., dim: Dim)` dispatchers were added
  that unconditionally forward both `Dim::One | Dim::Two` arms to the `_1d`
  body.

- **Plain renames (no Dim):** Five genuinely-1D functions (`fdata_to_pc_1d`,
  `fdata_to_pls_1d`, `fourier_fit_1d`, `pspline_fit_1d`, `select_basis_auto_1d`)
  had the `_1d` suffix dropped at their definition and at all ~157 internal
  call-sites, doctests, and crate-root re-exports.

All five focus areas requested in the brief were examined:

### 1. Body identity

Spot-checked: `regression.rs::fdata_to_pc`, `basis/fourier_fit.rs::fourier_fit`,
`basis/pspline.rs::pspline_fit`, `fdata.rs::center_1d` / `norm_lp_1d`,
`metric/dtw.rs::dtw_self_1d` / `dtw_cross_1d`, `metric/pca.rs::pca_self_1d` /
`pca_cross_1d`, `depth/band.rs::band_1d`.

In every case only the symbol name or visibility keyword on the signature line
changed. The one cosmetic body delta allowed by the brief — updating the
`fourier_fit` error string from `"…in fourier_fit_1d;…"` to `"…in fourier_fit;…"`
— is the sole intra-body text change and has zero behavioral effect.

### 2. Dispatcher correctness

Every `pub fn X(.., dim: Dim)` dispatcher uses the pattern:
```rust
match dim {
    Dim::One | Dim::Two => x_1d(arg1, arg2, ..),
}
```
Args are forwarded in the identical order they appear in the `_1d` signature.
`#[must_use]` is preserved on dispatchers whose `_1d` originals carried it
(all depth functions, `fdata_to_pc`, `fdata_to_pls`, `rpd_depth`, `kl_self`,
`kl_cross`). Functions whose `_1d` had no `#[must_use]` (dtw, soft_dtw, fourier,
hshift, basis_coef) also have no `#[must_use]` on the dispatcher — consistent
with the originals.

### 3. `pca_cross` signature deviation

The pre-phase source confirms `pca_cross_1d(data1, data2, ncomp)` — no
`argvals` parameter. The new dispatcher is `pca_cross(data1, data2, ncomp, dim)`,
which correctly appends only the new `Dim` parameter. The executor's reported
"RESEARCH claimed argvals" discrepancy was a research artifact; the real
implementation never had an argvals argument, and the dispatcher correctly reflects
the actual signature.

### 4. Plain-rename completeness

`fdata_to_pc` / `fdata_to_pls`: searched all source files — no residual
`fdata_to_pc_1d` / `fdata_to_pls_1d` call-sites remain (remaining occurrences
are test function *names* like `fn test_fdata_to_pc_1d_basic()`, not call-sites).
Re-exports updated in `lib.rs`, `prelude.rs`, and all module `mod.rs` files.
No pre-existing `fdata_to_pc` name existed before this phase, so no collision.
Same pattern confirmed for `fourier_fit`, `pspline_fit`, `select_basis_auto`.

### 5. `#[cfg(test)]` re-export hygiene

`depth/mod.rs` gates `modal_1d`, `random_projection_1d`, `random_tukey_1d`,
`rpd_depth_1d` behind `#[cfg(test)] pub(crate) use …`. The unconditional
`pub(crate) use …` block covers the _1d primitives that are called in
non-test production code (dispatch.rs, outliers.rs, fdata.rs, explain/,
classification/dd.rs, alignment/phase_boxplot.rs) — those remain correctly
always-available within the crate.

`metric/mod.rs` test-only block covers `dtw_1d`, `fourier_1d`, `hshift_1d`,
`kl_1d`, `pca_1d`, `soft_dtw_{self,cross,div_self}_1d`. `soft_dtw_div_cross_1d`
was correctly removed from that block in the follow-up commit (014ae217) because
no bit-identity test referenced it — the dispatcher (the only caller) has private
intra-module access without needing the re-export.

### 6. No divergent Dim::Two paths

All 41 dispatchers collapse both arms to the same `_1d` body. No function
examined had a pre-existing divergent 2D implementation that was incorrectly
unified. The design is correct: the `Dim` parameter serves as a future extension
seam, not a current dispatch fork.

All reviewed files meet quality standards. No issues found.

---

_Reviewed: 2026-09-08T21:07:22Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
