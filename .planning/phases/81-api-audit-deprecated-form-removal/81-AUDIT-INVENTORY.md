# 81 — Public-API Breaking-Change Audit Inventory

**Phase:** 81 (API Audit & Deprecated-Form Removal) · **Requirement:** AUDIT-01
**Produced:** 2026-09-07 · **Crate:** `fdars-core` (workspace `/home/simonm/projects/rust/fdars/fdars-core`)
**Milestone:** v0.41.0 — 1.0 API stabilization pass (first breaking milestone; still 0.x)

This inventory ranks candidate breaking changes across the four audit scopes. Every location is grounded against real source (grep/read verified). Each entry has a stable `AUD-NN` ID that Phases 82/83 cite when drawing their concrete change sets. **Analysis only — no `fdars-core/src/` code is modified by this plan.**

Aggressiveness policy (from `81-CONTEXT.md`): naming = **Balanced** (RECOMMENDED vs OPTIONAL), pub-sealing = **Aggressive**, `#[non_exhaustive]` = **Full**.

---

## Scope Summary

| Scope | Requirement | Entries | Recommended / Seal-now | Optional / Review | Notes |
|-------|-------------|---------|------------------------|-------------------|-------|
| A — deprecated-form removal | API-01 | 6 (`AUD-01`–`AUD-06`) | 6 (all — pre-specified, executed by plan 81-02) | 0 | No approval needed to proceed |
| B — accidental `pub` exposure | API-02 | 3 (`AUD-07`–`AUD-09`) | 2 seal-now | 1 review (`wire`) | Re-export discipline strong; no dependency leaks found |
| C — `#[non_exhaustive]` gaps | API-03 | 4 grouped (`AUD-10`–`AUD-13`) | `AUD-10` (10 enums) + `AUD-11` (2 result structs) | `AUD-12` (config structs, caution) + `AUD-13` (`wire` layer structs) | Crate already ~330 `#[non_exhaustive]`; result structs already ~fully covered |
| D — naming inconsistencies | API-04 | 10 (`AUD-14`–`AUD-23`) | 4 RECOMMENDED | 6 OPTIONAL (→ STAB-03 if declined) | `Dim` dispatcher pattern already established for mean/depth |

**Totals:** 23 stable entry IDs (several bundle related items). Scope A is informational (already executing). The live approval decision is over Scopes B, C, D (`AUD-07`–`AUD-23`).

Baseline facts (grounded): crate has **66 public enums**, **~330 `#[non_exhaustive]` occurrences**, **78 functions** carrying `_1d`/`_2d`/`_nd` suffixes. Only **2 public `...Result` structs** and **10 public enums** currently lack `#[non_exhaustive]`.

---

## Ranked Inventory

### Scope A — Deprecated-form removal (→ API-01)

All 6 forms carry `#[deprecated(since = "0.30.0", …)]` and forward to a `Dim`/`_seeded` replacement. **Already fully specified; plan 81-02 executes the hard removal + caller migration. Listed here for completeness — no approval required.** Removal touches each definition site, the crate-root re-export block (`lib.rs`), the prelude re-export block (where present), plus the internal callers/tests below.

| ID | Location (def) | Remove | Replacement | Blast radius (callers) | Value / Risk |
|----|----------------|--------|-------------|------------------------|--------------|
| `AUD-01` | `src/fdata.rs:206` `mean_2d` | fn + `lib.rs:684` re-export | `mean(…, Dim::Two)` | src: 1 unit test (`fdata.rs:1103`); tests: `validate_against_r.rs:2517`, `equivalence_phase50.rs:242,244`; examples: 0; doctests: 0 | High / Low |
| `AUD-02` | `src/function_on_scalar.rs:917` `fanova` | fn + `lib.rs:396` re-export (`#[allow(deprecated)]`) | `fanova_seeded(…, 42)` | src callers: `inference/anova.rs:244,266`, `inference/permutation.rs:402`, `function_on_scalar.rs` unit tests (1084,1118,1151,1163,1166); tests: `equivalence_phase50.rs:71,73`; examples: **0 code calls** (example 21 already uses `fanova_seeded`; only a doc-comment mentions `fanova`); doctests: 0 | High / Med (LCG-seed behavior must be preserved by callers migrating to seed=42) |
| `AUD-03` | `src/depth/random_tukey.rs:61` `random_tukey_2d` | fn + `lib.rs:660` + `depth/mod.rs:48` re-exports | `random_tukey(…, Dim::Two)` | src: 0 non-test callers; tests: `equivalence_phase50.rs:263,265`; examples: 0; doctests: 0 | High / Low |
| `AUD-04` | `src/depth/random_projection.rs:87` `random_projection_2d` | fn + `lib.rs:659` + `depth/mod.rs:45` re-exports | `random_projection(…, Dim::Two)` | src: `depth/tests.rs:358`; tests: `equivalence_phase50.rs:252,254`; examples: 0; doctests: 0 | High / Low |
| `AUD-05` | `src/depth/fraiman_muniz.rs:66` `fraiman_muniz_2d` | fn + `lib.rs:654` + `prelude.rs:44` + `depth/mod.rs:37` re-exports | `fraiman_muniz(…, Dim::Two)` | src: `depth/tests.rs:327`; tests: `validate_against_r.rs:2285`, `equivalence_phase50.rs:228,233`; examples: 0; doctests: 0 | High / Low |
| `AUD-06` | `src/depth/modal.rs:67` `modal_2d` | fn + `lib.rs:657` + `prelude.rs:45` + `depth/mod.rs:42` re-exports | `modal(…, Dim::Two)` | src: `depth/tests.rs:338`; tests: `equivalence_phase50.rs:217,220`; examples: 0; doctests: 0 | High / Low |

Note: several removals also delete the now-unnecessary `#[allow(deprecated)]` guards on the `lib.rs`/`prelude.rs` re-export blocks (`lib.rs:395,651,681`; `prelude.rs:42`).

---

### Scope B — Accidental `pub` exposure (→ API-02) · Aggressive

Finding: the crate's re-export discipline is **strong** — no dependency-type leaks (no `pub use nalgebra/faer/rayon/statrs/…`), and `linalg`, `distributions`, `permutation_test`, `test_helpers` are already `pub(crate)`. Only three sealing candidates surfaced.

| ID | Location | Item | Re-exported? | Proposed change | Blast radius | Value / Risk | Recommendation |
|----|----------|------|--------------|-----------------|--------------|--------------|----------------|
| `AUD-07` | `src/helpers.rs:10` | `pub fn sort_nan_safe(&mut [f64])` | No | `pub` → `pub(crate)` | Internal-only: used across outliers/seasonal/classification/scalar_on_function/tolerance/spm/detrend/conformal; **0 external re-exports, 0 test/example/doctest uses of the public path**. Sealing is source-invisible to users. | High / Low | **SEAL NOW** |
| `AUD-08` | `src/smoothing.rs:336` | `pub fn solve_gaussian_pub(&mut [f64], &mut [f64], usize)` | No | `pub` → `pub(crate)` (and consider dropping the `_pub` suffix) | Internal cross-module solver wrapper (concurrent_regression, scalar_on_function/cv); not re-exported; 0 external uses. The `_pub` suffix itself signals an accidental widening. | High / Low | **SEAL NOW** |
| `AUD-09` | `src/wire.rs` (`pub mod wire`, 24 pub types incl. `FdaData`, `Layer`, `LayerKey`, `*Layer`) | Whole `pub mod wire` | No (zero crate-root/prelude re-exports) | **REVIEW:** either `pub mod` → `pub(crate) mod`, OR keep public as a deliberate interchange surface | `FdaData`/`wire::` is referenced **only inside `wire.rs`** (grep: 0 uses in any other module, test, example, or the re-export blocks). BUT `wire.rs` is documented as the pipeline/JS interchange container with a `use fdars_core::wire::*` doctest — so it may be a deliberate (if currently unwired) public seam for the JS/R bindings. **User judgment call at the gate.** | Med / Med | **REVIEW** (seal aggressively unless kept as a binding seam) |

---

### Scope C — `#[non_exhaustive]` gaps (→ API-03) · Full

The crate is already heavily covered (~330 occurrences). Gaps concentrate in **enums** and **user-constructed config structs**; result structs are essentially already sealed. Config structs have **all-`pub` fields** and are built with `Config { .. }` literals — adding `#[non_exhaustive]` **blocks cross-crate struct-literal construction**, so it is user-hostile without a builder/`Default` migration. Hence configs are flagged OPTIONAL/caution, enums + result structs RECOMMENDED.

| ID | Kind | Proposed change | Members (grounded file:line) | Blast radius | Value / Risk | Recommendation |
|----|------|-----------------|------------------------------|--------------|--------------|----------------|
| `AUD-10` | Public enums (10) missing `#[non_exhaustive]` | Add `#[non_exhaustive]` to each | `PeerPenalty` `src/peer.rs:68`; `LambdaChoice` `src/peer.rs:99`; `LambdaMethod` `src/peer.rs:117`; `DesignCriterion` `src/optimal_design.rs:92`; `OptimalityKind` `src/optimal_design.rs:103`; `ExtrapolationPolicy` `src/helpers.rs:884`; `ImputationMethod` `src/helpers.rs:1015`; `SelectionCriterion` `src/scalar_on_function/mod.rs:268`; `BasisType` `src/smooth_basis.rs:22`; `BasisCriterion` `src/smooth_basis.rs:1117` | Cross-crate `match` on these must add a `_ =>` arm (breaking for exhaustive matchers). In-crate matches are updated in-phase. All are method/criterion enums that plausibly grow variants. | High / Low | **RECOMMENDED (all 10)** |
| `AUD-11` | Public result structs (2) missing `#[non_exhaustive]` | Add `#[non_exhaustive]` | `OptimBandwidthResult` `src/smoothing.rs:544`; `KnnCvResult` `src/smoothing.rs:767` | Users only read these (never construct) → sealing is a pure, non-breaking-forward gain. 0 construction sites outside the crate. | High / Low | **RECOMMENDED (both)** |
| `AUD-12` | Public config structs (~22, all-`pub` fields) missing `#[non_exhaustive]` | Add `#[non_exhaustive]` **only if** paired with a builder/`Default`+`..default()` construction path | Confirmed-missing incl.: SPM — `SpmConfig` `spm/phase.rs:29`, `CusumConfig` `spm/cusum.rs:29`, `EwmaConfig` `spm/ewma.rs:46`, `MewmaConfig` `spm/mewma.rs:39`, `ProfileMonitorConfig` `spm/profile.rs:61`, `ArlConfig` `spm/arl.rs:43`, `FrccConfig` `spm/frcc.rs:57`, `MfpcaConfig` `spm/mfpca.rs:48`; Alignment — `ShapeCiConfig` `alignment/shape_ci.rs:17`, `LambdaCvConfig` `alignment/lambda_cv.rs:13`, `KMedoidsConfig` `alignment/clustering.rs:38`, `BayesianAlignConfig` `alignment/bayesian.rs:20`, `TransferAlignConfig` `alignment/transfer.rs:16`; Regression/other — `ElasticPcrConfig` + `ScalarOnShapeConfig` `elastic_regression/mod.rs:63,90`, `BoostingConfig` `boosting_regression/mod.rs:44`, `OptDesConfig` `optimal_design.rs:206`, `PeerConfig` `peer.rs:129`; Outliers — `TvdMssConfig` `outliers.rs:467`, `MuodConfig` `outliers.rs:564`, `SeqTransformConfig` `outliers.rs:734`, `DepthgramConfig` `outliers.rs:870`. (Already covered, NOT gaps: `PaceFpcaConfig`, `FofReConfig`, `DenseFlmmConfig`, `FastFmmConfig`, `MultiFammConfig`, `GmmClusterConfig`, `WcrConfig`.) | **BREAKING for any external `Config { … }` literal.** Requires a construction escape hatch or it locks users out. High churn, user-hostile if done naively. | Med / High | **OPTIONAL (caution)** — recommend only bundled with a `Default`/builder; else defer to STAB-03 |
| `AUD-13` | `wire` module `*Layer` structs (~18, all-`pub` fields) | Add `#[non_exhaustive]` | `FdaData` `wire.rs:49` + all `*Layer` (`FpcaLayer` 157, `PlsLayer` 174, … `CustomLayer` 467) | Only relevant if `AUD-09` keeps `wire` public. Same struct-literal-blocking concern as `AUD-12`. | Low / Med | **OPTIONAL** — resolve jointly with `AUD-09` (moot if `wire` is sealed) |

---

### Scope D — Naming inconsistencies (→ API-04) · Balanced

`Dim`-dispatcher unification is already established (`mean`, `modal`, `fraiman_muniz`, `random_tukey`, `random_projection`). RECOMMENDED = high-value collapses / correctness fixes; OPTIONAL = low-value suffix noise deferrable to the STAB-03 1.0 gap checklist (Phase 84).

**Signature caveat (grounded):** the `_2d` siblings of `deriv`/`geometric_median`/`lp`/`hausdorff`/`functional_spatial` do **not** share the `_1d` signature — the 2D forms take separate `argvals_s`/`argvals_t` grids (and `deriv_2d` returns `Option<Deriv2DResult>`, not `FdMatrix`). So a `Dim` collapse here is a *signature-carrying* dispatcher (heavier than the depth collapses), which tempers value/raises effort. Ratings reflect this.

| ID | Family / item (file:line) | Proposed change | Blast radius | Value / Risk | Recommendation |
|----|---------------------------|-----------------|--------------|--------------|----------------|
| `AUD-14` | `funhddC_cluster` `src/gmm/subspace.rs:554` | Rename → `fun_hddc_cluster` (snake_case; deprecate old) | Re-export `lib.rs:481`; camelCase-in-snake_case is an outright Rust-convention violation. Callers: grep in-crate + examples. | High / Low | **RECOMMENDED** |
| `AUD-15` | `FosrResult2d` `src/function_on_scalar_2d.rs:69` vs `FosrResult` `src/function_on_scalar.rs:29` | Rename `FosrResult2d` → `Fosr2dResult` (adjective-before-noun, matches `Grid2d`/`Deriv2DResult` house style) | Re-export `lib.rs:399`; type-only rename, users updating imports. Low count. | Med / Low | **RECOMMENDED** |
| `AUD-16` | `GmmResult` `src/gmm/mod.rs:39` vs `GmmClusterResult` `src/gmm/mod.rs:69` | Disambiguate: `GmmResult` → `GmmFitResult` (single-K fit) leaving `GmmClusterResult` for K-selection | Re-export `lib.rs:482`; two distinct public types today, easily confused. | Med / Low | **RECOMMENDED** |
| `AUD-17` | `deriv_1d` `src/fdata.rs:874` / `deriv_2d` `src/fdata.rs:956` | Collapse → `deriv(…, Dim)` (signature-carrying) | src+tests+examples refs: `deriv_1d` 22, `deriv_2d` 7. Core, widely used → high value, but 2D signature differs (returns `Option<Deriv2DResult>`), so non-trivial dispatcher. | High / Med | **RECOMMENDED** |
| `AUD-18` | `lp_self_1d`/`lp_cross_1d`/`lp_self_2d`/`lp_cross_2d` `src/metric/lp.rs:40,94,126,160` | Collapse self/cross pairs → `lp_self(…, Dim)` / `lp_cross(…, Dim)` | refs: `lp_self_1d` 19, `lp_self_2d` 6 (+cross). Distance-matrix backbone. 2D takes `argvals_s/_t`. | Med / Med | **RECOMMENDED** |
| `AUD-19` | `geometric_median_1d`/`_2d` `src/fdata.rs:1022,1047` | Collapse → `geometric_median(…, Dim)` | refs: `geometric_median_1d` 10. Less common; 2D differs in signature. | Med / Med | **OPTIONAL** (→ STAB-03 if declined) |
| `AUD-20` | `hausdorff_self_1d`/`cross_1d`/`self_2d`/`cross_2d` + `hausdorff_3d` `src/metric/hausdorff.rs:46,63,81,137,148` | Collapse self/cross → `Dim` dispatchers; leave `hausdorff_3d` (point-cloud, distinct) | refs: `hausdorff_self_1d` 13. 2D has surface-extraction logic → messiest collapse. | Low / Med | **OPTIONAL** (→ STAB-03) |
| `AUD-21` | `functional_spatial_1d`/`_2d` + `kernel_functional_spatial_1d`/`_2d` `src/depth/spatial.rs:18,77,199,219` | Collapse → `functional_spatial(…, Dim)` / `kernel_functional_spatial(…, Dim)` | refs: `functional_spatial_1d` 17. Depth measures; clean-ish 1D/2D split. | Low / Med | **OPTIONAL** (→ STAB-03) |
| `AUD-22` | Lone `_1d`/`_2d` suffix noise with **no** sibling (~15+ fns): `center_1d` `fdata.rs:234`, `norm_lp_1d` `fdata.rs:771`, `fourier_fit_1d` `basis/fourier_fit.rs:42`, `pspline_fit_1d` `basis/pspline.rs:66`, `select_basis_auto_1d` `basis/auto_select.rs:274`, `fdata_to_basis_1d`/`basis_to_fdata_1d` `basis/projection.rs:160,226`, `fdata_to_pc_1d`/`fdata_to_pls_1d` `regression.rs:387,714`, depth lone-`_1d` (`band_1d`, `extremal_depth_1d`, `total_variation_depth_1d`, `linfinity_depth_1d`, `hypograph_index_1d`, …), metric 1D-only self/cross (`dtw_*_1d`, `soft_dtw_*_1d`, `kl_*_1d`, `fourier_*_1d`, `hshift_*_1d`, `pca_*_1d`, `basis_coef_*_1d`, `deriv_*_1d`), helper `simpsons_weights_2d` `helpers.rs:173` | Drop the decorative suffix (e.g. `center_1d` → `center`); no `Dim` needed (no sibling) | Very large aggregate call-site churn across the crate + all 28 examples + doctests, for cosmetic gain. Each is individually trivial but the batch is the biggest, highest-risk edit in the milestone. | Low / High | **OPTIONAL** — recommend deferring the bulk to STAB-03; pick at most a few high-traffic ones (`center_1d`, `norm_lp_1d`) if the user wants |
| `AUD-23` | `LpeerResult` `src/peer.rs:190` vs `PeerResult` `src/peer.rs:148` | Optional: `LpeerResult` → `LocalPeerResult` for symmetry | Re-export `lib.rs:618`; purely cosmetic clarity. | Low / Low | **OPTIONAL** (→ STAB-03) |

**Explicitly NOT flagged (correct as-is):** `_nd` alignment functions (`srsf_transform_nd`, `elastic_align_pair_nd`, `karcher_mean_nd`, `pca_nd`, …) — `_nd` correctly means R^d, not a 1D/2D bifurcation. `fosr_2d`/`predict_fosr_2d` — module is legitimately 2D-grid scoped. `_seeded` suffix (`fanova_seeded`, `random_tukey_1d_seeded`, …) — consistent, semantically distinct, keep. `FdCurveSet::from_1d` — describes input shape, appropriate.

---

## How Phases 82/83 Consume This

- **Phase 82 (API-02 + API-03)** draws its concrete change set from the **APPROVED** entries in Scope B (`AUD-07`–`AUD-09`) and Scope C (`AUD-10`–`AUD-13`). It cites entries by `AUD-NN` ID. Config-struct sealing (`AUD-12`/`AUD-13`) proceeds only if the user approves *and* accepts the `Default`/builder construction path (or the entry is reduced/deferred).
- **Phase 83 (API-04)** draws its rename/dispatcher set from the **APPROVED** Scope D entries. **RECOMMENDED** entries (`AUD-14`–`AUD-18`) are the default Phase-83 scope; **OPTIONAL** entries the user declines route to the **STAB-03 1.0 gap checklist (Phase 84)**, not forced into Phase 83.
- **Scope A (`AUD-01`–`AUD-06`)** is already executed by plan **81-02** and needs no approval.
- IDs are stable and never renumbered — downstream plans reference them directly.

---

## Approval

**Approved:** 2026-09-07 (user, at the Phase 81 human-verify checkpoint).

**APPROVED — proceed in Phases 82/83:**
- Scope A (`AUD-01`–`AUD-06`) — deprecated-form removal, executed by plan 81-02 (informational; no approval was required).
- Scope B (Phase 82 / API-02): `AUD-07` (seal `sort_nan_safe`), `AUD-08` (seal `solve_gaussian_pub`).
- Scope C (Phase 82 / API-03): `AUD-10` (add `#[non_exhaustive]` to the 10 enums), `AUD-11` (add `#[non_exhaustive]` to `OptimBandwidthResult` + `KnnCvResult`).
- Scope D (Phase 83 / API-04): `AUD-14` (`funhddC_cluster` → `fun_hddc_cluster`), `AUD-15` (`FosrResult2d` → `Fosr2dResult`), `AUD-16` (`GmmResult` → `GmmFitResult`), `AUD-17` (`deriv_1d`/`_2d` → `deriv(…, Dim)`), `AUD-18` (`lp_self`/`lp_cross` → `Dim` dispatchers).

**DEFERRED — route to STAB-03 1.0 gap checklist (Phase 84), NOT forced into Phases 82/83:**
- `AUD-09` + `AUD-13` — **`wire` module kept public** as a deliberate (currently-unwired) JS/R interchange seam; revisit/wire-up-or-seal before the 1.0 cut. (`AUD-13` non_exhaustive on `wire` structs is moot while `wire` stays public.)
- `AUD-12` — config-struct `#[non_exhaustive]` deferred: breaks external `Config {..}` literals without a builder/`Default` path; the "seal configs + add construction escape hatch" work is scoped to STAB-03.
- `AUD-19` (`geometric_median`), `AUD-20` (`hausdorff`), `AUD-21` (`functional_spatial`), `AUD-22` (lone-`_1d`/`_2d` suffix batch — the largest/highest-risk edit), `AUD-23` (`LpeerResult` → `LocalPeerResult`) — optional naming, deferred to STAB-03.

**User notes:** Accepted orchestrator recommendations across all four decisions (Scope B/C core approved; `wire` kept public; configs deferred; naming = recommended-only). Reduced-scope approval exercised on naming (Phase 83 = `AUD-14`–`AUD-18` only).

---

## Raw Evidence (unranked)

The ranked inventory above supersedes this section; it is retained as the grounded scan trail (Task 1). Every hit was verified with ripgrep/read against the live source.

### Scope A (deprecated-form removal → API-01)
6 `#[deprecated(since = "0.30.0", …)]` forms, each forwarding to a `Dim`/`_seeded` replacement:
`mean_2d` `src/fdata.rs:206`→`mean(…,Dim::Two)`; `fanova` `src/function_on_scalar.rs:917`→`fanova_seeded(…,42)`; `random_tukey_2d` `src/depth/random_tukey.rs:61`→`random_tukey(…,Dim::Two)`; `random_projection_2d` `src/depth/random_projection.rs:87`→`random_projection(…,Dim::Two)`; `fraiman_muniz_2d` `src/depth/fraiman_muniz.rs:66`→`fraiman_muniz(…,Dim::Two)`; `modal_2d` `src/depth/modal.rs:67`→`modal(…,Dim::Two)`. Re-export sites: `lib.rs:654,657,659,660,684,396`; `prelude.rs:44,45`; `depth/mod.rs:37,42,45,48` (all under `#[allow(deprecated)]` guards).

### Scope B (accidental `pub` → API-02)
`sort_nan_safe` `src/helpers.rs:10` (pub, not re-exported, internal-only); `solve_gaussian_pub` `src/smoothing.rs:336` (pub, `_pub` suffix, internal cross-module); `pub mod wire` `src/wire.rs` (24 pub types, `FdaData`/`*Layer`, referenced only inside wire.rs, 0 re-exports, but documented interchange seam). No dependency-type leaks found; `linalg`/`distributions`/`permutation_test`/`test_helpers` already `pub(crate)`.

### Scope C (`#[non_exhaustive]` gaps → API-03)
Baseline ~330 `#[non_exhaustive]` present. Confirmed gaps — enums (10): `PeerPenalty` `peer.rs:68`, `LambdaChoice` `peer.rs:99`, `LambdaMethod` `peer.rs:117`, `DesignCriterion` `optimal_design.rs:92`, `OptimalityKind` `optimal_design.rs:103`, `ExtrapolationPolicy` `helpers.rs:884`, `ImputationMethod` `helpers.rs:1015`, `SelectionCriterion` `scalar_on_function/mod.rs:268`, `BasisType` `smooth_basis.rs:22`, `BasisCriterion` `smooth_basis.rs:1117`. Result structs (only 2): `OptimBandwidthResult` `smoothing.rs:544`, `KnnCvResult` `smoothing.rs:767`. Config structs (~22, all-`pub` fields — caution) listed under `AUD-12`. `wire` layer structs under `AUD-13`.

### Scope D (naming → API-04)
78 `_1d`/`_2d`/`_nd`-suffixed functions. camelCase violation `funhddC_cluster` `gmm/subspace.rs:554`. Result naming clashes `FosrResult`/`FosrResult2d` (`function_on_scalar.rs:29`,`function_on_scalar_2d.rs:69`), `GmmResult`/`GmmClusterResult` (`gmm/mod.rs:39,69`), `PeerResult`/`LpeerResult` (`peer.rs:148,190`). Signature-carrying `_1d`/`_2d` families (2D uses `argvals_s/_t`): `deriv` `fdata.rs:874,956`, `geometric_median` `fdata.rs:1022,1047`, `lp_*` `metric/lp.rs:40,94,126,160`, `hausdorff_*` `metric/hausdorff.rs`, `functional_spatial_*`/`kernel_functional_spatial_*` `depth/spatial.rs:18,77,199,219`. Lone-suffix noise (no sibling) enumerated under `AUD-22`. Correct-as-is: `_nd` alignment fns, `fosr_2d`, `_seeded` suffix, `FdCurveSet::from_1d`.
