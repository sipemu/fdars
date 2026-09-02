# Survey: Python-beyond-scikit-fda Ecosystem (PYX-01)

**Milestone:** v0.31.0 Multi-Ecosystem Gap Audit — Phase 52
**Survey date:** 2026-09-02
**Ecosystem:** Python FDA / functional-time-series libraries OTHER than scikit-fda
**Packages surveyed (version-pinned as of 2026-09):**
- `FDApy@1.0.4` — functional data on multi/different-dimensional domains, irregular sampling, (multivariate) dimension reduction, simulation toolbox (Golovkine; JOSS 2025).
- `tslearn@0.9.0` (0.10 dev line) — DTW/soft-DTW, DTW barycenter averaging, shapelets, k-Shape, kernel k-means (GAK), metrics for variable-length series.
- `sktime@0.3x` — time-series ML framework; FDA-relevant components: shapelet transforms, distance-based clustering (wraps tslearn), interval/dictionary classifiers.
- Other: `pyts@0.13.x` (SAX/PAA/bag-of-patterns, GAF/MTF imaging), `tsfresh`/`catch22` (mass TS feature extraction).

**EXCLUSION:** `scikit-fda` is entirely EXCLUDED (covered by v0.14.0 `BACKLOG.md`). No scikit-fda capability appears below.
**Scope discipline:** capabilities must be FDA-relevant (functional/shape/curve methods), not generic time-series ML.
**De-dup baselines:** shipped fdars, `BACKLOG.md`, `R-BACKLOG.md`, `R-AUDIT-REPORT.md`.
**Audit-only fence:** zero `fdars-core/src/` edits.

---

## Capability Inventory & fdars Parity Mapping

| Capability | Reference (pkg@ver) | fdars status | Searched-for | Net-new? | Notes |
|---|---|---|---|---|---|
| (Multivariate) functional data on multi/different-dimensional domains; irregular sampling | FDApy@1.0.4 | partial | `grep multi_fdata.rs irreg_fdata/ function_on_scalar_2d.rs spm/mfpca.rs`: multivariate FD + irregular + 2D surfaces present, but MFPCA jointly across *heterogeneous* domains (curve + image) is not first-class | Yes (missing slice) | fdars `mfpca` takes multiple same-type variables; FDApy models jointly across different-dimensional domains. See gap list. |
| Dimension reduction / MFPCA (multivariate FPCA) | FDApy@1.0.4 | present | `grep spm/mfpca.rs`: `mfpca`, `MfpcaConfig`, `MfpcaResult` | No | Same-domain multivariate FPCA covered. |
| Simulation toolbox (basis-decomposition, clustered functional data) | FDApy@1.0.4 | present | `grep simulation.rs`: functional-data simulators present | No | Covered. |
| Smoothing methods | FDApy@1.0.4 | present | `grep smoothing.rs smooth_basis.rs fem_smoothing.rs` | No | Covered. |
| DTW / soft-DTW distance + soft-DTW divergence | tslearn@0.9.0 | present | `grep metric/soft_dtw.rs`: `soft_dtw_distance`, `soft_dtw_divergence` | No | Covered. |
| DTW barycenter averaging (DBA) / soft-DTW barycenter | tslearn@0.9.0 | present | `grep metric/soft_dtw.rs`: full barycenter (gradient descent, `update_barycenter`) | No | Covered. |
| Elastic/warping curve alignment | tslearn@0.9.0 | present | `grep alignment/ warping.rs elastic.rs` | No | fdars exceeds (SRVF). |
| **Shapelets / shapelet transform / shapelet-based classification** | tslearn@0.9.0, sktime, pyts@0.13.x | absent | `grep -ri "shapelet" src/` → **none**; backlogs+R-AUDIT = 0 | **Yes** | Interpretable local-shape primitives for classification. Net-new. |
| **k-Shape clustering (SBD — cross-correlation shape-based distance)** | tslearn@0.9.0, sktime | absent | `grep -riE "kshape\|shape.based.distance\|sbd" src/` → **none** (`alignment/shape.rs` is SRVF, not SBD); backlogs+R-AUDIT = 0 | **Yes** | FFT cross-correlation shape clustering, distinct from fdars' elastic/SRVF clustering. Net-new. |
| **Global Alignment Kernel (GAK) — PSD kernel for kernel k-means / SVM on curves** | tslearn@0.9.0 | absent | `grep -riE "global.alignment.kernel\|\bgak\b\|triangular.kernel" src/` → **none**; backlogs = 0 | **Yes (slice)** | fdars has soft-DTW *divergence* (a distance), not a PSD alignment *kernel* enabling kernel machines on curves. Net-new slice. |
| Kernel k-means on time series | tslearn@0.9.0 | partial | `grep clustering.rs gmm/ metric/`: k-means + kernels present, but not GAK-kernel k-means specifically | No | Blocked on GAK (above). |
| SAX / PAA / bag-of-patterns / GAF-MTF imaging (symbolic & image representations) | pyts@0.13.x, sktime | absent | `grep -riE "\bsax\b\|piecewise.aggregate\|bag.of.pattern\|paa" src/` → **none**; backlogs = 0 | Flagged | Symbolic/imaging TS-ML *representations*, not functional numeric methods. **Likely out-of-scope** — recorded for RPT-03 triage. |
| Mass TS feature extraction (tsfresh/catch22) | tsfresh, catch22 | partial | `grep utility.rs scoring.rs`: some functional features (local_averages tracked in `BACKLOG` PREP-08) | No | Generic TS-ML feature banks — out-of-scope; the FDA-relevant feature APIs are already backlogged (PREP-08). |

---

## Net-New Gap List (Python-beyond-scikit-fda)

| Capability | Reference (pkg@ver) | fdars status | Searched-for | Net-new? | Notes |
|---|---|---|---|---|---|
| **Shapelet transform / shapelet-based classification** | tslearn@0.9.0 / sktime / pyts@0.13.x | absent | `shapelet` → none in `src/`, backlogs, R-AUDIT | Yes | Value: interpretable local-shape features + high-accuracy shape classifiers for curves. Effort: M–L (shapelet discovery/learning + transform). Reference: tslearn `ShapeletModel`, sktime shapelet transform. |
| **k-Shape clustering (SBD)** | tslearn@0.9.0 / sktime | absent | `kshape/sbd/shape.based.distance` → none | Yes | Value: fast FFT cross-correlation shape clustering; a different inductive bias from SRVF/elastic clustering. Effort: M (SBD via `rustfft` cross-correlation + centroid refinement). Reference: tslearn `KShape`. |
| **Global Alignment Kernel (GAK) for kernel methods on curves** | tslearn@0.9.0 | absent | `global.alignment.kernel/gak` → none | Yes (slice) | Value: PSD kernel enabling kernel k-means / SVM directly on curves. Effort: S–M (GAK atop existing soft-DTW machinery; add PSD kernel + kernel-k-means glue). Reference: tslearn `gak`, `KernelKMeans`. |
| **Multi-dimensional heterogeneous-domain MFPCA** | FDApy@1.0.4 | partial | `spm/mfpca.rs` = same-type multivariate FPCA; joint curve+surface not first-class | Yes (slice) | Value: joint dimension reduction across mixed-domain components (e.g. a 1D curve + a 2D image per subject). Effort: M (generalize `mfpca` component handling across domain dimensionalities). Reference: FDApy `MFPCA`. |

**Net-new gap count: 4** (3 solid + 1 partial/missing-slice), plus 1 flagged likely-out-of-scope (SAX/symbolic representations).

Python's FDA-specific numeric surface (FDApy, tslearn's DTW/soft-DTW/barycenter) is largely covered by fdars. The net-new gaps concentrate in **time-series-ML shape methods** (shapelets, k-Shape, GAK) that sit at the FDA/ML boundary, plus a multi-domain MFPCA slice from FDApy.

---

## Reverse-Parity Note (where fdars LEADS these Python libraries)

- **Statistical depth & robustness**: depth measures, functional boxplots, outlier detection, streaming depth — absent from tslearn/sktime/FDApy.
- **Inference & regression breadth**: scalar-on/function-on-function/concurrent/Fréchet/boosting regression, permutation & ANOVA inference — far beyond FDApy's dimension-reduction focus and tslearn/sktime's ML focus.
- **Elastic/SRVF shape analysis**: full square-root framework, elastic FPCA/regression/changepoint — tslearn offers only DTW-family alignment.
- **Explainability**: PDP/SHAP/LIME/ALE for functional models — no analog.
- **Determinism & deployment**: reproducible seeded parallelism; compiled Rust core to WASM/JS + R bindings vs Python-runtime-bound libraries.
