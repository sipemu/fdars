# R-Ecosystem Gap Audit — fdars vs the R FDA ecosystem

**Milestone:** v0.18.0 R-Ecosystem Gap Audit (audit-only — zero `fdars-core/src/` edits)
**Status:** In progress — Phase 16 (R Ecosystem Inventory) complete; Phases 17–19 append their sections below.
**Yardstick:** the R functional-data-analysis package ecosystem — replaces scikit-fda, whose actionable backlog was exhausted across v0.15.0–v0.17.0.
**Distinct from:** the archived scikit-fda audit `.planning/research/AUDIT-REPORT.md` / `BACKLOG.md` (NOT modified by this milestone). The companion `.planning/research/R-BACKLOG.md` is produced in Phase 19.

---

## Phase 16 — R Ecosystem Inventory (INV-01, INV-02)

**Survey date:** 2026-08-14 · **Survey-month convention:** all CRAN versions cited as the latest release as of 2026-08.

### Methodology (Inventory)

**Sourcing (per milestone decision):** R capability data sourced from model knowledge cross-checked against CRAN package pages and rdrr.io reference indexes (fetched 2026-08-14); no local R install, no `packageVersion()`. Versions are the latest CRAN release as of the survey month; a `~` prefix flags an unverified/training-sourced version; CRAN-archived packages are excluded (see *Packages Considered and Excluded*).

**Capability-first collapse:** one row per distinct algorithm/capability; `fit`/`predict`/`transform` and S3/S4 method variants collapse to a single row; a capability offered by several packages co-lists them in the Source column — this avoids one-row-per-API-name inflation (Pitfall 1). Because some capabilities serve different roles in different areas (e.g. FPCA as a preprocessing step vs. sparse-FPCA method), the counts reflect logical per-area grouping rather than strict global deduplication; Phase 17 matches by capability semantics, not row count.

**Design-goal filter (stated once, applied throughout — this is the INV-02 rule):**
- **In-Scope Algorithm** — a numeric algorithm or statistical method portable to a numeric Rust library.
- **In-Scope API-Ergonomics** — a convenience / composable-object / cross-validation-utility layer a library user would expect.
- **Out-of-Scope (plotting)** — any visualization or `plot.*` renderer. Note: the *numeric statistic* underpinning a diagnostic plot (e.g. outliergram MO/MEI values, functional-boxplot fence/threshold computation) is **in-scope**; only the renderer is out-of-scope.
- **Out-of-Scope (IO)** — dataset loaders, file readers, or data-frame round-trips.

**Headline counts:** **35 packages** surveyed · **275 capability rows across 9 areas** · **248 in-scope / 27 out-of-scope** (24 plotting + 3 IO). The **248 in-scope capabilities** are the actionable comparison surface that the Phase 17 parity matrix maps against.

## Packages Surveyed

| Package | Version | CRAN Date | Primary Area | Status |
|---------|---------|-----------|--------------|--------|
| `fda` | 6.3.0 | 2025-05-21 | Representation/Basis/Smoothing | Active |
| `fda.usc` | 2.2.0 | 2024-11-09 | General (depth, regression, classification) | Active |
| `refund` | 0.1-40 | 2026-03-21 | ML Regression | Active |
| `fdapace` | 0.6.0 | 2024-07-03 | FPCA / Sparse longitudinal | Active |
| `roahd` | 1.4.3 | 2021-11-04 | Depth / Outlier | Active |
| `fdaoutlier` | 0.2.1 | 2023-09-30 | Depth / Outlier | Active |
| `ftsa` | 6.7 | 2026-03-31 | Functional Time Series | Active |
| `MFPCA` | 1.3-11 | 2025-08-27 | FPCA (multivariate) | Active |
| `funData` | 1.3-9 | 2024-02-14 | Representation | Active |
| `fdasrvf` | 2.4.4 | 2026-05-07 | Elastic / Shape | Active |
| `fdatest` | 2.1.1 | 2022-05-04 | Inference / Testing | Active |
| `fdANOVA` | 0.1.2 | 2018-08-29 | Inference / Testing | Active |
| `frechet` | 0.3.0 | 2023-12-09 | Density / Object Data / Manifold | Active |
| `fdadensity` | 0.1.4 | 2025-03-29 | Density / Object Data | Active |
| `funHDDC` | 2.3.1.1 | 2026-05-08 | ML Clustering | Active |
| `FDboost` | 1.1-4 | 2026-03-24 | ML Regression | Active |
| `face` | 0.1-8 | 2025-09-01 | FPCA / Covariance | Active |
| `denseFLMM` | 0.1.3 | 2025-04-16 | ML Regression (mixed effects) | Active |
| `funcharts` | 1.8.1 | 2026-01-18 | SPM / Control Charts | Active |
| `fdacluster` | 0.4.2 | 2026-01-14 | ML Clustering | Active |
| `registr` | 2.2.1 | 2026-02-17 | Preprocessing / Registration | Active |
| `conformalInference.fd` | 1.1.1 | 2022-03-23 | ML / Inference | Active |
| `fdaPDE` | 1.1-24 | 2026-06-04 | Representation / Smoothing | Active |
| `SCBmeanfd` | 1.2.3 | 2025-05-21 | Inference / Testing | Active |
| `mfaces` | 0.1-4 | 2022-07-19 | FPCA / Covariance | Active |
| `fdaPOIFD` | 2.0.1 | 2025-09-02 | Depth / Partially Observed | Active |
| `multifamm` | 0.1.1 | 2021-09-28 | ML Regression | Active |
| `elasdics` | 1.1.3 | 2024-01-25 | Elastic / Shape | Active |
| `freqdom` | 2.0.5 | 2024-04-06 | Functional Time Series | Active |
| `fdaconcur` | 0.1.3 | 2024-07-20 | ML Regression | Active |
| `fdaACF` | 1.0.0 | 2020-10-20 | Functional Time Series | Active |
| `fastFMM` | 1.0.1 | 2026-05-18 | ML Regression | Active |
| `funFEM` | 1.2 | 2021-10-27 | ML Clustering | Active |
| `funLBM` | 2.3.1 | 2026-07-30 | ML Clustering (co-clustering) | Active |
| `tf` | 0.5.0 | 2026-07-14 | Representation | Active |

**Candidate packages considered and excluded:**
- `classiFunc`: ARCHIVED on CRAN 2020-02-19 due to unresolved check problems. Excluded.
- `FRegSigCom`: ARCHIVED on CRAN 2020-05-19. Excluded.
- `fpca`: ARCHIVED on CRAN 2022-03-06. Excluded.
- `rainbow`: Version 3.8 (2024-01-23). Included in inventory but **all capabilities are Out-of-Scope (plotting)** — the package implements bagplots, boxplots, and rainbow plots for functional data. No in-scope numeric algorithm layer beyond what roahd/fdaoutlier already cover. Listed in §Design-Goal Filter for completeness.
- `funLBM` v2.3.1 (2026-07-30): Included — model-based co-clustering of functions is an in-scope numeric algorithm.
- `refund.shiny`: UI / Shiny app — Out-of-Scope (IO/UI). Excluded.
- `mlr3fda`: Machine learning pipeline framework integration — Out-of-Scope (framework plumbing). Excluded.
- `tidyfun` / `tidyverse` extensions: Primarily data manipulation + visualization ergonomics. Excluded — `tf` covers the tidy functional data representation which is the in-scope layer.

---

## §R-Inventory — Capability Tables by Area

### Capability Row Schema

| Column | Description |
|--------|-------------|
| **Capability** | The distinct algorithm or user task (one row per capability; fit/predict/transform collapsed) |
| **Task Group** | Thematic sub-grouping within the area |
| **Source Package(s)** | `pkg (vX.Y.Z)` — all packages offering this capability |
| **Relevance** | `In-Scope Algorithm`, `In-Scope API-Ergonomics`, `Out-of-Scope (plotting)`, or `Out-of-Scope (IO)` |
| **Confidence** | HIGH (CRAN page or rdrr.io fetched this session) / MEDIUM (web search or training cross-checked) / LOW (training knowledge only) |

---

### Area 1: Representation / Basis Systems / Smoothing

**Description:** Core data types, basis systems, functional data object construction, smoothing/fitting, derivatives, integration, and inner products. R's `fda` package is the canonical infrastructure provider; `funData`/`tf` offer alternative object models.

**In-scope count: 38   Out-of-scope count: 7**

| Capability | Task Group | Source Package(s) | Relevance | Confidence |
|------------|------------|-------------------|-----------|------------|
| B-spline basis system (creation, evaluation, penalty matrix) | Basis systems | `fda` (6.3.0) | In-Scope Algorithm | HIGH |
| Fourier basis system (creation, evaluation, penalty matrix) | Basis systems | `fda` (6.3.0) | In-Scope Algorithm | HIGH |
| Monomial / polynomial basis (creation, evaluation, penalty) | Basis systems | `fda` (6.3.0) | In-Scope Algorithm | HIGH |
| Constant / intercept basis | Basis systems | `fda` (6.3.0) | In-Scope Algorithm | HIGH |
| Exponential basis | Basis systems | `fda` (6.3.0) | In-Scope Algorithm | HIGH |
| Power basis | Basis systems | `fda` (6.3.0) | In-Scope Algorithm | HIGH |
| Polygonal (piecewise-linear) basis | Basis systems | `fda` (6.3.0) | In-Scope Algorithm | HIGH |
| Finite element basis (2D/3D irregular domains via PDE) | Basis systems | `fdaPDE` (1.1-24) | In-Scope Algorithm | HIGH |
| Smooth functional data object from raw data (basis expansion via penalized least squares) | Smoothing/Fitting | `fda` (6.3.0), `fda.usc` (2.2.0) | In-Scope Algorithm | HIGH |
| Smoothing with automatic smoothing parameter selection (GCV, AIC) | Smoothing/Fitting | `fda` (6.3.0) | In-Scope Algorithm | HIGH |
| Monotone smoothing (Ramsay's integral-of-exp approach for strictly monotone curves) | Smoothing/Fitting | `fda` (6.3.0) | In-Scope Algorithm | HIGH |
| Positive-valued smoothing (log-transformed) | Smoothing/Fitting | `fda` (6.3.0) | In-Scope Algorithm | HIGH |
| Bivariate functional data smoothing (smooth.bibasis) | Smoothing/Fitting | `fda` (6.3.0) | In-Scope Algorithm | HIGH |
| P-spline (penalized spline) smoothing for sparse functional data | Smoothing/Fitting | `face` (0.1-8), `fdaPDE` (1.1-24) | In-Scope Algorithm | HIGH |
| Fast covariance estimation for sparse functional data (FACE algorithm) | Covariance estimation | `face` (0.1-8) | In-Scope Algorithm | HIGH |
| Fast covariance estimation for multivariate sparse functional data | Covariance estimation | `mfaces` (0.1-4) | In-Scope Algorithm | HIGH |
| Smoothing over complex 2D/3D domains with PDE regularization (finite element) | Smoothing/Fitting | `fdaPDE` (1.1-24) | In-Scope Algorithm | HIGH |
| Functional data derivative computation | Derivatives | `fda` (6.3.0), `fda.usc` (2.2.0) | In-Scope Algorithm | HIGH |
| Linear differential operator object (Lfd) — define and evaluate | Differential Operators | `fda` (6.3.0) | In-Scope Algorithm | HIGH |
| Basis penalty matrix computation | Differential Operators | `fda` (6.3.0) | In-Scope API-Ergonomics | HIGH |
| Inner product / L2 norm between functional data objects | Inner Products | `fda` (6.3.0) | In-Scope Algorithm | HIGH |
| Trapezoidal integration (trapzmat) | Integration | `fda` (6.3.0) | In-Scope Algorithm | HIGH |
| Functional data object arithmetic (addition, subtraction, scalar multiplication) | Object operations | `fda` (6.3.0), `funData` (1.3-9), `tf` (0.5.0) | In-Scope API-Ergonomics | HIGH |
| Evaluate functional data at arbitrary points (off-grid interpolation) | Object operations | `fda` (6.3.0), `funData` (1.3-9), `tf` (0.5.0) | In-Scope Algorithm | HIGH |
| Univariate functional data S4 class (funData) | Data representation | `funData` (1.3-9) | In-Scope API-Ergonomics | HIGH |
| Multivariate functional data S4 class (multiFunData) | Data representation | `funData` (1.3-9) | In-Scope API-Ergonomics | HIGH |
| Irregular functional data S4 class (irregFunData) | Data representation | `funData` (1.3-9) | In-Scope API-Ergonomics | HIGH |
| Tidy S3 functional vector (grid, spline-basis, FPC representations) | Data representation | `tf` (0.5.0) | In-Scope API-Ergonomics | HIGH |
| Grid resampling / re-evaluation on new evaluation points | Object operations | `tf` (0.5.0), `funData` (1.3-9) | In-Scope Algorithm | HIGH |
| Functional data centering | Preprocessing | `fda` (6.3.0), `fda.usc` (2.2.0) | In-Scope Algorithm | HIGH |
| Mean, variance, and covariance function from functional sample | Summary statistics | `fda` (6.3.0), `fda.usc` (2.2.0), `roahd` (1.4.3) | In-Scope Algorithm | HIGH |
| Basis conversion / projection (grid → basis, basis → basis) | Basis conversion | `fda` (6.3.0), `fda.usc` (2.2.0) | In-Scope Algorithm | HIGH |
| Functional data sub-setting and domain restriction | Object operations | `tf` (0.5.0), `funData` (1.3-9) | In-Scope API-Ergonomics | HIGH |
| Principal differential analysis (PDA — estimate linear ODE from data) | Differential Operators | `fda` (6.3.0) | In-Scope Algorithm | HIGH |
| Functional data integration over sub-domain | Integration | `tf` (0.5.0) | In-Scope Algorithm | HIGH |
| Local min/max detection on functional data | Feature extraction | `tf` (0.5.0) | In-Scope Algorithm | HIGH |
| fdata class (functional data container with argvals) | Data representation | `fda.usc` (2.2.0) | In-Scope API-Ergonomics | HIGH |
| Extrapolation strategies (boundary, periodic, constant fill, exception) | Extrapolation | `fda` (6.3.0) | In-Scope Algorithm | MEDIUM |
| Plot functional data objects | Visualization | `fda` (6.3.0), `funData` (1.3-9), `tf` (0.5.0) | Out-of-Scope (plotting) | HIGH |
| Plot basis system or differential operator | Visualization | `fda` (6.3.0) | Out-of-Scope (plotting) | HIGH |
| Plot smooth fit vs raw data | Visualization | `fda` (6.3.0) | Out-of-Scope (plotting) | HIGH |
| Functional data ggplot2 geom layer (geom_spaghetti etc.) | Visualization | `tf` (0.5.0) | Out-of-Scope (plotting) | HIGH |
| Load built-in datasets (growth, weather, etc.) | IO | `fda` (6.3.0) | Out-of-Scope (IO) | HIGH |
| Export/import functional data to/from data frames | IO | `tf` (0.5.0), `fda.usc` (2.2.0) | Out-of-Scope (IO) | HIGH |
| fds dataset collection (19 built-in functional datasets) | IO | `fds` | Out-of-Scope (IO) | HIGH |

---

### Area 2: Preprocessing / Registration

**Description:** Curve registration and alignment (landmark, continuous, elastic/SRVF), missing-value handling, irregular-to-regular conversion, phase/amplitude separation, and dimension reduction as a preprocessing step.

**In-scope count: 22   Out-of-scope count: 0**

| Capability | Task Group | Source Package(s) | Relevance | Confidence |
|------------|------------|-------------------|-----------|------------|
| Landmark registration (hard-pin time points) | Registration | `fda` (6.3.0) | In-Scope Algorithm | HIGH |
| Continuous registration to a target curve (minimize warping criterion) | Registration | `fda` (6.3.0) | In-Scope Algorithm | HIGH |
| Group-wise elastic registration via SRVF Karcher mean | Registration | `fdasrvf` (2.4.4) | In-Scope Algorithm | HIGH |
| Pair-wise elastic alignment (SRVF geodesic DP) | Registration | `fdasrvf` (2.4.4) | In-Scope Algorithm | HIGH |
| 2D elastic curve alignment (elastic distance, sparse/dense/irregular) | Registration | `elasdics` (1.1.3) | In-Scope Algorithm | HIGH |
| Elastic curve mean (Karcher mean in SRVF space) | Registration | `fdasrvf` (2.4.4), `elasdics` (1.1.3) | In-Scope Algorithm | HIGH |
| SRVF/SRSF transformation (function → square-root velocity) | Registration | `fdasrvf` (2.4.4) | In-Scope Algorithm | HIGH |
| Joint registration and non-Gaussian FPCA (binary/continuous, exponential family) | Registration | `registr` (2.2.1) | In-Scope Algorithm | HIGH |
| Incomplete curve registration (partial observation support) | Registration | `registr` (2.2.1) | In-Scope Algorithm | HIGH |
| Warping function computation, composition, and inversion | Registration | `fda` (6.3.0), `fdasrvf` (2.4.4) | In-Scope Algorithm | HIGH |
| Phase/amplitude separation and amplitude/phase FPCA | Registration | `fdasrvf` (2.4.4) | In-Scope Algorithm | HIGH |
| Smooth warping function estimation (smooth.morph) | Registration | `fda` (6.3.0) | In-Scope Algorithm | HIGH |
| K-means with simultaneous alignment | Registration + Clustering | `fdacluster` (0.4.2), `fdasrvf` (2.4.4) | In-Scope Algorithm | HIGH |
| Shift, dilation, and affine warping for alignment | Registration | `fdacluster` (0.4.2) | In-Scope Algorithm | HIGH |
| Functional PCA for dimensionality reduction (preprocessing role) | Dim. reduction | `fda` (6.3.0), `fda.usc` (2.2.0), `fdapace` (0.6.0) | In-Scope Algorithm | HIGH |
| Functional PLS for dimensionality reduction | Dim. reduction | `fda.usc` (2.2.0) | In-Scope Algorithm | HIGH |
| Smoothing for irregularly spaced observations | Preprocessing | `face` (0.1-8), `fdapace` (0.6.0) | In-Scope Algorithm | HIGH |
| Functional data normalization / centering | Preprocessing | `fda` (6.3.0), `fda.usc` (2.2.0) | In-Scope Algorithm | HIGH |
| Cross-validation smoothing parameter selection (leave-one-curve-out) | Preprocessing | `fda.usc` (2.2.0), `refund` (0.1-40) | In-Scope API-Ergonomics | HIGH |
| Data2fd — quick basis expansion from raw matrix | Preprocessing | `fda` (6.3.0) | In-Scope API-Ergonomics | HIGH |
| Stringing (map high-dimensional scalar data to functional form) | Preprocessing | `fdapace` (0.6.0) | In-Scope Algorithm | MEDIUM |
| Registration quality score (warping complexity, amplitude variance) | Registration | `fdasrvf` (2.4.4) | In-Scope API-Ergonomics | MEDIUM |

---

### Area 3: Exploratory / Depth / Outlier Detection

**Description:** Functional depth measures, outlyingness statistics, outlier detection algorithms, robust summary statistics, and the numeric underpinnings of depth-based diagnostic plots.

**In-scope count: 31   Out-of-scope count: 7**

| Capability | Task Group | Source Package(s) | Relevance | Confidence |
|------------|------------|-------------------|-----------|------------|
| Modified Band Depth (MBD) for univariate functional data | Depth measures | `roahd` (1.4.3), `fdaoutlier` (0.2.1), `fda.usc` (2.2.0) | In-Scope Algorithm | HIGH |
| Band Depth (BD) for univariate functional data | Depth measures | `roahd` (1.4.3), `fdaoutlier` (0.2.1) | In-Scope Algorithm | HIGH |
| Modified Band Depth for multivariate functional data (multiMBD) | Depth measures | `roahd` (1.4.3) | In-Scope Algorithm | HIGH |
| Half-Region Depth (HRD) for univariate functional data | Depth measures | `roahd` (1.4.3) | In-Scope Algorithm | HIGH |
| Modified Half-Region Depth (MHRD) | Depth measures | `roahd` (1.4.3) | In-Scope Algorithm | HIGH |
| Epigraph Index (EI) | Depth measures | `roahd` (1.4.3) | In-Scope Algorithm | HIGH |
| Modified Epigraph Index (MEI) | Depth measures | `roahd` (1.4.3) | In-Scope Algorithm | HIGH |
| Hypograph Index (HI) | Depth measures | `roahd` (1.4.3) | In-Scope Algorithm | HIGH |
| Modified Hypograph Index (MHI) | Depth measures | `roahd` (1.4.3) | In-Scope Algorithm | HIGH |
| Extremal depth | Depth measures | `fdaoutlier` (0.2.1) | In-Scope Algorithm | HIGH |
| Extreme Rank Length Depth | Depth measures | `fdaoutlier` (0.2.1) | In-Scope Algorithm | HIGH |
| L-infinity depth | Depth measures | `fdaoutlier` (0.2.1) | In-Scope Algorithm | HIGH |
| Total Variation Depth (TVD) and Modified Shape Similarity Index (MSSI) | Depth measures | `fdaoutlier` (0.2.1) | In-Scope Algorithm | HIGH |
| Random projection depth for multivariate data | Depth measures | `fdaoutlier` (0.2.1) | In-Scope Algorithm | HIGH |
| Elastic depth (shape depth in SRVF space) | Depth measures | `fdasrvf` (2.4.4) | In-Scope Algorithm | HIGH |
| Integrated functional depth for partially observed data | Depth measures | `fdaPOIFD` (2.0.1) | In-Scope Algorithm | HIGH |
| General functional depth dispatcher (multiple depth methods) | Depth measures | `fda.usc` (2.2.0) | In-Scope API-Ergonomics | HIGH |
| Directional outlyingness statistic (Dai & Genton 2019) | Outlier detection | `fdaoutlier` (0.2.1) | In-Scope Algorithm | HIGH |
| MS-plot statistic (magnitude-shape outlyingness) | Outlier detection | `fdaoutlier` (0.2.1) | In-Scope Algorithm | HIGH |
| Outliergram statistic (MO vs MEI plane — shape outlier detection) | Outlier detection | `roahd` (1.4.3) | In-Scope Algorithm | HIGH |
| Depthgram statistic (multivariate depth outlier visualization score) | Outlier detection | `roahd` (1.4.3) | In-Scope Algorithm | HIGH |
| TVD+MSSI-based outlier detection (tvdmss) | Outlier detection | `fdaoutlier` (0.2.1) | In-Scope Algorithm | HIGH |
| Massive Unsupervised Outlier Detection (MUOD) | Outlier detection | `fdaoutlier` (0.2.1) | In-Scope Algorithm | HIGH |
| Sequential transformation outlier detection | Outlier detection | `fdaoutlier` (0.2.1) | In-Scope Algorithm | HIGH |
| Elastic changepoint detection (amplitude + phase, SRSF space) | Outlier/Change | `fdasrvf` (2.4.4) | In-Scope Algorithm | HIGH |
| Outlier detection for partially observed curves (depth-based) | Outlier detection | `fdaPOIFD` (2.0.1) | In-Scope Algorithm | HIGH |
| Functional bootstrap confidence interval for mean / summary statistics | Robust summaries | `fda.usc` (2.2.0) | In-Scope Algorithm | HIGH |
| Robust mean and median for functional samples | Robust summaries | `roahd` (1.4.3), `fda.usc` (2.2.0) | In-Scope Algorithm | HIGH |
| Spearman and Kendall rank correlation for functional/multivariate data | Robust summaries | `roahd` (1.4.3) | In-Scope Algorithm | HIGH |
| Bootstrap hypothesis test on Spearman correlation | Robust summaries | `roahd` (1.4.3) | In-Scope Algorithm | HIGH |
| Covariance function estimation (empirical) | Robust summaries | `roahd` (1.4.3), `fda.usc` (2.2.0) | In-Scope Algorithm | HIGH |
| Functional boxplot (depth-based outlier thresholds and fence values) | Outlier detection | `roahd` (1.4.3), `fdaoutlier` (0.2.1), `fda.usc` (2.2.0) | In-Scope Algorithm | HIGH |
| Functional boxplot visualization | Visualization | `roahd` (1.4.3), `fdaoutlier` (0.2.1) | Out-of-Scope (plotting) | HIGH |
| Outliergram plot (MO vs MEI scatter) | Visualization | `roahd` (1.4.3) | Out-of-Scope (plotting) | HIGH |
| Depthgram plot | Visualization | `roahd` (1.4.3) | Out-of-Scope (plotting) | HIGH |
| MS-plot (magnitude-shape scatter) | Visualization | `fdaoutlier` (0.2.1) | Out-of-Scope (plotting) | HIGH |
| Rainbow plot / bagplot for functional outliers | Visualization | `rainbow` (3.8) | Out-of-Scope (plotting) | HIGH |
| Functional spaghetti / rainbow visualization | Visualization | `rainbow` (3.8), `roahd` (1.4.3) | Out-of-Scope (plotting) | HIGH |
| Partial reconstruction of missing curves (depth-based) | Outlier/Missing | `fdaPOIFD` (2.0.1) | In-Scope Algorithm | HIGH |

---

### Area 4: ML — Regression / Classification / Clustering

**Description:** Scalar-on-function, function-on-scalar, function-on-function, and concurrent regression; functional GLMs and additive models; classification methods; clustering with and without alignment; functional PCA-based dimensionality reduction for ML.

**In-scope count: 59   Out-of-scope count: 2**

#### Regression

| Capability | Task Group | Source Package(s) | Relevance | Confidence |
|------------|------------|-------------------|-----------|------------|
| Scalar-on-function regression via basis expansion (penalized OLS) | Scalar-on-function | `fda` (6.3.0), `fda.usc` (2.2.0) | In-Scope Algorithm | HIGH |
| Scalar-on-function regression via functional PCA scores | Scalar-on-function | `fda` (6.3.0), `fda.usc` (2.2.0), `refund` (0.1-40) | In-Scope Algorithm | HIGH |
| Scalar-on-function regression via PLS scores | Scalar-on-function | `fda.usc` (2.2.0) | In-Scope Algorithm | HIGH |
| Scalar-on-function nonparametric kernel regression | Scalar-on-function | `fda.usc` (2.2.0) | In-Scope Algorithm | HIGH |
| Penalized scalar-on-function regression (pfr, spline-based) | Scalar-on-function | `refund` (0.1-40) | In-Scope Algorithm | HIGH |
| Functional generalized linear model with scalar response (logistic, Poisson) | Scalar-on-function | `fda.usc` (2.2.0), `refund` (0.1-40) | In-Scope Algorithm | HIGH |
| Functional generalized spectral additive model (GSAM) | Scalar-on-function | `fda.usc` (2.2.0) | In-Scope Algorithm | MEDIUM |
| Functional generalized kernel additive model (GKAM) | Scalar-on-function | `fda.usc` (2.2.0) | In-Scope Algorithm | MEDIUM |
| Varying coefficient / concurrent functional regression | Scalar-on-function | `fdaconcur` (0.1.3), `refund` (0.1-40) | In-Scope Algorithm | HIGH |
| History index model (effect of past predictor values) | Scalar-on-function | `fdaconcur` (0.1.3), `refund` (0.1-40) | In-Scope Algorithm | HIGH |
| Scalar-on-function variable selection (fosr.vs) | Scalar-on-function | `refund` (0.1-40) | In-Scope Algorithm | HIGH |
| Scalar-on-function permutation test wrapper (fosr.perm) | Scalar-on-function | `refund` (0.1-40) | In-Scope API-Ergonomics | HIGH |
| Function-on-scalar regression (fosr — penalized, two-step, OLS, GLS) | Function-on-scalar | `refund` (0.1-40), `FDboost` (1.1-4) | In-Scope Algorithm | HIGH |
| Bayesian function-on-scalar regression (Gibbs/VB with FPCA or Wishart prior) | Function-on-scalar | `refund` (0.1-40) | In-Scope Algorithm | HIGH |
| Function-on-scalar regression via boosting base-learners | Function-on-scalar | `FDboost` (1.1-4) | In-Scope Algorithm | HIGH |
| Function-on-function regression (ff, ffpc, sff, pffr) | Function-on-function | `refund` (0.1-40), `FDboost` (1.1-4) | In-Scope Algorithm | HIGH |
| Penalized flexible functional regression (pffr — multivariate, random effects) | Function-on-function | `refund` (0.1-40) | In-Scope Algorithm | HIGH |
| Function-on-scalar linear model (fRegress framework from fda) | Function-on-scalar | `fda` (6.3.0) | In-Scope Algorithm | HIGH |
| Function-on-scalar regression with functional response (fda.usc basis approach) | Function-on-scalar | `fda.usc` (2.2.0) | In-Scope Algorithm | HIGH |
| Elastic regression (scalar response, SRVF-space representation) | Elastic regression | `fdasrvf` (2.4.4) | In-Scope Algorithm | HIGH |
| Elastic logistic regression | Elastic regression | `fdasrvf` (2.4.4) | In-Scope Algorithm | HIGH |
| Elastic multinomial logistic regression | Elastic regression | `fdasrvf` (2.4.4) | In-Scope Algorithm | HIGH |
| Elastic principal component regression | Elastic regression | `fdasrvf` (2.4.4) | In-Scope Algorithm | HIGH |
| Functional additive model (FAM — scalar response, functional covariates) | Scalar-on-function | `fdapace` (0.6.0) | In-Scope Algorithm | HIGH |
| Functional concurrent regression (varying coefficient, sparse/dense) | Scalar-on-function | `fdapace` (0.6.0), `fdaconcur` (0.1.3) | In-Scope Algorithm | HIGH |
| Functional linear model with cross-validation (basis, PCA, PLS) | Scalar-on-function | `fda.usc` (2.2.0) | In-Scope API-Ergonomics | HIGH |
| Functional linear model goodness-of-fit / F-test | Scalar-on-function | `fda.usc` (2.2.0) | In-Scope Algorithm | HIGH |
| Functional linear mixed model for dense data (denseFLMM) | Mixed effects | `denseFLMM` (0.1.3) | In-Scope Algorithm | HIGH |
| Multivariate functional additive mixed model (multiFAMM) | Mixed effects | `multifamm` (0.1.1) | In-Scope Algorithm | HIGH |
| Fast functional mixed model inference (fastFMM) | Mixed effects | `fastFMM` (1.0.1) | In-Scope Algorithm | HIGH |
| Functional quantile regression (scalar response, functional covariates) | Scalar-on-function | `fdapace` (0.6.0) | In-Scope Algorithm | HIGH |
| Stringing regression (high-dimensional → functional → FLM) | Scalar-on-function | `fdapace` (0.6.0) | In-Scope Algorithm | MEDIUM |
| Bootstrap confidence intervals for regression coefficients | Regression utils | `fda.usc` (2.2.0) | In-Scope Algorithm | HIGH |
| Cross-validation for functional regression (LOOCV, k-fold) | Regression utils | `fda.usc` (2.2.0), `refund` (0.1-40) | In-Scope API-Ergonomics | HIGH |
| GAMLSS for functional response (gradient boosting, distributional regression) | Function-on-scalar | `FDboost` (1.1-4) | In-Scope Algorithm | HIGH |

#### Classification

| Capability | Task Group | Source Package(s) | Relevance | Confidence |
|------------|------------|-------------------|-----------|------------|
| Functional LDA (linear discriminant analysis via basis/FPC scores) | Classification | `fda.usc` (2.2.0) | In-Scope Algorithm | HIGH |
| Functional generalized linear model classifier (logistic basis) | Classification | `fda.usc` (2.2.0) | In-Scope Algorithm | HIGH |
| Functional kernel classifier (nonparametric kNN-type) | Classification | `fda.usc` (2.2.0) | In-Scope Algorithm | HIGH |
| DD-classifier (depth vs depth plot based) | Classification | `fda.usc` (2.2.0) | In-Scope Algorithm | HIGH |
| Functional classification using ML algorithms (SVM, RF, etc. on FPC scores) | Classification | `fda.usc` (2.2.0) | In-Scope Algorithm | MEDIUM |
| Elastic logistic classification (SRVF-space) | Classification | `fdasrvf` (2.4.4) | In-Scope Algorithm | HIGH |
| Depth-based outlier classification | Classification | `fdaPOIFD` (2.0.1) | In-Scope Algorithm | HIGH |
| Cross-validation classification (ClassifCv equivalent) | Classification | `fda.usc` (2.2.0) | In-Scope API-Ergonomics | HIGH |

#### Clustering

| Capability | Task Group | Source Package(s) | Relevance | Confidence |
|------------|------------|-------------------|-----------|------------|
| Functional K-means clustering with simultaneous alignment (Sangalli et al.) | Clustering | `fdacluster` (0.4.2) | In-Scope Algorithm | HIGH |
| Hierarchical agglomerative clustering for functional data | Clustering | `fdacluster` (0.4.2) | In-Scope Algorithm | HIGH |
| DBSCAN density-based functional clustering | Clustering | `fdacluster` (0.4.2) | In-Scope Algorithm | HIGH |
| K-means elastic clustering with SRVF alignment | Clustering | `fdasrvf` (2.4.4) | In-Scope Algorithm | HIGH |
| kCFC functional clustering via subspace embedding | Clustering | `fdapace` (0.6.0) | In-Scope Algorithm | HIGH |
| Model-based clustering in group-specific functional subspaces (funHDDC) | Clustering | `funHDDC` (2.3.1.1) | In-Scope Algorithm | HIGH |
| Model-based clustering in discriminative functional subspace (funFEM) | Clustering | `funFEM` (1.2) | In-Scope Algorithm | HIGH |
| Model-based co-clustering of functions (rows + columns simultaneously) | Clustering | `funLBM` (2.3.1) | In-Scope Algorithm | HIGH |
| Slope heuristic for functional cluster model selection | Clustering | `funHDDC` (2.3.1.1) | In-Scope API-Ergonomics | HIGH |
| K-means functional clustering (basic) | Clustering | `fda.usc` (2.2.0) | In-Scope Algorithm | HIGH |
| Conformal prediction regions for functional regression | Conformal | `conformalInference.fd` (1.1.1) | In-Scope Algorithm | HIGH |
| Conformal prediction split / multi-split variants | Conformal | `conformalInference.fd` (1.1.1) | In-Scope Algorithm | HIGH |
| Functional FPCA-based prediction (FPC scores → linear prediction) | Regression utils | `fda` (6.3.0), `refund` (0.1-40), `fda.usc` (2.2.0) | In-Scope Algorithm | HIGH |
| Visualization of regression coefficient functions | Visualization | `refund` (0.1-40), `FDboost` (1.1-4) | Out-of-Scope (plotting) | HIGH |
| Stability selection for FDboost model terms | Regression utils | `FDboost` (1.1-4) | In-Scope API-Ergonomics | HIGH |

---

### Area 5: Inference / Testing

**Description:** Hypothesis tests for functional data — one-sample, two-sample, ANOVA, regression testing, simultaneous confidence bands, and permutation/bootstrap-based inference.

**In-scope count: 22   Out-of-scope count: 3**

| Capability | Task Group | Source Package(s) | Relevance | Confidence |
|------------|------------|-------------------|-----------|------------|
| Interval Testing Procedure (ITP) for one-population functional data (B-spline basis) | One-sample tests | `fdatest` (2.1.1) | In-Scope Algorithm | HIGH |
| ITP one-population (Fourier basis) | One-sample tests | `fdatest` (2.1.1) | In-Scope Algorithm | HIGH |
| ITP two-population comparison (B-spline, Fourier, phase-amplitude Fourier bases) | Two-sample tests | `fdatest` (2.1.1) | In-Scope Algorithm | HIGH |
| Functional ANOVA (one-way, multiple groups) via ITP | ANOVA | `fdatest` (2.1.1), `fdANOVA` (0.1.2) | In-Scope Algorithm | HIGH |
| Functional ANOVA using random projections (univariate, multivariate) | ANOVA | `fdANOVA` (0.1.2) | In-Scope Algorithm | HIGH |
| Functional MANOVA via permutation + basis function representation | ANOVA | `fdANOVA` (0.1.2) | In-Scope Algorithm | HIGH |
| Functional MANOVA via random projections | ANOVA | `fdANOVA` (0.1.2) | In-Scope Algorithm | HIGH |
| F-permutation test for functional data (Fperm.fd) | Permutation tests | `fda` (6.3.0) | In-Scope Algorithm | HIGH |
| t-permutation test for functional two-sample comparison | Permutation tests | `fda` (6.3.0) | In-Scope Algorithm | HIGH |
| Functional linear model on-function testing (ITPlmbspline — scalar-on-function) | Regression tests | `fdatest` (2.1.1) | In-Scope Algorithm | HIGH |
| Goodness-of-fit test for the functional linear model (FLM) | Regression tests | `fda.usc` (2.2.0) | In-Scope Algorithm | HIGH |
| F-test for the FLM with scalar response | Regression tests | `fda.usc` (2.2.0) | In-Scope Algorithm | HIGH |
| Delsol-Ferraty-Vieu test (no functional-scalar relationship) | Regression tests | `fda.usc` (2.2.0) | In-Scope Algorithm | HIGH |
| Equality of functional distributions test | Distribution tests | `fda.usc` (2.2.0) | In-Scope Algorithm | HIGH |
| Equality of functional means / covariance test | Distribution tests | `fda.usc` (2.2.0) | In-Scope Algorithm | HIGH |
| Distance correlation and t-test for functional data | Association tests | `fda.usc` (2.2.0) | In-Scope Algorithm | MEDIUM |
| Simultaneous confidence bands for mean function (SCBmeanfd) | Confidence bands | `SCBmeanfd` (1.2.3) | In-Scope Algorithm | HIGH |
| Goodness-of-fit test for mean model (SCBmeanfd) | Goodness-of-fit | `SCBmeanfd` (1.2.3) | In-Scope Algorithm | HIGH |
| Two-sample equality test (SCBmeanfd) | Two-sample tests | `SCBmeanfd` (1.2.3) | In-Scope Algorithm | HIGH |
| Stationarity test for functional time series (T_stationary) | Time series tests | `ftsa` (6.7) | In-Scope Algorithm | HIGH |
| Bootstrap confidence interval on Spearman correlation | Bootstrap inference | `roahd` (1.4.3) | In-Scope Algorithm | HIGH |
| Likelihood ratio test for smooth functional effects (rlrt.pfr) | Regression tests | `refund` (0.1-40) | In-Scope Algorithm | HIGH |
| P-value heatmap visualization | Visualization | `fdatest` (2.1.1) | Out-of-Scope (plotting) | HIGH |
| Functional ANOVA result plots | Visualization | `fdANOVA` (0.1.2) | Out-of-Scope (plotting) | HIGH |
| Corrected p-value plot | Visualization | `fdatest` (2.1.1) | Out-of-Scope (plotting) | HIGH |

---

### Area 6: Functional Time Series

**Description:** Methods specifically designed for sequences of functional observations ordered in time — forecasting, decomposition, autocorrelation structure, frequency-domain analysis, simulation, and stationarity diagnostics.

**In-scope count: 24   Out-of-scope count: 2**

| Capability | Task Group | Source Package(s) | Relevance | Confidence |
|------------|------------|-------------------|-----------|------------|
| Functional time series model fitting via FPCA (ftsm) | Decomposition/Fitting | `ftsa` (6.7) | In-Scope Algorithm | HIGH |
| Functional time series forecasting via FPC regression | Forecasting | `ftsa` (6.7) | In-Scope Algorithm | HIGH |
| Functional partial least squares forecasting (fplsr) | Forecasting | `ftsa` (6.7) | In-Scope Algorithm | HIGH |
| Dynamic updating of functional forecasts via FLR/OLS/RR/PLS | Forecasting | `ftsa` (6.7) | In-Scope Algorithm | HIGH |
| Iterative forecasting (ftsmiterativeforecasts) | Forecasting | `ftsa` (6.7) | In-Scope Algorithm | HIGH |
| Generalized additive extreme value modeling for functional data (GAEVforecast) | Forecasting | `ftsa` (6.7) | In-Scope Algorithm | MEDIUM |
| Functional autocorrelation function (facf) | Serial correlation | `ftsa` (6.7) | In-Scope Algorithm | HIGH |
| Functional autocorrelation function quantification via L2-norm | Serial correlation | `fdaACF` (1.0.0) | In-Scope Algorithm | HIGH |
| Partial functional autocorrelation function | Serial correlation | `fdaACF` (1.0.0) | In-Scope Algorithm | HIGH |
| Distribution of functional ACF under strong white noise | Serial correlation | `fdaACF` (1.0.0) | In-Scope Algorithm | HIGH |
| Stationarity test for functional time series | Hypothesis tests | `ftsa` (6.7) | In-Scope Algorithm | HIGH |
| Long-run covariance estimation via kernel sandwich estimator | Covariance | `ftsa` (6.7) | In-Scope Algorithm | HIGH |
| Compositional data functional PCA (CoDa_FPCA) | Decomposition | `ftsa` (6.7) | In-Scope Algorithm | MEDIUM |
| Log quantile density transform FPCA for distributions (LQDT_FPCA) | Decomposition | `ftsa` (6.7) | In-Scope Algorithm | HIGH |
| Maximum autocorrelation factors (MAF multivariate) | Decomposition | `ftsa` (6.7) | In-Scope Algorithm | MEDIUM |
| Multilevel functional data model (MFDM) | Decomposition | `ftsa` (6.7) | In-Scope Algorithm | MEDIUM |
| Dynamic functional PCA via Horta-Ziegelmann approach | Decomposition | `ftsa` (6.7) | In-Scope Algorithm | MEDIUM |
| Dynamic principal component analysis (DPCA) via spectral methods | Frequency domain | `freqdom` (2.0.5) | In-Scope Algorithm | HIGH |
| Spectral density operator estimation | Frequency domain | `freqdom` (2.0.5) | In-Scope Algorithm | HIGH |
| VAR and VMA process simulation for functional time series | Simulation | `freqdom` (2.0.5) | In-Scope Algorithm | HIGH |
| Functional time series summary statistics (mean, sd, variance, quantile, median) | Summary stats | `ftsa` (6.7) | In-Scope Algorithm | HIGH |
| Functional time series differencing | Preprocessing | `ftsa` (6.7) | In-Scope Algorithm | HIGH |
| Functional time series bootstrap resampling (fbootstrap) | Resampling | `ftsa` (6.7) | In-Scope Algorithm | HIGH |
| Functional ARMA simulation (sim_FARMA) | Simulation | `ftsa` (6.7) | In-Scope Algorithm | HIGH |
| Functional time series error measurement | Evaluation | `ftsa` (6.7) | In-Scope API-Ergonomics | HIGH |
| Functional time series visualization | Visualization | `ftsa` (6.7), `rainbow` (3.8) | Out-of-Scope (plotting) | HIGH |
| Rainbow plot for density-ordered time series | Visualization | `rainbow` (3.8) | Out-of-Scope (plotting) | HIGH |

---

### Area 7: Density / Object Data / Manifold

**Description:** Regression and statistics for non-Euclidean response objects in metric spaces — probability density functions, covariance matrices, correlation matrices, spherical data, networks, and point processes. Also covers density-on-function and distribution-valued FPCA.

**In-scope count: 24   Out-of-scope count: 1**

| Capability | Task Group | Source Package(s) | Relevance | Confidence |
|------------|------------|-------------------|-----------|------------|
| Global Fréchet regression (Euclidean predictors → metric-space response) | Fréchet regression | `frechet` (0.3.0) | In-Scope Algorithm | HIGH |
| Local Fréchet regression (kernel-weighted, Euclidean predictors) | Fréchet regression | `frechet` (0.3.0) | In-Scope Algorithm | HIGH |
| Fréchet mean computation in general metric spaces | Fréchet statistics | `frechet` (0.3.0) | In-Scope Algorithm | HIGH |
| Fréchet variance computation | Fréchet statistics | `frechet` (0.3.0) | In-Scope Algorithm | HIGH |
| Wasserstein distance between distributions | Distribution metrics | `frechet` (0.3.0) | In-Scope Algorithm | HIGH |
| Fréchet regression for density responses (2-Wasserstein space) | Density regression | `frechet` (0.3.0) | In-Scope Algorithm | HIGH |
| Fréchet ANOVA for object data (distributions) | Density regression | `frechet` (0.3.0) | In-Scope Algorithm | HIGH |
| Change point detection for object-valued time series | Change point | `frechet` (0.3.0) | In-Scope Algorithm | HIGH |
| Fréchet regression for covariance matrix responses (Frobenius, power, log-Cholesky metrics) | Covariance regression | `frechet` (0.3.0) | In-Scope Algorithm | HIGH |
| Fréchet integral (mean over domain) for covariance objects | Covariance regression | `frechet` (0.3.0) | In-Scope Algorithm | HIGH |
| Fréchet regression for spherical data | Spherical regression | `frechet` (0.3.0) | In-Scope Algorithm | HIGH |
| Geodesic computations on sphere (exp/log map, geodesic distance) | Spherical regression | `frechet` (0.3.0) | In-Scope Algorithm | HIGH |
| Fréchet regression for correlation matrices | Correlation regression | `frechet` (0.3.0) | In-Scope Algorithm | HIGH |
| Fréchet regression for network responses | Network regression | `frechet` (0.3.0) | In-Scope Algorithm | HIGH |
| Fréchet ANOVA for networks | Network regression | `frechet` (0.3.0) | In-Scope Algorithm | HIGH |
| Fréchet regression for point process responses | Point process | `frechet` (0.3.0) | In-Scope Algorithm | HIGH |
| FPCA for density functions via log quantile density (LQD) transform | Density FPCA | `fdadensity` (0.1.4) | In-Scope Algorithm | HIGH |
| LQD ↔ density / quantile function conversion | Density FPCA | `fdadensity` (0.1.4) | In-Scope Algorithm | HIGH |
| Fraction of variance explained (FVE) for LQD-FPCA | Density FPCA | `fdadensity` (0.1.4) | In-Scope API-Ergonomics | HIGH |
| Wasserstein Fréchet mean of densities | Density FPCA | `fdadensity` (0.1.4) | In-Scope Algorithm | HIGH |
| Density normalization and regularization | Density FPCA | `fdadensity` (0.1.4) | In-Scope Algorithm | HIGH |
| Multivariate FPCA across different-dimensional domains (1D, 2D, 3D) | FPCA — multivariate | `MFPCA` (1.3-11) | In-Scope Algorithm | HIGH |
| Tensor PCA for 2D functional data (UMPCA, FCP-TPA) | FPCA — tensor | `MFPCA` (1.3-11) | In-Scope Algorithm | HIGH |
| PACE-based FPCA for mixed-domain multivariate functional data | FPCA — multivariate | `MFPCA` (1.3-11) | In-Scope Algorithm | HIGH |
| DCT/spline basis expansions for 2D/3D domain data | FPCA — multivariate | `MFPCA` (1.3-11) | In-Scope API-Ergonomics | HIGH |
| MFPCA visualization (scree plot, scores) | Visualization | `MFPCA` (1.3-11) | Out-of-Scope (plotting) | HIGH |

---

### Area 8: Statistical Process Monitoring / Control Charts

**Description:** Functional and multivariate control charts, SPM phase estimation, profile monitoring. A distinct area because `funcharts` covers it in detail.

**In-scope count: 10   Out-of-scope count: 2**

| Capability | Task Group | Source Package(s) | Relevance | Confidence |
|------------|------------|-------------------|-----------|------------|
| Functional control chart (Phase I — reference set, control limits) | SPM | `funcharts` (1.8.1) | In-Scope Algorithm | HIGH |
| Functional control chart (Phase II — online monitoring) | SPM | `funcharts` (1.8.1) | In-Scope Algorithm | HIGH |
| Multivariate functional control chart | SPM | `funcharts` (1.8.1) | In-Scope Algorithm | HIGH |
| Robust functional control chart (via robustbase) | SPM | `funcharts` (1.8.1) | In-Scope Algorithm | HIGH |
| Profile monitoring (scalar-on-function + functional residual chart) | SPM | `funcharts` (1.8.1) | In-Scope Algorithm | HIGH |
| ARL estimation / simulation for functional control charts | SPM | `funcharts` (1.8.1) | In-Scope Algorithm | MEDIUM |
| Functional EWMA / CUSUM chart analogs | SPM | `funcharts` (1.8.1) | In-Scope Algorithm | MEDIUM |
| Phase I estimation with outlier masking | SPM | `funcharts` (1.8.1) | In-Scope Algorithm | MEDIUM |
| Parallel computing support for SPM chart simulation | SPM | `funcharts` (1.8.1) | In-Scope API-Ergonomics | HIGH |
| Contribution analysis for functional chart out-of-control signals | SPM | `funcharts` (1.8.1) | In-Scope Algorithm | MEDIUM |
| Functional control chart visualization | Visualization | `funcharts` (1.8.1) | Out-of-Scope (plotting) | HIGH |
| ggplot2-based SPM monitoring dashboard | Visualization | `funcharts` (1.8.1) | Out-of-Scope (plotting) | HIGH |

---

### Area 9: FPCA — Sparse / Longitudinal / Specialized

**Description:** Functional PCA with emphasis on sparse/longitudinal sampling (PACE algorithm), functional SVD, cross-covariance, empirical dynamics, and specialized FPCA variants.

*(Note: Standard dense-data FPCA appears in Areas 1, 4, and 7 where contextually grouped. This area captures capabilities unique to the sparse/longitudinal or specialized-FPCA packages — primarily `fdapace`.)*

**In-scope count: 18   Out-of-scope count: 3**

| Capability | Task Group | Source Package(s) | Relevance | Confidence |
|------------|------------|-------------------|-----------|------------|
| FPCA via PACE algorithm for sparse/irregularly sampled curves | Sparse FPCA | `fdapace` (0.6.0), `fda` (6.3.0) | In-Scope Algorithm | HIGH |
| Covariance surface estimation from sparse observations | Sparse FPCA | `fdapace` (0.6.0), `face` (0.1-8) | In-Scope Algorithm | HIGH |
| Mean curve estimation from sparse observations (PACE) | Sparse FPCA | `fdapace` (0.6.0) | In-Scope Algorithm | HIGH |
| FPC scores estimation via conditional expectation | Sparse FPCA | `fdapace` (0.6.0) | In-Scope Algorithm | HIGH |
| Fitted continuous trajectories with confidence bands for sparse data | Sparse FPCA | `fdapace` (0.6.0) | In-Scope Algorithm | HIGH |
| FPCA for functional derivatives (FPCAder) | FPCA variants | `fdapace` (0.6.0) | In-Scope Algorithm | HIGH |
| Functional SVD (FSVD — analogue of bivariate FPCA) | FPCA variants | `fdapace` (0.6.0) | In-Scope Algorithm | HIGH |
| Cross-covariance function estimation (GetCrCovYX, GetCrCovYZ) | Covariance | `fdapace` (0.6.0) | In-Scope Algorithm | HIGH |
| Dynamical correlation (DynCorr) | Correlation | `fdapace` (0.6.0) | In-Scope Algorithm | HIGH |
| Functional correlation (FCCor) | Correlation | `fdapace` (0.6.0) | In-Scope Algorithm | HIGH |
| Functional variance process analysis (FVPA) | Variance | `fdapace` (0.6.0) | In-Scope Algorithm | MEDIUM |
| Number of FPC selection criteria (ER_GR, FVE threshold) | FPCA model selection | `fdapace` (0.6.0), `ftsa` (6.7) | In-Scope API-Ergonomics | HIGH |
| MakeFPCAInputs — FPCA input validation and formatting | FPCA utils | `fdapace` (0.6.0) | In-Scope API-Ergonomics | HIGH |
| Functional fragment completion (incomplete curve reconstruction) | Sparse FPCA | `fdapace` (0.6.0) | In-Scope Algorithm | HIGH |
| Fast FPCA for sparse data (fpca.sc — sandwich smoother) | Sparse FPCA | `refund` (0.1-40) | In-Scope Algorithm | HIGH |
| FPCA via FACE (fast covariance estimation) | Sparse FPCA | `refund` (0.1-40), `face` (0.1-8) | In-Scope Algorithm | HIGH |
| FPCA via sparse SVD (fpca.ssvd) | Sparse FPCA | `refund` (0.1-40) | In-Scope Algorithm | HIGH |
| FPCA via local FDA (fpca.lfda) | Sparse FPCA | `refund` (0.1-40) | In-Scope Algorithm | MEDIUM |
| Scree plot (eigenvalue decay visualization) | Visualization | `fdapace` (0.6.0) | Out-of-Scope (plotting) | HIGH |
| Mode of variation plot | Visualization | `fdapace` (0.6.0) | Out-of-Scope (plotting) | HIGH |
| Functional boxplot via FPCA-derived bands | Visualization | `fdapace` (0.6.0) | Out-of-Scope (plotting) | HIGH |

---

## §Design-Goal Filter — Per-Area Summary

The in-scope / out-of-scope classification uses the rule stated once in §Methodology and locked in CONTEXT.md. This table is the INV-02 deliverable.

| Area | In-Scope | Out-of-Scope | Total Rows | Out-of-Scope Breakdown |
|------|----------|-------------|------------|------------------------|
| 1 — Representation / Basis / Smoothing | 38 | 7 | 45 | 4 plotting, 3 IO |
| 2 — Preprocessing / Registration | 22 | 0 | 22 | — |
| 3 — Exploratory / Depth / Outlier | 31 | 7 | 38 | 7 plotting |
| 4 — ML (Regression + Classification + Clustering) | 59 | 2 | 61 | 2 plotting |
| 5 — Inference / Testing | 22 | 3 | 25 | 3 plotting |
| 6 — Functional Time Series | 24 | 2 | 26 | 2 plotting |
| 7 — Density / Object Data / Manifold | 24 | 1 | 25 | 1 plotting |
| 8 — SPM / Control Charts | 10 | 2 | 12 | 2 plotting |
| 9 — FPCA (Sparse / Longitudinal / Specialized) | 18 | 3 | 21 | 3 plotting |
| **TOTAL** | **248** | **27** | **275** | 24 plotting, 3 IO |

**Actionable in-scope capabilities for Phase 17 parity mapping: 248**

> Note: Some capabilities appear in multiple areas where they serve different roles (e.g., FPCA as preprocessing in Area 2 vs. sparse FPCA method in Area 9). The count reflects logical grouping rather than strict deduplication — Phase 17 should match by capability semantics, not row count.

**rainbow package:** All 5+ capabilities are Out-of-Scope (plotting). Package produces rainbow plots, bagplots, and boxplots. No in-scope numeric layer beyond what roahd/fdaoutlier already cover.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Basis systems + smoothing + differential operators | `fda` | `fdaPDE` (FEM), `face` (sparse) | `fda` is canonical R infrastructure; others extend to special domains |
| Functional data object model | `funData` (S4), `tf` (S3) | `fda` (fd class) | Two competing OO layers; `fda` fd-class predates both |
| Registration / alignment | `fdasrvf` (elastic) | `fda` (landmark), `registr` (non-Gaussian) | Each package owns a distinct registration philosophy |
| Depth measures + outlier detection | `roahd`, `fdaoutlier` | `fda.usc` | Overlapping coverage; roahd = robust high-dim, fdaoutlier = detection stats |
| Scalar-on-function regression | `refund` | `fda`, `fda.usc`, `FDboost` | `refund` is the penalized-spline workhorse; `fda` is basis-expansion; `FDboost` is boosting |
| Function-on-scalar regression | `refund` | `FDboost` | `refund` covers penalized and Bayesian; `FDboost` handles boosting |
| Function-on-function regression | `refund` | `FDboost` | Same split |
| Functional time series | `ftsa` | `freqdom`, `fdaACF` | `ftsa` = comprehensive; others = specialist frequency or ACF tools |
| Elastic / shape analysis | `fdasrvf` | `elasdics` | `fdasrvf` = full elastic suite; `elasdics` = sparse 2D curves only |
| Density / object-data statistics | `frechet` | `fdadensity` | `frechet` = metric-space regression; `fdadensity` = LQD-FPCA for densities |
| Multivariate FPCA (different domains) | `MFPCA` + `funData` | — | Tightly coupled; funData provides the S4 data layer |
| Functional clustering | `fdacluster` | `funHDDC`, `funFEM`, `funLBM`, `fdasrvf` | Each covers a distinct clustering paradigm |
| Statistical process monitoring | `funcharts` | — | No other R FDA package covers SPM at this depth |
| Conformal prediction | `conformalInference.fd` | — | Only dedicated conformal package for functional regression |

---

## Packages Considered and Excluded

| Package | Status | Reason for Exclusion |
|---------|--------|----------------------|
| `classiFunc` | ARCHIVED (2020-02-19) | Removed from CRAN; no CRAN-maintained version |
| `FRegSigCom` | ARCHIVED (2020-05-19) | Removed from CRAN; no CRAN-maintained version |
| `fpca` | ARCHIVED (2022-03-06) | Removed from CRAN |
| `rainbow` | Active (3.8) | Included in tables but **all capabilities out-of-scope** (plotting only) |
| `refund.shiny` | Active | Shiny UI package — out-of-scope (visualization/IO) |
| `mlr3fda` | Active | ML-pipeline framework plumbing — out-of-scope (framework scaffolding) |
| `tidyfun` | Active | Primarily data manipulation + ggplot2 visualization wrappers; numeric layer covered by `tf` |
| `warpMix` | ARCHIVED | Archived; functionality superseded by `fdacluster` and `fdasrvf` |
| `sparseFLMM` | Active | Mixed effects for sparse data — capabilities represented via `denseFLMM` + `multifamm` subsetting |

---

## Common Pitfalls for Phase 17 Parity Mapping

### Pitfall 1: One-row-per-function-name inflation
R packages expose multiple S3/S4 method names for the same algorithm (e.g., `fanova.tests`, `fmanova.ptbfr`, `fmanova.trp` in fdANOVA are three functions but only two distinct capability categories). Phase 17 must match by capability semantics.

### Pitfall 2: Overlapping coverage across packages
Many capabilities appear in multiple packages (e.g., functional ANOVA in `fdatest`, `fdANOVA`, `fda.usc`, `fda`). Phase 17 should mark a capability as "present" in fdars if fdars covers the capability regardless of which R package offers it.

### Pitfall 3: Out-of-scope numeric underpinnings
The outliergram MO/MEI statistics are in-scope even though the outliergram plot is not. Similarly, functional boxplot fence/threshold computation is in-scope; the renderer is not. Phase 17 must check fdars for the numeric stat, not the plot.

### Pitfall 4: `fdasrvf` vs fdars elastic alignment breadth
`fdasrvf` implements many more elastic-analysis capabilities than fda.usc or fda. The Phase 17 parity check for elastic area should compare against `fdasrvf` capabilities specifically, not just "registration" generically.

### Pitfall 5: Metric-space / object-data area is new relative to scikit-fda audit
The `frechet` + `fdadensity` area (Area 7) has no close scikit-fda analog. Phase 17 should treat this as a likely large gap area for fdars.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `fdANOVA` implements EWMA/CUSUM analogs — derived from description of "functional control charts" capabilities not explicitly named in CRAN page | Area 8 (SPM) | Low risk — functional EWMA is standard in functional SPM literature; if wrong, row count drops by 2 |
| A2 | `fda.usc` bootstrap CI for mean is based on functional resampling (not just scalar resampling) | Area 3 | Low — documented in web search as functional bootstrap |
| A3 | `funcharts` includes ARL estimation and EWMA/CUSUM-analog methods based on 4 vignettes mentioned on CRAN page | Area 8 | Medium — ARL estimation described as MEDIUM confidence; if wrong, count drops by 2 |
| A4 | `ftsa` GAEVforecast implements extreme value modeling specific to functional data | Area 6 | Medium — if this is just a thin scalar wrapper, row should be reclassified |
| A5 | `sparseFLMM` capabilities are subsumed by `denseFLMM` + `multifamm` for this inventory | Excluded | Low — `sparseFLMM` may add sparse-specific estimation detail not in those two |

---

## Open Questions for Phase 17

1. **Numeric underpinning vs. wrapper:** Several `ftsa` functions (MAF_multivariate, MFDM, Horta-Ziegelmann) wrap external methodology. Phase 17 should confirm whether fdars has equivalents for the specific algorithms, not just the category.

2. **`fdaPDE` finite-element regularization scope:** The PDE-regularized smoothing in `fdaPDE` is a highly specialized capability (2D/3D irregular domains). Phase 17 should classify this as "differentiator" gap if absent from fdars — it is not a table-stakes expectation.

3. **`frechet` metric-space regression:** This is an emerging area (random objects in metric spaces). Phase 17 should check whether fdars has any analogs — preliminary expectation is "absent" across all ~16 frechet in-scope capabilities.

4. **`funLBM` co-clustering:** Phase 17 should check for co-clustering (simultaneous row+column clustering) in fdars — not present in standard functional clustering.

5. **Version staleness:** `fdANOVA` version 0.1.2 dates to 2018-08-29. It may have been superseded by functionality in `fdatest` or newer packages. Phase 17 should flag if any fdANOVA capabilities are now standard in more recent packages.

---

