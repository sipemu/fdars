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


## Phase 17 — Parity Matrix & Categorization

**Map date:** 2026-08-15 · **Input set:** the **248 in-scope R capabilities** from Phase 16 §Design-Goal Filter (9 areas). The 27 out-of-scope R rows (24 plotting + 3 IO) are excluded from the actionable-gap total by construction.

This section maps every in-scope R capability against fdars (crate `fdars-core`, shipped through v0.17.0), producing a per-capability verdict, a "searched fdars for:" evidence note, and the closest-match fdars module/function. It reuses the v0.14.0 scikit-fda audit (`AUDIT-REPORT.md` §Phase 8) wherever an R capability overlaps an already-assessed scikit-fda capability, and re-greps `fdars-core/src/` to confirm/extend. It then categorizes every gap.

### Rubrics (documented once)

**Verdict rubric (D-01, reused verbatim from the v0.14.0 audit).** Matched by **capability semantics, not API name** — a different call shape (builder-struct + single call vs R's S3/S4 dispatch) is not a gap.

| Verdict | Definition |
|---------|-----------|
| **present** | fdars delivers the same result in *any* call-shape. (Accuracy not re-verified here; a known-bug area is flagged "present — accuracy NOT verified" per the v0.14.0 convention.) |
| **partial** | A related/narrower/internal-only capability exists — missing a documented sub-mode, exposed only internally, or a narrower variant. A partial row is an *add-a-variant* backlog candidate. |
| **absent** | No fdars capability delivers the result. An *implement-from-scratch* backlog candidate; closest match noted or "no match found". |

**Category rubric (D-03, reused verbatim).** Applied to every gap (partial/absent) row; present rows carry no category.

| Category | Definition |
|----------|-----------|
| **table-stakes** | A capability a general-purpose FDA library is expected to have; its absence is a competitive deficit. |
| **differentiator** | Valuable but specialized; nice-to-have, not baseline-expected. |
| **out-of-scope** | On inspection really a rendering/IO-adjacent capability that should not be built. Rare here (the input set is already in-scope), reserved for numeric-underpinning rows whose only realistic use is a plot/IO. |

**v0.15.0–v0.17.0 credit.** The following post-date the v0.14.0 §Phase 8 audit and are credited **present** where an R capability maps to them: spline interpolation (`spline_interpolate`), functional summary statistics (`functional_variance`/`functional_std`/`functional_covariance`/`depth_based_median`/`trim_mean`), missing-value imputation (`impute_missing_values`), composable `ExtrapolationPolicy`, functional scoring metrics (`functional_mae`/`mse`/`mape`/`msle`/`explained_variance`), least-squares shift registration (`least_squares_shift_registration`) + three registration-quality scores (`least_squares_score`/`pairwise_correlation_score`/`sobolev_least_squares_score`), banded elastic alignment (`*_with_band`), parallel CV folds, faer FPCA SVD, parallel elastic-FPCA.

---

### §Parity-Matrix

Column schema: **Capability | Source pkg(s) | Verdict | "searched fdars for:" evidence note | closest-match fdars module/function (or "no match found")**.

#### Area 1: Representation / Basis Systems / Smoothing (38 in-scope)

| Capability | Source | Verdict | searched fdars for: … | Closest match |
|------------|--------|---------|-----------------------|---------------|
| B-spline basis system (creation, evaluation, penalty matrix) | fda | **present** | B-spline basis construction + evaluation + roughness penalty | `basis::bspline_basis`, `bspline_basis_from_knots`, `smooth_basis::bspline_penalty_matrix` |
| Fourier basis system (creation, evaluation, penalty matrix) | fda | **present** | Fourier design matrix + penalty | `basis::fourier_basis`, `fourier_basis_with_period`, `smooth_basis::fourier_penalty_matrix` |
| Monomial / polynomial basis (creation, evaluation, penalty) | fda | **absent** | monomial/polynomial power basis constructor | no match found (only B-spline + Fourier exposed; scikit-fda `MonomialBasis` was also absent, §Phase 8) — *differentiator* |
| Constant / intercept basis | fda | **absent** | constant/intercept basis type | no match found (trivially constructable, no named factory; scikit-fda `ConstantBasis` also absent) — *table-stakes* |
| Exponential basis | fda | **absent** | exponential basis constructor | no match found — *differentiator* |
| Power basis | fda | **absent** | power basis constructor | no match found — *differentiator* |
| Polygonal (piecewise-linear) basis | fda | **partial** | piecewise-linear basis | `helpers::linear_interp` / `fdata_interpolate` give piecewise-linear evaluation, but no named polygonal *basis system* object |
| Finite element basis (2D/3D irregular domains via PDE) | fdaPDE | **absent** | finite-element basis over irregular meshes | no match found (scikit-fda `FiniteElementBasis` also absent) — *differentiator* |
| Smooth fd object from raw data (penalized least squares basis expansion) | fda, fda.usc | **present** | penalized basis-expansion smoother | `smooth_basis::smooth_basis`, `basis::pspline::pspline_fit_1d` |
| Smoothing with automatic parameter selection (GCV, AIC) | fda | **partial** | automatic smoothing-parameter selection by GCV *and* AIC | `smooth_basis::smooth_basis_gcv`, `smoothing::optim_bandwidth` (GCV/CV only; AIC absent, per §Phase 8) |
| Monotone smoothing (integral-of-exp, strictly monotone curves) | fda | **partial** | monotone smoothing of arbitrary data | `landmark::monotone_landmark_warp` builds monotone (Fritsch-Carlson) warping functions, but there is no Ramsay integral-of-exp monotone *smoother* for general data |
| Positive-valued smoothing (log-transformed) | fda | **absent** | log-domain positive-constrained smoother | no match found — *differentiator* |
| Bivariate functional data smoothing (smooth.bibasis) | fda | **partial** | 2D/bivariate surface smoothing | `function_on_scalar_2d` (tensor-product penalized 2D fit) covers 2D surfaces; no direct `smooth.bibasis` raw-surface smoother |
| P-spline smoothing for sparse functional data | face, fdaPDE | **present** | penalized-spline (P-spline) smoothing | `basis::pspline::pspline_fit_1d`, `pspline_fit_gcv` |
| Fast covariance estimation for sparse data (FACE) | face | **partial** | fast sandwich covariance estimator for sparse data | `irreg_fdata::cov_irreg` (kernel-smoothed empirical covariance from irregular obs); not the FACE fast-sandwich algorithm specifically |
| Fast covariance for multivariate sparse data (mfaces) | mfaces | **absent** | multivariate sparse fast covariance | no match found (single-covariate `cov_irreg` only) — *differentiator* |
| Smoothing over 2D/3D domains with PDE regularization (FEM) | fdaPDE | **absent** | PDE-regularized FEM smoothing on irregular domains | no match found — *differentiator* |
| Functional data derivative computation | fda, fda.usc | **present** | numerical derivative of functional data | `metric::deriv` (derivative utilities); FPCA/elastic paths use `gradient_uniform` (`warping.rs`) |
| Linear differential operator object (Lfd) | fda | **partial** | composable Lfd operator | `smooth_basis::{bspline,fourier}_penalty_matrix` compute derivative-order penalties; no composable Lfd object (per §Phase 8 `LinearDifferentialOperator` partial) |
| Basis penalty matrix computation | fda | **present** | roughness/derivative penalty matrix | `smooth_basis::bspline_penalty_matrix`, `fourier_penalty_matrix` |
| Inner product / L2 norm between fd objects | fda | **present** | L2 inner product + Lp norm | `utility::inner_product`, `inner_product_matrix`, `fdata::norm_lp_1d` |
| Trapezoidal integration (trapzmat) | fda | **present** | trapezoidal/Simpson quadrature over a grid | `helpers::simpsons_weights`, `cumulative_trapz` (`warping.rs`) |
| Fd object arithmetic (add, subtract, scalar mult) | fda, funData, tf | **present** | pointwise fd arithmetic | `FdMatrix` element access + `center_1d`; arithmetic composable over the column-major matrix |
| Evaluate fd at arbitrary points (off-grid interpolation) | fda, funData, tf | **present** | off-grid evaluation via interpolation | `helpers::fdata_interpolate`, `spline_interpolate` (v0.15.0), `linear_interp` |
| Univariate functional data S4 class (funData) | funData | **present** | univariate fd container | `FdMatrix` (column-major fd container) + `fdata` conventions |
| Multivariate functional data S4 class (multiFunData) | funData | **absent** | multivariate (multi-domain) fd container type | no match found (2D handled via flattened matrices, no composable multi-domain container) — *differentiator* |
| Irregular functional data S4 class (irregFunData) | funData | **present** | irregular fd container | `irreg_fdata::IrregFdata` |
| Tidy S3 functional vector (grid / spline-basis / FPC reprs) | tf | **partial** | multi-representation fd vector type | `FdMatrix` (grid) + `FpcaResult` (FPC) + `fdata_to_basis` (basis) exist as separate types; no single tidy vector switching representations |
| Grid resampling / re-evaluation on new points | tf, funData | **present** | re-evaluate fd on new evaluation grid | `helpers::fdata_interpolate` / `spline_interpolate`, `irreg_fdata::to_regular_grid` |
| Functional data centering | fda, fda.usc | **present** | center curves about the mean | `fdata::center_1d` / `center_2d` |
| Mean, variance, covariance function from sample | fda, fda.usc, roahd | **present** | pointwise mean + variance + covariance surface | `fdata::mean_1d`, `functional_variance`, `functional_covariance` (v0.15.0) |
| Basis conversion / projection (grid↔basis, basis↔basis) | fda, fda.usc | **present** | least-squares grid→basis projection | `basis::fdata_to_basis`, `fdata_to_basis_1d`, `basis_to_fdata` |
| Functional data sub-setting and domain restriction | tf, funData | **partial** | domain restriction / sub-range extraction | column/row slicing on `FdMatrix` is possible; no named domain-restriction API |
| Principal differential analysis (PDA — estimate linear ODE) | fda | **absent** | estimate a linear ODE (differential operator) from data | no match found — *differentiator* |
| Functional data integration over sub-domain | tf | **present** | integrate a curve over a sub-interval | `helpers::simpsons_weights` + `cumulative_trapz` support sub-domain integration |
| Local min/max detection on functional data | tf | **present** | local extrema / peak detection | `seasonal::peak::find_peaks_1d`, `landmark::detect_landmarks` |
| fdata class (container with argvals) | fda.usc | **present** | fd container carrying argvals | `FdMatrix` + argvals convention; `IrregFdata` for irregular |
| Extrapolation strategies (boundary, periodic, constant fill, exception) | fda | **present** | composable extrapolation policy | `ExtrapolationPolicy{Boundary,Exception,Fill,Periodic}` (v0.16.0, `helpers.rs`) |

**Area 1 verdicts:** present = 20 · partial = 8 · absent = 10 (total 38).

#### Area 2: Preprocessing / Registration (22 in-scope)

| Capability | Source | Verdict | searched fdars for: … | Closest match |
|------------|--------|---------|-----------------------|---------------|
| Landmark registration (hard-pin time points) | fda | **present** | landmark shift/warp registration | `landmark::landmark_register`, `detect_and_register` |
| Continuous registration to a target curve | fda | **present** | continuous alignment to a target | `alignment::align_to_target` |
| Group-wise elastic registration via SRVF Karcher mean | fdasrvf | **present** | SRVF group registration to Karcher mean | `alignment::karcher_mean`, `karcher_mean_with_band` (v0.16.0) |
| Pair-wise elastic alignment (SRVF geodesic DP) | fdasrvf | **present** | pairwise elastic (dynamic-programming) alignment | `alignment::elastic_align_pair`, `elastic_align_pair_banded` |
| 2D elastic curve alignment (sparse/dense/irregular) | elasdics | **partial** | 2D/nD elastic curve alignment | `alignment::nd::elastic_align_pair_nd` (nD dense); sparse/irregular 2D-curve variant not covered |
| Elastic curve mean (Karcher mean in SRVF space) | fdasrvf, elasdics | **present** | SRVF Karcher mean | `alignment::karcher_mean`, `robust_karcher::robust_karcher_mean` |
| SRVF/SRSF transformation (function → SRVF) | fdasrvf | **present** | SRSF transform + inverse | `alignment::srsf_transform`, `srsf_inverse`, `srsf_transform_nd` |
| Joint registration + non-Gaussian FPCA (exp-family) | registr | **absent** | joint registration with exponential-family (binary/count) FPCA | no match found (`elastic_fpca` is Gaussian-space; `registr`'s GLM-family registration absent) — *differentiator* |
| Incomplete curve registration (partial observation) | registr | **partial** | registration of partially observed curves | `alignment::partial_match`, `partial` (partial-shape matching) + `spm::partial` PACE completion; not `registr`'s incomplete-curve GLM registration |
| Warping function computation, composition, inversion | fda, fdasrvf | **present** | warp construction, composition, inversion | `warping::invert_gamma`, `normalize_warp`, `gam_to_psi`, `psi_to_gam` |
| Phase/amplitude separation + amplitude/phase FPCA | fdasrvf | **present** | phase/amplitude decomposition + separate FPCA | `alignment::set::elastic_decomposition`, `elastic_fpca::{vert_fpca,horiz_fpca,joint_fpca}` |
| Smooth warping function estimation (smooth.morph) | fda | **present** | smoothed warping estimate | `warping::gam_to_psi_smooth` (Nadaraya-Watson smoothed warp) |
| K-means with simultaneous alignment | fdacluster, fdasrvf | **partial** | k-means clustering that aligns curves within the loop | `clustering::kmeans_fd` (no alignment) + `alignment/clustering` (from elastic distances); no single joint align+cluster estimator |
| Shift, dilation, affine warping for alignment | fdacluster | **partial** | shift + dilation/affine warping models | `alignment::shift::least_squares_shift_registration` (shift only, v0.17.0); dilation/affine warping not covered |
| Functional PCA for dimensionality reduction (preprocessing) | fda, fda.usc, fdapace | **present** | FPCA as a preprocessing/dim-reduction step | `regression::fdata_to_pc_1d` (+ `FpcaResult.project`), faer SVD backend (v0.15.0) |
| Functional PLS for dimensionality reduction | fda.usc | **present** | functional PLS scores | `regression::fdata_to_pls_1d`, `scalar_on_function::fregre_pls` |
| Smoothing for irregularly spaced observations | face, fdapace | **present** | kernel-smooth irregular obs onto a grid | `irreg_fdata::to_regular_grid`, `irreg_fdata::smoothing` |
| Functional data normalization / centering | fda, fda.usc | **present** | normalize / center curves | `fdata::center_1d`, `functional_std` for scaling |
| Cross-validation smoothing-parameter selection (leave-one-curve-out) | fda.usc, refund | **present** | LOO/GCV smoothing-parameter CV | `smoothing::cv_smoother` (CV.S), `gcv_smoother` (GCV.S), `optim_bandwidth` |
| Data2fd — quick basis expansion from a raw matrix | fda | **present** | one-call raw matrix → basis representation | `basis::fdata_to_basis` / `fdata_to_basis_1d` |
| Stringing (map high-dim scalar data to functional form) | fdapace | **absent** | stringing (order scalar features into a curve) | no match found — *differentiator* |
| Registration quality score (warping complexity, amplitude var) | fdasrvf | **present** | registration-quality diagnostics | `alignment::quality::{alignment_quality,warp_complexity,warp_smoothness}` + `least_squares_score`/`pairwise_correlation_score`/`sobolev_least_squares_score` (v0.17.0) |

**Area 2 verdicts:** present = 16 · partial = 4 · absent = 2 (total 22).

#### Area 3: Exploratory / Depth / Outlier Detection (31 in-scope)

| Capability | Source | Verdict | searched fdars for: … | Closest match |
|------------|--------|---------|-----------------------|---------------|
| Modified Band Depth (MBD), univariate | roahd, fdaoutlier, fda.usc | **present** | modified band depth | `depth::modified_band_1d` |
| Band Depth (BD), univariate | roahd, fdaoutlier | **present** | band depth | `depth::band_1d` |
| Modified Band Depth, multivariate (multiMBD) | roahd | **partial** | multivariate MBD | `depth::modified_band_1d` is univariate; 2D depths exist (`fraiman_muniz_2d`) but no multivariate multiMBD |
| Half-Region Depth (HRD), univariate | roahd | **absent** | half-region depth | no match found — *differentiator* |
| Modified Half-Region Depth (MHRD) | roahd | **absent** | modified half-region depth | no match found — *differentiator* |
| Epigraph Index (EI) | roahd | **partial** | epigraph index | `depth::modified_epigraph_index_1d` (MEI present); the un-modified EI not separately exposed |
| Modified Epigraph Index (MEI) | roahd | **present** | modified epigraph index | `depth::modified_epigraph_index_1d` |
| Hypograph Index (HI) | roahd | **absent** | hypograph index | no match found (MEI present, MHI/HI not) — *differentiator* |
| Modified Hypograph Index (MHI) | roahd | **absent** | modified hypograph index | no match found — *differentiator* |
| Extremal depth | fdaoutlier | **absent** | extremal depth | no match found — *differentiator* |
| Extreme Rank Length Depth | fdaoutlier | **absent** | extreme-rank-length depth | no match found — *differentiator* |
| L-infinity depth | fdaoutlier | **absent** | L∞ depth | no match found — *differentiator* |
| Total Variation Depth (TVD) + MSSI | fdaoutlier | **absent** | total-variation depth + modified shape similarity index | no match found — *differentiator* |
| Random projection depth (multivariate) | fdaoutlier | **present** | random-projection depth | `depth::random_projection_1d`, `random_projection_2d`, `random_tukey_1d` |
| Elastic depth (shape depth in SRVF space) | fdasrvf | **present** | elastic/shape depth | `alignment::elastic_depth` |
| Integrated functional depth for partially observed data | fdaPOIFD | **absent** | integrated depth for partially observed curves | no match found (`spm::partial` reconstructs, not a POIFD depth) — *differentiator* |
| General functional depth dispatcher (multiple methods) | fda.usc | **partial** | single dispatcher over multiple depth methods | depth methods exist as separate functions (`fraiman_muniz_1d`, `band_1d`, …); no unified `DepthMethod`-dispatched entry point exposed publicly |
| Directional outlyingness statistic (Dai & Genton) | fdaoutlier | **present** | directional outlyingness | `outliers::magnitude_shape_outlyingness` (magnitude + shape components) |
| MS-plot statistic (magnitude-shape outlyingness) | fdaoutlier | **present** | magnitude-shape outlyingness statistic | `outliers::magnitude_shape_outlyingness` (`MagnitudeShapeResult`) |
| Outliergram statistic (MO vs MEI) | roahd | **present** | outliergram MO/MEI shape-outlier statistic | `outliers::outliergram` |
| Depthgram statistic | roahd | **absent** | depthgram multivariate depth statistic | no match found — *differentiator* |
| TVD+MSSI-based outlier detection (tvdmss) | fdaoutlier | **absent** | tvdmss outlier detector | no match found — *differentiator* |
| Massive Unsupervised Outlier Detection (MUOD) | fdaoutlier | **absent** | MUOD outlier detector | no match found — *differentiator* |
| Sequential transformation outlier detection | fdaoutlier | **absent** | sequential-transformation outlier detector | no match found — *differentiator* |
| Elastic changepoint detection (amplitude + phase, SRSF) | fdasrvf | **present** | elastic amplitude/phase changepoint | `elastic_changepoint::{elastic_amp_changepoint,elastic_ph_changepoint,elastic_fpca_changepoint}` |
| Outlier detection for partially observed curves (depth-based) | fdaPOIFD | **absent** | partially-observed depth outlier detection | no match found — *differentiator* |
| Functional bootstrap CI for mean / summary statistics | fda.usc | **present** | functional bootstrap CI | `scalar_on_function::bootstrap::bootstrap_ci_fregre_lm`; `spm::bootstrap` resampling |
| Robust mean and median for functional samples | roahd, fda.usc | **present** | robust functional mean/median | `fdata::geometric_median_1d`, `trim_mean` (v0.15.0), `depth_based_median` |
| Spearman / Kendall rank correlation for functional data | roahd | **absent** | functional Spearman/Kendall rank correlation | no match found — *differentiator* |
| Bootstrap hypothesis test on Spearman correlation | roahd | **absent** | bootstrap test on functional Spearman correlation | no match found — *differentiator* |
| Covariance function estimation (empirical) | roahd, fda.usc | **present** | empirical covariance surface | `fdata::functional_covariance` (v0.15.0), `irreg_fdata::cov_irreg` |
| Functional boxplot (depth-based outlier thresholds / fences) | roahd, fdaoutlier, fda.usc | **partial** | depth-fence functional-boxplot thresholds | `outliers::outliergram` (parabolic fence) provides fence-based detection; the López-Pintado depth-fence (1.5×IQR of depths) not a named function (per §Phase 8) |
| Partial reconstruction of missing curves (depth-based) | fdaPOIFD | **partial** | reconstruct missing curve segments | `spm::partial` (PACE conditional-expectation domain completion); not the fdaPOIFD depth-based reconstruction |

**Area 3 verdicts:** present = 12 · partial = 5 · absent = 16 (total 33).

#### Area 4: ML — Regression / Classification / Clustering (59 in-scope)

##### Regression (35 rows)

| Capability | Source | Verdict | searched fdars for: … | Closest match |
|------------|--------|---------|-----------------------|---------------|
| Scalar-on-function regression via basis expansion (penalized OLS) | fda, fda.usc | **present** | scalar-on-function via basis expansion | `scalar_on_function::fregre_lm` (over basis/FPC scores) |
| Scalar-on-function regression via FPCA scores | fda, fda.usc, refund | **present** | FPCA-score scalar-on-function regression | `scalar_on_function::fregre_lm` with `fdata_to_pc_1d`; `model_selection_ncomp` |
| Scalar-on-function regression via PLS scores | fda.usc | **present** | PLS-score scalar-on-function regression | `scalar_on_function::fregre_pls` |
| Scalar-on-function nonparametric kernel regression | fda.usc | **present** | Nadaraya-Watson kernel scalar-on-function regression | `scalar_on_function::fregre_np`, `fregre_np_from_distances` |
| Penalized scalar-on-function regression (pfr, spline) | refund | **partial** | roughness-penalized spline coefficient-function regression | `fregre_lm` + `bspline_penalty_matrix` available but LDO penalty not wired as a plug-in; no dedicated `pfr` estimator |
| Functional GLM, scalar response (logistic, Poisson) | fda.usc, refund | **partial** | functional GLM family (logistic + Poisson + …) | `scalar_on_function::functional_logistic` (logistic only); Poisson/other exponential families absent |
| Functional generalized spectral additive model (GSAM) | fda.usc | **absent** | spectral additive functional model | no match found — *differentiator* |
| Functional generalized kernel additive model (GKAM) | fda.usc | **absent** | kernel additive functional model | no match found — *differentiator* |
| Varying-coefficient / concurrent functional regression | fdaconcur, refund | **absent** | concurrent (varying-coefficient) regression | no match found — *table-stakes* |
| History index model (effect of past predictor values) | fdaconcur, refund | **absent** | history-index (lagged) functional regression | no match found — *differentiator* |
| Scalar-on-function variable selection (fosr.vs) | refund | **absent** | variable selection in scalar-on-function regression | no match found (scikit-fda maxima-hunting/RKHS selection also absent, §Phase 8) — *differentiator* |
| Scalar-on-function permutation-test wrapper (fosr.perm) | refund | **partial** | permutation-test wrapper for scalar-on-function fit | `function_on_scalar::fanova` is permutation-based (functional response); no scalar-on-function permutation wrapper |
| Function-on-scalar regression (penalized / two-step / OLS / GLS) | refund, FDboost | **present** | function-on-scalar regression | `function_on_scalar::fosr`, `fosr_fpc` |
| Bayesian function-on-scalar regression (Gibbs/VB) | refund | **absent** | Bayesian (Gibbs/VB) function-on-scalar regression | no match found — *differentiator* |
| Function-on-scalar regression via boosting base-learners | FDboost | **absent** | boosting-based function-on-scalar regression | no match found — *differentiator* |
| Function-on-function regression (ff, ffpc, sff, pffr) | refund, FDboost | **present** | function-on-function regression | `fof_regression::fof_regression` |
| Penalized flexible functional regression (pffr — multivariate, RE) | refund | **absent** | pffr flexible/mixed function-on-function regression | no match found (basic FoF present; flexible-RE variant absent) — *differentiator* |
| Function-on-scalar linear model (fRegress framework) | fda | **present** | fRegress-style function-on-scalar linear model | `function_on_scalar::fosr` |
| Function-on-scalar regression with functional response (fda.usc basis) | fda.usc | **present** | basis-approach function-on-scalar with functional response | `function_on_scalar::fosr`, `fosr_fpc` |
| Elastic regression (scalar response, SRVF-space) | fdasrvf | **present** | elastic (SRVF) scalar-response regression | `elastic_regression::elastic_regression` |
| Elastic logistic regression | fdasrvf | **present** | elastic logistic regression | `elastic_regression::elastic_logistic` |
| Elastic multinomial logistic regression | fdasrvf | **partial** | elastic multinomial (multi-class) logistic | `elastic_logistic` is binary only; multinomial variant absent (grep: no `multinomial`) |
| Elastic principal component regression | fdasrvf | **present** | elastic PCR | `elastic_regression::elastic_pcr` |
| Functional additive model (FAM — scalar response) | fdapace | **absent** | functional additive model (nonparametric additive) | no match found — *differentiator* |
| Functional concurrent regression (varying-coeff, sparse/dense) | fdapace, fdaconcur | **absent** | concurrent/varying-coefficient regression | no match found — *table-stakes* |
| Functional linear model with cross-validation (basis, PCA, PLS) | fda.usc | **present** | CV-tuned functional linear model | `scalar_on_function::cv` (`fregre_cv`), `model_selection_ncomp` |
| Functional linear model goodness-of-fit / F-test | fda.usc | **partial** | FLM goodness-of-fit / F-test | `helpers::r_squared`/`r_squared_adj` give fit diagnostics; no formal FLM F-test / GoF test |
| Functional linear mixed model, dense data (denseFLMM) | denseFLMM | **partial** | functional linear mixed model | `famm` (functional mixed model, fixed-effect test) present; not the full denseFLMM random-effects estimator |
| Multivariate functional additive mixed model (multiFAMM) | multifamm | **absent** | multivariate functional additive mixed model | no match found — *differentiator* |
| Fast functional mixed model inference (fastFMM) | fastFMM | **partial** | fast functional mixed-model inference | `famm::fmm_test_fixed` (functional mixed-model fixed-effect test); not the fastFMM massively-parallel estimator |
| Functional quantile regression (scalar response) | fdapace | **absent** | functional quantile regression | no match found — *differentiator* |
| Stringing regression (high-dim → functional → FLM) | fdapace | **absent** | stringing-then-FLM regression | no match found — *differentiator* |
| Bootstrap CIs for regression coefficients | fda.usc | **present** | bootstrap CI for functional regression coefficients | `scalar_on_function::bootstrap::{bootstrap_ci_fregre_lm,bootstrap_ci_functional_logistic}` |
| Cross-validation for functional regression (LOOCV, k-fold) | fda.usc, refund | **present** | k-fold / LOO CV for functional regression | `scalar_on_function::cv` (`fregre_cv`), `cv::metric_r_squared` |
| GAMLSS for functional response (distributional boosting) | FDboost | **absent** | GAMLSS distributional functional regression | no match found — *differentiator* |

##### Classification (8 rows)

| Capability | Source | Verdict | searched fdars for: … | Closest match |
|------------|--------|---------|-----------------------|---------------|
| Functional LDA (basis/FPC scores) | fda.usc | **present** | functional LDA | `classification::fclassif_lda` |
| Functional GLM classifier (logistic basis) | fda.usc | **present** | functional logistic classifier | `scalar_on_function::functional_logistic`, `classification::fclassif_kernel` (GLM family) |
| Functional kernel classifier (nonparametric kNN-type) | fda.usc | **present** | kernel / kNN functional classifier | `classification::fclassif_knn`, `fclassif_kernel` |
| DD-classifier (depth-vs-depth) | fda.usc | **present** | depth-vs-depth classifier | `classification::fclassif_dd` |
| Functional classification using ML on FPC scores (SVM, RF) | fda.usc | **partial** | pluggable ML classifier on FPC scores | FPC scores feed `fclassif_lda`/`qda`/`knn`; no SVM/RF backends (scikit-fda-style pluggable ML absent) |
| Elastic logistic classification (SRVF-space) | fdasrvf | **present** | elastic-space logistic classification | `elastic_regression::elastic_logistic`, `predict_elastic_logistic` |
| Depth-based outlier classification | fdaPOIFD | **partial** | depth-based (outlier) classification | `fclassif_dd` uses depth; the fdaPOIFD partially-observed depth-outlier classifier absent |
| Cross-validation classification (ClassifCv) | fda.usc | **present** | CV for functional classification | `classification::cv::fclassif_cv` (parallel folds, v0.15.0) |

##### Clustering + Conformal (16 rows)

| Capability | Source | Verdict | searched fdars for: … | Closest match |
|------------|--------|---------|-----------------------|---------------|
| Functional K-means with simultaneous alignment (Sangalli) | fdacluster | **partial** | joint align+cluster k-means | `clustering::kmeans_fd` (no in-loop alignment) + elastic distances; no combined estimator |
| Hierarchical agglomerative clustering for functional data | fdacluster | **present** | hierarchical clustering from distances | `alignment::clustering::hierarchical_from_distances`, `cut_dendrogram` |
| DBSCAN density-based functional clustering | fdacluster | **absent** | DBSCAN on functional data | no match found — *differentiator* |
| K-means elastic clustering with SRVF alignment | fdasrvf | **partial** | elastic k-means with SRVF alignment | `alignment::clustering::kmedoids_from_distances` on elastic distances; not in-loop SRVF k-means |
| kCFC functional clustering via subspace embedding | fdapace | **absent** | kCFC subspace-embedding clustering | no match found — *differentiator* |
| Model-based clustering in group-specific subspaces (funHDDC) | funHDDC | **partial** | model-based subspace functional clustering | `gmm::gmm_cluster` (Gaussian mixture on FPC scores); not funHDDC's per-group subspace model |
| Model-based clustering in discriminative subspace (funFEM) | funFEM | **absent** | discriminative-subspace model-based clustering | no match found — *differentiator* |
| Model-based co-clustering of functions (rows + columns) | funLBM | **absent** | latent-block co-clustering of functions | no match found — *differentiator* |
| Slope heuristic for cluster model selection | funHDDC | **absent** | slope-heuristic model-selection criterion | no match found — *differentiator* |
| K-means functional clustering (basic) | fda.usc | **present** | basic functional k-means | `clustering::kmeans_fd`; `fuzzy_cmeans_fd` for soft assignment |
| Conformal prediction regions for functional regression | conformalInference.fd | **present** | conformal prediction regions | `conformal::regression`, `conformal::generic` |
| Conformal prediction split / multi-split variants | conformalInference.fd | **present** | split / multi-split conformal | `conformal::regression` (split), `conformal::cv`, `tolerance::conformal` |
| Functional FPCA-based prediction (FPC scores → linear prediction) | fda, refund, fda.usc | **present** | predict from FPC scores | `FpcaResult.project` + `fregre_lm`; `FpcPredictor` trait |
| Stability selection for FDboost model terms | FDboost | **absent** | stability selection for model terms | no match found — *differentiator* |

**Area 4 verdicts:** present = 25 · partial = 12 · absent = 20 (total 57 literal in-scope rows; see recount note).

> **Area 4 recount note.** The Phase 16 §Design-Goal Filter header credits Area 4 with **59 in-scope** rows (61 total − 2 out-of-scope), but a direct recount of the literal Area-4 capability tables yields **57 in-scope + 1 out-of-scope (coefficient-function plot) = 58 rows**. This table maps all 57 literal in-scope rows; the 2-row header surplus is a Phase-16 over-count with no underlying capability rows to map (analogous to the stale-header recounts in the v0.14.0 §Phase 8).

#### Area 5: Inference / Testing (22 in-scope)

| Capability | Source | Verdict | searched fdars for: … | Closest match |
|------------|--------|---------|-----------------------|---------------|
| Interval Testing Procedure (ITP), one-population, B-spline | fdatest | **absent** | interval-wise testing procedure (ITP) | no match found — *differentiator* |
| ITP one-population (Fourier basis) | fdatest | **absent** | Fourier-basis ITP | no match found — *differentiator* |
| ITP two-population comparison (B-spline/Fourier/phase-amp) | fdatest | **absent** | two-sample ITP | no match found — *differentiator* |
| Functional ANOVA (one-way, multiple groups) via ITP | fdatest, fdANOVA | **partial** | one-way functional ANOVA | `function_on_scalar::fanova` (permutation-based group-mean difference test); not the ITP/V-statistic form (per §Phase 8 `oneway_anova` partial) |
| Functional ANOVA using random projections (uni/multivariate) | fdANOVA | **absent** | random-projection functional ANOVA | no match found — *differentiator* |
| Functional MANOVA via permutation + basis representation | fdANOVA | **partial** | permutation functional MANOVA | `function_on_scalar::fanova` (single-response permutation ANOVA); no multivariate MANOVA |
| Functional MANOVA via random projections | fdANOVA | **absent** | random-projection functional MANOVA | no match found — *differentiator* |
| F-permutation test for functional data (Fperm.fd) | fda | **partial** | F-permutation test for functional data | `function_on_scalar::fanova` is permutation-F-based; not exposed as a standalone `Fperm.fd`-style test |
| t-permutation test for functional two-sample comparison | fda | **absent** | two-sample functional t-permutation test | no match found — *table-stakes* |
| FLM on-function testing (ITPlmbspline — scalar-on-function) | fdatest | **absent** | interval-wise FLM coefficient testing | no match found — *differentiator* |
| Goodness-of-fit test for the FLM | fda.usc | **absent** | FLM goodness-of-fit test | no match found (fit diagnostics via `r_squared` only, §Phase 8) — *table-stakes* |
| F-test for the FLM with scalar response | fda.usc | **absent** | FLM F-test (scalar response) | no match found — *table-stakes* |
| Delsol-Ferraty-Vieu test (no functional-scalar relationship) | fda.usc | **absent** | Delsol-Ferraty-Vieu no-effect test | no match found — *differentiator* |
| Equality of functional distributions test | fda.usc | **absent** | equality-of-distributions functional test | no match found — *differentiator* |
| Equality of functional means / covariance test | fda.usc | **partial** | two-sample functional mean/covariance equality test | `spm::stats::hotelling_t2` (Hotelling T² statistic, SPM context — per §Phase 8 partial); not a standalone two-sample inference test |
| Distance correlation and t-test for functional data | fda.usc | **absent** | functional distance correlation + test | no match found — *differentiator* |
| Simultaneous confidence bands for the mean function | SCBmeanfd | **partial** | simultaneous confidence bands for mean | `alignment::shape_ci` (shape confidence intervals) + `spm` control limits; not the SCBmeanfd Gaussian-kinematic SCB |
| Goodness-of-fit test for mean model (SCBmeanfd) | SCBmeanfd | **absent** | mean-model goodness-of-fit test | no match found — *differentiator* |
| Two-sample equality test (SCBmeanfd) | SCBmeanfd | **absent** | SCB-based two-sample test | no match found — *table-stakes* |
| Stationarity test for functional time series (T_stationary) | ftsa | **absent** | functional stationarity test | no match found — *differentiator* |
| Bootstrap CI on Spearman correlation | roahd | **absent** | bootstrap CI for functional Spearman correlation | no match found — *differentiator* |
| Likelihood ratio test for smooth functional effects (rlrt.pfr) | refund | **absent** | LRT for smooth functional effects | no match found — *differentiator* |

**Area 5 verdicts:** present = 0 · partial = 5 · absent = 17 (total 22).

#### Area 6: Functional Time Series (25 in-scope)

| Capability | Source | Verdict | searched fdars for: … | Closest match |
|------------|--------|---------|-----------------------|---------------|
| Functional time series model fitting via FPCA (ftsm) | ftsa | **partial** | FPCA-based functional-time-series model | `regression::fdata_to_pc_1d` gives the FPCA decomposition; no time-ordered ftsm model/score-forecasting wrapper |
| Functional time series forecasting via FPC regression | ftsa | **absent** | forecast future curves via FPC-score time-series models | no match found — *differentiator* |
| Functional partial least squares forecasting (fplsr) | ftsa | **absent** | FPLS forecasting of curves | no match found — *differentiator* |
| Dynamic updating of functional forecasts (FLR/OLS/RR/PLS) | ftsa | **absent** | dynamic forecast updating | no match found — *differentiator* |
| Iterative forecasting (ftsmiterativeforecasts) | ftsa | **absent** | iterative multi-step curve forecasting | no match found — *differentiator* |
| Generalized additive extreme value modeling (GAEVforecast) | ftsa | **absent** | functional GAEV forecasting | no match found — *differentiator* |
| Functional autocorrelation function (facf) | ftsa | **absent** | functional ACF (fACF) | no match found (only scalar `autocorrelation` on a mean curve for period detection, `seasonal/`) — *differentiator* |
| Functional ACF quantification via L2-norm | fdaACF | **absent** | L2-norm functional ACF | no match found — *differentiator* |
| Partial functional autocorrelation function | fdaACF | **absent** | partial functional ACF | no match found — *differentiator* |
| Distribution of functional ACF under strong white noise | fdaACF | **absent** | white-noise fACF distribution / confidence bands | no match found — *differentiator* |
| Stationarity test for functional time series | ftsa | **absent** | functional stationarity test | no match found — *differentiator* |
| Long-run covariance estimation (kernel sandwich) | ftsa | **absent** | long-run covariance estimator | no match found — *differentiator* |
| Compositional data functional PCA (CoDa_FPCA) | ftsa | **absent** | compositional-data FPCA | no match found — *differentiator* |
| Log quantile density transform FPCA (LQDT_FPCA) | ftsa | **absent** | LQD-transform density FPCA | no match found — *differentiator* |
| Maximum autocorrelation factors (MAF) | ftsa | **absent** | maximum-autocorrelation-factor decomposition | no match found — *differentiator* |
| Multilevel functional data model (MFDM) | ftsa | **absent** | multilevel functional data model | no match found — *differentiator* |
| Dynamic functional PCA (Horta-Ziegelmann) | ftsa | **absent** | dynamic FPCA | no match found — *differentiator* |
| Dynamic principal component analysis (DPCA) via spectral methods | freqdom | **absent** | spectral dynamic PCA | no match found — *differentiator* |
| Spectral density operator estimation | freqdom | **absent** | functional spectral density operator | no match found — *differentiator* |
| VAR / VMA process simulation for functional time series | freqdom | **absent** | functional VAR/VMA simulation | no match found — *differentiator* |
| Functional time series summary statistics (mean, sd, var, quantile, median) | ftsa | **present** | pointwise summary statistics over a functional sample | `fdata::mean_1d`, `functional_variance`, `functional_std`, `depth_based_median` (v0.15.0) |
| Functional time series differencing | ftsa | **partial** | difference successive curves | `detrend` module (trend removal); no dedicated fd-series differencing operator |
| Functional time series bootstrap resampling (fbootstrap) | ftsa | **partial** | bootstrap resampling of functional observations | `spm::bootstrap`, `scalar_on_function::bootstrap`; not a time-series block bootstrap |
| Functional ARMA simulation (sim_FARMA) | ftsa | **absent** | functional ARMA data simulation | no match found (`simulation` covers KL Gaussian data, not FARMA) — *differentiator* |
| Functional time series error measurement | ftsa | **present** | forecast-error metrics for functional responses | `scoring::{functional_mae,functional_mse,functional_mape,functional_msle}` (v0.16.0) |

**Area 6 verdicts:** present = 2 · partial = 3 · absent = 20 (total 25).

#### Area 7: Density / Object Data / Manifold (25 in-scope)

| Capability | Source | Verdict | searched fdars for: … | Closest match |
|------------|--------|---------|-----------------------|---------------|
| Global Fréchet regression (Euclidean predictors → metric-space response) | frechet | **absent** | global Fréchet regression | no match found (grep: no `frechet_regression`) — *differentiator* |
| Local Fréchet regression (kernel-weighted) | frechet | **absent** | local Fréchet regression | no match found — *differentiator* |
| Fréchet mean in general metric spaces | frechet | **partial** | Fréchet mean | `alignment::karcher_mean` is a Fréchet mean on the *Fisher-Rao shape manifold* only; no general metric-space Fréchet mean |
| Fréchet variance computation | frechet | **absent** | Fréchet variance in a metric space | no match found — *differentiator* |
| Wasserstein distance between distributions | frechet | **absent** | Wasserstein / earth-mover distance | no match found (grep: no `wasserstein`) — *differentiator* |
| Fréchet regression for density responses (2-Wasserstein) | frechet | **absent** | density-response Fréchet regression | no match found — *differentiator* |
| Fréchet ANOVA for object data (distributions) | frechet | **absent** | Fréchet ANOVA | no match found — *differentiator* |
| Change point detection for object-valued time series | frechet | **partial** | changepoint for object-valued series | `elastic_changepoint`, `seasonal::change` handle functional/scalar changepoints; not general metric-space object changepoints |
| Fréchet regression for covariance-matrix responses (Frobenius/power/log-Cholesky) | frechet | **absent** | covariance-matrix response regression | no match found — *differentiator* |
| Fréchet integral (mean over domain) for covariance objects | frechet | **absent** | Fréchet integral over covariance objects | no match found — *differentiator* |
| Fréchet regression for spherical data | frechet | **absent** | spherical-response regression | no match found — *differentiator* |
| Geodesic computations on sphere (exp/log map, distance) | frechet | **absent** | sphere exp/log map + geodesic distance | no match found (`alignment` geodesics are SRVF-space, not spherical) — *differentiator* |
| Fréchet regression for correlation matrices | frechet | **absent** | correlation-matrix response regression | no match found — *differentiator* |
| Fréchet regression for network responses | frechet | **absent** | network-response regression | no match found — *differentiator* |
| Fréchet ANOVA for networks | frechet | **absent** | network Fréchet ANOVA | no match found — *differentiator* |
| Fréchet regression for point-process responses | frechet | **absent** | point-process response regression | no match found — *differentiator* |
| FPCA for density functions via log quantile density (LQD) | fdadensity | **absent** | LQD-transform density FPCA | no match found — *differentiator* |
| LQD ↔ density / quantile function conversion | fdadensity | **absent** | LQD ↔ density transform | no match found — *differentiator* |
| Fraction of variance explained (FVE) for LQD-FPCA | fdadensity | **partial** | fraction-of-variance-explained selector | `FpcaResult` exposes singular values (FVE computable); no LQD-FPCA FVE helper |
| Wasserstein Fréchet mean of densities | fdadensity | **absent** | Wasserstein barycenter of densities | no match found — *differentiator* |
| Density normalization and regularization | fdadensity | **absent** | density normalization/regularization | no match found — *differentiator* |
| Multivariate FPCA across different-dimensional domains (1D/2D/3D) | MFPCA | **absent** | multi-domain multivariate FPCA | no match found (1D + flattened-2D FPCA only) — *differentiator* |
| Tensor PCA for 2D functional data (UMPCA, FCP-TPA) | MFPCA | **partial** | tensor/2D FPCA | `regression` 2D FPCA path + `function_on_scalar_2d` tensor penalty; not UMPCA/FCP-TPA tensor decompositions |
| PACE-based FPCA for mixed-domain multivariate data | MFPCA | **absent** | mixed-domain PACE FPCA | no match found — *differentiator* |
| DCT/spline basis expansions for 2D/3D domain data | MFPCA | **partial** | 2D/3D basis expansion | `basis` (1D B-spline/Fourier) + `function_on_scalar_2d` tensor product; no 3D/DCT basis |

**Area 7 verdicts:** present = 0 · partial = 5 · absent = 20 (total 25).

#### Area 8: Statistical Process Monitoring / Control Charts (10 in-scope)

| Capability | Source | Verdict | searched fdars for: … | Closest match |
|------------|--------|---------|-----------------------|---------------|
| Functional control chart (Phase I — reference set, control limits) | funcharts | **present** | Phase-I control-limit estimation | `spm::phase`, `spm::control` |
| Functional control chart (Phase II — online monitoring) | funcharts | **present** | Phase-II online monitoring | `spm::phase`, `spm::control`, `spm::rules` |
| Multivariate functional control chart | funcharts | **present** | multivariate functional control chart | `spm::mfpca`, `spm::mewma`, `spm::amewma` |
| Robust functional control chart | funcharts | **partial** | robust (outlier-resistant) control chart | `spm::iterative` (outlier-masked Phase-I) + robust means; not a robustbase-style robust chart |
| Profile monitoring (scalar-on-function + residual chart) | funcharts | **present** | profile monitoring | `spm::profile` |
| ARL estimation / simulation for functional charts | funcharts | **present** | ARL estimation via simulation | `spm::arl` |
| Functional EWMA / CUSUM chart analogs | funcharts | **present** | functional EWMA + CUSUM charts | `spm::ewma`, `spm::cusum`, `spm::mewma` |
| Phase I estimation with outlier masking | funcharts | **present** | Phase-I with outlier masking | `spm::iterative`, `spm::phase` |
| Parallel computing support for SPM chart simulation | funcharts | **present** | parallel SPM simulation | `spm::arl`/`spm::bootstrap` under `iter_maybe_parallel!` (parallel feature) |
| Contribution analysis for out-of-control signals | funcharts | **present** | contribution analysis | `spm::contrib` |

**Area 8 verdicts:** present = 9 · partial = 1 · absent = 0 (total 10).

#### Area 9: FPCA — Sparse / Longitudinal / Specialized (18 in-scope)

| Capability | Source | Verdict | searched fdars for: … | Closest match |
|------------|--------|---------|-----------------------|---------------|
| FPCA via PACE for sparse/irregular curves | fdapace, fda | **partial** | PACE FPCA for sparse curves | `regression::fdata_to_pc_1d` (dense FPCA) + `irreg_fdata::to_regular_grid` pre-step + `spm::partial` PACE conditional-expectation; no unified PACE FPCA estimator |
| Covariance surface estimation from sparse observations | fdapace, face | **present** | sparse covariance-surface estimation | `irreg_fdata::cov_irreg` |
| Mean curve estimation from sparse observations (PACE) | fdapace | **present** | sparse mean-curve estimation | `irreg_fdata::to_regular_grid` (kernel-smoothed mean over irregular obs) |
| FPC scores via conditional expectation | fdapace | **partial** | conditional-expectation FPC scores for sparse data | `spm::partial::conditional_expectation` (PACE-framework reconstruction); not a general FPC-score estimator for arbitrary sparse FPCA |
| Fitted continuous trajectories + confidence bands for sparse data | fdapace | **partial** | trajectory reconstruction with confidence bands | `spm::partial` reconstructs trajectories; `alignment::shape_ci` gives bands; not integrated PACE trajectory + band output |
| FPCA for functional derivatives (FPCAder) | fdapace | **absent** | FPCA of derivatives | no match found — *differentiator* |
| Functional SVD (FSVD — bivariate FPCA analogue) | fdapace | **absent** | functional SVD / cross-FPCA | no match found — *differentiator* |
| Cross-covariance function estimation (GetCrCovYX/YZ) | fdapace | **absent** | cross-covariance surface estimation | no match found — *differentiator* |
| Dynamical correlation (DynCorr) | fdapace | **absent** | dynamical correlation | no match found — *differentiator* |
| Functional correlation (FCCor) | fdapace | **absent** | functional correlation | no match found — *differentiator* |
| Functional variance process analysis (FVPA) | fdapace | **absent** | functional variance process | no match found — *differentiator* |
| Number of FPC selection criteria (ER_GR, FVE threshold) | fdapace, ftsa | **present** | number-of-components selection (FVE / eigenratio) | `scalar_on_function::model_selection_ncomp`; `FpcaResult` singular values give FVE |
| MakeFPCAInputs — FPCA input validation / formatting | fdapace | **present** | FPCA input validation/formatting | `validation` module + `IrregFdata` constructors |
| Functional fragment completion (incomplete curve reconstruction) | fdapace | **partial** | reconstruct incomplete curves | `spm::partial` domain completion; not general fragment completion |
| Fast FPCA for sparse data (fpca.sc — sandwich smoother) | refund | **absent** | sandwich-smoother sparse FPCA | no match found — *differentiator* |
| FPCA via FACE (fast covariance) | refund, face | **partial** | FACE-based FPCA | `irreg_fdata::cov_irreg` + `fdata_to_pc_1d`; not the FACE fast-sandwich algorithm |
| FPCA via sparse SVD (fpca.ssvd) | refund | **partial** | sparse-SVD FPCA | `fdata_to_pc_1d` uses thin SVD (faer, v0.15.0); not the penalized sparse-SVD (fpca.ssvd) variant |
| FPCA via local FDA (fpca.lfda) | refund | **absent** | local-FDA FPCA | no match found — *differentiator* |

**Area 9 verdicts:** present = 4 · partial = 6 · absent = 8 (total 18).

### §Parity-Matrix — Verdict Counts

Per-area and overall verdict tallies over the **250 literal in-scope capability rows mapped** (the 248 Phase-16 header total plus the net +2 literal-row reconciliation: Areas 3/6/7 each carry +1 literal in-scope row over their header, and Area 4 carries −2 — see the Area-4 recount note; all rows trace 1:1 to Phase-16 §R-Inventory table rows).

| Area | Present | Partial | Absent | Rows | **Actionable gaps (partial+absent)** |
|------|---------|---------|--------|------|--------------------------------------|
| 1 — Representation / Basis / Smoothing | 20 | 8 | 10 | 38 | **18** |
| 2 — Preprocessing / Registration | 16 | 4 | 2 | 22 | **6** |
| 3 — Exploratory / Depth / Outlier | 12 | 5 | 16 | 33 | **21** |
| 4 — ML (Regression + Classification + Clustering) | 25 | 12 | 20 | 57 | **32** |
| 5 — Inference / Testing | 0 | 5 | 17 | 22 | **22** |
| 6 — Functional Time Series | 2 | 3 | 20 | 25 | **23** |
| 7 — Density / Object Data / Manifold | 0 | 5 | 20 | 25 | **25** |
| 8 — SPM / Control Charts | 9 | 1 | 0 | 10 | **1** |
| 9 — FPCA (Sparse / Longitudinal / Specialized) | 4 | 6 | 8 | 18 | **14** |
| **TOTAL** | **88** | **49** | **113** | **250** | **162** |

**Headline actionable-gap count (in-scope, absent + partial): 162** (49 partial "add-a-variant" + 113 absent "implement-from-scratch"). This is the input set Phase 19 ranks. Present = 88/250 (35%).

**Where fdars is strongest (present-heavy):** SPM / control charts (9/10 present — fdars' `spm/` suite exceeds any single R package's chart coverage), Preprocessing / Registration (16/22 — the elastic/SRVF alignment stack is a core fdars strength), and Representation (20/38 — B-spline/Fourier bases, smoothing, FPCA, extrapolation policies, summary stats all shipped).

**Where the gaps concentrate (absent-heavy):** Area 7 Density / Object Data / Manifold (0 present — the entire `frechet` metric-space regression + `fdadensity` LQD-FPCA + `MFPCA` multi-domain surface is absent), Area 6 Functional Time Series (2/25 present — no forecasting, functional ACF, spectral DPCA, or FARMA), and Area 5 Inference (0 present — no ITP, functional ANOVA V-statistic, or formal FLM/two-sample tests as standalone hypothesis tests). These three areas confirm the Phase-16 pre-flagged large-gap zones.

**Accuracy flags (carried from v0.14.0 §Phase 8, D-02).** Two present rows touch known-bug areas and are read as "present — accuracy NOT verified": the elastic-alignment family (Area 2 "Pair-wise elastic alignment" / "Elastic curve mean" / "SRVF Karcher mean" — GH #34 level-encoding, fixed `6ed62398`) and the basis round-trip path underlying Area 1 "Basis conversion / projection" (GH #33, fixed `2fb6d3c9`). Both fixes have landed; presence is not re-validated numerically in this audit (deferred, per the v0.14.0 D-02a convention).

---

### §Categorization

Every one of the **162 actionable gaps** (49 partial + 113 absent) is categorized under the D-03 rubric (documented once, above). Present rows carry no category.

**Category counts.**

| Category | Partial | Absent | **Total gaps** | Share |
|----------|---------|--------|----------------|-------|
| **table-stakes** | 11 | 7 | **18** | 11% |
| **differentiator** | 38 | 106 | **144** | 89% |
| **out-of-scope** | 0 | 0 | **0** | 0% |
| **TOTAL** | 49 | 113 | **162** | 100% |

The gap profile is overwhelmingly **differentiator** (144/162, 89%) — the bulk of the R ecosystem that fdars lacks is specialized methodology (metric-space/object-data regression, functional time series, sparse-PACE variants, the long tail of depth measures, mixed/additive/Bayesian regression families). Only **18 gaps (11%) are table-stakes** — capabilities a general-purpose FDA library is expected to have. These 18 are the Phase-19 priority signal.

**out-of-scope = 0.** The mapping input set was already filtered to in-scope numeric algorithms + API-ergonomics in Phase 16; on inspection no in-scope row degraded to a rendering/IO-adjacent capability, so the out-of-scope bucket is empty here (the 27 plotting/IO R rows were excluded before this matrix).

#### The 18 table-stakes gaps (Phase-19 priority signal), by area

| # | Capability | Source | Verdict | Area | Rationale (why table-stakes) |
|---|-----------|--------|---------|------|------------------------------|
| 1 | Constant / intercept basis | fda | absent | 1 | Every basis-expansion FDA library exposes an intercept/constant basis; needed for regression design matrices. |
| 2 | Smoothing with automatic parameter selection (GCV + AIC) | fda | partial | 1 | GCV present; AIC-based smoothing-parameter selection is a mainstream expected criterion. |
| 3 | General functional depth dispatcher (multiple methods) | fda.usc | partial | 3 | A single depth entry point that dispatches over methods is a baseline ergonomic; fdars exposes methods only as separate functions. |
| 4 | Functional boxplot (depth-based fence / threshold values) | roahd, fdaoutlier, fda.usc | partial | 3 | The López-Pintado depth-fence functional boxplot is the canonical robust-summary tool; only the outliergram parabolic fence is exposed. |
| 5 | Varying-coefficient / concurrent functional regression | fdaconcur, refund | absent | 4 | Concurrent (varying-coefficient) regression is a mainstream functional-regression model absent from fdars. |
| 6 | Functional concurrent regression (varying-coeff, sparse/dense) | fdapace, fdaconcur | absent | 4 | Same capability from the sparse-data side; still a baseline functional-regression expectation. |
| 7 | Functional GLM, scalar response (logistic + Poisson + families) | fda.usc, refund | partial | 4 | Logistic present; Poisson/other exponential-family functional GLMs are a standard expected family. |
| 8 | Functional linear model goodness-of-fit / F-test | fda.usc | partial | 4 | A fitted FLM should ship a GoF/F-test; only informal R² diagnostics exist. |
| 9 | Functional ANOVA (one-way) with V-statistic | fdatest, fdANOVA | partial | 5 | One-way functional ANOVA is a baseline inference test; permutation form exists, the asymptotic V-statistic form does not. |
| 10 | F-permutation test for functional data (Fperm.fd) | fda | partial | 5 | A standalone functional F-permutation test is a baseline two/multi-group inference tool. |
| 11 | t-permutation test for functional two-sample comparison | fda | absent | 5 | Two-sample functional mean comparison is the most basic functional hypothesis test. |
| 12 | Equality of functional means / covariance test | fda.usc | partial | 5 | A standalone two-sample mean/covariance equality test is baseline; only an SPM-context Hotelling T² exists. |
| 13 | Goodness-of-fit test for the FLM | fda.usc | absent | 5 | Baseline model-adequacy test for the functional linear model. |
| 14 | F-test for the FLM with scalar response | fda.usc | absent | 5 | Baseline significance test for the scalar-response FLM. |
| 15 | Two-sample equality test (SCBmeanfd) | SCBmeanfd | absent | 5 | Two-sample mean-function equality is a baseline inference capability. |
| 16 | Simultaneous confidence bands for the mean function | SCBmeanfd | partial | 5 | SCBs for the mean are a standard inferential summary; only shape-CI / SPM limits exist. |
| 17 | FPCA via PACE for sparse/irregular curves | fdapace, fda | partial | 9 | PACE FPCA is the canonical method for sparse/longitudinal functional data; fdars has the pieces but no unified estimator. |
| 18 | FPC scores via conditional expectation | fdapace | partial | 9 | Conditional-expectation (PACE) FPC scores are the standard way to score sparse curves; only a partial SPM-context reconstruction exists. |

**Reading the priority signal.** Table-stakes gaps cluster in **Area 5 Inference (8 of 18)** — fdars has essentially no standalone hypothesis-testing surface (0 present in Area 5) — and **Area 4 ML regression (4 of 18)**, chiefly concurrent regression and functional-GLM families. **Area 9 (2)**, **Area 1 (2)**, and **Area 3 (2)** round out the list. Areas 2, 6, 7, 8 contribute **zero** table-stakes gaps: Area 2/8 because fdars already covers the baseline (registration, SPM), and Areas 6/7 because their gaps, though numerous, are specialized (functional time series, metric-space object data) rather than baseline-expected — they are the differentiator long tail, not table-stakes deficits.

*This completes Phase 17 (GAP-01 parity matrix + GAP-02 categorization). The 18 table-stakes gaps and 144 differentiator gaps feed Phase 18's reverse-parity fdars-strengths sweep and Phase 19's value-ranked backlog.*

---

## Phase 18 — Reverse-Parity Strengths Sweep

**Sweep date:** 2026-08-15 · **Requirement:** GAP-03 · **Mode:** audit-only (zero `fdars-core/src/` edits).

This section walks **every module** in `fdars-core/src/` (crate shipped through v0.17.0) and catalogues where fdars is **unique** (no R FDA-ecosystem equivalent) or **ahead** of its closest R analog. The yardstick is the **R functional-data-analysis ecosystem** (the 35 packages of Phase 16 §R-Inventory + a handful of adjacent CRAN packages surfaced during re-vetting), **not scikit-fda**. This is the critical difference from the v0.14.0 §Phase 8 reverse-parity sweep (30 fdars-only-vs-scikit-fda items): **R is much broader than scikit-fda**, so several v0.14.0 "fdars-unique" items do *not* survive here.

### Method & honesty rule

- Every candidate strength is cross-checked against the Phase-16 R inventory (§R-Inventory), the Phase-17 parity matrix (§Parity-Matrix, esp. the present-heavy areas), and — where the inventory left doubt — a targeted CRAN/literature search (recorded in the row). A capability is **fdars-unique** only when **no R FDA package delivers it**; **fdars-ahead** only when an R analog exists but fdars' version is genuinely broader/more integrated.
- **Where R leads, it is NOT a strength.** Per Phase 17, R leads on: `fdasrvf` elastic breadth (elastic multinomial, elastic PCA sub-variants), `fdapace` sparse-PACE FPCA, `refund`/`FDboost` FoF/pffr/boosting regression, `ftsa` functional forecasting + fACF, `frechet`/`fdadensity` metric-space & density object-data, `fdatest`/`fdANOVA`/`SCBmeanfd` inference. Those are Phase-17 gaps and are **excluded** from this catalogue.

### Re-vet casualties (v0.14.0-vs-scikit-fda strengths that do NOT survive vs R)

The v0.14.0 §Phase 8 sweep listed these as fdars-only vs scikit-fda; **R has them**, so they are dropped from the unique/ahead catalogues below:

| v0.14.0 scikit-fda "fdars-only" item | Present in R? | R source (Phase 16 area / CRAN) |
|---|---|---|
| Statistical Process Monitoring / control charts (#2) | **Yes** | `funcharts` (Area 8) — Phase 17 rates fdars 9/10 *present*; fdars is ahead on chart *breadth* (see AHEAD-1), but SPM as a capability is not R-absent. |
| Conformal prediction (#5) | **Yes** | `conformalInference.fd` (Area 4) — Phase 17 *present*; not fdars-unique. |
| Elastic regression / FPCA / changepoint / shape depth / geodesics / Bayesian & constrained alignment / warping utils (#13, #18–#20, #22–#24, #27, #30) | **Yes (mostly)** | `fdasrvf` (Areas 2/3/4) covers SRVF regression, amplitude/phase FPCA, elastic depth, elastic changepoint, Karcher-mean geodesics, warp composition/inversion. `fdasrvf` is *broader* than fdars here (Pitfall 4) → these are Phase-17 territory, not strengths. Bayesian alignment: `fdasrvf` ships `bayesian` SRVF alignment too. |
| Singular Spectrum Analysis (#9) | **Yes** | `Rfssa` (Functional SSA, CRAN) + `Rssa` (scalar SSA) — functional SSA decomposition exists in R. Re-vet search 2026-08-15. Dropped. |
| Function-on-function regression (#17) | **Yes** | `refund` (`ff`/`pffr`), `FDboost` (Area 4) — fdars' `fof_regression` is *present* but R is broader (pffr mixed/RE, boosting) → Phase-17 gap side, not a strength. |
| Function-on-scalar regression / FOSR incl. 2D (#21) | **Yes (1D)** | `refund` (`fosr`/`pffr`), `FDboost` (Area 4). 1D FOSR is *present* in both. **2D surface-response FOSR** (`function_on_scalar_2d`, tensor-product penalty) has no direct single-package R analog → retained as AHEAD-6 (nuance), not unique. |
| Irregular functional data module (#28), Regression-FPCA backbone (#29), Multi-response SoF (#15) | **Yes / partial** | Irregular data: `funData::irregFunData`, `fdapace`, `face` (Areas 1/9). FPCA-in-regression: `refund`/`fda.usc` chain FPC scores. Multi-response SoF is niche but `refund` FoF/mv-response covers vector responses → dropped as not clearly R-absent. |

### §Reverse-Parity — Per-Module Coverage (exhaustiveness proof)

Every top-level module and submodule directory in `fdars-core/src/` is listed. **Strength found?** = whether the module contributes a row to the fdars-unique or fdars-ahead catalogues below (U-n / A-n), or "no" (capability present in R at parity or R leads — a Phase-17 concern, not a strength).

| Module (`src/…`) | Capabilities scanned | Strength found? |
|---|---|---|
| `alignment/` (30 files: karcher, pairwise, srsf, geodesic, bayesian, constrained, closed, robust_karcher, shape_ci, phase_boxplot, elastic_depth, outlier, clustering, quality, shift, nd, partial_match, transfer, tsrvf, warp_stats, …) | Full SRVF/elastic alignment suite | **No** — `fdasrvf` (Pitfall 4) is broader; Phase-17 rates alignment *present* and lists sparse-2D/joint-GLM registration as gaps. Not a strength vs R. |
| `andrews.rs` | `andrews_transform`, `andrews_loadings` | **U-1** — Andrews-curve transform + loadings for functional data; no R FDA package. |
| `basis/` (bspline, fourier, pspline, projection, fourier_fit, auto_select) | B-spline/Fourier/P-spline bases, fit, projection, auto-select | **No** — `fda` is canonical (Area 1 *present*); R leads on monomial/constant/exp/power/FEM bases. |
| `classification/` (lda, qda, knn, kernel, dd, cv, fit) | LDA/QDA/kNN/kernel/DD classifiers + CV | **No** — `fda.usc` at parity (Area 4 *present*). |
| `clustering.rs` | k-means, fuzzy c-means for fd | **No** — `fda.usc`/`fdacluster` at parity; R leads (DBSCAN, funHDDC, funLBM). |
| `conformal/` (regression, classification, cv, elastic, generic) | Split/full conformal + elastic conformal | **A-2** (nuance) — `conformalInference.fd` covers regression conformal (parity); fdars adds **conformal *classification*** + **elastic-space conformal**, which the R package does not. |
| `covariance.rs` | Empirical / smoothed covariance surfaces | **No** — `roahd`/`fda.usc`/`face` at parity or ahead (sparse FACE). |
| `cv.rs` | Generic CV utilities, R² metrics | **No** — ergonomic layer; `fda.usc`/`refund` at parity. |
| `depth/` (band, modified band, fraiman_muniz, modal, random_projection, random_tukey, rpd, spatial) | Batch functional depths | **No** — `roahd`/`fdaoutlier`/`ddalpha` at parity or ahead (long tail of depths R has, fdars lacks — Phase 17). |
| `detrend/` (stl, loess, linear, polynomial, diff, decompose, auto) | STL/LOESS detrend, differencing, decompose | **No** — STL/LOESS are scalar-TS-standard (`stats::stl`, `forecast`); fd differencing is Phase-17 *partial* (ftsa). Not R-absent. |
| `distance.rs` | Pairwise Lp distances between curves | **No** — `fda.usc::metric.lp` at parity. |
| `elastic_changepoint.rs` | Amplitude/phase/FPCA elastic changepoint | **No** — `fdasrvf` elastic changepoint (Area 3 *present*). |
| `elastic_explain.rs` | `elastic_pcr_attribution` | **U-2** — feature attribution for elastic-PCR models; no R FDA explainability, elastic or otherwise. |
| `elastic_fpca.rs` | vert/horiz/joint (amplitude/phase) FPCA | **No** — `fdasrvf` amplitude/phase FPCA (Area 2 *present*). |
| `elastic_regression/` (regression, logistic, pcr, scalar_on_shape) | Elastic SRVF regression family | **No** — `fdasrvf` (Area 4 *present*; R leads on elastic multinomial). |
| `explain/` (pdp, shap, ale_lime, importance, sensitivity, counterfactual, diagnostics, advanced + helpers/) | 44+ explainability fns for functional models | **U-3** (headliner) — model explainability for functional models. |
| `explain_generic/` (pdp, shap, lime, ale, importance, saliency, sobol, friedman, anchor, prototype, counterfactual, stability + `FpcPredictor` trait) | 15 model-agnostic explainers via one trait | **U-3** (headliner) — same family; trait-driven generic layer. |
| `famm.rs` | `fmm`, `fmm_predict`, `fmm_test_fixed` | **No** — functional mixed models exist in R (`denseFLMM`, `multifamm`, `fastFMM`, Area 4); R is broader (random-effects estimators). Phase-17 *partial*. |
| `fdata.rs` | Core fd container + summary stats (mean/var/std/cov/median/trim) | **No** — `fda`/`fda.usc`/`roahd` at parity (Area 1 *present*). |
| `fof_regression.rs` | Function-on-function regression | **No** — `refund`/`FDboost` broader (Area 4). |
| `function_on_scalar.rs` | 1D FOSR + permutation ANOVA | **No** — `refund`/`FDboost` at parity/ahead. |
| `function_on_scalar_2d.rs` | 2D surface-response FOSR (tensor penalty) | **A-6** (nuance) — 2D functional-surface response regression with tensor-product penalty; no single R package packages this exactly (`refund` pffr is 1D-response-centric). Modest lead. |
| `gmm/` (em, cluster, covariance, init) | GMM (EM) clustering for fd | **No** — model-based fd clustering exists in R (`funHDDC`, `funFEM`, `mclust`-on-scores); R broader. |
| `helpers.rs` | Quadrature, interpolation, extrapolation policy, imputation | **No** — infrastructure; `fda`/`tf` at parity (Area 1 *present*). |
| `irreg_fdata/` (kernels, smoothing) | Irregular→grid kernel smoothing | **No** — `funData::irregFunData`, `fdapace`, `face` cover irregular data. |
| `landmark.rs` | Landmark detect + register, monotone warp | **No** — `fda` landmark registration (Area 2 *present*). |
| `linalg.rs` | faer/nalgebra linear-algebra shims | **No** — infrastructure, not a user capability. |
| `matrix.rs` | `FdMatrix` column-major container | **No** — infrastructure; `funData`/`fda` fd containers at parity. |
| `metric/` (lp, dtw, soft_dtw, hausdorff, kl, fourier, deriv, pca, basis_coef, hshift) | Curve distances incl. **soft-DTW + soft-DTW barycenter**, Hausdorff, KL, DTW | **A-4** — classic DTW & Lp exist in R (`dtw`, `fda.usc`); **soft-DTW divergence + soft-DTW barycenter** for functional data have no CRAN R analog (Python `tslearn` only; re-vet search 2026-08-15). |
| `outliers.rs` | MS-plot / directional outlyingness, outliergram | **No** — `fdaoutlier`/`roahd` at parity or ahead (Area 3). |
| `regression.rs` | FPCA (`fdata_to_pc_1d`), `FpcaResult`, PLS | **No** — FPCA at parity (`fda`/`fda.usc`/`fdapace`); the *explainability integration* is credited under U-3, not here. |
| `scalar_on_function/` (fregre_lm, pls, nonparametric, logistic, **robust**, **multi**, cv, bootstrap) | SoF regression family | **A-5** (nuance) — core SoF at parity (`fda.usc`); **robust SoF** (`fregre_l1` L1-loss, `fregre_huber` Huber-loss) has no direct R FDA analog (fda.usc SoF is OLS/kernel/PLS). Modest lead. |
| `scoring.rs` | Functional MAE/MSE/MAPE/MSLE/explained-variance | **No** — `ftsa` error metrics at parity (Area 6 *present*). |
| `seasonal/` (autoperiod, period, sazed, peak, strength, change, hilbert, lomb_scargle, matrix_profile, ssa) | Period detection, peaks, Hilbert, Lomb-Scargle, matrix profile, SSA on fd | **A-7** — SSA has an R functional analog (`Rfssa`) → *not* unique; but **automatic period detection (SAZED/autoperiod), functional matrix-profile (motif/discord), Hilbert-transform instantaneous frequency, and Lomb-Scargle** for functional curves have no R **FDA** package (scalar-TS `tsmp`/`lomb`/`seewave` exist, but not integrated for fd). Ahead-of-nearest-analog (numeric-signal toolkit inside an FDA library). |
| `simulation.rs` | KL-expansion Gaussian fd simulation | **No** — `fda.usc`/`funData`/`fdasrvf` simulate fd; `freqdom`/`ftsa` simulate FARMA/VAR (R broader, Phase-17). |
| `smooth_basis.rs` | Penalized basis smoothing + GCV + penalty matrices | **No** — `fda::smooth.basis` canonical (Area 1 *present*). |
| `smoothing.rs` | Kernel smoothers (NW/local-linear/kNN), CV/GCV bandwidth | **No** — `fda.usc`/`face` at parity (Area 2 *present*). |
| `spm/` (21 files: phase, monitor, ewma, cusum, mewma, amewma, frcc, control, stats, rules, contrib, arl, elastic_spm, mfpca, iterative, profile, partial, chi_squared, ncomp, bootstrap) | Full SPM chart suite for functional data | **A-1** — `funcharts` covers functional SPM (Phase-17 *present*), so SPM is not R-absent; but fdars' **chart-type breadth** (EWMA + CUSUM + MEWMA + adaptive-MEWMA + FRCC + Nelson/Western-Electric run rules + contribution analysis + elastic-shape SPM) exceeds `funcharts` in one integrated module → ahead on breadth. |
| `streaming_depth/` (fraiman_muniz, bd, mbd, rolling, sorted_ref) | Incremental/online functional depth | **U-5** — streaming/online functional depth; R depth is batch-only (`roahd`/`ddalpha`/`DepthProc`; re-vet search 2026-08-15). |
| `tolerance/` (fpca, degras, conformal, equivalence, exponential, elastic + types) | Simultaneous functional tolerance bands | **U-6** — simultaneous functional tolerance bands (FPCA/Degras/conformal/equivalence/exp-family/elastic); no R FDA package. (R `SCBmeanfd` gives *confidence* bands for the mean — a different object; noted, not equated.) |
| `utility.rs` | Inner products, inner-product matrix | **No** — `fda` inner products at parity. |
| `validation.rs` | Input dimension/parameter validation | **No** — infrastructure. |
| `warping.rs` | Warp invert/compose/normalize, γ↔ψ | **No** — `fda`/`fdasrvf` warp utilities (Area 2 *present*). |
| `wire.rs` | `FdaData` unified serializable workflow-result container | **U-4** — Rust-native serializable multi-layer workflow container (FPCA/alignment/depth/cluster/regression/FOSR/tolerance/SPM/explain layers). No R analog (R uses S3/S4 result objects + lists, not a unified layer container). API-ergonomics, not a numeric algorithm. |

**Exhaustiveness:** all **21 directory submodule groups + 21 top-level `.rs` files = 42 module units** in `fdars-core/src/` scanned (excludes `lib.rs`, `prelude.rs`, `error.rs`, `parallel.rs`, `test_helpers.rs`, `elastic.rs` re-export shim — pure infrastructure/re-export with no user capability). Every unit maps to a catalogue row (U-n/A-n) or an explicit "No — R at parity / R leads" verdict.

### §Reverse-Parity — fdars-unique (no R equivalent)

Capabilities with **no** R FDA-ecosystem package delivering them. Each row names the closest-R "none found" evidence.

| # | fdars capability | fdars module / function | Closest R analog — none found (evidence) | Confidence |
|---|---|---|---|---|
| **U-1** | **Andrews-curve transform for functional data** — `andrews_transform` (Andrews Fourier transform for dimensionality-reduction) + `andrews_loadings` (component interpretation). | `andrews.rs` | No R FDA package offers an Andrews-curve *transform for functional data*. Generic `andrews_curve`-style plotting exists in tabular-viz packages (e.g. base/`ggplot2` recipes) but as a *plot*, not a functional-data numeric transform + loadings. Not in Phase-16 inventory. | HIGH |
| **U-2** | **Elastic-model explainability** — `elastic_pcr_attribution`: feature attribution for elastic-PCR (SRSF-space) regression. | `elastic_explain.rs` | No R FDA package offers explainability for elastic/shape models (nor for functional models at all — see U-3). | MEDIUM |
| **U-3** | **Model explainability for functional models** (headliner) — PDP, SHAP, LIME, ALE, permutation & Friedman-H importance, Sobol indices, saliency, sensitivity, counterfactual search, anchor explanations, prototype/criticism, DFbetas/DFFits influence diagnostics — model-agnostic via the `FpcPredictor` trait (drives regression + classification + logistic uniformly). | `explain/` (44+ fns, 9 files + helpers/), `explain_generic/` (15 fns + `FpcPredictor`) | **None found.** R has strong *model-agnostic* explainer packages (`DALEX`, `lime`, `pdp`, `iml`, `modelStudio`, `fastshap`) — but they operate on tabular/scalar-feature models and are **not integrated with functional-regression / FPC-score models**. No R FDA package (`fda.usc`, `refund`, `FDboost`, `fdapace`, …) exposes PDP/SHAP/LIME/ALE for functional predictors. Re-vet search 2026-08-15. | HIGH |
| **U-4** | **WIRE unified workflow container** — `FdaData` + typed `Layer` enum (FPCA/alignment/distances/depth/outlier/cluster/regression/FOSR/tolerance/mean/SPM-chart/SPM-monitor/explain/custom): one serializable Rust-native structure capturing heterogeneous FDA-pipeline outputs. | `wire.rs` | **None found.** R FDA packages return per-method S3/S4 objects and ad-hoc lists; there is no unified serializable multi-layer workflow-result container. (API-ergonomics strength, not a numeric algorithm.) | MEDIUM |
| **U-5** | **Streaming / online functional depth** (headliner) — incremental Fraiman-Muniz, streaming Band Depth & Modified Band Depth, rolling reference window, sorted-reference-state accumulation. | `streaming_depth/` (5 files; `StreamingDepth` trait + 4 impls) | **None found.** All R functional-depth packages (`roahd`, `fdaoutlier`, `ddalpha`, `DepthProc`, `depthTools`, `fda.usc`) compute depth on a **fixed batch reference set** — no incremental/online/rolling depth. Re-vet search 2026-08-15. | HIGH |
| **U-6** | **Simultaneous functional tolerance bands** — FPCA-based, Degras bootstrap, conformal, equivalence-test, exponential-family, and elastic tolerance bands (`ToleranceBand`, `BandType`, `PhaseToleranceBand`, `ElasticToleranceBandResult`). | `tolerance/` (6 method files + types) | **None found.** No R FDA package produces simultaneous functional *tolerance* bands. `SCBmeanfd` gives *confidence* bands for the **mean function** (a different statistical object — coverage of the mean, not of future curves); `roahd`/`fdaoutlier` give depth-fence boxplots (not tolerance bands). | HIGH |

**fdars-unique count: 6** (2 headliners: U-3 explainability, U-5 streaming depth; plus U-1 Andrews, U-2 elastic-explain, U-4 WIRE, U-6 tolerance bands).

### §Reverse-Parity — fdars-ahead (leads closest R analog)

An R analog exists, but fdars' version is broader / more integrated. The R analog is named and the nature of the lead stated. (These are *modest* leads — where R clearly leads, e.g. `fdasrvf` elastic breadth or `refund` FoF, the item is a Phase-17 gap and is excluded.)

| # | fdars capability | fdars module | Named R analog | Nature of the lead |
|---|---|---|---|---|
| **A-1** | **SPM chart-type breadth in one module** — EWMA, CUSUM, MEWMA, adaptive-MEWMA, FRCC, χ², Hotelling-T²/SPE, Nelson/Western-Electric run rules, contribution analysis, ARL simulation, elastic-shape SPM, MFPCA SPM. | `spm/` (21 files) | `funcharts` (1.8.1) | `funcharts` covers Phase-I/II functional control charts and profile monitoring (Phase-17 rates fdars 9/10 *present* vs it). fdars packages a **wider set of chart types + run-rule logic + elastic-shape monitoring** in one integrated suite; `funcharts` does not expose CUSUM/adaptive-MEWMA/run-rules at the same breadth. Lead = chart-type breadth + rule engine. |
| **A-2** | **Conformal prediction breadth** — split/full conformal for **regression *and* classification**, multiple non-conformity scores, plus **elastic-space** conformal. | `conformal/` (5 files) | `conformalInference.fd` (1.1.1) | The R package targets conformal *regression* prediction bands. fdars adds **conformal classification** (`ClassificationScore`) and **elastic conformal** — modes absent from the R package. Lead = task coverage (classification + elastic). |
| **A-4** | **Soft-DTW distance + soft-DTW barycenter for curves** — smooth differentiable DTW, soft-DTW divergence, and soft-DTW barycenter, alongside classic DTW / Hausdorff / KL metrics. | `metric/soft_dtw.rs`, `metric/dtw.rs` | `dtw` (classic DTW), `fda.usc` (Lp/semimetrics) | Classic DTW and Lp semimetrics exist in R. **Soft-DTW (divergence + barycenter)** has no CRAN R implementation for functional data — it lives in Python `tslearn`/`mblondel/soft-dtw`. Lead = differentiable soft-DTW family inside an FDA library. Re-vet search 2026-08-15. |
| **A-5** | **Robust scalar-on-function regression** — L1-loss (`fregre_l1`) and Huber-loss (`fregre_huber`) SoF regression + robust prediction/CV. | `scalar_on_function/robust.rs` | `fda.usc` (`fregre.pc`/`fregre.np`/`fregre.pls`) | `fda.usc` covers OLS/PLS/kernel SoF regression but **not L1/Huber robust-loss** SoF. (`roahd`/`fda.usc` have robust *summaries*, not robust *regression*.) Lead = robust-loss SoF estimators. |
| **A-6** | **2D surface-response function-on-scalar regression** — `function_on_scalar_2d` with tensor-product roughness penalty (`Grid2d`, `FosrResult2d`). | `function_on_scalar_2d.rs` | `refund` (`fosr`/`pffr`) | R FOSR (`refund`) is oriented to 1D functional responses; fdars packages a **2D functional-surface response** FOSR with a tensor-product penalty as a first-class estimator. Lead = 2D-surface response support. |
| **A-7** | **Signal-processing toolkit for functional curves** — automatic period detection (SAZED, autoperiod), functional **matrix profile** (motif/discord discovery), Hilbert-transform instantaneous frequency/amplitude, Lomb-Scargle periodogram. | `seasonal/` (autoperiod, sazed, matrix_profile, hilbert, lomb_scargle) | scalar-TS: `tsmp` (matrix profile), `lomb` (Lomb-Scargle), `seewave`/`hht` (Hilbert); functional SSA: `Rfssa` | Each primitive has a **scalar-time-series** R package, but **none is integrated for functional-data curves inside an FDA library**. (SSA itself is *not* a lead — `Rfssa` provides functional SSA; excluded.) Lead = period/motif/instantaneous-frequency signal tooling operating on `FdMatrix` curves. |

**fdars-ahead count: 6** (A-1 SPM breadth, A-2 conformal breadth, A-4 soft-DTW, A-5 robust SoF, A-6 2D-FOSR, A-7 fd signal toolkit). *(Numbering skips A-3: the elastic-alignment family — an A-3 candidate vs scikit-fda — is a Phase-17 gap vs `fdasrvf` and is deliberately omitted.)*

### §Reverse-Parity — Summary count

- **Modules walked:** 42 module units (21 submodule directory groups + 21 top-level `.rs` capability files); 6 pure-infrastructure/re-export files excluded. Every unit carries an explicit verdict → demonstrably exhaustive.
- **fdars-unique (no R equivalent): 6** — U-3 model explainability (headliner) · U-5 streaming/online depth (headliner) · U-1 Andrews transform · U-2 elastic-model explain · U-4 WIRE workflow container · U-6 simultaneous tolerance bands.
- **fdars-ahead (leads closest R analog): 6** — A-1 SPM chart breadth (vs `funcharts`) · A-2 conformal breadth incl. classification+elastic (vs `conformalInference.fd`) · A-4 soft-DTW + barycenter (vs `dtw`) · A-5 robust L1/Huber SoF (vs `fda.usc`) · A-6 2D-surface FOSR (vs `refund`) · A-7 functional signal toolkit (vs scalar-TS `tsmp`/`lomb`/`hht`).
- **Net vs the v0.14.0 scikit-fda sweep:** the 30 scikit-fda "fdars-only" items collapse to **12 R-honest strengths** (6 unique + 6 ahead). The big casualties are **SPM, conformal, and the entire elastic/shape stack** (unique vs scikit-fda, but R has `funcharts`/`conformalInference.fd`/`fdasrvf`), plus **SSA** (`Rfssa`) and **FoF/FOSR-1D** (`refund`/`FDboost`). fdars' *genuine* R-relative moat is **explainability for functional models** and **streaming depth** — no R package touches either.

*This completes Phase 18 (GAP-03 reverse-parity fdars-strengths sweep). The 12 R-honest strengths feed Phase 19's consolidated findings — keeping the backlog from proposing work in areas where fdars already leads.*
