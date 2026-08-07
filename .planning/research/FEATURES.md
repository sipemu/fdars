# Feature Research: scikit-fda API Surface vs fdars Gap Analysis

**Domain:** Functional Data Analysis (FDA) library — numeric Rust crate vs Python reference
**Researched:** 2026-08-07
**Confidence:** MEDIUM (official docs verified at scikit-fda 0.10.1 stable; fdars coverage from codebase map)

---

## Purpose

This document maps scikit-fda's complete public API surface so the roadmap gap analysis can determine what fdars is missing, partially implements, or deliberately does not need. scikit-fda 0.10.1 is the agreed sole baseline.

Coverage is organized by area: Representation, Preprocessing (Smoothing + Registration + Dimensionality Reduction), Exploratory Analysis, Machine Learning, Inference, and Misc/Infrastructure.

---

## Area 1: Representation

### scikit-fda Public API

| Class / Function | Purpose |
|-----------------|---------|
| `FData` | Abstract base with shared interface (evaluate, arithmetic, derivatives) |
| `FDataGrid` | Discretized representation on a common evaluation grid; supports arithmetic, finite-difference derivatives, integration, inner products |
| `FDataBasis` | Parametric representation as linear combination of basis functions; analytical derivatives |
| `FDataIrregular` | Sparse/irregularly sampled observations per curve (added v0.10.0); covariance estimation |
| `BSplineBasis` | B-spline basis (R→R) |
| `FourierBasis` | Fourier (trigonometric) basis (R→R) |
| `MonomialBasis` | Monomial/polynomial basis (R→R) |
| `ConstantBasis` | Constant (intercept) basis (R→R) |
| `CustomBasis` | Arbitrary user-supplied basis functions |
| `TensorBasis` | Tensor product of 1D bases (Rn→R multivariate domain) |
| `FiniteElementBasis` | Finite element basis (Rn→R, irregular meshes) |
| `VectorValuedBasis` | Stack of bases for vector-valued output (Rn→Rm) |
| `SplineInterpolation` | Spline interpolation for evaluating FDataGrid at off-grid points |
| `BoundaryExtrapolation` | Extrapolation: repeat boundary value |
| `ExceptionExtrapolation` | Extrapolation: raise on out-of-domain query |
| `FillExtrapolation` | Extrapolation: fill with constant |
| `PeriodicExtrapolation` | Extrapolation: wrap periodically |
| `MinimizeMixedEffectsConverter` | Convert FDataIrregular → FDataBasis via mixed effects (optimization) |
| `EMMixedEffectsConverter` | Convert FDataIrregular → FDataBasis via EM algorithm |
| `FDataGrid.to_basis()` | Convert grid representation to basis representation |

### fdars Current Status

fdars uses a flat `FdMatrix` (column-major `Vec<f64>`) as its core data structure — it is a discretized grid equivalent without an object-oriented functional data type. There is no:
- Named FDataGrid/FDataBasis type distinction
- Basis system hierarchy (BSpline, Fourier, Monomial, etc.) as first-class objects
- Irregular/sparse functional data type (only `irreg_fdata/` module for some ops)
- Formal extrapolation/interpolation layer
- Grid-to-basis conversion pipeline

The `function_on_scalar_2d.rs` uses a tensor-product concept internally, and `basis` modules exist, but they are not exposed as composable basis-type objects.

### Categorization

**Table Stakes (any FDA library needs):**
- Discretized representation (grid) — fdars has FdMatrix, partial coverage
- B-spline basis — partial (internal use in smoothing/alignment)
- Fourier basis — partial (used in spectral ops)
- Grid-to-basis conversion — gap
- Spline interpolation at arbitrary query points — gap (fdars evaluates at stored grid only)
- Extrapolation policies — gap

**Differentiators:**
- FDataIrregular with covariance estimation — gap; scikit-fda only added this in v0.10.0
- FiniteElementBasis — advanced; unlikely needed in v1 of gap-filling
- VectorValuedBasis — advanced; relevant for multivariate functional data
- Mixed-effects irregular→basis conversion (EM, optimization) — advanced

**Anti-features for a numeric Rust crate:**
- Python-style object hierarchy with `__repr__`, `__getitem__` magic methods — Rust idiomatic API is preferable; do not replicate the Python class hierarchy literally

**Porting complexity:** HIGH overall. A first-class `FDataGrid`/`FDataBasis` type system with composable basis objects is an architectural refactor of fdars' data layer, not an additive feature.

---

## Area 2: Preprocessing — Smoothing

### scikit-fda Public API

| Class / Function | Purpose |
|-----------------|---------|
| `KernelSmoother` | Non-parametric kernel smoothing; hat matrix is pluggable |
| `NadarayaWatsonHatMatrix` | Nadaraya-Watson kernel smoother strategy |
| `LocalLinearRegressionHatMatrix` | Local linear regression smoother strategy |
| `KNeighborsHatMatrix` | k-nearest-neighbor smoother strategy |
| `BasisSmoother` | Penalized basis expansion smoother (penalizes derivatives via `LinearDifferentialOperator`) |
| `SmoothingParameterSearch` | Grid search over smoothing parameters (like sklearn GridSearchCV) |
| `LinearSmootherLeaveOneOutScorer` | LOO-CV scorer for linear smoothers |
| `LinearSmootherGeneralizedCVScorer` | GCV scorer for linear smoothers |
| `akaike_information_criterion` | AIC bandwidth selection criterion |
| `finite_prediction_error` | FPE criterion |
| `shibata` | Shibata's selector |
| `rice` | Rice's selector |
| `MissingValuesInterpolation` | Impute missing values in functional data |

### fdars Current Status

fdars does not expose a standalone smoothing module. Basis expansion is used internally in several modules (alignment, detrending) but not as a standalone, user-facing smoother with CV bandwidth selection. No Nadaraya-Watson, local linear regression, or k-NN smoothers as public API.

### Categorization

**Table Stakes:**
- Kernel smoothing (Nadaraya-Watson) — gap
- Basis smoothing with penalty — gap (internal use only; not exposed)
- CV bandwidth selection (LOO, GCV) — gap

**Differentiators:**
- Local linear regression smoother — gap; moderately advanced
- Multiple CV criteria (AIC, FPE, Shibata, Rice) — gap; useful for auto-tuning
- Missing value imputation for functional data — gap

**Anti-features:**
- None specific to smoothing; all of these are numerically implementable and appropriate for a Rust crate

**Porting complexity:** MEDIUM. Nadaraya-Watson and basis smoothers are numerically straightforward given fdars' matrix infrastructure. The pluggable hat-matrix pattern is an API design choice. CV bandwidth selection requires a scoring loop.

---

## Area 3: Preprocessing — Registration / Alignment

### scikit-fda Public API

| Class / Function | Purpose |
|-----------------|---------|
| `LeastSquaresShiftRegistration` | Shift-only alignment by minimizing LS criterion |
| `FisherRaoElasticRegistration` | Full elastic alignment via SRSF/Fisher-Rao metric |
| `landmark_shift_registration` | Align curves by shifting to match a landmark |
| `landmark_shift_deltas` | Compute shift deltas for landmark registration |
| `landmark_elastic_registration` | Elastic landmark registration (non-linear warping) |
| `landmark_elastic_registration_warping` | Return the warping function used in landmark elastic reg |
| `invert_warping` | Invert a warping function |
| `normalize_warping` | Normalize a warping function to [0,1] |
| `AmplitudePhaseDecomposition` | Validate registration: decompose amplitude vs phase variation |
| `LeastSquares` | Registration validation via LS criterion |
| `SobolevLeastSquares` | Registration validation via Sobolev-penalized LS |
| `PairwiseCorrelation` | Registration validation via pairwise correlation |

### fdars Current Status

fdars has strong elastic registration coverage in `src/alignment/`:
- Elastic curve registration (`elastic_align_pair`)
- Elastic shape analysis, SRSF framework
- Elastic FPCA, elastic regression

**Gaps identified:**
- `LeastSquaresShiftRegistration` (shift-only) — gap
- `landmark_shift_registration` and related landmark utilities — gap
- Registration quality validation classes (AmplitudePhaseDecomposition, SobolevLeastSquares, PairwiseCorrelation) — gap
- `invert_warping`, `normalize_warping` utilities — partial/unclear

### Categorization

**Table Stakes:**
- Shift registration — gap; simpler than elastic but widely expected
- Landmark registration — gap; standard FDA preprocessing step

**Differentiators:**
- Fisher-Rao elastic registration (SRSF) — fdars has this
- Registration validation / quality metrics — gap; useful for assessing registration quality

**Anti-features:** None; all relevant for a numeric library.

**Porting complexity:** LOW for shift/landmark registration. Registration validation is MEDIUM (requires amplitude-phase decomposition math).

---

## Area 4: Preprocessing — Dimensionality Reduction & Feature Construction

### scikit-fda Public API

| Class / Function | Purpose |
|-----------------|---------|
| `FPCA` | Functional PCA; sklearn transformer interface; supports regularization via `LinearDifferentialOperator` |
| `FPLS` | Functional PLS (partial least squares); added v0.9.1 |
| `DiffusionMap` | Functional diffusion maps; manifold learning; added v0.10.0 |
| `MaximaHunting` | Variable selection by identifying maxima of relevance measure |
| `RecursiveMaximaHunting` | Iterative maxima hunting with multiple correction strategies |
| `RKHSVariableSelection` | Variable selection via RKHS-based relevance |
| `MinimumRedundancyMaximumRelevance` | mRMR variable selection for functional data |
| `FDAFeatureUnion` | Combine multiple FDA feature transformers (like sklearn FeatureUnion) |
| `PerClassTransformer` | Apply different transformers per class label |
| `LocalAveragesTransformer` | Compute local averages on intervals as features |
| `OccupationMeasureTransformer` | Measure time spent in value ranges as features |
| `NumberCrossingsTransformer` | Count crossings of a threshold as features |
| `modified_epigraph_index` | Functional feature: modified epigraph index |
| `local_averages` | Functional feature: local averages |
| `occupation_measure` | Functional feature: occupation measure |
| `number_crossings` | Functional feature: threshold crossing count |

### fdars Current Status

fdars has FPCA (`FpcaResult`, `fdata_to_pc_1d`) and uses it extensively as the projection step before classification and regression. No standalone FPLS, DiffusionMap, or variable selection. No feature construction transformers.

### Categorization

**Table Stakes:**
- FPCA — fdars has this (solid coverage)
- FPLS — gap; expected companion to FPCA for supervised reduction

**Differentiators:**
- DiffusionMap — gap; manifold-based reduction; less common
- MaximaHunting / RecursiveMaximaHunting — gap; point-selection approach useful for interpretability
- mRMR variable selection — gap; advanced
- Feature construction transformers (local averages, occupation measure, crossings) — gap; useful for pipeline-based workflows

**Anti-features:**
- `FDAFeatureUnion` / `PerClassTransformer` are scikit-learn pipeline plumbing — equivalent in Rust would be trait composition, not a direct port; don't replicate the Python API shape

**Porting complexity:** MEDIUM for FPLS (PLS math is well-understood). HIGH for DiffusionMap. MEDIUM for feature construction transformers. LOW-MEDIUM for variable selection.

---

## Area 5: Exploratory Analysis — Depth & Outlier Detection

### scikit-fda Public API

| Class / Function | Purpose |
|-----------------|---------|
| `IntegratedDepth` | Fraiman-Muniz integrated depth |
| `BandDepth` | Band depth (López-Pintado & Romo) |
| `ModifiedBandDepth` | Modified band depth (faster, approximation) |
| `DistanceBasedDepth` | Depth based on distance to center |
| `OutlyingnessBasedDepth` | Depth = 1 / (1 + outlyingness) |
| `ProjectionDepth` | Depth via random projections |
| `SimplicialDepth` | Simplicial depth (multivariate, less common for FDA) |
| `StahelDonohoOutlyingness` | Stahel-Donoho outlyingness measure (multivariate) |
| `BoxplotOutlierDetector` | Detect outliers using functional boxplot |
| `MSPlotOutlierDetector` | Magnitude-shape plot outlier detector |
| `directional_outlyingness_stats` | Compute directional outlyingness statistics |

### fdars Current Status

fdars has strong depth coverage in `src/depth/`:
- Fraiman-Muniz depth
- Modal depth
- Band depth (standard + modified)
- Random projection depth
- Streaming depth variants in `src/streaming_depth/`
- SPM-based outlier detection in `src/spm/`

**Gaps:**
- `DistanceBasedDepth` — unclear; likely gap
- `OutlyingnessBasedDepth` — gap
- `SimplicialDepth` — gap (unusual for FDA; low priority)
- `StahelDonohoOutlyingness` — gap
- `MSPlotOutlierDetector` — fdars has SPM outlier detection; MS-plot specifically may be gap
- `directional_outlyingness_stats` — gap

### Categorization

**Table Stakes:**
- Fraiman-Muniz / Integrated depth — fdars has this
- Band depth / Modified band depth — fdars has this
- Functional boxplot outlier detection — fdars has this via SPM
- Random projection depth — fdars has this

**Differentiators:**
- MS-plot (MagnitudeShapePlot) outlier detector — partial gap; important for functional outlier diagnosis
- Directional outlyingness — gap
- SimplicialDepth / StahelDonoho — gap; less commonly required

**Anti-features:** None; depth measures are core numeric operations.

**Porting complexity:** LOW-MEDIUM (most depth measures are straightforward numerical integrals or combinatorics).

---

## Area 6: Exploratory Analysis — Summary Statistics & Visualization

### scikit-fda Public API

| Class / Function | Purpose |
|-----------------|---------|
| `mean` | Functional mean (pointwise) |
| `gmean` | Geometric mean |
| `trim_mean` | Trimmed mean (depth-based trimming) |
| `depth_based_median` | Deepest function as median |
| `geometric_median` | Geometric (Fréchet) median |
| `fisher_rao_karcher_mean` | Fisher-Rao Riemannian mean (elastic mean) |
| `cov` | Functional covariance (bivariate function) |
| `var` | Functional variance |
| `std` | Functional standard deviation |
| `GraphPlot` | Plot functional data as curves |
| `ScatterPlot` | Scatter plot of functional data |
| `ParametricPlot` | Parametric (phase-space) plot |
| `Boxplot` | Functional boxplot |
| `SurfaceBoxplot` | Boxplot for surface (2D domain) functional data |
| `Outliergram` | Outliergram visualization |
| `MagnitudeShapePlot` | MS-plot for outlier diagnosis |
| `ClusterPlot` | Visualize clustering results |
| `ClusterMembershipLinesPlot` | Soft membership visualization |
| `ClusterMembershipPlot` | Membership as color-coded plot |
| `FPCAPlot` | Plot FPCA components |

### fdars Current Status

fdars has functional mean, covariance, and related statistics as internal helpers but these may not all be exposed as distinct public functions. Depth-based median via deepest-curve concept is present in depth module. Fisher-Rao Karcher mean is not exposed separately (elastic mean available via alignment).

All visualization classes (GraphPlot, Boxplot, MagnitudeShapePlot, etc.) are Python/matplotlib-based.

### Categorization

**Table Stakes (numeric):**
- Functional mean, variance, std — partial; verify public API exposure
- Trimmed mean — partial
- Depth-based median — partial
- Functional covariance (as a function, not just a matrix) — gap

**Differentiators:**
- Geometric median / Fréchet mean — gap
- Fisher-Rao Karcher mean — gap as standalone function (elastic alignment achieves it but not named as such)
- `geometric_mean` of curves — gap

**Anti-features (visualization layer — explicitly out of scope per PROJECT.md):**
- `GraphPlot`, `ScatterPlot`, `ParametricPlot`, `Boxplot`, `SurfaceBoxplot`, `Outliergram`, `MagnitudeShapePlot`, `ClusterPlot`, `FPCAPlot` — fdars is a Rust numeric library; matplotlib-dependent visualization is Python-specific. Deliberate non-goal. Consumers can plot results using their own plotting tools.

**Porting complexity:** LOW for statistics. Visualization — NOT to be ported.

---

## Area 7: Machine Learning — Classification

### scikit-fda Public API

| Class | Purpose |
|-------|---------|
| `KNeighborsClassifier` | Functional kNN with any functional metric (Lp, Fisher-Rao, etc.) |
| `RadiusNeighborsClassifier` | Fixed-radius neighbor classifier |
| `NearestCentroid` | Classify by closest class centroid |
| `DTMClassifier` | Distance-to-trimmed-means; outlier-robust |
| `MaximumDepthClassifier` | Assign to class with maximum depth |
| `DDClassifier` | Depth-vs-depth plot classifier |
| `DDGClassifier` | Generalized DD classifier (polynomial/any classifier in DD space) |
| `LogisticRegression` | Functional logistic regression |
| `QuadraticDiscriminantAnalysis` | Functional QDA |

No explicit LDA class — scikit-fda achieves LDA behavior via `NearestCentroid` with Mahalanobis distance.

### fdars Current Status

fdars has: LDA, QDA, kNN, kernel classifier, DD classifier — all fitting in FPC score space. No `RadiusNeighbors`, `NearestCentroid` as named public API, no `DTMClassifier`, no `DDGClassifier` (generalized DD). Functional logistic regression exists.

**Gaps:**
- `RadiusNeighborsClassifier` — gap
- `NearestCentroid` — gap (LDA in FPC space is similar but not identical)
- `DTMClassifier` (distance-to-trimmed-means) — gap; useful for robust classification
- `DDGClassifier` (generalized DD with arbitrary classifier in DD space) — gap

### Categorization

**Table Stakes:**
- kNN classifier with functional metric — fdars has this
- Depth-based classifiers (MaximumDepth, DD) — fdars has these
- Logistic regression — fdars has this
- QDA — fdars has this

**Differentiators:**
- `DTMClassifier` — gap; outlier-robust; moderately complex
- `DDGClassifier` — gap; generalization that allows any classifier in DD space; LOW porting complexity
- `RadiusNeighborsClassifier` — gap; LOW complexity, minor utility gain

**Anti-features:** None specific to classification.

**Porting complexity:** LOW for NearestCentroid, RadiusNeighbors. MEDIUM for DTMClassifier. LOW for DDGClassifier.

---

## Area 8: Machine Learning — Regression

### scikit-fda Public API

| Class | Purpose |
|-------|---------|
| `LinearRegression` | Scalar-on-function and function-on-scalar in one unified class; accepts functional predictors and responses; supports `LinearDifferentialOperator` regularization |
| `HistoricalLinearRegression` | Function-on-function regression using only past values as predictors (causal) |
| `KNeighborsRegressor` | Functional kNN regression |
| `RadiusNeighborsRegressor` | Fixed-radius kNN regression |
| `KernelRegression` | Functional kernel regression (Nadaraya-Watson style, scalar response) |
| `FPCARegression` | Project to FPCA scores, then OLS |
| `FPLSRegression` | Project to FPLS scores, then OLS |

scikit-fda does **not** have a general function-on-function regression — only the historical (causal) variant.

### fdars Current Status

fdars has: scalar-on-function linear (`fregre_lm`), functional logistic, FPCA-based regression, elastic regression. No kNN regressor, no kernel regression, no HistoricalLinearRegression, no FPLS regression.

**Gaps:**
- `KNeighborsRegressor` / `RadiusNeighborsRegressor` — gap
- `KernelRegression` — gap (Nadaraya-Watson for regression)
- `HistoricalLinearRegression` — gap; causal FDA regression model
- `FPLSRegression` — gap (requires FPLS first)
- `LinearRegression` with function-on-scalar direction (vectorial response) — partial; `function_on_scalar_2d.rs` covers 2D FOSR but not the general sklearn-compatible interface

### Categorization

**Table Stakes:**
- Scalar-on-function linear regression — fdars has this
- kNN regression — gap; expected given kNN classification exists
- Kernel regression — gap; standard non-parametric baseline

**Differentiators:**
- HistoricalLinearRegression — gap; moderately advanced; unique to FDA
- FPLSRegression — gap; requires FPLS implementation first
- Function-on-scalar with regularization (unified interface) — partial gap

**Anti-features:** None.

**Porting complexity:** LOW for kNN/kernel regression. MEDIUM for HistoricalLinearRegression (requires careful causal indexing). MEDIUM for FPLSRegression (depends on FPLS).

---

## Area 9: Machine Learning — Clustering

### scikit-fda Public API

| Class | Purpose |
|-------|---------|
| `KMeans` | Functional k-means with functional metrics |
| `FuzzyCMeans` | Fuzzy c-means for functional data |
| `NearestNeighbors` | Unsupervised neighbor search (for index building) |
| `AgglomerativeClustering` | Hierarchical clustering using functional distance matrix |

### fdars Current Status

fdars has GMM clustering (`GmmClusterConfig`). No k-means, fuzzy c-means, or hierarchical clustering exposed.

**Gaps:**
- `KMeans` — gap; most common clustering baseline
- `FuzzyCMeans` — gap (fdars has GMM which is a soft assignment method; fuzzy c-means is different)
- `AgglomerativeClustering` — gap
- `NearestNeighbors` (unsupervised) — gap

### Categorization

**Table Stakes:**
- Functional k-means — gap; expected baseline
- Fuzzy c-means — fdars has GMM as an alternative; fuzzy c-means is a gap

**Differentiators:**
- AgglomerativeClustering — gap; requires functional distance matrix; MEDIUM complexity

**Anti-features:** None.

**Porting complexity:** MEDIUM for k-means (requires functional mean and distance). LOW for fuzzy c-means if k-means is done first. MEDIUM for agglomerative (needs linkage logic on top of distance matrix).

---

## Area 10: Inference (Statistical Testing)

### scikit-fda Public API

| Function | Purpose |
|----------|---------|
| `oneway_anova` | One-way functional ANOVA (asymptotic) |
| `v_sample_stat` | V-statistic for functional ANOVA |
| `v_asymptotic_stat` | Asymptotic V-statistic |
| `hotelling_t2` | Functional Hotelling T² test (two-sample mean comparison) |
| `hotelling_test_ind` | Independent-sample Hotelling test |

### fdars Current Status

No statistical inference / hypothesis testing module in fdars.

### Categorization

**Table Stakes:**
- Functional ANOVA — gap; expected for any complete FDA toolkit
- Hotelling T² — gap; two-sample test is a standard need

**Differentiators:** None — these are core statistical tests.

**Anti-features:** None.

**Porting complexity:** MEDIUM (requires functional inner products and asymptotic distribution computation; well-documented in literature).

---

## Area 11: Metrics & Norms

### scikit-fda Public API

| Class / Function | Purpose |
|-----------------|---------|
| `LpNorm` | Lp norm for functional data (p=1,2,∞) |
| `LpDistance` | Lp distance |
| `MahalanobisDistance` | Mahalanobis distance via covariance |
| `NormInducedMetric` | Metric induced by any norm |
| `PairwiseMetric` | Compute full pairwise distance matrix |
| `TransformationMetric` | Apply transform then compute metric |
| `lp_norm` | Functional Lp norm (function) |
| `lp_distance` | Functional Lp distance (function) |
| `angular_distance` | Angular distance between functions |
| `fisher_rao_distance` | Fisher-Rao geodesic distance |
| `fisher_rao_amplitude_distance` | Fisher-Rao amplitude component |
| `fisher_rao_phase_distance` | Fisher-Rao phase component |
| `inner_product` | L2 inner product |
| `inner_product_matrix` | Gram matrix of inner products |
| `cosine_similarity` | Cosine similarity |
| `cosine_similarity_matrix` | Pairwise cosine similarity matrix |

### fdars Current Status

fdars computes L2 inner products (via Simpson weights), Mahalanobis distance (in QDA), elastic/Fisher-Rao distances (in alignment module). `lp_norm` and `lp_distance` are not exposed as standalone public functions. No `PairwiseMetric` abstraction for computing distance matrices.

**Gaps:**
- Standalone `lp_norm`, `lp_distance` — gap; useful utilities
- `angular_distance`, `cosine_similarity` — gap
- `PairwiseMetric` (distance matrix computation) — gap; required for kNN and agglomerative clustering
- `TransformationMetric` — gap; low priority

### Categorization

**Table Stakes:**
- Lp norms/distances — gap as standalone public API
- Pairwise distance matrix computation — gap; required by kNN and clustering
- Fisher-Rao distance — fdars has via alignment

**Differentiators:**
- `TransformationMetric` — gap; low priority; composable but niche

**Anti-features:** None.

**Porting complexity:** LOW for Lp norms and distance matrix. MEDIUM for angular/cosine when applied to functional data with integration weights.

---

## Area 12: Infrastructure — Datasets, Covariances, Operators, Regularization

### scikit-fda Public API

**Covariance functions (for Gaussian process generation):**
`Brownian`, `Exponential`, `Gaussian` (RBF), `Matern`, `Linear`, `Polynomial`, `WhiteNoise`, `Covariance` (base)

**Operators:**
`Identity`, `LinearDifferentialOperator` (e.g., penalize second derivative), `SRSF` (square-root slope function)

**Regularization:**
`L2Regularization` (Tikhonov/ridge, used with `LinearDifferentialOperator` in regression/smoothing/FPCA)

**Datasets (fetch):**
`fetch_aemet`, `fetch_gait`, `fetch_growth`, `fetch_handwriting`, `fetch_mco`, `fetch_medflies`, `fetch_nox`, `fetch_octane`, `fetch_phoneme`, `fetch_tecator`, `fetch_weather`, `fetch_bone_density`, `fetch_cran`, `fetch_ucr`

**Data generation:**
`make_gaussian`, `make_gaussian_process`, `make_sinusoidal_process`, `make_multimodal_samples`, `make_multimodal_landmarks`, `make_random_warping`, `make_sde_trajectories` (Itô SDE via Euler-Maruyama/Milstein, added v0.10.0)

**Scoring metrics (sklearn-compatible):**
`explained_variance_score`, `mean_absolute_error`, `mean_absolute_percentage_error`, `mean_squared_error`, `mean_squared_log_error`, `r2_score`

### fdars Current Status

fdars has 28 examples and data embedded in test fixtures. No named benchmark datasets as `fetch_*` functions. No formal Gaussian process simulation module (though some test data generation exists). `LinearDifferentialOperator` concept used internally in smoothing but not exposed as a user-facing operator object. `SRSF` operator is implemented internally in elastic module.

**Gaps:**
- Named benchmark dataset loaders (`fetch_*`) — gap; important for reproducible benchmarking and user onboarding
- Gaussian process data generation with named covariance kernels — gap
- SDE trajectory generation — gap (advanced)
- `LinearDifferentialOperator` as a user-facing composable operator — gap
- Scoring functions as a public utility module — gap; consumers can compute themselves

### Categorization

**Table Stakes:**
- A few standard benchmark datasets — gap; important for user onboarding and reproducibility
- Basic data generation (sinusoidal, Gaussian process) — gap for public API

**Differentiators:**
- `LinearDifferentialOperator` as composable operator for penalties — gap; important enabler for penalized smoothing/FPCA
- SDE trajectory generation — gap; advanced; low priority initially
- Named covariance kernels (Matern, Brownian, etc.) — gap; useful for simulation studies

**Anti-features:**
- Mirroring all 14 `fetch_*` datasets by re-hosting data in a Rust crate — licensing and binary size concerns; instead, expose a loader API that reads user-provided files in standard formats (CSV, npy)

**Porting complexity:** LOW for data generation helpers. MEDIUM for `LinearDifferentialOperator` abstraction. LOW for scoring utilities.

---

## Feature Landscape Summary

### Table Stakes (Any Serious FDA Library Must Have)

| Feature | fdars Status | Complexity to Add | Gap Priority |
|---------|-------------|-------------------|--------------|
| Discretized functional data representation (grid) | Partial (`FdMatrix`) | HIGH — requires type refactor | P2 |
| Basis systems as composable objects (BSpline, Fourier, Monomial) | Partial (internal) | HIGH — architectural | P2 |
| Grid-to-basis conversion | Gap | HIGH | P2 |
| Spline interpolation at arbitrary query points | Gap | MEDIUM | P2 |
| Kernel smoothing (Nadaraya-Watson) | Gap | MEDIUM | P1 |
| Basis smoothing with derivative penalty | Gap (internal only) | MEDIUM | P1 |
| CV bandwidth selection (LOO/GCV) | Gap | MEDIUM | P2 |
| Shift registration | Gap | LOW | P1 |
| Landmark registration | Gap | LOW-MEDIUM | P1 |
| Registration quality validation | Gap | MEDIUM | P2 |
| FPCA as standalone transformer | Partial (embedded in regression) | LOW | P1 |
| FPLS | Gap | MEDIUM | P1 |
| Depth: IntegratedDepth, BandDepth, ModifiedBandDepth | Present | — | Done |
| Depth: ProjectionDepth | Present | — | Done |
| Outlier detection (MS-plot, boxplot) | Partial (SPM-based) | LOW | P2 |
| Functional mean, var, std (public API) | Partial | LOW | P1 |
| Trimmed mean, depth-based median | Partial | LOW | P1 |
| Functional covariance (as bivariate function) | Gap | MEDIUM | P2 |
| kNN classifier with functional metric | Present | — | Done |
| Depth-based classifiers (MaxDepth, DD) | Present | — | Done |
| Logistic regression | Present | — | Done |
| QDA | Present | — | Done |
| DTMClassifier | Gap | MEDIUM | P2 |
| DDGClassifier | Gap | LOW | P2 |
| Scalar-on-function linear regression | Present | — | Done |
| kNN regression | Gap | LOW | P1 |
| Kernel regression | Gap | MEDIUM | P1 |
| HistoricalLinearRegression | Gap | MEDIUM | P2 |
| FPLSRegression | Gap (depends on FPLS) | MEDIUM | P2 |
| Functional k-means | Gap | MEDIUM | P1 |
| Fuzzy c-means | Gap | MEDIUM | P2 |
| Functional ANOVA (oneway) | Gap | MEDIUM | P2 |
| Hotelling T² test | Gap | MEDIUM | P2 |
| Lp norms/distances (public functions) | Gap (internal only) | LOW | P1 |
| Pairwise distance matrix | Gap | LOW | P1 |
| Standard benchmark datasets (fetch_*) | Gap | LOW-MEDIUM | P2 |
| Gaussian process data generation | Gap | LOW | P2 |

### Differentiators (Advanced / Less Common)

| Feature | fdars Status | Notes |
|---------|-------------|-------|
| Elastic/SRSF registration (FisherRaoElasticRegistration) | Present | Strong coverage |
| Elastic regression & FPCA | Present | Differentiator fdars has |
| GMM clustering | Present | scikit-fda uses k-means/fuzzy; fdars has GMM |
| Streaming depth measures | Present | Not in scikit-fda |
| Model explainability (PDP, SHAP, LIME, ALE) | Present | Not in scikit-fda |
| SPM / control charts | Present | Not in scikit-fda |
| Seasonal decomposition (STL) | Present | Not in scikit-fda |
| Irregular FDA (FDataIrregular) | Partial | scikit-fda added v0.10.0; fdars has basic module |
| DiffusionMap dimensionality reduction | Gap | Added in scikit-fda v0.10.0; manifold learning |
| Variable selection (MaximaHunting, mRMR) | Gap | Useful for interpretable FDA |
| Feature construction transformers | Gap | Pipeline utilities; lower priority |
| SDE trajectory generation | Gap | Advanced simulation |
| LinearDifferentialOperator (composable) | Gap | Key enabler for penalized methods |
| Mixed-effects irregular→basis conversion | Gap | Advanced preprocessing |
| AgglomerativeClustering | Gap | Straightforward with distance matrix |
| SimplicialDepth | Gap | Less common; low priority |

### Anti-Features (Deliberately Out of Scope for fdars)

| Feature | Why It Exists in scikit-fda | Why fdars Should Not Chase It | Alternative |
|---------|----------------------------|-------------------------------|-------------|
| matplotlib visualization (`GraphPlot`, `Boxplot`, `MagnitudeShapePlot`, etc.) | Python ecosystem standard; FDA users expect plotting in the same library | fdars is a numeric Rust library; no graphics runtime; PROJECT.md explicitly lists as out-of-scope | Expose numeric results (depth values, FPCA scores, cluster labels) that consumers can plot using their preferred tools |
| sklearn pipeline API (`.fit()`, `.transform()`, `Pipeline`, `GridSearchCV`) | Python ecosystem convention; enables composability with sklearn | Rust equivalent is trait composition; a literal sklearn API clone is unidiomatic and brittle | Design Rust traits (e.g., `FdaTransformer`) that are idiomatic and composable within Rust's type system |
| Python-style `__repr__`, `__getitem__`, `__add__` magic methods on data types | Python convention | Implement Rust standard traits (`Display`, `Index`, `Add`) idiomatically instead | Rust operator overloading and trait impls |
| `fetch_*` dataset loaders that bundle data in the library | Convenience for Python users; datasets packaged with PyPI distribution | Binary data in a Rust crate inflates crate size; licensing concerns; CRAN-distributed data has its own terms | Document where to obtain datasets; provide loader functions that accept file paths or standard formats |
| `PerClassTransformer` / `FDAFeatureUnion` pipeline metaclasses | Scikit-learn pipeline conventions | Rust generics and trait bounds express this more naturally without Python metaclass machinery | Trait-based transformer composition |

---

## Feature Dependencies

```
Basis systems (BSpline, Fourier, Monomial)
    └──requires──> Grid-to-basis conversion
    └──enables──>  Basis smoothing with penalty
    └──enables──>  LinearDifferentialOperator as operator
                       └──enables──> Penalized FPCA
                       └──enables──> Penalized smoothing/regression

FPLS
    └──requires──> FPCA infrastructure (already present)
    └──enables──>  FPLSRegression

Lp distance / Pairwise distance matrix
    └──requires──> Integration weights (already present)
    └──enables──>  kNN regression
    └──enables──>  Functional k-means
    └──enables──>  AgglomerativeClustering

Functional k-means
    └──requires──> Functional mean (already present)
    └──requires──> Lp distance

Functional ANOVA / Hotelling T²
    └──requires──> Functional mean (already present)
    └──requires──> Functional covariance

Shift registration
    └──no dependencies beyond FdMatrix
    └──enables──>  Landmark registration (shift variant)

HistoricalLinearRegression
    └──requires──> Scalar-on-function regression infrastructure (present)

Outlier detection (MS-plot)
    └──requires──> Depth measures (present)
    └──requires──> Directional outlyingness stats
```

---

## Biggest Likely Gaps for fdars (Gap Analysis Head Start)

Ranked by expected impact on a user migrating from scikit-fda:

1. **Smoothing module** — Nadaraya-Watson, basis smoothing, CV bandwidth selection. Any real FDA workflow starts with smoothing raw data. This is the highest-impact missing area.

2. **Public Lp norm/distance functions + pairwise distance matrix** — These are required by kNN regression, k-means clustering, and agglomerative clustering. Unblocking this unblocks a chain of other features.

3. **Functional k-means clustering** — Scikit-fda's most-used clustering method; fdars has only GMM. Gap is wide and high visibility.

4. **FPLS + FPLSRegression** — Companion to FPCA/FPCARegression; expected by users coming from R/Python FDA ecosystems.

5. **Shift registration + landmark registration** — The SRSF elastic registration exists but the simpler shift/landmark methods are missing; users often want the cheaper method first.

6. **kNN regression + Kernel regression** — Simpler regression baselines expected alongside the linear model.

7. **Statistical inference (ANOVA, Hotelling T²)** — No hypothesis testing at all; significant gap for a scientific computing library.

8. **FPCA as a standalone public transformer** — Currently embedded inside regression result types; should be extractable and usable independently in any pipeline.

9. **Standard benchmark dataset loaders** — Critical for reproducibility, examples, and user onboarding. Low implementation cost.

10. **Representation layer formalization** — The `FdMatrix` + separate domain-specific usage is workable but not as discoverable as scikit-fda's `FDataGrid`/`FDataBasis` distinction; medium-term architectural work.

---

## Sources

- [scikit-fda API Reference (stable 0.10.1)](https://fda.readthedocs.io/en/stable/apilist.html)
- [scikit-fda Registration docs](https://fda.readthedocs.io/en/latest/modules/preprocessing/registration.html)
- [scikit-fda Feature Construction docs](https://fda.readthedocs.io/en/stable/modules/preprocessing/feature_construction.html)
- [scikit-fda Representation docs](https://fda.readthedocs.io/en/latest/modules/representation.html)
- [scikit-fda GitHub Releases](https://github.com/GAA-UAM/scikit-fda/releases)
- [scikit-fda paper (arXiv:2211.02566)](https://arxiv.org/abs/2211.02566)
- fdars codebase: `.planning/codebase/ARCHITECTURE.md` (2026-08-07)
- fdars milestone: `.planning/PROJECT.md` (v0.14.0 audit milestone)

---

*Feature research for: FDA library capability gap analysis (fdars vs scikit-fda 0.10.1)*
*Researched: 2026-08-07*
*Confidence: MEDIUM — scikit-fda API verified against official docs; fdars coverage inferred from codebase map*
