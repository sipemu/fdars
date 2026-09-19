# Feature Research: v0.47.0 Cross-Language Benchmark Harness & Optimization Pass

**Domain:** Cross-language performance benchmarking for Rust FDA library vs Python scikit-fda and R FDA ecosystem
**Milestone:** v0.47.0 (benchmark harness + optimization pass)
**Researched:** 2026-09-19
**Confidence:** MEDIUM — scikit-fda API verified from official readthedocs (MEDIUM confidence via context7/web); R package operations confirmed from CRAN Task View and package documentation (LOW confidence via websearch, cross-checked across multiple sources); benchmark methodology from established projects (benchLANG, ASV, hyperfine); no independent performance numbers exist for fdars vs scikit-fda.

---

## Overlapping Operation Matrix

This table is the primary deliverable. It maps each benchmarkable FDA operation to its implementation across all three ecosystems. Only operations that exist in at least two ecosystems — or have a clear fdars-side equivalent with measurement value — are included.

### Legend
- `fdars-core` = the Rust crate under optimization
- `scikit-fda` = Python package (v0.10.x)
- `fda` / `fda.usc` / `refund` / `fdapace` / `roahd` / `fdasrvf` = R packages
- `EXISTS` = direct equivalent with matching semantics
- `PARTIAL` = exists but with different assumptions or narrower scope
- `ABSENT` = no published equivalent; operation is exclusive to one ecosystem

| # | Operation | fdars-core | scikit-fda (Python) | R ecosystem | Notes / Fairness caveat |
|---|-----------|------------|---------------------|-------------|-------------------------|
| B01 | **Basis smoothing — B-spline** | `smooth_fd` / `BSplineBasis` backend | `BasisSmoother` + `BSplineBasis` | `fda::smooth.basis`, `refund::fbps` | All three use penalized least squares; verify penalty order matches |
| B02 | **Basis smoothing — Fourier** | `smooth_fd` / `FourierBasis` backend | `BasisSmoother` + `FourierBasis` | `fda::smooth.basis` (Fourier) | Identical algorithm; smoothing parameter selection may differ |
| B03 | **FPCA — dense** | `fdata_to_pc_1d` (nalgebra SVD) / `fdata_to_pc_1d` with `linalg` feature (faer thin_svd) | `skfda.preprocessing.dim_reduction.FPCA` | `fda::pca.fd`, `fdapace::FPCA` (Dense), `refund::fpca.sc`, `refund::fpca.face` | All center data and perform eigendecomposition of covariance; integration weights must match |
| B04 | **Functional depth — Fraiman-Muniz** | `fraiman_muniz_1d` | `skfda.exploratory.depth.IntegratedDepth` | `fda.usc::depth.FM`, `roahd` (via MBD family) | Direct equivalent in all three; verify integration method (trapezoidal vs Simpson's) |
| B05 | **Functional depth — Band / Modified Band** | `band_depth_1d`, `modified_band_depth` | `skfda.exploratory.depth.BandDepth`, `ModifiedBandDepth` | `fda.usc::depth.mode`, `roahd::MBD` | MBD semantics identical; verify J=2 band parameter default |
| B06 | **Functional boxplot / outlier detection** | `functional_boxplot`, `spm_phase1` outlier flags | `skfda.exploratory.depth.OutlyingnessBasedDepth` + outlier step | `roahd::fbplot`, `fda.usc::Outliers.fdata` | Functional boxplot = depth-sorted envelope + 1.5*IQR fence; verify IQR factor |
| B07 | **Scalar-on-function regression — FPCR** | `fregre_lm` (FPCR path) | `skfda.ml.regression.FPCARegression` | `refund::pfr(lf(...))`, `fda.usc::fregre.pc` | All project onto FPCs then fit OLS; comparison is fair if ncomp is fixed |
| B08 | **Scalar-on-function regression — kernel** | `fregre_kernel` | `skfda.ml.regression.KernelRegression` | `fda.usc::fregre.np` | Nonparametric; bandwidth must match across implementations |
| B09 | **Elastic alignment — Fisher-Rao / SRVF** | `elastic_align_pair` / `elastic_align_group` | `skfda.preprocessing.registration.FisherRaoElasticRegistration` | `fdasrvf::time_warping`, `fdasrvf::align_fPCA` | This is the known #1 fdars bottleneck; O(N²·M²); all use SRVF — verify DP vs gradient approach |
| B10 | **Karcher mean (elastic)** | `elastic_karcher_mean` | `skfda.exploratory.stats.fisher_rao_karcher_mean` | `fdasrvf::time_warping` (template = Karcher mean) | Direct equivalent; used as subroutine of elastic align |
| B11 | **Clustering — k-means on curves** | `gmm_cluster` / functional k-means | `skfda.ml.clustering.KMeans`, `FuzzyCMeans` | `fda.usc::kmeans.fd` | Distance metric must match (L2 vs elastic); use L2 for fair comparison |
| B12 | **Classification — kNN** | `fclassif_knn` | `skfda.ml.classification.KNeighborsClassifier` | `fda.usc::classif.knn` | Direct equivalent; L2 distance by default |
| B13 | **Covariance / FPCA — PACE (sparse)** | `pace_fpca` | ABSENT (scikit-fda has no sparse PACE) | `fdapace::FPCA` (Sparse) | fdars vs R only; scikit-fda comparison not applicable |

### Operations exclusive to fdars-core (no comparison target)

| Operation | Why not benchmarkable cross-language |
|-----------|--------------------------------------|
| Soft-DTW barycenter | No scikit-fda or R equivalent with matching semantics |
| SPM control charts (MEWMA, MCUSUM) | No equivalent in scikit-fda; partial in `spc` R pkg (not FDA-specific) |
| Wavelet-domain regression (wcr/wnet) | `refund` has `wcr`/`wnet` but different penalty; defer to parity audit |
| Reverse-mode autodiff | No scikit-fda or R FDA equivalent |
| Elastic FPCA (joint alignment + PCA) | `fdasrvf::align_fPCA` is the only R equivalent; absent from scikit-fda |

---

## Table Stakes

Features that a credible cross-language benchmark harness must have. Missing any of these = results are not publishable or reproducible.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **Identical input across all languages** | Without the same data, speedup numbers are fabricated | LOW | Generate in Rust/Python with a seeded PRNG, serialize to CSV/NPZ/RDS, load in all runners |
| **Pinned library versions** | Results shift when upstream changes; must cite exact versions | LOW | Python: `requirements.txt` or `pyproject.toml` locked; R: `renv.lock`; Rust: `Cargo.lock` |
| **Warmup before timing** | Python and R have JIT/import overhead that inflates the first call | LOW | One untimed warmup call per operation before the timed loop |
| **Multiple trials + median reported** | Single-run timing is unreliable on a modern OS | LOW | Minimum 5 timed trials; report median and stddev |
| **Wall-clock time as primary metric** | This is what the user experiences | LOW | `std::time::Instant` (Rust), `timeit` (Python), `proc.time()` (R) |
| **Thread-count parity** | Rayon uses all cores by default; scikit-fda uses NumPy/LAPACK threads; R uses OpenBLAS | MEDIUM | Set `RAYON_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1` for single-thread baseline; provide separate parallel run |
| **Numeric agreement validation** | Without it you may be timing different computations | MEDIUM | For each operation, verify outputs agree to relative tolerance ≤ 1e-3 (eigenvalues/scores/depth values) before publishing speedup numbers |
| **Environment disclosure** | Reproducibility requires knowing the machine | LOW | Record OS, CPU model, RAM, kernel version, toolchain versions in a `bench/env.json` |
| **Reproduction instructions** | Others must be able to re-run | LOW | README with `make bench` or `./bench/run.sh` entry point |
| **Per-operation speedup table** | The primary output of the comparison | LOW | `fdars vs scikit-fda` and `fdars vs best-R` columns; negative results are fine |

## Differentiators

Features that make the benchmark harness and report stand out beyond the minimum.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Before / after optimization comparison** | Proves the optimization pass actually helped; the killer evidence for the milestone | MEDIUM | Run harness against the pre-optimization Rust binary and the post-optimization binary; show both columns |
| **Peak RSS measurement** | Memory footprint matters for production deployments | MEDIUM | Use `/usr/bin/time -v` (Linux) or `getrusage(RUSAGE_CHILDREN)` wrapper; report peak RSS in MB alongside wall-clock |
| **N×M scaling table** | Shows which library degrades faster at scale | MEDIUM | Run each operation at 3 input sizes (small/medium/large); plot or tabulate scaling exponents |
| **Parallel vs single-thread breakdown** | Shows Rayon's contribution; prevents cherry-picking | LOW | Two harness runs: `RAYON_NUM_THREADS=1` and default; report both |
| **Numeric agreement appendix** | Builds credibility; shows the comparison is like-for-like | LOW | Table of max relative error per operation across all library pairs |
| **Existing criterion suite integration** | Links the external harness to the internal benchmarks already in-crate | LOW | criterion before/after numbers appear alongside cross-language numbers in the report |
| **HTML / Markdown report auto-generation** | Makes results sharable as a repo artifact | MEDIUM | `bench/report.md` written by the harness driver; tables formatted for GitHub rendering |
| **fdasrvf as elastic alignment reference** | fdasrvf is the gold-standard R elastic analysis package; more relevant than scikit-fda for B09 | LOW | Include fdasrvf::time_warping in B09/B10 even though it is not in the "named" primary list |

## Anti-Features

Features that appear useful but should not be built for this milestone.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| **ASV (airspeed velocity) integration** | Tracks performance over commits automatically | Adds a Python tool as a permanent CI dependency; the milestone goal is a one-shot comparison report, not continuous tracking | Generate a self-contained `bench/report.md` per run; CI can re-run on demand |
| **Docker / Nix environment** | Guarantees identical OS across machines | Adds days of infra work; not needed for a credible single-machine report | Document the `env.json` snapshot; pin versions explicitly in lockfiles |
| **Exhaustive operation coverage (all 70+ fdars functions)** | More = better benchmark | Unmaintainable; most operations have no cross-language equivalent | Benchmark the 13 overlapping operations above plus 3–5 optimization targets from the criterion hotspot list |
| **Micro-allocation benchmarks** | Shows allocation count; useful for memory tuning | Requires `dhat` / `binggan` integration in Rust; `/proc/self/statm` polling in Python; impractical cross-language | Report peak RSS only; allocation detail stays in the Rust-only criterion pass |
| **Statistical significance testing across runs** | Makes speedup claims rigorous | Requires 30+ trials per operation per library, total harness runtime becomes hours | Report median ± stddev from 5–10 trials; add a disclaimer that results are indicative |
| **Interactive web frontend (ASV-style)** | Nice for browsing regression history | Requires JavaScript build tooling; overkill for a one-shot milestone report | Static `bench/report.md` in the repo; embed criterion HTML link |

---

## Input-Size Grid

Based on research into FDA benchmark practice and known fdars complexity:

| Level | N (curves) | M (grid points) | Use | Notes |
|-------|-----------|----------------|-----|-------|
| **Small** | 50 | 50 | Smoke test; confirms harness works | All 13 operations finish in <5 s per library |
| **Medium** | 200 | 100 | Primary benchmark | Comparable to typical published FDA studies; elastic alignment feasible at ~10–30 s |
| **Large** | 500 | 200 | Stress / bottleneck identification | Known fdars elastic-alignment bottleneck at this size; expect minutes in naive path |
| **Stress** | 1000 | 500 | Elastic alignment only | Skip all others at this size; proves the optimization fix at scale |

**Data generation scheme:** For each size, generate Gaussian process curves on a uniform grid [0,1] with squared-exponential covariance `k(t,s) = exp(-|t-s|²/(2*l²))`, `l = 0.2`. Generate the seed matrix in Python (or Rust) and serialize to CSV so all three language drivers load the exact same floating-point values. This eliminates PRNG-parity complexity across language implementations.

---

## Metric Set

| Metric | Primary | How to measure | Threshold for publication |
|--------|---------|----------------|--------------------------|
| **Wall-clock time (median)** | Yes | `timeit` / `proc.time()` / `Instant`; median of 5–10 trials | Must be reported for all 13 operations |
| **Wall-clock stddev** | Yes | Same trials | Report; flag if CV > 15% (noisy machine) |
| **Speedup factor vs slowest** | Yes | `t_slowest / t_fdars` | Report as "fdars is Xx faster/slower than Y" |
| **Peak RSS (MB)** | Yes for large size | `/usr/bin/time -v` or `getrusage` wrapper | Report for medium + large; omit for small |
| **Numeric agreement (max rel error)** | Yes for validity | Compare output arrays per operation across all pairs | Must be ≤ 1e-3 for timing comparison to be publishable; document any that fail |
| **Allocation count (Rust only)** | No | `dhat`; criterion `iter_with_setup` | Keep in criterion suite; not cross-language |
| **Throughput (curves/sec)** | Optional | `N / median_time` | Useful for scalar-on-function and depth; compute from timing |

---

## Numeric Agreement Protocol

This is mandatory for a credible comparison. The harness must validate before reporting speedup.

For each operation B01–B13, after one library run:
1. Save the primary output array (eigenvalues for FPCA, depth scores for depth, fitted values for regression, warped curves for elastic alignment).
2. Load the same-input output from a reference library (use scikit-fda as reference for Python; fdapace as reference for FPCA in R).
3. Check `max(|a - b|) / (|a| + |b| + 1e-10) < tolerance`.
4. Tolerance = `1e-3` for depth scores and regression coefficients; `1e-2` for elastic alignment (warping is non-unique up to group action).
5. Sign ambiguity in FPCA eigenfunctions: flip sign of each component to maximize dot product with reference before comparing.
6. Record results in `bench/numeric_agreement.md`.

Operations where numeric agreement cannot be established (e.g., because algorithms differ fundamentally) must be flagged in the report as "timing only, no agreement check."

---

## Feature Dependencies

```
Input generation (CSV serialization)
    └──required by──> All 13 benchmark operations

Numeric agreement validation
    └──required before──> Speedup table publication

Single-thread timing (RAYON_NUM_THREADS=1)
    └──required before──> Parallel timing (default threads)
    (establish baseline first, then layer parallelism)

Large-size run (N=500, M=200)
    └──required before──> Optimization pass targets
    (need the bottleneck evidence before choosing what to fix)

Criterion before/after
    └──required by──> Published perf report (optimization section)
```

---

## MVP Definition

### Launch With (Phase 1 of milestone)

Minimum viable harness that produces publishable numbers for the core operations.

- [ ] Input generation: Python script produces CSV for each size grid (seeded Gaussian process)
- [ ] B03: FPCA (dense) — fdars vs scikit-fda vs fdapace
- [ ] B04: Fraiman-Muniz depth — fdars vs scikit-fda vs fda.usc
- [ ] B09: Elastic alignment — fdars vs scikit-fda vs fdasrvf (the #1 bottleneck)
- [ ] B07: Scalar-on-function FPCR — fdars vs scikit-fda vs refund
- [ ] Numeric agreement check for B03, B04, B09, B07
- [ ] `bench/report.md` with environment + timing + speedup columns

### Add After Bottleneck Evidence (Phase 2)

- [ ] B01/B02: Basis smoothing
- [ ] B05/B06: Modified band depth + functional boxplot
- [ ] B08: Kernel regression
- [ ] B10: Karcher mean
- [ ] Peak RSS measurement for large size
- [ ] Before/after optimization comparison (requires optimization pass to be done)

### Future Consideration

- [ ] B11: k-means clustering (scikit-fda uses L2; fdars uses GMM by default — requires special-casing)
- [ ] B13: PACE sparse FPCA vs fdapace (requires sparse data generation)
- [ ] ASV integration for continuous tracking

---

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| B09 Elastic alignment benchmark | HIGH (known bottleneck) | LOW (one driver per ecosystem) | P1 |
| B03 FPCA benchmark | HIGH (SVD is second bottleneck) | LOW | P1 |
| Numeric agreement validation | HIGH (credibility) | MEDIUM | P1 |
| Input CSV generation | HIGH (reproducibility) | LOW | P1 |
| Before/after optimization numbers | HIGH (proves value) | LOW (run harness twice) | P1 |
| B04/B05/B06 Depth benchmarks | MEDIUM | LOW | P2 |
| B07 Regression benchmark | MEDIUM | LOW | P2 |
| Peak RSS measurement | MEDIUM | LOW | P2 |
| N×M scaling table | MEDIUM | LOW (3 harness runs) | P2 |
| B01/B02 Smoothing benchmarks | LOW (less interesting) | LOW | P3 |
| Parallel vs single-thread breakdown | MEDIUM | LOW | P2 |
| HTML report | LOW | MEDIUM | P3 |

---

## Competitor Feature Analysis

| Feature | scikit-fda (Python) | R ecosystem (best package) | fdars-core (Rust) |
|---------|---------------------|---------------------------|-------------------|
| FPCA (dense) | FPCA class, NumPy/LAPACK SVD | fdapace::FPCA (dense), pca.fd | fdata_to_pc_1d; faer thin_svd (1.8–4.1× faster per v0.14 audit) |
| Basis smoothing | BasisSmoother (BSpline, Fourier) | fda::smooth.basis | smooth_fd (existing) |
| Fraiman-Muniz depth | IntegratedDepth (default) | fda.usc::depth.FM | fraiman_muniz_1d |
| Modified band depth | ModifiedBandDepth | roahd::MBD | modified_band_depth |
| Functional boxplot | depth-based (OutlyingnessBasedDepth) | roahd::fbplot | functional_boxplot |
| Scalar-on-function regression (FPCR) | FPCARegression | refund::pfr, fda.usc::fregre.pc | fregre_lm (FPCR path) |
| Elastic registration (Fisher-Rao) | FisherRaoElasticRegistration | fdasrvf::time_warping | elastic_align_pair/group |
| Elastic Karcher mean | fisher_rao_karcher_mean | fdasrvf mean computation | elastic_karcher_mean |
| PACE / sparse FPCA | ABSENT | fdapace::FPCA (sparse) | pace_fpca |
| Soft-DTW barycenter | ABSENT | ABSENT | soft_dtw_barycenter |
| SPM (MEWMA/MCUSUM) | ABSENT | spc package (not FDA-specific) | spm_phase1/phase2 |
| Classification (kNN) | KNeighborsClassifier | fda.usc::classif.knn | fclassif_knn |

---

## Sources

- scikit-fda 0.10.1 API Reference: [https://fda.readthedocs.io/en/stable/apilist.html](https://fda.readthedocs.io/en/stable/apilist.html) — MEDIUM confidence
- scikit-fda paper (arXiv 2211.02566): [https://arxiv.org/pdf/2211.02566](https://arxiv.org/pdf/2211.02566) — MEDIUM confidence
- CRAN Task View: Functional Data Analysis: [https://cran.r-project.org/web/views/FunctionalData.html](https://cran.r-project.org/web/views/FunctionalData.html) — LOW confidence
- roahd R Journal paper: [https://journal.r-project.org/articles/RJ-2019-032/](https://journal.r-project.org/articles/RJ-2019-032/) — LOW confidence (cross-checked)
- fdasrvf CRAN: [https://www.rdocumentation.org/packages/fdasrvf/versions/2.4.4](https://www.rdocumentation.org/packages/fdasrvf/versions/2.4.4) — LOW confidence
- refund pfr docs: [https://rdrr.io/cran/refund/man/pfr.html](https://rdrr.io/cran/refund/man/pfr.html) — LOW confidence
- ASV (airspeed velocity): [https://github.com/airspeed-velocity/asv](https://github.com/airspeed-velocity/asv) — LOW confidence
- benchLANG cross-language methodology: [https://github.com/GlebOlsen/benchLANG](https://github.com/GlebOlsen/benchLANG) — LOW confidence
- hyperfine + /usr/bin/time -v for RSS: [https://configcrate.com/hyperfine-bench-stats.html](https://configcrate.com/hyperfine-bench-stats.html) — LOW confidence
- Binggan (Rust peak memory benchmarking): [https://github.com/PSeitz/binggan](https://github.com/PSeitz/binggan) — LOW confidence

---
*Feature research for: v0.47.0 cross-language FDA performance benchmark harness*
*Researched: 2026-09-19*
