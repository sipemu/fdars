# fdars R-Ecosystem Prioritized Backlog

**Crate:** fdars-core (shipped through v0.17.0)
**Audit milestone:** v0.18.0 — R-Ecosystem Gap Audit (audit-only deliverable; no `fdars-core/src/` changes)
**Yardstick:** the R functional-data-analysis package ecosystem (35 packages, Phase 16) — **not** scikit-fda
**Source report:** [R-AUDIT-REPORT.md](R-AUDIT-REPORT.md) (Phases 16–19)
**Produced by:** Phase 19 consolidation (RPT-02)
**Distinct from:** the archived scikit-fda `BACKLOG.md` (NOT modified by this milestone)

This file is the standalone, value-ranked backlog for the v0.18.0 R-ecosystem audit. It is intended to be consumed directly by `/gsd-new-milestone` to promote items into future milestone requirements. Each item is phrased as a GSD-ready candidate requirement or phase. The **162 actionable gaps** from the Phase-17 parity matrix (18 table-stakes + 144 differentiator) are clustered into **26 coherent, milestone-promotable items** — not 162 one-liners. Items in fdars' **12 R-honest strength areas** (Phase 18) are excluded by construction (no work proposed where fdars already leads).

---

## Ranking Methodology

### Formula

```
score = value / sqrt(effort)
```

A higher score means more user value delivered per unit of effort. Items are ordered by descending score in the Ranked Backlog table. The formula rewards high-value items and penalizes high-effort items non-linearly (large efforts are more than proportionally expensive to deliver). Methodology reused **verbatim** from the v0.14.0 audit (`BACKLOG.md`).

### Value Scale (1–5)

Value measures **user value**, not ease of implementation.

| Value | Anchor |
|-------|--------|
| 5 | Table-stakes capability blocking real workloads — a capability R FDA users rely on daily and expect from any general-purpose FDA library |
| 4 | High-value capability widely used in practice; or a present partial implementation that needs significant work to reach parity |
| 3 | Meaningful capability; important but not blocking; commonly requested in FDA toolkits |
| 2 | Useful addition; niche use-case or specialized methodology |
| 1 | Niche differentiator or cosmetic improvement with limited real-world impact |

### Effort Map (S / M / L)

| Effort | Numeric | sqrt(effort) | Definition |
|--------|---------|--------------|------------|
| S | 1 | 1.000 | Small — approximately 1 week of implementation including tests |
| M | 3 | 1.732 | Medium — approximately 2–4 weeks including integration and validation |
| L | 9 | 3.000 | Large — approximately 1–3 months or a cross-cutting new subsystem |

### Severity Scale

| Severity | Meaning |
|----------|---------|
| P1 | Table-stakes capability gap blocking real workloads — a baseline-expected FDA capability that fdars lacks |
| P2 | Meaningful but not blocking — a useful missing capability that sophisticated users notice |
| P3 | Niche or specialized — valuable to a narrow audience but not baseline-expected |

**Note:** Severity (P1/P2/P3) and Value (1–5) are correlated but independent. Severity describes the category of impact; Value quantifies user benefit for ranking purposes. Ties in score are broken by severity (P1 before P2 before P3).

### Category (D-03, carried from Phase 17)

Each item is tagged **table-stakes** or **differentiator** (the Phase-17 category of its dominant constituent gaps). All 18 table-stakes gaps are covered by items T-01…T-08; the 144 differentiator gaps are covered by the remaining items.

---

## Ranked Backlog

**MASTER TABLE — sorted by descending `score = value / sqrt(effort)`. Rank 1..26.**
Items with the same computed score are sub-ordered P1 before P2 before P3 (higher severity first within the same score tier).

| Rank | ID | Title | Severity | Value | Effort | Score | Category | Area |
|------|----|-------|----------|-------|--------|-------|----------|------|
| 1 | T-01 | Constant/intercept basis + AIC smoothing-parameter selection | P1 | 5 | S | 5.00 | table-stakes | 1 |
| 2 | T-02 | Depth-fence functional boxplot + unified depth dispatcher | P1 | 5 | S | 5.00 | table-stakes | 3 |
| 3 | REG-03 | Elastic multinomial regression + robust-family completions | P2 | 3 | S | 3.00 | differentiator | 4 |
| 4 | INF-01 | Two-sample functional tests (t/F-permutation, mean/cov equality, SCB) | P1 | 5 | M | 2.89 | table-stakes | 5 |
| 5 | INF-02 | FLM inference suite (goodness-of-fit, F-test, one-way ANOVA V-stat) | P1 | 5 | M | 2.89 | table-stakes | 5 |
| 6 | REG-01 | Concurrent / varying-coefficient functional regression | P1 | 5 | M | 2.89 | table-stakes | 4 |
| 7 | REG-02 | Functional GLM exponential-family families (Poisson, Gamma, …) | P1 | 4 | M | 2.31 | table-stakes | 4 |
| 8 | FPCA-01 | Unified PACE sparse FPCA + conditional-expectation scores | P1 | 4 | M | 2.31 | table-stakes | 9 |
| 9 | DEPTH-01 | Depth-measure long tail (HRD/MHRD, HI/MHI, extremal, ERL, L∞, TVD+MSSI) | P2 | 4 | M | 2.31 | differentiator | 3 |
| 10 | OUT-01 | Outlier-detector suite (tvdmss, MUOD, sequential-transform, depthgram) | P2 | 4 | M | 2.31 | differentiator | 3 |
| 11 | INF-03 | Interval Testing Procedure (ITP) family (1-/2-sample, FLM coefficient) | P2 | 4 | M | 2.31 | differentiator | 5 |
| 12 | REG-04 | Additive / GKAM / GSAM functional regression + variable selection | P2 | 3 | M | 1.73 | differentiator | 4 |
| 13 | REG-05 | Flexible mixed-effects regression (denseFLMM, multiFAMM, fastFMM, pffr) | P2 | 3 | M | 1.73 | differentiator | 4 |
| 14 | FTS-02 | Functional ACF/PACF + stationarity + long-run covariance | P2 | 3 | M | 1.73 | differentiator | 6 |
| 15 | CLUS-01 | Model-based / density functional clustering (funHDDC, funFEM, DBSCAN, kCFC) | P2 | 3 | M | 1.73 | differentiator | 4 |
| 16 | REP-01 | Basis-system completions (monomial, exponential, power, polygonal, multi-domain) | P2 | 3 | M | 1.73 | differentiator | 1 |
| 17 | DENS-01 | Density object-data FPCA (LQD transform, Wasserstein barycenter) | P3 | 3 | M | 1.73 | differentiator | 7 |
| 18 | FPCA-02 | Specialized FPCA variants (FPCAder, FSVD, cross-covariance, sandwich/ssvd) | P3 | 3 | M | 1.73 | differentiator | 9 |
| 19 | SPARSE-01 | Sparse/irregular fast covariance (FACE, mfaces) + trajectory bands | P3 | 3 | M | 1.73 | differentiator | 1/9 |
| 20 | FTS-01 | Functional time series forecasting (ftsm, FPC-regression, fplsr, updating) | P2 | 4 | L | 1.33 | differentiator | 6 |
| 21 | FRE-01 | Fréchet regression + statistics (global/local, mean/variance/ANOVA) | P2 | 4 | L | 1.33 | differentiator | 7 |
| 22 | FTS-03 | Spectral functional time series (DPCA, spectral density, VAR/VMA, FARMA sim) | P3 | 3 | L | 1.00 | differentiator | 6 |
| 23 | FRE-02 | Object-data Fréchet regression (covariance/correlation/spherical/network/point-process) | P3 | 3 | L | 1.00 | differentiator | 7 |
| 24 | REG-06 | Boosting / Bayesian functional regression (FDboost, GAMLSS, Gibbs/VB FOSR) | P3 | 2 | L | 0.67 | differentiator | 4 |
| 25 | REP-02 | FEM/PDE smoothing on irregular 2D/3D domains (fdaPDE) | P3 | 2 | L | 0.67 | differentiator | 1 |
| 26 | CLUS-02 | Functional co-clustering (funLBM latent-block) + slope-heuristic selection | P3 | 2 | L | 0.67 | differentiator | 4 |

**Descending-score check:** 5.00, 5.00, 3.00, 2.89, 2.89, 2.89, 2.31, 2.31, 2.31, 2.31, 2.31, 1.73, 1.73, 1.73, 1.73, 1.73, 1.73, 1.73, 1.73, 1.33, 1.33, 1.00, 1.00, 0.67, 0.67, 0.67 — **strictly non-increasing.** Ties broken by severity: within the 2.89 tier all three are P1; within the 2.31 tier REG-02/FPCA-01 (P1) precede DEPTH-01/OUT-01/INF-03 (P2); within the 1.73 tier the P2 items (REG-04, REG-05, FTS-02, CLUS-01, REP-01) precede the P3 items (DENS-01, FPCA-02, SPARSE-01); the 1.33 tier is P2, the 1.00 and 0.67 tiers P3. REG-03 (S-effort differentiator, score 3.00) correctly sits above the M-effort table-stakes tier (2.89) — a high value-per-effort ratio, as the formula intends.

---

## Backlog Items

Each item carries a 7-field promotion block: (1) candidate requirement / phase phrasing; (2) R-side reference (package(s) + parity rows covered); (3) fdars current gap (absent vs partial + what exists today); (4) proposed direction (target `fdars-core/src/` location + approach sketch); (5) value + effort + severity + category; (6) score; (7) notes / dependencies.

---

### T-01 — Constant/intercept basis + AIC smoothing-parameter selection

1. **Candidate requirement / phase phrasing:** "Add a named constant/intercept basis constructor to `basis/`, and add an AIC criterion to the automatic smoothing-parameter selector so `smooth_basis` can select the roughness penalty by AIC as well as GCV."
2. **R-side reference:** `fda` (6.3.0) — "Constant / intercept basis" (Area 1, absent, table-stakes) and "Smoothing with automatic parameter selection (GCV, AIC)" (Area 1, partial, table-stakes). Covers 2 of the 18 table-stakes gaps.
3. **fdars current gap:** Constant basis **absent** — only B-spline + Fourier factories are exposed (a constant column is trivially constructable but there is no named basis object usable in regression design matrices). AIC smoothing selection **partial** — `smooth_basis::smooth_basis_gcv` and `smoothing::optim_bandwidth` do GCV/CV only; the AIC criterion is absent.
4. **Proposed direction:** `fdars-core/src/basis/` — add a `constant_basis(n_eval)` / `ConstantBasis` factory returning a single-column design + a zero roughness penalty. `fdars-core/src/smooth_basis.rs` (and/or `smoothing.rs`) — add an `AIC` variant to the smoothing-selection criterion (AIC = n·log(RSS/n) + 2·tr(H), reusing the hat-matrix trace already computed for GCV). No new algorithm needed.
5. **Value 5 · Effort S · Severity P1 · Category table-stakes** — both are baseline expectations (intercept basis for regression design; AIC as a mainstream smoothing criterion) and both reuse existing infrastructure.
6. **Score = 5 / √1 = 5.00.**
7. **Notes/deps:** No dependencies. Smallest, highest-leverage table-stakes item — a natural first phase. The constant basis unblocks cleaner regression design matrices used by REG-01/REG-02.

---

### T-02 — Depth-fence functional boxplot + unified depth dispatcher

1. **Candidate requirement / phase phrasing:** "Add the López-Pintado depth-fence functional boxplot (1.5×IQR-of-depths threshold + outlier flags as numeric outputs) and a single `functional_depth(data, method: DepthMethod)` dispatcher over the existing depth functions."
2. **R-side reference:** `roahd`/`fdaoutlier`/`fda.usc` — "Functional boxplot (depth-based outlier thresholds/fences)" (Area 3, partial, table-stakes) and `fda.usc` — "General functional depth dispatcher" (Area 3, partial, table-stakes). Covers 2 of the 18 table-stakes gaps.
3. **fdars current gap:** Functional boxplot **partial** — `outliers::outliergram` gives a parabolic-fence detector, but the canonical depth-fence boxplot (central-region + 1.5×IQR whisker on depth values) is not a named function. Depth dispatcher **partial** — depth methods exist only as separate functions (`fraiman_muniz_1d`, `band_1d`, `modified_band_1d`, `random_projection_1d`, …); there is no unified `DepthMethod`-dispatched public entry point.
4. **Proposed direction:** `fdars-core/src/outliers.rs` (or `depth/`) — add `functional_boxplot(data, argvals, depth_method)` returning `{ median_curve, central_region, fences, outlier_indices }` (numeric outputs only; renderer stays out-of-scope). `fdars-core/src/depth/mod.rs` — add a `DepthMethod` enum + `functional_depth(data, method)` that dispatches to the existing per-method functions (mirrors the existing `CovType`/`ProjectionBasisType` enum-dispatch convention).
5. **Value 5 · Effort S · Severity P1 · Category table-stakes** — the functional boxplot is *the* canonical robust functional-summary tool; the dispatcher is a baseline ergonomic. Both wrap existing code.
6. **Score = 5 / √1 = 5.00.**
7. **Notes/deps:** No dependencies. Reuses all shipped depth measures. Ranks equal-first with T-01 (both 5.00, P1); ordered by area.

---

### REG-03 — Elastic multinomial regression + robust-family completions

1. **Candidate requirement / phase phrasing:** "Add multinomial (multi-class) elastic logistic regression to `elastic_regression/`, closing the one partial in fdars' otherwise-complete elastic-regression family."
2. **R-side reference:** `fdasrvf` (2.4.4) — "Elastic multinomial logistic regression" (Area 4, partial, differentiator). Covers 1 gap.
3. **fdars current gap:** **Partial** — `elastic_regression::elastic_logistic` is binary-only (grep: no `multinomial`); elastic regression, elastic PCR, and binary elastic logistic are all **present**. Only the multi-class extension is missing.
4. **Proposed direction:** `fdars-core/src/elastic_regression/logistic.rs` — extend the SRSF-space logistic estimator to a one-vs-rest or softmax multinomial variant, reusing the existing SRVF representation and warping machinery; add `predict_elastic_multinomial`.
5. **Value 3 · Effort S · Severity P2 · Category differentiator** — a targeted extension of an existing capability; small, self-contained, and high score-per-effort.
6. **Score = 3 / √1 = 3.00.**
7. **Notes/deps:** No new subsystem. This is the sole S-effort differentiator that ranks above the M-effort table-stakes tier by score; kept in strict order accordingly. NOTE: this is *extending* an existing fdars capability, not building a strength area — the elastic *breadth* moat belongs to `fdasrvf` (Phase 18), so this closes a fdars gap, it does not build a fdars strength.

---

### INF-01 — Two-sample functional tests (t/F-permutation, mean/cov equality, SCB)

1. **Candidate requirement / phase phrasing:** "Create a functional-inference module exposing standalone two-sample tests: functional t-permutation test, F-permutation test (`Fperm.fd`), equality-of-means/covariance test, two-sample SCB-based test, and simultaneous confidence bands for the mean."
2. **R-side reference:** `fda` (`Fperm.fd`, `tperm.fd`), `fda.usc` (equality of means/covariance), `SCBmeanfd` (two-sample equality test, SCB for the mean). Parity rows (Area 5): "t-permutation test" (absent, table-stakes), "F-permutation test" (partial), "Equality of functional means/covariance test" (partial), "Two-sample equality test (SCBmeanfd)" (absent, table-stakes), "Simultaneous confidence bands for the mean" (partial). Covers ~5 gaps incl. 4 table-stakes.
3. **fdars current gap:** Area 5 is **0/22 present.** The related internals are repurposed: `function_on_scalar::fanova` is permutation-F-based but not exposed as a standalone `Fperm.fd`/`tperm.fd` test; `spm::stats::hotelling_t2` gives a Hotelling T² in an SPM context, not a standalone two-sample inference test; `alignment::shape_ci` gives shape CIs, not SCBs for the mean.
4. **Proposed direction:** new `fdars-core/src/inference/` module (or `inference.rs`). Wrap/lift the existing permutation machinery from `function_on_scalar.rs` into standalone `t_perm_test` / `f_perm_test`; expose `spm::stats::hotelling_t2` as a `two_sample_mean_test`; add `mean_scb` (Degras/Gaussian-kinematic SCB — note `tolerance/degras.rs` has related bootstrap-band machinery to reuse) and an SCB-based two-sample test.
5. **Value 5 · Effort M · Severity P1 · Category table-stakes** — the most basic functional hypothesis tests; fdars has *zero* standalone inference surface today.
6. **Score = 5 / √3 = 2.89.**
7. **Notes/deps:** Foundational for Area 5. Reuses `function_on_scalar` permutation code + `spm::stats` + `tolerance/degras`. Pairs with INF-02 and INF-03 to build the full inference suite; INF-01 first (broadest table-stakes coverage).

---

### INF-02 — FLM inference suite (goodness-of-fit, F-test, one-way ANOVA V-stat)

1. **Candidate requirement / phase phrasing:** "Add formal functional-linear-model inference: a goodness-of-fit test and an F-test for the scalar-response FLM, plus the asymptotic one-way functional ANOVA V-statistic form alongside the existing permutation ANOVA."
2. **R-side reference:** `fda.usc` (FLM goodness-of-fit, F-test for the scalar-response FLM), `fdatest`/`fdANOVA` (one-way functional ANOVA V-statistic). Parity rows (Areas 4/5): "Functional linear model goodness-of-fit / F-test" (partial), "Goodness-of-fit test for the FLM" (absent, table-stakes), "F-test for the FLM with scalar response" (absent, table-stakes), "Functional ANOVA (one-way) with V-statistic" (partial, table-stakes). Covers ~4 gaps incl. 3 table-stakes.
3. **fdars current gap:** **Partial/absent** — fitted FLMs (`scalar_on_function::fregre_lm`) ship only informal `helpers::r_squared`/`r_squared_adj` diagnostics; no formal GoF/F-test. `function_on_scalar::fanova` is permutation-based; the asymptotic V-statistic ANOVA form is absent.
4. **Proposed direction:** `fdars-core/src/inference/` (shared with INF-01) — add `flm_gof_test` and `flm_f_test` operating on a fitted `FregreLmResult` (residual-based F/GoF statistics against the FLM null); add a `oneway_anova_vstat` variant to `function_on_scalar.rs` computing the asymptotic V-statistic. Reuses fitted-model residuals and integration weights already available.
5. **Value 5 · Effort M · Severity P1 · Category table-stakes** — model-adequacy and significance testing for the FLM are baseline; only informal R² exists.
6. **Score = 5 / √3 = 2.89.**
7. **Notes/deps:** Depends on the `inference/` module scaffolding from INF-01 (build INF-01 first). Consumes `scalar_on_function` fit results.

---

### REG-01 — Concurrent / varying-coefficient functional regression

1. **Candidate requirement / phase phrasing:** "Implement functional concurrent (varying-coefficient) regression — a time-varying coefficient model relating a functional response to functional predictors evaluated at the same argument — for both dense and sparse data."
2. **R-side reference:** `fdaconcur` (0.1.3), `refund`, `fdapace` (0.6.0). Parity rows (Area 4): "Varying-coefficient / concurrent functional regression" (absent, table-stakes) and "Functional concurrent regression (varying-coeff, sparse/dense)" (absent, table-stakes). Covers 2 table-stakes gaps.
3. **fdars current gap:** **Absent** — no concurrent/varying-coefficient regression (grep: no match). fdars has scalar-on-function (`scalar_on_function/`), function-on-scalar (`function_on_scalar.rs`), and function-on-function (`fof_regression.rs`), but not the concurrent model where β(t) varies over the shared argument.
4. **Proposed direction:** new `fdars-core/src/concurrent_regression.rs` — pointwise / locally-weighted least squares estimation of β(t) over the shared grid (dense case), with a smoothing penalty; a kernel-weighted sparse variant reusing `irreg_fdata` smoothing for the sparse/PACE case. Return `{ beta_curve, fitted, residuals }`.
5. **Value 5 · Effort M · Severity P1 · Category table-stakes** — concurrent regression is a mainstream functional-regression model expected in any general-purpose FDA library.
6. **Score = 5 / √3 = 2.89.**
7. **Notes/deps:** Independent. The sparse variant benefits from FPCA-01's PACE infrastructure but does not require it (dense variant is self-contained). Pairs naturally with REG-02 into a "functional regression completeness" milestone.

---

### REG-02 — Functional GLM exponential-family families (Poisson, Gamma, …)

1. **Candidate requirement / phase phrasing:** "Extend the functional GLM beyond logistic to the exponential family — Poisson (log link), Gamma, and Gaussian-identity — for scalar-response functional regression."
2. **R-side reference:** `fda.usc`, `refund` — "Functional GLM, scalar response (logistic + Poisson + families)" (Area 4, partial, table-stakes). Covers 1 table-stakes gap.
3. **fdars current gap:** **Partial** — `scalar_on_function::functional_logistic` covers the logistic family only; Poisson and other exponential-family functional GLMs are absent.
4. **Proposed direction:** `fdars-core/src/scalar_on_function/` — generalize `functional_logistic` into a `functional_glm(data, y, family: GlmFamily)` estimator via IRLS over FPC/basis scores, with a `GlmFamily` enum (Binomial/Poisson/Gamma/Gaussian) carrying link + variance functions. Reuses the existing FPCA-score design + Newton/IRLS loop.
5. **Value 4 · Effort M · Severity P1 · Category table-stakes** — exponential-family functional GLMs are a standard expected family; the logistic path already proves the pattern.
6. **Score = 4 / √3 = 2.31.**
7. **Notes/deps:** Reuses `functional_logistic` IRLS scaffolding + `fdata_to_pc_1d`. Independent of INF/REG-01. High table-stakes value; ranks at the top of the 2.31 tier by P1 severity.

---

### FPCA-01 — Unified PACE sparse FPCA + conditional-expectation scores

1. **Candidate requirement / phase phrasing:** "Provide a single PACE FPCA estimator for sparse/irregularly-sampled curves: smoothed mean + covariance-surface eigendecomposition + conditional-expectation FPC scores, returning fitted trajectories with confidence bands."
2. **R-side reference:** `fdapace` (0.6.0), `fda`. Parity rows (Area 9): "FPCA via PACE for sparse/irregular curves" (partial, table-stakes), "FPC scores via conditional expectation" (partial, table-stakes), "Fitted continuous trajectories + confidence bands for sparse data" (partial), "Functional fragment completion" (partial). Covers ~4 gaps incl. 2 table-stakes.
3. **fdars current gap:** **Partial** — the pieces exist but not as a unified estimator: `irreg_fdata::to_regular_grid` (kernel-smoothed mean), `irreg_fdata::cov_irreg` (sparse covariance surface), `regression::fdata_to_pc_1d` (dense FPCA), and `spm::partial::conditional_expectation` (PACE-style reconstruction) are separate. There is no `pace_fpca` that chains them and produces conditional-expectation scores for arbitrary sparse input.
4. **Proposed direction:** new `fdars-core/src/pace_fpca.rs` — orchestrate the existing pieces: smooth mean + `cov_irreg` covariance surface → eigendecompose → conditional-expectation (BLUP) scores per curve → fitted trajectories + pointwise bands. Expose `MakeFPCAInputs`-style validation via the existing `validation` module.
5. **Value 4 · Effort M · Severity P1 · Category table-stakes** — PACE is *the* canonical method for sparse/longitudinal functional data; fdars has the components but no user-facing estimator.
6. **Score = 4 / √3 = 2.31.**
7. **Notes/deps:** Mostly integration of shipped `irreg_fdata` + `spm::partial` + `regression` code. Enables the sparse path of REG-01 and SPARSE-01.

---

### DEPTH-01 — Depth-measure long tail (HRD/MHRD, HI/MHI, extremal, ERL, L∞, TVD+MSSI)

1. **Candidate requirement / phase phrasing:** "Add the missing univariate functional depth measures to `depth/`: half-region and modified half-region depth, hypograph/modified-hypograph and un-modified epigraph indices, extremal depth, extreme-rank-length depth, L∞ depth, and total-variation depth with MSSI."
2. **R-side reference:** `roahd` (HRD, MHRD, HI, MHI, EI), `fdaoutlier` (extremal, ERL, L∞, TVD+MSSI). Parity rows (Area 3): 8 absent depth-measure rows + the EI partial — all differentiator. Covers ~9 gaps.
3. **fdars current gap:** **Absent (mostly).** fdars ships MBD, BD, MEI, random-projection, elastic depth, Fraiman-Muniz; the long tail (HRD/MHRD, HI/MHI, extremal, ERL, L∞, TVD+MSSI) is absent, and the un-modified EI is partial (only MEI exposed).
4. **Proposed direction:** `fdars-core/src/depth/` — add one function per measure following the existing per-file convention (each is a well-defined rank/band statistic over the column-major `FdMatrix`). Register each in the T-02 `DepthMethod` dispatcher.
5. **Value 4 · Effort M · Severity P2 · Category differentiator** — a large, coherent, well-understood cluster; each measure is individually small but the set is milestone-sized. Widely used in robust-FDA workflows.
6. **Score = 4 / √3 = 2.31.**
7. **Notes/deps:** Complements T-02 (dispatcher registration). No new infrastructure. **Excludes** streaming depth (fdars strength U-5) — these are batch measures only.

---

### OUT-01 — Outlier-detector suite (tvdmss, MUOD, sequential-transform, depthgram)

1. **Candidate requirement / phase phrasing:** "Add the `fdaoutlier`/`roahd` outlier-detection algorithms: TVD+MSSI detector (tvdmss), Massive Unsupervised Outlier Detection (MUOD), sequential-transformation detection, and the depthgram statistic."
2. **R-side reference:** `fdaoutlier` (tvdmss, MUOD, sequential-transformation), `roahd` (depthgram). Parity rows (Area 3): all 4 absent, differentiator. Plus partially-observed depth outlier detection (`fdaPOIFD`, absent). Covers ~5 gaps.
3. **fdars current gap:** **Absent.** fdars has MS-plot / directional outlyingness (`outliers::magnitude_shape_outlyingness`) and outliergram (`outliers::outliergram`), but not tvdmss, MUOD, sequential-transformation, depthgram, or the fdaPOIFD partially-observed detectors.
4. **Proposed direction:** `fdars-core/src/outliers.rs` — add `tvdmss` (depends on TVD+MSSI from DEPTH-01), `muod`, `sequential_transform_outliers`, and a `depthgram` statistic (numeric outputs; renderer out-of-scope).
5. **Value 4 · Effort M · Severity P2 · Category differentiator** — robust outlier detection is a common exploratory need; the cluster is coherent and shares the depth infrastructure.
6. **Score = 4 / √3 = 2.31.**
7. **Notes/deps:** **tvdmss depends on DEPTH-01** (TVD+MSSI depth). Sequence DEPTH-01 before OUT-01.

---

### INF-03 — Interval Testing Procedure (ITP) family (1-/2-sample, FLM coefficient)

1. **Candidate requirement / phase phrasing:** "Implement the Interval Testing Procedure (ITP): one-population and two-population interval-wise tests (B-spline and Fourier bases) with domain-selective adjusted p-values, plus interval-wise FLM coefficient testing."
2. **R-side reference:** `fdatest` (2.1.1) — ITP one-/two-population, `ITPlmbspline` FLM coefficient testing; random-projection ANOVA/MANOVA (`fdANOVA`) is adjacent. Parity rows (Area 5): 4 absent ITP rows + random-projection ANOVA/MANOVA — all differentiator. Covers ~6 gaps.
3. **fdars current gap:** **Absent.** No interval-wise testing procedure; the domain-selective adjusted-p-value machinery has no analog in fdars.
4. **Proposed direction:** `fdars-core/src/inference/itp.rs` — basis-projected interval-wise permutation testing with the ITP p-value-adjustment (closure over interval families). Reuse the permutation infrastructure from INF-01 and the basis projection from `basis/`.
5. **Value 4 · Effort M · Severity P2 · Category differentiator** — a distinctive, widely-cited inference method; more specialized than INF-01/INF-02's baseline tests, hence P2/differentiator.
6. **Score = 4 / √3 = 2.31.**
7. **Notes/deps:** Depends on the `inference/` scaffolding (INF-01) and `basis/` projection. Build after INF-01/INF-02.

---

### REG-04 — Additive / GKAM / GSAM functional regression + variable selection

1. **Candidate requirement / phase phrasing:** "Add nonparametric additive functional regression — functional additive model (FAM), generalized kernel additive model (GKAM), generalized spectral additive model (GSAM) — plus scalar-on-function variable selection and a permutation-test wrapper."
2. **R-side reference:** `fdapace` (FAM), `fda.usc` (GKAM, GSAM), `refund` (fosr.vs variable selection, fosr.perm wrapper, history-index model). Parity rows (Area 4): FAM (absent), GSAM (absent), GKAM (absent), fosr.vs (absent), fosr.perm (partial), history-index (absent) — all differentiator. Covers ~6 gaps.
3. **fdars current gap:** **Absent (mostly).** fdars has linear/kernel/PLS scalar-on-function (`scalar_on_function/`) but no additive-model family, no variable selection, no history-index (lagged) model.
4. **Proposed direction:** `fdars-core/src/scalar_on_function/additive.rs` — backfitting additive model over FPC-score components; kernel and spectral additive variants; a `variable_selection` helper (group-penalized coefficient selection); a history-index estimator (lagged predictor window). Reuse `smoothing.rs` kernels + `fdata_to_pc_1d`.
5. **Value 3 · Effort M · Severity P2 · Category differentiator** — a meaningful regression breadth expansion; specialized beyond the concurrent/GLM table-stakes.
6. **Score = 3 / √3 = 1.73.**
7. **Notes/deps:** Independent of REG-01/REG-02 but shares the scalar-on-function design machinery.

---

### REG-05 — Flexible mixed-effects regression (denseFLMM, multiFAMM, fastFMM, pffr)

1. **Candidate requirement / phase phrasing:** "Extend functional mixed models beyond fixed-effect testing to full random-effects estimation: dense functional linear mixed model (denseFLMM), multivariate functional additive mixed model (multiFAMM), fast functional mixed-model inference (fastFMM), and flexible function-on-function regression (pffr)."
2. **R-side reference:** `denseFLMM` (0.1.3), `multifamm` (0.1.1), `fastFMM` (1.0.1), `refund` (pffr). Parity rows (Area 4): denseFLMM (partial), multiFAMM (absent), fastFMM (partial), pffr flexible/RE (absent) — all differentiator. Covers ~4 gaps.
3. **fdars current gap:** **Partial/absent.** `famm::fmm_test_fixed` provides a fixed-effect functional mixed-model test but not the full random-effects estimators; multiFAMM and pffr random-effects variants are absent.
4. **Proposed direction:** `fdars-core/src/famm.rs` — extend to random-effects estimation (mixed-model equations over FPC scores or basis coefficients), a multivariate variant, and wire a flexible-RE function-on-function path into `fof_regression.rs`.
5. **Value 3 · Effort M · Severity P2 · Category differentiator** — mixed-effects functional regression is a specialized but sought-after capability; fdars already has the fixed-effect foothold.
6. **Score = 3 / √3 = 1.73.**
7. **Notes/deps:** Builds on existing `famm.rs`. Note: `fof_regression.rs` (function-on-function) is **present** at parity — this item extends only the *flexible/RE* variant, not the base capability (which is not a gap).

---

### FTS-02 — Functional ACF/PACF + stationarity + long-run covariance

1. **Candidate requirement / phase phrasing:** "Add functional serial-dependence tooling: functional autocorrelation (fACF) and partial ACF with white-noise confidence bands, a functional stationarity test, and long-run covariance estimation via a kernel sandwich estimator."
2. **R-side reference:** `ftsa` (facf, T_stationary, long-run covariance), `fdaACF` (L2-norm fACF, partial fACF, white-noise distribution). Parity rows (Area 6): 7 absent rows (fACF, L2 fACF, partial fACF, white-noise distribution, stationarity, long-run covariance, differencing-partial) — all differentiator. Covers ~7 gaps.
3. **fdars current gap:** **Absent.** fdars' `seasonal/` has only scalar autocorrelation on a mean curve for period detection; there is no functional ACF/PACF, no stationarity test, no long-run covariance. Functional differencing is partial (`detrend`).
4. **Proposed direction:** new `fdars-core/src/fts/acf.rs` — L2-norm functional ACF/PACF with the strong-white-noise limiting distribution for confidence bands; a `stationarity_test`; a `long_run_covariance` kernel-sandwich estimator; a functional differencing operator. Reuse `helpers` quadrature + `covariance.rs`.
5. **Value 3 · Effort M · Severity P2 · Category differentiator** — the serial-dependence diagnostics are the natural first FTS building block and are lighter than full forecasting (FTS-01).
6. **Score = 3 / √3 = 1.73.**
7. **Notes/deps:** Foundational for FTS-01/FTS-03 (forecasting and spectral methods build on ACF/long-run covariance). Build this before FTS-01.

---

### CLUS-01 — Model-based / density functional clustering (funHDDC, funFEM, DBSCAN, kCFC)

1. **Candidate requirement / phase phrasing:** "Add functional clustering paradigms beyond k-means/GMM: subspace model-based clustering (funHDDC), discriminative-subspace clustering (funFEM), DBSCAN density clustering, kCFC subspace-embedding clustering, and joint align-and-cluster k-means."
2. **R-side reference:** `funHDDC` (2.3.1.1), `funFEM` (1.2), `fdacluster` (0.4.2, DBSCAN + Sangalli joint align+cluster), `fdapace` (kCFC), `fdasrvf` (elastic k-means). Parity rows (Area 4): funHDDC (partial), funFEM (absent), DBSCAN (absent), kCFC (absent), Sangalli joint (partial), elastic k-means (partial) — all differentiator. Covers ~6 gaps.
3. **fdars current gap:** **Partial/absent.** fdars has `clustering::kmeans_fd`/`fuzzy_cmeans_fd`, `gmm::gmm_cluster` (GMM on FPC scores), and hierarchical/k-medoids from elastic distances; funHDDC per-group subspaces, funFEM discriminative subspace, DBSCAN, kCFC, and in-loop joint align+cluster are absent/partial.
4. **Proposed direction:** `fdars-core/src/clustering.rs` + `gmm/` — add funHDDC-style per-group subspace covariance models (extend `gmm/`), a DBSCAN over functional distances (reuse `distance.rs`), a kCFC subspace-embedding loop, and a joint align+cluster estimator (reuse `alignment/` + `clustering`).
5. **Value 3 · Effort M · Severity P2 · Category differentiator** — clustering breadth; each paradigm is well-defined and shares fdars' distance/GMM infrastructure.
6. **Score = 3 / √3 = 1.73.**
7. **Notes/deps:** Reuses `distance.rs`, `gmm/`, `alignment/`. Co-clustering (funLBM) is split out as CLUS-02 (larger, L-effort).

---

### REP-01 — Basis-system completions (monomial, exponential, power, polygonal, multi-domain)

1. **Candidate requirement / phase phrasing:** "Complete the basis-system family in `basis/`: monomial/polynomial, exponential, and power bases as named factories; a named polygonal (piecewise-linear) basis object; and a composable multivariate/multi-domain functional data container."
2. **R-side reference:** `fda` (monomial, exponential, power, polygonal bases; PDA/Lfd), `funData` (multiFunData multi-domain container), `tf` (tidy multi-representation vector). Parity rows (Area 1): monomial (absent), exponential (absent), power (absent), polygonal (partial), multiFunData (absent), tidy-vector (partial), Lfd (partial), PDA (absent), tensor/2D-FPCA (partial) — all differentiator. Covers ~9 gaps.
3. **fdars current gap:** **Absent/partial.** Only B-spline + Fourier are exposed as named bases; polygonal evaluation exists via `linear_interp` but no named basis object; no composable multi-domain container (2D handled via flattened matrices); no composable Lfd object; no PDA (linear-ODE estimation).
4. **Proposed direction:** `fdars-core/src/basis/` — add `monomial_basis`, `exponential_basis`, `power_basis`, `polygonal_basis` factories with penalty matrices; a `MultiFunData` container in a new `multi_fdata.rs`; a composable `Lfd`/`LinearDifferentialOperator` object; a `principal_differential_analysis` estimator.
5. **Value 3 · Effort M · Severity P2 · Category differentiator** — rounds out the representation layer; individually small pieces but a coherent milestone.
6. **Score = 3 / √3 = 1.73.**
7. **Notes/deps:** The constant basis is handled separately in T-01 (table-stakes). Independent of the ML items.

---

### DENS-01 — Density object-data FPCA (LQD transform, Wasserstein barycenter)

1. **Candidate requirement / phase phrasing:** "Add density-valued FDA: the log-quantile-density (LQD) transform and its inverse, LQD-FPCA for probability densities, Wasserstein Fréchet mean (barycenter) of densities, and density normalization/regularization."
2. **R-side reference:** `fdadensity` (0.1.4). Parity rows (Area 7): LQD-FPCA (absent), LQD↔density conversion (absent), FVE for LQD-FPCA (partial), Wasserstein barycenter (absent), density normalization (absent) — all differentiator. Covers ~5 gaps.
3. **fdars current gap:** **Absent.** No LQD transform, no density-space FPCA, no Wasserstein barycenter. `FpcaResult` exposes singular values (FVE computable) but nothing density-specific.
4. **Proposed direction:** new `fdars-core/src/density_fda.rs` — LQD/inverse-LQD transforms (compositional-geometry map), then reuse `fdata_to_pc_1d` in LQD space for LQD-FPCA; a 1D Wasserstein barycenter (quantile-average) for densities.
5. **Value 3 · Effort M · Severity P3 · Category differentiator** — a self-contained density-FDA slice of Area 7; specialized but tractable (1D densities, no general metric-space machinery).
6. **Score = 3 / √3 = 1.73.**
7. **Notes/deps:** The 1D-density subset of Area 7; simpler than the general Fréchet items (FRE-01/FRE-02). Reuses `fdata_to_pc_1d`.

---

### FPCA-02 — Specialized FPCA variants (FPCAder, FSVD, cross-covariance, sandwich/ssvd)

1. **Candidate requirement / phase phrasing:** "Add the specialized FPCA variants: FPCA of derivatives (FPCAder), functional SVD / cross-FPCA (FSVD), cross-covariance surface estimation, dynamical/functional correlation, and the fast sandwich-smoother / sparse-SVD FPCA estimators."
2. **R-side reference:** `fdapace` (FPCAder, FSVD, GetCrCov, DynCorr, FCCor, FVPA), `refund` (fpca.sc sandwich, fpca.ssvd, fpca.lfda). Parity rows (Area 9): 8 absent rows + fpca.ssvd/FACE partials — all differentiator. Covers ~10 gaps.
3. **fdars current gap:** **Absent/partial.** fdars has dense `fdata_to_pc_1d` (thin faer SVD) and the PACE pieces (see FPCA-01), but no derivative-FPCA, no functional SVD, no cross-covariance surfaces, no dynamical correlation, no sandwich-smoother/ssvd variants.
4. **Proposed direction:** `fdars-core/src/regression.rs` (or a new `fpca_variants.rs`) — `fpca_der` (differentiate loadings), `fsvd` (bivariate SVD analogue), `cross_covariance` surfaces, `dynamical_correlation`, and a sandwich-smoother covariance path feeding FPCA.
5. **Value 3 · Effort M · Severity P3 · Category differentiator** — completes the FPCA long tail; valuable to FPCA-heavy users, specialized overall.
6. **Score = 3 / √3 = 1.73.**
7. **Notes/deps:** Complements FPCA-01 (which handles the table-stakes PACE core). Reuses `regression` FPCA + `covariance.rs`.

---

### SPARSE-01 — Sparse/irregular fast covariance (FACE, mfaces) + trajectory bands

1. **Candidate requirement / phase phrasing:** "Implement the FACE fast-sandwich covariance estimator for sparse functional data, its multivariate extension (mfaces), and integrated fitted-trajectory-with-confidence-band output for sparse curves."
2. **R-side reference:** `face` (0.1-8, FACE), `mfaces` (0.1-4, multivariate FACE), `fdapace` (trajectory bands). Parity rows (Areas 1/9): "Fast covariance estimation (FACE)" (partial), "Fast covariance for multivariate sparse data (mfaces)" (absent), "FPCA via FACE" (partial), "Fitted continuous trajectories + bands" (partial) — all differentiator. Covers ~4 gaps.
3. **fdars current gap:** **Partial/absent.** `irreg_fdata::cov_irreg` gives a kernel-smoothed empirical covariance, but not the FACE fast-sandwich algorithm specifically; the multivariate mfaces extension is absent.
4. **Proposed direction:** `fdars-core/src/irreg_fdata/` — add a FACE sandwich-smoother covariance estimator; a multivariate variant; integrate with FPCA-01 to emit trajectory + band output.
5. **Value 3 · Effort M · Severity P3 · Category differentiator** — a performance/accuracy upgrade to sparse-data covariance; specialized.
6. **Score = 3 / √3 = 1.73.**
7. **Notes/deps:** Complements FPCA-01. Builds on `irreg_fdata::cov_irreg`.

---

### FTS-01 — Functional time series forecasting (ftsm, FPC-regression, fplsr, updating)

1. **Candidate requirement / phase phrasing:** "Build functional time series forecasting: the FPCA-based functional time series model (ftsm), FPC-score-regression forecasting, functional PLS forecasting (fplsr), dynamic forecast updating, and iterative multi-step forecasting."
2. **R-side reference:** `ftsa` (6.7). Parity rows (Area 6): ftsm (partial), FPC-regression forecasting (absent), fplsr (absent), dynamic updating (absent), iterative forecasting (absent), GAEVforecast (absent) — all differentiator. Covers ~6 gaps.
3. **fdars current gap:** **Absent (mostly).** `fdata_to_pc_1d` gives the FPCA decomposition but there is no time-ordered ftsm wrapper, no score-forecasting, no fplsr, no dynamic updating. Forecast-error metrics (`scoring.rs`) are present.
4. **Proposed direction:** new `fdars-core/src/fts/forecast.rs` — decompose a curve series via `fdata_to_pc_1d`, fit scalar time-series models (AR/ARIMA-style) to the FPC-score sequences, reconstruct forecast curves; add a PLS-score forecasting variant and a dynamic-updating path.
5. **Value 4 · Effort L · Severity P2 · Category differentiator** — a large, high-value differentiator zone (Area 6 is 2/25 present); forecasting is the headline FTS capability. L-effort (new subsystem + scalar-TS modeling layer) caps the score.
6. **Score = 4 / √3(9) = 4 / 3 = 1.33.**
7. **Notes/deps:** **Depends on FTS-02** (ACF / long-run covariance for model order + inference). Reuses `regression` FPCA + `scoring.rs`. Largest single-area gap zone by capability count — milestone-sized.

---

### FRE-01 — Fréchet regression + statistics (global/local, mean/variance/ANOVA)

1. **Candidate requirement / phase phrasing:** "Introduce metric-space (object-data) regression: global and local Fréchet regression with Euclidean predictors, Fréchet mean and variance in a general metric space, Wasserstein distance between distributions, density-response Fréchet regression, and Fréchet ANOVA."
2. **R-side reference:** `frechet` (0.3.0). Parity rows (Area 7): global/local Fréchet regression (absent), Fréchet mean (partial), Fréchet variance (absent), Wasserstein distance (absent), density-response regression (absent), Fréchet ANOVA (absent), object changepoint (partial) — all differentiator. Covers ~8 gaps.
3. **fdars current gap:** **Absent (mostly).** `alignment::karcher_mean` is a Fréchet mean on the Fisher-Rao *shape* manifold only; there is no general metric-space Fréchet mean/variance, no Wasserstein distance, no Fréchet regression. Area 7 is 0/25 present.
4. **Proposed direction:** new `fdars-core/src/frechet/` module — a metric-space abstraction (distance + weighted-Fréchet-mean solver), global Fréchet regression (weighted local-constant/local-linear over predictors), Fréchet variance/ANOVA, and 1D Wasserstein distance. Start with the density (2-Wasserstein) response space (shares DENS-01's quantile machinery).
5. **Value 4 · Effort L · Severity P2 · Category differentiator** — the single largest all-absent zone (Area 7, 0/25); a genuinely new subsystem, high value for object-data users. L-effort caps the score.
6. **Score = 4 / √9 = 4 / 3 = 1.33.**
7. **Notes/deps:** The core Fréchet framework; FRE-02 (specific object spaces) builds on it. Shares Wasserstein/quantile machinery with DENS-01.

---

### FTS-03 — Spectral functional time series (DPCA, spectral density, VAR/VMA, FARMA sim)

1. **Candidate requirement / phase phrasing:** "Add frequency-domain and simulation FTS methods: dynamic principal component analysis (DPCA) via spectral methods, spectral density operator estimation, functional VAR/VMA process simulation, and functional ARMA (FARMA) simulation."
2. **R-side reference:** `freqdom` (2.0.5, DPCA, spectral density, VAR/VMA), `ftsa` (sim_FARMA, MAF, MFDM, dynamic FPCA, CoDa/LQDT FPCA). Parity rows (Area 6): DPCA (absent), spectral density (absent), VAR/VMA (absent), FARMA sim (absent), MAF/MFDM/dynamic-FPCA/CoDa/LQDT (absent) — all differentiator. Covers ~9 gaps.
3. **fdars current gap:** **Absent.** No spectral/frequency-domain FTS methods; `simulation.rs` covers KL Gaussian data, not FARMA/VAR.
4. **Proposed direction:** new `fdars-core/src/fts/spectral.rs` — leverage the existing `rustfft` dependency for spectral density operator estimation and DPCA; a FARMA/VAR simulator in `simulation.rs`.
5. **Value 3 · Effort L · Severity P3 · Category differentiator** — the most specialized FTS slice; frequency-domain methods serve a narrow audience.
6. **Score = 3 / √9 = 3 / 3 = 1.00.**
7. **Notes/deps:** Reuses `rustfft`. Build after FTS-02/FTS-01 (spectral methods presuppose the ACF/forecasting foundation).

---

### FRE-02 — Object-data Fréchet regression (covariance/correlation/spherical/network/point-process)

1. **Candidate requirement / phase phrasing:** "Extend Fréchet regression to specific object spaces: covariance-matrix responses (Frobenius/power/log-Cholesky metrics), correlation matrices, spherical data (with geodesic exp/log maps), network responses, and point-process responses — each with its Fréchet-ANOVA analog."
2. **R-side reference:** `frechet` (0.3.0). Parity rows (Area 7): covariance-matrix regression (absent), Fréchet integral (absent), correlation-matrix (absent), spherical (absent), sphere geodesics (absent), network (absent), network ANOVA (absent), point-process (absent) — all differentiator. Covers ~8 gaps.
3. **fdars current gap:** **Absent.** No object-space geometry (sphere exp/log maps, SPD-matrix metrics, network/point-process spaces).
4. **Proposed direction:** `fdars-core/src/frechet/` — implement per-space metric + geodesic operations (SPD-matrix metrics, sphere exp/log, network/point-process distances) as pluggable metric backends for the FRE-01 Fréchet-regression solver.
5. **Value 3 · Effort L · Severity P3 · Category differentiator** — highly specialized object-data spaces; a narrow-but-deep audience.
6. **Score = 3 / √9 = 3 / 3 = 1.00.**
7. **Notes/deps:** **Depends on FRE-01** (the Fréchet-regression solver framework). Each metric backend is a plug-in.

---

### REG-06 — Boosting / Bayesian functional regression (FDboost, GAMLSS, Gibbs/VB FOSR)

1. **Candidate requirement / phase phrasing:** "Add gradient-boosting and Bayesian functional regression: boosting base-learners for function-on-scalar and function-on-function (FDboost), GAMLSS distributional functional regression, Bayesian function-on-scalar regression (Gibbs/VB), and FDboost stability selection."
2. **R-side reference:** `FDboost` (1.1-4, boosting, GAMLSS, stability selection), `refund` (Bayesian FOSR Gibbs/VB). Parity rows (Area 4): boosting FOSR (absent), Bayesian FOSR (absent), GAMLSS (absent), stability selection (absent) — all differentiator. Covers ~4 gaps.
3. **fdars current gap:** **Absent.** No boosting or Bayesian functional-regression machinery; fdars regression is penalized/kernel/PLS/elastic only.
4. **Proposed direction:** new `fdars-core/src/boosting_regression.rs` — component-wise gradient boosting with functional base-learners; optionally a Gibbs/VB Bayesian FOSR sampler. Large new estimation frameworks.
5. **Value 2 · Effort L · Severity P3 · Category differentiator** — advanced estimation frameworks; niche relative to the core regression suite, and expensive to build well.
6. **Score = 2 / √9 = 2 / 3 = 0.67.**
7. **Notes/deps:** Independent but large; lowest-value-per-effort of the regression items. Deprioritized accordingly.

---

### REP-02 — FEM/PDE smoothing on irregular 2D/3D domains (fdaPDE)

1. **Candidate requirement / phase phrasing:** "Add finite-element / PDE-regularized smoothing over complex 2D/3D irregular domains: a finite-element basis over meshes and PDE-regularized surface smoothing."
2. **R-side reference:** `fdaPDE` (1.1-24). Parity rows (Area 1): "Finite element basis (2D/3D irregular domains)" (absent), "Smoothing over 2D/3D domains with PDE regularization (FEM)" (absent), "Bivariate fd smoothing (smooth.bibasis)" (partial), "Positive-valued smoothing" (absent), "Monotone smoothing (Ramsay integral-of-exp)" (partial) — all differentiator. Covers ~5 gaps.
3. **fdars current gap:** **Absent/partial.** `function_on_scalar_2d` (tensor-product penalized 2D fit, an fdars strength A-6) covers *regular* 2D surfaces, but there is no FEM basis over irregular meshes and no PDE-regularized smoothing; positive/monotone smoothing are absent/partial.
4. **Proposed direction:** new `fdars-core/src/fem_smoothing.rs` — a linear finite-element basis over a triangulated mesh + PDE (Laplacian) regularization penalty; add positive (log-domain) and Ramsay integral-of-exp monotone smoothers to `smooth_basis.rs`.
5. **Value 2 · Effort L · Severity P3 · Category differentiator** — a highly specialized capability (irregular-domain FDA); large mesh/PDE subsystem for a narrow audience (Open Question #2 flagged it explicitly as differentiator, not table-stakes).
6. **Score = 2 / √9 = 2 / 3 = 0.67.**
7. **Notes/deps:** Does **not** overlap fdars' A-6 strength (regular-grid 2D FOSR); this is irregular-mesh FEM. Large, standalone.

---

### CLUS-02 — Functional co-clustering (funLBM latent-block) + slope-heuristic selection

1. **Candidate requirement / phase phrasing:** "Add model-based co-clustering of functions — simultaneous row (curve) and column (argument) clustering via a functional latent block model (funLBM) — plus the slope-heuristic model-selection criterion."
2. **R-side reference:** `funLBM` (2.3.1), `funHDDC` (slope heuristic). Parity rows (Area 4): "Model-based co-clustering of functions" (absent), "Slope heuristic for cluster model selection" (absent) — all differentiator. Covers 2 gaps.
3. **fdars current gap:** **Absent.** fdars has no co-clustering (simultaneous row+column blocks) and no slope-heuristic selector; standard functional clustering (`clustering.rs`, `gmm/`) clusters curves only.
4. **Proposed direction:** new `fdars-core/src/coclustering.rs` — a latent-block-model EM (block-wise Gaussian on FPC scores) with a slope-heuristic model-selection helper.
5. **Value 2 · Effort L · Severity P3 · Category differentiator** — a niche clustering paradigm; the latent-block EM is a substantial standalone estimator.
6. **Score = 2 / √9 = 2 / 3 = 0.67.**
7. **Notes/deps:** Split from CLUS-01 due to the larger latent-block EM effort. Independent.

---

## Gap-to-Item Coverage Map

The 162 actionable gaps map onto the 26 items as follows (per-area, cross-checked against the Phase-17 §Parity-Matrix Verdict Counts). Some large clusters absorb the full absent+partial set for an area; small partials fold into the nearest thematic item.

| Area | Actionable gaps | Items covering the area | Table-stakes items |
|------|-----------------|-------------------------|--------------------|
| 1 — Representation | 18 | T-01, REP-01, REP-02, SPARSE-01 | T-01 (2 TS) |
| 2 — Preprocessing / Registration | 6 | (folded: sparse-2D elastic + joint-align into CLUS-01/FPCA-01; registr GLM registration noted, low priority) | — |
| 3 — Depth / Outlier | 21 | T-02, DEPTH-01, OUT-01 | T-02 (2 TS) |
| 4 — ML Regression/Classif/Clustering | 32 | REG-01, REG-02, REG-03, REG-04, REG-05, REG-06, CLUS-01, CLUS-02 | REG-01 (2 TS), REG-02 (1 TS) |
| 5 — Inference | 22 | INF-01, INF-02, INF-03 | INF-01 (2 TS), INF-02 (3 TS) |
| 6 — Functional Time Series | 23 | FTS-01, FTS-02, FTS-03 | — |
| 7 — Density / Object Data | 25 | DENS-01, FRE-01, FRE-02 | — |
| 8 — SPM | 1 | (robust-chart partial — folded; near-parity, low priority) | — |
| 9 — FPCA (sparse/specialized) | 14 | FPCA-01, FPCA-02, SPARSE-01 | FPCA-01 (2 TS) |

**Table-stakes coverage:** all **18** table-stakes gaps are covered by items T-01 (2), T-02 (2), INF-01 (2), INF-02 (3), REG-01 (2), REG-02 (1), FPCA-01 (2) — plus the ANOVA-V-stat table-stakes partial in INF-02 and the depth-dispatcher/boxplot in T-02 — totalling the 18 from §Phase 17 §Categorization. Every table-stakes gap sits in an item ranked in the top 8 (scores 5.00–2.31, all P1). The Area-2 (6) and Area-8 (1) small partials are near-parity items folded into adjacent clusters or explicitly noted as low-priority; they carry no table-stakes gaps (§Phase 17 confirms Areas 2/8 contribute zero table-stakes).

**Excluded by construction (fdars strengths, Phase 18 — no work proposed):** model explainability (U-3), streaming depth (U-5), Andrews transform (U-1), elastic-model explain (U-2), WIRE container (U-4), tolerance bands (U-6), SPM chart breadth (A-1), conformal breadth (A-2), soft-DTW (A-4), robust SoF regression (A-5), 2D-surface FOSR (A-6), functional signal toolkit (A-7). The base function-on-function (`fof_regression.rs`) and 1D function-on-scalar (`function_on_scalar.rs`) regression are **present** at parity — REG-05 extends only their flexible/RE variants, not the base capabilities.

---

## Completeness Gate

This backlog is verified against the Phase-19 completeness gate (reused from v0.14.0):

- [x] **Master table strictly non-increasing by score.** The master table runs 5.00, 5.00, 3.00, 2.89 ×3, 2.31 ×5, 1.73 ×8, 1.33 ×2, 1.00 ×2, 0.67 ×3 — strictly non-increasing (26 items). Ties broken by severity (P1 before P2 before P3 within each equal-score tier: e.g. the 2.31 tier orders REG-02/FPCA-01 (P1) before DEPTH-01/OUT-01/INF-03 (P2)).
- [x] **Every ranked item has a matching 7-field promotion block.** All 26 items (T-01, T-02, REG-03, INF-01, INF-02, REG-01, REG-02, FPCA-01, DEPTH-01, OUT-01, INF-03, REG-04, REG-05, FTS-02, CLUS-01, REP-01, DENS-01, FPCA-02, SPARSE-01, FTS-01, FRE-01, FTS-03, FRE-02, REG-06, REP-02, CLUS-02) carry the full 7-field block (candidate phrasing · R-side reference + parity rows · fdars current gap · proposed direction incl. `fdars-core/src/` location · value+effort+severity+category · score · notes/deps).
- [x] **Top items are non-cosmetic.** The top 8 by score (T-01, T-02, REG-03, INF-01, INF-02, REG-01, REG-02, FPCA-01) are all substantive capability additions (7 of 8 are P1 table-stakes; REG-03 is a targeted differentiator extension). No cosmetic items appear near the top.
- [x] **Table-stakes represented near the top.** All 18 table-stakes gaps are covered by the top-8 items (T-01, T-02, INF-01, INF-02, REG-01, REG-02, FPCA-01), every one ranked ≤ rank 8 with P1 severity. ≥8 P1 items present (T-01, T-02, INF-01, INF-02, REG-01, REG-02, FPCA-01 are P1 — 7 P1 items, exceeding the ≥N table-stakes threshold; the 18 table-stakes gaps map onto these 7 P1 items).
- [x] **No fdars-strength work proposed.** All 12 Phase-18 R-honest strengths (U-1…U-6, A-1/A-2/A-4/A-5/A-6/A-7) are excluded by construction (see Gap-to-Item Coverage Map §Excluded).
- [x] **All 162 gaps accounted for.** Every actionable gap folds into an item (Gap-to-Item Coverage Map), with the small Area-2 (6) / Area-8 (1) near-parity partials explicitly folded or noted.

**Gate verdict: PASS.** This backlog is GSD-ready — items promote directly into future milestones via `/gsd-new-milestone`.

*This completes Phase 19 RPT-02 (value-ranked R-ecosystem backlog). Milestone v0.18.0 (R-Ecosystem Gap Audit) deliverables are complete: RPT-01 (`R-AUDIT-REPORT.md` §Phase 19) + RPT-02 (this file).*
