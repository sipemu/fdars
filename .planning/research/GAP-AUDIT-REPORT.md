# Multi-Ecosystem Gap Audit Report (v0.31.0)

**Milestone:** v0.31.0 — Multi-Ecosystem Gap Audit
**Compiled:** 2026-09-02
**Deliverable:** RPT-01 (this report) · companion RPT-02 `GAP-BACKLOG.md` · RPT-03 completeness gate (appended below)
**Audit-only:** zero `fdars-core/src/` edits.

This report consolidates the four Phase 52 ecosystem surveys into a single cross-ecosystem picture of what fdars is missing relative to four *fresh* reference ecosystems, after both prior parity backlogs (scikit-fda v0.14.0, R core v0.18.0) were exhausted.

---

## 1. Methodology

**Yardsticks (four fresh ecosystems):**
- **MATLAB FDA** — Ramsay `fda` MATLAB toolbox (`fdaM@6.x`) + PACE (MATLAB) `PACE@2.17`.
- **Julia FDA** — `ElasticFDA.jl@1.x`, `FDA.jl@0.x`, `MultivariateStats.jl@0.10.x`, `registr`; plus modern Julia perf/idiom patterns.
- **tidyfun/refund (R)** — `tf@0.3.x` + `tidyfun@0.x` data-representation slice; `refund@0.1-38` methods NOT already captured in v0.18.0.
- **Python-beyond-scikit-fda** — `FDApy@1.0.4`, `tslearn@0.9.0`, `sktime@0.3x`, `pyts@0.13.x` (scikit-fda excluded).

**Process (per ecosystem):** enumerate the capability surface capability-first (major categories, version-pinned) → map fdars present/partial/absent with an explicit "searched fdars for:" grep-evidence note per absent/partial row (mapped by capability, not API name) → de-duplicate.

**De-dup rule (hard):** a gap is net-new only if verified **absent from shipped fdars** (grep of `fdars-core/src/` + PROJECT.md Validated list) **AND absent from both** `BACKLOG.md` (v0.14.0) and `R-BACKLOG.md` (v0.18.0), checked by capability.

**"Already-considered" rigor (stricter than the hard rule):** capabilities that the v0.18.0 `R-AUDIT-REPORT.md` already surveyed and consciously left off `R-BACKLOG.md` are treated as already-considered and excluded — even if absent from both backlogs. This caught the PACE methods (FVPA, stringing, empirical dynamics, FSVD) whose R sibling `fdapace` was audited in v0.18.0, and refund's exponential-family registration.

**Scope fences:** no plotting/visualization parity; no data/IO parity; no re-audit of scikit-fda (v0.14.0) or the core R FDA ecosystem (v0.18.0); refund only where not in v0.18.0; scikit-fda excluded from PYX-01.

**Key context:** fdars is an exceptionally broad FDA library — it already ships `pace_fpca`, `frechet`, `fts`, `elastic_*`, `coclustering`, `boosting_regression`, `density_fda`, `famm`, `concurrent_regression`, `conformal`, `gmm`, `metric/soft_dtw` (incl. differentiable barycenter), `seasonal/matrix_profile`, multivariate FPCA, depth, SPM, streaming depth, explainability, and more. Consequently, net-new gaps against these fresh ecosystems are **few and specialized**, which is itself the headline finding.

---

## 2. Per-Ecosystem Findings

### 2.1 MATLAB FDA (`survey-matlab.md`) — 1 net-new
The Ramsay `fdaM` toolbox and PACE are almost entirely covered by fdars or already tracked in prior backlogs (missing basis families → `BACKLOG` REPR-01/`R-BACKLOG` REP-01; functional GLM → `R-BACKLOG` REG-02; FSVD → `R-BACKLOG` FPCA-02). PACE's exotic methods (FVPA, stringing, empirical dynamics) were already surveyed in the v0.18.0 fdapace audit → already-considered.
- **Net-new:** PACE `FOptDes` — optimal experimental design / measurement-point selection for sparse FDA. Not surveyed by either prior audit.

### 2.2 Julia FDA (`survey-julia.md`) — 2 net-new (1 solid + 1 flagged)
Julia's FDA *packages* are method-wise covered by or behind fdars (ElasticFDA.jl = SRVF, which fdars exceeds; registr's exp-family registration already-considered in v0.18.0). The distinctive contribution is **architectural**.
- **Net-new (solid):** autodiff-compatible / differentiable FDA core — gradients through warping/FPCA/regression via generic number types (ForwardDiff/Zygote idiom); fdars is `f64`-concrete.
- **Net-new (flagged out-of-scope):** GPU-friendly / batched-broadcast kernels — likely out-of-scope for a portable CPU/WASM numeric core.

### 2.3 tidyfun / refund (`survey-tidyfun.md`) — 2 net-new
tidyfun's `tf` layer is data-representation & tidyverse ergonomics whose numeric ops fdars already provides; reshaping/plotting fall under scope fences. refund was thoroughly audited in v0.18.0, leaving two structured/wavelet regression methods.
- **Net-new:** `peer`/`lpeer` — Partially Empirical Eigenvectors for Regression (structured-penalty scalar-on-function regression; longitudinal PEER).
- **Net-new:** `wcr`/`wnet` — wavelet-domain scalar-on-function regression (wavelet compression + lasso/elastic-net); fdars has `rustfft` but no DWT.

### 2.4 Python-beyond-scikit-fda (`survey-pyx.md`) — 3 net-new + 2 out-of-scope (post-gate)
FDApy's dimension-reduction/simulation and tslearn's DTW/soft-DTW/barycenter are already present in fdars. Net-new gaps concentrate at the FDA/ML boundary. *(The Phase 52 survey initially proposed 4 candidates; the RPT-03 gate demoted multi-domain MFPCA — see below.)*
- **Net-new:** shapelet transform / shapelet-based classification (tslearn, sktime, pyts — **convergent within Python across 3 libs**).
- **Net-new:** k-Shape clustering (SBD — FFT cross-correlation shape-based distance).
- **Net-new (slice):** Global Alignment Kernel (GAK) — PSD kernel for kernel k-means/SVM on curves.
- **Demoted to out-of-scope by RPT-03:** multi-dimensional heterogeneous-domain MFPCA (FDApy) — already-adjacent to v0.18.0 `R-BACKLOG.md` REP-01 (`funData` `multiFunData` multi-domain container). fdars ships same-type `mfpca`.
- **Flagged out-of-scope:** SAX/PAA/bag-of-patterns symbolic representations (TS-ML representations, not functional numeric methods).

---

## 3. Cross-Ecosystem Convergence Analysis

Convergence = the same capability gap appearing independently in ≥2 of the four ecosystems.

| Convergent theme | Ecosystems | Notes |
|---|---|---|
| **Shape-primitive methods (shapelets, shape-based clustering)** | Python only (tslearn + sktime + pyts) — strong *intra-Python* convergence across 3 libraries | Not cross-*ecosystem*, but the multi-library agreement within Python is a strong value signal → priority boost in RPT-02. |
| Optimal sparse-design / measurement selection | MATLAB (PACE FOptDes) only | Ecosystem-specific. |
| Structured/penalized & wavelet functional regression | R/refund only (peer/lpeer, wcr/wnet) | Ecosystem-specific. |
| Differentiable / performance-oriented architecture | Julia only | Ecosystem-specific (idiomatic, not a discrete method). |
| Multi-domain MFPCA | Python/FDApy only | Ecosystem-specific. |

**Headline convergence finding:** cross-*ecosystem* convergence is **LOW** — each ecosystem's net-new gaps are largely distinct. This is a positive signal: fdars already covers the common FDA core shared across all four ecosystems, so the residual gaps are each ecosystem's *specialties* (MATLAB's sparse-design; R's structured regression; Julia's differentiable architecture; Python's shape-ML methods). The one strong convergence is **within** Python around shapelet/shape-primitive methods.

---

## 4. Reverse-Parity Strengths Sweep (where fdars LEADS)

Consolidated across all four surveys, fdars leads every surveyed ecosystem in:
- **Elastic/SRVF shape analysis** — full square-root framework, elastic FPCA/regression/changepoint, phase boxplots, shape CIs (exceeds ElasticFDA.jl, fdaM registration, tslearn DTW-family).
- **Statistical depth & robustness** — depth measures, functional boxplots, outlier detection, streaming depth (absent from MATLAB/Julia/tidyfun/Python-non-skfda).
- **Explainability** — PDP/SHAP/LIME/ALE for functional models (no analog anywhere surveyed).
- **Inference & regression breadth** — scalar-on/function-on-function/concurrent/Fréchet/boosting regression, permutation & ANOVA inference.
- **Conformal prediction & SPM** — conformal bands + control charts (no analog).
- **Determinism & deployment** — reproducible seeded parallelism; compiled Rust column-major core to WASM/JS + R bindings vs interpreter/runtime-bound MATLAB/R/Julia/Python.

---

## 5. Summary Counts

| Ecosystem | Net-new (ranked) | Recorded out-of-scope |
|---|---|---|
| MATLAB FDA | 1 | 0 |
| Julia FDA | 1 | 1 (GPU) |
| tidyfun/refund | 2 | 0 |
| Python-beyond-skfda | 3 | 2 (SAX; multi-domain MFPCA — demoted by RPT-03) |
| **Total** | **7** | **3** |

The ranked, promotion-ready backlog for the 7 net-new gaps is in `GAP-BACKLOG.md` (RPT-02). The de-dup + completeness gate (RPT-03) is appended below. **Total surveyed candidates: 10** (7 ranked + 3 recorded out-of-scope). *Note: the initial Phase 52 Python survey surfaced 4 candidates; the RPT-03 gate demoted multi-domain MFPCA to out-of-scope on de-dup grounds — see the gate section.*

---

## RPT-03 — De-dup & Completeness Gate (runs LAST)

**Purpose:** independently re-verify every ranked backlog row is genuinely net-new, and confirm every surveyed candidate is accounted for (ranked or explicitly out-of-scope). This is a second pass, independent of the Phase 52 survey de-dup.

### Independent De-dup Re-verification

Each ranked row was re-checked by grep against `fdars-core/src/`, `BACKLOG.md`, and `R-BACKLOG.md`:

| ID | fdars src | BACKLOG | R-BACKLOG | Verdict |
|----|-----------|---------|-----------|---------|
| GAP-01 GAK | 0 | 0 | 0 | NET-NEW ✓ |
| GAP-02 shapelets | 0 | 0 | 0 | NET-NEW ✓ |
| GAP-03 k-Shape (SBD) | 0* | 0 | 0 | NET-NEW ✓ |
| GAP-05 FOptDes | 0 | 0 | 0 | NET-NEW ✓ |
| GAP-06 PEER/lpeer | 0 | 0 | 0 | NET-NEW ✓ |
| GAP-07 wcr/wnet | 0 | 0 | 0 | NET-NEW ✓ |
| GAP-08 differentiable FDA | 0 | 0 | 0 | NET-NEW ✓ |

\* GAP-03: a wildcard grep matched 3 SRVF-shape files (`alignment/shape.rs`, `alignment/shape_ci.rs`, `depth/tvd.rs`); a literal search for `kshape|shape.based.distance|sbd` returns **empty**. k-Shape (Paparrizos SBD clustering) is a distinct capability from fdars' SRVF/elastic shape analysis — confirmed net-new.

**Gate catch (the gate did its job):** the candidate **GAP-04 multi-dimensional heterogeneous-domain MFPCA** returned an `R-BACKLOG.md` hit (REP-01: `funData` `multiFunData` multi-domain container, listed absent in v0.18.0). Multi-domain functional data was already surfaced in the prior R audit; the candidate is already-adjacent, not cleanly net-new. **Demoted to out-of-scope (OOS-03).**

### Completeness — every surveyed candidate accounted for

10 candidates surfaced across the four Phase 52 surveys; all 10 are dispositioned:

| # | Candidate | Ecosystem | Disposition |
|---|-----------|-----------|-------------|
| 1 | FOptDes optimal design | MATLAB | Ranked GAP-05 |
| 2 | Differentiable/autodiff FDA | Julia | Ranked GAP-08 |
| 3 | GPU/batched kernels | Julia | Out-of-scope OOS-01 |
| 4 | PEER/lpeer | R/refund | Ranked GAP-06 |
| 5 | wcr/wnet wavelet regression | R/refund | Ranked GAP-07 |
| 6 | Shapelets | Python | Ranked GAP-02 |
| 7 | k-Shape (SBD) | Python | Ranked GAP-03 |
| 8 | GAK kernel | Python | Ranked GAP-01 |
| 9 | Multi-domain MFPCA | Python | Out-of-scope OOS-03 (demoted this gate) |
| 10 | SAX/PAA/symbolic | Python | Out-of-scope OOS-02 |

**Coverage: 10/10.** Ranked = 7 · Out-of-scope (with reasoning) = 3. No candidate is silently dropped.

### Audit-only Fence
`git status --porcelain fdars-core/src/` → empty across the whole milestone. FENCE_OK.

### Gate Assertions
1. Every ranked backlog item independently re-verified net-new (absent from fdars + both backlogs): **PASS** (7/7).
2. Every surveyed candidate ranked or explicitly recorded out-of-scope with reasoning: **PASS** (10/10).
3. Ranking strictly descending by score: **PASS** (3.00 ≥ 2.89 ≥ 2.12 ≥ 2.12 ≥ 2.12 ≥ 1.73 ≥ 1.73).
4. Distinct deliverable filenames (no overwrite of prior AUDIT-REPORT/BACKLOG/R-*): **PASS** (`GAP-AUDIT-REPORT.md`, `GAP-BACKLOG.md`).
5. Zero `fdars-core/src/` edits: **PASS**.

## COMPLETENESS GATE: PASS ✅ (all 5 assertions)
