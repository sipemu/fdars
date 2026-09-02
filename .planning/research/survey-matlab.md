# Survey: MATLAB FDA Ecosystem (MAT-01)

**Milestone:** v0.31.0 Multi-Ecosystem Gap Audit — Phase 52
**Survey date:** 2026-09-02
**Ecosystem:** MATLAB functional data analysis
**Packages surveyed (version-pinned as of 2026-09):**
- `fdaM@6.x` — Ramsay's `fda` MATLAB toolbox (the "fdaM" functions; MATLAB port tracks the R `fda` package, latest confirmable line 6.x). Source: McGill FDA downloads / functional-data-analysis.net.
- `PACE@2.17` (MATLAB) — Müller/Wang group Principal Analysis by Conditional Estimation toolbox (anson.ucdavis.edu/~mueller/data/pace.html). The R sibling `fdapace@0.6.0` was already audited in v0.18.0; the MATLAB PACE toolbox "contains some methods not available on fdapace and vice versa."

**De-dup baselines:** shipped fdars (`fdars-core/src/` + PROJECT.md Validated list), `BACKLOG.md` (v0.14.0 scikit-fda), `R-BACKLOG.md` (v0.18.0 R), and — for rigor — `R-AUDIT-REPORT.md` (fdapace was surveyed there; PACE-MATLAB methods already surveyed in v0.18.0 are treated as already-considered, not net-new).

**Audit-only fence:** zero `fdars-core/src/` edits. Only this file was written.

---

## Capability Inventory & fdars Parity Mapping

Standardized columns (Phase 53 merge contract):
`Capability | Reference (pkg@ver) | fdars status | Searched-for | Net-new? | Notes`

### Ramsay `fda` MATLAB toolbox (fdaM)

| Capability | Reference (pkg@ver) | fdars status | Searched-for | Net-new? | Notes |
|---|---|---|---|---|---|
| Basis systems: B-spline, Fourier, constant, monomial, polygonal, power, exponential (8 types) | fdaM@6.x | partial | `grep basis/ smooth_basis.rs`: B-spline, Fourier, constant present; monomial/power/exponential/polygonal not all present | No | Missing basis families already tracked in `BACKLOG.md` REPR-01 and `R-BACKLOG.md` REP-01. Not net-new. |
| Data2fd / smooth_basis / smooth.basisPar (raw→fd conversion) | fdaM@6.x | present | `grep smooth_basis.rs smoothing.rs`: smoothing + basis conversion present | No | Covered. |
| Penalized smoothing w/ Lfd roughness penalties (int2Lfd, vec2Lfd) | fdaM@6.x | present | `grep pda.rs smoothing.rs fem_smoothing.rs`: linear differential operator + penalties present | No | LDO tracked; PDA present (`pda.rs`). |
| Registration: landmarkreg, register.fd (continuous), smooth-monotone | fdaM@6.x | present | `grep landmark.rs warping.rs alignment/`: landmark + elastic/continuous registration present | No | Covered broadly. |
| AmpPhaseDecomp (amplitude/phase variance decomposition) | fdaM@6.x | present | `grep alignment/pairwise.rs warping.rs`: `amplitude_distance`, `phase_distance` present | No | Covered. |
| fRegress (functional linear model, scalar & functional response) | fdaM@6.x | present | `grep scalar_on_function/ function_on_scalar.rs fof_regression.rs`: FLM present | No | Covered. |
| glmfit_fda (functional GLM, exponential-family response) | fdaM@6.x | absent | `grep -riE "poisson\|binomial\|link_function" src/`: only distribution helpers + functional_logistic | No | Functional GLM exponential-family (Poisson/Gamma) already in `R-BACKLOG.md` REG-02. Not net-new. |
| pca.fd / FPCA + varmx rotation | fdaM@6.x | present | `grep regression.rs fpca_variants.rs`: FPCA + variants present | No | Covered. |
| Principal differential analysis (pda.fd) | fdaM@6.x | present | `grep pda.rs` | No | Covered. |
| Functional canonical correlation (cca.fd) | fdaM@6.x | present | `grep -ri "canonical\|cca" src/` → covariance/inference | No | Functional CCA present/adjacent; not surfaced as gap. |

### PACE (MATLAB) — methods beyond the v0.18.0-audited fdapace

| Capability | Reference (pkg@ver) | fdars status | Searched-for | Net-new? | Notes |
|---|---|---|---|---|---|
| Sparse/longitudinal FPCA via conditional expectation (PACE core) + trajectory CI bands | PACE@2.17 | present | `grep pace_fpca.rs`: `pace_fpca`, `PaceFpcaConfig`, `PaceFpcaResult` | No | Shipped (closed `R-BACKLOG.md` FPCA-01). |
| FClust (functional clustering) | PACE@2.17 | present | `grep clustering.rs gmm/ coclustering.rs` | No | Covered. |
| FCReg (functional concurrent regression, 2D smoothing) | PACE@2.17 | present | `grep concurrent_regression.rs` | No | Covered. |
| FSVD (functional singular value decomposition) | PACE@2.17 | absent | `grep -ri "fsvd\|functional.*svd" src/` → none | No | Already in `R-BACKLOG.md` FPCA-02. Not net-new. |
| FVPA (functional variance process analysis) | PACE@2.17 | absent | `grep -riE "variance.process\|fvpa" src/` → none | No | Already surveyed in `R-AUDIT-REPORT.md` (fdapace). Already-considered, not net-new. |
| WFDA (pairwise time-warping / dynamic time warping alignment) | PACE@2.17 | present | `grep warping.rs alignment/ metric/soft_dtw.rs`: warping + soft-DTW present | No | Covered. |
| Stringing (high-dim vector → functional via optimal ordering) | PACE@2.17 | absent | `grep -ri "stringing" src/` → none | No | Surveyed in `R-AUDIT-REPORT.md`; left off `R-BACKLOG.md` in v0.18.0. Already-considered, not net-new. |
| Empirical dynamics / dynamic correlation (FADynamics, GetDynamicalCorr) | PACE@2.17 | absent | `grep -riE "dynamic.correl\|empirical.dynamic" src/` → none | No | "empirical dynamic" appears in `R-AUDIT-REPORT.md` (fdapace scope). Already-considered, not net-new. |
| **FOptDes — optimal experimental design / measurement-point selection for sparse FDA** | PACE@2.17 | absent | `grep -riE "optimal.design\|foptdes\|sensor.placement" src/` → **none**; `grep R-AUDIT-REPORT.md R-BACKLOG.md BACKLOG.md` → **none** | **Yes** | Genuinely net-new: choose sampling/measurement locations to minimize FPCA-score prediction error under a sparse-design budget. Surveyed by neither prior audit. |

---

## Net-New Gap List (MATLAB FDA)

Filtered to capabilities VERIFIED absent from shipped fdars AND both `BACKLOG.md`/`R-BACKLOG.md`, AND not already surveyed-and-declined in the v0.18.0 `R-AUDIT-REPORT.md`.

| Capability | Reference (pkg@ver) | fdars status | Searched-for | Net-new? | Notes |
|---|---|---|---|---|---|
| **Optimal experimental design for sparse FDA (FOptDes)** | PACE@2.17 (MATLAB) | absent | `optimal.design/foptdes/sensor.placement` → none in `src/`; none in either backlog or R-AUDIT | Yes | Value: enables principled sparse-sampling designs (which time points to measure) minimizing FPCA prediction MSE. Effort: M (build on existing `pace_fpca` covariance machinery + a greedy/convex design criterion). Reference baseline: PACE `FOptDes`. |

**Net-new gap count: 1.**

The MATLAB FDA surface is almost entirely covered by fdars or already tracked in prior backlogs. PACE's more exotic methods (FVPA, stringing, empirical dynamics, FSVD) were already surveyed in the v0.18.0 fdapace audit and are excluded as already-considered per the hard de-dup rule.

---

## Reverse-Parity Note (where fdars LEADS MATLAB FDA)

- **Elastic/SRVF shape analysis** (`elastic_*`, `warping.rs`) — fuller SRVF/SRSF framework, elastic FPCA, elastic changepoint, and shape CIs than fdaM's registration.
- **Explainability** (`explain/`, `explain_generic/`) — PDP/SHAP/LIME/ALE for functional models has no MATLAB-FDA analog.
- **Conformal prediction** (`conformal/`), **SPM control charts** (`spm/`), **streaming depth** (`streaming_depth/`) — no MATLAB-FDA equivalent.
- **Soft-DTW with differentiable barycenter** (`metric/soft_dtw.rs`) — beyond WFDA's classical warping.
- **Memory-safe column-major numeric core in Rust** — deployable to WASM/JS and R; MATLAB toolboxes are interpreter-bound.
