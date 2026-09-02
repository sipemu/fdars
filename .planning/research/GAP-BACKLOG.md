# Multi-Ecosystem Gap Backlog (v0.31.0)

**Milestone:** v0.31.0 — Multi-Ecosystem Gap Audit · RPT-02
**Compiled:** 2026-09-02
**Source:** `GAP-AUDIT-REPORT.md` (RPT-01), which consolidates the four Phase 52 surveys.
**Audit-only:** zero `fdars-core/src/` edits. Every item is a *candidate* for a future implementation milestone — nothing here is built in v0.31.0.

---

## Ranking Methodology

Consistent with the v0.14.0 `BACKLOG.md` and v0.18.0 `R-BACKLOG.md` conventions.

**Formula:** `score = value / √effort`

**Value scale (1–5):** 1 = marginal · 2 = minor · 3 = solid/useful · 4 = high-leverage · 5 = high-leverage + corroborated across multiple reference libraries.

**Effort map:** S = 1 (small, atop existing machinery) · M = 2 (moderate, new module reusing primitives) · L = 3 (large, invasive or needs new infrastructure). Denominator is `√effort`: √1=1.00, √2≈1.41, √3≈1.73.

**Convergence boost:** a gap corroborated across multiple reference libraries gets +1 value (evidence of real demand). Applied to **shapelets** (tslearn + sktime + pyts).

**Tie-break rule (equal score):** (a) higher reuse of existing fdars machinery ranks higher (lower integration risk); (b) core-capability generalizations rank above niche additions; (c) concrete methods rank above architectural refactors.

---

## Ranked Backlog

| Rank | ID | Gap | Value | Effort | Score | Ecosystem | Reference (pkg@ver) |
|------|----|-----|-------|--------|-------|-----------|---------------------|
| 1 | GAP-01 | Global Alignment Kernel (GAK) + kernel-k-means/SVM on curves | 3 | S | 3.00 | Python | tslearn@0.9.0 |
| 2 | GAP-02 | Shapelet transform / shapelet-based classification | 5 | L | 2.89 | Python (×3 libs) | tslearn@0.9.0 / sktime / pyts@0.13.x |
| 3 | GAP-03 | k-Shape clustering (SBD, FFT cross-correlation) | 3 | M | 2.12 | Python | tslearn@0.9.0 |
| 4 | GAP-05 | Optimal experimental design for sparse FDA (FOptDes) | 3 | M | 2.12 | MATLAB | PACE@2.17 |
| 5 | GAP-06 | PEER / longitudinal PEER (structured-penalty SoF regression) | 3 | M | 2.12 | R/refund | refund@0.1-38 |
| 6 | GAP-07 | Wavelet-domain functional regression (wcr/wnet) | 3 | L | 1.73 | R/refund | refund@0.1-38 |
| 7 | GAP-08 | Autodiff-compatible / differentiable FDA core | 3 | L | 1.73 | Julia | ElasticFDA.jl + Zygote idiom |

*7 ranked net-new items. Rank 3–5 share score 2.12; ordered by tie-break rule (reuse-proximity + generalization). Rank 6–7 share 1.73; concrete method (wcr) above architectural refactor (differentiable). **GAP-04 (multi-domain MFPCA) was demoted to out-of-scope by the RPT-03 completeness gate** — see below — because v0.18.0 `R-BACKLOG.md` REP-01 already flagged `funData`'s `multiFunData` multi-domain container; the capability is already-adjacent, not cleanly net-new. Its ID is retired from the ranked list; IDs GAP-05..08 keep their original identifiers.*

---

## Backlog Items (promotion-ready blocks)

### GAP-01 — Global Alignment Kernel (GAK) + kernel methods on curves
- **Candidate requirement/phase:** `GAK-01` — a PSD global-alignment kernel + kernel-k-means (and kernel-SVM glue) for curve sets.
- **Value 3 · Effort S · Score 3.00**
- **Reference baseline:** tslearn@0.9.0 (`gak`, `KernelKMeans`).
- **Rationale:** fdars already ships `metric/soft_dtw` (distance + differentiable barycenter) but no PSD *kernel* enabling kernel machines directly on curves. GAK is a small addition atop existing soft-DTW machinery that unlocks a whole class of kernel methods — highest score by virtue of low effort / high leverage.
- **Reuse:** `metric/soft_dtw.rs`, `distance.rs`, existing clustering.

### GAP-02 — Shapelet transform / shapelet-based classification
- **Candidate requirement/phase:** `SHP-01` — shapelet discovery/learning + shapelet transform + classifier.
- **Value 5 (convergence-boosted) · Effort L · Score 2.89**
- **Reference baseline:** tslearn@0.9.0 (`ShapeletModel`), sktime shapelet transform, pyts@0.13.x.
- **Rationale:** interpretable local-shape primitives for classification; the *only* gap corroborated across three independent Python libraries → strongest demand signal in the audit. Higher effort (discovery + learning + transform) keeps it at rank 2 behind the cheap GAK win.
- **Reuse:** `classification/`, `distance.rs`, `metric/`.

### GAP-03 — k-Shape clustering (SBD)
- **Candidate requirement/phase:** `KSH-01` — shape-based distance (SBD) via FFT cross-correlation + k-Shape centroid refinement.
- **Value 3 · Effort M · Score 2.12**
- **Reference baseline:** tslearn@0.9.0 (`KShape`), sktime.
- **Rationale:** a distinct inductive bias from fdars' SRVF/elastic clustering — normalized cross-correlation shape matching, fast via FFT. Complements existing clustering families.
- **Reuse:** `rustfft` (already a dependency), `clustering.rs`, `alignment/`.

### GAP-04 — Multi-dimensional heterogeneous-domain MFPCA — ⚠ DEMOTED TO OUT-OF-SCOPE (OOS-03)
- **RPT-03 gate ruling:** the v0.18.0 `R-BACKLOG.md` REP-01 already lists `funData`'s `multiFunData` multi-domain container as an absent, differentiator capability. Multi-domain functional data (container + joint analysis) was therefore already surfaced in the prior R audit; treating FDApy's multi-domain MFPCA as net-new would re-litigate an already-backlogged theme, violating the hard de-dup rule ("by capability, not name"). Recorded as **OOS-03** below. The base same-type `mfpca` already ships (`spm/mfpca.rs`).

### GAP-05 — Optimal experimental design for sparse FDA (FOptDes)
- **Candidate requirement/phase:** `FOD-01` — choose measurement/sampling locations minimizing FPCA-score prediction MSE under a sparse-design budget.
- **Value 3 · Effort M · Score 2.12**
- **Reference baseline:** PACE@2.17 (MATLAB) `FOptDes`.
- **Rationale:** principled sparse-sampling design; builds directly on fdars' existing `pace_fpca` covariance/eigen machinery.
- **Reuse:** `pace_fpca.rs`, `covariance.rs`.

### GAP-06 — PEER / longitudinal PEER
- **Candidate requirement/phase:** `PER-01` — Partially Empirical Eigenvectors for Regression (structured a-priori-penalty scalar-on-function regression; longitudinal variant).
- **Value 3 · Effort M · Score 2.12**
- **Reference baseline:** refund@0.1-38 (`peer`, `lpeer`). NOT captured in v0.18.0.
- **Rationale:** a structured-penalty SoF regression distinct from FPCR/`pfr`; lets users inject a-priori signal structure into the penalty.
- **Reuse:** `scalar_on_function/`, `regression.rs`, `smoothing.rs`.

### GAP-07 — Wavelet-domain functional regression (wcr/wnet)
- **Candidate requirement/phase:** `WAV-01` — DWT + regularized (ridge/lasso/elastic-net) scalar-on-function regression in the wavelet domain.
- **Value 3 · Effort L · Score 1.73**
- **Reference baseline:** refund@0.1-38 (`wcr`, `wnet`). NOT captured in v0.18.0.
- **Rationale:** sparse regression for spiky/localized functional predictors. Higher effort because fdars has `rustfft` but no discrete wavelet transform — a DWT must be built first.
- **Reuse:** would add a DWT module; `scalar_on_function/`, existing regularized-regression paths.

### GAP-08 — Autodiff-compatible / differentiable FDA core
- **Candidate requirement/phase:** `DIF-01` — a scoped differentiable subset (e.g. differentiable elastic distance / FPCA) supporting gradients for embedding in larger optimization/ML pipelines.
- **Value 3 · Effort L · Score 1.73**
- **Reference baseline:** Julia generic-programming idiom (ElasticFDA.jl + ForwardDiff/Zygote).
- **Rationale:** strategic/architectural — Julia's generic code lets AD flow through FDA ops. High long-term value but invasive (generics-over-scalar refactor of hot paths), so ranked last among rankable items and scoped to a subset. Concrete methods rank above it per the tie-break rule.
- **Reuse:** `metric/soft_dtw.rs` (already has a hand-written gradient — the natural pilot), `elastic_*`.

---

## Recorded Out-of-Scope (not ranked — see RPT-03 completeness gate)

| ID | Candidate | Ecosystem | Reason recorded out-of-scope |
|----|-----------|-----------|------------------------------|
| OOS-01 | GPU-friendly / batched-broadcast FDA kernels | Julia idiom | fdars targets a portable CPU/WASM numeric core; GPU acceleration conflicts with the deployment model. Revisit only if a GPU backend becomes a project goal. |
| OOS-02 | SAX / PAA / bag-of-patterns symbolic & imaging representations | Python (pyts/sktime) | Time-series-ML *representations* (discretization/imaging), not functional numeric methods; outside the FDA-numeric scope fence. |
| OOS-03 | Multi-dimensional heterogeneous-domain MFPCA (was GAP-04) | Python (FDApy) | Already-adjacent to v0.18.0 `R-BACKLOG.md` REP-01 (`funData` `multiFunData` multi-domain container). Demoted by the RPT-03 gate — not cleanly net-new. Base same-type `mfpca` already ships. |

---

*Backlog compiled 2026-09-02. Promote top-first into a future implementation milestone via `/gsd-review-backlog`.*
