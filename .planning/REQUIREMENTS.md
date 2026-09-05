# Requirements: fdars — v0.38.0 VEESA

**Defined:** 2026-09-05
**Core Value:** Close the highest-leverage capability gaps against reference FDA ecosystems, additively and reuse-first — this milestone brings fdars to parity with three Tucker-affiliated elastic-shape-analysis works (VEESA pipeline + elastic conformal anomaly detection).

## Sources

- **VEESA paper** — Goode, Tucker & Ries, *"Explainable Machine Learning for Functional Data"*, Journal of Data Science (`jds1212.pdf`).
- **`sandialabs/veesa`** — R package implementing the VEESA pipeline (`prep_training_data`, `prep_testing_data`, `compute_pfi`, jfPCA).
- **arXiv 2504.01172** — Adams, Berman, Michalenko & Tucker, *"Conformal Anomaly Detection for Functional Data with Elastic Distance Metrics"*.

## v1 Requirements (this milestone)

Real `fdars-core/src/` code. All additive/non-breaking, reuse-first, **no new crate dependency**.

### VEESA Pipeline (VEE) — jfPCA fit/transform seam + model-agnostic explainability

- [ ] **VEE-01**: A public jfPCA **fit** step produces a reusable transformer (a model object, or an extension of `JointFpcaResult`) that stores everything needed to project new curves — the trained Karcher-mean template, `mean_psi`, joint eigenvector components (`vert_component`/`horiz_component`), `balance_c`, and `argvals`. Training scores from the fit reproduce `joint_fpca`'s scores within 1e-8.
- [ ] **VEE-02**: A public **out-of-sample transform** (`prep_testing_data` equivalent) aligns new raw curves to the *trained* Karcher-mean template and projects them onto the *trained* joint-FPCA basis, returning scores in the trained coordinate system. Transforming the original training curves reproduces the training scores within tolerance (fit→transform round-trip). Reuses the existing private `project_onto_eigenvectors`, now exposed via this seam.
- [ ] **VEE-03**: **Model-agnostic permutation feature importance (PFI)** over jfPCA principal-component scores — computes importance for a caller-supplied trained predictor (generic over a predictor trait / scoring closure, not tied to `elastic_pcr`), by permuting each PC-score column and measuring the degradation in a user-selected error metric. Deterministic under a seed; on a known-signal design the informative PC ranks above noise PCs.
- [ ] **VEE-04**: **Principal-direction reconstruction** for visualization — reconstruct functions at μ ± c·σⱼ along each joint PC and split the perturbation into its amplitude (warped-function) and phase (warping-function) parts, returning curves suitable for plotting/interpretation. `c = 0` reproduces the jfPCA mean function.
- [ ] **VEE-05**: **Cohesive VEESA pipeline + integration** — an end-to-end convenience path tying fit (align + jfPCA) → out-of-sample transform → PFI, with full crate-root + prelude re-exports and a running end-to-end module doctest (`cargo test --doc`).

### Elastic Conformal Anomaly Detection (ECA)

- [ ] **ECA-01**: **Elastic nonconformity scores** — extend `NonConformityScore` (currently `{SupNorm, L2}`) with elastic variants scoring a curve against a reference template: amplitude distance, phase distance, and combined elastic distance (reusing `amplitude_distance`/`phase_distance`/`elastic_distance`). Each score is non-negative and zero for a curve identical to the reference.
- [ ] **ECA-02**: **Inductive conformal anomaly detector** — calibrate elastic nonconformity scores on a reference/calibration set (against a template such as the calibration Karcher mean), then for new curves return conformal **p-values** and **anomaly flags** at a chosen level α. On exchangeable clean data the flag rate is approximately α (marginal validity); injected **magnitude** and **shape** outliers are flagged.
- [ ] **ECA-03**: **Result type + integration** — a `ConformalAnomalyResult` (per-curve conformal p-values, nonconformity scores, boolean flags, and the calibrated threshold), crate-root + prelude re-exports, and a running module doctest. Additive/non-breaking; the existing `conformal_prediction_band` path is unchanged.

## Future Requirements (deferred)

- **VEE-F1**: Native random-forest / tree-ensemble predictor for the VEESA pipeline — deferred; the milestone keeps PFI model-agnostic (works with any external/existing predictor) rather than adding a new learner.
- **ECA-F1**: Full conditional / Mondrian conformal anomaly detection (class-conditional validity) — deferred; v1 covers the inductive (marginal) case from arXiv 2504.01172.
- **VEE-F2**: Plotting/rendering of principal directions and PFI — out of the numeric-library scope (VEE-04 returns the curves; rendering is a caller concern).

## Out of Scope

| Feature | Reason |
|---------|--------|
| Plotting / ggplot-style rendering (`veesa` R plots) | fdars is a numeric library — VEE-04 returns the curves to plot; rendering stays with the caller (consistent with prior audit fences) |
| New learner (random forest, etc.) | PFI is kept model-agnostic; adding a learner is a separate, larger scope (VEE-F1) |
| New crate dependency | Milestone constraint — all three groups build on existing jfPCA / elastic-distance / conformal machinery |
| Breaking changes to existing signatures | Additive-only; protects R + WASM bindings + 28 examples (deprecate, never remove) |
| Data/IO loaders for the paper datasets | Out of scope, consistent with prior audit fences |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| VEE-01 | Phase 72 | Pending |
| VEE-02 | Phase 72 | Pending |
| VEE-03 | Phase 73 | Pending |
| VEE-04 | Phase 73 | Pending |
| VEE-05 | Phase 73 | Pending |
| ECA-01 | Phase 74 | Pending |
| ECA-02 | Phase 74 | Pending |
| ECA-03 | Phase 74 | Pending |

**Coverage:**
- v1 requirements: 8 total
- Mapped to phases: 8 ✓
- Unmapped: 0 ✓

---
*Requirements defined: 2026-09-05*
*Last updated: 2026-09-05 after roadmap creation (Phases 72–74 mapped)*
