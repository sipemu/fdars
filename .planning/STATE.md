---
gsd_state_version: 1.0
milestone: v0.38.0
milestone_name: "VEESA: Elastic Shape Explainability & Conformal Anomaly Detection"
current_phase: 74
current_phase_name: Elastic Conformal Anomaly Detection
status: planning
stopped_at: Phase 73 complete, ready to plan Phase 74
last_updated: "2026-09-05T18:53:37.918Z"
last_activity: 2026-09-05
last_activity_desc: Phase 73 complete, transitioned to Phase 74
state_head: 0288489426085e54fe8d8418a6a86174437361ea
progress:
  total_phases: 3
  completed_phases: 2
  total_plans: 2
  completed_plans: 2
  percent: 67
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-09-05)

**Core value:** A comprehensive, fast Rust functional-data-analysis library that closes the highest-leverage capability gaps against reference ecosystems — this milestone brings fdars to parity with three Tucker-affiliated elastic-shape-analysis works (VEESA pipeline + elastic conformal anomaly detection), additively and reuse-first.
**Current focus:** Phase 73 — VEESA Explainability Pipeline & Integration

## Current Position

Phase: 74 — Elastic Conformal Anomaly Detection
Plan: Not started
Status: Ready to plan
Last activity: 2026-09-05 — Phase 73 complete, transitioned to Phase 74

Progress: [███░░░░░░░] 33%

## Milestone Roadmap (v0.38.0)

Three phases, 8 requirements (VEE-01..05, ECA-01..03) — an implementation milestone closing the gaps against the VEESA paper (Goode, Tucker & Ries), the `sandialabs/veesa` R package, and arXiv 2504.01172 (elastic conformal anomaly detection). Additive/non-breaking, reuse-first, **no new crate dependency**. Two dependency groups: the VEE group (Phases 72→73) is a chain (fit/transform seam → explainability pipeline built on it); the ECA group (Phase 74) is fully independent and sequenced last. Phase numbering continues from v0.37.0 (ended at 71) → Phase 72. Fine granularity; 3 phases (one per requirement group) matches recent milestone shape (WAV/PEER).

| Phase | Requirements | Notes |
|-------|--------------|-------|
| 72 — jfPCA Fit/Transform Seam | VEE-01, VEE-02 | Public jfPCA **fit** transformer (stores Karcher-mean template, `mean_psi`, `vert_component`/`horiz_component`, `balance_c`, `argvals`; training scores reproduce `joint_fpca` within 1e-8) + public **out-of-sample transform** (`prep_testing_data` equivalent: align new curves to the trained template, project onto the trained joint-FPCA basis, scores in trained coords; fit→transform round-trip within tolerance). Reuses `elastic_fpca.rs` (`joint_fpca`/`vert_fpca`/`horiz_fpca`, `JointFpcaResult`, private `project_onto_eigenvectors` now exposed via this seam) + `alignment/` (Karcher-mean alignment). Foundational — Phase 73 consumes it. |
| 73 — VEESA Explainability Pipeline & Integration | VEE-03, VEE-04, VEE-05 | **Model-agnostic PFI** over jfPCA PC scores (generic over any trained predictor / scoring closure, NOT tied to `elastic_pcr` — crate has no random forest, so PFI must be model-agnostic; reuses `elastic_explain.rs` permutation machinery; deterministic under seed; informative PC ranks above noise on known-signal design). **Principal-direction reconstruction** (functions at μ ± c·σⱼ per joint PC, split into amplitude + phase parts for plotting; `c=0` reproduces the jfPCA mean). **Cohesive pipeline + integration** (end-to-end fit → transform → PFI convenience path, full crate-root + prelude re-exports, running module doctest under `cargo test --doc`). Depends on Phase 72's seam. **UI hint**: yes (VEE-04 returns curves for plotting — rendering stays with the caller, VEE-F2 fence). |
| 74 — Elastic Conformal Anomaly Detection | ECA-01, ECA-02, ECA-03 | **Elastic nonconformity scores** — extend `NonConformityScore` (`{SupNorm, L2}` → adds amplitude / phase / combined elastic variants scoring a curve vs a reference template; each non-negative, zero for an identical curve; reuses `amplitude_distance`/`phase_distance`/`elastic_distance`). **Inductive conformal anomaly detector** — calibrate on a reference/calibration set (template e.g. calibration Karcher mean) → per-curve conformal p-values + anomaly flags at level α (flag rate ≈ α on exchangeable clean data; injected magnitude AND shape outliers flagged). **Result type + integration** — `ConformalAnomalyResult` (p-values, scores, flags, calibrated threshold) + crate-root/prelude re-exports + module doctest. Additive: existing `conformal_prediction_band` path unchanged. **Independent of the VEE group** — can sequence anywhere. |

**Execution order (dependency-driven):** 72 → 73 → 74. Phase 72 (fit/transform seam) is foundational to Phase 73 (explainability pipeline consumes the trained transform + PFI). Phase 74 (ECA) is fully independent of the VEE group and sequenced last for a clean linear order (could equally run first). All 8 requirements mapped, no orphans, no duplicates.

## Performance Metrics

**Velocity:**

- Total plans completed: 111+ (across v0.14.0–v0.37.0)
- Average duration: — min
- Total execution time: — hours

**By Phase (prior milestones):**

| Phase | Milestone | Plans |
|-------|-----------|-------|
| 01–09 | v0.14.0 | 21 |
| 10–45 | v0.15.0–v0.29.0 | 63 |
| 46–51 | v0.30.0 | 23 |
| 52–65 | v0.31.0–v0.35.0 | 21 |
| 66–68 | v0.36.0 | 3 |
| 69–71 | v0.37.0 | 5 |
| 72–74 | v0.38.0 | 0/? (pending) |

**Recent Trend:**

- Last milestone: v0.37.0 WAV phases 69–71 (5 plans) — audit PASSED 6/6, integration SOUND; crate code-complete 0.36.0 → 0.37.0 (tag/publish deferred). Promoted GAP-07 (`wcr`/`wnet`).
- Trend: v0.38.0 stays in implementation shape — real code, normal test/clippy/fmt gates. **Heavily reuse-first**: `elastic_fpca.rs` jfPCA machinery + private `project_onto_eigenvectors`, `alignment/` Karcher-mean + elastic distances, `elastic_explain.rs` permutation-importance, `tolerance/conformal.rs` conformal scaffolding + `NonConformityScore`. Lower net-new-code risk than WAV's from-scratch DWT — the core algorithms already exist; the work is exposing a fit→transform seam, a model-agnostic PFI generic, principal-direction reconstruction, and an elastic-distance conformal anomaly path. 3 phases driven by two requirement groups (VEE chain of 2 + independent ECA).

*Updated after each plan completion*
**Per-Plan Metrics:**

| Plan | Duration | Tasks | Files |
|------|----------|-------|-------|
| (none yet — v0.38.0) | — | — | — |
| Phase 72 P01 | 18 | 4 tasks | 4 files |
| Phase 73-veesa-explainability-pipeline-integration P01 | 50m | 5 tasks | 4 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.

Relevant to current work (v0.38.0 VEESA):

- **Implementation milestone, ships on tag** — v0.38.0 makes real `fdars-core/src/` changes; crate bump + `v0.38.0` tag + crates.io publish is a deferred operator ship step (per the established release-decoupling convention; crate is 0.28.0 published with 0.29.0–0.37.0 unreleased). Normal test/clippy (`--all-targets --features linalg,parallel`)/fmt gates apply.
- **Reuse-first, no new dependency** — do NOT re-implement: `elastic_fpca.rs` (`joint_fpca`/`vert_fpca`/`horiz_fpca`, `JointFpcaResult`, private `project_onto_eigenvectors`), `alignment/` (`amplitude_distance`/`phase_distance`/`elastic_distance`, `karcher_mean`), `elastic_explain.rs` (permutation-importance machinery), `tolerance/conformal.rs` (`conformal_prediction_band`, `NonConformityScore`). No `Cargo.toml` change.
- **PFI is model-agnostic** — the crate has NO built-in random forest. The VEE-03 PFI layer must be generic over any trained predictor / scoring closure (matching VEESA's design), NOT tied to `elastic_pcr`. A native tree-ensemble learner is explicitly deferred (VEE-F1).
- **jfPCA fit→transform contract** — VEE-01 exposes a reusable fit transformer on the `JointFpcaResult` family (stores template + eigen components + `balance_c` + `argvals`); VEE-02 exposes out-of-sample transform reusing the existing private `project_onto_eigenvectors`. Training-score reproduction (1e-8) and fit→transform round-trip are the make-or-break numerical gates.
- **Elastic conformal extends, never breaks** — ECA-01 adds elastic variants to `NonConformityScore` (`{SupNorm, L2}` → + amplitude/phase/combined); the existing `conformal_prediction_band` path stays unchanged (additive-only, `#[non_exhaustive]` enum forward-compat).
- **Additive/non-breaking** — zero changes to existing public signatures (protects R + WASM bindings + 28 examples); only new modules/types + additive `lib.rs`/`prelude.rs` re-exports. Deprecate, never remove.
- **Phase numbering continues** — v0.37.0 ended at Phase 71 → v0.38.0 starts at Phase 72. No reset.
- **8 requirements → 3 phases** (fine granularity): Phase 72 VEE-01/02, Phase 73 VEE-03/04/05, Phase 74 ECA-01/02/03. All 8 mapped, no orphans, no duplicates. VEE group is a dependency chain (72→73); ECA group (74) is independent, sequenced last.
- [Phase 72]: Added training_gammas/training_aligned to JfpcaModel to expose Karcher alignment output for exact round-trip scoring
- [Phase 72]: score_training() method provides < 1e-8 round-trip via stored alignment; transform() uses align_to_target for out-of-sample curves
- [Phase 73]: Stored karcher.mean_srsf (mu_q_centered) as JfpcaModel::mean_srsf field for exact c=0 principal-direction reconstruction
- [Phase 73]: Used karcher_mean[0] as f0 anchor for srsf_inverse (augmented element encodes midpoint, not initial value)

### Pending Todos

- **Migrate `fdars-r` R wrapper to use the `FdMatrix` API** (issue `fdars-j75`) — carried forward; the additive VEESA + conformal-anomaly surface should be exposed to R/WASM bindings in a follow-up, not this milestone.

### Blockers/Concerns

- **No research/SUMMARY.md** — broad ecosystem research was intentionally skipped this milestone (the three source papers/repo + a codebase scan already pinned the gaps precisely). Non-blocking for the roadmap. Numerical make-or-break gates warrant known-answer tests: training-score reproduction (1e-8), fit→transform round-trip, PFI seed-determinism + known-signal PC ranking, principal-direction `c=0` reproduces the mean, elastic nonconformity zero-for-identical, marginal-validity flag rate ≈ α, injected magnitude+shape outliers flagged, module doctests under `cargo test --doc`.
- **jfPCA transform correctness** — the private `project_onto_eigenvectors` must be exposed and driven correctly for out-of-sample curves (align to trained template first, then project onto trained basis). Confirm at plan time that alignment of new curves to the *trained* Karcher mean matches the training-time alignment convention exactly, else the round-trip gate fails.
- **PFI generic surface** — confirm at plan time whether `elastic_explain.rs`'s permutation machinery can be lifted to a predictor-agnostic trait/closure without breaking its existing (`elastic_pcr`-specific) callers; if not, add a thin additive generic layer alongside (still additive/non-breaking).
- Historical build/CI hazards (MEMORY.md) apply this implementation milestone: run clippy with `--all-targets --features linalg,parallel -- -D warnings` (CI lints test/bench code); run `cargo fmt` per commit (`--no-verify` commits leave fmt drift); watch `/tmp` and `target/` disk pressure on full builds (`rm -rf target/debug/{incremental,examples}` to free space); prefer inline execution + `commit --no-verify` after out-of-band gates if executor subagents stall on long cargo builds; the per-phase impl-subagent pattern (plan→code→test→--no-verify commit→artifacts) dodged executor stalls on v0.32.0 GAK.
- Pre-existing serde build break (NOT v0.38.0): `fdars-core/src/shapelet/classifier.rs` `ShapeletTransformClassifier` (Phase 60, commit ea39c623) embeds non-serde `ClassifFit` → `cargo build --features serde` fails. Independent of VEESA; new types should be serde-clean (or serde-gated) if they derive serde. Candidate GSD-ready backlog fix: add serde-gated derives to `ClassifFit`.

## Deferred Items

Items acknowledged and deferred, most recent first:

| Category | Item | Status | Deferred At | Milestone |
|----------|------|--------|-------------|-----------|
| VEESA | VEE-F1 (native random-forest / tree-ensemble predictor — PFI kept model-agnostic instead); VEE-F2 (plotting/rendering of principal directions + PFI — VEE-04 returns curves, rendering is a caller concern) | Deferred | v0.38.0 | future milestone |
| Conformal-anomaly | ECA-F1 (full conditional / Mondrian conformal anomaly detection — class-conditional validity; v1 covers the inductive marginal case) | Deferred | v0.38.0 | future milestone |
| Backlog | GAP-08 (autodiff-compatible / differentiable FDA core — invasive generics refactor, score 1.73, L) — the last remaining v0.31.0 backlog item | Deferred | v0.36.0 | future milestone |
| Wavelet-regression | WAV-F1 (binomial/logistic GLM-family); WAV-F2 (Symlets/Coiflets/biorthogonal + wavelet packets); WAV-F3 (2D/surface DWT) | Deferred | v0.37.0 | future milestone |
| API-breaking | APIB-01 — breaking removal of the 6 `#[deprecated]` forms from v0.30.0 | Deferred | v0.30.0 | future 1.0-readiness |

## Session Continuity

Last session: 2026-09-05T18:33:19.099Z
Stopped at: Phase 73 complete, ready to plan Phase 74
Resume file: None

## Operator Next Steps

- Plan the first phase with /gsd-plan-phase 72
- Deferred crate-release steps (bump + tag + publish for v0.29.0–v0.38.0) remain an operator concern.
