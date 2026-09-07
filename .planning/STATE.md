---
gsd_state_version: 1.0
milestone: v0.40.0
milestone_name: Correctness & Release Hardening
current_phase: 80
current_phase_name: Release Hardening & Ship v0.40.0
status: planning
stopped_at: Phase 79 complete, ready to plan Phase 80
last_updated: "2026-09-07T05:44:25.096Z"
last_activity: 2026-09-07
last_activity_desc: Phase 79 complete, transitioned to Phase 80
state_head: 51abecc006a8a6bc054752b41ac74ce1940d5af1
progress:
  total_phases: 3
  completed_phases: 2
  total_plans: 3
  completed_plans: 3
  percent: 67
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-09-06)

**Core value:** A comprehensive, fast Rust functional-data-analysis library that closes the highest-leverage capability and performance gaps against reference ecosystems. This milestone is a **correctness & release-hardening** pass: fix the bugs and build breakage found during recent milestones, formally validate the outstanding v0.39.0 phases, then bump/tag/publish fdars' first crates.io release since v0.38.0 (folding in the unpublished v0.39.0 forward-mode AD core).
**Current focus:** Phase 79 — Serde Feature Repair

## Current Position

Phase: 80 — Release Hardening & Ship v0.40.0
Plan: Not started
Status: Ready to plan
Last activity: 2026-09-07 — Phase 79 complete, transitioned to Phase 80

## Milestone Roadmap (v0.40.0)

Three phases, 5 requirements (CORR-01/02, BUILD-01, REL-01/02) — a small fixup/hardening milestone. Implementation milestone with real `fdars-core/src/` changes scoped to fixes/hardening (no new algorithms); additive/non-breaking (protects R + WASM bindings + 28 examples); no new crate dependency; behavior-preserving except where correcting the acknowledged `soft_dtw` bug. Real code → this milestone **does** get a `v0.40.0` git tag. Phase numbering continues from v0.39.0 (ended at 77) → Phase 78. Fine granularity, but the work compresses naturally to 3 phases (paired requirements + a final ship phase).

| Phase | Requirements | Notes |
|-------|--------------|-------|
| 78 — Gradient Correctness — soft_dtw Fix & Backward-Pass Audit | CORR-01, CORR-02 | **CORR-01:** concrete `soft_dtw_backward` endpoint-seed fix (`fdars-core/src/metric/soft_dtw.rs:262` — the reverse loop overwrites the `E[n][m]=1.0` seed with 0, zeroing the whole gradient; fix prototyped in that file's test `corrected_oracle_gradient`, ~line 467) + a regression test asserting non-zero gradient AND real barycenter movement vs the pointwise mean, cross-checked against the v0.39.0 `Dual` path; tighten existing `test_soft_dtw_barycenter_*` so they can't pass on an all-zero gradient. **CORR-02:** audit the ~10 sibling hand-written backward/gradient passes (`alignment/differentiable`, `autodiff`, `boosting_regression/gamlss`, `elastic_regression/logistic`, `explain_generic/counterfactual`, `regression`, `seasonal/mod`, `smooth_basis`, `metric/soft_dtw`) for analogous boundary-seed bugs — each disposed "clean" (one-line rationale) or "fixed" (+ regression test), all traceable. Behavior-preserving except the intended `soft_dtw` correction. Independent of Phase 79. |
| 79 — Serde Feature Repair | BUILD-01 | Self-contained. `ClassifFit` (`fdars-core/src/classification/fit.rs:49`) lacks the `#[cfg_attr(feature = "serde", derive(...))]` the rest of the crate uses; add derives to it + any embedded non-serde types until `cargo build --features serde` compiles again (broken since Phase 60), plus a serde round-trip test and a CI-runnable `--features serde` guard against re-breakage. Additive/non-breaking. Independent of Phase 78. |
| 80 — Release Hardening & Ship v0.40.0 | REL-01, REL-02 | **Must land last** — validates + folds in everything. **REL-01:** Nyquist sign-off of phases 75/76/77 `VALIDATION.md` (draft → validated) via the validate-phase flow; fill or record any coverage gaps. **REL-02:** crate bump 0.38.0 → 0.40.0, CHANGELOG (v0.39.0 AD core + v0.40.0), README/`documentation/` refresh (tracked docs live in `documentation/`; `docs/` is gitignored), whole-crate gates green. The actual `git tag v0.40.0` push → crates.io publish via `release.yml` is the final operator-driven step (documented in the SUMMARY, gated on all prior phases green) — the phase prepares + verifies release-readiness, it does not itself tag/publish. Depends on Phase 78 + Phase 79. |

**Execution order:** 78 and 79 are independent (either order / parallelizable); 80 lands last. All 5 requirements mapped, no orphans, no duplicates.

**Gates (this implementation milestone):** `cargo fmt --check`, `cargo clippy --all-targets --features linalg,parallel -- -D warnings` (CI lints test/bench code — use `--all-targets`, not a plain `-p` lint), `cargo test`. Plus a `--features serde` build/round-trip guard (Phase 79). Ships to crates.io on the operator `v0.40.0` tag (Phase 80 prepares + verifies release-readiness).

## Performance Metrics

**Velocity:**

- Total plans completed: 111+ (across v0.14.0–v0.39.0)
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
| 72–74 | v0.38.0 | 3 |
| 75–77 | v0.39.0 | 4 |
| 78–80 | v0.40.0 | 3/TBD (78 complete, 79 complete, 80 pending) |

**Recent Trend:**

- Last milestone: v0.39.0 DIFF phases 75–77 (4 plans) — audit 4/4, integration INTEGRATED; crate code-complete (tag/publish deferred, now folded into v0.40.0).
- Trend: v0.40.0 is a **fixup/hardening** milestone — lower net-new-code risk than a feature milestone. Real code changes but scoped to two concrete fixes (a known `soft_dtw` gradient bug + a known serde build break) plus an audited sweep and a validate-and-ship phase. Normal test/clippy/fmt gates.

*Updated after each plan completion*
**Per-Plan Metrics:**

| Plan | Duration | Tasks | Files |
|------|----------|-------|-------|
| 78-01 (CORR-01/02) | — | 3 | soft_dtw + gradient audit |
| 79-01 (BUILD-01) | — | 3 | serde derives on 5 types + round-trip test |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.

Relevant to current work (v0.40.0 Correctness & Release Hardening):

- **Fixup/hardening milestone, not features** — all parity/gap backlogs are exhausted; scope is correctness + build repair + validation + ship. No new algorithms.
- **CORR-01 = the acknowledged `soft_dtw_backward` endpoint-seed bug** — the reverse loop at `metric/soft_dtw.rs:262` overwrites the `E[n][m]=1.0` seed with 0, zeroing the whole gradient → `soft_dtw_barycenter` silently returns the pointwise mean without refining. Fix is prototyped in that file's `corrected_oracle_gradient` test (~line 467). Behavior-changing (that is the point) — but scoped to the endpoint-seed defect, NOT a redesign of the barycenter optimizer.
- **CORR-01 + CORR-02 grouped** (Phase 78) — CORR-01 is the concrete fix + regression test; CORR-02 is the sweep of the ~10 sibling hand-written gradient passes for analogous boundary-seed bugs, each disposed clean/fixed and traceable.
- **BUILD-01 self-contained** (Phase 79) — `ClassifFit` (`classification/fit.rs:49`) lacks the crate's conditional serde derives; add them + any embedded non-serde types until `cargo build --features serde` compiles, plus a CI guard. Independent of Phase 78.
- **REL-01 + REL-02 grouped** (Phase 80) — Nyquist sign-off of 75/76/77 then bump/CHANGELOG/docs/gates; lands last because it validates + folds in the earlier fixes.
- **This milestone gets a `v0.40.0` git tag** — real code changes (unlike audit milestones). The tag folds the unpublished v0.39.0 AD core + these fixes into fdars' first crates.io release since v0.38.0. Tag/publish is the final operator-driven step.
- **Additive/non-breaking, no new crate dependency** — carried conventions. Fixes reuse existing machinery.
- **Phase numbering continues** — v0.39.0 ended at Phase 77 → v0.40.0 starts at Phase 78. No reset.
- **5 requirements → 3 phases** (fine granularity, compressed): Phase 78 CORR-01/02; Phase 79 BUILD-01; Phase 80 REL-01/02. All 5 mapped, no orphans, no duplicates. 78 and 79 independent; 80 last.

### Pending Todos

- **Migrate `fdars-r` R wrapper to use the `FdMatrix` API** (issue `fdars-j75`) — carried forward; not this milestone.

### Blockers/Concerns

- **No research/SUMMARY.md** — intentional: this is a correctness/hardening pass on fdars' own code, not a feature/parity milestone, so no domain research was run. Non-blocking for the roadmap.
- **CORR-01 is behavior-changing** — unlike every recent milestone's additive-only stance, the `soft_dtw` fix intentionally changes gradient/barycenter output. Scope guard: fix ONLY the endpoint-seed defect + validating test; do NOT redesign the barycenter algorithm. Existing `test_soft_dtw_barycenter_*` must be tightened so they can't pass on an all-zero gradient.
- **CORR-02 sweep risk** — the audit may surface additional real bugs in the ~10 sibling gradient passes; each must be dispositioned clean/fixed and traceable. Budget for possible extra fix work inside Phase 78.
- Historical build/CI hazards (MEMORY.md) apply: run clippy with `--all-targets --features linalg,parallel -- -D warnings` (CI lints test/bench code); run `cargo fmt` per commit (`--no-verify` commits leave fmt drift); watch `/tmp` and `target/` disk pressure on full builds (`rm -rf target/debug/{incremental,examples}` to free space; doctests link in a small `/tmp` tmpfs — full → all commits fail with a bogus "No space left"); prefer inline execution + `commit --no-verify` after out-of-band gates if executor subagents stall on long cargo builds; the per-phase impl-subagent pattern dodged executor stalls on recent milestones.
- **serde build break is the BUILD-01 target** (`fdars-core/src/classification/fit.rs` `ClassifFit`; surfaced via `shapelet/classifier.rs` embedding it) — Phase 79 fixes it.

## Deferred Items

Items acknowledged and deferred, most recent first:

| Category | Item | Status | Deferred At | Milestone |
|----------|------|--------|-------------|-----------|
| Soft-DTW-optimizer | SDTW-O1 — replace the `soft_dtw_barycenter` inverse-curvature / soft-DBA majorization-minimization step (added in Phase 78 to stop the fixed-`lr` divergence that CORR-01 exposed) with a proper global optimizer (L-BFGS and/or multi-restart) for the non-convex soft-DTW barycenter objective. The MM step is locally stable + converging but not globally optimal. Surfaced by v0.40.0 Phase 78 code review (WR-03). | Deferred | v0.40.0 | future milestone |
| Differentiable-core | DIF-F1 (reverse-mode / VJP autodiff — needs a tape/graph engine); DIF-F2 (broaden the differentiable subset beyond elastic + FPCA — basis eval, inner products, SRSF/warping, other regressions); DIF-F3 (make existing f64 hot-path signatures themselves generic — breaking risk to R/WASM/examples) | Deferred | v0.39.0 | future milestone |
| VEESA | VEE-F1 (native random-forest / tree-ensemble predictor); VEE-F2 (plotting/rendering of principal directions + PFI) | Deferred | v0.38.0 | future milestone |
| Conformal-anomaly | ECA-F1 (full conditional / Mondrian conformal anomaly detection) | Deferred | v0.38.0 | future milestone |
| Wavelet-regression | WAV-F1 (binomial/logistic GLM-family); WAV-F2 (Symlets/Coiflets/biorthogonal + wavelet packets); WAV-F3 (2D/surface DWT) | Deferred | v0.37.0 | future milestone |
| API-breaking | APIB-01 — breaking removal of the 6 `#[deprecated]` forms from v0.30.0 | Deferred | v0.30.0 | future 1.0-readiness |

## Session Continuity

Last session: 2026-09-06T20:10:00.000Z
Stopped at: Phase 79 complete, ready to plan Phase 80
Resume file: None

## Operator Next Steps

- Plan the first phase with `/gsd-plan-phase 78` (or `79` — independent of 78).
