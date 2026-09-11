---
gsd_state_version: 1.0
milestone: v0.44.0
milestone_name: Differentiable Core — Reverse-Mode & Broadened Subset
current_phase: 100
current_phase_name: Release Preparation & Verification
status: planning
stopped_at: Phase 99 complete, ready to plan Phase 100
last_updated: "2026-09-11T19:06:25.230Z"
last_activity: 2026-09-11
last_activity_desc: Phase 99 complete, transitioned to Phase 100
state_head: 2f487dcdba10eef35ab5e32ce339d9a1e2e0d7e9
progress:
  total_phases: 7
  completed_phases: 6
  total_plans: 15
  completed_plans: 15
  percent: 86
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-09-09)

**Core value:** A comprehensive, fast Rust functional-data-analysis library. This milestone completes the **differentiable-core** section of the 1.0 gap checklist (`documentation/ROADMAP-TO-1.0.md`, DIF-F1/F2/F3) — an in-crate reverse-mode (VJP) autodiff core alongside the v0.39.0 forward-mode `Dual`, a broadened differentiable operation set across four algorithm families, and generic-over-scalar hot-path signatures. Implementation milestone, **strictly additive/non-breaking** (defaulted type params `T = f64` — protects R + WASM bindings + 28 examples), **no new crate dependency** (in-crate hand-written tape).
**Current focus:** Phase 97 — Differentiable Regression Prediction & Smoothing Penalties

## Current Position

Phase: 100 — Release Preparation & Verification
Plan: Not started
Status: Ready to plan
Last activity: 2026-09-11 — Phase 99 complete, transitioned to Phase 100

## Milestone Roadmap (v0.44.0)

Seven phases, 11 requirements — an implementation milestone completing the differentiable-core section of `documentation/ROADMAP-TO-1.0.md` (DIF-F1/F2/F3). Strictly additive/non-breaking (defaulted type params `T = f64`; protects R + WASM + 28 examples). No new crate dependency (in-crate hand-written reverse-mode tape, exactly as forward-mode `Dual` was built in v0.39.0). Fine granularity. Phase numbering continues from v0.43.0 (ended at 93) → **Phase 94**. No reset.

| Phase | Requirements | Notes |
|-------|--------------|-------|
| 94 — Reverse-Mode Autodiff Core (VJP Tape) | RAD-01, RAD-02, RAD-03 | **DIF-F1.** In-crate Wengert-list tape (`Var`) + full op set (RAD-01), backward pass + `vjp` entry point for many-input→scalar objectives (RAD-02), validated vs forward-mode `Dual` and finite differences on the existing differentiable subset — elastic soft-DTW + FPCA scores (RAD-03). Largely independent; sequenced first because RAD-03 validates against the already-shipped v0.39.0 subset. No new crate dependency. First phase. |
| 95 — Generic Scalar Hot-Path Signatures | GEN-01 | **DIF-F3 enabler.** Generalize targeted hot-path signatures over the scalar type via defaulted type params (`T = f64`) so every existing f64 call site, R + WASM binding, and all 28 examples compile unchanged. GEN-01 is the compile-time proof of non-breakingness and the substrate the DOP families are written against. Depends on 94 (both autodiff types validated to flow). |
| 96 — Differentiable Basis Evaluation & Inner Products | DOP-01 | **DIF-F2 family 1.** Basis eval (B-spline/Fourier) + functional inner products generic over `Scalar` and differentiable; f64 path reproduces current numerics. Depends on 95. |
| 97 — Differentiable Regression Prediction & Smoothing Penalties | DOP-02, DOP-03 | **DIF-F2 families 2+3.** `fregre_lm`/FPCR prediction differentiable w.r.t. inputs (DOP-02) + roughness-penalty evaluation differentiable (DOP-03); f64 parity preserved, both FD-checked. Grouped because prediction composes the penalty/basis machinery. Depends on 96. |
| 98 — Differentiable Depth & Curve Distances | DOP-04 | **DIF-F2 family 4.** Functional depth + curve distances (beyond soft-DTW) generic over `Scalar` and differentiable; FD-checked, f64 parity preserved. Independent of 96/97 — only needs the 95 substrate. Depends on 95. |
| 99 — End-to-End Autodiff Flow & Gradient API | GEN-02, API-01 | **DIF-F3 + API.** Autodiff types flow through the generalized hot-paths end-to-end; a composed objective yields FD-checked gradients (GEN-02). Unified `grad`/`jacobian`/`vjp` entry points + worked composition demo + full crate-root/prelude re-exports + running module doctest (API-01). Depends on 96, 97, 98 (composition exercises the broadened subset) and 94 (`vjp`). |
| 100 — Release Preparation & Verification | REL-01 | **Terminal.** Bump 0.43.0 → 0.44.0, CHANGELOG `[0.44.0]` (root + crate), check off DIF-F1/F2/F3 on `ROADMAP-TO-1.0.md`; all whole-crate gates green — fmt/clippy `--all-targets`/full `cargo test`/`--features serde` build/28 examples+doctests/`cargo package`. The full-suite gate is the end-to-end proof of the milestone. The `git tag v0.44.0` → crates.io publish is the DEFERRED operator step. Depends on all prior phases. |

**Execution order:** 94 → 95 → 96 → 97 → 98 → 99 → 100. Reverse-mode core first (validates against the shipped subset), then the generic hot-path enabler (GEN-01), then the four DOP families (96/97 chained, 98 independent), then end-to-end flow + gradient API, then release prep last. All 11 requirements mapped, no orphans, no duplicates.

**Gates (this additive, non-breaking milestone):** `cargo fmt --check`, `cargo clippy --all-targets --features linalg,parallel -- -D warnings` (CI lints test/bench code — use `--all-targets`), full `cargo test` (the differentiability + non-regression proof), a `--features serde` build guard, all 28 examples + doctests, and `cargo package`. **No new crate dependency** (in-crate tape).

## Performance Metrics

**Velocity:**

- Total plans completed: 129+ (across v0.14.0–v0.43.0)
- Average duration: — min
- Total execution time: — hours

**By Phase (prior milestones):**

| Phase | Milestone | Plans |
|-------|-----------|-------|
| 01–45 | v0.14.0–v0.29.0 | 84 |
| 46–51 | v0.30.0 | 23 |
| 52–77 | v0.31.0–v0.39.0 | ~24 |
| 78–80 | v0.40.0 | 5 |
| 81–85 | v0.41.0 | 9 |
| 86–89 | v0.42.0 | 4 |
| 90–93 | v0.43.0 | 4 |
| 94–100 | v0.44.0 | 0/? (planned) |

**Recent Trend:**

- Last milestone: v0.43.0 (phases 90–93, 4 plans) — audit 7/7, release-ready (operator tag/publish pending). Cleared the **Quality** blocker on the 1.0 checklist.
- Trend: v0.44.0 is an implementation milestone completing the **differentiable-core** section of the 1.0 checklist — additive/non-breaking, no new crate dependency, building on the v0.39.0 forward-mode AD core. After this, only SDTW-O1 (algorithm) and fdars-j75 (R ecosystem) remain before the terminal 1.0-CUT.

*Updated after each plan completion*
**Per-Plan Metrics:**

| Plan | Duration | Tasks | Files |
|------|----------|-------|-------|
| Phase 94 P01 | 7min | 3 tasks | 4 files |
| Phase 94 P02 | 4min | 2 tasks | 1 files |
| Phase 94 P03 | 5min | 2 tasks | 1 files |
| Phase 94 P04 | 10min | 3 tasks | 1 files |
| Phase 95 P01 | 12min | 3 tasks | 1 files |
| Phase 95 P02 | 18 | 3 tasks | 3 files |
| Phase 95 P03 | 5min | 3 tasks | 0 files |

## Accumulated Context

### Decisions

Decisions relevant to current work (v0.44.0):

- **Strictly additive/non-breaking** — all generalization via defaulted type params (`T = f64`); existing f64 signatures, R + WASM bindings, and all 28 examples must compile unchanged. GEN-01 (Phase 95) is the compile-time proof.
- **No new crate dependency** — the reverse-mode tape (RAD, Phase 94) is hand-written in-crate, matching how the forward-mode `Dual` was built in v0.39.0.
- **GEN-01 lands early as the enabler (Phase 95)** — you cannot make an operation differentiable (DOP) without its hot path first being generic over the scalar type; the DOP families (96/97/98) are written against the generic substrate.
- **Reverse-mode core first (Phase 94)** — RAD-01/02/03 are largely independent, but RAD-03 validates against the *existing* differentiable subset (soft-DTW, FPCA scores), which already exists from v0.39.0; sequenced first.
- **DOP families split by natural boundary** — 96 (basis eval + inner products), 97 (regression prediction + smoothing penalties, grouped because prediction composes the penalty/basis machinery), 98 (depth + curve distances, independent of 96/97).
- **API-01 + GEN-02 land together (Phase 99)** — the composition demo exercises the broadened subset end-to-end and uses the reverse-mode `vjp` entry point; near-last.
- **REL-01 lands last (Phase 100)** — bump/CHANGELOG/checklist/gates are the final proof; the full-suite `cargo test` gate is the end-to-end milestone proof. The `git tag v0.44.0` → crates.io publish is the DEFERRED operator step (GSD `git.create_tag` is off because `release.yml` couples tag-push to publish) — never tagged/published inside a phase.
- **Phase numbering continues** — v0.43.0 ended at Phase 93 → v0.44.0 starts at Phase 94. No reset.
- **11 requirements → 7 phases:** 94 RAD-01/02/03; 95 GEN-01; 96 DOP-01; 97 DOP-02/03; 98 DOP-04; 99 GEN-02/API-01; 100 REL-01. All mapped, no orphans, no duplicates.
- [Phase 94]: Scalar trait moved to mod.rs (shared between forward and reverse modes) — Enables both forward.rs and reverse.rs to share the same trait definition without duplication
- [Phase 94]: Tape fully opaque (not exported from prelude); only vjp is the public entry point — Per CONTEXT.md discretion: Tape is fully hidden behind vjp; Phase 99 can add inspection if needed
- [Phase 94]: Div three-case constant-folding for Var: both-const→SENTINEL, rhs-const→push_unary(1/v), general→push_binary quotient rule (common x/gamma path is the rhs-const fast path)
- [Phase 94]: signum for Var returns SENTINEL constant (no tape node) — piecewise-constant function has zero gradient, matches Dual zero-tangent convention
- [Phase 94]: Double-clear tape lifecycle was already correct in Plan 02 — confirmed by repeated-call stability tests showing zero gradient drift across 3 successive vjp calls
- [Phase 94]: Tier-3 agreement tests import crate::autodiff::{Dual, grad} from inside reverse.rs tests — valid because autodiff/mod.rs re-exports both at crate::autodiff path
- [Phase 94]: Tier-4 FD cross-check uses <Var as Scalar>::zero() disambiguation in test closures — Rust cannot infer the Scalar impl from Fn(&[Var])->Var alone when calling free-standing trait methods
- [Phase 94]: prelude.rs Var+vjp re-exports were already correct from Plan 01; Plan 04 only updated the vjp doctest to show 2-input Var-annotated example
- [Phase 95]: Removed invalid = f64 default on free function (rustc 1.97 rejects invalid_type_param_default on fns); T=f64 inference at existing call sites is unaffected
- [Phase 95]: [Phase 95 Plan 01]: use crate::autodiff::Scalar added at module top of helpers.rs (not cfg(test)) — required for public generic function bound
- [Phase 95]: trapz/inner_product/inner_product_l2 generalized in-place to T: Scalar; .sum() rewritten to T::zero() accumulator; f64 call sites unchanged via T=f64 inference
- [Phase 95]: serde build GREEN — the pre-existing ShapeletTransformClassifier/ClassifFit serde issue did not manifest; build clean at 18.42s
- [Phase 95]: fdars-r not local — non-breaking proof via Gates 2+4; four kernels remain pub and more general

### Pending Todos

- **Operator ship steps still pending** — `git tag v0.41.0`, `git tag v0.42.0`, `git tag v0.43.0` → push → crates.io publish not yet performed; registry is still at 0.40.0. A future 0.44.0 publish (or an earlier catch-up publish) supersets the unpublished versions.
- **Migrate `fdars-r` R wrapper to the `FdMatrix` API** (issue `fdars-j75`) — carried forward; separate package, out of `fdars-core` scope; a 1.0-ecosystem gap.

### Blockers/Concerns

- **No research/SUMMARY.md** — intentional: internal extension of the v0.39.0 forward-mode AD core, not an ecosystem-parity audit. Reference baseline is the existing v0.39.0 design (`Scalar` trait, `Dual<T>`, `grad`/`jacobian`) plus the Julia ForwardDiff / reverse-mode idiom. Non-blocking for the roadmap.
- **Non-breakingness is the central constraint** — GEN-01 (Phase 95) is the compile-time proof: all 28 examples + R/WASM bindings + every f64 call site must compile unchanged. Guard it at every DOP phase (f64 parity preserved).
- **Warp-searched `elastic_distance` (the DP) stays non-differentiable** — deferred at v0.39.0 (DIF-02) and unchanged here; the amplitude-at-warp / soft-DTW surrogates remain the differentiable paths. Do NOT try to differentiate the discrete DP argmin.
- **`soft_dtw_barycenter` optimizer redesign (SDTW-O1) is out of scope** — differentiability of the distance is in scope; the barycenter optimizer is a separate algorithm-quality item.
- Historical build/CI hazards (MEMORY.md) apply: run clippy `--all-targets --features linalg,parallel -- -D warnings` (CI lints test/bench code); run `cargo fmt` per commit (`--no-verify` commits leave fmt drift → CI fmt-check fails); keep the `--features serde` build green (repaired v0.40.0 — do not regress); watch `/tmp` and `target/` disk pressure on full builds (`rm -rf target/debug/{incremental,examples}` to free space; doctests link in a small `/tmp` tmpfs); the pre-commit hook runs the full cargo gate and times out — prefer inline execution + `commit --no-verify` after out-of-band gates; full combined gate as one background bash gets killed mid-run — run gates per-gate FOREGROUND with 600s timeout.

## Deferred Items

Items acknowledged and deferred, most recent first:

| Category | Item | Status | Deferred At | Milestone |
|----------|------|--------|-------------|-----------|
| 1.0-cut | 1.0-CUT — bump to 1.0.0 and declare the public API stable, once every ROADMAP-TO-1.0.md item clears | Deferred | v0.41.0 | future (deliberate 1.0 cut) |
| Algorithm | SDTW-O1 — replace the `soft_dtw_barycenter` MM-step descent with a proper global optimizer (L-BFGS / multi-restart) | Deferred | v0.40.0 | future milestone |
| fdars-r | `fdars-r` FdMatrix migration (issue `fdars-j75`) — migrate the external R wrapper to the `FdMatrix` API; separate package, out of `fdars-core` scope | Deferred | v0.41.0 | future milestone |

## Session Continuity

Last session: 2026-09-11T06:45:06.985Z
Stopped at: Phase 99 complete, ready to plan Phase 100
Resume file: None

## Operator Next Steps

- Review the v0.44.0 roadmap (`.planning/ROADMAP.md`), then plan the first phase with `/gsd-plan-phase 94`.
