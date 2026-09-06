---
gsd_state_version: 1.0
milestone: v0.39.0
milestone_name: "DIFF: Differentiable FDA Core (Forward-Mode Autodiff)"
status: Awaiting next milestone
stopped_at: Phase 77 complete — all phases complete
last_updated: "2026-09-06T17:55:52.021Z"
last_activity: 2026-09-06
last_activity_desc: Milestone v0.39.0 completed and archived
state_head: f7cd35c9f305f7d220cf081fa40595002abac396
progress:
  total_phases: 3
  completed_phases: 3
  total_plans: 4
  completed_plans: 4
  percent: 100
current_phase: 77
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-09-05)

**Core value:** A comprehensive, fast Rust functional-data-analysis library that closes the highest-leverage capability gaps against reference ecosystems — this milestone promotes GAP-08 (the last v0.31.0 `GAP-BACKLOG.md` item) by adding an in-crate forward-mode automatic-differentiation core so exact gradients flow through a scoped subset of FDA operations (elastic distance + FPCA scores) into optimization/ML pipelines.
**Current focus:** Phase 75 — Scalar Trait & Forward-Mode Dual Substrate

## Current Position

Phase: Milestone v0.39.0 complete
Plan: —
Status: Awaiting next milestone
Last activity: 2026-09-06 — Milestone v0.39.0 completed and archived

## Milestone Roadmap (v0.39.0)

Three phases, 4 requirements (DIF-01..04) — an implementation milestone promoting **GAP-08** (score 1.73, L-effort), the last remaining item in the v0.31.0 `GAP-BACKLOG.md`. Adds an in-crate forward-mode automatic-differentiation core (a `Scalar` trait + `Dual<T>` number) and makes a scoped subset of FDA ops (**elastic distance + FPCA scores**) generic over the scalar type, so exact gradients compose through arbitrary op chains. Reference baseline: the Julia generic-programming idiom (ElasticFDA.jl + ForwardDiff). Additive/non-breaking (existing f64 signatures untouched; generic versions live alongside — protects R + WASM bindings + 28 examples), **no new crate dependency** (in-crate dual numbers, forward-mode only). Phase numbering continues from v0.38.0 (ended at 74) → Phase 75. Fine granularity; 3 phases matches recent milestone shape (WAV/PEER/VEESA/FOptDes used 2–3).

| Phase | Requirements | Notes |
|-------|--------------|-------|
| 75 — Scalar Trait & Forward-Mode Dual Substrate | DIF-01 | In-crate `Scalar` trait + forward-mode `Dual<T>` number (value + tangent) implementing the ops the differentiable subset needs (±, ×, ÷, `sqrt`, `exp`, `ln`, `sin`, `cos`, `powf`, `abs`, partial comparisons) + gradient seed (set an input's tangent to 1) / extract helpers. `Scalar` implemented for `f64` so f64-instantiated generic code is identical to current numerics. **Make-or-break gate:** dual arithmetic reproduces analytical derivatives of composed elementary functions to ≤1e-10 (known-answer tests). No new crate dependency; existing f64 signatures untouched. **Foundational — blocks Phases 76 & 77** (everything is generic over this substrate). |
| 76 — Differentiable Elastic Distance & FPCA Scores | DIF-02, DIF-03 | The two scoped ops made generic-over-`Scalar`, both written against the Phase-75 substrate. **Elastic distance (DIF-02):** generic soft-DTW / amplitude+phase path; at `Dual` yields exact forward-mode gradients w.r.t. a curve's values — validated vs central finite differences AND the existing hand-written `soft_dtw` gradient; at `f64` reproduces `elastic_distance`/`amplitude_distance` within 1e-12. Pilot on `metric/soft_dtw.rs` (existing gradient to validate against) + `elastic_*`. **FPCA scores (DIF-03):** generic FPCA score projection so gradients of FPC scores w.r.t. input-curve values flow through at `Dual` — validated vs central finite differences; f64 reproduces existing FPCA scores within tolerance. Pilot on `regression.rs` FPCA. The two ops are **independent of each other** (disjoint code areas) — either order within the phase. Additive/non-breaking, no new dependency. Depends on Phase 75. |
| 77 — Gradient API, Composition Demo & Integration | DIF-04 | The wrap-up. Ergonomic public `(value, gradient)` / directional-derivative / Jacobian entry point over the `Scalar`-generic subset; a worked end-to-end example composing the differentiable ops into a scalar objective and taking its gradient (proving AD flows through composition); full crate-root + prelude re-exports; a running module doctest under `cargo test --doc`. Consumes the differentiable ops from Phase 76 — **must land last**. |

**Execution order (dependency-driven):** 75 → 76 → 77. Hard chain: DIF-01 (Phase 75) is the substrate DIF-02/DIF-03 are generic over; DIF-04 (Phase 77) consumes the differentiable ops from Phase 76. All 4 requirements mapped, no orphans, no duplicates.

**Normal gates (this implementation milestone):** `cargo test`, `cargo clippy --all-targets --features linalg,parallel -- -D warnings`, `cargo fmt`. Numerical make-or-break gates warrant known-answer tests: dual arithmetic vs analytical derivatives (≤1e-10); elastic-distance gradient vs finite differences AND vs the existing soft_dtw gradient; f64 parity of elastic distance (1e-12) and FPCA scores; FPCA-score gradient vs finite differences; module doctest under `cargo test --doc`. Ships to crates.io on the `v0.39.0` tag (crate bump is a deferred operator ship step).

## Performance Metrics

**Velocity:**

- Total plans completed: 111+ (across v0.14.0–v0.38.0)
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
| 75–77 | v0.39.0 | 0/? (pending) |

**Recent Trend:**

- Last milestone: v0.38.0 VEESA phases 72–74 (3 plans) — audit PASSED 8/8, integration SOUND; crate code-complete (tag/publish deferred).
- Trend: v0.39.0 stays in implementation shape — real code, normal test/clippy/fmt gates. **Reuse-first pilot** on `metric/soft_dtw.rs` (existing hand-written gradient), `elastic_*`, `regression.rs` FPCA. Lower net-new-code risk than a from-scratch subsystem — the core algorithms already exist; the work is making a scoped subset generic over a new in-crate `Scalar`/`Dual` substrate, additively alongside the f64 paths. 3 phases driven by a hard dependency chain (substrate → two independent differentiable ops → gradient API + integration).

*Updated after each plan completion*
**Per-Plan Metrics:**

| Plan | Duration | Tasks | Files |
|------|----------|-------|-------|
| (none yet — v0.39.0) | — | — | — |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.

Relevant to current work (v0.39.0 DIFF):

- **AD mechanism = in-crate forward-mode dual numbers + a `Scalar` trait** — a scoped subset of ops is generic over the scalar so gradients **compose** through arbitrary op chains. NOT hand-written per-op gradients; NOT an external AD crate.
- **Forward-mode only (JVP / dual numbers)** — matching the ForwardDiff reference. Reverse-mode/VJP deferred (DIF-F1).
- **Scope = elastic distance + FPCA scores** (the two ops DIF-01 names). Broadening deferred (DIF-F2).
- **Strictly additive API** — existing f64 public signatures are untouched; generic versions live alongside (protects R + WASM bindings + 28 examples). Deprecate, never remove. Refactoring f64 hot-path signatures to be generic is deferred (DIF-F3).
- **No new crate dependency** — an in-crate `Dual` type is sufficient for forward-mode. Reuse-first pilot: `metric/soft_dtw.rs` (existing hand-written gradient to validate against), `elastic_*`, `regression.rs` FPCA.
- **Reference baseline:** Julia generic-programming idiom (ElasticFDA.jl + ForwardDiff).
- **Implementation milestone, ships on tag** — real `fdars-core/src/` changes; crate bump + `v0.39.0` tag + crates.io publish is a deferred operator ship step (release.yml publishes on the tag).
- **Phase numbering continues** — v0.38.0 ended at Phase 74 → v0.39.0 starts at Phase 75. No reset.
- **4 requirements → 3 phases** (fine granularity): Phase 75 DIF-01; Phase 76 DIF-02/DIF-03; Phase 77 DIF-04. All 4 mapped, no orphans, no duplicates. Hard dependency chain 75 → 76 → 77.

### Pending Todos

- **Migrate `fdars-r` R wrapper to use the `FdMatrix` API** (issue `fdars-j75`) — carried forward; the additive differentiable surface should be exposed to R/WASM bindings in a follow-up, not this milestone.

### Blockers/Concerns

- **No research/SUMMARY.md** — broad ecosystem research was intentionally skipped this milestone (the design is pinned from the GAP-08 backlog block + codebase; key architectural decisions are recorded in REQUIREMENTS.md "Milestone Design Decisions"). Non-blocking for the roadmap.
- **Numerical make-or-break gates warrant known-answer tests:** dual arithmetic vs analytical derivatives (≤1e-10); elastic-distance gradient vs central finite differences AND vs the existing hand-written `soft_dtw` gradient; f64 parity of elastic distance (1e-12) and FPCA scores; FPCA-score gradient vs central finite differences; module doctest under `cargo test --doc`. These are the pass/fail seams for Phases 75–77.
- **`soft_dtw` differentiability** — confirm at plan time that the soft-DTW recurrence is differentiable through its `min`/soft-min at `Dual` (soft-DTW is smooth by construction; the hard-min elastic path may need the soft variant for a valid gradient). The existing hand-written `soft_dtw` gradient is the validation oracle for DIF-02.
- **FPCA score generic surface** — the eigenbasis/mean are precomputed constants of the fit; DIF-03 differentiates the *score projection* of a curve onto a fixed basis w.r.t. the input-curve values. Confirm at plan time the projection can be made generic-over-`Scalar` without dragging the SVD/eigendecomposition (which stays f64) into the generic path.
- Historical build/CI hazards (MEMORY.md) apply this implementation milestone: run clippy with `--all-targets --features linalg,parallel -- -D warnings` (CI lints test/bench code); run `cargo fmt` per commit (`--no-verify` commits leave fmt drift); watch `/tmp` and `target/` disk pressure on full builds (`rm -rf target/debug/{incremental,examples}` to free space); prefer inline execution + `commit --no-verify` after out-of-band gates if executor subagents stall on long cargo builds; the per-phase impl-subagent pattern dodged executor stalls on v0.32.0 GAK.
- Pre-existing serde build break (NOT v0.39.0): `fdars-core/src/shapelet/classifier.rs` `ShapeletTransformClassifier` embeds non-serde `ClassifFit` → `cargo build --features serde` fails. Independent of DIFF; new types should be serde-clean (or serde-gated) if they derive serde.

## Deferred Items

Items acknowledged and deferred, most recent first:

| Category | Item | Status | Deferred At | Milestone |
|----------|------|--------|-------------|-----------|
| Differentiable-core | DIF-F1 (reverse-mode / VJP autodiff — needs a tape/graph engine); DIF-F2 (broaden the differentiable subset beyond elastic + FPCA — basis eval, inner products, SRSF/warping, other regressions); DIF-F3 (make existing f64 hot-path signatures themselves generic — breaking risk to R/WASM/examples) | Deferred | v0.39.0 | future milestone |
| VEESA | VEE-F1 (native random-forest / tree-ensemble predictor); VEE-F2 (plotting/rendering of principal directions + PFI) | Deferred | v0.38.0 | future milestone |
| Conformal-anomaly | ECA-F1 (full conditional / Mondrian conformal anomaly detection) | Deferred | v0.38.0 | future milestone |
| Wavelet-regression | WAV-F1 (binomial/logistic GLM-family); WAV-F2 (Symlets/Coiflets/biorthogonal + wavelet packets); WAV-F3 (2D/surface DWT) | Deferred | v0.37.0 | future milestone |
| API-breaking | APIB-01 — breaking removal of the 6 `#[deprecated]` forms from v0.30.0 | Deferred | v0.30.0 | future 1.0-readiness |

## Session Continuity

Last session: 2026-09-06T00:00:00.000Z
Stopped at: Phase 77 complete — all phases complete
Resume file: None

## Operator Next Steps

- Start the next milestone with /gsd-new-milestone
