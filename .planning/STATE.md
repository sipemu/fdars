---
gsd_state_version: 1.0
milestone: v0.43.0
milestone_name: Test Determinism & Release Hardening
current_phase: 93
current_phase_name: Release Preparation & Readiness Verification
status: planning
stopped_at: Phase 92 complete, ready to plan Phase 93
last_updated: "2026-09-09T12:13:44.292Z"
last_activity: 2026-09-09
last_activity_desc: Phase 92 complete, transitioned to Phase 93
state_head: b3c6641a620989413f58db59c0c6f2d34eca3c78
progress:
  total_phases: 4
  completed_phases: 3
  total_plans: 3
  completed_plans: 3
  percent: 75
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-09-09)

**Core value:** A comprehensive, fast Rust functional-data-analysis library. This milestone clears the **Quality** blocker on the 1.0 gap checklist (`documentation/ROADMAP-TO-1.0.md`) — make `cargo test` reliably deterministic under full parallel runs, harden CI against determinism regressions, and prepare a release that supersets the unpublished 0.41.0/0.42.0. Implementation milestone, additive/non-breaking, mostly `tests/` + CI (possibly minor `src/`).
**Current focus:** Phase 90 — Golden-Flake Root-Cause & Deterministic Fix

## Current Position

Phase: 93 — Release Preparation & Readiness Verification
Plan: Not started
Status: Ready to plan
Last activity: 2026-09-09 — Phase 92 complete, transitioned to Phase 93

Progress: [░░░░░░░░░░] 0%

## Milestone Roadmap (v0.43.0)

Four phases, 7 requirements — an **investigate-first** quality + release-hardening milestone. Additive/non-breaking (protects R + WASM + 28 examples). No new crate dependency UNLESS the flake fix genuinely needs one (dev-only; `serial_test`/nextest — decided in Phase 90). Fine granularity; the four tightly-coupled requirement pairs/singletons (diagnose→fix, audit→fix, guardrail, prep→verify) form four coherent phases. Phase numbering continues from v0.42.0 (ended at 89) → **Phase 90**.

| Phase | Requirements | Notes |
|-------|--------------|-------|
| 90 — Golden-Flake Root-Cause & Deterministic Fix | FLAKE-01, FLAKE-02 | **Investigate-first.** FLAKE-01 (evidence-backed diagnosis of why the three golden tests pass per-binary but flake under full parallel `cargo test`) **gates** FLAKE-02 (the deterministic fix — tolerance vs. serialization chosen from the evidence). They share a phase because the fix design is wholly determined by the diagnosis. First phase; no dependency. |
| 91 — Suite-Wide Robustness Sweep | ROBUST-01, ROBUST-02 | Audit the whole suite for other fragile bit-identity / nondeterministic / env-BLAS-disk-dependent assertions (ROBUST-01) and fix or justify each (ROBUST-02). Reuses the Phase 90 diagnosis technique; sequenced after the golden-flake fix. Depends on 90. |
| 92 — CI Determinism Guardrail | CI-01 | Add a CI gate exercising the full parallel `cargo test` path (cross-binary interference) and/or a nextest serialization group, so a determinism regression fails CI instead of silently returning. Lands after the fixes so the gate reflects a green baseline. Depends on 91. |
| 93 — Release Preparation & Readiness Verification | REL-01, REL-02 | **Must land last.** Bump 0.42.0 → 0.43.0, CHANGELOG `[0.43.0]` + docs, check off the **Quality** item in `ROADMAP-TO-1.0.md` (REL-01); verify all gates green — fmt/clippy `--all-targets`/full `cargo test`/`--features serde` build/28 examples+doctests/`cargo package` — with the full-suite gate as the proof the flake is fixed (REL-02). The `git tag v0.43.0` → crates.io publish is the deferred operator step. Depends on 90, 91, 92. |

**Execution order:** 90 → 91 → 92 → 93. Investigate-first (diagnose before fix), reuse the technique in the sweep, lock the green baseline with CI, then prep + verify release last. All 7 requirements mapped, no orphans, no duplicates.

**Gates (this additive, non-breaking milestone):** `cargo fmt --check`, `cargo clippy --all-targets --features linalg,parallel -- -D warnings` (CI lints test/bench code — use `--all-targets`), full `cargo test` (the determinism proof), a `--features serde` build guard, all 28 examples + doctests, and `cargo package`. No new crate dependency unless the flake fix genuinely needs one (dev-only).

## Performance Metrics

**Velocity:**

- Total plans completed: 125+ (across v0.14.0–v0.42.0)
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
| 90–93 | v0.43.0 | 0/? (planned) |

**Recent Trend:**

- Last milestone: v0.42.0 (phases 86–89, 4 plans) — audit 9/9, release-ready (operator tag/publish pending).
- Trend: v0.43.0 is a quality + release-hardening milestone — investigate-first, additive/non-breaking. The long-standing `co_cluster`/`svd_sign` golden flake (logged on the 1.0 checklist since v0.41.0) is finally root-caused and fixed here.

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions relevant to current work (v0.43.0):

- **Investigate-first** — FLAKE-01's evidence-backed diagnosis gates FLAKE-02's fix approach; they share Phase 90 because the tolerance-vs-serialization choice depends entirely on the root cause.
- **REL-01/REL-02 land last (Phase 93)** — REL-02's full-suite `cargo test` gate is the proof the flake is fixed; release prep only happens once all test/CI work is green.
- **ROBUST sweep after the golden-flake fix (Phase 91)** — reuses the same diagnosis technique; CI-01 guardrail (Phase 92) lands after the fixes so the gate reflects a green baseline.
- **No new crate dependency unless the fix genuinely needs one** — `serial_test`/nextest serialization decided during FLAKE-01; if added it must be dev-only and justified.
- **Additive/non-breaking** — mostly `tests/` + CI config, possibly minor `src/` determinism changes; protects R + WASM bindings + 28 examples.
- **0.43.0 supersets the unpublished 0.41.0/0.42.0** — registry is still at 0.40.0; one publish catches it up. The `git tag v0.43.0` push → crates.io publish is the deferred OPERATOR step (GSD `git.create_tag` is off because `release.yml` couples tag-push to publish) — never tagged/published inside a phase.
- **Phase numbering continues** — v0.42.0 ended at Phase 89 → v0.43.0 starts at Phase 90. No reset.
- **7 requirements → 4 phases:** 90 FLAKE-01/02; 91 ROBUST-01/02; 92 CI-01; 93 REL-01/02. All mapped, no orphans, no duplicates.

### Pending Todos

- **Operator ship steps still pending** — `git tag v0.41.0` and `git tag v0.42.0` → push → crates.io publish not yet performed; 0.43.0's single publish supersets both (registry at 0.40.0).
- **Migrate `fdars-r` R wrapper to the `FdMatrix` API** (issue `fdars-j75`) — carried forward; separate package, out of `fdars-core` scope.

### Blockers/Concerns

- **No research/SUMMARY.md** — intentional: internal quality + release-hardening milestone, not an ecosystem-parity audit. Non-blocking for the roadmap.
- **The golden-test flake is the central target** — `co_cluster`/`svd_sign` (equivalence_phase48/49) fails ONLY under full parallel `cargo test`, passes per-binary in isolation (MEMORY.md). Treat as an environment/cross-binary-interference flake, not a numeric regression. Verify fixes via repeated full runs AND isolated binaries.
- Historical build/CI hazards (MEMORY.md) apply: run clippy `--all-targets --features linalg,parallel -- -D warnings` (CI lints test/bench code); run `cargo fmt` per commit (`--no-verify` commits leave fmt drift → CI fmt-check fails); keep the `--features serde` build green (repaired v0.40.0 — do not regress); watch `/tmp` and `target/` disk pressure on full builds (`rm -rf target/debug/{incremental,examples}` to free space; doctests link in a small `/tmp` tmpfs); the pre-commit hook runs the full cargo gate and times out — prefer inline execution + `commit --no-verify` after out-of-band gates; full combined gate as one background bash gets killed mid-run — run gates per-gate FOREGROUND with 600s timeout.

## Deferred Items

Items acknowledged and deferred, most recent first:

| Category | Item | Status | Deferred At | Milestone |
|----------|------|--------|-------------|-----------|
| 1.0-cut | 1.0-CUT — bump to 1.0.0 and declare the public API stable, once every ROADMAP-TO-1.0.md item clears | Deferred | v0.41.0 | future (deliberate 1.0 cut) |
| Algorithm | SDTW-O1 — replace the `soft_dtw_barycenter` MM-step descent with a proper global optimizer (L-BFGS / multi-restart) | Deferred | v0.40.0 | future milestone |
| Differentiable-core | DIF-F1 (reverse-mode/VJP), DIF-F2 (broaden differentiable subset), DIF-F3 (generic f64 hot-path signatures) | Deferred | v0.39.0 | future milestone |
| fdars-r | `fdars-r` FdMatrix migration (issue `fdars-j75`) — migrate the external R wrapper to the `FdMatrix` API; separate package, out of `fdars-core` scope | Deferred | v0.41.0 | future milestone |

## Session Continuity

Last session: 2026-09-09T00:00:00.000Z
Stopped at: Phase 92 complete, ready to plan Phase 93
Resume file: None

## Operator Next Steps

- Plan the first phase with /gsd-plan-phase 90
- Deferred ship steps: `git tag v0.41.0` / `v0.42.0` / (eventually) `v0.43.0` → push → crates.io publish, on a disk-healthy machine after a clean full `cargo test`
