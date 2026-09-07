---
gsd_state_version: 1.0
milestone: v0.41.0
milestone_name: 1.0 API Stabilization Pass
status: Awaiting next milestone
stopped_at: ROADMAP.md + STATE.md created for v0.41.0 (5 phases 81–85); REQUIREMENTS.md traceability filled
last_updated: "2026-09-07T13:52:39.047Z"
last_activity: 2026-09-07
last_activity_desc: Milestone v0.41.0 completed and archived
state_head: 8d059338b3f1172dc814123bffc505604486b285
progress:
  total_phases: 5
  completed_phases: 3
  total_plans: 9
  completed_plans: 9
current_phase: 85
current_phase_name: Release Preparation & Verification
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-09-06)

**Core value:** A comprehensive, fast Rust functional-data-analysis library. This milestone is a **1.0-readiness / API-stabilization** pass: audit the whole public surface, then land the breaking cleanups now while still in 0.x — producing a settled API + the stability deliverables (semver policy, MSRV, 1.0 gap checklist) that will govern a deliberate future 1.0 cut. Ships as **0.41.0** (NOT 1.0).
**Current focus:** Phase 85 — Release Preparation & Verification

## Current Position

Phase: Milestone v0.41.0 complete
Plan: —
Status: Awaiting next milestone
Last activity: 2026-09-07 — Milestone v0.41.0 completed and archived

## Milestone Roadmap (v0.41.0)

Five phases, 9 requirements — the **first breaking milestone** after a long additive-only run (legitimate under 0.x). Real `fdars-core/src/` changes limited to API shape (names, visibility, exhaustiveness) — numeric outputs unchanged. No new crate dependency. Workspace is `fdars-core` only (external `fdars-r` migration is a separate todo, `fdars-j75`). All 28 examples + doctests updated (compile-time proof). Fine granularity; the risk spread (mechanical vs. large/high-risk naming vs. non-code docs) justifies distinct phases. Phase numbering continues from v0.40.0 (ended at 80) → **Phase 81**.

| Phase | Requirements | Notes |
|-------|--------------|-------|
| 81 — API Audit & Deprecated-Form Removal | AUDIT-01, API-01 | **Opens the milestone, lands first.** AUDIT-01 produces the ranked breaking-change inventory across all four scopes (deprecated-form removal, accidental `pub` exposure, `#[non_exhaustive]` gaps, naming) → **presented for user approval**; Phases 82/83 change sets are drawn from the approved list. API-01 (remove the 6 deprecated forms → `Dim`/`_seeded` replacements; migrate callers/tests/doctests/example 21) is already fully specified, does NOT depend on the findings, and is low-risk/mechanical — rides in this phase. |
| 82 — Public-Surface Sealing & Non-Exhaustive Coverage | API-02, API-03 | Both drawn from the approved AUDIT-01 inventory. API-02: seal accidental `pub` exposure (`pub` → `pub(crate)`, removed re-exports, hidden leaked helper types). API-03: correct `#[non_exhaustive]` on public enums/result structs. Visibility/attribute-only — no behavior change. Depends on Phase 81. |
| 83 — Naming Unification | API-04 | The **largest, highest-risk** item (deferred as breaking back in v0.30.0). Unify `_1d`/`_2d`/`_nd` suffix sprawl + config/result naming into consistent dispatchers; touches many fns, all 28 examples, and docs. Exact scope set by the approved AUDIT-01 list (user may approve a reduced scope). Own phase. Depends on Phase 82. |
| 84 — Stability Deliverables | STAB-01, STAB-02, STAB-03 | Non-code deliverables (share a phase): semver/API-stability policy in `documentation/` (STAB-01), MSRV finalization + pin (1.81 crate / 1.84 `linalg`), Cargo.toml↔docs consistency (STAB-02), 1.0 gap checklist scoping the next milestone (STAB-03). Depends on Phase 81 (inventory informs the checklist); can run alongside 82/83. |
| 85 — Release Preparation & Verification | REL-01 | **Must land last.** Bump 0.40.0 → 0.41.0, CHANGELOG `[0.41.0]` with breaking changes explicitly called out, docs refresh, whole-crate gates green + `--features serde` build; all 28 examples + doctests pass. The `git tag v0.41.0` push → crates.io publish is the final operator-driven step (this phase prepares + verifies release-readiness only). Depends on Phases 82, 83, 84. |

**Execution order:** 81 → 82 → 83 → 84 → 85. 81 first (audit + approval gate). 82/83 draw from the approved inventory; 84 can run alongside once approved. 85 last. All 9 requirements mapped, no orphans, no duplicates.

**Gates (this breaking implementation milestone):** `cargo fmt --check`, `cargo clippy --all-targets --features linalg,parallel -- -D warnings` (CI lints test/bench code — use `--all-targets`, not a plain `-p` lint), `cargo test`, plus a `--features serde` build guard. No new crate dependency. All 28 examples + doctests must compile/pass — the compile-time proof the breaking changes are complete.

## Performance Metrics

**Velocity:**

- Total plans completed: 116+ (across v0.14.0–v0.40.0)
- Average duration: — min
- Total execution time: — hours

**By Phase (prior milestones):**

| Phase | Milestone | Plans |
|-------|-----------|-------|
| 01–45 | v0.14.0–v0.29.0 | 84 |
| 46–51 | v0.30.0 | 23 |
| 52–77 | v0.31.0–v0.39.0 | ~24 |
| 78–80 | v0.40.0 | 5 |
| 81–85 | v0.41.0 | 0/7 (planned) |

**Recent Trend:**

- Last milestone: v0.40.0 (phases 78–80, 5 plans) — audit 5/5, release-ready.
- Trend: v0.41.0 is the **first breaking** milestone — API-shape-only changes (no numeric/behavioral change), but higher blast-radius risk (Phase 83 naming touches all 28 examples). Audit-gated: nothing breaking executes before the user approves the inventory.

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table. Relevant to current work (v0.41.0):

- **First breaking milestone under 0.x** — departs from the long additive-only run; legitimate because the crate is still 0.x. Breaking limited to API shape (names, visibility, exhaustiveness); numeric outputs unchanged.
- **Ships as 0.41.0, NOT 1.0** — settle the API under 0.x first; the actual 1.0 cut is a separate, governed commitment (per STAB-01/STAB-03).
- **Audit-gated execution** — AUDIT-01 (Phase 81) produces a ranked inventory across four scopes, **user-approved before any execution phase runs**. API-02/03/04 change sets are drawn from the approved list; the user may approve a reduced scope (esp. naming).
- **API-01 rides in the audit phase** — the 6 deprecated-form removals are already fully specified and independent of the audit findings; low-risk/mechanical.
- **API-04 (naming) gets its own phase** — the largest, highest-risk item, deferred as breaking back in v0.30.0; touches many fns + all 28 examples + docs.
- **STAB-01/02/03 share one non-code phase** (84) — docs/policy deliverables, no `fdars-core/src/` code.
- **REL-01 lands last** (85) — prepares + verifies release-readiness; operator does the `v0.41.0` tag/publish.
- **No new crate dependency; workspace is `fdars-core` only** — carried conventions; external `fdars-r` migration is separate (`fdars-j75`).
- **Phase numbering continues** — v0.40.0 ended at Phase 80 → v0.41.0 starts at Phase 81. No reset.
- **9 requirements → 5 phases:** 81 AUDIT-01/API-01; 82 API-02/03; 83 API-04; 84 STAB-01/02/03; 85 REL-01. All mapped, no orphans, no duplicates.

### Pending Todos

- **Migrate `fdars-r` R wrapper to use the `FdMatrix` API** (issue `fdars-j75`) — carried forward; separate package, out of `fdars-core` scope this milestone.

### Blockers/Concerns

- **No research/SUMMARY.md** — intentional: this is an internal API-audit/stabilization pass, not an ecosystem-parity milestone. Non-blocking for the roadmap.
- **Breaking-change blast radius** — Phase 83 (naming) touches all 28 examples + docs; Phases 81–83 all break some external usage by design. Scope guard: nothing breaking executes before the AUDIT-01 inventory is user-approved; API shape only (no numeric/behavioral change).
- Historical build/CI hazards (MEMORY.md) apply: run clippy with `--all-targets --features linalg,parallel -- -D warnings` (CI lints test/bench code); run `cargo fmt` per commit (`--no-verify` commits leave fmt drift); keep the `--features serde` build green (repaired in v0.40.0 — do not regress it); watch `/tmp` and `target/` disk pressure on full builds (`rm -rf target/debug/{incremental,examples}` to free space; doctests link in a small `/tmp` tmpfs); prefer inline execution + `commit --no-verify` after out-of-band gates if executor subagents stall on long cargo builds.

## Deferred Items

Items acknowledged and deferred, most recent first:

| Category | Item | Status | Deferred At | Milestone |
|----------|------|--------|-------------|-----------|
| 1.0-cut | 1.0-CUT — bump to 1.0.0 and declare the public API stable, once the STAB-03 gap checklist is cleared | Deferred | v0.41.0 | future (deliberate 1.0 cut) |
| fdars-r | `fdars-r` FdMatrix migration (issue `fdars-j75`) — migrate the external R wrapper to the `FdMatrix` API; separate package, out of `fdars-core` scope | Deferred | v0.41.0 | future milestone |
| Soft-DTW-optimizer | SDTW-O1 — replace the `soft_dtw_barycenter` inverse-curvature / soft-DBA MM step with a proper global optimizer (L-BFGS / multi-restart) | Deferred | v0.40.0 | future milestone |
| Differentiable-core | DIF-F1 (reverse-mode/VJP), DIF-F2 (broaden differentiable subset), DIF-F3 (generic f64 hot-path signatures) | Deferred | v0.39.0 | future milestone |

## Session Continuity

Last session: 2026-09-07T08:20:00.000Z
Stopped at: ROADMAP.md + STATE.md created for v0.41.0 (5 phases 81–85); REQUIREMENTS.md traceability filled
Resume file: None

## Operator Next Steps

- Start the next milestone with /gsd-new-milestone
