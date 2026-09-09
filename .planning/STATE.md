---
gsd_state_version: 1.0
milestone: v0.43.0
milestone_name: Test Determinism & Release Hardening
status: planning
last_updated: "2026-09-09T05:21:12.264Z"
last_activity: 2026-09-09
progress:
  total_phases: 0
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
  percent: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-09-07)

**Core value:** A comprehensive, fast Rust functional-data-analysis library. This milestone clears the entire **API section** of the 1.0 gap checklist (`documentation/ROADMAP-TO-1.0.md`): seal `wire`, seal the config structs (with construction escape hatches), and finish the naming unification. Breaking but **API-shape-only** (names/visibility/exhaustiveness) — no numeric or behavioral change. Ships as **0.42.0** (NOT 1.0).
**Current focus:** Phase 86 — Surface Sealing (planning)

## Current Position

Phase: Not started (defining requirements)
Plan: —
Status: Defining requirements
Last activity: 2026-09-09 — Milestone v0.43.0 started

## Milestone Roadmap (v0.42.0)

Four phases, 9 requirements — the **second breaking milestone** (0.x → breaking permitted). API-shape-only: any function collapsed onto a `Dim`-dispatch signature routes to a byte-identical private `_impl` body (the v0.41.0 pattern; a code-review gate confirms no numeric drift). `fdars-core` only, no new crate dependency. All 28 examples + doctests updated (compile-time proof). Fine granularity; the risk spread (sealing vs. small renames vs. the large high-blast-radius suffix batch vs. non-code release prep) justifies distinct phases. Phase numbering continues from v0.41.0 (ended at 85) → **Phase 86**.

| Phase | Requirements | Notes |
|-------|--------------|-------|
| 86 — Surface Sealing | SEAL-01, SEAL-02, SEAL-03 | Seal `wire` `pub(crate)` (SEAL-01, independent + low-risk) and mark every public config struct `#[non_exhaustive]` (SEAL-02) **paired with** a `Default`/`..Default::default()` (and/or builder) construction escape hatch (SEAL-03). SEAL-02+SEAL-03 are **inseparable** — sealing a config without a non-literal construction path breaks external construction — so they land together; SEAL-01 rides along. First phase; independent of the naming work. |
| 87 — Targeted Renames | NAME-01, NAME-02, NAME-03, NAME-05 | The small, low-blast-radius renames: `Dim`-dispatch consolidation for `geometric_median` (NAME-01/AUD-19), the `hausdorff_*` family (NAME-02/AUD-20), and `functional_spatial_*` / `kernel_functional_spatial_*` (NAME-03/AUD-21), plus the `LpeerResult` → `LocalPeerResult` rename (NAME-05/AUD-23). Sequenced before the large suffix batch. Depends on Phase 86. |
| 88 — Large Suffix Batch | NAME-04 | The **highest-risk, highest-blast-radius** item (AUD-22): the large remaining lone-`_1d`/`_2d` suffix functions across the crate consolidated onto `Dim` dispatch, plus all 28 examples + docs updated. Isolated in its own phase, sequenced after the smaller renames. Depends on Phase 87. |
| 89 — Release Preparation & Verification | REL-01 | **Must land last.** Bump 0.41.0 → 0.42.0, breaking-framed CHANGELOG `[0.42.0]` (root + crate-shipped) calling out sealed `wire`/config + renames, docs refresh, whole-crate gates green + `--features serde` build + all 28 examples + doctests + `cargo package`, and check off the cleared API items in `documentation/ROADMAP-TO-1.0.md`. The `git tag v0.42.0` push → crates.io publish is the deferred operator step. Depends on Phases 86, 87, 88. |

**Execution order:** 86 → 87 → 88 → 89. 86 first (independent sealing). 87 then 88 sequence the naming smallest-first, isolating the high-blast-radius suffix batch. 89 last. All 9 requirements mapped, no orphans, no duplicates.

**Gates (this breaking, API-shape-only milestone):** `cargo fmt --check`, `cargo clippy --all-targets --features linalg,parallel -- -D warnings` (CI lints test/bench code — use `--all-targets`, not a plain `-p` lint), `cargo test`, plus a `--features serde` build guard. No new crate dependency; `fdars-core` only. All 28 examples + doctests must compile/pass — the compile-time proof the breaking changes are complete.

## Performance Metrics

**Velocity:**

- Total plans completed: 116+ (across v0.14.0–v0.41.0)
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
| 86–89 | v0.42.0 | 0/? (planned) |

**Recent Trend:**

- Last milestone: v0.41.0 (phases 81–85, 9 plans) — audit 9/9, INTEGRATED, release-ready.
- Trend: v0.42.0 is the **second breaking** milestone — API-shape-only changes (no numeric/behavioral change), higher blast-radius risk concentrated in Phase 88 (the large suffix batch touches all 28 examples + docs). The byte-identical `_impl` + code-review pattern from v0.41.0 is reused throughout.

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions relevant to current work (v0.42.0):

- **Second breaking milestone under 0.x** — clears the entire API section of `documentation/ROADMAP-TO-1.0.md`. Breaking limited to API shape (names, visibility, exhaustiveness); numeric outputs unchanged (byte-identical `_impl` bodies, code-review confirmed).
- **Ships as 0.42.0, NOT 1.0** — quality (golden-flake), algorithm (SDTW-O1), diff-core (DIF-F1/F2/F3), and ecosystem (`fdars-j75`) checklist items remain; the 1.0-CUT is deliberately deferred.
- **`wire` is sealed, not wired up** — `pub mod wire` → `pub(crate)`; no current JS/R consumer, so sealing is the low-risk choice (re-expose deliberately later). AUD-13 (`#[non_exhaustive]` on wire structs) becomes moot once sealed.
- **SEAL-02 + SEAL-03 are inseparable** — a config struct cannot be sealed `#[non_exhaustive]` without a non-literal construction path (`Default`/builder) or external construction breaks; they land in the same phase (86). SEAL-01 (wire, independent + low-risk) rides along.
- **Config-struct target set enumerated during planning** — the Phase 81 audit estimated ~22; the current tree has ~66 `*Config` definitions with ~15 already sealed. The exact set (public, not `pub(crate)`, not yet sealed) is enumerated in Phase 86 planning.
- **NAME-04 (the large suffix batch) gets its own phase (88)** — highest-risk, highest-blast-radius (many functions + all 28 examples + docs); isolated from the smaller renames and sequenced after them.
- **Small renames share one phase (87)** — NAME-01/02/03 `Dim`-dispatch consolidations + NAME-05 (`LpeerResult` → `LocalPeerResult`) are low-blast-radius.
- **REL-01 lands last (89)** — prepares + verifies release-readiness; operator does the `v0.42.0` tag/publish. Assumes 0.41.0 ships first.
- **No new crate dependency; `fdars-core` only** — the config escape hatches use `Default`/hand-written builders, not a derive-builder crate.
- **Phase numbering continues** — v0.41.0 ended at Phase 85 → v0.42.0 starts at Phase 86. No reset.
- **9 requirements → 4 phases:** 86 SEAL-01/02/03; 87 NAME-01/02/03/05; 88 NAME-04; 89 REL-01. All mapped, no orphans, no duplicates.

### Pending Todos

- **Migrate `fdars-r` R wrapper to use the `FdMatrix` API** (issue `fdars-j75`) — carried forward; separate package, out of `fdars-core` scope this milestone.
- **Operator ship step for v0.41.0** — `git tag v0.41.0` → push → crates.io publish still pending; 0.42.0 assumes 0.41.0 ships first.

### Blockers/Concerns

- **No research/SUMMARY.md** — intentional: this is an internal API-shape milestone, not an ecosystem-parity milestone. Non-blocking for the roadmap.
- **Breaking-change blast radius** — Phase 88 (NAME-04) touches all 28 examples + docs; the sealing + renaming break some external usage by design. Scope guard: API shape only (no numeric/behavioral change); every consolidated signature routes to a byte-identical private `_impl`, confirmed by a code-review gate.
- Historical build/CI hazards (MEMORY.md) apply: run clippy with `--all-targets --features linalg,parallel -- -D warnings` (CI lints test/bench code); run `cargo fmt` per commit (`--no-verify` commits leave fmt drift); keep the `--features serde` build green (repaired in v0.40.0 — do not regress it); watch `/tmp` and `target/` disk pressure on full builds (`rm -rf target/debug/{incremental,examples}` to free space; doctests link in a small `/tmp` tmpfs); prefer inline execution + `commit --no-verify` after out-of-band gates if executor subagents stall on long cargo builds.
- **Golden-test flake** — the pre-existing `co_cluster`/`svd_sign` golden flake (fails under full parallel `cargo test`, passes per-binary in isolation) is a known env flake, NOT a v0.42.0 regression; verify the REL-01 `cargo test` gate via isolated binaries if it surfaces. It remains on the 1.0 checklist (out of scope this milestone).

## Deferred Items

Items acknowledged and deferred, most recent first:

| Category | Item | Status | Deferred At | Milestone |
|----------|------|--------|-------------|-----------|
| 1.0-cut | 1.0-CUT — bump to 1.0.0 and declare the public API stable, once every ROADMAP-TO-1.0.md item clears | Deferred | v0.41.0 | future (deliberate 1.0 cut) |
| Quality | golden-test flake (co_cluster / svd_sign) — make deterministic under full parallel `cargo test` | Deferred | v0.41.0 | future milestone (quality) |
| Algorithm | SDTW-O1 — replace the `soft_dtw_barycenter` MM-step descent with a proper global optimizer (L-BFGS / multi-restart) | Deferred | v0.40.0 | future milestone |
| Differentiable-core | DIF-F1 (reverse-mode/VJP), DIF-F2 (broaden differentiable subset), DIF-F3 (generic f64 hot-path signatures) | Deferred | v0.39.0 | future milestone |
| fdars-r | `fdars-r` FdMatrix migration (issue `fdars-j75`) — migrate the external R wrapper to the `FdMatrix` API; separate package, out of `fdars-core` scope | Deferred | v0.41.0 | future milestone |

## Session Continuity

Last session: 2026-09-07T21:00:00.000Z
Stopped at: Phase 89 complete — all phases complete
Resume file: None

## Operator Next Steps

- Start the next milestone with /gsd-new-milestone
