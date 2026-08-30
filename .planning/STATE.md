---
gsd_state_version: 1.0
milestone: v0.30.0
milestone_name: Performance & Consolidation Pass
status: executing
last_updated: "2026-08-30T22:30:00.000Z"
last_activity: 2026-08-30
progress:
  total_phases: 6
  completed_phases: 1
  total_plans: 5
  completed_plans: 5
  percent: 17
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-08-30)

**Core value:** A comprehensive, fast Rust functional-data-analysis library — with both parity backlogs exhausted, v0.30.0 pivots from breadth to depth: profile the whole crate, then land behavior-preserving improvements across hot-path performance, code duplication, additive API consistency, and benchmark coverage.
**Current focus:** Phase 47 — Hot-Path & Allocation Performance (consumes PROF-01's ranked targets)

## Current Position

Phase: 47 of 51 (Hot-Path & Allocation Performance) — Phase 46 complete
Plan: — (not yet planned)
Status: Phase 46 complete & verified (passed 4/4); ready to plan Phase 47
Last activity: 2026-08-30 — Phase 46 executed (5 plans) and verified passed; three ranked inventories produced (PROF-00/01/02/03)

Progress: [█░░░░░░░░░] 17%

### Phase 46 outcomes (feed downstream phases)
- **PROF-01** (→ Phase 47/51): top hot paths `face_covariance` (984ms), `fem_smooth` (452ms), `frechet_anova` (133ms); top allocation `fts::dpca` (42MB churn). All 9 subsystems ranked with file:line anchors.
- **PROF-02** (→ Phase 49): top dedup = χ²/F survival (2 independent gamma kernels, `inference/dist.rs:99` vs `spm/chi_squared.rs:164`); then permutation loops, seeded-RNG, SVD sign-fix.
- **PROF-03** (→ Phase 50): 4 Config structs missing `Default`; non-seedable `fanova`; breaking items deferred to APIB-01.
- Measure-only confirmed: zero `src/` edits, no new dependency, full suite green, clippy `--all-targets` clean.

## Milestone Roadmap (v0.30.0)

Six phases, 13 requirements — the first internally-driven implementation milestone. **Measure-first:** Phase 46 (PROF) is a hard prerequisite whose three ranked inventories drive the implementation phases. Behavior-preserving (numeric outputs unchanged or provably-equivalent within tolerance, proven by tests + before/after criterion benchmarks). Additive/non-breaking API only (deprecate, never remove — protects R/WASM bindings + 28 examples). No new crate dependency (profiling uses existing dev-deps: criterion, feature-gated `dhat-heap`). Phase numbering continues from v0.29.0 (43/44/45) → Phase 46+.

| Phase | Requirements | Notes |
|-------|--------------|-------|
| 46 — Whole-Crate Profiling & Measurement | PROF-01, PROF-02, PROF-03 | Ranked hot-path target list (N×M-scaled, v0.19–v0.29 subsystems prioritized) + duplication inventory (file:line anchors) + API-inconsistency inventory (canonical form per item). **Hard prerequisite** for 47–50. Audit-only: zero behavior-changing src edits. |
| 47 — Hot-Path & Allocation Performance | PERF-01, PERF-02 | Optimize top-ranked compute-bound paths + allocation hotspots (FdMatrix↔DMatrix copies, per-iteration allocs). Before/after criterion + dhat-heap allocation profile + equivalence tests. Consumes PROF-01. |
| 48 — Parallelism-Gap Closure | PERF-03 | Close feature-gated rayon gaps in newer subsystems via `parallel.rs` macros; equivalence-tested vs sequential (ON/OFF); payback-threshold N guard where small-input regression possible. Depends on 46 (+ 47 shares harness). |
| 49 — Code Consolidation / Dedup | CONS-01, CONS-02 | Factor duplicated numerical machinery (FPCA scoring, Cholesky/ridge, Simpson/quadrature, χ²/F survival, SVD sign-fix) + statistical-test scaffolding (permutation loops, seeded-RNG) into shared `pub(crate)` helpers; migrate all call sites; behavior unchanged. Consumes PROF-02. |
| 50 — Additive API-Surface Consolidation | API-01, API-02, API-03 | Unified alternatives for inconsistent config/result patterns + redundant functions; `#[deprecated]` old forms (never remove); tighten crate-root re-exports; 28 examples + R/WASM bindings pass with deprecation warnings only. Consumes PROF-03. |
| 51 — Benchmark Coverage & Regression Guards | BENCH-01, BENCH-02 | New `[[bench]]` entries for unbenchmarked modules (`fts`/`frechet`/`boosting_regression`/`coclustering`/`fem_smoothing`/`density_fda`/`inference`/`fpca_variants`/`face`); commit PERF-proof benches as regression guards with documented before/after. BENCH-02 depends on 47/48. |

**Execution order (dependency-driven):** 46 → 47 → 48 → 49 → 50 → 51. Phase 46 gates all implementation phases (PROF-01→PERF, PROF-02→CONS, PROF-03→API). 47 precedes 48 (both PERF, shared harness). 49 and 50 each depend only on 46 and are otherwise independent of each other. 51 is last so BENCH-02's regression guards follow the PERF phases.

## Performance Metrics

**Velocity:**

- Total plans completed: 84 (across v0.14.0–v0.29.0)
- Average duration: — min
- Total execution time: — hours

**By Phase (prior milestones):**

| Phase | Milestone | Plans |
|-------|-----------|-------|
| 01–09 | v0.14.0 | 21 |
| 10–42 | v0.15.0–v0.28.0 | 52 |
| 43–45 | v0.29.0 | 11 |
| 46–51 | v0.30.0 | TBD |

**Recent Trend:**

- Last milestone: v0.29.0 phases 43–45 (11 plans) — milestone audit PASSED 12/12, archived. Exhausted the R-parity backlog.
- Trend: v0.30.0 is a **different shape** — measure-first performance/consolidation pass (not a feature-add milestone). Phase 46 profiling gates the rest; plan counts TBD until the ranked inventories exist.

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.

Relevant to current work (v0.30.0):

- **v0.30.0 pivots from external gap-audit to an internal performance/consolidation pass** — both parity backlogs (scikit-fda + R) are exhausted; profiling evidence replaces an external yardstick for prioritization.
- **Measure-first:** Phase 46 whole-crate profiling produces three ranked inventories (hot-path targets, duplication, API inconsistencies) before any implementation. No PERF/CONS/API phase is plannable until 46 lands its evidence.
- **Behavior-preserving:** numeric outputs unchanged or provably-equivalent within documented tolerance; every change proven by existing tests + before/after criterion benchmarks (equivalence tests where a split/optimized path exists).
- **Additive/non-breaking API only:** API consolidation adds a unified alternative + `#[deprecated]` on the old form — never removes an existing public signature (protects R/WASM bindings, 28 examples, external callers). The breaking removal sweep is deferred (APIB-01, future 1.0-readiness milestone).
- **No new crate dependency** carries forward — profiling uses existing dev-deps only (criterion, feature-gated `dhat-heap`).
- **Phase numbering continues** from v0.29.0 (ended at Phase 45) → v0.30.0 starts at Phase 46. No reset.
- **13 requirements → 6 phases** (fine granularity): PROF (46), PERF hot-path+alloc (47), PERF parallelism (48), CONS (49), API (50), BENCH (51). All 13 mapped, no orphans.

Conventions carried from prior milestones (relevant to implementation):

- Column-major `FdMatrix` (`src/matrix.rs`), `Result<T, FdarError>` on all public fns, feature-gated rayon (`iter_maybe_parallel!` etc.), per-thread RNG seeding `StdRng::seed_from_u64(seed + k)` — Phase 48 parallelism + Phase 49 CONS-02 seeded-RNG consolidation must preserve determinism.
- Full clippy gate: `cargo clippy --all-targets --features linalg,parallel -- -D warnings` (CI lints test/bench code; a plain `-p ... -D warnings` false-greens — MEMORY.md pointer). Relevant to Phase 51 bench code.
- `TMPDIR=/home/simonm/.cache/fdars-bench-tmp` required for build/doctest/bench linking; /tmp tmpfs exhaustion causes bogus "No space left" (MEMORY.md pointer) — this milestone runs criterion benches heavily.
- `target/` grows to 100+GB and fills /home; `rm -rf target/debug/{incremental,examples}` frees ~108G if example LINK fails (MEMORY.md pointer) — relevant since Phase 50 must build the 28 examples.

### Pending Todos

- **Migrate `fdars-r` R wrapper to use the `FdMatrix` API** — preserved from the removed `.beads` issue tracker (issue `fdars-j75`). Relevant to Phase 50: the R-binding call sites are one of the "must still pass" surfaces (API-03).

### Blockers/Concerns

- Local main is ahead of origin/HEAD → harness worktrees may fork the wrong base; GSD phases fall back to sequential no-worktree dispatch (MEMORY.md pointer). Prior milestones (v0.26.0–v0.29.0) executed phases inline (not via gsd-executor subagents) for this reason. v0.30.0's phases are largely sequential by dependency anyway (46 gates all).
- Executor subagents trip the 600s stream watchdog on long fdars cargo builds; `--no-verify` commits leave fmt drift → run `cargo fmt` per commit + a whole-crate sweep at milestone end (MEMORY.md pointers). **Especially relevant this milestone** — criterion bench compiles are long.
- /tmp tmpfs exhaustion + `target/` filling /home (see Decisions above) — both bite harder in a bench-heavy milestone; free space before running full `cargo bench`.
- **Governor/CPU-pinning caveat (from v0.14.0 audit):** multi-thread criterion cells were flagged LOW-CONFIDENCE with the governor unpinned. Phase 46/47/48 benchmark evidence should note environment (governor, RAYON_NUM_THREADS) to keep before/after comparisons honest.
- **Audit-milestone-no-git-tag pointer:** Phase 46 (PROF) is profiling-only. But v0.30.0 overall ships real code — the crate release (version bump + publish + tag) remains a deferred operator step (REL-01), not part of these phases.

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 260822-pvk | Update README: link Python package pyfda, note R package outdated | 2026-08-22 | 0c634e55 | [260822-pvk-update-readme-link-python-package-pyfda-](./quick/260822-pvk-update-readme-link-python-package-pyfda-/) |
| 260823-bds | Remove unused .beads issue tracker + stale AGENTS.md | 2026-08-23 | 198b5566 | [260823-bds-remove-beads-issue-tracker](./quick/260823-bds-remove-beads-issue-tracker/) |

## Deferred Items

Both parity backlogs (scikit-fda + R) are **exhausted** — no further ranked external-parity items remain. Deferred beyond v0.30.0: **APIB-01** (breaking removal of the functions/configs deprecated this milestone — a future 1.0-readiness / breaking release) and **REL-01** (crate version bump + `cargo publish` + tag folding in v0.29.0 + v0.30.0 — deferred operator ship-time step).

Advisory tech-debt carried forward (not necessarily v0.30.0 work, but some may surface in PROF-02/PROF-03 inventories): weakened MEWMA test assertion; `fix_svd_signs` NaN no-op; over-broad Phase 11 test name; prior VALIDATION.md files remain `draft` (Nyquist TODO); intentional R-baseline divergences documented in rustdoc across prior milestones.

**Release status:** `fdars-core` **0.28.0 published to crates.io** (tag `v0.28.0`, 2026-08-23). v0.29.0's additive code (REG-06/REP-02/CLUS-02) is unreleased; v0.30.0 will fold it into the next crate release (REL-01, operator-driven).

## Session Continuity

Last session: 2026-08-30T20:15:00.000Z
Stopped at: Created v0.30.0 roadmap (ROADMAP.md Phases 46–51; REQUIREMENTS.md traceability filled, 13/13 mapped)
Resume file: None

## Operator Next Steps

- Review the roadmap draft (ROADMAP.md, Phases 46–51).
- Plan the first phase: `/gsd-plan-phase 46` (Whole-Crate Profiling & Measurement — gates the whole milestone).
