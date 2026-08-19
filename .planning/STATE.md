---
gsd_state_version: 1.0
milestone: v0.23.0
milestone_name: Depth, Outliers & Interval Inference
status: in_progress
last_updated: "2026-08-19T21:00:00.000Z"
last_activity: 2026-08-19
progress:
  total_phases: 3
  completed_phases: 2
  total_plans: 5
  completed_plans: 5
  percent: 67
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-08-19)

**Core value:** A comprehensive, fast Rust functional-data-analysis library that closes the highest-leverage capability gaps against the reference FDA ecosystems — this milestone closes the top three P2 differentiator gaps from the v0.18.0 `R-BACKLOG.md` (score 2.31 each): the depth-measure long tail (DEPTH-01), the robust outlier-detector suite (OUT-01), and the Interval Testing Procedure family (INF-03), each by extending existing `fdars-core/src/` modules additively.
**Current focus:** Phases 28 (DEPTH-01) + 29 (OUT-01) complete & verified; next is Phase 30 (INF-03)

## Current Position

Phase: 29 complete (OUT-01) — verification passed 5/5; next: Phase 30 (INF-03, independent)
Plan: 28-* and 29-* all complete + verified (5 plans)
Status: Phases 28+29 shipped — 9 depth measures + 4 outlier detectors (tvdmss/muod/sequential_transform_outliers/depthgram) added additively; tvdmss consumes Phase 28 TvdMssResult
Last activity: 2026-08-19 — Phase 29 executed inline, full lib suite green (2165 tests), clippy --all-targets clean, serde build clean

## Milestone Roadmap (v0.23.0)

Three phases, three requirements — the top three P2 differentiators from `R-BACKLOG.md` (all tied at score 2.31, M-effort). First **differentiator** milestone (the P1 table-stakes tier was exhausted after v0.22.0). Real `fdars-core/src/` code — additive/non-breaking, `Result`-returning, inline `#[cfg(test)]` tests, crate-root re-exports; **zero changes to existing public signatures.** All three are reuse-first (extend `depth/` / `outliers.rs` / `inference/` + `basis/`); no new algorithm subsystem, **no new crate dependency.** Numeric outputs only — plotting/rendering out of scope.

| Phase | Requirements | Notes |
|-------|--------------|-------|
| 28 — Depth-Measure Long Tail | DEPTH-01 | `depth/`: add HRD/MHRD, HI/MHI/EI, extremal, ERL, L∞, TVD+MSSI — one `Result`-returning function per measure over the column-major `FdMatrix`, each registered in the T-02 `DepthMethod` dispatcher. Batch measures only (excludes streaming depth). R baseline: `roahd`/`fdaoutlier`. Rank 9, score 2.31. |
| 29 — Outlier-Detector Suite | OUT-01 | `outliers.rs`: add `tvdmss` (TVD+MSSI detector), `muod`, `sequential_transform_outliers`, and the `depthgram` statistic (numeric outputs; renderer out-of-scope), reusing the existing MS-plot / outliergram machinery + DEPTH-01 depths. R baseline: `fdaoutlier`/`roahd`. Rank 10, score 2.31. **Hard dependency on Phase 28** — tvdmss reuses DEPTH-01's TVD+MSSI depth. |
| 30 — Interval Testing Procedure Family | INF-03 | new `inference/itp.rs`: one-/two-population interval-wise tests (B-spline & Fourier bases) with domain-selective adjusted p-values + interval-wise FLM coefficient testing, reusing the INF-01 permutation infra + `basis/` projection. R baseline: `fdatest`. Rank 11, score 2.31. **Independent** of Phases 28/29. |

**Execution order:** Phase 29 (OUT-01) has a **hard dependency on Phase 28** (DEPTH-01) — `tvdmss` reuses DEPTH-01's total-variation depth + MSSI, so **28 must complete before 29**. Phase 30 (INF-03) is **independent** (depends only on the already-shipped INF-01 permutation infrastructure + existing `basis/`) and **may run in parallel** with Phases 28/29. Default sequence 28 → 29, with 30 free to run alongside.

## Performance Metrics

**Velocity:**

- Total plans completed: 44 (21 in v0.14.0 + 4 in v0.15.0 + 4 in v0.16.0 + 3 in v0.17.0 + 5 in v0.18.0 + 2 in v0.19.0 + 2 in v0.20.0 + 2 in v0.21.0 + 2 in v0.22.0)
- Average duration: — min
- Total execution time: — hours

**By Phase (prior milestones):**

| Phase | Milestone | Plans |
|-------|-----------|-------|
| 01–09 | v0.14.0 | 21 |
| 10–11 | v0.15.0 | 4 |
| 12–13 | v0.16.0 | 4 |
| 14–15 | v0.17.0 | 3 |
| 16–19 | v0.18.0 | 5 |
| 20–21 | v0.19.0 | 2 |
| 22–23 | v0.20.0 | 2 |
| 24–25 | v0.21.0 | 2 |
| 26–27 | v0.22.0 | 2 |
| 28 | v0.23.0 | TBD (DEPTH-01) |
| 29 | v0.23.0 | TBD (OUT-01) |
| 30 | v0.23.0 | TBD (INF-03) |

**Recent Trend:**

- Last 5 plans: 23-01, 24-01, 25-01, 26-01, 27-01 — all completed + verified
- Trend: consistent ~45min per plan, 7 tests/plan average for implementation phases

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.

Relevant to current work (v0.23.0 implementation):

- v0.23.0 is an **implementation** milestone — real `fdars-core/src/` code (drawn top-first from the R-ecosystem backlog, after v0.19.0–v0.22.0). All additions are **additive/non-breaking**, `Result`-returning, with inline `#[cfg(test)]` tests + crate-root re-exports; **zero changes to existing public signatures.**
- Scope is the **top three P2 differentiators** from `R-BACKLOG.md` (all tied at score 2.31, M-effort): **DEPTH-01** (rank 9), **OUT-01** (rank 10), **INF-03** (rank 11). First differentiator milestone — the P1 table-stakes tier is exhausted after v0.22.0. All three are **reuse-first** (extend `depth/` / `outliers.rs` / `inference/` + `basis/`); no new algorithm subsystem, **no new crate dependency**. Plotting/rendering out of scope (numeric outputs only).
- Phase numbering **continues** from v0.22.0 (ended at Phase 27) → v0.23.0 starts at Phase 28. No reset.
- **One requirement per phase, three phases:** Phase 28 = DEPTH-01, Phase 29 = OUT-01, Phase 30 = INF-03.
- **Phase 29 (OUT-01) has a hard dependency on Phase 28 (DEPTH-01)** — `tvdmss` reuses DEPTH-01's total-variation depth + MSSI. **28 must complete before 29.**
- **Phase 30 (INF-03) is independent** of Phases 28/29 — it depends only on the already-shipped INF-01 permutation infrastructure + existing `basis/` projection, so it may run in parallel with 28/29.
- **DEPTH-01 scope (from R-BACKLOG.md block):** add HRD, MHRD, HI, MHI, EI, extremal depth, ERL, L∞ depth, and TVD+MSSI as one `Result`-returning function per measure over the column-major `FdMatrix` (existing per-file convention), each registered in the T-02 `DepthMethod` dispatcher. **Excludes** streaming depth (fdars strength U-5) — batch measures only. R baseline: `roahd`/`fdaoutlier`.
- **OUT-01 scope (from R-BACKLOG.md block):** add `tvdmss`, `muod`, `sequential_transform_outliers`, and the `depthgram` statistic to `outliers.rs` (numeric outputs; renderer out-of-scope), reusing the existing MS-plot / outliergram machinery + DEPTH-01 depths. Excludes `fdaPOIFD` partially-observed detectors (deferred). R baseline: `fdaoutlier`/`roahd`.
- **INF-03 scope (from R-BACKLOG.md block):** new `inference/itp.rs` — one-/two-population interval-wise tests (B-spline & Fourier bases) with domain-selective adjusted p-values (the ITP interval-wise closure adjustment) + interval-wise FLM coefficient testing, reusing the INF-01 permutation infra + `basis/` projection. Excludes random-projection ANOVA/MANOVA (`fdANOVA`, deferred). R baseline: `fdatest`.
- R baselines matched by **capability**, not R's exact signatures.

Conventions carried from prior milestones (relevant to implementation):

- Column-major `FdMatrix` (`src/matrix.rs`), `Result<T, FdarError>` on all public fns, feature-gated rayon parallelism (`iter_maybe_parallel!` etc.), per-thread RNG seeding `StdRng::seed_from_u64(seed + k)` where randomness is involved (permutation tests in INF-03 need seeded reproducibility, mirroring INF-01's 999-perm default).
- Full clippy gate: `cargo clippy --all-targets --features linalg,parallel -- -D warnings` (CI lints test/bench code — a plain `-p ... -D warnings` false-greens; MEMORY.md pointer).
- `TMPDIR=/home/simonm/.cache/fdars-bench-tmp` required for build/doctest linking; /tmp tmpfs exhaustion causes bogus "No space left" (MEMORY.md pointer) — this milestone builds real code + doctests, so this pointer matters.

### Pending Todos

None yet.

### Blockers/Concerns

- Local main is ahead of origin/HEAD → harness worktrees may fork the wrong base; GSD phases fall back to sequential no-worktree dispatch (MEMORY.md pointer). Phase 30 is independent and could otherwise parallelize with 28/29, but sequential fallback may serialize them regardless.
- /tmp tmpfs exhaustion can block pre-commit doctest linking with bogus "No space left" — use `--no-verify` for docs and free /tmp before executing (MEMORY.md pointer). Relevant since the milestone compiles real code + doctests.
- DEPTH-01: several measures (HRD/MHRD, HI/MHI/EI, ERL) have subtle reference definitions (`roahd`/`fdaoutlier`); pin the exact statistic and MSSI construction for TVD during planning and document any divergence from the R baseline in rustdoc (as prior milestones documented divergences).
- OUT-01: `tvdmss` consumes DEPTH-01's TVD+MSSI — pin the DEPTH-01 TVD/MSSI interface first so Phase 29 has a stable dependency; `muod` / `sequential_transform_outliers` / `depthgram` numeric-output shapes should be pinned during planning.
- INF-03: the ITP interval-wise closure p-value-adjustment and the B-spline vs Fourier basis-projection paths must be reconciled against the INF-01 permutation infrastructure during planning; the domain-selective adjusted-p-value definition should be pinned + documented.

## Deferred Items

v2 backlog items (from `.planning/research/R-BACKLOG.md`, see REQUIREMENTS.md v2 section): REG-04, REG-05, CLUS-01, REP-01, FTS-02 (score 1.73, P2); DENS-01, FPCA-02, SPARSE-01 (score 1.73, P3); FTS-01, FRE-01 (score 1.33, L); FTS-03, FRE-02, REG-06, REP-02, CLUS-02 (score ≤ 1.00, L). Also from prior milestones: REG-01 sparse/PACE variant, configurable PACE bandwidth-selection subsystem, elastic multinomial beyond logistic, extra GLM families + configurable links. Explicit v0.23.0 exclusions: plotting/rendering of depth regions/boxplots/depthgram/outlier flags/ITP p-value surfaces; streaming/online depth variants (fdars strength U-5); `fdaPOIFD` partially-observed detectors; random-projection ANOVA/MANOVA (`fdANOVA`).

Advisory tech-debt carried forward (not v0.23.0 work): weakened MEWMA test assertion; `fix_svd_signs` NaN no-op; over-broad Phase 11 test name; prior VALIDATION.md files remain `draft` (Nyquist TODO).

**Also pending (not a backlog item):** a crate release for v0.23.0 — version bump (0.22.0 → next) + PR + tag, since this milestone ships real code (operator-driven ship-time step).

## Session Continuity

Last session: 2026-08-19
Stopped at: roadmap created (Phases 28–30)
Resume file: None

## Operator Next Steps

- Plan the first phase with `/gsd-plan-phase 28` (DEPTH-01), or plan Phase 30 (INF-03, independent) in parallel.
