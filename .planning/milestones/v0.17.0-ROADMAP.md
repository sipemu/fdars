# Roadmap: fdars

## Milestones

- ✅ **v0.14.0 — Performance & scikit-fda Gap Audit** — Phases 1–9 (shipped 2026-08-09) — [archive](milestones/v0.14.0-ROADMAP.md)
- ✅ **v0.15.0 — Top-Backlog Quick Wins** — Phases 10–11 (shipped 2026-08-11) — [archive](milestones/v0.15.0-ROADMAP.md)
- ✅ **v0.16.0 — Elastic Feasibility + Parity Quick Wins** — Phases 12–13 (complete 2026-08-12, release pending) — [archive](milestones/v0.16.0-ROADMAP.md)
- 🔵 **v0.17.0 — Registration Parity & Elastic-FPCA Performance** — Phases 14–15 (active)

## Phases

- [x] **Phase 14: Shift Registration** - Least-squares rigid-shift registration + the three scikit-fda registration-quality scores (completed 2026-08-12)
- [x] **Phase 15: Elastic-FPCA Performance** - Parallelize the three per-curve elastic-FPCA loops, equivalence-tested (completed 2026-08-12)

## Phase Details

### Phase 14: Shift Registration

**Goal**: A user can register a set of curves by simple rigid horizontal shift to the sample mean, and quantify how well any registration worked with the three standard scikit-fda diagnostics — closing the "simplest registration method" gap (fdars currently jumps from landmark shifts straight to full elastic SRSF warping).
**Depends on**: Nothing (first phase of this milestone; disjoint files from Phase 15)
**Requirements**: FEAT-06, FEAT-07
**Success Criteria** (what must be TRUE):

  1. A user can call `least_squares_shift_registration(data, argvals, ...)` and get back the registered curves plus the per-curve shift value `δᵢ` for every input curve, where each curve has been rigidly shifted (via linear-interpolation resampling) to minimize its L2 distance to the sample mean using golden-section / ternary search.
  2. On an already-aligned set of curves the estimated shifts are ≈ 0 and the curves are returned essentially unchanged; on a set of identical curves offset by known constant shifts, the recovered `δᵢ` match the injected offsets within tolerance.
  3. A user can score a registration with `least_squares_score` (∑‖registeredᵢ − mean‖²/n), `pairwise_correlation_score` (mean pairwise correlation between registered curves), and `sobolev_least_squares_score` (derivative-penalized LS), all living in `alignment/quality.rs` alongside the existing `alignment_quality` / `warp_complexity` / `warp_smoothness`.
  4. The quality scores move in the expected direction: a well-registered curve set yields a lower `least_squares_score` and higher `pairwise_correlation_score` than the same curves before registration (verified on a synthetic shifted-bumps set).
  5. All new public functions return `Result<T, FdarError>` (never panic on bad input), are re-exported at the crate root, carry inline `#[cfg(test)]` tests, and add no new API breakage — existing `alignment/` signatures are untouched.

**Plans**: 2 plans

- [x] 14-01-PLAN.md — FEAT-06: shift.rs tracer + least_squares_shift_registration + ShiftRegistrationResult (FEAT-06-A…E)
- [x] 14-02-PLAN.md — FEAT-07: three quality scores in quality.rs (FEAT-07-A…F) + consolidated mod.rs/lib.rs re-exports

### Phase 15: Elastic-FPCA Performance

**Goal**: The elastic-FPCA critical path runs in parallel under the `parallel` feature, cutting wall-clock on the registration-aware FPCA path for realistic curve counts (N ≥ 50) while producing output numerically equivalent to the sequential path.
**Depends on**: Nothing (independent of Phase 14 — isolated to `elastic_fpca.rs`, disjoint files; may run before, after, or concurrently with Phase 14)
**Requirements**: PERF-04
**Success Criteria** (what must be TRUE):

  1. The three per-curve loops in `elastic_fpca.rs` — `shooting_vectors_from_psis` (:701), `build_augmented_srsfs` (:720), and `svd_scores_and_eigenvalues` (:764) — execute via `iter_maybe_parallel!(0..n)` under the `parallel` feature, and compile/run sequentially (identical code path) when the feature is off.
  2. An inline `#[cfg(test)]` equivalence test confirms the parallel `vert_fpca` / `joint_fpca` output (scores and eigenvalues) matches the sequential path within numerical tolerance — floating-point summation order is controlled so elastic-geometry results agree.
  3. The light `:764` body is guarded by a size threshold (N ≳ 50) — or a documented small-N regression is explicitly accepted — so parallel dispatch is only taken where it pays back, per the audit's streaming-sentinel payback rule.
  4. The change is additive and non-breaking: `vert_fpca` / `joint_fpca` public signatures and their `Result<T, FdarError>` returns are unchanged, no new dependencies are added, and the feasibility win (parallel completes correctly at N ≥ 50) is demonstrated rather than pinned to a precise speedup number — the audit flagged elastic cells LOW-CONFIDENCE under an unpinned governor.

**Plans**: 1 plan

- [x] 15-01-PLAN.md — PERF-04: parallelize the three elastic-FPCA loops (:701/:720/:764) via iter_maybe_parallel! with N≥50 guard, equivalence-tested (PERF-04-A…F)

---

<details>
<summary>✅ v0.14.0 Performance & scikit-fda Gap Audit (Phases 1–9) — SHIPPED 2026-08-09</summary>

Audit-only milestone — every phase produced analysis artifacts, zero `fdars-core/src/` edits. Deliverables: `.planning/research/AUDIT-REPORT.md` (consolidated report) + `.planning/research/BACKLOG.md` (32-item value-ranked backlog).

- [x] Phase 1: Measurement Discipline & Baselines (2/2 plans) — completed 2026-08-07
- [x] Phase 2: Static Hot-Path Analysis (2/2 plans) — completed 2026-08-07
- [x] Phase 3: Elastic Alignment Hot Path (2/2 plans) — completed 2026-08-08
- [x] Phase 4: FPCA/SVD & Allocation Audit (3/3 plans) — completed 2026-08-08
- [x] Phase 5: Parallelism Gap Assessment (3/3 plans) — completed 2026-08-08
- [x] Phase 6: Conditional SVD Library Comparison (1/1 plans) — completed 2026-08-09
- [x] Phase 7: scikit-fda Capability Enumeration (2/2 plans) — completed 2026-08-09
- [x] Phase 8: Capability Parity Matrix & Categorization (3/3 plans) — completed 2026-08-09
- [x] Phase 9: Consolidated Report & Prioritized Backlog (3/3 plans) — completed 2026-08-09

Full phase detail: [milestones/v0.14.0-ROADMAP.md](milestones/v0.14.0-ROADMAP.md)

</details>

<details>
<summary>✅ v0.15.0 Top-Backlog Quick Wins (Phases 10–11) — SHIPPED 2026-08-11</summary>

First implementation milestone — the top-4 audit-backlog quick wins delivered as real `fdars-core/src/` code, each with inline tests and numerical verification. Full suite green; milestone audit passed (4/4); shipped via PR #38, `fdars-core` 0.15.0 on crates.io.

- [x] Phase 10: Capability Gaps — Spline Interpolation & Functional Summary Statistics (2/2 plans) — completed 2026-08-10 (FEAT-01, FEAT-02)
- [x] Phase 11: Performance Wins — Parallel CV Folds & faer FPCA SVD (2/2 plans) — completed 2026-08-11 (PERF-01, PERF-02)

Full phase detail: [milestones/v0.15.0-ROADMAP.md](milestones/v0.15.0-ROADMAP.md)

</details>

<details>
<summary>✅ v0.16.0 Elastic Feasibility + Parity Quick Wins (Phases 12–13) — COMPLETE 2026-08-12 (release pending)</summary>

Second implementation milestone — the next tier of the v0.14.0 audit backlog: the P1 elastic-feasibility headline plus three effort-S scikit-fda parity gaps, all additive/non-breaking. Milestone audit passed (4/4 requirements, cross-phase integration clean, 2663 tests green). Not yet released — needs a version bump (0.15.0 → 0.16.0) + PR to protected `main` + tag.

- [x] Phase 12: Elastic Feasibility — Banded Alignment Default & `band_frac` (1/1 plans) — completed 2026-08-12 (PERF-03: opt-in `*_with_band` wrappers, large grids feasible)
- [x] Phase 13: Parity Quick Wins — Imputation, Extrapolation Policy & Scoring Metrics (2 plans + 1 gap-closure) — completed 2026-08-12 (FEAT-03 imputation, FEAT-04 `ExtrapolationPolicy` both interp paths, FEAT-05 five scoring metrics)

Full phase detail: [milestones/v0.16.0-ROADMAP.md](milestones/v0.16.0-ROADMAP.md)

</details>

---
*Latest: v0.17.0 Registration Parity & Elastic-FPCA Performance started 2026-08-12 — Phases 14–15, 3 requirements (FEAT-06 shift registration + FEAT-07 registration-quality scores → Phase 14; PERF-04 parallelize elastic-FPCA → Phase 15). Prior: v0.16.0 code-complete 2026-08-12 (release pending — version bump + PR + tag); v0.15.0 shipped 2026-08-11 (crates.io 0.15.0); v0.14.0 audit shipped 2026-08-09.*
