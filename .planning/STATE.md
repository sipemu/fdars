---
gsd_state_version: "1.0"
milestone: v0.45.0
current_phase: 103
current_phase_name: SoftDtwBarycenterConfig API & Legacy Wrapper
status: executing
stopped_at: Phase 102 complete, ready to plan Phase 103
last_updated: "2026-09-13T00:10:51.902Z"
last_activity: 2026-09-13
last_activity_desc: Phase 103 execution started
state_head: 8ff8b573815c3624e9e8f83a5d57e75a7b82f686
progress:
  total_phases: 5
  completed_phases: 0
  total_plans: 2
  completed_plans: 2
milestone_name: Soft-DTW Barycenter — Global Optimizer (SDTW-O1)
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-09-13)

**Core value:** A comprehensive, fast Rust functional-data-analysis library that closes the highest-leverage capability and performance gaps on the road to a deliberate 1.0 cut.
**Current focus:** Phase 103 — SoftDtwBarycenterConfig API & Legacy Wrapper

## Current Position

Phase: 103 (SoftDtwBarycenterConfig API & Legacy Wrapper) — EXECUTING
Plan: 1 of 1
Status: Executing Phase 103
Last activity: 2026-09-13 — Phase 103 execution started

## Milestone Roadmap (v0.45.0)

Five phases, 5 requirements — an implementation milestone replacing the MM-step descent in `soft_dtw_barycenter` with a proper global optimizer (in-crate L-BFGS + deterministic multi-restart). Strictly additive/non-breaking (legacy 4-arg `soft_dtw_barycenter(data, gamma, max_iter, tol)` preserved as a thin wrapper; new config API additive; protects R + WASM + 28 examples). **No new crate dependency** — L-BFGS hand-rolled in-crate (consistent with the v0.44.0 reverse-mode tape decision); must compile in default features + `wasm32` (no `linalg`/`argmin` gating). Fine granularity. Phase numbering continues from v0.44.0 (ended at 100) → **Phase 101**. No reset.

| Phase | Requirements | Notes |
|-------|--------------|-------|
| 101 — In-Crate L-BFGS Barycenter Optimizer | SDTW-01 | Replace the fixed-step MM/soft-DBA descent with a hand-rolled limited-memory L-BFGS (two-loop recursion + backtracking/Armijo line search) driven by the exact soft-DTW gradient already in-crate (correct since v0.40.0 CORR-01). Compiles + runs in default features and on `wasm32` (no `linalg`/`argmin` gating). No new crate dependency. First phase. |
| 102 — Deterministic Multi-Restart Wrapper | SDTW-02 | Wrap the local optimizer in a multi-restart loop: pointwise mean + deterministically-seeded perturbations (`StdRng::seed_from_u64(seed + k)` convention); keep the barycenter with the lowest total soft-DTW cost; reproducible for a fixed seed. Escapes local minima of the non-convex objective. Depends on 101. |
| 103 — `SoftDtwBarycenterConfig` API & Legacy Wrapper | SDTW-03 | Add the builder-convention config struct (`n_restarts`, `max_iter`, `tol`, `gamma`, `seed`) + a new config-taking entry point; preserve the existing `soft_dtw_barycenter(data, gamma, max_iter, tol)` fn as a thin, behavior-compatible wrapper so no call site, prelude re-export, R/WASM binding, or example changes. Additive crate-root/prelude re-exports + running doctest. Depends on 102. |
| 104 — Quality Evidence — Escapes-Local-Minimum & f64 Parity | SDTW-04 | A test (and/or example) shows the new optimizer reaches a strictly lower total soft-DTW barycenter cost than the MM baseline where MM stalls in a local optimum, and that the legacy-wrapper path preserves f64 behavior (bounded barycenter within the data's convex hull; no divergence). Deterministic (fixed seed). Depends on 103. |
| 105 — Release Preparation & Verification | REL-01 | **Terminal.** Bump 0.44.0 → 0.45.0, CHANGELOG `[0.45.0]` (root + crate), check off `SDTW-O1` on `ROADMAP-TO-1.0.md`; all whole-crate gates green — `cargo fmt --check`, `clippy --all-targets --features linalg,parallel -- -D warnings`, full `cargo test`, `--features serde` build guard, all 28 examples + doctests, `cargo package`. The `git tag v0.45.0` → crates.io publish is the DEFERRED operator step. Depends on all prior phases. |

**Execution order:** 101 → 102 → 103 → 104 → 105. Strict dependency chain: local optimizer first, then the multi-restart wrapper around it, then the config API + legacy wrapper, then the quality-evidence tests, then release prep last. All 5 requirements mapped, no orphans, no duplicates.

**Gates (this additive, non-breaking milestone):** `cargo fmt --check`, `cargo clippy --all-targets --features linalg,parallel -- -D warnings` (CI lints test/bench code — use `--all-targets`), full `cargo test` (the non-regression + quality proof), a `--features serde` build guard, all 28 examples + doctests, and `cargo package`. **No new crate dependency** (in-crate hand-rolled L-BFGS). Must compile in default features + `wasm32`.

## Performance Metrics

**Velocity:**

- Total plans completed: 145+ (across v0.14.0–v0.44.0)
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
| 94–100 | v0.44.0 | 16 |
| 101–105 | v0.45.0 | 0/? (planned) |

**Recent Trend:**

- Last milestone: v0.44.0 (phases 94–100, 16 plans) — audit 11/11, integration COMPLETE, release-ready (operator tag/publish pending). Cleared the **differentiable-core** (DIF-F1/F2/F3) section of the 1.0 checklist.
- Trend: v0.45.0 is an implementation milestone clearing the **algorithm (SDTW-O1)** blocker on the 1.0 checklist — additive/non-breaking, no new crate dependency, in-crate L-BFGS + multi-restart. After this, only fdars-j75 (R ecosystem) remains before the terminal 1.0-CUT.

*Updated after each plan completion*

**Per-Plan Metrics:**

| Plan | Duration | Tasks | Files |
|------|----------|-------|-------|
| (none yet — v0.45.0) | — | — | — |
| Phase 101 P01 | 10 | 4 tasks | 2 files |

## Accumulated Context

### Decisions

Decisions relevant to current work (v0.45.0):

- **Strictly additive/non-breaking** — the legacy 4-arg `soft_dtw_barycenter(data, gamma, max_iter, tol)` fn is preserved as a thin, behavior-compatible wrapper; the new `SoftDtwBarycenterConfig` API is additive alongside it. All 28 examples, prelude re-exports, and R/WASM bindings must compile unchanged.
- **No new crate dependency** — L-BFGS is hand-rolled in-crate (two-loop recursion + backtracking/Armijo line search), consistent with the v0.44.0 reverse-mode tape and v0.39.0 forward-mode `Dual` decisions. `argmin` is only reachable transitively behind the optional `linalg` feature; the barycenter must keep working in default + WASM builds — **no `linalg`/`argmin` gating**.
- **Optimizer consumes the existing gradient** — the soft-DTW objective and its gradient are unchanged (gradient correct since v0.40.0 CORR-01); this milestone changes only the *optimizer* consuming it. Do NOT touch the objective, gradient, or `Var`/`Dual` autodiff core.
- **Deterministic multi-restart** — restart inits are the pointwise mean + deterministically-seeded perturbations via `StdRng::seed_from_u64(seed + k)`; keep the lowest-total-cost restart; reproducible for a fixed seed. Restart parallelism (feature-gated `rayon`) is an optional implementation detail, NOT a scoped requirement — determinism + correctness are the goal.
- **Config struct fields** — `SoftDtwBarycenterConfig { n_restarts, max_iter, tol, gamma, seed }`, builder/`Default` convention (matches the crate's config-struct idiom; note config structs are `#[non_exhaustive]` since v0.42.0 — construct via `default()` + field assignment).
- **Warp-searched `elastic_distance` stays out of scope** — a discrete DP argmin has no meaningful gradient (deferred at v0.39.0 DIF-02); only the smooth soft-DTW objective is optimized here.
- **Phase numbering continues** — v0.44.0 ended at Phase 100 → v0.45.0 starts at Phase 101. No reset.
- **5 requirements → 5 phases:** 101 SDTW-01; 102 SDTW-02; 103 SDTW-03; 104 SDTW-04; 105 REL-01. One requirement per phase (fine granularity, tight dependency chain). All mapped, no orphans, no duplicates.
- **REL-01 lands last (Phase 105)** — bump/CHANGELOG/checklist/gates are the final proof; the full-suite `cargo test` gate is the end-to-end milestone proof. The `git tag v0.45.0` → crates.io publish is the DEFERRED operator step (GSD `git.create_tag` is off because `release.yml` couples tag-push to publish) — never tagged/published inside a phase.
- [Phase 101]: L-BFGS replaces MM descent in soft_dtw_barycenter: two-loop recursion (m=10 VecDeque history), Armijo backtracking (c1=1e-4, beta=0.5, max_bt=20), pure std Rust — no new crate dependency, wasm32-safe, public 4-arg signature preserved
- [Phase 101]: Symmetric-init degenerate case (4 quarter-period shifts) produces zero gradient at L-BFGS initialization — correct optimizer behavior; test updated to use asymmetric 3-shift dataset preserving CORR-01 spirit

### Pending Todos

- **Operator ship steps still pending** — `git tag v0.41.0`/`v0.42.0`/`v0.43.0`/`v0.44.0` → push → crates.io publish not yet performed; registry is still at 0.40.0. A future 0.45.0 publish supersets all the unpublished versions in one shot.
- **Migrate `fdars-r` R wrapper to the `FdMatrix` API** (issue `fdars-j75`) — carried forward; separate package, out of `fdars-core` scope; the remaining 1.0-ecosystem gap after this milestone.

### Blockers/Concerns

- **No research/SUMMARY.md relevant to this milestone** — intentional: internal algorithm improvement, not an ecosystem-parity audit. Reference baseline is the soft-DBA / L-BFGS literature + the exact soft-DTW gradient already in-crate (v0.40.0). The existing `research/SUMMARY.md` (dated 2026-09-02) is stale for this milestone; REQUIREMENTS.md is authoritative. Non-blocking for the roadmap.
- **Non-breakingness is the central constraint** — the legacy 4-arg fn signature must survive byte-for-byte; the config API is additive. All 28 examples + R/WASM bindings + every call site must compile unchanged. Guard it at Phase 103.
- **Default + WASM compatibility** — the optimizer must compile and run with the default feature set and on `wasm32-unknown-unknown` (no `linalg`/`argmin` gating). Guard it at Phase 101.
- **`soft_dtw_backward` gradient is correct** — fixed in v0.40.0 (CORR-01, endpoint-seed bug); this milestone consumes it as-is and must not regress or redefine it.
- Historical build/CI hazards (MEMORY.md) apply: run clippy `--all-targets --features linalg,parallel -- -D warnings` (CI lints test/bench code); run `cargo fmt` per commit (`--no-verify` commits leave fmt drift → CI fmt-check fails); keep the `--features serde` build green (repaired v0.40.0 — do not regress); watch `/tmp` and `target/` disk pressure on full builds (`rm -rf target/debug/{incremental,examples}` to free space; doctests link in a small `/tmp` tmpfs); the pre-commit hook runs the full cargo gate and times out — prefer inline execution + `commit --no-verify` after out-of-band gates; full combined gate as one background bash gets killed mid-run — run gates per-gate FOREGROUND with 600s timeout.

## Deferred Items

Items acknowledged and deferred, most recent first:

| Category | Item | Status | Deferred At | Milestone |
|----------|------|--------|-------------|-----------|
| 1.0-cut | 1.0-CUT — bump to 1.0.0 and declare the public API stable, once every ROADMAP-TO-1.0.md item clears | Deferred | v0.41.0 | future (deliberate 1.0 cut) |
| fdars-r | `fdars-r` FdMatrix migration (issue `fdars-j75`) — migrate the external R wrapper to the `FdMatrix` API; separate package, out of `fdars-core` scope | Deferred | v0.41.0 | future milestone |

## Session Continuity

Last session: 2026-09-12T23:20:08.817Z
Stopped at: Phase 102 complete, ready to plan Phase 103
Resume file: None

## Operator Next Steps

- Plan the first phase with `/gsd-plan-phase 101`

</content>
