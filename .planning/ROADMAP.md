# Roadmap: fdars

## Milestones

- ✅ **v0.40.0 Correctness & Release Hardening** — Phases 78–80 (shipped 2026-09-07)
- ✅ **v0.41.0 1.0 API Stabilization Pass** — Phases 81–85 (shipped 2026-09-07)
- ✅ **v0.42.0 1.0 API Finalization** — Phases 86–89 (shipped 2026-09-08)
- ✅ **v0.43.0 Test Determinism & Release Hardening** — Phases 90–93 (shipped 2026-09-09)
- ✅ **v0.44.0 Differentiable Core — Reverse-Mode & Broadened Subset** — Phases 94–100 (shipped 2026-09-12)
- 🔨 **v0.45.0 Soft-DTW Barycenter — Global Optimizer (SDTW-O1)** — Phases 101–105 (active)

## Overview

**Active milestone: v0.45.0 Soft-DTW Barycenter — Global Optimizer (SDTW-O1).** Replace the single-init MM-descent `soft_dtw_barycenter` with a proper global optimizer — an in-crate L-BFGS local optimizer (two-loop recursion + backtracking/Armijo line search) wrapped in a deterministically-seeded multi-restart loop — closing the last in-core algorithm-quality item (`SDTW-O1`) on `documentation/ROADMAP-TO-1.0.md`. Strictly additive/non-breaking (the legacy 4-arg `soft_dtw_barycenter(data, gamma, max_iter, tol)` fn is preserved as a thin wrapper; new config API is additive). **No new crate dependency** — L-BFGS hand-rolled in-crate (consistent with the v0.44.0 reverse-mode tape decision); must stay WASM- and default-feature-compatible (no `linalg`/`argmin` gating). Ships as 0.45.0 (still 0.x).

With **API** (v0.42.0), **Quality** (v0.43.0), and **differentiable-core** (v0.44.0, DIF-F1/F2/F3) cleared on the 1.0 gap checklist, this milestone clears the **algorithm (SDTW-O1)** blocker — leaving only **fdars-j75** (R ecosystem — `fdars-r` FdMatrix migration) before the terminal **1.0-CUT**.

**Operator ship step pending (prior + this milestone):** `git push origin main` → `git tag v0.45.0` → `release.yml` publishes to crates.io — **0.45.0 supersedes the unpublished 0.41.0/0.42.0/0.43.0/0.44.0** (registry is still at 0.40.0, so one publish catches all up). `git.create_tag` stays OFF because the tag auto-publishes.

## Phases

- [x] **Phase 101: In-Crate L-BFGS Barycenter Optimizer** - Replace the MM-step descent with a hand-rolled limited-memory L-BFGS driven by the exact soft-DTW gradient (completed 2026-09-13)
- [x] **Phase 102: Deterministic Multi-Restart Wrapper** - Run the local optimizer from multiple seeded inits and keep the lowest-cost barycenter, reproducibly (completed 2026-09-13)
- [x] **Phase 103: `SoftDtwBarycenterConfig` API & Legacy Wrapper** - Add the builder-convention config struct + new entry point; preserve the 4-arg fn as a thin non-breaking wrapper (completed 2026-09-13)
- [x] **Phase 104: Quality Evidence — Escapes-Local-Minimum & f64 Parity** - Demonstrate a strictly lower barycenter cost than MM where MM stalls, and preserved legacy f64 behavior (completed 2026-09-13)
- [ ] **Phase 105: Release Preparation & Verification** - Bump to 0.45.0, CHANGELOG, check off SDTW-O1, all whole-crate gates green

## Phase Details

### Phase 101: In-Crate L-BFGS Barycenter Optimizer

**Goal**: The soft-DTW barycenter is refined by a proper local optimizer — an in-crate limited-memory L-BFGS (two-loop recursion + backtracking/Armijo line search) driven by the exact soft-DTW gradient the crate already computes — replacing the fixed-step MM/soft-DBA descent, and it works in the default feature set and on `wasm32`.
**Depends on**: Phase 100 (v0.44.0 complete); reuses the correct soft-DTW gradient from v0.40.0 (CORR-01)
**Requirements**: SDTW-01
**Success Criteria** (what must be TRUE):

  1. `soft_dtw_barycenter` (single-init path) reaches a soft-DTW cost no worse than — and on curved objectives strictly lower than — the previous MM-step descent for the same iteration budget.
  2. The optimizer converges to a bounded barycenter within the data's convex hull (no divergence) and honors `max_iter` / `tol` stopping.
  3. The code compiles and runs with the default feature set and on the `wasm32-unknown-unknown` target — no `linalg`/`argmin` gating and no new crate dependency.
  4. The L-BFGS step consumes the exact soft-DTW gradient already in-crate; the objective and its gradient definitions are unchanged.

**Plans**: 1 plan

- [x] 101-01-PLAN.md — Replace MM descent with in-crate L-BFGS (two-loop + Armijo), safeguards, SDTW-01 unit tests, wasm32/serde compile guards

### Phase 102: Deterministic Multi-Restart Wrapper

**Goal**: A multi-restart wrapper runs the L-BFGS local optimizer from multiple initializations (pointwise mean + deterministically-seeded perturbations) and returns the barycenter with the lowest total soft-DTW cost across restarts, escaping local minima of the non-convex objective — reproducibly for a fixed seed.
**Depends on**: Phase 101
**Requirements**: SDTW-02
**Success Criteria** (what must be TRUE):

  1. The optimizer runs from the pointwise mean plus one or more deterministically-seeded perturbed inits and keeps the restart with the lowest total soft-DTW cost.
  2. Restart seeding follows the project's `StdRng::seed_from_u64(seed + k)` convention.
  3. Repeated runs with the same seed and inputs produce a bit-identical barycenter (reproducible).
  4. The kept barycenter's total soft-DTW cost is no worse than any individual restart's cost.

**Plans**: 1 plan

- [x] 102-01-PLAN.md — Extract single-init L-BFGS refiner; add deterministic seeded multi-restart driver selecting lowest-total-cost barycenter; SDTW-02 tests; wasm32/serde/clippy guards

### Phase 103: `SoftDtwBarycenterConfig` API & Legacy Wrapper

**Goal**: A `SoftDtwBarycenterConfig` builder-convention config struct (`n_restarts`, `max_iter`, `tol`, `gamma`, `seed`) and a new config-taking entry point expose the optimizer, while the existing `soft_dtw_barycenter(data, gamma, max_iter, tol)` signature is preserved as a thin, behavior-compatible wrapper so no existing call site, prelude re-export, R/WASM binding, or example changes.
**Depends on**: Phase 102
**Requirements**: SDTW-03
**Success Criteria** (what must be TRUE):

  1. A `SoftDtwBarycenterConfig` struct exists with `n_restarts`, `max_iter`, `tol`, `gamma`, and `seed` fields following the project's builder/`Default` convention, and a new config-taking entry point drives the optimizer.
  2. The existing 4-arg `soft_dtw_barycenter(data, gamma, max_iter, tol)` fn still compiles with its original signature and delegates to the new path as a thin wrapper (single-restart, legacy behavior).
  3. All 28 examples, the prelude re-exports, and doctests compile unchanged — the change is additive/non-breaking.
  4. The new public items (`SoftDtwBarycenterConfig` + new entry point) are re-exported at the crate root and prelude with a running doctest.

**Plans**: 1 plan

- [x] 103-01-PLAN.md — Config struct + config entry point + legacy delegation + PartialEq on result, additive three-site re-exports, non-breaking guards (SDTW-03)

### Phase 104: Quality Evidence — Escapes-Local-Minimum & f64 Parity

**Goal**: A test (and/or example) demonstrates the new optimizer reaches a strictly lower total soft-DTW barycenter cost than the MM baseline on a case where MM stalls in a local optimum, and that the legacy-wrapper path preserves f64 behavior (converges to a bounded barycenter within the data's convex hull; no divergence).
**Depends on**: Phase 103
**Requirements**: SDTW-04
**Success Criteria** (what must be TRUE):

  1. A committed test (and/or example) constructs a case where the MM baseline stalls in a local optimum and asserts the new optimizer's total soft-DTW cost is strictly lower.
  2. A test asserts the legacy-wrapper path converges to a bounded barycenter within the data's convex hull with no divergence (f64 behavior preserved).
  3. The quality-evidence tests are deterministic (fixed seed) and pass under the full `cargo test` run.

**Plans**: 1 plan

- [x] 104-01-PLAN.md — escape-from-local-minimum + legacy convex-hull f64-parity tests (SDTW-04)

### Phase 105: Release Preparation & Verification

**Goal**: `fdars-core` is release-ready at 0.45.0 — version bumped, both CHANGELOGs written, `SDTW-O1` checked off on `documentation/ROADMAP-TO-1.0.md`, and all whole-crate release gates green; the `git tag v0.45.0` → crates.io publish remains the deferred operator step.
**Depends on**: Phases 101, 102, 103, 104
**Requirements**: REL-01
**Success Criteria** (what must be TRUE):

  1. `fdars-core` is bumped 0.44.0 → 0.45.0 with a `[0.45.0]` entry in both CHANGELOGs (root + crate), and `SDTW-O1` is checked off on `documentation/ROADMAP-TO-1.0.md`.
  2. All whole-crate gates pass green: `cargo fmt --check`, `cargo clippy --all-targets --features linalg,parallel -- -D warnings`, full `cargo test`, a `--features serde` build guard, all 28 examples + doctests, and `cargo package`.
  3. No `git tag`/publish is performed inside the phase — the `git tag v0.45.0` → crates.io publish (via `release.yml`) is left as the deferred operator step (GSD `git.create_tag` stays off).

**Plans**: 1 plan

Plans:
- [ ] 105-01-PLAN.md — Bump 0.45.0 + both CHANGELOGs + SDTW-O1 checkoff; full whole-crate gate battery green (no tag/publish)

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 101. In-Crate L-BFGS Barycenter Optimizer | 1/1 | Complete    | 2026-09-13 |
| 102. Deterministic Multi-Restart Wrapper | 1/1 | Complete    | 2026-09-13 |
| 103. `SoftDtwBarycenterConfig` API & Legacy Wrapper | 1/1 | Complete    | 2026-09-13 |
| 104. Quality Evidence — Escapes-Local-Minimum & f64 Parity | 1/1 | Complete    | 2026-09-13 |
| 105. Release Preparation & Verification | 0/? | Not started | - |

<details>
<summary>✅ v0.44.0 Differentiable Core — Reverse-Mode & Broadened Subset (Phases 94–100) — SHIPPED 2026-09-12</summary>

Implementation milestone completing the entire **differentiable-core** section of the 1.0 gap checklist (DIF-F1/F2/F3). Added a hand-written in-crate reverse-mode (VJP) tape (`Var`/Wengert-list, `vjp` entry point) alongside the v0.39.0 forward-mode `Dual`; broadened the differentiable operation set across four families (basis eval + inner products, regression prediction, smoothing penalties, depth + curve distances); generalized targeted f64 hot-path signatures over the scalar type via defaulted type params (`T = f64`), proven non-breaking at compile time. Strictly additive/non-breaking, no new crate dependency. Full detail: [`milestones/v0.44.0-ROADMAP.md`](milestones/v0.44.0-ROADMAP.md); archived phase artifacts: `milestones/v0.44.0-phases/`.

- [x] Phase 94: Reverse-Mode Autodiff Core (VJP Tape) (4/4 plans) — completed 2026-09-11
- [x] Phase 95: Generic Scalar Hot-Path Signatures (3/3 plans) — completed 2026-09-11
- [x] Phase 96: Differentiable Basis Evaluation & Inner Products (3/3 plans) — completed 2026-09-11
- [x] Phase 97: Differentiable Regression Prediction & Smoothing Penalties (3/3 plans) — completed 2026-09-11
- [x] Phase 98: Differentiable Depth & Curve Distances (1/1 plan) — completed 2026-09-11
- [x] Phase 99: End-to-End Autodiff Flow & Gradient API (1/1 plan) — completed 2026-09-11
- [x] Phase 100: Release Preparation & Verification (1/1 plan) — completed 2026-09-11

**Requirements:** 11/11 satisfied (RAD-01/02/03, DOP-01/02/03/04, GEN-01/02, API-01, REL-01). Audit: passed; integration COMPLETE (9/9 wiring, 6/6 E2E flows). Release-ready — operator `git tag v0.44.0` → crates.io publish pending (the deferred operator step).

</details>

<details>
<summary>✅ v0.43.0 Test Determinism & Release Hardening (Phases 90–93) — SHIPPED 2026-09-09</summary>

Quality + release-hardening milestone. Root-caused the long-standing `co_cluster`/`svd_sign` golden flake as a **deterministic faer-vs-nalgebra SVD backend divergence** in `fdata_to_pc` when built without `linalg` (NOT the previously-hypothesized environmental/disk-pressure flake — that was overturned in Phase 90), and fixed it with a least-invasive test-side `#[cfg_attr(not(feature = "linalg"), ignore)]` guard on the three affected tests — no new dependency, no tolerance relaxation, no `src/` change. A suite-wide robustness audit (all three fragility classes) found **zero further fragile tests**; a new `determinism-guardrail` CI job (repeat + `RAYON_NUM_THREADS=1` + anti-silent-skip golden assertions) locks the green baseline and gates the crates.io publish. Cleared the **Quality** blocker on the 1.0 gap checklist. Full detail: [`milestones/v0.43.0-ROADMAP.md`](milestones/v0.43.0-ROADMAP.md); archived phase artifacts: `milestones/v0.43.0-phases/`.

- [x] Phase 90: Golden-Flake Root-Cause & Deterministic Fix (1/1 plan) — completed 2026-09-09
- [x] Phase 91: Suite-Wide Robustness Sweep (1/1 plan) — completed 2026-09-09
- [x] Phase 92: CI Determinism Guardrail (1/1 plan) — completed 2026-09-09
- [x] Phase 93: Release Preparation & Readiness Verification (1/1 plan) — completed 2026-09-09

**Requirements:** 7/7 satisfied (FLAKE-01/02, ROBUST-01/02, CI-01, REL-01/02). Audit: passed. Release-ready — operator `git tag v0.43.0` → crates.io publish pending (the deferred operator step; 0.43.0 supersedes the unpublished 0.41.0/0.42.0 — registry is still at 0.40.0, so one publish catches all three up).

</details>

<details>
<summary>✅ v0.42.0 1.0 API Finalization (Phases 86–89) — SHIPPED 2026-09-08</summary>

The second breaking milestone — clears the entire **API section** of the 1.0 gap checklist (`documentation/ROADMAP-TO-1.0.md`): sealed `wire` `pub(crate)` (SEAL-01), `#[non_exhaustive]` on 38 config structs + the construction-idiom change (SEAL-02/03), and the full `_1d`/`_2d` → `Dim`-dispatch naming unification. API-shape-only — no numeric/behavioral change (byte-identical `_impl` bodies, code-review confirmed; 2857-test non-regression). Full detail: [`milestones/v0.42.0-ROADMAP.md`](milestones/v0.42.0-ROADMAP.md); archived phase artifacts: `milestones/v0.42.0-phases/`.

- [x] Phase 86: Surface Sealing (1/1 plan) — completed 2026-09-08
- [x] Phase 87: Targeted Renames (1/1 plan) — completed 2026-09-08
- [x] Phase 88: Large Suffix Batch (1/1 plan) — completed 2026-09-08
- [x] Phase 89: Release Preparation & Verification (1/1 plan) — completed 2026-09-08

**Requirements:** 9/9 satisfied (SEAL-01/02/03, NAME-01/02/03/04/05, REL-01). Audit: passed. Release-ready — operator `git tag v0.42.0` → crates.io publish pending (the deferred operator step).

</details>

<details>
<summary>✅ v0.41.0 1.0 API Stabilization Pass (Phases 81–85) — SHIPPED 2026-09-07</summary>

First breaking milestone after a long additive-only run. Audit → user-approved breaking-change inventory → per-category cleanups → stability deliverables → release prep. Ships as 0.41.0 (NOT 1.0). Full detail: [`milestones/v0.41.0-ROADMAP.md`](milestones/v0.41.0-ROADMAP.md); archived phase artifacts: `milestones/v0.41.0-phases/`.

- [x] Phase 81: API Audit & Deprecated-Form Removal (2/2 plans) — completed 2026-09-07
- [x] Phase 82: Public-Surface Sealing & Non-Exhaustive Coverage (2/2 plans) — completed 2026-09-07
- [x] Phase 83: Naming Unification (3/3 plans) — completed 2026-09-07
- [x] Phase 84: Stability Deliverables (1/1 plan) — completed 2026-09-07
- [x] Phase 85: Release Preparation & Verification (1/1 plan) — completed 2026-09-07

**Requirements:** 9/9 satisfied (AUDIT-01, API-01/02/03/04, STAB-01/02/03, REL-01). Audit: passed, integration INTEGRATED. Release-ready — operator `git tag v0.41.0` → publish pending.

</details>

<details>
<summary>✅ v0.40.0 Correctness & Release Hardening (Phases 78–80) — SHIPPED 2026-09-07</summary>

Full detail: [`milestones/v0.40.0-ROADMAP.md`](milestones/v0.40.0-ROADMAP.md).

</details>

**Shipped:** v0.14.0 → v0.44.0 (see the Milestones list above and the `milestones/` archives).
</content>
</invoke>
