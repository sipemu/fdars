# Roadmap: fdars

## Milestones

- ✅ **v0.40.0 Correctness & Release Hardening** — Phases 78–80 (shipped 2026-09-07)
- ✅ **v0.41.0 1.0 API Stabilization Pass** — Phases 81–85 (shipped 2026-09-07)
- ✅ **v0.42.0 1.0 API Finalization** — Phases 86–89 (shipped 2026-09-08)
- ✅ **v0.43.0 Test Determinism & Release Hardening** — Phases 90–93 (shipped 2026-09-09)
- ✅ **v0.44.0 Differentiable Core — Reverse-Mode & Broadened Subset** — Phases 94–100 (shipped 2026-09-12)

## Overview

No active milestone. Start the next with `/gsd-new-milestone`.

With the **API** (v0.42.0), **Quality** (v0.43.0), and **differentiable-core** (v0.44.0, DIF-F1/F2/F3) sections of `documentation/ROADMAP-TO-1.0.md` all cleared, the remaining 1.0 blockers are **SDTW-O1** (algorithm — global `soft_dtw_barycenter` optimizer) and **fdars-j75** (R ecosystem — `fdars-r` FdMatrix migration), before the terminal **1.0-CUT**.

**Operator ship step pending:** `git push origin main` → `git tag v0.44.0` → `release.yml` publishes to crates.io — **0.44.0 supersedes the unpublished 0.41.0/0.42.0/0.43.0** (registry is still at 0.40.0, so one publish catches all four up). `git.create_tag` stays OFF because the tag auto-publishes.

## Phases

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
