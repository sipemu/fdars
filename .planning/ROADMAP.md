# Roadmap: fdars

## Milestones

- ✅ **v0.40.0 Correctness & Release Hardening** — Phases 78–80 (shipped 2026-09-07)
- ✅ **v0.41.0 1.0 API Stabilization Pass** — Phases 81–85 (shipped 2026-09-07)
- ✅ **v0.42.0 1.0 API Finalization** — Phases 86–89 (shipped 2026-09-08)
- ✅ **v0.43.0 Test Determinism & Release Hardening** — Phases 90–93 (shipped 2026-09-09)
- 🚧 **v0.44.0 Differentiable Core — Reverse-Mode & Broadened Subset** — Phases 94–100 (in progress)

## Overview

v0.44.0 completes the entire **differentiable-core** section of `documentation/ROADMAP-TO-1.0.md` (DIF-F1/F2/F3). It adds an in-crate **reverse-mode (VJP) autodiff** core alongside the existing v0.39.0 forward-mode `Dual`, **broadens the differentiable operation set** across four algorithm families (basis eval + inner products, scalar-on-function regression prediction, smoothing/roughness penalties, depth/curve distances), and **generalizes existing f64 hot-path signatures over the scalar type** via defaulted type params (`T = f64`). The whole milestone is **strictly additive/non-breaking**: every existing f64 call site, the R + WASM bindings, and all 28 examples must compile unchanged. No new crate dependency — the reverse-mode tape is hand-written in-crate, exactly as the forward-mode `Dual` was built in v0.39.0. Ships as 0.44.0 (still 0.x). After this milestone the remaining 1.0 blockers are SDTW-O1 (algorithm) and fdars-j75 (R ecosystem), before the terminal 1.0-CUT.

## Phases

**Phase Numbering:**

- Integer phases (94, 95, …): Planned milestone work
- Decimal phases (94.1, …): Urgent insertions (marked INSERTED)

Phase numbering continues from v0.43.0 (ended at 93) → **Phase 94**. No reset.

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

### 🚧 v0.44.0 Differentiable Core — Reverse-Mode & Broadened Subset (In Progress)

**Milestone Goal:** Complete the differentiable-core section of `documentation/ROADMAP-TO-1.0.md` (DIF-F1/F2/F3) — an in-crate reverse-mode (VJP) tape alongside forward-mode `Dual`, a broadened differentiable operation set across four families, and generic-over-scalar hot-path signatures — strictly additive so no existing call site, R/WASM binding, or example changes.

- [x] **Phase 94: Reverse-Mode Autodiff Core (VJP Tape)** - Hand-written in-crate reverse-mode tape + backward pass + VJP entry point, validated vs forward-mode and finite differences (completed 2026-09-11)
- [ ] **Phase 95: Generic Scalar Hot-Path Signatures** - Generalize targeted hot-path signatures over the scalar type via defaulted type params (`T = f64`); non-breakingness proven at compile time
- [ ] **Phase 96: Differentiable Basis Evaluation & Inner Products** - Basis eval (B-spline/Fourier) + functional inner products generic over `Scalar` and differentiable
- [ ] **Phase 97: Differentiable Regression Prediction & Smoothing Penalties** - `fregre_lm`/FPCR prediction and roughness-penalty evaluation generic over `Scalar` and differentiable
- [ ] **Phase 98: Differentiable Depth & Curve Distances** - Functional depth + curve distances (beyond soft-DTW) generic over `Scalar` and differentiable
- [ ] **Phase 99: End-to-End Autodiff Flow & Gradient API** - Autodiff types flow through the generalized hot-paths end-to-end; unified `grad`/`jacobian`/`vjp` API + composition demo + re-exports + doctest
- [ ] **Phase 100: Release Preparation & Verification** - Bump 0.43.0 → 0.44.0, CHANGELOG, DIF-F1/F2/F3 checked off, all whole-crate gates green incl. `cargo package`

## Phase Details

### Phase 94: Reverse-Mode Autodiff Core (VJP Tape)

**Goal**: An in-crate reverse-mode (vector-Jacobian-product) autodiff core exists alongside the forward-mode `Dual`, with backward-pass gradient accumulation validated against the existing differentiable subset.
**Depends on**: Nothing (first phase of this milestone; builds on the shipped v0.39.0 `Scalar`/`Dual` design, which already exists)
**Requirements**: RAD-01, RAD-02, RAD-03
**Success Criteria** (what must be TRUE):

  1. A reverse-mode tape (Wengert-list) records operations on a `Var`/tape scalar type supporting the full forward-mode operation set (±, ×, ÷, sqrt, exp, ln, sin/cos, powf, abs, comparisons).
  2. A backward pass seeds the output adjoint and accumulates input gradients, exposed through a `vjp` entry point efficient for many-input→scalar objectives.
  3. Reverse-mode gradients match the forward-mode `Dual` path and central finite differences within tolerance on elastic soft-DTW distance and FPCA scores (the existing differentiable subset).
  4. No new crate dependency is added (the tape is hand-written in-crate, matching how `Dual` was built).

**Plans**: 4/4 plans executed

- [x] 94-01-tracer-tape-skeleton-PLAN.md — TRACER: refactor autodiff.rs → autodiff/{mod,forward}, stand up reverse.rs skeleton (Var/Tape/Node/Mul/minimal vjp) + end-to-end known-answer test (RAD-01, RAD-02)
- [x] 94-02-full-op-set-PLAN.md — full op set on Var (arithmetic + transcendentals), complete Scalar impl, Tier-1 known-answer + Tier-2 singular-point tests (RAD-01)
- [x] 94-03-vjp-hardening-agreement-PLAN.md — harden vjp lifecycle (double-clear, edge cases) + Tier-3 reverse-vs-Dual agreement tests (RAD-02)
- [x] 94-04-validation-prelude-PLAN.md — Tier-4 FD cross-checks (soft_dtw + FPCA scores + composed objective) + prelude re-exports + vjp doctest (RAD-03)

### Phase 95: Generic Scalar Hot-Path Signatures

**Goal**: Targeted f64 hot-path signatures are generalized over the scalar type via defaulted type params (`T = f64`), proven non-breaking at compile time — the enabling substrate the DOP families are written against.
**Depends on**: Phase 94 (both autodiff scalar types — `Dual` and reverse-mode `Var` — should be the concrete types the generalized signatures are validated to flow)
**Requirements**: GEN-01
**Success Criteria** (what must be TRUE):

  1. The targeted hot-path signatures accept a scalar type parameter defaulted to `f64` (`T = f64`), so unannotated call sites resolve exactly as before.
  2. Every existing f64 call site inside `fdars-core` compiles unchanged (no signature churn at call sites).
  3. All 28 examples and the R + WASM binding surfaces compile unchanged against the generalized signatures.
  4. The `--features serde` build stays green (no regression from the generalization).

**Plans**: 3/3 plans executed

- [x] 95-01-tracer-l2-distance-PLAN.md — TRACER: generalize l2_distance to `<T: Scalar = f64>` + parity/Dual/Var tests + confirm l2_distance_matrix & crate compile
- [x] 95-02-remaining-kernels-PLAN.md — generalize trapz, inner_product (`.sum()`→accumulator rewrite), inner_product_l2 in place + parity/Dual/Var tests
- [x] 95-03-non-breaking-compile-gate-PLAN.md — GEN-01 compile-gate: 28 examples + serde + wasm + clippy --all-targets + full test + doctests + churn diff + fdars-r grep

### Phase 96: Differentiable Basis Evaluation & Inner Products

**Goal**: Basis evaluation (B-spline / Fourier) and functional inner products are generic over `Scalar` and differentiable, with f64 numerics preserved (family 1 of DIF-F2).
**Depends on**: Phase 95 (needs the generic hot-path substrate)
**Requirements**: DOP-01
**Success Criteria** (what must be TRUE):

  1. Basis evaluation (B-spline and Fourier) is generic over `Scalar`; at `f64` it reproduces the current numerics bit-for-bit (or within documented tolerance).
  2. Functional inner products are generic over `Scalar` and differentiable through their inputs.
  3. Gradients of a basis-eval / inner-product objective match finite differences within tolerance at both `Dual` and reverse-mode `Var`.

**Plans**: TBD

### Phase 97: Differentiable Regression Prediction & Smoothing Penalties

**Goal**: Scalar-on-function regression prediction (`fregre_lm`/FPCR path) and smoothing/roughness-penalty evaluation are generic over `Scalar` and differentiable w.r.t. inputs, with f64 parity preserved (families 2 and 3 of DIF-F2).
**Depends on**: Phase 96 (regression prediction composes basis eval / inner products / FPCA projection)
**Requirements**: DOP-02, DOP-03
**Success Criteria** (what must be TRUE):

  1. Scalar-on-function regression prediction (`fregre_lm` / FPCR path) is generic over `Scalar` and differentiable w.r.t. inputs; f64 predictions are unchanged.
  2. Smoothing / roughness-penalty evaluation is generic over `Scalar` and differentiable (penalty w.r.t. curve values / smoothing inputs); f64 penalty values are unchanged.
  3. Gradients of both the prediction and the penalty match central finite differences within tolerance.

**Plans**: TBD

### Phase 98: Differentiable Depth & Curve Distances

**Goal**: Functional depth measures and curve distances (beyond the existing soft-DTW) are generic over `Scalar` and differentiable, with f64 parity preserved (family 4 of DIF-F2).
**Depends on**: Phase 95 (needs the generic hot-path substrate; independent of Phases 96–97)
**Requirements**: DOP-04
**Success Criteria** (what must be TRUE):

  1. At least one functional depth measure is generic over `Scalar` and differentiable w.r.t. curve values; f64 depth values are unchanged.
  2. At least one curve distance beyond soft-DTW is generic over `Scalar` and differentiable.
  3. Gradients of the depth and distance paths match central finite differences within tolerance at both `Dual` and reverse-mode `Var`.

**Plans**: TBD

### Phase 99: End-to-End Autodiff Flow & Gradient API

**Goal**: Autodiff types flow through the generalized hot-paths end-to-end into a composed scalar objective, exposed through an ergonomic unified gradient API with a worked, finite-difference-checked composition demo.
**Depends on**: Phases 96, 97, 98 (composition demo exercises the broadened differentiable subset); Phase 94 (uses the `vjp` entry point)
**Requirements**: GEN-02, API-01
**Success Criteria** (what must be TRUE):

  1. Both autodiff types (`Dual` and reverse-mode `Var`) flow through the generalized hot-paths end-to-end; a composed objective built from the broadened subset yields correct gradients validated by finite differences.
  2. Ergonomic gradient entry points (`grad` / `jacobian` / `vjp`) are exposed and full crate-root + prelude re-exports cover all new public surface.
  3. A worked end-to-end composition demo (composing differentiable ops into a scalar objective and taking its gradient) exists and is finite-difference-checked.
  4. A running module doctest demonstrates the gradient API and passes under `cargo test`.

**Plans**: TBD

### Phase 100: Release Preparation & Verification

**Goal**: fdars-core is release-ready at 0.44.0 with the differentiable-core section of the 1.0 checklist cleared and every whole-crate gate green — the full-suite gate standing as end-to-end proof of the milestone.
**Depends on**: Phases 94, 95, 96, 97, 98, 99 (terminal phase; its full-suite gate proves the whole milestone)
**Requirements**: REL-01
**Success Criteria** (what must be TRUE):

  1. The crate is bumped 0.43.0 → 0.44.0 with a `[0.44.0]` entry in both CHANGELOGs (root + crate).
  2. The differentiable-core items (DIF-F1 / DIF-F2 / DIF-F3) are checked off on `documentation/ROADMAP-TO-1.0.md`.
  3. All whole-crate gates pass: `cargo fmt --check`, `cargo clippy --all-targets --features linalg,parallel -- -D warnings`, full `cargo test`, a `--features serde` build guard, all 28 examples + doctests, and `cargo package`.
  4. The `git tag v0.44.0` → crates.io publish is left as the deferred operator step (never tagged/published inside the phase — `git.create_tag` is off because `release.yml` couples tag-push to publish).

**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 94 → 95 → 96 → 97 → 98 → 99 → 100

**Dependency notes:** Phase 94 (reverse-mode core) is independent-in-principle and could be built in parallel with Phase 95, but RAD-03 validates against the existing (already-differentiable) subset, so it is sequenced first. Phase 95 (generic hot-paths, GEN-01) is the enabler for the DOP families (96/97/98). Phases 96→97 chain (regression composes basis/inner-product/FPCA); Phase 98 (depth/distances) is independent of 96/97 and only needs 95. Phase 99 composes the broadened subset and uses the reverse-mode `vjp` entry point. Phase 100 is terminal — its full-suite gate proves the milestone.

**Milestone gates (additive, non-breaking):** `cargo fmt --check`, `cargo clippy --all-targets --features linalg,parallel -- -D warnings` (CI lints test/bench code — use `--all-targets`), full `cargo test` (the differentiability + non-regression proof), a `--features serde` build guard, all 28 examples + doctests, and `cargo package`. No new crate dependency. Build/CI hazards (from MEMORY.md): run `cargo fmt` per commit (`--no-verify` commits leave fmt drift → CI fmt-check fails); the pre-commit hook runs the full cargo gate and times out — prefer inline execution + `commit --no-verify` after out-of-band gates; watch `/tmp` and `target/` disk pressure on full builds (`rm -rf target/debug/{incremental,examples}` to free space).

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 94. Reverse-Mode Autodiff Core (VJP Tape) | v0.44.0 | 4/4 | Complete    | 2026-09-11 |
| 95. Generic Scalar Hot-Path Signatures | v0.44.0 | 3/3 | In Progress|  |
| 96. Differentiable Basis Evaluation & Inner Products | v0.44.0 | 0/? | Not started | - |
| 97. Differentiable Regression Prediction & Smoothing Penalties | v0.44.0 | 0/? | Not started | - |
| 98. Differentiable Depth & Curve Distances | v0.44.0 | 0/? | Not started | - |
| 99. End-to-End Autodiff Flow & Gradient API | v0.44.0 | 0/? | Not started | - |
| 100. Release Preparation & Verification | v0.44.0 | 0/? | Not started | - |

**Shipped:** v0.14.0 → v0.43.0 (see the Milestones list above and the `milestones/` archives).
