# Roadmap: fdars

## Milestones

- ✅ **v0.40.0 Correctness & Release Hardening** — Phases 78–80 (shipped 2026-09-07)
- ✅ **v0.41.0 1.0 API Stabilization Pass** — Phases 81–85 (shipped 2026-09-07)
- ✅ **v0.42.0 1.0 API Finalization** — Phases 86–89 (shipped 2026-09-08)
- 🚧 **v0.43.0 Test Determinism & Release Hardening** — Phases 90–93 (in progress)

## Phases

### 🚧 v0.43.0 Test Determinism & Release Hardening (In Progress)

**Milestone Goal:** Make `cargo test` reliably deterministic under full parallel runs, harden CI so the determinism can't silently regress, and prepare a release (0.43.0, superseding the unpublished 0.41.0/0.42.0) — clearing the **Quality** blocker on `documentation/ROADMAP-TO-1.0.md`. Investigate-first: root-cause the known golden-test flake before fixing it, reuse that technique in a suite-wide sweep, lock in the green baseline with a CI guardrail, then prepare + verify release-readiness last (the full-suite gate is the proof the flake is gone). Implementation milestone — mostly `tests/` + CI config (possibly minor `src/` for determinism), **additive/non-breaking** (protects R + WASM bindings + 28 examples). No new crate dependency UNLESS the flake fix genuinely needs one (dev-only; `serial_test`/nextest — decided in Phase 90). Phase numbering continues from v0.42.0 (ended at 89) → **Phase 90**.

**Phase Numbering:** Integer phases are planned work; decimal phases (e.g., 90.1) are urgent insertions in numeric order. Numbering never resets.

- [x] **Phase 90: Golden-Flake Root-Cause & Deterministic Fix** - Diagnose why the three golden tests flake under full parallel `cargo test`, then fix them deterministically. (completed 2026-09-09)
- [ ] **Phase 91: Suite-Wide Robustness Sweep** - Audit the whole suite for other fragile/nondeterministic/env-dependent assertions and fix or justify each.
- [ ] **Phase 92: CI Determinism Guardrail** - Add a CI gate exercising the full parallel `cargo test` path so a determinism regression fails CI.
- [ ] **Phase 93: Release Preparation & Readiness Verification** - Bump to 0.43.0, CHANGELOG/docs, clear the Quality checklist item, and verify all release gates green.

#### Phase 90: Golden-Flake Root-Cause & Deterministic Fix

**Goal**: The three known flaky golden tests (`golden_co_cluster_parallel`, `golden_co_cluster_below_threshold` in `equivalence_phase48`; `svd_sign_fpca_two_matrix_bit_identical` in `equivalence_phase49`) pass reliably under repeated full parallel `cargo test`, backed by an evidence-based diagnosis of the root cause.
**Depends on**: Nothing (first phase of this milestone)
**Requirements**: FLAKE-01, FLAKE-02
**Success Criteria** (what must be TRUE):

  1. An artifact records an evidence-backed root-cause diagnosis (BLAS/thread scheduling vs. SVD sign vs. test ordering vs. disk pressure) distinguishing why the tests pass per-binary but flake under full parallel `cargo test`.
  2. The fix approach (tolerance-comparison vs. serialization) is chosen from and justified by that diagnosis, not guessed.
  3. The three affected tests pass across repeated full parallel `cargo test` runs (no flake reproduced over multiple consecutive runs).
  4. If any dependency was added to achieve determinism, it is dev-only and its necessity is justified against the diagnosis; otherwise no new dependency was introduced.

**Plans**: 1 plan

- [x] 90-01-PLAN.md — Diagnose the golden-test flake (reproduce + reconcile intermittent-vs-deterministic + commit 90-DIAGNOSIS.md), then apply the evidence-chosen cfg-guard fix and prove it (10 consecutive green full-parallel runs + per-binary green)

Notes: FLAKE-01's diagnosis gates FLAKE-02's fix approach — they share this phase because the fix design is wholly determined by the evidence. The flake has stood since at least v0.41.0 (logged on the 1.0 checklist); per MEMORY.md it fails ONLY under full `cargo test`, passing per-binary in isolation — treat as an environment/cross-binary-interference flake, not a numeric regression. No new crate dependency UNLESS the fix genuinely needs one (`serial_test` / nextest serialization — decided here); if added it must be dev-only. Build hazards: `/tmp` tmpfs is small (doctests link there); `target/` can fill `/home` (`rm -rf target/debug/{incremental,examples}` to free space); prefer inline execution + `commit --no-verify` after out-of-band gates (the pre-commit hook runs the full cargo gate and times out).

#### Phase 91: Suite-Wide Robustness Sweep

**Goal**: Beyond the three known tests, the rest of the suite is audited for analogous fragility and every additional fragile test is either fixed or documented as safe-to-leave, so the whole suite is reliably green under full parallel runs.
**Depends on**: Phase 90
**Requirements**: ROBUST-01, ROBUST-02
**Success Criteria** (what must be TRUE):

  1. A findings list enumerates every other fragile bit-identity, nondeterministic, or environment/BLAS/disk-dependent assertion discovered in the suite (or records that none were found).
  2. Each additional fragile test found is either fixed deterministically or has a documented justification for why it is safe to leave as-is.
  3. The full `cargo test` suite passes under full parallel runs with no residual flake attributable to the swept categories.

**Plans**: TBD

Notes: Reuses the diagnosis technique established in Phase 90; sequenced after the golden-flake fix so the known case informs what to look for. Same build/disk hazards as Phase 90 apply. Additive/non-breaking — test + possibly minor `src/` determinism changes only.

#### Phase 92: CI Determinism Guardrail

**Goal**: A CI guardrail exercises the full parallel `cargo test` path (cross-binary interference) and/or a nextest serialization group, so any future determinism regression fails CI loudly instead of silently returning.
**Depends on**: Phase 91
**Requirements**: CI-01
**Success Criteria** (what must be TRUE):

  1. CI runs a job that exercises the full parallel `cargo test` path (the cross-binary-interference condition that triggered the original flake) and/or a nextest serialization group.
  2. The guardrail passes against the now-green baseline from Phases 90–91.
  3. A reintroduced determinism regression would fail this CI gate rather than pass silently (the gate targets the specific failure mode diagnosed in Phase 90).

**Plans**: TBD

Notes: Lands after the fixes exist so the gate reflects a green baseline (a guardrail added before the fix would start red). CI config change (`.github/workflows/`); if a nextest serialization group is used, its config must align with whatever dependency decision was made in Phase 90. Clippy in CI uses `--all-targets --features linalg,parallel -- -D warnings` (lints test/bench code).

#### Phase 93: Release Preparation & Readiness Verification

**Goal**: fdars-core is release-ready at 0.43.0 — version bumped, CHANGELOG/docs written, the Quality item on the 1.0 checklist cleared, and every whole-crate gate green — with the full-suite gate standing as the proof the flake is fixed. The `git tag v0.43.0` push → crates.io publish is left as the deferred operator step.
**Depends on**: Phase 90, Phase 91, Phase 92
**Requirements**: REL-01, REL-02
**Success Criteria** (what must be TRUE):

  1. `Cargo.toml` is bumped 0.42.0 → 0.43.0, a CHANGELOG `[0.43.0]` entry (root + crate-shipped) describes the determinism fix + CI guardrail, and docs are refreshed.
  2. The **Quality** item in `documentation/ROADMAP-TO-1.0.md` is checked off.
  3. All release gates pass green: `cargo fmt --check`, `cargo clippy --all-targets --features linalg,parallel -- -D warnings`, full `cargo test` (the determinism proof), `--features serde` build, all 28 examples + doctests, and `cargo package`.
  4. The CHANGELOG/notes frame 0.43.0 as superseding the unpublished 0.41.0/0.42.0 (registry still at 0.40.0 — one publish catches it up), and the operator-driven `git tag v0.43.0` → crates.io publish is documented as the remaining step (not performed in any phase).

**Plans**: TBD

Notes: **Must land last** — REL-02's full-suite gate is the evidence the flake fix holds end-to-end. Per project convention the `git tag v0.43.0` push auto-publishes via `release.yml`, so tagging/publishing is deliberately NOT done inside a phase (GSD `git.create_tag` is off for this repo). Run `cargo fmt` per commit to avoid the `--no-verify` fmt-drift trap; keep the `--features serde` build green (repaired in v0.40.0 — do not regress). Watch `/tmp` and `target/` disk pressure on the full gate run.

---

<details>
<summary>✅ v0.42.0 1.0 API Finalization (Phases 86–89) — SHIPPED 2026-09-08</summary>

The second breaking milestone — clears the entire **API section** of the 1.0 gap checklist (`documentation/ROADMAP-TO-1.0.md`): sealed `wire` `pub(crate)` (SEAL-01), `#[non_exhaustive]` on 38 config structs + the construction-idiom change (SEAL-02/03), and the full `_1d`/`_2d` → `Dim`-dispatch naming unification — `geometric_median`/`hausdorff_*`/`functional_spatial_*` + `LpeerResult`→`LocalPeerResult` (NAME-01/02/03/05) and the large batch of 41 consolidations + 5 plain-renames (NAME-04). API-shape-only — no numeric/behavioral change (byte-identical `_impl` bodies, code-review confirmed; 2857-test non-regression). Full detail: [`milestones/v0.42.0-ROADMAP.md`](milestones/v0.42.0-ROADMAP.md); archived phase artifacts: `milestones/v0.42.0-phases/`.

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

## Progress

**Execution Order:**
v0.43.0 phases execute in numeric order: 90 → 91 → 92 → 93

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 90. Golden-Flake Root-Cause & Deterministic Fix | v0.43.0 | 1/1 | Complete    | 2026-09-09 |
| 91. Suite-Wide Robustness Sweep | v0.43.0 | 0/TBD | Not started | - |
| 92. CI Determinism Guardrail | v0.43.0 | 0/TBD | Not started | - |
| 93. Release Preparation & Readiness Verification | v0.43.0 | 0/TBD | Not started | - |
