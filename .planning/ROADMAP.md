# Roadmap: fdars

## Milestones

- ✅ **…v0.40.0 Correctness & Release Hardening** — Phases 78-80 (shipped 2026-09-07)
- 🚧 **v0.41.0 1.0 API Stabilization Pass** — Phases 81-85 (in progress)

## Overview

v0.41.0 is fdars' **first breaking milestone** after a long additive-only run — legitimate under 0.x, where semver permits breaking changes. It opens with a whole-surface **audit** that produces a ranked, user-approved breaking-change inventory across four scopes (deprecated-form removal, accidental `pub` exposure, `#[non_exhaustive]` gaps, naming inconsistencies). The concrete change sets for the surface-sealing and naming-unification phases are **drawn from that approved list**, so those execution phases depend on the audit. The mechanical removal of the 6 already-specified deprecated forms rides in the audit phase (it does not depend on the findings). Then per-category execution phases land the breaking cleanups, a non-code phase produces the stability deliverables (semver policy, MSRV, 1.0 gap checklist) that will govern a deliberate future 1.0 cut, and a final release phase bumps 0.40.0 → 0.41.0, writes the CHANGELOG with breaking changes called out, and verifies release-readiness. Ships as **0.41.0** (NOT 1.0). All 28 examples + doctests are the compile-time proof the breaking changes are complete.

## Phases

**Phase Numbering:**
- Integer phases: Planned milestone work (continuing from Phase 80 → **Phase 81 onward**)
- Decimal phases (81.1, 81.2): Urgent insertions (marked with INSERTED)

- [x] **Phase 81: API Audit & Deprecated-Form Removal** - Produce the user-approved ranked breaking-change inventory; remove the 6 deprecated forms
- [x] **Phase 82: Public-Surface Sealing & Non-Exhaustive Coverage** - Seal accidental `pub` exposure and correct `#[non_exhaustive]` per the approved inventory
- [x] **Phase 83: Naming Unification** - Unify `_1d`/`_2d`/`_nd` suffix sprawl + config/result naming into consistent dispatchers (largest, highest-risk)
- [x] **Phase 84: Stability Deliverables** - Semver/stability policy, MSRV finalization, and 1.0 gap checklist (non-code)
- [x] **Phase 85: Release Preparation & Verification** - Bump 0.40.0 → 0.41.0, CHANGELOG with breaking changes, docs refresh, whole-crate gates green

## Phase Details

### Phase 81: API Audit & Deprecated-Form Removal
**Goal**: The full public surface is audited into a ranked, user-approved breaking-change inventory, and the 6 already-specified deprecated forms are removed from the crate.
**Depends on**: Nothing (first phase)
**Requirements**: AUDIT-01, API-01
**Success Criteria** (what must be TRUE):
  1. A ranked breaking-change inventory exists covering all four scopes (deprecated-form removal, accidental `pub` exposure, `#[non_exhaustive]` gaps, naming inconsistencies), each entry listing location, proposed change, blast radius (internal callers, examples, doctests), and a value/risk rating.
  2. The inventory has been presented to and approved by the user — the concrete change sets consumed by Phases 82 and 83 are drawn from this approved list (the user may approve a reduced scope, especially for naming).
  3. The 6 deprecated forms (`mean_2d`, `fanova`, `random_tukey_2d`, `random_projection_2d`, `fraiman_muniz_2d`, `modal_2d`) and their crate-root/prelude re-exports are removed; all internal callers, unit tests, doctests, and example 21 are migrated to the `Dim`/`_seeded` replacements.
  4. The crate builds and all 28 examples compile after the deprecated-form removal; whole-crate gates (fmt, `clippy --all-targets --features linalg,parallel -- -D warnings`, `cargo test`, `--features serde` build) are green.
**Plans**: 2 plans

Plans:
- [ ] 81-01-PLAN.md — Whole-surface audit → ranked, ID-addressable breaking-change inventory (four scopes) → user-approval checkpoint (reduced scope allowed)
- [ ] 81-02-PLAN.md — Hard-remove the 6 deprecated forms + all re-exports; migrate internal callers/doctests/tests/example-21 to `Dim`/`_seeded`; whole-crate gates green

### Phase 82: Public-Surface Sealing & Non-Exhaustive Coverage
**Goal**: The accidental/unintended public exposure and `#[non_exhaustive]` gaps identified in the approved AUDIT-01 inventory are corrected, shrinking and future-proofing what a 1.0 must commit to.
**Depends on**: Phase 81 (change sets drawn from the approved inventory)
**Requirements**: API-02, API-03
**Success Criteria** (what must be TRUE):
  1. Accidental/unintended public exposure from the approved inventory is sealed (`pub` → `pub(crate)`, removed re-exports, hidden leaked helper types); the public surface compiles and no in-crate, example, test, or doctest usage breaks.
  2. `#[non_exhaustive]` coverage on public enums and result structs is corrected per the approved inventory so future field/variant additions stay non-breaking after 1.0.
  3. No numeric or behavioral output changes — the edits are visibility/attribute-only.
  4. Whole-crate gates (fmt, `clippy --all-targets --features linalg,parallel -- -D warnings`, `cargo test`, `--features serde` build) are green and all 28 examples compile.
**Plans**: 2 plans

Plans:
- [x] 82-01: Seal accidental `pub` exposure per approved inventory (API-02)
- [x] 82-02: Correct `#[non_exhaustive]` coverage on public enums/result structs (API-03)

### Phase 83: Naming Unification
**Goal**: The approved naming unification is applied — `_1d`/`_2d`/`_nd` suffix sprawl and config/result naming collapse into consistent dispatchers across the whole crate.
**Depends on**: Phase 82 (surface sealed first; scope set by the approved inventory)
**Requirements**: API-04
**Success Criteria** (what must be TRUE):
  1. The approved `_1d`/`_2d`/`_nd` suffix + config/result naming changes are applied and the surface exposes consistent dispatchers (scope as approved in AUDIT-01 — possibly reduced).
  2. All call sites, all 28 examples, and all doctests are updated to the new names; the crate compiles.
  3. The whole-crate test suite passes with no numeric/behavioral change — the breaking is limited to API shape.
  4. Whole-crate gates (fmt, `clippy --all-targets --features linalg,parallel -- -D warnings`, `cargo test`, `--features serde` build) are green.
**Plans**: 3 plans

Plans:
- [ ] 83-01-PLAN.md — AUD-14/15/16 simple hard renames (funhddC_cluster→fun_hddc_cluster, FosrResult2d→Fosr2dResult, GmmResult→GmmFitResult)
- [ ] 83-02-PLAN.md — AUD-17 deriv grid-enum collapse (deriv_1d/deriv_2d → deriv via DerivDomain/DerivResult)
- [ ] 83-03-PLAN.md — AUD-18 lp grid-enum collapse (lp_self/cross_1d/2d → lp_self/lp_cross via LpDomain)

### Phase 84: Stability Deliverables
**Goal**: The non-code stability deliverables that will govern the eventual 1.0 cut exist — semver/stability policy, a finalized MSRV policy, and a 1.0 gap checklist.
**Depends on**: Phase 81 (approved inventory informs the gap checklist; can run alongside 82/83)
**Requirements**: STAB-01, STAB-02, STAB-03
**Success Criteria** (what must be TRUE):
  1. A documented semver + API-stability policy exists in `documentation/` — the definition of "stable", the deprecation process, and the breaking-change policy governing the eventual 1.0.
  2. The MSRV policy is reviewed, pinned (1.81 crate / 1.84 for the `linalg` feature), documented, and consistent between `Cargo.toml` and the docs.
  3. A 1.0 gap checklist enumerating what remains before a real 1.0 cut (items deferred out of this milestone) exists and scopes the next milestone.
**Plans**: 1 plan

Plans:
- [ ] 84-01-PLAN.md — Author semver/stability policy (STABILITY.md, STAB-01/02) + 1.0 gap checklist (ROADMAP-TO-1.0.md, STAB-03) in `documentation/`

### Phase 85: Release Preparation & Verification
**Goal**: The crate is bumped to 0.41.0, the CHANGELOG documents the breaking changes, docs are refreshed, and whole-crate release-readiness is verified (the operator tag/publish is the final external step).
**Depends on**: Phase 82, Phase 83, Phase 84 (validates and folds in all breaking changes + deliverables)
**Requirements**: REL-01
**Success Criteria** (what must be TRUE):
  1. `fdars-core/Cargo.toml` is bumped 0.40.0 → 0.41.0 and a CHANGELOG `[0.41.0]` entry exists with the breaking changes (deprecated-form removal, surface sealing, non-exhaustive corrections, naming unification) explicitly called out.
  2. Docs are refreshed to the new surface; whole-crate gates are green: `cargo fmt --check`, `cargo clippy --all-targets --features linalg,parallel -- -D warnings`, `cargo test`, and a `--features serde` build.
  3. All 28 examples + doctests compile/pass against the new surface — the compile-time proof the breaking changes are complete.
  4. Release-readiness is verified and documented; the `git tag v0.41.0` push → crates.io publish is left as the final operator-driven step (this phase does not tag/publish).
**Plans**: 1 plan

Plans:
- [ ] 85-01-PLAN.md — Bump version 0.40.0→0.41.0 + CHANGELOG `[0.41.0]` breaking entry + docs refresh + whole-crate release-readiness verification (no tag/publish)

## Progress

**Execution Order:**
Phases execute in numeric order: 81 → 82 → 83 → 84 → 85. Phase 81 (audit + approval) must land first — Phases 82/83 draw their change sets from its approved inventory. Phase 84 (non-code deliverables) can run alongside 82/83 once the inventory is approved. Phase 85 lands last.

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 81. API Audit & Deprecated-Form Removal | 2/2 | ✓ Complete | 2026-09-07 |
| 82. Public-Surface Sealing & Non-Exhaustive Coverage | 2/2 | ✓ Complete | 2026-09-07 |
| 83. Naming Unification | 3/3 | ✓ Complete | 2026-09-07 |
| 84. Stability Deliverables | 1/1 | ✓ Complete | 2026-09-07 |
| 85. Release Preparation & Verification | 1/1 | ✓ Complete | 2026-09-07 |
