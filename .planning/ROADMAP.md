# Roadmap: fdars

## Milestones

- ✅ **v0.40.0 Correctness & Release Hardening** — Phases 78–80 (shipped 2026-09-07)
- ✅ **v0.41.0 1.0 API Stabilization Pass** — Phases 81–85 (shipped 2026-09-07)
- 🔵 **v0.42.0 1.0 API Finalization** — Phases 86–89 (active)

## Phases

- [x] **Phase 86: Surface Sealing** - Seal the `wire` module `pub(crate)` and mark every public config struct `#[non_exhaustive]` with a `Default`/builder construction escape hatch (completed 2026-09-08)
- [x] **Phase 87: Targeted Renames** - Consolidate the small `Dim`-dispatch families (`geometric_median`, `hausdorff_*`, `functional_spatial_*`) and rename `LpeerResult` → `LocalPeerResult` (completed 2026-09-08)
- [ ] **Phase 88: Large Suffix Batch** - Consolidate the large remaining lone-`_1d`/`_2d` suffix functions onto `Dim` dispatch and update all 28 examples + docs (highest blast radius)
- [ ] **Phase 89: Release Preparation & Verification** - Bump 0.41.0 → 0.42.0, breaking-framed CHANGELOG, whole-crate gates green, ROADMAP-TO-1.0.md API items checked off

## Phase Details

### Phase 86: Surface Sealing

**Goal**: The public interchange (`wire`) surface is removed and every public config struct is future-proofed against field additions without breaking external construction.
**Depends on**: Nothing (first phase of milestone; builds on the shipped v0.41.0 surface)
**Requirements**: SEAL-01, SEAL-02, SEAL-03
**Success Criteria** (what must be TRUE):

  1. External callers can no longer name any `wire` type (`pub mod wire` is `pub(crate)`, and all crate-root/prelude re-exports of wire types are gone).
  2. Every public config struct not already sealed carries `#[non_exhaustive]`, so fields can be added post-1.0 without a breaking change.
  3. Every newly-sealed config struct can still be constructed by external code via a documented `Default` + `..Default::default()` path (and/or builder) — no `Config { .. }` literal is the only way in.
  4. The crate, all 28 examples, and all doctests compile with the sealed surfaces; `cargo build --features serde` still compiles.

**Plans**: 1 plan

- [x] 86-01-PLAN.md — Seal `wire` pub(crate) + doctest fix (SEAL-01); mark 38 config structs `#[non_exhaustive]` with documented `Default` path (SEAL-02/03); full-gate verification

### Phase 87: Targeted Renames

**Goal**: The small, low-blast-radius naming inconsistencies are resolved — spatial/median families collapse onto single `Dim`-dispatched signatures and the PEER result type name matches its sibling.
**Depends on**: Phase 86
**Requirements**: NAME-01, NAME-02, NAME-03, NAME-05
**Success Criteria** (what must be TRUE):

  1. `geometric_median` is callable through one `Dim`-dispatched signature (no lone `_1d`/`_2d` public forms), routing to byte-identical private `_impl` bodies.
  2. The `hausdorff_*` family and the `functional_spatial_*` / `kernel_functional_spatial_*` families are each callable through one `Dim`-dispatched signature, routing to byte-identical private `_impl` bodies.
  3. The result type is named `LocalPeerResult` (not `LpeerResult`) everywhere — definition, all references, `lib.rs`/`prelude.rs` re-exports, examples, and doctests.
  4. The crate, all 28 examples, and all doctests compile against the consolidated signatures; a code-review gate confirms no numeric drift.

**Plans**: 1 plan

- [x] 87-01-PLAN.md — NAME-05 LocalPeerResult rename (tracer) + NAME-01/02/03 Dim-dispatch consolidations + all ~35 caller updates and full gate set

### Phase 88: Large Suffix Batch

**Goal**: The remaining crate-wide lone-`_1d`/`_2d` suffix sprawl is unified onto `Dim` dispatch and the entire example/doc surface is migrated to the new signatures.
**Depends on**: Phase 87
**Requirements**: NAME-04
**Success Criteria** (what must be TRUE):

  1. The large remaining batch of lone-`_1d`/`_2d` suffix functions is callable through a single `Dim`-dispatched signature per family, routing to byte-identical private `_impl` bodies.
  2. All 28 examples and all docs/doctests are updated to the new surface and compile.
  3. The whole crate compiles with no lingering references to the removed lone-suffix public names; a code-review gate confirms no numeric drift.

**Plans**: 1 plan

- [ ] 88-01-PLAN.md — Consolidate all in-scope lone-`_1d`/`_2d` public functions onto `Dim` dispatch (Cat 1/2/3) or plain renames, migrate all 28 examples + tests + benches + doctests, full gate set

### Phase 89: Release Preparation & Verification

**Goal**: The crate is version-bumped, the breaking API changes are documented, every gate is green, and the 1.0 checklist reflects the cleared API items — release-ready for the operator to tag and publish.
**Depends on**: Phase 86, Phase 87, Phase 88
**Requirements**: REL-01
**Success Criteria** (what must be TRUE):

  1. The crate version is bumped 0.41.0 → 0.42.0 with a breaking-framed `[0.42.0]` entry (root + crate-shipped CHANGELOG) explicitly calling out the sealed `wire`/config surfaces and the renames.
  2. Whole-crate gates are green: `cargo fmt --check`, `cargo clippy --all-targets --features linalg,parallel -- -D warnings`, `cargo test`, a `--features serde` build, all 28 examples + doctests, and `cargo package`.
  3. `documentation/ROADMAP-TO-1.0.md` is updated to check off the cleared API items (AUD-09/12/13/19–23).
  4. Release-readiness is prepared and verified only — the `git tag v0.42.0` push → crates.io publish is left as the deferred operator step.

**Plans**: TBD

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 86. Surface Sealing | 1/1 | Complete    | 2026-09-08 |
| 87. Targeted Renames | 1/1 | Complete    | 2026-09-08 |
| 88. Large Suffix Batch | 0/? | Not started | - |
| 89. Release Preparation & Verification | 0/? | Not started | - |

**Execution order:** 86 → 87 → 88 → 89. 86 first (independent sealing). 87 then 88 sequence the naming work smallest-first, isolating the high-blast-radius suffix batch. 89 last (depends on all sealing + naming phases). All 9 requirements mapped, no orphans, no duplicates.

**Gates (this breaking, API-shape-only milestone):** `cargo fmt --check`, `cargo clippy --all-targets --features linalg,parallel -- -D warnings`, `cargo test`, plus a `--features serde` build guard. No new crate dependency; `fdars-core` only. All 28 examples + doctests must compile — the compile-time proof each breaking change is complete. Every consolidated signature routes to a byte-identical private `_impl` body (no numeric/behavioral change), confirmed by a code-review gate.

---

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

---

_Next milestone starts with `/gsd-new-milestone` (questioning → research → requirements → roadmap). Phase numbering continues from 89._
