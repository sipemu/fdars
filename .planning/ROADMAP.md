# Roadmap: fdars

## Milestones

- ✅ **v0.40.0 Correctness & Release Hardening** — Phases 78–80 (shipped 2026-09-07)
- ✅ **v0.41.0 1.0 API Stabilization Pass** — Phases 81–85 (shipped 2026-09-07)
- ✅ **v0.42.0 1.0 API Finalization** — Phases 86–89 (shipped 2026-09-08)

## Phases

_No active milestone. Start the next with `/gsd-new-milestone` (questioning → research → requirements → roadmap). Phase numbering continues from 89._

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

---

_Next milestone starts with `/gsd-new-milestone` (questioning → research → requirements → roadmap). Phase numbering continues from 89._
