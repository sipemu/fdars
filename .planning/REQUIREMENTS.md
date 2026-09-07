# Requirements: fdars — v0.41.0 1.0 API Stabilization Pass

**Defined:** 2026-09-07
**Core Value:** A comprehensive, fast Rust functional-data-analysis library. This milestone is a **1.0-readiness pass**: audit the whole public surface, then land the breaking cleanups now while still in 0.x — producing a settled API and the stability deliverables that will govern a deliberate future 1.0 cut. Ships as **0.41.0** (not 1.0).

## Milestone Scope

First **breaking** milestone after a long additive-only run — legitimate because the crate is still in 0.x, where breaking changes are permitted under semver. The milestone opens with an **audit/inventory phase** whose ranked breaking-change list is **approved before any execution**; API-01 (deprecated-form removal) is already fully specified and does not depend on the audit's findings. No new crate dependency. Workspace is `fdars-core` only — the external `fdars-r` CRAN package is the R maintainer's concern (its `FdMatrix` migration, issue `fdars-j75`, stays a separate todo). All 28 examples + doctests are updated to the new surface — they are the compile-time proof the breaking changes are complete.

## v1 Requirements

### API Audit

- [ ] **AUDIT-01**: A ranked public-API breaking-change inventory is produced covering all four scopes — deprecated-form removal, accidental `pub`-surface exposure, `#[non_exhaustive]` gaps, and naming inconsistencies (`_1d`/`_2d`/`_nd` sprawl + config/result patterns). Each entry lists location, proposed change, blast radius (internal callers, examples, doctests), and a value/risk rating. The inventory is presented for approval; the concrete API-02/03/04 change sets are drawn from the approved list.

### Breaking API Cleanup

- [ ] **API-01**: The 6 deprecated forms (`mean_2d`, `fanova`, `random_tukey_2d`, `random_projection_2d`, `fraiman_muniz_2d`, `modal_2d`) are removed, along with their crate-root/prelude re-exports. All internal callers, unit tests, doctests, and example 21 are migrated to the `Dim`/`_seeded` replacements. The crate and all 28 examples compile.
- [ ] **API-02**: Accidental/unintended public exposure identified in AUDIT-01 is sealed (`pub` → `pub(crate)`, removed re-exports, hidden leaked helper types). The public surface compiles and no in-crate, example, test, or doctest usage breaks.
- [ ] **API-03**: `#[non_exhaustive]` coverage on public enums and result structs is corrected per AUDIT-01, so future field/variant additions remain non-breaking after 1.0.
- [ ] **API-04**: The approved naming unification is applied — `_1d`/`_2d`/`_nd` suffix sprawl and config/result naming unified into consistent dispatchers. All call sites, the 28 examples, and doctests are updated. The crate compiles and the whole-crate test suite passes.

### Stability Deliverables

- [ ] **STAB-01**: A documented semver + API-stability policy exists in `documentation/` — the definition of "stable", the deprecation process, and the breaking-change policy that will govern the eventual 1.0.
- [ ] **STAB-02**: The MSRV policy is reviewed and pinned (1.81 crate / 1.84 for the `linalg` feature), documented, and consistent between `Cargo.toml` and the docs.
- [ ] **STAB-03**: A 1.0 gap checklist enumerating what remains before a real 1.0 cut (items deferred out of this milestone) is produced, scoping the next milestone.

### Release

- [ ] **REL-01**: Crate bumped 0.40.0 → 0.41.0; CHANGELOG `[0.41.0]` entry added with breaking changes explicitly called out; docs refreshed. Whole-crate gates green: `cargo fmt --check`, `cargo clippy --all-targets --features linalg,parallel -- -D warnings`, `cargo test`, and a `--features serde` build. The `git tag v0.41.0` push → crates.io publish is the final operator-driven step (the phase prepares + verifies release-readiness).

## Future Requirements

Deferred to a later, deliberate **1.0 cut** (governed by STAB-01/03):

- **1.0-CUT**: Bump to 1.0.0 and declare the public API stable, once the STAB-03 gap checklist is cleared.
- **fdars-r FdMatrix migration** (issue `fdars-j75`): migrate the external R wrapper to the `FdMatrix` API — separate package, out of `fdars-core` scope.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Cutting 1.0.0 this milestone | Deliberate decision: settle the API under 0.x first; 1.0 is a separate, governed commitment (STAB-01/03) |
| New algorithms / features | This is a stabilization pass — no net-new capability; all parity/gap backlogs remain exhausted |
| Changes to external `fdars-r` | External CRAN package, R maintainer's concern; its `FdMatrix` migration is a separate todo (`fdars-j75`) |
| New crate dependency | Carried convention — the cleanup reuses existing machinery |
| Behavior/numeric changes | Breaking is limited to API shape (names, visibility, exhaustiveness); numeric outputs unchanged |

## Traceability

Filled during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| AUDIT-01 | TBD | Pending |
| API-01 | TBD | Pending |
| API-02 | TBD | Pending |
| API-03 | TBD | Pending |
| API-04 | TBD | Pending |
| STAB-01 | TBD | Pending |
| STAB-02 | TBD | Pending |
| STAB-03 | TBD | Pending |
| REL-01 | TBD | Pending |

**Coverage:**
- v1 requirements: 9 total
- Mapped to phases: 0 (roadmap pending)
- Unmapped: 9 ⚠️

---
*Requirements defined: 2026-09-07*
*Last updated: 2026-09-07 after initial definition*
