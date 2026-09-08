---
phase: 86-surface-sealing
plan: "01"
subsystem: api
tags: [rust, non_exhaustive, visibility, api-sealing, config-structs, breaking-change]

requires: []
provides:
  - "wire module sealed pub(crate) — ~24 interchange types no longer in the public API"
  - "#[non_exhaustive] on all 38 previously-unsealed public *Config structs (66 total, 28 already sealed)"
  - "documented Default-based construction escape hatch on every newly-sealed config"
  - "all external-crate construction sites (doctests, examples, integration tests, benches) migrated to default() + field-assignment"
affects: [87-targeted-renames, 88-large-suffix-batch, 89-release-preparation]

actuals:
  tokens: 60000
  tasks: 3
  commits: 1

tech-stack:
  added: []
  patterns:
    - "External construction of #[non_exhaustive] configs uses Config::default() + field assignment (struct literals, even ..Default::default(), are E0639 outside the crate)"
    - "Retained-but-internal module gets a documented module-level #![allow(dead_code)] instead of per-item allows"

key-files:
  created: []
  modified:
    - "fdars-core/src/lib.rs (pub mod wire → pub(crate) mod wire)"
    - "fdars-core/src/wire.rs (module doctest → ignore; #![allow(dead_code)])"
    - "38 domain module files (#[non_exhaustive] + Default-path doc on each config)"
    - "fdars-core/examples/27_spm/main.rs, tests/validate_spm_math.rs, 2 benches (construction-site migration)"

key-decisions:
  - "Wire module doctest marked `ignore` (not `use crate::wire`) — its types are pub(crate), so it can never compile as an external doctest; ignore is the only correct choice."
  - "Added module-level #![allow(dead_code)] to wire.rs — sealing made its 24 retained interchange types dead code; documents the intentional retain-for-future-bindings decision."
  - "SEAL-03 doc wording corrected to the accurate idiom: default() + field assignment, NOT ..Default::default() (which E0639 rejects externally)."

patterns-established:
  - "non_exhaustive external construction: `let mut c = T::default(); c.field = v;` (block-expr form `{ let mut c = T::default(); c.f = v; c }` when inline)."

requirements-completed: [SEAL-01, SEAL-02, SEAL-03]

coverage:
  - id: D1
    description: "wire module sealed pub(crate); external callers can no longer name any wire type; zero re-exports; crate + examples compile"
    requirement: "SEAL-01"
    verification:
      - kind: integration
        ref: "cargo clippy --all-targets --features linalg,parallel -- -D warnings"
        status: pass
      - kind: integration
        ref: "cargo build --examples"
        status: pass
    human_judgment: false
  - id: D2
    description: "all 38 unsealed public *Config structs carry #[non_exhaustive] (28 pre-existing untouched)"
    requirement: "SEAL-02"
    verification:
      - kind: integration
        ref: "cargo test --features linalg,parallel (unit+integration, incl. validate_spm_math 34/0)"
        status: pass
    human_judgment: false
  - id: D3
    description: "each newly-sealed config is externally constructible via a documented Default path; doctests demonstrating it pass"
    requirement: "SEAL-03"
    verification:
      - kind: unit
        ref: "cargo test --doc --features linalg,parallel (208 passed, 0 failed, 5 ignored)"
        status: pass
    human_judgment: false
  - id: D4
    description: "crate + 28 examples + doctests + serde build all compile with sealed surfaces (success criterion #4)"
    requirement: "SEAL-01"
    verification:
      - kind: integration
        ref: "cargo build --features serde"
        status: pass
      - kind: integration
        ref: "cargo fmt --check"
        status: pass
    human_judgment: false

duration: 55min
completed: 2026-09-08
status: complete
---

# Phase 86: Surface Sealing Summary

**`wire` sealed `pub(crate)` and all 38 unsealed public `*Config` structs marked `#[non_exhaustive]`, with every external construction site migrated to the `Default`-based escape hatch — a pure API-shape-only breaking change with zero behavioral drift.**

## Performance

- **Duration:** ~55 min
- **Tasks:** 3 (A: wire seal, B: 38× non_exhaustive + SEAL-03 docs, C: full gate verification)
- **Files modified:** 41 source files (+ 2 planning docs)

## Accomplishments
- **SEAL-01:** `pub mod wire` → `pub(crate) mod wire`. Zero re-exports existed to remove; zero examples/tests referenced wire. The module doctest was marked `ignore` (its types are now crate-internal) and a documented `#![allow(dead_code)]` added so the retained-but-internal interchange surface doesn't trip `-D warnings`.
- **SEAL-02:** `#[non_exhaustive]` added to all 38 previously-unsealed public `*Config` structs (of 66 total; 28 already sealed), placed as the first attribute above each derive stack, preserving all existing derives/cfg_attr.
- **SEAL-03:** each newly-sealed struct documents its construction escape hatch. All 38 already had `Default`; no new impls were needed.
- **Blast-radius migration (success criterion #4):** because `#[non_exhaustive]` forbids *any* struct expression from an external crate (even functional-update `..Default::default()`), every external-crate construction site was converted to `default()` + field assignment: 30 doctests, the `27_spm` example, 16 sites in `tests/validate_spm_math.rs`, and 3 bench sites.

## Task Commits
1. **Tasks A + B + C (SEAL-01/02/03 + blast-radius fixes)** — `ae0fbe34` (feat!) — committed atomically after all gates passed out-of-band (repo pre-commit hook exceeds the 30s timeout; `--no-verify` + out-of-band gates is the established pattern).

**Plan/context metadata:** `3c923a0e` (context), `ae0fbe34` (impl).

## Files Created/Modified
- `fdars-core/src/lib.rs` — wire visibility.
- `fdars-core/src/wire.rs` — doctest `ignore`, module `#![allow(dead_code)]`.
- 38 domain module files — `#[non_exhaustive]` + SEAL-03 doc line.
- `fdars-core/examples/27_spm/main.rs`, `tests/validate_spm_math.rs`, `benches/{boosting_regression_benchmarks,shapelet}.rs` — construction-site migration.

## Decisions Made
- **wire doctest → `ignore`, not `use crate::wire`:** the plan suggested `use crate::wire::*;`, but a doctest compiles as an *external* crate where `crate` is the doctest crate, not `fdars_core`; and the types are `pub(crate)` regardless. `ignore` is the only correct choice.
- **`#![allow(dead_code)]` on wire:** sealing turned 24 public types into unused internal code; a documented module-level allow beats 24 per-item allows.
- **SEAL-03 idiom corrected:** the requirement's phrasing (`..Default::default()`) is not a valid *external* construction path for a non_exhaustive struct (E0639). The correct, documented, and tested idiom is `Config::default()` + field assignment.

## Deviations from Plan

### Auto-fixed Issues

**1. [Missing scope — success criterion #4] External-crate construction sites broke under `#[non_exhaustive]`**
- **Found during:** Task C (gate verification).
- **Issue:** The plan scoped SEAL-02 as "add the attribute" and assumed the crate/examples/doctests would still compile. In fact `#[non_exhaustive]` rejects *every* struct expression from an external crate (E0639), including functional-update form — so 30 doctests, the `27_spm` example, 16 integration-test sites, and 3 bench sites failed to compile. This is squarely inside Phase 86 (success criterion #4 requires all examples + doctests compile).
- **Fix:** converted every external construction site to `default()` + field assignment (inline sites use the block-expr `{ let mut c = T::default(); c.f = v; c }` form). SEAL-03 doc lines reworded to the accurate idiom.
- **Files modified:** the `27_spm` example, `tests/validate_spm_math.rs`, 2 benches, and ~10 additional `src/` files carrying affected doctests (beyond the plan's 32 `files_modified`).
- **Verification:** `cargo test --doc` 208/0; `clippy --all-targets` clean; `cargo test` unit+integration green.

**2. [Missing — lint] Sealing wire produced 25 `dead_code` warnings**
- **Found during:** Task C (clippy `-D warnings`).
- **Issue:** wire's now-internal types/methods are unused inside the crate → `dead_code` warnings would fail the lint gate.
- **Fix:** documented module-level `#![allow(dead_code)]` in `wire.rs`.
- **Verification:** `clippy --all-targets --features linalg,parallel -- -D warnings` exits 0.

**3. [Lint] SEAL-03 doc line tripped `clippy::doc_lazy_continuation`**
- **Found during:** Task C.
- **Issue:** an added doc line directly under a markdown list item in `spm/profile.rs` was parsed as a lazy list continuation.
- **Fix:** separated every SEAL-03 doc line as its own paragraph with a blank `///` line.
- **Verification:** clippy clean.

---

**Total deviations:** 3 auto-fixed (1 missing-scope blast-radius migration, 2 lint). All necessary for the success criteria / gates. The blast-radius migration expands `files_modified` beyond the plan's 32 but stays strictly within Phase 86's stated goal (no numeric/behavioral change).

## Issues Encountered
- `/home` partition at 99% (6.7G free); freed `target/debug/{incremental,examples}` and routed the separate-feature serde build to `/tmp` tmpfs to avoid ENOSPC.
- Long combined gate runs were killed as background jobs; split into per-gate foreground runs to completion.

## User Setup Required
None — no external service configuration required.

## Next Phase Readiness
- Phase 87 (Targeted Renames) is unblocked. The `default()` + field-assignment migration pattern established here is the reference for any future config-construction edits.
- **Note for Phase 89 (CHANGELOG):** the breaking surface is larger than "sealed wire + sealed configs" — it also changes the *supported construction idiom* for 38 configs (struct literals no longer work for external callers). Call this out explicitly in the breaking-changes notes.

---
*Phase: 86-surface-sealing*
*Completed: 2026-09-08*
