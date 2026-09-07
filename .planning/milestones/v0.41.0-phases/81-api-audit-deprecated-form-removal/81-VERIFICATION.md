---
phase: 81-api-audit-deprecated-form-removal
verified: 2026-09-07T09:47:09Z
status: passed
score: 4/4 must-haves verified
behavior_unverified: 0
overrides_applied: 0
re_verification:
  # No previous VERIFICATION.md — initial verification
requirements_verified:
  - AUDIT-01
  - API-01
---

# Phase 81: API Audit & Deprecated-Form Removal Verification Report

**Phase Goal:** The full public surface is audited into a ranked, user-approved breaking-change inventory, and the 6 already-specified deprecated forms are removed from the crate.
**Verified:** 2026-09-07T09:47:09Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths (Roadmap Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | A ranked breaking-change inventory exists covering all four scopes, each entry listing location, proposed change, blast radius, and value/risk rating | ✓ VERIFIED | `81-AUDIT-INVENTORY.md` — Scope A (`AUD-01`–`AUD-06`), Scope B (`AUD-07`–`AUD-09`), Scope C (`AUD-10`–`AUD-13`), Scope D (`AUD-14`–`AUD-23`). Every row carries a grounded `file:line` Location, Remove/Proposed-change, Blast-radius (callers/examples/doctests counted), and a Value/Risk rating. 23 stable IDs. |
| 2 | The inventory was presented to and approved by the user | ✓ VERIFIED | `81-AUDIT-INVENTORY.md` `## Approval` (lines 104–119): "Approved: 2026-09-07 (user, at the Phase 81 human-verify checkpoint)." APPROVED lists (`AUD-07`,`AUD-08`,`AUD-10`,`AUD-11`,`AUD-14`–`AUD-18`) and DEFERRED lists (`AUD-09`/`AUD-13`,`AUD-12`,`AUD-19`–`AUD-23`) are both filled. Reduced-scope on naming exercised as permitted. |
| 3 | The 6 deprecated forms + crate-root/prelude re-exports are removed; all internal callers/unit tests/doctests/example 21 migrated to `Dim`/`_seeded` | ✓ VERIFIED | grep: 0 `fn (mean_2d\|fanova\|...)` definitions, 0 bare calls `(form)\s*\(`, 0 `#[deprecated]` on those files, 0 re-exports in `lib.rs`/`prelude.rs`/`depth/mod.rs`, 0 stray `#[allow(deprecated)]`. Remaining `fanova` textual hits are prose comments/doc-links only. `fanova_seeded(...,42)` present at all migrated sites (`inference/anova.rs:243,264`, `inference/permutation.rs:401`, `function_on_scalar.rs` unit tests). Example 21 uses `fanova_seeded(&data, &groups, 500, 42)` + `//!` doc references `fanova_seeded`. |
| 4 | Crate builds + all 28 examples compile; whole-crate gates green (fmt, clippy --all-targets, cargo test, --features serde build) | ✓ VERIFIED | Ran myself (see gate table below). fmt PASS, clippy `-D warnings` PASS (zero warnings), serde build PASS (forced rebuild after `touch lib.rs`), 28/28 examples compile, lib tests 2857/0, migrated integration binaries green. Two golden "failures" cited by SUMMARY confirmed pre-existing environmental flakes (pass in isolation). |

**Score:** 4/4 truths verified (0 present, behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `81-AUDIT-INVENTORY.md` | Ranked 4-scope inventory + filled Approval | ✓ VERIFIED | 139 lines; ranked tables per scope; `## Approval` section filled with APPROVED/DEFERRED AUD-NN lists |
| `fdars-core/src/fdata.rs` | `mean_2d` removed | ✓ VERIFIED | No `fn mean_2d`, no `#[deprecated]` |
| `fdars-core/src/function_on_scalar.rs` | `fanova` removed, `fanova_seeded` retained | ✓ VERIFIED | `fanova_seeded` at :804 retained; no `fn fanova(`; callers migrated to `...,42` |
| `fdars-core/src/depth/{random_tukey,random_projection,fraiman_muniz,modal}.rs` | 4 `_2d` forms removed | ✓ VERIFIED | No `fn *_2d` definitions, no `#[deprecated]` |
| `fdars-core/src/lib.rs`, `prelude.rs`, `depth/mod.rs` | re-exports removed | ✓ VERIFIED | 0 hits for the 6 forms; 0 stray `#[allow(deprecated)]` |
| `fdars-core/examples/21_function_on_scalar/main.rs` | migrated to `fanova_seeded` | ✓ VERIFIED | `use ...{fanova_seeded, ...}`; `fanova_seeded(&data,&groups,500,42)`; doc line updated |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `inference/anova.rs` | `function_on_scalar::fanova_seeded` | `fanova_seeded(...,42)` | ✓ WIRED | Legacy seed-42 stream preserved at :243, :264 |
| `inference/permutation.rs` | `fanova_seeded` | `fanova_seeded(...,42)` | ✓ WIRED | :401 |
| `example 21` | `function_on_scalar::fanova_seeded` | `use` + call | ✓ WIRED | Compiles; seed 42 pinned |
| Phase 82/83 consumption | approved `AUD-NN` set | AUD-ID citation in Approval | ✓ WIRED | Consumption contract explicit; DEFERRED entries routed to STAB-03 (Phase 84) |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Removed forms absent (defs) | `rg 'fn (mean_2d\|fanova\|...)'` src | NO fn DEFINITIONS | ✓ PASS |
| Removed forms absent (calls) | `rg '(form)\s*\('` src/tests/examples | NO BARE CALLS | ✓ PASS |
| Migration pins seed 42 | `rg 'fanova_seeded' ... \| grep 42` | 4 call sites `...,42` | ✓ PASS |
| Lib behavior unchanged | `cargo test --lib --features linalg,parallel` | 2857 passed / 0 failed | ✓ PASS |
| Dispatch equivalence preserved | `cargo test --test equivalence_phase50` | 7 passed / 0 failed | ✓ PASS |
| R-validation preserved | `cargo test --test validate_against_r` | 172 passed / 0 failed | ✓ PASS |

### Gate Execution (Criterion 4 — verifier-run)

| Gate | Command | Result | Status |
|------|---------|--------|--------|
| fmt | `cargo fmt --check` | clean | ✓ PASS |
| clippy | `cargo clippy --all-targets --features linalg,parallel -- -D warnings` | Finished, zero warnings | ✓ PASS |
| serde build | `cargo build --features serde` (forced rebuild) | Compiling → Finished, exit 0 | ✓ PASS |
| examples | `cargo build --examples --features linalg,parallel` | Finished; 28/28 examples | ✓ PASS |
| lib tests | `cargo test --lib --features linalg,parallel` | 2857 passed / 0 failed | ✓ PASS |
| golden phase48 (isolated) | `cargo test --test equivalence_phase48` | 5 passed / 0 failed | ✓ PASS (flake only under full-run disk pressure) |
| golden phase49 (isolated) | `cargo test --test equivalence_phase49` | 8 passed / 0 failed | ✓ PASS (flake only under full-run disk pressure) |

**Note on the two golden "failures" in the SUMMARY:** `golden_co_cluster_*` (phase48) and `svd_sign_fpca_two_matrix_bit_identical` (phase49) pass when their binaries run in isolation. They match the known environmental disk-pressure flake pattern (/home at 96%), touch none of the 6 removed forms, and are outside API-01 scope. Not a Phase 81 regression.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| AUDIT-01 | 81-01 | Ranked 4-scope inventory, presented for approval; API-02/03/04 sets drawn from approved list | ✓ SATISFIED | Truths 1 & 2; `81-AUDIT-INVENTORY.md` + filled Approval |
| API-01 | 81-02 | 6 deprecated forms + re-exports removed; callers/tests/doctests/example 21 migrated; crate + 28 examples compile | ✓ SATISFIED | Truths 3 & 4; grep + gate results |

No orphaned requirements — REQUIREMENTS.md maps only AUDIT-01 and API-01 to Phase 81, both claimed by plans.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| — | — | No TBD/FIXME/XXX debt markers in modified src files | — | None |

Remaining `fanova` textual occurrences are historical prose comments and intra-doc references (e.g. "reproduces the legacy `fanova` permutation stream") — no code calls, no definitions, no broken doc-links. These are acceptable per the phase directive.

### Gaps Summary

None. All 4 roadmap success criteria hold against the codebase, verified independently of SUMMARY claims. Both requirements (AUDIT-01, API-01) satisfied. All load-bearing gates (fmt, clippy `-D warnings`, serde build, 28 examples) run green by the verifier. The inventory was approved by the user at the checkpoint (criterion 2 already resolved) — no human re-approval requested.

---

_Verified: 2026-09-07T09:47:09Z_
_Verifier: Claude (gsd-verifier)_
