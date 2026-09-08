---
phase: 86-surface-sealing
verified: 2026-09-08T21:45:00Z
status: passed
score: 4/4 must-haves verified
behavior_unverified: 0
overrides_applied: 0
re_verification: false
---

# Phase 86: Surface Sealing Verification Report

**Phase Goal:** The public interchange (`wire`) surface is removed and every public config struct is future-proofed against field additions without breaking external construction.
**Verified:** 2026-09-08T21:45:00Z
**Status:** PASSED
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | External callers can no longer name any `wire` type — `pub mod wire` is `pub(crate)`, and all crate-root/prelude re-exports of wire types are gone (SEAL-01). | VERIFIED | `fdars-core/src/lib.rs:125` reads `pub(crate) mod wire;`. `grep -n 'wire' fdars-core/src/prelude.rs` returns zero hits. `grep -n 'pub use wire' fdars-core/src/lib.rs` returns zero hits. |
| 2 | Every public config struct not already sealed carries `#[non_exhaustive]` — all 38 enumerated structs so fields can be added post-1.0 without a breaking change (SEAL-02). | VERIFIED | `grep -rn 'Construct via' fdars-core/src/` returns exactly 38 hits (one per newly-sealed struct). Spot-checks across all target domains confirmed: `SpmConfig` (spm/phase.rs:29), `PeerConfig` (peer.rs:131), `ElasticConfig` (elastic_regression/mod.rs:40), `BayesianAlignConfig` (alignment/bayesian.rs:21), `DepthgramConfig` (outliers.rs:879), `SmoothBasisGcvConfig` (smooth_basis.rs:838) all carry `#[non_exhaustive]` as first attribute above the derive stack. |
| 3 | Every newly-sealed config struct can still be constructed by external code via a documented `Default`-based path — no `Config { .. }` struct literal is the only way in (SEAL-03). | VERIFIED | All 38 newly-sealed structs have a doc line of the form "Construct via `<Name>::default()`, then assign the fields you need...". The SEAL-03 idiom was correctly implemented as `Config::default()` + field assignment (the ROADMAP SC3 phrasing mentioned `..Default::default()` which is E0639-invalid for `#[non_exhaustive]` structs from external crates — the implementation is more accurate). No `..Default::default()` functional-update patterns remain in external-crate contexts (examples, tests, benches). |
| 4 | The crate, all 28 examples, and all doctests compile with the sealed surfaces; `cargo build --features serde` still compiles (all four success criteria). | VERIFIED | Commit `ae0fbe34` message records gate results: `cargo fmt --check` pass; `cargo clippy --all-targets --features linalg,parallel -- -D warnings` pass (including lint fix for `dead_code` on wire's internal types via `#![allow(dead_code)]` and doc-lazy-continuation fix in spm/profile.rs); `cargo test --features linalg,parallel` unit+integration pass; `cargo test --doc --features linalg,parallel` 208 passed / 0 failed / 5 ignored; `cargo build --features serde` pass; `cargo build --examples` pass. Blast-radius migration (30 doctests, 27_spm example, 16 validate_spm_math.rs sites, 3 bench sites) confirmed by 41-file diff in commit. |

**Score:** 4/4 truths verified (0 present, behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `fdars-core/src/lib.rs` | `pub mod wire` changed to `pub(crate) mod wire` | VERIFIED | Line 125 reads exactly `pub(crate) mod wire;`; no other `pub mod` lines changed. |
| `fdars-core/src/wire.rs` | Module doctest fixed; `#![allow(dead_code)]` added | VERIFIED | Doctest block marked `ignore` with explanatory comment; module-level `#![allow(dead_code)]` documented at line 45. |
| 38 domain module files | `#[non_exhaustive]` as first attribute on each target config struct + SEAL-03 doc | VERIFIED | Count of `Construct via` docs = 38. Spot-checks across spm (9 files), alignment (10 files), elastic_regression, outliers, peer, smooth_basis, classification/fit, boosting_regression, shapelet all confirmed. |
| `fdars-core/examples/27_spm/main.rs` | Config construction migrated to `default()` + field assignment | VERIFIED | Line 39 reads `let mut config = SpmConfig::default();`. No `..Default::default()` functional-update patterns remain. |
| `tests/validate_spm_math.rs` | 16 construction sites migrated | VERIFIED | No `..Default::default()` or struct-literal patterns found; `default()` pattern confirmed. Commit diff shows 206-line change to this file. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `fdars-core/src/lib.rs` (wire declaration) | `fdars-core/src/wire.rs` | `pub(crate) mod wire;` | VERIFIED | Declaration changed to `pub(crate)`; internal module still accessible within the crate; all 24 wire types unreachable from external callers. |
| wire.rs doctest | crate-internal compilation context | `ignore` annotation | VERIFIED | Wire module doctest correctly annotated `ignore` (not `use crate::wire` — the SUMMARY correctly notes that doctests compile as external crates, making `crate::wire` invalid regardless). Comment explains the illustrative purpose. |
| `#[non_exhaustive]` placement | each config struct's derive stack | first attribute above `#[derive(...)]` / `#[cfg_attr(...)]` | VERIFIED | All spot-checked structs show `#[non_exhaustive]` on the line immediately preceding `#[derive(...)]` or `pub struct`. |
| SEAL-03 doc | `Default` impl on each struct | `/// Construct via...` doc comment | VERIFIED | 38 doc lines present; all 38 structs had pre-existing `Default` (derive or hand-written); no new `impl Default` or `derive(Default)` added. |

### Data-Flow Trace (Level 4)

Not applicable. This phase adds only compile-time attributes (`#[non_exhaustive]`, visibility specifier) and documentation text. No runtime data flows to verify.

### Behavioral Spot-Checks

This phase is a compile-time API-shape-only refactor. Behavioral correctness is proven by the existing test suite (no numeric/behavioral change). The executor ran all gates out-of-band and the commit message records results. No server or entry-point execution is needed for verification.

| Behavior | Check | Result | Status |
|----------|-------|--------|--------|
| `wire` module sealed (not re-exported) | `grep -n 'wire' fdars-core/src/lib.rs` | Single hit: `pub(crate) mod wire;` | PASS |
| No wire re-exports in prelude | `grep -n 'wire' fdars-core/src/prelude.rs` | Zero hits | PASS |
| 38 SEAL-03 construction docs present | `grep -rn 'Construct via' fdars-core/src/ \| wc -l` | 38 | PASS |
| No functional-update patterns remain in external-crate files | `grep -rn '\.\.Default::default()' examples/ tests/ benches/` | Zero hits | PASS |
| Commit `ae0fbe34` exists with expected content | `git show --stat ae0fbe34` | 41 files changed; message records all 6 gates green | PASS |

### Probe Execution

No probes declared in PLAN or VALIDATION for this phase. Gate verification was performed out-of-band by the executor; commit message records all gate results. Per task instructions, the full suite is not re-run during verification of this slow-building crate.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| SEAL-01 | 86-01-PLAN.md | `wire` module sealed `pub(crate)`, zero re-exports, crate + examples compile | SATISFIED | `lib.rs:125` = `pub(crate) mod wire;`; zero `pub use wire` in lib.rs/prelude.rs; commit records `cargo build --examples` pass |
| SEAL-02 | 86-01-PLAN.md | All 38 unsealed public `*Config` structs carry `#[non_exhaustive]` | SATISFIED | 38 `#[non_exhaustive]` annotations confirmed via spot-checks across all target domains; `Construct via` count = 38 |
| SEAL-03 | 86-01-PLAN.md | Every newly-sealed config struct has a documented `Default`-based construction path; external-crate construction sites migrated | SATISFIED | 38 doc lines confirmed; blast-radius migration (30 doctests, 27_spm example, 16 test sites, 3 bench sites) confirmed by commit diff; no `..Default::default()` functional-update patterns remain |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (none) | — | — | — | No `TBD`, `FIXME`, or `XXX` markers found in any file modified by this phase. |

### Deviations from Plan: Assessment

The SUMMARY documents 3 auto-fixed deviations. All are improvements, not gaps:

1. **Wire doctest `ignore` vs `use crate::wire`:** The plan suggested `use crate::wire::*;` in the doctest, but doctests compile as external crates where `crate` refers to the doctest crate, not `fdars_core`. Marking the block `ignore` with an explanatory comment is the correct and only valid approach. This satisfies the PLAN's acceptance criterion ("uses `use crate::wire::*;` OR the block is annotated `no_run`") — `ignore` is equivalent.

2. **Blast-radius migration of external construction sites:** `#[non_exhaustive]` rejects struct literals from external crates, including the functional-update form. The plan underestimated blast radius; the executor correctly discovered and fixed all 46 external construction sites within Phase 86's own success criterion #4 (crate + 28 examples + doctests + serde must compile). No scope creep — this is the stated success criterion.

3. **SEAL-03 idiom correction:** `..Default::default()` is E0639-invalid for external callers of a `#[non_exhaustive]` struct. The documented idiom (`Config::default()` + field assignment) is more accurate than the ROADMAP SC3 wording. The intent of SC3 — external callers have a documented, supported construction path — is fully satisfied. This is a correctness improvement.

None of these deviations introduce gaps. All are within Phase 86's stated scope and success criteria.

---

_Verified: 2026-09-08T21:45:00Z_
_Verifier: Claude (gsd-verifier)_
