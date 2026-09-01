---
phase: 50-additive-api-surface-consolidation
verified: 2026-09-01T19:51:44Z
status: passed
score: 10/10 must-haves verified
behavior_unverified: 0
overrides_applied: 0
---

# Phase 50: Additive API-Surface Consolidation Verification Report

**Phase Goal:** A user gets a single canonical, consistent entry point for previously-inconsistent config/result patterns and redundant public functions — with the old forms still compiling and passing (now emitting deprecation warnings), so R/WASM bindings, the 28 examples, and external callers all keep working with zero breakage.
**Verified:** 2026-09-01T19:51:44Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
| --- | --- | --- | --- |
| 1 | 3 boosting configs gain additive `impl Default` (API-01) | ✓ VERIFIED | `impl Default for BoostingConfig` (mod.rs:67), `BayesianConfig` (:115), `StabilityConfig` (:150); pin-test `config_defaults_match_documented_values` passes |
| 2 | StlConfig NOT touched (already derives Default) | ✓ VERIFIED | No `impl Default for StlConfig` in diff; no E0119; crate builds |
| 3 | `fanova_seeded(data, groups, n_perm, seed)` exists, keeps hand-rolled LCG, seed threads into rng_state (API-01/02) | ✓ VERIFIED | Signature at function_on_scalar.rs:804; `let mut rng_state: u64 = seed;` (:857) with multiplier `6_364_136_223_846_793_005` (:862); test `fanova_seeded_different_seed_changes_pvalue_not_statistic` passes |
| 4 | `fanova` is `#[deprecated(since="0.30.0")]` shim delegating to `fanova_seeded(…, 42)`, bit-identical (API-01/02/03) | ✓ VERIFIED | `#[deprecated]` (:912) + body `fanova_seeded(data, groups, n_perm, 42)` (:918); goldens `fanova_shim_seed42_bit_identical` + `fanova_seeded_seed42_bit_identical` pass |
| 5 | Shared `Dim` enum (`#[non_exhaustive]`, One/Two) in src/dim.rs, re-exported at crate root (API-02) | ✓ VERIFIED | dim.rs present; `#[non_exhaustive] pub enum Dim { One, Two }`; `pub mod dim;` (lib.rs:89) + `pub use dim::Dim;` (:592) |
| 6 | Exactly 5 unified Dim-dispatchers ship, each forwards to `_1d` for both arms (API-02) | ✓ VERIFIED | modal.rs:55, fraiman_muniz.rs:54, random_projection.rs:70, random_tukey.rs:49, fdata.rs:193 — each `Dim::One \| Dim::Two => name_1d(…)`; 10 dispatch equivalence tests pass |
| 7 | ONLY 5 `_2d` shims + fanova are deprecated; NO `_1d` deprecated; deriv/geometric_median/functional_spatial/_nd untouched (API-02) | ✓ VERIFIED | Deprecated set = {fanova, mean_2d, modal_2d, fraiman_muniz_2d, random_projection_2d, random_tukey_2d}; grep confirms no `_1d` deprecated; deriv_2d/geometric_median_2d/functional_spatial/_nd not deprecated; dispatch.rs uses only `_1d` |
| 8 | Item #3 doc-only: canonical FpcaResult vocabulary note, NO field rename (API-01) | ✓ VERIFIED | regression.rs:5 `# Canonical result-field vocabulary (API-01)` note; `fitted` accepted; no pub field removed/renamed in diff |
| 9 | Additive-only: no public signature/field REMOVED or renamed; fanova signature byte-identical (API-03) | ✓ VERIFIED | `git diff ecad1248..HEAD` shows only `pub use` lines with names ADDED and fanova line byte-identical (attribute added above); zero pub-field removals; all `_1d`/`_2d`/fanova names retained in exports |
| 10 | No new crate dependency (API-03 / SC4) | ✓ VERIFIED | `git diff ecad1248..HEAD` on Cargo.toml files: empty |

**Score:** 10/10 truths verified (0 present, behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
| --- | --- | --- | --- |
| `fdars-core/src/boosting_regression/mod.rs` | 3 Default impls + pin-test | ✓ VERIFIED | 3 impls at :67/:115/:150, pinned by test |
| `fdars-core/src/function_on_scalar.rs` | fanova_seeded + deprecated fanova shim | ✓ VERIFIED | fanova_seeded:804, deprecated fanova:912 |
| `fdars-core/src/dim.rs` | Dim enum | ✓ VERIFIED | Present, non_exhaustive, re-exported |
| `fdars-core/src/depth/{modal,fraiman_muniz,random_projection,random_tukey}.rs` | 4 dispatchers + deprecated `_2d` | ✓ VERIFIED | Dispatchers forward to `_1d`; `_2d` deprecated |
| `fdars-core/src/fdata.rs` | mean dispatcher + deprecated mean_2d | ✓ VERIFIED | mean:193 forwards to mean_1d; mean_2d:206 deprecated |
| `fdars-core/src/regression.rs` | canonical-vocabulary doc note | ✓ VERIFIED | Module note at :5 (API-01) |
| `fdars-core/tests/equivalence_phase50.rs` | fanova + dispatch goldens | ✓ VERIFIED | 13 tests pass |

### Key Link Verification

| From | To | Via | Status | Details |
| --- | --- | --- | --- | --- |
| fanova (deprecated shim) | fanova_seeded(…, 42) | delegating call | ✓ WIRED | body `fanova_seeded(data, groups, n_perm, 42)`; bit-identical golden green |
| 5 dispatchers | `_1d` primitives | `Dim::One \| Dim::Two => name_1d(…)` | ✓ WIRED | all forward to `_1d`; equivalence tests green |
| src/dim.rs::Dim | crate root | `pub use dim::Dim` | ✓ WIRED | lib.rs:592 |
| lib.rs re-exports | fanova + fanova_seeded + Dim + 5 dispatchers | `pub use` | ✓ WIRED | all present, `_1d`/`_2d` retained |
| depth/dispatch.rs | `_1d` only | (no `_2d`) | ✓ WIRED | warning-free, zero `#[allow(deprecated)]` churn |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
| --- | --- | --- | --- |
| fanova bit-identity + dispatch equivalence | `cargo test --test equivalence_phase50` | 13 passed; 0 failed | ✓ PASS |
| boosting Default documented values | `cargo test default` | `config_defaults_match_documented_values` ok | ✓ PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
| --- | --- | --- | --- | --- |
| API-01 | 50-01, 50-02, 50-03 | Config/result patterns gain unified alternatives; old forms `#[deprecated]` with note; both paths compile+pass | ✓ SATISFIED | 3 Default impls; fanova_seeded + deprecated fanova; canonical-vocabulary doc note |
| API-02 | 50-02, 50-03 | Redundant fns gain single canonical entry point; superseded `#[deprecated]`; re-export surface tightened; no signature change | ✓ SATISFIED | 5 Dim dispatchers + fanova_seeded; exactly 6 deprecated (`_2d`×5 + fanova); no `_1d` deprecated; no signature change |
| API-03 | 50-01, 50-02, 50-03 | build/test/28 examples/R+WASM pass with deprecation warnings only — zero breakage | ✓ SATISFIED | Additive-only diff (no removal/rename); fanova byte-identical; all gates green this session (full suite both configs, 28 examples deprecation-free, wasm32 --features js, clippy --all-targets) |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
| --- | --- | --- | --- | --- |
| (none) | — | No TBD/FIXME/XXX in any phase-modified file | — | Clean |

### Human Verification Required

None. Phase is entirely additive Rust API surface with deterministic goldens; all behavior verified programmatically (bit-identity + dispatch-equivalence tests), and the additive-only guarantee verified structurally from the diff.

### Gaps Summary

No gaps. All 10 must-haves verified against the actual codebase. The additive-only guarantee holds: the only `-` lines in the `ecad1248..HEAD` src diff are `pub use` lines that gained names and the fanova line that gained a `#[deprecated]` attribute above it — no public item or field was removed or renamed. The deprecated set is exactly the intended six (`fanova`, `mean_2d`, `modal_2d`, `fraiman_muniz_2d`, `random_projection_2d`, `random_tukey_2d`); no `_1d` workhorse and no genuine `_nd`/divergent function is deprecated. No new dependency. The load-bearing back-compat criterion (API-03) is met.

---

_Verified: 2026-09-01T19:51:44Z_
_Verifier: Claude (gsd-verifier)_
