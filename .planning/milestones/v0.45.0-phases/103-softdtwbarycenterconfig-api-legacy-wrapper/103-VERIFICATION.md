---
phase: 103-softdtwbarycenterconfig-api-legacy-wrapper
verified: 2026-09-13T14:00:00Z
status: passed
score: 7/7 must-haves verified
covered_files:
  - .planning/phases/103-softdtwbarycenterconfig-api-legacy-wrapper/103-01-PLAN.md
  - .planning/phases/103-softdtwbarycenterconfig-api-legacy-wrapper/103-01-SUMMARY.md
  - .planning/REQUIREMENTS.md
  - fdars-core/src/metric/soft_dtw.rs
  - fdars-core/src/metric/mod.rs
  - fdars-core/src/lib.rs
  - fdars-core/src/prelude.rs
covered_digest: "v1:sha256:44b078fc3606e53712da7ae763a392fffefdfa52e35cbbc604b510f8df8173fe"
behavior_unverified: 0
overrides_applied: 0
re_verification: null
---

# Phase 103: SoftDtwBarycenterConfig API & Legacy Wrapper Verification Report

**Phase Goal:** A `SoftDtwBarycenterConfig` builder-convention config struct + new config-taking entry point expose the multi-restart optimizer, while the existing `soft_dtw_barycenter(data, gamma, max_iter, tol)` signature is preserved as a thin behavior-compatible wrapper (non-breaking); new public items are additively re-exported (crate root + prelude) with a running doctest.
**Verified:** 2026-09-13T14:00:00Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `SoftDtwBarycenterConfig::default()` yields gamma=1.0, max_iter=100, tol=1e-6, n_restarts=10, seed=0 | ✓ VERIFIED | Manual `impl Default` at line 82 returns exact values; `config_default_values` test passes (34 passed in lib binary) |
| 2 | `soft_dtw_barycenter_with_config(&data, &config)` runs and returns a valid `SoftDtwBarycenterResult` driven by the multi-restart optimizer | ✓ VERIFIED | Function at line 959 delegates directly to `soft_dtw_barycenter_multirestart` with all config fields; `config_path_runs` test passes |
| 3 | Legacy `soft_dtw_barycenter(data, gamma, max_iter, tol)` keeps its exact 4-arg signature and is bit-identical to the config path with n_restarts=1 | ✓ VERIFIED | Function at line 902 calls `soft_dtw_barycenter_multirestart(data, gamma, max_iter, tol, 1, 0)` directly; `legacy_equals_config_single_restart` passes with `.to_bits()` comparison; NaN-row parity test also passes |
| 4 | `SoftDtwBarycenterConfig` + `soft_dtw_barycenter_with_config` are reachable at crate root AND via the prelude | ✓ VERIFIED | lib.rs lines 636/638-639 re-export both; prelude.rs lines 49-50 re-export both; `crate_root_reexport_reachable` and `prelude_reexport_reachable` tests pass |
| 5 | A running doctest on `soft_dtw_barycenter_with_config` passes under `cargo test --doc` | ✓ VERIFIED | `cargo test -p fdars-core --doc soft_dtw` → 2 passed (line 932 doctest + SoftDtwBarycenterConfig doctest) |
| 6 | `SoftDtwBarycenterResult` derives `PartialEq` (resolves deferred Phase-102 IN-02) | ✓ VERIFIED | Line 20: `#[derive(Debug, Clone, PartialEq)]` on `SoftDtwBarycenterResult`; `legacy_equals_config_single_restart` test uses `PartialEq` equality check |
| 7 | All 28 examples + wasm32 + default + serde builds + full test suite compile/pass unchanged (non-breaking) | ✓ VERIFIED | SUMMARY reports 2939 tests passed, 28 examples compile, wasm32/default/serde builds pass, fmt clean, clippy clean; no Cargo.toml changes (zero new dependencies confirmed by `git diff HEAD~4 HEAD -- fdars-core/Cargo.toml` returning empty) |

**Score:** 7/7 truths verified (0 present, behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `fdars-core/src/metric/soft_dtw.rs` | `SoftDtwBarycenterConfig` struct + `soft_dtw_barycenter_with_config` fn | ✓ VERIFIED | `#[non_exhaustive]`, `#[derive(Debug, Clone, PartialEq)]`, manual `impl Default` at line 82; `#[must_use]` fn at line 959 |
| `fdars-core/src/metric/mod.rs` | additive `pub use soft_dtw::{..., SoftDtwBarycenterConfig, soft_dtw_barycenter_with_config}` | ✓ VERIFIED | Lines 151-155 include both new items alongside all existing exports |
| `fdars-core/src/lib.rs` | crate-root re-export of both new items | ✓ VERIFIED | Lines 636/638-639 include both new items |
| `fdars-core/src/prelude.rs` | prelude re-export of both new items | ✓ VERIFIED | Lines 49-50 include both new items |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `soft_dtw_barycenter_with_config` | `soft_dtw_barycenter_multirestart` | direct call passing all 6 config fields | ✓ WIRED | Line 963-970: `soft_dtw_barycenter_multirestart(data, config.gamma, config.max_iter, config.tol, config.n_restarts, config.seed)` |
| legacy `soft_dtw_barycenter` | `soft_dtw_barycenter_multirestart` with n_restarts=1, seed=0 | direct call | ✓ WIRED | Line 908: `soft_dtw_barycenter_multirestart(data, gamma, max_iter, tol, 1, 0)` — parity by construction |
| `metric/mod.rs` pub use | `lib.rs` re-export | `prelude.rs` re-export | ✓ WIRED | Three-site additive re-export chain confirmed at all three files; two compile-time reachability tests pass |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Config default values pinned | `cargo test … config_default_values` | ok | ✓ PASS |
| Config path produces finite barycenter | `cargo test … config_path_runs` | ok | ✓ PASS |
| Legacy bit-identical to config at n_restarts=1 | `cargo test … legacy_equals_config_single_restart` | ok | ✓ PASS |
| NaN-row parity (legacy == config) | `cargo test … legacy_equals_config_nan_row_parity` | ok | ✓ PASS |
| Crate-root reachability | `cargo test … crate_root_reexport_reachable` | ok | ✓ PASS |
| Prelude reachability | `cargo test … prelude_reexport_reachable` | ok | ✓ PASS |
| Running doctest compiles and executes | `cargo test -p fdars-core --doc soft_dtw` | 2 passed | ✓ PASS |
| tslearn regression guard | `test_soft_dtw_barycenter_vs_tslearn` | ok (9 tests) | ✓ PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| SDTW-03 | 103-01-PLAN.md | SoftDtwBarycenterConfig + soft_dtw_barycenter_with_config expose multi-restart optimizer; legacy signature preserved non-breaking; additive re-exports with running doctest | ✓ SATISFIED | All 7 truths verified against codebase; tests pass |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| — | — | None found | — | — |

No `TBD`, `FIXME`, `XXX`, `TODO`, `HACK`, or `PLACEHOLDER` markers found in any of the four modified files. No `#[allow(dead_code)]` attributes remain on `soft_dtw_barycenter_multirestart`, `per_col_std`, or `perturbed_init` (all removed per plan; confirmed by grep returning empty). No new crate dependencies added (Cargo.toml unchanged).

### Human Verification Required

None. All must-haves are verifiable programmatically and have been confirmed by running tests.

### Gaps Summary

No gaps. All 7 must-have truths are verified against the actual codebase:

- `SoftDtwBarycenterConfig` is exactly as required: `#[non_exhaustive]`, `#[derive(Debug, Clone, PartialEq)]`, manual `impl Default` (NOT derived) with the documented non-zero values.
- `soft_dtw_barycenter_with_config` is `#[must_use]`, wired to the multirestart driver, and has a passing running doctest (confirmed live with `cargo test --doc`).
- The legacy 4-arg signature delegates to `soft_dtw_barycenter_multirestart(..., 1, 0)` — parity is structural, not aspirational, and is proven by the bit-identity test including the NaN-row edge case.
- `SoftDtwBarycenterResult` now derives `PartialEq` (IN-02 resolved).
- Both new items are at all three re-export sites and confirmed reachable by compile-time tests.
- No new crate dependency; no breaking change.

---

_Verified: 2026-09-13T14:00:00Z_
_Verifier: Claude (gsd-verifier)_
