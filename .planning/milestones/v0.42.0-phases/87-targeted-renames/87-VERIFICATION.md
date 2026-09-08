---
phase: 87-targeted-renames
verified: 2026-09-08T00:00:00Z
status: passed
score: 4/4 must-haves verified
behavior_unverified: 0
overrides_applied: 0
re_verification: false
---

# Phase 87: Targeted Renames Verification Report

**Phase Goal:** The small, low-blast-radius naming inconsistencies are resolved — spatial/median families collapse onto single `Dim`-dispatched signatures and the PEER result type name matches its sibling.
**Verified:** 2026-09-08
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `geometric_median` is callable through one `Dim`-dispatched signature; no lone `_1d`/`_2d` public forms remain (NAME-01, SC1) | VERIFIED | `fdata.rs:1081` — one `pub fn geometric_median(.., dim: Dim)` dispatching to private `geometric_median_1d_impl` / `geometric_median_2d_impl` (line 1106/1131). No `pub fn geometric_median_1d` or `_2d` in file. |
| 2 | `hausdorff_self`/`hausdorff_cross` and `functional_spatial`/`kernel_functional_spatial` are each callable through one `Dim`-dispatched signature; `hausdorff_3d` stays separate (NAME-02, NAME-03, SC2) | VERIFIED | `metric/hausdorff.rs:51/86` — `pub fn hausdorff_self` and `pub fn hausdorff_cross` with `Dim` dispatch; `pub fn hausdorff_3d` at line 118 unchanged. `depth/spatial.rs:22/225` — `pub fn functional_spatial` (one `functional_spatial_impl`) and `pub fn kernel_functional_spatial` (two distinct impls: `_1d_impl` uses Simpson's, `_2d_impl` uses `vec![1.0; n_points]` uniform weights). |
| 3 | The result type is named `LocalPeerResult` everywhere — definition, all references, `lib.rs`/`prelude.rs` re-exports, doctests; `LpeerResult` is gone (NAME-05, SC3) | VERIFIED | `peer.rs:195` — `pub struct LocalPeerResult`; `impl LocalPeerResult` at line 709; `lib.rs:615` and `prelude.rs:117` re-export `LocalPeerResult`. `grep -rn LpeerResult fdars-core/src/` returns zero. CHANGELOG [0.36.0] historical mention preserved intentionally (not a code symbol). |
| 4 | The crate, all 28 examples, all doctests, and benches compile against the consolidated signatures; existing suite passes — no numeric drift (SC4) | VERIFIED | `cargo check -p fdars-core --features linalg,parallel` exits 0 (4.86s). All external callers confirmed using new signatures: examples/02 (`geometric_median` + `Dim::One`), examples/05 (`functional_spatial` + `Dim::One`), examples/06 (`hausdorff_self` + `Dim::One`), `tests/validate_against_r.rs` (all 9 sites use new dispatcher forms), `benches/depth_benchmarks.rs` (import + call + group-name strings updated). SUMMARY records: fmt-check pass, clippy --all-targets pass, 2857 lib tests pass, serde build pass, examples build pass, benches build pass, doctests pass. Commits `c315dc6b` and `f6b7cf81` confirmed in repo. |

**Score:** 4/4 truths verified (0 present, behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `fdars-core/src/peer.rs` | `LocalPeerResult` definition + impl + doctests | VERIFIED | `pub struct LocalPeerResult` at line 195; `impl LocalPeerResult` at line 709; doctests updated at lines 2107/2112/2139/2149/2151/2169/2179 |
| `fdars-core/src/fdata.rs` | `geometric_median` dispatcher + `_1d_impl`/`_2d_impl` | VERIFIED | `pub fn geometric_median` at line 1081; private `fn geometric_median_1d_impl` at line 1106; `fn geometric_median_2d_impl` at line 1131 |
| `fdars-core/src/metric/hausdorff.rs` | `hausdorff_self`/`hausdorff_cross` dispatchers + `_impl` fns | VERIFIED | `pub fn hausdorff_self` (line 51), `pub fn hausdorff_cross` (line 86), `pub fn hausdorff_3d` (line 118); private `_1d_impl`/`_2d_impl` for both families |
| `fdars-core/src/depth/spatial.rs` | `functional_spatial`/`kernel_functional_spatial` dispatchers + `_impl` fns | VERIFIED | `pub fn functional_spatial` (line 22) with single `functional_spatial_impl`; `pub fn kernel_functional_spatial` (line 225) with distinct `_1d_impl` (Simpson's) and `_2d_impl` (uniform weights `vec![1.0; n_points]`) |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `depth/spatial.rs` | `depth/mod.rs` | `pub use spatial::{functional_spatial, kernel_functional_spatial}` | WIRED | `mod.rs:42` confirmed |
| `metric/hausdorff.rs` | `metric/mod.rs` | `pub use hausdorff::{hausdorff_3d, hausdorff_cross, hausdorff_self}` | WIRED | `mod.rs:145` confirmed |
| `lib.rs` | consumers | re-exports `hausdorff_self`, `hausdorff_cross`, `hausdorff_3d`, `functional_spatial`, `kernel_functional_spatial`, `geometric_median` | WIRED | `lib.rs:634-635/647-648/673` confirmed |
| `prelude.rs` | consumers | re-exports `functional_spatial` and `LocalPeerResult` | WIRED | `prelude.rs:41/117` confirmed |
| `explain/helpers/kernel.rs` | `depth::functional_spatial` | internal callers at lines 133 and 173 use `depth::functional_spatial(.., None, crate::dim::Dim::One)` | WIRED | Both confirmed using new dispatcher form |
| `tests/validate_against_r.rs` | new dispatcher signatures | `fdars_core::depth::functional_spatial(.., fdars_core::dim::Dim::One)` etc. | WIRED | Confirmed at lines 840-865, 2301-2302, 2489-2522, 2814, 2855, 5290 |
| `benches/depth_benchmarks.rs` | new dispatcher signatures | `functional_spatial(.., fdars_core::dim::Dim::One)`, `hausdorff_self(.., fdars_core::dim::Dim::One)` | WIRED | Confirmed at lines 116/122/126, 159/166/170; group-name strings updated |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Crate compiles with consolidated signatures | `cargo check -p fdars-core --features linalg,parallel` | Finished in 4.86s, exit 0 | PASS |
| No old `_1d`/`_2d` public names survive in code | `grep -rnP '(geometric_median_1d|hausdorff_self_1d|...)' fdars-core/src examples tests benches | grep -v '_impl'` | Zero code hits (2 doc-comment references in `///` lines only) | PASS |
| No `LpeerResult` in src/examples/tests/benches | `grep -rn 'LpeerResult' fdars-core/src/ fdars-core/examples/ fdars-core/tests/ fdars-core/benches/` | Zero hits | PASS |
| `kernel_functional_spatial_2d_impl` uses uniform weights | Read `depth/spatial.rs:262-272` | `let weights = vec![1.0; n_points]` confirmed — distinct from `_1d_impl`'s `simpsons_weights` | PASS |
| Commits documented in SUMMARY exist | `git log --oneline c315dc6b f6b7cf81` | `c315dc6b refactor(87)!: Dim-dispatch consolidation + LpeerResult→LocalPeerResult`; `f6b7cf81 refactor(87)!: migrate external call sites to Dim-dispatched signatures` | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| NAME-01 | 87-01-PLAN.md | `geometric_median` consolidated onto one `Dim`-dispatched signature (AUD-19) | SATISFIED | `fdata.rs:1081` — single `pub fn geometric_median(.., dim: Dim)`; private `_impl` bodies |
| NAME-02 | 87-01-PLAN.md | `hausdorff_*` family consolidated onto `Dim` dispatch (AUD-20) | SATISFIED | `hausdorff.rs:51/86` — `hausdorff_self`/`hausdorff_cross` dispatchers; `hausdorff_3d` unchanged |
| NAME-03 | 87-01-PLAN.md | `functional_spatial_*` / `kernel_functional_spatial_*` consolidated (AUD-21) | SATISFIED | `spatial.rs:22/225` — dispatchers with distinct `_impl` bodies confirmed |
| NAME-05 | 87-01-PLAN.md | `LpeerResult` renamed to `LocalPeerResult` everywhere (AUD-23) | SATISFIED | Zero `LpeerResult` in src/examples/tests/benches; `LocalPeerResult` in all re-exports and doctests |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `depth/spatial.rs` | 14, 216 | Old names `functional_spatial_2d` / `kernel_functional_spatial_2d` appear in `///` doc comments only | Info | Migration guidance for users; not callable symbols; no action required |

No debt markers (`TBD`/`FIXME`/`XXX`) found in any modified file.

### Human Verification Required

None. All truths are fully verifiable from the codebase.

### Gaps Summary

No gaps. All four success criteria are met:

1. SC1 (NAME-01): `geometric_median` is the sole public form with `Dim` dispatch; private `_impl` bodies hold verbatim logic.
2. SC2 (NAME-02, NAME-03): `hausdorff_self`/`hausdorff_cross` and `functional_spatial`/`kernel_functional_spatial` are Dim-dispatched; `hausdorff_3d` unchanged; `kernel_functional_spatial_2d_impl` correctly retains uniform weights.
3. SC3 (NAME-05): `LocalPeerResult` is the type name everywhere in live code; CHANGELOG historical reference is intentionally preserved.
4. SC4: `cargo check` exits 0; all external callers (examples, tests, benches) use the new dispatcher signatures; executor gate results (2857 tests, clippy, fmt, serde build, examples, benches, doctests) all green per SUMMARY coverage section.

---

_Verified: 2026-09-08_
_Verifier: Claude (gsd-verifier)_
