---
phase: 50-additive-api-surface-consolidation
plan: 03
subsystem: depth + fdata + regression
tags: [api, additive, deprecated, dim-dispatch, golden, doc-only, phase-close]
requires:
  - phase: 50
    plan: 02
    provides: "#[deprecated] + deprecation-hygiene pattern; tests/equivalence_phase50.rs (extended here)"
provides:
  - "Dim enum (Dim::One/Dim::Two, #[non_exhaustive]) in src/dim.rs, re-exported at crate root (API-02)"
  - "5 unified Dim-dispatchers: depth::{modal, fraiman_muniz, random_projection, random_tukey}, fdata::mean — each forwards to its _1d primitive for both Dim arms (API-02)"
  - "5 redundant _2d shims #[deprecated(since=0.30.0)] (modal_2d, fraiman_muniz_2d, random_projection_2d, random_tukey_2d, mean_2d); no _1d deprecated (API-02)"
  - "Canonical FpcaResult vocabulary doc note in regression.rs — DOC-ONLY, no field rename (API-01)"
  - "equivalence_phase50.rs dispatch goldens (==_1d/==_2d deterministic + structural RNG); closes API-01/02/03 for Phase 50"
affects: []
actuals:
  tokens: 4500
  tasks: 3
  commits: 3
tech-stack:
  added: []
  patterns: [dim-dispatch-enum, forward-by-construction-single-arm-match, deterministic-vs-structural-golden-split, deprecation-hygiene]
key-files:
  created:
    - fdars-core/src/dim.rs
  modified:
    - fdars-core/src/lib.rs
    - fdars-core/src/prelude.rs
    - fdars-core/src/depth/mod.rs
    - fdars-core/src/depth/modal.rs
    - fdars-core/src/depth/fraiman_muniz.rs
    - fdars-core/src/depth/random_projection.rs
    - fdars-core/src/depth/random_tukey.rs
    - fdars-core/src/depth/tests.rs
    - fdars-core/src/fdata.rs
    - fdars-core/src/regression.rs
    - fdars-core/tests/validate_against_r.rs
    - fdars-core/tests/equivalence_phase50.rs
key-decisions:
  - "SOURCE-VERIFIED SCOPE: 5 dispatchers = 4 depth (modal, fraiman_muniz, random_projection, random_tukey) + 1 fdata (mean), NOT 6. functional_spatial is arity-excluded (_1d takes argvals, _2d does not); band/regression have no _2d; deriv/geometric_median genuinely diverge; _nd algorithms (pca_nd, karcher_*_nd, srsf_*_nd) are different algorithms — all untouched."
  - "Each dispatcher forwards to its _1d primitive for BOTH Dim arms (Dim::One | Dim::Two => name_1d(…)) — the _2d path never diverged. The dim param is intent-explicit + a single future seam."
  - "DETERMINISTIC/STRUCTURAL golden split: modal/fraiman_muniz/mean assert_eq! vs BOTH _1d and _2d (bit-identical). random_projection/random_tukey call _seeded(…, None) -> thread_rng() (fresh entropy, no public seed) so assert_eq! would flake — verified STRUCTURALLY (len == n_obs AND every value in [0,1]); forwarding to _1d is a compile-time single-arm-match guarantee, not a runtime one."
  - "ONLY the 5 _2d shims deprecated (#[deprecated(since=0.30.0)]); NO _1d deprecated (workhorse called by depth/dispatch.rs). depth/dispatch.rs verified: uses only _1d/_seeded forms -> stayed warning-free with ZERO #[allow(deprecated)] churn."
  - "Item #3 DOC-ONLY: regression.rs module header documents canonical FPCA vocab (scores/rotation/mean/weights/singular_values/centered) + `fitted` as accepted response-field name; FpcPredictor trait noted as cross-model unification layer. NO field renamed (renames deferred to APIB-01)."
deviations:
  - "Rule 3 (blocking issue): the plan assumed `pub use` re-exports of deprecated items emit NO deprecation warning; empirically this rustc (1.97) DOES warn on re-exporting a #[deprecated] fn. Under clippy --all-targets -D warnings this was a false-red blocker. Fix: added targeted #[allow(deprecated)] to the back-compat re-export blocks that still export the _2d names — lib.rs (depth + fdata blocks), depth/mod.rs (fraiman_muniz/modal/random_projection/random_tukey lines), prelude.rs (depth block). The _2d names remain re-exported per API-03; only the deprecation lint is silenced on the re-export statements."
patterns-established:
  - "Dim-dispatch: a shared #[non_exhaustive] Dim enum + name(…, dim: Dim) wrappers forwarding to the _1d primitive, retiring redundant _2d shims via #[deprecated] without touching any primitive."
  - "When re-exporting a #[deprecated] item for back-compat under -D warnings, #[allow(deprecated)] the re-export statement (the deprecation lint fires on `pub use`, contrary to common assumption)."
requirements-completed: [API-01, API-02, API-03]
coverage:
  - id: D1
    description: "5 unified Dim-dispatchers equal their _1d/_2d primitives (deterministic) or produce valid depth vectors (RNG)"
    requirement: API-02
    verification:
      - kind: integration
        ref: "equivalence_phase50 dispatch_* (11 tests): modal/fraiman_muniz/mean assert_eq! vs _1d AND _2d; random_projection/random_tukey structural [0,1]+len vs dispatcher and _2d — 11/11 pass under linalg,parallel"
        status: pass
    human_judgment: false
  - id: D2
    description: "Only the 5 _2d shims deprecated; no _1d deprecated; dispatch.rs warning-free"
    requirement: API-02
    verification:
      - kind: integration
        ref: "clippy --all-targets --features linalg,parallel -D warnings clean; depth/dispatch.rs uses only _1d/_seeded (zero #[allow])"
        status: pass
    human_judgment: false
  - id: D3
    description: "Item #3 canonical-vocabulary doc note added; no field renamed"
    requirement: API-01
    verification:
      - kind: manual
        ref: "regression.rs module header carries the vocab table + fitted note; FpcaResult fields unchanged; grep confirms no rename"
        status: pass
    human_judgment: false
  - id: D4
    description: "Whole-phase gate: both feature configs + 28 examples + wasm + -D warnings + fmt green"
    requirement: API-03
    verification:
      - kind: integration
        ref: "cargo test both --features linalg,parallel AND --no-default-features --features linalg (18 test-result groups each, zero failures); cargo build --examples (all 28); wasm32 --features js compiles; clippy --all-targets -D warnings clean; cargo fmt --check clean"
        status: pass
    human_judgment: false
status: complete
---

# Phase 50 Plan 03: Dim-Dispatch Unification + Field-Naming Docs Summary

Unified the cleanly-unifiable `_1d`/`_2d` depth/fdata pairs behind a single ergonomic
`name(…, dim: Dim)` entry point (item #4), deprecated only the redundant `_2d` shims, and
added the doc-only canonical-vocabulary note (item #3) — closing API-01/02/03 and Phase 50.

## What shipped

**Dim enum (`src/dim.rs`):** `Dim { One, Two }`, `#[derive(Debug, Clone, Copy, PartialEq, Eq)]`,
`#[non_exhaustive]`, re-exported at the crate root (`fdars_core::Dim`). Shared by the depth and
fdata dispatchers.

**5 unified dispatchers** (each `#[must_use]`, each `Dim::One | Dim::Two => name_1d(…)`):

| Dispatcher | File | Forwards to |
|-----------|------|-------------|
| `depth::modal` | depth/modal.rs | `modal_1d` |
| `depth::fraiman_muniz` | depth/fraiman_muniz.rs | `fraiman_muniz_1d` |
| `depth::random_projection` | depth/random_projection.rs | `random_projection_1d` |
| `depth::random_tukey` | depth/random_tukey.rs | `random_tukey_1d` |
| `fdata::mean` | fdata.rs | `mean_1d` |

**5 `_2d` shims deprecated** `#[deprecated(since = "0.30.0", note = "…forwards to <name>_1d")]`:
`modal_2d`, `fraiman_muniz_2d`, `random_projection_2d`, `random_tukey_2d`, `mean_2d`. Bodies and
signatures unchanged; no `_1d` deprecated; `_nd` algorithms and functional_spatial/deriv/
geometric_median/regression untouched.

**Item #3 (doc-only):** `regression.rs` module header now documents the canonical `FpcaResult`
vocabulary and that `fitted` is the accepted response-field name, pointing at `FpcPredictor` as
the cross-model unification layer. No field renamed.

## Deterministic vs structural golden split

- **Deterministic pairs (modal, fraiman_muniz, mean):** `assert_eq!(unified(…, Dim::One), _1d(…))`
  AND `assert_eq!(unified(…, Dim::Two), _2d(…))` — bit-identical on a fixed fixture.
- **RNG pairs (random_projection, random_tukey):** `_1d` calls `_seeded(…, None) → thread_rng()`
  (fresh entropy, no public seed), so two calls are independent draws — `assert_eq!` would flake.
  Verified STRUCTURALLY (len == n_obs AND every value ∈ [0,1]), mirroring the existing
  `test_random_projection_2d_returns_valid`. Forwarding to `_1d` is a compile-time single-arm-match
  guarantee, not a runtime one.

## Deprecation hygiene (callers handled)

`#[allow(deprecated)]` on the specific in-crate `_2d`-caller test fns (not blanket module allows):
`depth/tests.rs::{test_modal_2d_delegates, test_fraiman_muniz_2d_delegates,
test_random_projection_2d_returns_valid}`, `fdata.rs::test_mean_2d_delegates`,
`validate_against_r.rs::{test_2d_delegates_to_1d_fm, test_2d_mean_valid}`, and the new
`equivalence_phase50.rs` `dispatch_*_equals_2d` / `dispatch_*_2d_is_valid` tests.
`random_tukey_2d` had no non-re-export caller (confirmed). `depth/dispatch.rs` verified: uses only
`_1d`/`_seeded` forms → zero `#[allow]` needed there.

## Deviation

**Rule 3 (blocking issue):** The plan asserted that `pub use` re-exports of deprecated items emit
no deprecation warning. On rustc 1.97 they DO — 11 lib warnings appeared, a false-red under
`clippy --all-targets -D warnings`. Fix: targeted `#[allow(deprecated)]` on the back-compat
re-export blocks that still export the `_2d` names (lib.rs depth + fdata blocks, depth/mod.rs
lines, prelude.rs depth block). The `_2d` names stay re-exported per API-03; only the lint is
silenced on those re-export statements. This is now recorded as an established pattern.

## Gate results (whole-phase close)

- `cargo test --features linalg,parallel`: 18 test-result groups, 0 failures.
- `cargo test --no-default-features --features linalg`: 18 test-result groups, 0 failures.
- 11/11 `dispatch_*` goldens pass.
- `cargo build --examples --features linalg,parallel`: all 28 examples build.
- `cargo build --target wasm32-unknown-unknown --features js --no-default-features`: compiles.
- `cargo clippy --all-targets --features linalg,parallel -- -D warnings`: clean.
- `cargo fmt -p fdars-core --check`: clean.
- No public signature changed (all additive); no new dependency.

## Commits

- `518c9d91` feat(50-03): add Dim enum + 5 unified depth/fdata dispatchers
- `60a44c5f` feat(50-03): deprecate 5 redundant _2d shims + hygiene + ==_2d goldens
- `30f0669a` docs(50-03): canonical FpcaResult vocabulary note (item #3, doc-only)

## Self-Check: PASSED

- Files verified on disk: `fdars-core/src/dim.rs`, `fdars-core/tests/equivalence_phase50.rs`,
  `.planning/phases/50-additive-api-surface-consolidation/50-03-SUMMARY.md`.
- Commits verified in git log: `518c9d91`, `60a44c5f`, `30f0669a`.
