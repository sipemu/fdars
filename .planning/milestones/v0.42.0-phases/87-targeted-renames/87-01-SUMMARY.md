---
phase: 87-targeted-renames
plan: "01"
subsystem: api
tags: [rust, dim-dispatch, api-consolidation, rename, breaking-change, byte-identical]

requires:
  - phase: 86-surface-sealing
    provides: "sealed surface + non_exhaustive configs (independent; 87 builds on the same v0.42.0 breaking milestone)"
provides:
  - "geometric_median / hausdorff_self / hausdorff_cross / functional_spatial / kernel_functional_spatial consolidated onto single Dim-dispatched public signatures"
  - "private *_impl fns holding byte-identical bodies; hausdorff_3d kept separate"
  - "LpeerResult renamed to LocalPeerResult (matches PeerResult sibling)"
  - "all internal + external callers (examples, tests, benches) migrated to the new signatures"
affects: [88-large-suffix-batch, 89-release-preparation]

actuals:
  tokens: 55000
  tasks: 3
  commits: 3

tech-stack:
  added: []
  patterns:
    - "Dim-dispatch consolidation: pub fn name(.., dim: Dim) { match dim { Dim::One => name_1d_impl(..), Dim::Two => name_2d_impl(..) } } with a superset Option<&[f64]> grid param where the 2d form needs a second grid"
    - "External-crate call sites use fully-qualified fdars_core::dim::Dim to avoid import churn"

key-files:
  created: []
  modified:
    - "fdars-core/src/fdata.rs (geometric_median dispatcher + _1d/_2d_impl)"
    - "fdars-core/src/metric/hausdorff.rs (hausdorff_self/cross dispatchers; hausdorff_3d untouched)"
    - "fdars-core/src/depth/spatial.rs (functional_spatial + kernel_functional_spatial dispatchers; kernel _2d uniform-weights impl preserved)"
    - "fdars-core/src/peer.rs (LocalPeerResult), lib.rs/prelude.rs/metric/mod.rs/depth/mod.rs (re-exports)"
    - "fdars-core/src/{explain/helpers/kernel.rs,depth/tests.rs,metric/tests.rs} + examples/tests/benches (callers)"

key-decisions:
  - "functional_spatial_2d was a pure forward to _1d(.., None) → collapsed to ONE functional_spatial_impl; the Dim::Two arm forwards None (byte-identical to the old _2d)."
  - "kernel_functional_spatial_2d uses uniform weights (not Simpson's) → kept a DISTINCT _2d_impl; the two arms route to different impls."
  - "LpeerResult reference in the released CHANGELOG [0.36.0] section left intact (historically accurate); Phase 89's CHANGELOG documents the rename."

patterns-established:
  - "Dim-dispatch consolidation with byte-identical _impl bodies (the v0.41.0 pattern, extended to remove the old public _1d/_2d names)."

requirements-completed: [NAME-01, NAME-02, NAME-03, NAME-05]

coverage:
  - id: D1
    description: "geometric_median callable through one Dim-dispatched signature; no lone _1d/_2d public forms"
    requirement: "NAME-01"
    verification:
      - kind: integration
        ref: "cargo test --features linalg,parallel (2857 lib + validate_against_r geometric_median sites, 0 failed)"
        status: pass
    human_judgment: false
  - id: D2
    description: "hausdorff_self/cross and functional_spatial/kernel_functional_spatial Dim-dispatched; hausdorff_3d separate; byte-identical bodies"
    requirement: "NAME-02, NAME-03"
    verification:
      - kind: integration
        ref: "cargo test --features linalg,parallel (byte-identical non-regression) + clippy --all-targets"
        status: pass
    human_judgment: false
  - id: D3
    description: "LpeerResult renamed to LocalPeerResult everywhere in src (def, impl, doctests, re-exports)"
    requirement: "NAME-05"
    verification:
      - kind: integration
        ref: "grep -rn LpeerResult fdars-core/src = 0; cargo build + cargo test --doc"
        status: pass
    human_judgment: false
  - id: D4
    description: "crate + 28 examples + doctests + benches + serde build compile against consolidated signatures"
    requirement: "NAME-01"
    verification:
      - kind: integration
        ref: "cargo build --examples; cargo build --benches; cargo build --features serde; cargo fmt --check"
        status: pass
    human_judgment: false

duration: 50min
completed: 2026-09-08
status: complete
---

# Phase 87: Targeted Renames Summary

**Four `_1d`/`_2d` function families (`geometric_median`, `hausdorff_self`/`hausdorff_cross`, `functional_spatial`/`kernel_functional_spatial`) collapsed onto single `Dim`-dispatched signatures with byte-identical private `_impl` bodies, and `LpeerResult` renamed to `LocalPeerResult` — a pure API-shape change proven by 2857 passing lib tests.**

## Performance
- **Duration:** ~50 min
- **Tasks:** 3 (A: LpeerResult rename tracer, B: Dim-dispatch consolidations, C: external callers + gates)
- **Files modified:** 19 (11 src + examples/tests/benches + READMEs)

## Accomplishments
- **NAME-05 (tracer):** `LpeerResult` → `LocalPeerResult` across 15 src refs (definition, `impl`, doctests, `lib.rs`/`prelude.rs` re-exports). Zero external callers.
- **NAME-01:** `geometric_median_1d`/`_2d` → `geometric_median(.., argvals_t: Option<&[f64]>, dim: Dim)` + two private `_impl` fns (bodies differ: Simpson's 1D vs 2D weights).
- **NAME-02:** `hausdorff_self_1d`/`_2d` and `hausdorff_cross_1d`/`_2d` → `hausdorff_self`/`hausdorff_cross` (Dim-dispatched, superset `argvals_t`). `hausdorff_3d` kept separate (point-cloud input, still public).
- **NAME-03:** `functional_spatial_1d`/`_2d` → `functional_spatial` (one impl — `_2d` was a pure forward); `kernel_functional_spatial_1d`/`_2d` → `kernel_functional_spatial` (two distinct impls — `_2d` keeps uniform weights, `_1d` Simpson's).
- **Blast radius:** all internal callers (`explain/helpers/kernel.rs`, `depth/tests.rs`, `metric/tests.rs`, `fdata.rs` tests) + ~15 external sites (3 examples, `validate_against_r.rs` ×9, benches ×3 + group-name strings) + example READMEs migrated to the new signatures.

## Task Commits
1. **Task A (NAME-05 rename)** — `c315dc6b` (folded into the A+B commit)
2. **Task B (Dim-dispatch consolidations + in-crate callers)** — `c315dc6b` (refactor!)
3. **Task C (external callers + full gate set)** — `f6b7cf81` (refactor!)

## Files Created/Modified
- `fdars-core/src/fdata.rs`, `metric/hausdorff.rs`, `depth/spatial.rs` — dispatchers + `_impl` fns.
- `fdars-core/src/peer.rs`, `lib.rs`, `prelude.rs`, `metric/mod.rs`, `depth/mod.rs` — rename + re-exports.
- `fdars-core/src/explain/helpers/kernel.rs`, `depth/tests.rs`, `metric/tests.rs` — internal callers.
- `fdars-core/examples/{02,05,06}/main.rs` + READMEs, `tests/validate_against_r.rs`, `benches/depth_benchmarks.rs` — external callers.

## Decisions Made
- **`functional_spatial_2d` collapsed to one impl** (it was a pure `_1d(.., None)` forward); `Dim::Two` forwards `None` to stay byte-identical.
- **`kernel_functional_spatial_2d` kept a distinct impl** — it uses uniform weights, NOT Simpson's; treating it as a forward would have been a silent numeric change. The 2857-test suite + code-review gate confirm no drift.
- **Historical CHANGELOG `[0.36.0]` `LpeerResult` mention left intact** — accurate to that release; the rename is documented in Phase 89's `[0.42.0]` entry (deviation from the plan's overly-broad "zero `LpeerResult` in fdars-core/" grep, which would have falsified history).

## Deviations from Plan
### Auto-fixed / judgment calls
**1. Historical CHANGELOG reference preserved** — the plan's Task A verify grep spanned all of `fdars-core/`; I narrowed it to `fdars-core/src/` and left the released-`[0.36.0]` CHANGELOG entry naming `LpeerResult` intact (renaming it would misrepresent the 0.36.0 API). Verified `grep -rn LpeerResult fdars-core/src = 0`.
**2. Fully-qualified `fdars_core::dim::Dim` in external callers** — examples/tests/benches are external crates where `crate::` would misresolve; used the fully-qualified path to avoid per-file `use` churn.
**3. Example README references updated** — beyond the plan's code scope, updated 3 example READMEs that named the removed `_1d` functions, for doc accuracy (not compile-checked).

---
**Total deviations:** 3 judgment calls; none change scope or behavior. All 4 requirements delivered.

## Issues Encountered
- `/home` hit 100% (3.8G free) mid-phase; freed `target/debug/{incremental,examples}` and routed the examples/benches/serde builds to `/tmp` tmpfs (RAM-backed) to complete the gates without ENOSPC.
- The known co_cluster/svd_sign golden flake did NOT surface this run (all integration goldens passed).

## Next Phase Readiness
- Phase 88 (Large Suffix Batch, NAME-04) is unblocked — it applies this exact Dim-dispatch pattern to the remaining, higher-blast-radius lone-`_1d`/`_2d` functions across the crate + all 28 examples. The consolidation + call-site-rewrite tooling from this phase is the reference.

---
*Phase: 87-targeted-renames*
*Completed: 2026-09-08*
