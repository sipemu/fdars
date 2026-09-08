---
phase: 88-large-suffix-batch
plan: "01"
subsystem: api
tags: [rust, dim-dispatch, api-consolidation, rename, breaking-change, byte-identical, naming-unification]

requires:
  - phase: 87-targeted-renames
    provides: "the Dim-dispatch consolidation pattern + call-site rewrite tooling"
provides:
  - "41 lone _1d functions consolidated: Cat 1 (5) downgraded to pub(crate); Cat 2 depth+fdata (15) + Cat 3 metric self/cross (20) given Dim dispatchers"
  - "5 genuinely-1D functions plain-renamed (drop _1d, no dim): fdata_to_pc, fdata_to_pls, fourier_fit, pspline_fit, select_basis_auto"
  - "entire example/doc/test/bench surface migrated to the new signatures"
affects: [89-release-preparation]

actuals:
  tokens: 250000
  tasks: 8
  commits: 8

tech-stack:
  added: []
  patterns:
    - "Simplified consolidation mechanic: flip pub fn X_1d → pub(crate) fn X_1d (body byte-identical) + add pub fn X(.., dim: Dim) dispatcher — keeps dispatch.rs and all internal callers unchanged"
    - "Plain rename (drop _1d, no dim param) for genuinely-1D-only functions where a dim selector would be inert cruft at 1.0"

key-files:
  created: []
  modified:
    - "fdars-core/src/depth/*.rs (13 Cat 2 dispatchers), fdars-core/src/depth/mod.rs"
    - "fdars-core/src/metric/*.rs (20 Cat 3 self/cross dispatchers), fdars-core/src/metric/mod.rs"
    - "fdars-core/src/fdata.rs (center, norm_lp dispatchers; mean_1d pub(crate))"
    - "fdars-core/src/regression.rs (fdata_to_pc/pls plain rename, ~171 call sites), src/basis/* (fourier_fit/pspline_fit/select_basis_auto)"
    - "fdars-core/src/lib.rs, prelude.rs (re-exports); examples/tests/benches (external callers)"

key-decisions:
  - "USER DECISION: plain rename (drop _1d, no dim) for the 5 genuinely-1D-only functions (FPCA/PLS/basis fits) — a clean 1.0 API over an inert dim param."
  - "USER DECISION: _seeded variants + the 3 lone _2d (fosr_2d, predict_fosr_2d, simpsons_weights_2d) stay as-is (name collisions / load-bearing helper)."
  - "Simplified mechanic (visibility-flip, not _impl rename) chosen so src/depth/dispatch.rs and all internal callers compile unchanged."
  - "RESEARCH signature error corrected: pca_cross_1d has NO argvals param → pca_cross(data1, data2, ncomp, dim). Verified against source + by verifier/reviewer."

patterns-established:
  - "Dim-dispatch consolidation via pub(crate) visibility flip + dispatcher; plain-rename for 1D-only funcs — the terminal naming-unification pass before 1.0."

requirements-completed: [NAME-04]

coverage:
  - id: D1
    description: "41 lone _1d functions consolidated onto Dim dispatch (Cat 1/2/3); 5 genuinely-1D funcs plain-renamed"
    requirement: "NAME-04"
    verification:
      - kind: integration
        ref: "cargo test --features linalg,parallel (2857 lib + all integration + 208 doctests, 0 failed) — byte-identical non-regression"
        status: pass
    human_judgment: false
  - id: D2
    description: "all 28 examples + doctests + tests + benches compile against the new surface"
    requirement: "NAME-04"
    verification:
      - kind: integration
        ref: "cargo build --examples; cargo build --benches; cargo test --doc; clippy --all-targets --features linalg,parallel -D warnings"
        status: pass
    human_judgment: false
  - id: D3
    description: "no lingering public _1d/_2d removed names; 3 lone _2d + _seeded + Cat-5 exclusions unchanged"
    requirement: "NAME-04"
    verification:
      - kind: integration
        ref: "negative grep over src+examples+tests+benches = 0 (excluding kept names); cargo build --features serde"
        status: pass
    human_judgment: false

duration: 90min
completed: 2026-09-08
status: complete
---

# Phase 88: Large Suffix Batch Summary

**41 lone `_1d` functions across depth, metric, and fdata consolidated onto `Dim`-dispatched public signatures (byte-identical `pub(crate)` bodies), and 5 genuinely-1D functions (FPCA/PLS/basis) plain-renamed to drop `_1d` — the crate's terminal naming-unification pass, proven by 2857 passing tests with zero numeric drift.**

## Performance
- **Duration:** ~90 min (impl agent + inline gate verification)
- **Tasks:** 8 (Cat 1 depth, Cat 1 mean, Cat 2 depth ×13, Cat 2 fdata ×2, Cat 3 metric ×20, plain-rename basis ×3, plain-rename regression ×2, external-caller sweep + gates)
- **Commits:** 8

## Accomplishments
- **Cat 1 (5):** `fraiman_muniz_1d`, `modal_1d`, `random_projection_1d`, `random_tukey_1d`, `mean_1d` → `pub(crate)` (existing bare dispatchers retained).
- **Cat 2 depth (13):** `band`, `modified_band`, `modified_epigraph_index`, `extreme_rank_length_depth`, `extremal_depth`, `half_region_depth`, `modified_half_region_depth`, `hypograph_index`, `epigraph_index`, `modified_hypograph_index`, `linfinity_depth`, `rpd_depth`, `total_variation_depth` — new `Dim` dispatchers.
- **Cat 2 fdata (2):** `center`, `norm_lp` dispatchers.
- **Cat 3 metric (20):** `basis_coef`/`deriv`/`dtw`/`fourier`/`hshift`/`kl`/`pca`/`soft_dtw`/`soft_dtw_div` × self+cross — new dispatchers; 8 previously-missing bare names added to `lib.rs`.
- **Plain rename (5):** `fdata_to_pc` (~152 call sites), `fdata_to_pls` (~19), `fourier_fit`, `pspline_fit`, `select_basis_auto` — global symbol renames.
- **Blast radius:** all external callers (10 of 28 examples, `validate_against_r.rs`, benches + group-name strings), the `equivalence_phase50.rs` bit-identity goldens (migrated to dispatcher, still hold), `seasonal/peak.rs` internal `fourier_fit` caller, and the `lib.rs` module-doc doctest.

## Task Commits
1. `cdf1413f` — Cat 1/2 depth + fdata Dim dispatch
2. `81e722f2` — Cat 3 metric self/cross families
3. `ea423cf7` — plain-rename basis + regression
4. `076d9670` — external callers + inline tests migrated
5. `69ebeff3` — cfg(test) gating of test-only `_1d` re-exports
6. `899d566d` — trim_mean doctest → mean dispatcher
7. (metric/mod.rs) — drop unused `soft_dtw_div_cross_1d` test re-export (clippy `--all-targets` fix)

## Decisions Made
- Plain rename (not Dim dispatch) for the 5 genuinely-1D-only functions — **user decision** (clean 1.0 API).
- Visibility-flip mechanic (not `_impl` rename) — keeps `dispatch.rs` + internal callers untouched.
- `pca_cross(data1, data2, ncomp, dim)` — corrected the RESEARCH signature error (no `argvals`).

## Deviations from Plan
### Auto-fixed
**1. RESEARCH `pca_cross_1d` signature error** — research claimed an `argvals` param that doesn't exist in source. The impl matched the real body (`pca_cross(data1, data2, ncomp, dim)`). Verified by verifier + code review.
**2. clippy `--all-targets` unused-import** — the phase's per-task `cargo build -p` gate can't catch test-code warnings; `--all-targets` (CI's gate) flagged an unused `soft_dtw_div_cross_1d` `#[cfg(test)]` re-export. Removed it.
**3. cfg(test) re-exports + example-02 local shadow** — the impl agent added `#[cfg(test)] pub(crate) use` re-exports so inline tests resolve the `_1d` primitives, and renamed a `let mean = mean(...)` local (pre-existing shadow exposed once `mean_1d`→`mean`).
**4. Stray `.planning/tmp/roadmapper-skills.txt`** swept into a commit via `git add -A`; removed as part of finalization.

---
**Total deviations:** 4 auto-fixed; none change scope or behavior. NAME-04 delivered in full.

## Issues Encountered
- `/home` at ~99–100% throughout; freed `target/debug/{incremental,examples}` and routed examples/benches/serde builds to `/tmp` tmpfs.
- The impl agent's per-task `cargo build -p fdars-core` gate could not catch test/bench-code lint issues — the orchestrator's `clippy --all-targets` run caught the one that slipped through (a good argument for always running `--all-targets` at phase end).

## Next Phase Readiness
- Phase 89 (Release Preparation, REL-01) is unblocked. The **entire API section** of the 1.0 checklist is now cleared (SEAL + all NAME items). Phase 89 must document ALL of these breaking changes in the `[0.42.0]` CHANGELOG: sealed `wire`/configs (+ the construction-idiom change), the Phase-87 consolidations + LpeerResult→LocalPeerResult, and this phase's 41 consolidations + 5 plain-renames.

---
*Phase: 88-large-suffix-batch*
*Completed: 2026-09-08*
