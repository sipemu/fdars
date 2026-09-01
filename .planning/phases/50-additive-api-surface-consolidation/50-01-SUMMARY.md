---
phase: 50-additive-api-surface-consolidation
plan: 01
subsystem: boosting_regression
tags: [additive, api-consolidation, default-impl, ergonomics, tracer, API-01, API-03]
requires:
  - phase: 46
    provides: PROF-03 api-inventory (missing-Default configs ranked as item #1, additive-safe, HIGH)
provides:
  - "src/boosting_regression/mod.rs — impl Default for BoostingConfig / BayesianConfig / StabilityConfig (additive trait impls; no signature change)"
  - "boosting_regression::tests::config_defaults_match_documented_values — pins the documented default values (assert_eq! vs struct-literal) so a silent future default drift is caught"
  - "The full additive+examples+wasm+clippy+fmt gate pipeline proven end-to-end (the Phase-50 tracer) for the deprecation-introducing plans 50-02 (fanova_seeded) and 50-03 (Dim dispatch)"
affects: [50-02, 50-03]
actuals:
  tokens: 6500
  tasks: 2
  commits: 3
tech-stack:
  added: []
  patterns:
    - "additive Default impl: new `impl Default for Config` blocks placed immediately after each struct; purely additive (a new trait impl can never break existing construction) — no behavior golden exists because these configs only take effect when explicitly constructed"
    - "documented-default pin test (assert_eq! Config::default() vs struct-literal) so a silent default change is caught by a compile-checked test"
key-files:
  created: []
  modified:
    - fdars-core/src/boosting_regression/mod.rs
key-decisions:
  - "Shipped 3 impl Default blocks, NOT 4. The PROF-03 inventory listed StlConfig as a fourth missing-Default config, but the live tree shows StlConfig ALREADY `#[derive(Debug, Clone, PartialEq, Default)]` at src/detrend/stl.rs:47 — adding an impl would fail with error[E0119]: conflicting implementations. StlConfig was NOT touched."
  - "CONTEXT claimed the 3 boosting configs are `#[non_exhaustive]`; the live tree shows they are NOT (only the Result structs carry `#[non_exhaustive]`). This makes the struct-literal `Self { .. }` in each impl body valid and is irrelevant to additivity — adding Default is additive regardless."
  - "Default field values: doc-VERIFIED where the field doc-comment states a default; the [ASSUMED] numerics adopt the value the most-representative constructor uses. Any in-range value is back-compat-safe (no behavior golden), so [ASSUMED] carries no risk here."
  - "BayesianConfig defaults ncomp=4 / n_iter=400 / burn_in=200 confirmed to match bayesian::tests::default_config() verbatim; StabilityConfig n_resamples=100 / pi_thr=0.9 are doc-VERIFIED ('default: 100' / 'default: 0.9'); BoostingConfig mstop=100 is the FDboost convention (nu=0.1 / nbasis=10 / order=4 / lfd_order=2 / lambda=1.0 / ncomp_x=3 are doc-VERIFIED from field docs / doc-examples)."
requirements-completed: [API-01 (item #1 — Default impls), API-03 (back-compat: examples + wasm compile, deprecation-warning-free — none introduced here)]
coverage:
  - id: D1
    description: "BoostingConfig / BayesianConfig / StabilityConfig each gain an additive impl Default with the documented values; StlConfig untouched (no E0119)"
    requirement: API-01
    verification:
      - kind: unit
        ref: "boosting_regression::tests::config_defaults_match_documented_values — assert_eq! each Config::default() vs the documented struct-literal; passes under --features linalg,parallel AND --no-default-features --features linalg"
        status: pass
    human_judgment: false
  - id: D2
    description: "Additive-safe pipeline proven end-to-end (the Phase-50 tracer): both feature configs green, all 28 examples build, wasm compiles, clippy --all-targets -D warnings clean, fmt clean"
    requirement: API-03
    verification:
      - kind: integration
        ref: "cargo test boosting_regression under both configs => pin test 1/1 each; cargo build --examples --features linalg,parallel => all 28 built; cargo build --target wasm32-unknown-unknown --features js --no-default-features => compiled"
        status: pass
      - kind: lint
        ref: "cargo clippy --all-targets --features linalg,parallel -- -D warnings => clean"
        status: pass
    human_judgment: false
status: complete
---

# Phase 50 Plan 01: Boosting-Config Default Impls (API-01 Tracer) Summary

Added `impl Default` to the three `src/boosting_regression/mod.rs` config structs that genuinely
lacked it — `BoostingConfig`, `BayesianConfig`, `StabilityConfig` — using the documented field
defaults, giving users `Config::default()` ergonomics (52/56 configs already had it). Because a
`Default` trait impl is purely additive and cannot change any existing output, this is the Phase-50
tracer: it exercises the FULL additive+gate machinery (build → both-config suite → clippy
--all-targets -D warnings → 28-example build → wasm compile → fmt → `--no-verify` commit) at the
lowest possible risk, proving the pipeline before the deprecation-introducing plans 50-02
(`fanova_seeded`) and 50-03 (`Dim` dispatch).

## What shipped

- **3 additive `impl Default` blocks** (NOT 4), each placed immediately after its struct:
  - `BoostingConfig::default()` → `{ mstop: 100, nu: 0.1, nbasis: 10, order: 4, lfd_order: 2, lambda: 1.0, ncomp_x: 3, seed: 0 }`
  - `BayesianConfig::default()` → `{ ncomp: 4, tau2: 100.0, ig_a0: 0.001, ig_b0: 0.001, n_iter: 400, burn_in: 200, thin: 1, seed: 0 }`
  - `StabilityConfig::default()` → `{ n_resamples: 100, pi_thr: 0.9, seed: 0 }`
- **`StlConfig` was NOT touched** — it already `#[derive(..., Default)]` at `src/detrend/stl.rs:47`;
  adding an impl would be an `error[E0119]: conflicting implementations`. This is the RESEARCH
  correction to the PROF-03 inventory (which had listed it as a fourth target).
- **A documented-default pin test** (`config_defaults_match_documented_values`) asserting each
  `::default()` equals a struct-literal of the values above, so a future silent default drift is
  caught by a compile-checked test.

## Default values: doc-VERIFIED vs [ASSUMED]

| Config | Field | Value | Source |
|--------|-------|-------|--------|
| BoostingConfig | nu / nbasis / order / lfd_order / lambda / ncomp_x | 0.1 / 10 / 4 / 2 / 1.0 / 3 | doc-VERIFIED (field docs / doc-examples) |
| BoostingConfig | mstop | 100 | [ASSUMED] — FDboost convention (test uses 5 for speed; doc mandates a fixed mstop) |
| BoostingConfig | seed | 0 | matches stability.rs `default_boost` |
| BayesianConfig | tau2 / ig_a0 / ig_b0 / thin | 100.0 / 0.001 / 0.001 / 1 | doc-VERIFIED (field docs; thin min) |
| BayesianConfig | ncomp / n_iter / burn_in | 4 / 400 / 200 | [ASSUMED] — confirmed to match `bayesian::tests::default_config()` verbatim |
| BayesianConfig | seed | 0 | [ASSUMED] isolation-only |
| StabilityConfig | n_resamples / pi_thr | 100 / 0.9 | doc-VERIFIED ("default: 100" / "default: 0.9") |
| StabilityConfig | seed | 0 | [ASSUMED] isolation-only |

No [ASSUMED] value needed a genuine judgment call: every one either matches an existing in-tree
constructor (`bayesian::tests::default_config`, `stability::tests::default_boost`) or is a
back-compat-safe in-range value — and since these configs only take effect when explicitly
constructed, there is no behavior golden any choice could break.

## Commit count

3 atomic commits:
- `f34c91e1` feat(50-01): add impl Default for 3 boosting configs (item #1, additive)
- `cedd1068` test(50-01): pin documented ::default() values for 3 boosting configs
- (this SUMMARY commit)

## Gate results

| Gate | Config A (`--features linalg,parallel`) | Config B (`--no-default-features --features linalg`) |
|------|------------------------------------------|-------------------------------------------------------|
| pin test (`config_defaults...`) | 1/1 pass | 1/1 pass |
| `cargo build` | clean | (lib built via clippy/test) |
| 28-example build | all 28 built (no disk/link failure) | — |
| wasm (`--target wasm32-unknown-unknown --features js`) | compiled | — |
| `clippy --all-targets -- -D warnings` | clean | — |
| `cargo fmt -p fdars-core` | clean | — |

Full additive+examples+wasm+clippy+fmt pipeline is now proven green — the tracer is complete and
plans 50-02 / 50-03 can build on it.

## Deviations from Plan

- **[Rule 3 - Blocking] clippy `items_after_test_module`.** The initial placement of the new
  `#[cfg(test)] mod tests` block left a `pub use self::stability::stability_selection;` barrel
  re-export *after* the test module, which `clippy --all-targets -D warnings` rejects
  (`clippy::items_after_test_module`). Resolved by moving the test module to the very end of the
  file, after all barrel re-exports. No functional change; the pin test still passes under both
  configs.
- **[Note] Struct visibility.** CONTEXT asserted the 3 boosting configs are `#[non_exhaustive]`;
  the live tree shows they are not (only the Result structs are). This made the `Self { .. }`
  struct-literal in each impl body directly valid — no adjustment needed, and it does not affect
  additivity (a new trait impl is additive regardless).

Otherwise the plan was executed exactly: 3 impls (not 4), StlConfig untouched, documented values
pinned, both-config + examples + wasm + clippy gates green, `--no-verify` commits with `cargo fmt`
per commit.

## Known Stubs

None. Each `impl Default` returns a concrete struct-literal of real documented values; no
placeholders, empty returns, or TODOs.

## Threat Flags

None. Additive trait impl on a pure numerical library — no new network endpoint, auth path, file
access, or schema at a trust boundary. Threat T-50-01 (default-value tampering) mitigated exactly
as the register prescribed: the `::default()` pin test asserts the documented values under both
feature configs. Threat T-50-02 (28-example build fills /home) did not trigger — the example build
completed with no disk/link failure, so the `rm -rf target/debug/{incremental,examples}` retry was
not needed.

## Self-Check: PASSED

- `fdars-core/src/boosting_regression/mod.rs` (3 impl Default + pin test) — FOUND
- Commit `f34c91e1` — FOUND
- Commit `cedd1068` — FOUND
