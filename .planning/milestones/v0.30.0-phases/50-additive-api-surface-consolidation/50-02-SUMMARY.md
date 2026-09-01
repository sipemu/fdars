---
phase: 50-additive-api-surface-consolidation
plan: 02
subsystem: function_on_scalar
tags: [api, additive, deprecated, fanova, reproducibility, lcg, golden]
requires:
  - phase: 50
    plan: 01
    provides: additive+examples+wasm+clippy gate pipeline proven
provides:
  - fanova_seeded(data, groups, n_perm, seed) — reproducible functional ANOVA (API-01/02)
  - fanova() kept as #[deprecated(since="0.30.0")] delegating shim (seed=42) — bit-identical (API-03)
  - First in-crate use of #[deprecated] + the deprecation-hygiene pattern (migrate or #[allow])
  - tests/equivalence_phase50.rs (fanova goldens; extended by 50-03)
affects: [50-03]
actuals:
  tokens: 60000
  tasks: 3
  commits: 3
tech-stack:
  added: []
  patterns: [deprecated-delegating-shim, LCG-seed-threading, golden-capture-before-change, deprecation-hygiene (#[allow(deprecated)] on behavior-pinning tests)]
key-files:
  created:
    - fdars-core/tests/equivalence_phase50.rs
  modified:
    - fdars-core/src/function_on_scalar.rs
    - fdars-core/src/lib.rs
    - fdars-core/examples/21_function_on_scalar/main.rs
    - fdars-core/src/inference/anova.rs
    - fdars-core/src/inference/permutation.rs
    - fdars-core/tests/validate_new_modules.rs
key-decisions:
  - "fanova_seeded KEEPS the hand-rolled LCG (multiplier 6_364_136_223_846_793_005), threading `seed` into rng_state — NOT StdRng (seed_from_u64(42) != the LCG stream → would change p-values)."
  - "fanova() becomes a #[deprecated(since=0.30.0, note=…)] shim delegating to fanova_seeded(…, 42) → bit-identical to the pre-change fanova (golden-proven)."
  - "Deprecation hygiene (first #[deprecated] in-crate): example 21 + integration tests migrated to fanova_seeded; unit tests that intentionally pin the OLD fanova get #[allow(deprecated)] + local use — keeps clippy --all-targets -D warnings green."
patterns-established:
  - "Additive deprecation: add the canonical form, make the old a #[deprecated] delegating shim (behavior-preserving), migrate non-pinning callers, #[allow(deprecated)] the pin-tests."
requirements-completed: [API-01 (fanova consistency), API-02 (fanova_seeded), API-03 (back-compat)]
coverage:
  - id: D1
    description: "fanova_seeded reproducible + fanova shim bit-identical to pre-change output"
    requirement: API-02
    verification:
      - kind: integration
        ref: "equivalence_phase50: fanova_seeded_seed42_bit_identical + fanova_shim_seed42_bit_identical + fanova_seeded_different_seed_changes_pvalue_not_statistic — 3/3 pass both configs"
        status: pass
    human_judgment: false
  - id: D2
    description: "API-03 back-compat: 28 examples build (deprecation-free), wasm compiles, clippy --all-targets clean despite new #[deprecated]"
    requirement: API-03
    verification:
      - kind: integration
        ref: "cargo build --examples (all 28, no deprecation warnings); wasm32 --features js compiles; clippy --all-targets --features linalg,parallel -D warnings clean; full suite both configs green"
        status: pass
    human_judgment: false
---

# Plan 50-02 SUMMARY — fanova_seeded (reproducible functional ANOVA)

## What shipped

- **`fanova_seeded(data, groups, n_perm, seed)`** (`function_on_scalar.rs:804`) — the reproducible
  canonical form. Threads `seed` into the existing hand-rolled LCG (`rng_state = seed`), preserving
  the exact multiplier `6_364_136_223_846_793_005`. Matches the sibling `(…, n_perm, seed)` convention.
- **`fanova(...)` → `#[deprecated(since="0.30.0", note="use fanova_seeded for reproducible
  permutation p-values")]`** delegating shim calling `fanova_seeded(…, 42)` — bit-identical to the
  pre-change `fanova` (the old code hardcoded `rng_state=42`).
- **`tests/equivalence_phase50.rs`** — goldens capturing the pre-change `fanova` output
  (`global_statistic` + `p_value`) via `assert_eq!`, plus a seed-threading test (different seed ⇒
  different p_value, same statistic).
- **Deprecation hygiene** (crate's first `#[deprecated]`): example 21 + the `validate_new_modules`
  integration tests migrated to `fanova_seeded`; the anova/permutation unit tests that intentionally
  cross-check the OLD `fanova` decision carry `#[allow(deprecated)]`. `lib.rs` re-exports both.

## Evidence

| Gate | Result |
|------|--------|
| fanova goldens (both configs) | ✅ 3/3 (shim bit-identical + seed threads) |
| Full suite both configs | ✅ 18 ok-blocks each, 0 fail |
| 28 examples build | ✅ all, no deprecation warnings (ex-21 migrated) |
| wasm32 `--features js` | ✅ compiles |
| clippy --all-targets -D warnings | ✅ clean (deprecation hygiene holds) |

## Resume note

Plan 50-02's Task 3 (deprecation-hygiene) was executed by a subagent that the operator killed just
after it passed the config-B suite and while building examples/wasm — before writing this SUMMARY.
On resume the uncommitted hygiene work was gate-verified (all gates re-run green) and committed
(`57de3ff0`), then this SUMMARY written. Tasks 1–2 had already committed (`80a87ad0` golden,
`8d40727c` fanova_seeded + deprecate). Next: plan 50-03 (Dim dispatchers) extends equivalence_phase50.rs.

## Deviation

- None functional. Reconciliation-on-resume only: the killed executor's completed-but-uncommitted
  Task 3 changes were verified and committed rather than re-executed (avoids duplicate work).
