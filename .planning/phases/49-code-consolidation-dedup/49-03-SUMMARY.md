---
phase: 49-code-consolidation-dedup
plan: 03
subsystem: helpers
tags: [refactor, dedup, consolidation, seeded-rng, determinism, golden-equivalence, CONS-02]
requires:
  - phase: 49
    plan: 01
    provides: tests/equivalence_phase49.rs shared golden harness + #[doc(hidden)] __equivalence_test_support forwarding surface
  - phase: 48
    provides: equivalence_phaseNN.rs bit-identical golden pattern (assert_eq!, both feature configs)
provides:
  - "helpers::seed_for_thread(seed, k) — the single pub(crate) home for the per-thread determinism contract, body EXACTLY StdRng::seed_from_u64(seed.wrapping_add(k as u64))"
  - "tests/equivalence_phase49.rs rng_stream golden — helper draws bit-identical to seed_from_u64(seed+k) for seed=42, k in {0,1,3}"
  - CONS-02 seeded-RNG consolidation — all 14 exact thread-offset sites migrated, bit-identical under both feature configs
affects: []
actuals:
  tokens: 26000
  tasks: 2
  commits: 3
tech-stack:
  added: []
  patterns:
    - "one pub(crate) seed_for_thread(seed,k) helper (wrapping_add body) replaces 14 hand-rolled StdRng::seed_from_u64(seed+k) / .wrapping_add(k) sites — bit-identical for non-overflowing inputs"
    - "rng_stream golden pins the helper stream (first 8 u64 draws) against the captured pre-refactor seed_from_u64(seed+k) formula, independent of the per-site downstream goldens"
    - "test reach via #[doc(hidden)] __equivalence_test_support::helpers::seed_for_thread_draws forwarder (returns drawn u64s, not the crate-private StdRng)"
key-files:
  created: []
  modified:
    - fdars-core/src/helpers.rs
    - fdars-core/src/lib.rs
    - fdars-core/tests/equivalence_phase49.rs
    - fdars-core/src/alignment/generative.rs
    - fdars-core/src/outliers.rs
    - fdars-core/src/scalar_on_function/bootstrap.rs
    - fdars-core/src/tolerance/fpca.rs
    - fdars-core/src/tolerance/equivalence.rs
    - fdars-core/src/spm/bootstrap.rs
    - fdars-core/src/elastic_explain.rs
key-decisions:
  - "Helper body is EXACTLY StdRng::seed_from_u64(seed.wrapping_add(k as u64)) — the determinism contract. wrapping_add is bit-identical to plain seed + k for all non-overflowing inputs and matches the .wrapping_add sites verbatim, so both `+` and `.wrapping_add` spellings migrate without changing a single draw."
  - "The verified thread-offset set is 14 sites (RESEARCH estimated ~10). Migrated: generative.rs:108,276; outliers.rs:156; scalar_on_function/bootstrap.rs:89,212; tolerance/fpca.rs:68,112; tolerance/equivalence.rs:29,56,83,108; spm/bootstrap.rs:253; elastic_explain.rs:314,384."
  - "Deliberately NOT migrated (different contract or another plan's ownership): outliers.rs:987 (plain seed, test); frechet/anova.rs:180,272 (plan 49-04 permutation); explain_generic/importance.rs:66 + shap.rs:78 (their owning plans — left untouched, not confirmed pure thread-offset here); spm/arl.rs:177,248,301 + alignment/shape_ci.rs:121 + boosting_regression/stability.rs:137 (config.seed/stab_config.seed offset sites, NOT in this plan's files_modified); tolerance/degras.rs:131 (hardcoded 42 + b base); and all ~88 plain-`seed` single-RNG and LCG-seed calls."
  - "Call sites use the fully-qualified crate::helpers::seed_for_thread(seed, k) inline rather than adding a `use` per file — minimizes import churn and avoids unused-import risk where a file no longer references StdRng."
  - "Test reach: added a #[doc(hidden)] __equivalence_test_support::helpers::seed_for_thread_draws forwarder that returns the first n drawn u64s (via RngCore::next_u64) rather than the pub(crate) StdRng, so the crate-private RNG type never escapes while the exact stream is still exercised. The forwarder also keeps seed_for_thread live, so clippy reports no dead_code."
patterns-established:
  - "Bit-identical rng_stream golden (assert_eq! on u64 draws) captured from the pre-refactor seed+k formula; must hold under --features linalg,parallel AND --no-default-features --features linalg."
requirements-completed: [CONS-02]
coverage:
  - id: R1
    description: "seed_for_thread(seed,k) draws bit-identical to StdRng::seed_from_u64(seed+k) for fixed (seed,k) pairs — the determinism contract pinned on the helper alone"
    requirement: CONS-02
    verification:
      - kind: integration
        ref: "cargo test --test equivalence_phase49 rng_stream under --features linalg,parallel AND --no-default-features --features linalg => 1/1 pass both configs"
        status: pass
    human_judgment: false
  - id: R2
    description: "All 14 exact thread-offset sites migrated to seed_for_thread; RNG stream + downstream numeric output bit-identical under both feature configs"
    requirement: CONS-02
    verification:
      - kind: integration
        ref: "full seed-dependent suite (bootstrap CIs, tolerance bands, outliers, generative alignment) green under BOTH configs => 2583 lib each + integration + doctests, 0 fail"
        status: pass
    human_judgment: false
  - id: R3
    description: "Only exact thread-offset form migrated; plain-seed, LCG, and permutation/config-owned sites NOT touched"
    requirement: CONS-02
    verification:
      - kind: integration
        ref: "grep confirms outliers.rs:987 plain-seed, frechet/anova.rs, explain_generic reseeds, spm/arl.rs, shape_ci, degras, boosting stability all unchanged; 14 seed_for_thread call sites total"
        status: pass
    human_judgment: false
  - id: R4
    description: "Wave-3 gate: clippy --all-targets clean; no public signature change; no new dependency"
    requirement: CONS-02
    verification:
      - kind: lint
        ref: "cargo clippy --all-targets --features linalg,parallel -- -D warnings => clean (after dropping now-unused StdRng/SeedableRng imports in spm/bootstrap.rs)"
        status: pass
    human_judgment: false
status: complete
---

# Phase 49 Plan 03: Seeded-RNG Consolidation (CONS-02) Summary

Consolidated the per-thread determinism contract `seed + k` — hand-rolled at 14 parallel-loop
sites as `StdRng::seed_from_u64(seed + k as u64)` / `seed_from_u64(seed.wrapping_add(k as u64))` —
into ONE `pub(crate)` helper `helpers::seed_for_thread(seed, k)` whose body is EXACTLY
`StdRng::seed_from_u64(seed.wrapping_add(k as u64))`. `wrapping_add` is bit-identical to the plain
`seed + k as u64` form for all non-overflowing inputs and matches the `.wrapping_add` sites
verbatim, so both spellings migrated without changing a single RNG draw. An `rng_stream` golden
pins the helper's stream against the captured pre-refactor formula, and the full seed-dependent
suite (bootstrap CIs, tolerance bands, outlier detection, generative alignment) proves every
migrated site is bit-identical under both feature configs.

## What shipped

- **`src/helpers.rs`**: `pub(crate) fn seed_for_thread(seed: u64, k: usize) -> StdRng` — body EXACTLY
  `StdRng::seed_from_u64(seed.wrapping_add(k as u64))`, `#[inline]`, self-contained (fully-qualified
  `rand::rngs::StdRng` + `rand::SeedableRng`, no new import block).
- **`src/lib.rs`**: added a `#[doc(hidden)] __equivalence_test_support::helpers::seed_for_thread_draws`
  forwarder returning the first `n` drawn `u64`s (via `RngCore::next_u64`) — test-only reach into the
  `pub(crate)` helper without letting the crate-private `StdRng` escape and without widening any real
  signature.
- **`tests/equivalence_phase49.rs`** (appended): the `rng_stream` golden. First-8-`u64`-draws consts
  captured verbatim from the pre-refactor `seed_from_u64(42 + k)` formula for `k ∈ {0,1,3}`, asserted
  `assert_eq!` bit-identical against the helper's stream. Gates the contract on the helper alone.
- **Migrated 14 thread-offset sites** onto `crate::helpers::seed_for_thread(seed, k)` (fully-qualified,
  inline) — see the migrated set below. Dropped the now-unused `StdRng` / `SeedableRng` imports in
  `spm/bootstrap.rs` (its only thread-offset site was migrated; `rand::Rng` retained for `gen_range`).

## Migrated sites (the exact thread-offset set — 14 total)

| File | Lines | Loop var | Pre-refactor form |
|------|-------|----------|-------------------|
| `alignment/generative.rs` | 108, 276 | `i` | `seed + i as u64` |
| `outliers.rs` | 156 | `b` | `seed.wrapping_add(b as u64)` |
| `scalar_on_function/bootstrap.rs` | 89, 212 | `b` | `seed.wrapping_add(b as u64)` |
| `tolerance/fpca.rs` | 68, 112 | `b` | `seed + b as u64` |
| `tolerance/equivalence.rs` | 29, 56, 83, 108 | `b` | `seed + b as u64` |
| `spm/bootstrap.rs` | 253 | `rep` | `seed + rep as u64` |
| `elastic_explain.rs` | 314, 384 | `p` | `seed.wrapping_add(p as u64)` |

## Deliberately NOT migrated

- `outliers.rs:987` — plain `seed_from_u64(seed)` single-RNG (test module), not a thread offset.
- `frechet/anova.rs:180,272` — permutation reseeds owned by plan 49-04.
- `explain_generic/importance.rs:66` + `shap.rs:78` — per-component/per-`i` reseeds left to their
  owning plans (not confirmed as pure thread-offset with no other coupling here — the plan preferred
  leaving them).
- `spm/arl.rs:177,248,301`, `alignment/shape_ci.rs:121`, `boosting_regression/stability.rs:137` —
  `config.seed`/`stab_config.seed` offset sites, NOT in this plan's `files_modified`.
- `tolerance/degras.rs:131` — hardcoded `42 + b` base (not a caller `seed`).
- All ~88 plain-`seed` single-RNG and `.wrapping_mul` LCG-seed calls — different contract.

## Commit count

3 atomic commits:
- `4b5da143` feat(49-03): add pub(crate) seed_for_thread(seed,k) + rng_stream golden
- `0d09adfa` refactor(49-03): migrate thread-offset RNG sites to seed_for_thread (CONS-02)
- (this SUMMARY commit)

## Golden results (both feature configs)

| Config | `equivalence_phase49 rng_stream` | Full suite (lib) | clippy --all-targets |
|--------|----------------------------------|------------------|----------------------|
| `--features linalg,parallel` | 1/1 pass | 2583 pass, 0 fail | clean |
| `--no-default-features --features linalg` | 1/1 pass | 2583 pass, 0 fail | clean |

Every seed-dependent test (bootstrap CIs, tolerance bands, outlier detection, generative alignment)
stayed bit-identical — those tests ARE the per-site equivalence gate, and they pass unchanged under
both configs.

## Deviations from Plan

- **[Rule 3 — Blocking] Unused imports after migration.** `spm/bootstrap.rs` had explicit
  `use rand::rngs::StdRng;` and `use rand::{Rng, SeedableRng};`. Its only `StdRng::seed_from_u64`
  usage was the migrated site, so both `StdRng` and `SeedableRng` became unused (clippy `-D warnings`
  error). Dropped them, keeping `use rand::Rng;` (still used by `gen_range`). Directly caused by the
  migration; no behavior change.
- **Site count.** RESEARCH estimated ~10 thread-offset sites; the verified exact set is 14 (the plan's
  candidate list was correct — it enumerated all 14 across the seven files). No sites outside the
  candidate list were found or migrated.
- **explain_generic reseeds left untouched.** The plan permitted the executor to migrate
  `explain_generic/importance.rs:66` + `shap.rs:78` IF confirmed pure thread-offset. They were left to
  their owning plans (the plan's stated preference) to avoid coupling this consolidation to the
  importance/shap logic.

## Known Stubs

None. The helper is a real one-line delegator; every migrated site calls it with the correct loop
index. No placeholders, empty returns, or TODOs introduced.

## Threat Flags

None. Internal determinism-contract refactor — no new network endpoint, auth path, file access, or
schema at a trust boundary. Threat T-49-04 (RNG-stream / migrated-site output tampering) was
mitigated exactly as the register prescribed: the `rng_stream` golden (`assert_eq!` draws ==
`seed_from_u64(seed+k)`) plus the full seed-dependent suite bit-identical under both configs, before
and after migration. Threat T-49-05 (over-migration onto plain-`seed`/LCG sites) mitigated by
migrating ONLY the exact thread-offset form and explicitly excluding plain-seed, LCG, and
permutation/config-owned sites (verified by grep).

## Self-Check: PASSED

- `fdars-core/src/helpers.rs::seed_for_thread` — FOUND
- `fdars-core/tests/equivalence_phase49.rs::rng_stream_seed_for_thread_bit_identical` — FOUND
- Commit `4b5da143` — FOUND
- Commit `0d09adfa` — FOUND
