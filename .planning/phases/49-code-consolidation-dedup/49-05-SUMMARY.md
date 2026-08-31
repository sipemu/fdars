---
phase: 49-code-consolidation-dedup
plan: 05
subsystem: helpers
tags: [refactor, dedup, consolidation, seeded-rng, determinism, gap-closure, CONS-02]
requires:
  - phase: 49
    plan: 03
    provides: "helpers::seed_for_thread(seed,k) pub(crate) determinism helper + tests/equivalence_phase49.rs rng_stream golden"
provides:
  - "CONS-02 gap closure — the 5 residual same-contract thread-offset RNG sites (spm/arl.rs x3, alignment/shape_ci.rs, boosting_regression/stability.rs) plus explain_generic/importance.rs migrated to seed_for_thread; ROADMAP 'every call site migrated' now met for the exact thread-offset contract"
  - "explain/shap.rs in-code Phase-49 CONS-02 rationale comment documenting why its single plain-seed sequential RNG is intentionally NOT migrated"
affects: []
actuals:
  tokens: 9000
  tasks: 1
  commits: 2
tech-stack:
  added: []
  patterns:
    - "seed_for_thread(seed,k) now consumes ALL exact StdRng::seed_from_u64(seed_field + loop_var) thread-offset sites; residual RNG sites are single-plain-seed (sequential) and carry an in-code exclusion rationale"
key-files:
  created: []
  modified:
    - fdars-core/src/spm/arl.rs
    - fdars-core/src/alignment/shape_ci.rs
    - fdars-core/src/boosting_regression/stability.rs
    - fdars-core/src/explain_generic/importance.rs
    - fdars-core/src/explain/shap.rs
key-decisions:
  - "explain_generic/importance.rs:66 MIGRATED — it is the exact per-component thread offset StdRng::seed_from_u64(seed.wrapping_add(k as u64)) inside iter_maybe_parallel!(0..ncomp), with `seed` a plain u64 param; identical contract to the other migrated sites."
  - "explain/shap.rs EXCLUDED with in-code rationale — the verifier's 'shap.rs:78' reseed does not exist; the only RNG in fpc_shap_values_logistic is StdRng::seed_from_u64(seed) at line ~192, a single RNG advanced sequentially across observations (NOT a seed+k thread offset). Migrating would change the coalition-sampling stream and the SHAP values. Added a 4-line Phase-49 CONS-02 comment mirroring the 49-04 exclusion style."
  - "explain_generic/importance.rs:151 left untouched — plain StdRng::seed_from_u64(seed) single sequential RNG (conditional-permutation path), not a thread offset; the rand::prelude::* glob import stays because that site still uses StdRng."
  - "No new golden added. All 5 migrated sites use the identical seed_for_thread(seed_field, loop_var) form whose (seed+k) offset contract is already pinned by the existing equivalence_phase49 rng_stream golden. Per-site bit-identity is re-proven by each module's own seed-dependent suite (ARL sims, bootstrap shape CIs, stability selection, permutation importance) passing unchanged under both configs."
  - "Dropped now-unused imports: arl.rs SeedableRng (StdRng kept for sample_gamma param, Rng kept for .sample); shape_ci.rs StdRng+SeedableRng (Rng kept for .gen_range); stability.rs StdRng+SeedableRng (Rng kept for .gen_range). Mirrors plan 49-03's spm/bootstrap.rs cleanup."
requirements-completed: [CONS-02]
status: complete
---

# Phase 49 Plan 05: Seeded-RNG Gap Closure (CONS-02) Summary

Closed the CONS-02 verification gap: five genuine same-contract per-thread seeded-RNG sites were
missed by plan 49-03 because they fell outside that plan's `files_modified` (a scoping artifact, not
a contract difference). All are the exact `StdRng::seed_from_u64(seed_field + loop_var as u64)` /
`.wrapping_add(loop_var as u64)` pattern inside `iter_maybe_parallel!` loops that
`helpers::seed_for_thread` was built to eliminate — bit-identical because `wrapping_add == +` for the
small non-overflowing seeds these sites use. Migrated them, plus the one explain_generic per-component
reseed that is also a pure thread offset. Made an explicit, traced decision on the two verifier-flagged
explain sites.

## Sites migrated (6 total)

| File | Line(s) | Pre-refactor form | Loop var | New call |
|------|---------|-------------------|----------|----------|
| `spm/arl.rs` | 177, 248, 301 | `StdRng::seed_from_u64(config.seed + rep as u64)` | `rep` | `crate::helpers::seed_for_thread(config.seed, rep)` |
| `alignment/shape_ci.rs` | 121 | `StdRng::seed_from_u64(config.seed + b as u64)` | `b` | `seed_for_thread(config.seed, b)` |
| `boosting_regression/stability.rs` | 137 | `StdRng::seed_from_u64(stab_config.seed.wrapping_add(b as u64))` | `b` | `seed_for_thread(stab_config.seed, b)` |
| `explain_generic/importance.rs` | 66 | `StdRng::seed_from_u64(seed.wrapping_add(k as u64))` | `k` (component) | `seed_for_thread(seed, k)` |

All four files are `iter_maybe_parallel!` per-replicate / per-component loops with a plain `u64` seed
field offset by the loop index — the exact contract `seed_for_thread` owns.

## Sites explicitly excluded, with in-code rationale (no silent skip)

- **`explain/shap.rs`** (`fpc_shap_values_logistic`, ~line 192): the ONLY RNG is
  `StdRng::seed_from_u64(seed)` — a single plain-`seed` RNG advanced **sequentially across all `n`
  observations**, not a `seed + k` per-thread offset. The verifier's "shap.rs:78" thread-offset reseed
  does not exist in the file. Migrating (reseeding per observation) would change the coalition-sampling
  stream and therefore the SHAP values. Left un-migrated with a 4-line Phase-49 CONS-02 rationale
  comment (mirrors the 49-04 exclusion-comment style).
- **`explain_generic/importance.rs:151`** (conditional-permutation path): also a single plain-`seed`
  sequential `StdRng::seed_from_u64(seed)`, not a thread offset — untouched (and it keeps the
  `rand::prelude::*` import live).

## Golden / test evidence

- **No new golden required.** The existing `equivalence_phase49::rng_stream_seed_for_thread_bit_identical`
  golden already pins `seed_for_thread(seed, k) == seed_from_u64(seed + k)` for `seed=42, k ∈ {0,1,3}`.
  Every migrated site uses the identical `seed_for_thread(field, loop_var)` form, so no new offset form
  is introduced. Verified unchanged: `1 passed` under both configs.
- **Per-site bit-identity** is re-proven by each migrated module's own seed-dependent suite passing
  unchanged: ARL simulations (spm), bootstrap shape confidence intervals (alignment), stability
  selection (boosting), and permutation importance (explain_generic). Those tests ARE the per-site
  equivalence gate.

## Suite results — BOTH configs green

| Config | rng_stream golden | Full lib suite | Integration + doctests |
|--------|-------------------|----------------|------------------------|
| `--features linalg,parallel` | 1/1 pass | 2586 pass, 0 fail | all pass, 0 fail |
| `--no-default-features --features linalg` | 1/1 pass | 2586 pass, 0 fail | all pass, 0 fail |

clippy `--all-targets --features linalg,parallel -- -D warnings`: **clean** (the CI-representative
gate per project MEMORY). No unused-import warnings after dropping now-unused `StdRng`/`SeedableRng`.

## Commit count

2 commits:
- `c2444ee5` refactor(49-05): migrate 5 residual thread-offset RNG sites to seed_for_thread (CONS-02)
- (this SUMMARY commit)

## Deviations from Plan

- **[Rule 3 — Blocking] Unused imports after migration.** Dropped `SeedableRng` from `arl.rs`
  (kept `StdRng` for the `sample_gamma(rng: &mut StdRng, ...)` signature and `Rng` for `.sample`),
  and dropped both `StdRng` + `SeedableRng` from `shape_ci.rs` and `stability.rs` (their only
  `seed_from_u64` usage was the migrated site; `Rng` retained for `.gen_range`). Directly caused by the
  migration; no behavior change. Mirrors plan 49-03's spm/bootstrap.rs cleanup.
- **Verifier's "shap.rs:78" is a phantom reference.** No thread-offset reseed exists there; the file's
  only RNG is a plain sequential `seed_from_u64(seed)`. Resolved by the exclusion rationale above
  rather than a migration.

## Deferred (out of scope)

- `fdars-core/src/parallel.rs:172` — pre-existing `clippy::useless_vec` warning that surfaces ONLY under
  `--no-default-features --features linalg --all-targets`. Confirmed pre-existing (stashing all 49-05
  changes leaves the warning; last commit to that file is `bc7baefa`, unrelated to Phase 49). The
  CI-representative gate (`--features linalg,parallel --all-targets`) is clean. Logged to
  `deferred-items.md`; NOT fixed here (executor scope boundary — 49-05 touches none of `parallel.rs`).

## Known Stubs

None. Every migrated site calls `seed_for_thread` with the correct loop index; the shap.rs exclusion is
a real (unchanged) RNG with a documented rationale.

## Threat Flags

None. Internal determinism-contract refactor — no new network endpoint, auth path, file access, or
schema at a trust boundary. Bit-identity re-proven by the rng_stream golden plus every migrated
module's seed-dependent suite under both feature configs.

## Self-Check: PASSED

- `fdars-core/src/spm/arl.rs` uses `seed_for_thread(config.seed, rep)` x3 — FOUND
- `fdars-core/src/alignment/shape_ci.rs` uses `seed_for_thread(config.seed, b)` — FOUND
- `fdars-core/src/boosting_regression/stability.rs` uses `seed_for_thread(stab_config.seed, b)` — FOUND
- `fdars-core/src/explain_generic/importance.rs:66` uses `seed_for_thread(seed, k)` — FOUND
- `fdars-core/src/explain/shap.rs` Phase-49 CONS-02 exclusion comment — FOUND
- Commit `c2444ee5` — FOUND
