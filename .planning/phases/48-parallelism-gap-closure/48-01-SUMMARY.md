---
phase: 48-parallelism-gap-closure
plan: 01
subsystem: frechet
tags: [perf, parallelism, rayon, frechet_anova, criterion, golden-equivalence, tracer]
requires:
  - phase: 46
    provides: PROF-01 ranked hot-path targets (frechet_anova 133ms #4)
  - phase: 47
    provides: perf-bench + golden-equivalence harness pattern
provides:
  - Permanent proof pipeline (perf_parallelism bench, equivalence_phase48 golden, PERF-PARALLEL-RESULTS.md ledger)
  - frechet_anova parallelized via iter_maybe_parallel! with FRECHET_ANOVA_PERM_PARALLEL_THRESHOLD=200 outer-if
  - PERF-03 tracer: bit-identical output under both feature configs + 9.9× thread-scaling speedup
affects: [48-02, 48-03, 51-benchmark-coverage-regression-guards]
actuals:
  tokens: 24000
  tasks: 4
  commits: 3
tech-stack:
  added: []
  patterns: [golden-equivalence capture-then-assert (bit-identical), permanent thread-scaling bench, payback-threshold outer-if]
key-files:
  created:
    - fdars-core/benches/perf_parallelism.rs
    - fdars-core/tests/equivalence_phase48.rs
    - .planning/phases/48-parallelism-gap-closure/PERF-PARALLEL-RESULTS.md
  modified:
    - fdars-core/Cargo.toml
    - fdars-core/src/frechet/anova.rs
key-decisions:
  - "frechet_anova n_ge loop → iter_maybe_parallel!(0..n_perm).map(count_ge).sum(); per-perm reseed StdRng::seed_from_u64(seed+perm) makes .sum() order-independent → bit-identical."
  - "FRECHET_ANOVA_PERM_PARALLEL_THRESHOLD=200: below-threshold n_perm routes to the sequential branch (same golden); conservative, measurement confirmed no review needed."
  - "DEFER: frechet_anova_space<S: MetricSpace> NOT parallelized (would need S: Sync public-generic widening); concrete frechet_anova already covers the PROF-01 hotspot. Signed off in 48-03."
  - "Governor powersave (sudo/cpupower unavailable) → LOW-CONFIDENCE on absolute medians; 9.9× direction unambiguous."
patterns-established:
  - "Bit-identical golden (assert_eq!, not tolerance) captured from pre-parallel code; must hold under BOTH --features linalg,parallel AND --no-default-features --features linalg."
  - "Thread-scaling proof via RAYON_NUM_THREADS=1 vs =20 env-var sweep on a permanent criterion cell."
requirements-completed: [PERF-03 (frechet_anova path)]
coverage:
  - id: D1
    description: "frechet_anova parallelized bit-identically (goldens assert_eq! under both feature configs; per-perm seeding preserved; signature unchanged)"
    requirement: PERF-03
    verification:
      - kind: integration
        ref: "cargo test equivalence_phase48 under --features linalg,parallel AND --no-default-features --features linalg => golden_frechet_anova_parallel + _below_threshold pass (3/3 both configs)"
        status: pass
    human_judgment: false
  - id: D2
    description: "Thread-scaling speedup measured on permanent perf_parallelism bench"
    requirement: PERF-03
    verification:
      - kind: benchmark
        ref: "perf_parallelism_frechet_anova/n24_m81_nperm999: 1-thread 322.73ms → 20-thread 32.57ms = 9.9× (criterion change -89.8%, p<0.05)"
        status: pass
    human_judgment: false
  - id: D3
    description: "Wave-1 gate: full suite green both configs + clippy --all-targets clean; no new dependency"
    requirement: PERF-03
    verification:
      - kind: integration
        ref: "cargo test -p fdars-core (2583 lib tests) passes under both configs; cargo clippy --all-targets --features linalg,parallel -- -D warnings clean"
        status: pass
    human_judgment: false
---

# Plan 48-01 SUMMARY — frechet_anova Parallelism Tracer

## What shipped

The permanent PERF-03 proof pipeline, driven end-to-end through the primary parallelization
target `frechet_anova` (PROF-01 #4, 133 ms):

- **`fdars-core/benches/perf_parallelism.rs`** — permanent criterion bench, `harness=false` in
  Cargo.toml. Cell `perf_parallelism_frechet_anova/n24_m81_nperm999` (strictly-positive density
  data, no RNG in generator). Becomes a Phase 51 BENCH-02 regression guard.
- **`fdars-core/tests/equivalence_phase48.rs`** — permanent golden-equivalence module. Two
  bit-identical goldens (`assert_eq!`) for the parallel branch (n_perm=999) and the
  below-threshold sequential branch (n_perm=50), both passing under BOTH feature configs.
- **`src/frechet/anova.rs`** — `n_ge` permutation loop swapped to
  `iter_maybe_parallel!(0..n_perm).map(count_ge).sum()` behind a
  `FRECHET_ANOVA_PERM_PARALLEL_THRESHOLD = 200` payback outer-if. Per-perm reseeding preserved
  (determinism linchpin). Public signature byte-identical; `frechet_anova_space` untouched.
- **`PERF-PARALLEL-RESULTS.md`** — environment block, filled frechet_anova medians, threshold
  constants, deferrals.

## Evidence

| Check | Result |
|-------|--------|
| Bit-identical goldens (both feature configs) | ✅ 3/3 parallel-ON, 3/3 parallel-OFF |
| Thread-scaling | ✅ 322.73 ms (1t) → 32.57 ms (20t) = **9.9×**, criterion −89.8% p<0.05 |
| Full suite both configs | ✅ 2583 lib tests each, all integration/doctests green |
| clippy --all-targets | ✅ clean |
| Signature / dependency | ✅ unchanged / none added |

## Resume note

Plan 48-01 is complete (Tasks 1–4). Task 2/3 (golden + src edit) landed pre-interrupt in
`356de77f`; this session finished Task 1 (scaffold, `03a2624c`) and Task 4 (ledger + gate,
`21abda69`). Next: plan 48-02 (co_cluster parallelization), then 48-03 (finalize + deferrals +
sign-off).

## Deviations

- Golden data uses `two_group_densities` (m=20 in tests, m=81 in bench) rather than the plan's
  generic "sinusoid" sketch — `frechet_anova` runs on `WassersteinDensitySpace`, which requires
  strictly-positive rows. No behavioral impact; the density generator is deterministic and
  RNG-free as required.
- Governor could not be pinned to `performance` (no passwordless sudo); medians recorded under
  `powersave` with the LOW-CONFIDENCE caveat. The ~10× speedup direction is unaffected.
