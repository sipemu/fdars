---
phase: 48-parallelism-gap-closure
plan: 02
subsystem: coclustering
tags: [perf, parallelism, rayon, co_cluster, cem, golden-equivalence]
requires:
  - phase: 48
    plan: 01
    provides: perf_parallelism bench + equivalence_phase48 harness + PERF-PARALLEL-RESULTS ledger
provides:
  - co_cluster n_init multi-restart loop parallelized (iter_maybe_parallel! map → Result → sequential strict-> reduce)
  - CO_CLUSTER_INIT_PARALLEL_THRESHOLD=3 payback guard
  - bench_co_cluster thread-scaling cell (n200_m50_ninit8)
affects: [48-03, 51-benchmark-coverage-regression-guards]
actuals:
  tokens: 22000
  tasks: 3
  commits: 3
tech-stack:
  added: []
  patterns: [parallel map→collect::<Result>→sequential strict-> reduce (order-independent tie-break)]
key-files:
  created: []
  modified:
    - fdars-core/src/coclustering.rs
    - fdars-core/tests/equivalence_phase48.rs
    - fdars-core/benches/perf_parallelism.rs
    - .planning/phases/48-parallelism-gap-closure/PERF-PARALLEL-RESULTS.md
key-decisions:
  - "n_init loop → iter_maybe_parallel!(0..n_init).map(run_init).collect::<Result<Vec,_>>()? then a SEQUENTIAL reduce with strict `>` — lowest-init-index wins ties exactly as the old loop."
  - "collect::<Result<Vec,_>>() short-circuits on the first Err in index order = the sequential `?` behavior (error propagation preserved)."
  - "run_init closure captures only `&` to owned data → Sync; needed #[cfg(feature=parallel)] use rayon::iter::ParallelIterator (same as anova.rs)."
  - "CoClusterConfig is #[non_exhaustive] — goldens/bench build it via Default + field assignment."
patterns-established:
  - "For fallible parallel best-of loops: parallel map returning Result, collect Result (first-Err short-circuit), then sequential strict-> reduce for a deterministic, order-independent winner."
requirements-completed: [PERF-03 (co_cluster path)]
coverage:
  - id: D1
    description: "co_cluster n_init loop parallelized bit-identically (labels + log_likelihood; per-init seeding + strict-> tie-break preserved; signature unchanged)"
    requirement: PERF-03
    verification:
      - kind: integration
        ref: "golden_co_cluster_parallel (n_init=4) + _below_threshold (n_init=2) assert_eq! on LL + row_labels + col_labels, pass under both feature configs; 15 coclustering lib tests green both configs"
        status: pass
    human_judgment: false
  - id: D2
    description: "Wave-2 gate green + bench cell registered"
    requirement: PERF-03
    verification:
      - kind: integration
        ref: "full suite 2583 lib tests both configs; clippy --all-targets clean; perf_parallelism_co_cluster/n200_m50_ninit8 registered"
        status: pass
    human_judgment: false
---

# Plan 48-02 SUMMARY — co_cluster n_init Parallelization

## What shipped

- **`src/coclustering.rs`** — the multi-restart CEM `for init in 0..n_init` best-of loop is now a
  `run_init(init) -> Result<CoClusterResult, FdarError>` closure driven by
  `iter_maybe_parallel!(0..n_init).map(run_init).collect::<Result<Vec<_>, _>>()?` behind a
  `CO_CLUSTER_INIT_PARALLEL_THRESHOLD = 3` payback outer-if, followed by a **sequential** strict-`>`
  `reduce` that keeps the earliest (lowest-index) element on ties — byte-for-byte the old tie-break.
  Per-init seeding (`config.seed.wrapping_add(init*1000)`) untouched; public signature unchanged.
- **`tests/equivalence_phase48.rs`** — `golden_co_cluster_parallel` (n_init=4, parallel branch) and
  `golden_co_cluster_below_threshold` (n_init=2, sequential branch) assert bit-identical
  `log_likelihood` + `row_labels` + `col_labels`, captured from the pre-parallel code, passing under
  both feature configs.
- **`benches/perf_parallelism.rs`** — `bench_co_cluster` cell (n=200, m=50, n_init=8) registered.
- **`PERF-PARALLEL-RESULTS.md`** — threshold recorded.

## Evidence

| Check | Result |
|-------|--------|
| co_cluster goldens (both configs) | ✅ parallel + below-threshold, `assert_eq!` LL + labels |
| coclustering lib tests (both configs) | ✅ 15/15 |
| Full suite both configs | ✅ 2583 lib tests each, all integration green |
| clippy --all-targets | ✅ clean |
| Signature / dependency | ✅ unchanged / none added |

## Notes

- **Tie-break correctness (T-48-03):** `reduce(|acc, r| if r.log_likelihood > acc.log_likelihood { r } else { acc })`
  over the index-ordered `Vec` keeps `acc` unless strictly greater — identical to the old
  "only replace when strictly greater" loop. Golden asserts exact labels, so any tie-break drift
  would fail the test.
- **Error propagation (T-48-04):** only `kmeans_fd` is fallible; `collect::<Result<Vec,_>>()` returns
  the first `Err` in iteration order — the same error the sequential `?` short-circuited on.
- 1-vs-N thread-scaling medians for co_cluster are captured in Wave 3 (plan 48-03).
