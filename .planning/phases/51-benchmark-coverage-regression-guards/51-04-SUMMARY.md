---
phase: 51-benchmark-coverage-regression-guards
plan: 04
subsystem: benchmarks
tags: [bench, bench-02, regression-guard, documentation, ledger]
requires:
  - phase: 51
    plan: 03
    provides: all 9 BENCH-01 module benches + captured --quick baselines
  - phase: 47
    provides: PERF-RESULTS.md (dpca -54%, face_covariance -80.7%, …)
  - phase: 48
    provides: PERF-PARALLEL-RESULTS.md (frechet_anova 9.9×, co_cluster 6.4×)
provides:
  - BENCH-RESULTS.md — consolidated soft/hard regression-guard ledger (BENCH-02)
affects: []
actuals:
  tokens: 6000
  tasks: 1
  commits: 1
tech-stack:
  added: []
  patterns: [documented-soft-guard ledger + deterministic-hard-guard inventory]
key-files:
  created:
    - .planning/phases/51-benchmark-coverage-regression-guards/BENCH-RESULTS.md
  modified: []
key-decisions:
  - "Soft guards (criterion baseline-compare, no CI fail) for wall-time; deterministic hard guards (alloc_audit_dpca/fpca) unchanged; NO wall-time assert! (governor-sensitive)."
  - "Doc-only: no Cargo.toml or src/ edit; the 2 PERMANENT perf benches + 2 alloc guards confirmed intact; no v* tag (crate stays 0.29.0)."
requirements-completed: [BENCH-02]
coverage:
  - id: D1
    description: "BENCH-RESULTS.md consolidates Phase 47/48 before/after as documented soft guards + 9 new-module first baselines + governor caveat + guard inventory"
    requirement: BENCH-02
    verification:
      - kind: doc
        ref: "BENCH-RESULTS.md: 6 sections; grep gates (powersave, alloc_audit, perf_hotpaths) pass; alloc_audit_dpca/fpca on disk; perf_hotpaths+perf_parallelism registered; cargo build --benches green"
        status: pass
    human_judgment: false
---

# Plan 51-04 SUMMARY — BENCH-RESULTS.md (BENCH-02)

## What shipped

`.planning/phases/51-benchmark-coverage-regression-guards/BENCH-RESULTS.md` — the consolidated
regression-guard ledger, with all 6 sections:
1. **Guard model** — documented SOFT guards (criterion baseline-compare, no CI fail) vs deterministic
   HARD guards (`alloc_audit_dpca/fpca`, dhat-heap); explicit note that NO wall-time `assert!` was added.
2. **Phase 47 wins** table (dpca −54% blocks, face_covariance −80.7%, fsvd/ssvd/functional_acf/fem_smooth).
3. **Phase 48 wins** table (frechet_anova 9.9×, co_cluster 6.4×) + the RAYON_NUM_THREADS sweep command.
4. **New-module baselines** — all 9 BENCH-01 cells with captured `--quick` medians (inference 2.196ms,
   fts 12.7ms, frechet 264µs, boost_fosr 28.2/14.5ms, co_cluster_select 232ms, fem_smooth_gcv 305.8ms,
   lqd_fpca 3.04ms + wasserstein 389µs, fpca_der 9.76ms + fsvd 1.28ms, mface_covariance 1.232s).
5. **Governor caveat** — `powersave` unpinned → wall-time LOW-CONFIDENCE; alloc wins HIGH-confidence.
6. **Guard inventory** — perf_hotpaths + perf_parallelism registered PERMANENT; both alloc_audit guards
   intact; 9 new benches added.

## Evidence

| Check | Result |
|-------|--------|
| grep gates (powersave / alloc_audit / perf_hotpaths) | ✅ pass |
| alloc_audit_dpca.rs + alloc_audit_fpca.rs on disk | ✅ present |
| perf_hotpaths + perf_parallelism registered | ✅ (Cargo.toml, unchanged) |
| cargo build --benches | ✅ whole suite (9 new + 2 permanent + prior) compiles |
| Cargo/src edit | ✅ none (doc-only); no v* tag |

BENCH-01 (9 module benches) + BENCH-02 (this ledger) both complete — Phase 51 done.
