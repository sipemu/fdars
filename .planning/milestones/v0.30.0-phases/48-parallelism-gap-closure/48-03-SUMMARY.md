---
phase: 48-parallelism-gap-closure
plan: 03
subsystem: validation
tags: [perf, parallelism, thread-scaling, validation, sign-off]
requires:
  - phase: 48
    plan: 01
    provides: frechet_anova parallelization + bench cell
  - phase: 48
    plan: 02
    provides: co_cluster parallelization + bench cell
provides:
  - Thread-scaling medians (frechet_anova 9.9×, co_cluster 6.4×) recorded in PERF-PARALLEL-RESULTS.md
  - Finalized payback thresholds + documented deferrals (t_perm/f_perm, frechet_anova_space, explain)
  - 48-VALIDATION.md signed off (nyquist_compliant: true)
affects: [51-benchmark-coverage-regression-guards]
actuals:
  tokens: 14000
  tasks: 3
  commits: 1
tech-stack:
  added: []
  patterns: [RAYON_NUM_THREADS 1-vs-N env sweep for thread-scaling proof]
key-files:
  created: []
  modified:
    - .planning/phases/48-parallelism-gap-closure/PERF-PARALLEL-RESULTS.md
    - .planning/phases/48-parallelism-gap-closure/48-VALIDATION.md
key-decisions:
  - "Both thresholds KEPT (frechet 200, co_cluster 3) — validated by thread-scaling; large cells are far above break-even so exact crossover is not safety-critical. No src edit needed in Wave 3."
  - "Three deferred targets documented (not silent): t_perm/f_perm (RNG reseed changes p-values), frechet_anova_space (S:Sync widening), explain/importance (Phase 49 folds into parallel generic path)."
patterns-established: []
requirements-completed: [PERF-03]
coverage:
  - id: D1
    description: "Thread-scaling measured for both parallelized paths; parallel faster on large inputs"
    requirement: PERF-03
    verification:
      - kind: benchmark
        ref: "frechet_anova 322.73→32.57ms (9.9×); co_cluster 337.34→52.91ms (6.4×); both criterion change <-83%, p<0.05"
        status: pass
    human_judgment: false
  - id: D2
    description: "Deferred targets documented with rationale (not silent omissions)"
    requirement: PERF-03
    verification:
      - kind: doc
        ref: "PERF-PARALLEL-RESULTS.md 'Deferred (documented)' table: t_perm_test/f_perm_test, frechet_anova_space, explain/importance"
        status: pass
    human_judgment: false
  - id: D3
    description: "Final phase gate green + VALIDATION signed off"
    requirement: PERF-03
    verification:
      - kind: integration
        ref: "5/5 equivalence goldens both configs; full suite 2583 lib tests both configs; clippy --all-targets clean; frechet_anova + co_cluster signatures byte-identical; 48-VALIDATION.md nyquist_compliant: true"
        status: pass
    human_judgment: false
---

# Plan 48-03 SUMMARY — Thread-Scaling, Deferrals, Sign-Off

## What shipped (docs-only — no numeric/src changes)

- **Thread-scaling measured** (`RAYON_NUM_THREADS=1` vs `=20`, governor `powersave`):

  | Cell | 1-thread | 20-thread | Speedup |
  |------|----------|-----------|---------|
  | frechet_anova n24_m81_nperm999 | 322.73 ms | 32.57 ms | **9.9×** |
  | co_cluster n200_m50_ninit8 | 337.34 ms | 52.91 ms | **6.4×** |

  Both recorded in `PERF-PARALLEL-RESULTS.md` with the LOW-CONFIDENCE governor caveat (absolute
  numbers indicative; the ~6–10× direction is unambiguous).
- **Thresholds finalized** — both kept (`FRECHET_ANOVA_PERM_PARALLEL_THRESHOLD=200`,
  `CO_CLUSTER_INIT_PARALLEL_THRESHOLD=3`), annotated "validated by thread-scaling". No src edit was
  required in Wave 3 — the measured break-even confirms the research estimates.
- **Deferred (documented)** — `t_perm_test`/`f_perm_test` (shared advancing RNG → parallelizing
  changes p-values), `frechet_anova_space` (`S: Sync` public-generic widening), `explain/importance`
  (low fan-out; Phase 49 CONS-02 folds it into the already-parallel generic path). Each with a
  one-line rationale in the ledger.
- **`48-VALIDATION.md` signed off** — `status: validated`, `nyquist_compliant: true`,
  `wave_0_complete: true`, every checklist item ✅.

## Final gate

| Check | Result |
|-------|--------|
| Phase-48 goldens (both configs) | ✅ 5/5 each (frechet + co_cluster, parallel + below-threshold) |
| Full suite both configs | ✅ 2583 lib tests each |
| clippy --all-targets | ✅ clean |
| Public signatures | ✅ `frechet_anova` + `co_cluster` byte-identical |
| New dependency | ✅ none |

## Phase 48 outcome

PERF-03 met: the two bit-identical-safe parallelizations (`frechet_anova`, `co_cluster`) ship with
payback guards and 6–10× thread-scaling; marginal/unsafe targets are documented+deferred. Permanent
`perf_parallelism` bench cells feed Phase 51 BENCH-02.
