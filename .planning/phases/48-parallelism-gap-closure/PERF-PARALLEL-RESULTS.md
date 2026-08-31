# Phase 48 — Parallelism-Gap Closure: Thread-Scaling Results (PERF-03)

Permanent ledger of the before/after thread-scaling evidence for the paths parallelized this
phase. Cells come from `fdars-core/benches/perf_parallelism.rs` (PERMANENT — Phase 51 BENCH-02
regression guard). Equivalence (bit-identical output) is proven separately by
`fdars-core/tests/equivalence_phase48.rs`.

## Environment

| Field | Value |
|-------|-------|
| Date | 2026-08-31 |
| Host logical cores | 20 |
| CPU governor | **powersave** (sudo/cpupower unavailable — could not pin to `performance`) |
| Confidence | **LOW-CONFIDENCE** — governor unpinned (`powersave`); absolute medians and ratios are indicative, not authoritative. Thread-scaling *direction* (N-thread < 1-thread) remains a valid signal; treat exact ratios as lower bounds. |
| RAYON_NUM_THREADS default | unset (rayon uses all 20 logical cores) |
| Feature flags | `--features linalg,parallel` (parallel ON); equivalence also verified under `--no-default-features --features linalg` (parallel OFF) |
| Sweep method | env-var: `RAYON_NUM_THREADS=1` vs `RAYON_NUM_THREADS=20` |
| Host tmp | `TMPDIR=/home/simonm/.cache/fdars-bench-tmp` (/tmp tmpfs exhaustion guard) |

> Governor caveat (carried from v0.14.0 audit): multi-thread criterion cells are LOW-CONFIDENCE
> with the governor unpinned. Numbers below record the environment honestly so future
> `performance`-governor re-runs can supersede them.

## frechet_anova (src/frechet/anova.rs — PROF-01 #4, 133 ms)

Cell: `perf_parallelism_frechet_anova/n24_m81_nperm999` (n=24 curves, m=81 argvals, n_perm=999, seed=42).

| Cell | 1-thread median | 20-thread median | Ratio (20t / 1t) | Speedup (1t / 20t) |
|------|-----------------|------------------|------------------|--------------------|
| n24_m81_nperm999 | 322.73 ms | 32.57 ms | 0.101 | **9.9×** |

Captured 2026-08-31 (governor `powersave`, LOW-CONFIDENCE on absolute numbers; the ~10× direction
is unambiguous). Criterion reported `change: -89.8%` (20t vs 1t), p < 0.05. Comfortably beats the
0.6× target — no Wave 3 threshold review needed for frechet_anova.

## co_cluster (src/coclustering.rs — Wave 2 / plan 48-02)

Cell: `perf_parallelism_co_cluster/*` (added by plan 48-02).

| Cell | 1-thread median | 20-thread median | Ratio (20t / 1t) | Speedup (1t / 20t) |
|------|-----------------|------------------|------------------|--------------------|
| n200_m50_ninit8 | 337.34 ms | 52.91 ms | 0.157 | **6.4×** |

Captured 2026-08-31 (governor `powersave`, LOW-CONFIDENCE on absolute numbers; direction
unambiguous). Criterion reported `change: -84.0%` (20t vs 1t), p < 0.05. n_init=8 caps parallel
fan-out at 8-way, so 6.4× is near the achievable ceiling for this cell — well above the payback
break-even. No threshold review needed for co_cluster.

## Threshold constants

| Constant | Location | Value | Rationale |
|----------|----------|-------|-----------|
| `FRECHET_ANOVA_PERM_PARALLEL_THRESHOLD` | `src/frechet/anova.rs` | 200 | n_perm is the work driver; below ~200 perms rayon dispatch overhead can exceed the per-perm `compute_tn_generic` cost. **Validated by thread-scaling** (9.9× at n_perm=999); kept conservative — the large cell is far above break-even, so the exact crossover is not safety-critical. |
| `CO_CLUSTER_INIT_PARALLEL_THRESHOLD` | `src/coclustering.rs` | 3 | n_init is the work driver; below 3 restarts rayon dispatch overhead can exceed the per-init CEM cost. **Validated by thread-scaling** (6.4× at n_init=8). Mirrors the v0.17.0 `SCORES_PARALLEL_THRESHOLD` precedent; kept at 3 (default `n_init`=3 hits the parallel path only when a caller opts into ≥3 restarts). |

Both thresholds hold against the measured break-even — the parallelized cells are ~6–10× faster
well above threshold, and the sequential branch below threshold is bit-identical (proven by the
`*_below_threshold` goldens). No numeric computation was changed; only the guard consts exist.

## Deferred (documented)

These PROF-01/RESEARCH parallelization candidates were DELIBERATELY NOT parallelized this phase —
each with a rationale below, not a silent omission. They remain open for a future milestone.

| Target | Location | Why deferred |
|--------|----------|--------------|
| `t_perm_test` / `f_perm_test` | `src/inference/permutation.rs` | Use a SINGLE shared advancing `StdRng` across permutations. Parallelizing requires per-perm reseeding, which **changes the returned p-values** — a numeric-output change. Out of scope for this behavior-preserving milestone; revisit in a milestone that accepts a re-baseline. |
| `frechet_anova_space<S: MetricSpace>` | `src/frechet/anova.rs` | Parallelizing needs an added `S: Sync` bound — a public-generic signature widening that could break external non-`Sync` `MetricSpace` impls. The concrete `frechet_anova` (parallelized here) already captures the PROF-01 hotspot. Revisit if `MetricSpace` is confirmed sealed/internal. |
| `explain/importance.rs` (:131, :221) | `src/explain/importance.rs` | Low outer-loop fan-out (ncomp 3–10) + shared inner RNG. Phase 49 CONS-02 folds this into the already-parallel `explain_generic` path, gaining parallelism for free — parallelizing it standalone here would be throwaway work. |

## Regression-guard note

The permanent `perf_parallelism` cells (`frechet_anova`, `co_cluster`) feed **Phase 51 BENCH-02** as
regression guards for the wins recorded above. Re-run under a `performance` governor to supersede
the LOW-CONFIDENCE absolute medians when a pinned environment is available.
