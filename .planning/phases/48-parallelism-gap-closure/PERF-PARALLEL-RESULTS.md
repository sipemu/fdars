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
| _TBD (plan 48-02)_ | — | — | — | — |

## Threshold constants

| Constant | Location | Value | Rationale |
|----------|----------|-------|-----------|
| `FRECHET_ANOVA_PERM_PARALLEL_THRESHOLD` | `src/frechet/anova.rs` | 200 | n_perm is the work driver; below ~200 perms rayon dispatch overhead can exceed the per-perm `compute_tn_generic` cost. Conservative; Wave 3 criterion measurement confirms/adjusts. |
| `CO_CLUSTER_INIT_PARALLEL_THRESHOLD` | `src/coclustering.rs` | _TBD (plan 48-02)_ | _TBD_ |

## Deferrals

- **`frechet_anova_space<S: MetricSpace>`** (generic path, anova.rs) NOT parallelized — would require
  an `S: Sync` bound, a public-generic signature widening that could break external `MetricSpace`
  impls. DEFER-with-rationale. The concrete `frechet_anova` (parallelized here) already covers the
  PROF-01 hotspot. (Finalized/signed off in plan 48-03.)
