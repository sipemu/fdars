# PROF-01 — Ranked Hot-Path Optimization Target List

**Phase:** 46 (Whole-Crate Profiling & Measurement) · **Requirement:** PROF-01 · **Consumers:** Phase 47 (PERF), Phase 51 (BENCH-01 module list)
**Method:** throwaway criterion probe benches (copy of `benches/audit_hotpaths.rs`) + feature-gated `dhat-heap` allocation probes over the 9 reuse-first v0.19–v0.29 subsystems. Measure-only — all probes removed in Plan 02 Task 3; zero `src/` edits.
**Ranking metric (locked):** wall-time × representativeness, allocation count secondary.

---

## Environment

| Property | Value |
|----------|-------|
| CPU governor | `powersave` — **LOW-CONFIDENCE honesty caveat** (unpinned; multi-thread cells not directly comparable across runs — v0.14.0 audit note) |
| Logical cores | 20 (RAYON_NUM_THREADS default = 20 unless overridden) |
| Feature flags | `linalg,parallel` (criterion), `dhat-heap,linalg` (allocation probes) |
| Harness | criterion 0.5 (`harness = false`), dhat 0.3 |
| Date | 2026-08-30 |
| Host tmp | `TMPDIR=/home/simonm/.cache/fdars-bench-tmp` |

Downstream before/after comparisons (Phase 47) MUST re-capture this block and should pin the
governor to `performance` for honest deltas; `powersave` inflates and de-stabilizes timings.

---

## Measured Cells

Wall-time is criterion's median. Allocations from `dhat-heap` probes at the n200_m50 cell (serialized,
`--test-threads=1` — dhat allows one live profiler per process).

| Module | Function | N | M | median | allocs (blocks / total B / peak B) | anchor |
|--------|----------|---|---|--------|-----------------------------------|--------|
| fpca_variants | `fsvd` | 50 | 50 | 601 µs | — | `src/fpca_variants.rs:488` |
| fpca_variants | `fsvd` | 200 | 50 | 999 µs | 275 / 600 049 / 410 880 | `src/fpca_variants.rs:488` |
| fpca_variants | `fsvd` | 50 | 200 | 12.09 ms | — | `src/fpca_variants.rs:488` |
| fpca_variants | `fsvd` | 1000 | 50 | 2.88 ms | — | `src/fpca_variants.rs:488` |
| fpca_variants | `ssvd` | 200 | 50 | — | 22 / 314 416 / 182 384 | `src/fpca_variants.rs:740` |
| inference | `t_perm_test` | 50 | 50 | 439 µs | — | `src/inference/permutation.rs:152` |
| inference | `t_perm_test` | 200 | 50 | 1.74 ms | — | `src/inference/permutation.rs:152` |
| inference | `t_perm_test` | 50 | 200 | 992 µs | — | `src/inference/permutation.rs:152` |
| fts | `long_run_covariance` | 50 | 50 | 320 µs | — | `src/fts/acf.rs:337` |
| fts | `long_run_covariance` | 200 | 50 | 2.13 ms | 6 / 100 400 / 40 400 | `src/fts/acf.rs:337` |
| fts | `long_run_covariance` | 50 | 200 | 4.24 ms | — | `src/fts/acf.rs:337` |
| fts | `long_run_covariance` | 1000 | 50 | 21.31 ms | — | `src/fts/acf.rs:337` |
| fts | `dpca` | 200 | 50 | — | **17 739 / 42 084 568 / 8 637 712** | `src/fts/spectral.rs:203` |
| frechet | `frechet_anova` | 50 | 50 | 32.14 ms | — | `src/frechet/anova.rs:122` |
| frechet | `frechet_anova` | 200 | 50 | 128.97 ms | — | `src/frechet/anova.rs:122` |
| frechet | `frechet_anova` | 50 | 200 | 133.02 ms | — | `src/frechet/anova.rs:122` |
| density_fda | `wasserstein_barycenter` | 50 | 101 | 71.8 µs | — | `src/density_fda.rs:407` |
| density_fda | `wasserstein_barycenter` | 200 | 101 | 309 µs | — | `src/density_fda.rs:407` |
| density_fda | `wasserstein_barycenter` | 50 | 201 | 195 µs | — | `src/density_fda.rs:407` |
| irreg_fdata::face | `face_covariance` | 50 | 30 | **242.5 ms** | — | `src/irreg_fdata/face.rs:128` |
| irreg_fdata::face | `face_covariance` | 200 | 30 | **983.8 ms** | — | `src/irreg_fdata/face.rs:128` |
| boosting_regression | `boost_fosr` | 50 | 50 | 2.69 ms | — | `src/boosting_regression/boost_fosr.rs:263` |
| boosting_regression | `boost_fosr` | 100 | 50 | 4.61 ms | — | `src/boosting_regression/boost_fosr.rs:263` |
| boosting_regression | `boost_fosr` | 50 | 100 | 5.27 ms | — | `src/boosting_regression/boost_fosr.rs:263` |
| fem_smoothing | `fem_smooth` | 64 nodes | 98 tris | 595 µs | — | `src/fem_smoothing.rs:475` |
| fem_smoothing | `fem_smooth` | 256 nodes | 450 tris | 39.96 ms | — | `src/fem_smoothing.rs:475` |
| fem_smoothing | `fem_smooth` | 576 nodes | 1058 tris | **452.3 ms** | — | `src/fem_smoothing.rs:475` |
| coclustering | `co_cluster` | 50 | 50 | 4.66 ms | — | `src/coclustering.rs:874` |
| coclustering | `co_cluster` | 100 | 50 | 13.33 ms | — | `src/coclustering.rs:874` |
| coclustering | `co_cluster` | 50 | 100 | 12.15 ms | — | `src/coclustering.rs:874` |

_(fsvd @ n200_m50 is the Plan-01 tracer cell; ssvd/long_run_covariance/dpca dhat cells added in Plan 02.)_

---

## Ranked Targets

Top-10, ranked by wall-time × representativeness (allocation count secondary). Peak observed at a
tractable cell; anchors are the primary optimization site.

| Rank | Module::Function | Cost signal | Anchor | Optimization note (Phase 47) |
|------|------------------|-------------|--------|------------------------------|
| 1 | `irreg_fdata::face_covariance` | **984 ms** @ n200_m30 (3.8 s @ n50_m60) | `src/irreg_fdata/face.rs:128` | By far the slowest path. ~O(n·m²) kernel covariance smoothing over irregular pairs — the dominant PERF-01 target. |
| 2 | `fem_smoothing::fem_smooth` | **452 ms** @ 576 nodes | `src/fem_smoothing.rs:475` | Strongly superlinear in node count (595 µs → 40 ms → 452 ms for 64 → 256 → 576 nodes) — FEM stiffness assembly + linear solve. |
| 3 | `fts::dpca` | **42 MB / 8.6 MB peak alloc** (17 739 blocks) @ n200_m50 | `src/fts/spectral.rs:203` | **Top allocation hotspot** (PERF-02) — spectral-density estimation churns huge temporary DMatrices. |
| 4 | `frechet::frechet_anova` | **133 ms** @ n50_m200 | `src/frechet/anova.rs:122` | Permutation test × Wasserstein-Fréchet variance; N and M both raise cost sharply. Sequential permutation loop → also a PERF-03 parallelism candidate. |
| 5 | `fts::long_run_covariance` | 21.3 ms @ n1000_m50 (N-dominated) | `src/fts/acf.rs:337` | Lag-covariance accumulation; N-dominated. Lean on allocs (6 blocks) — compute-bound. |
| 6 | `coclustering::co_cluster` | 13.3 ms @ n100_m50 | `src/coclustering.rs:874` | Iterative CEM (n_init × max_iter); N-dominated. Parallelism candidate over inits (PERF-03). |
| 7 | `fpca_variants::fsvd` | 12.1 ms @ n50_m200 (O(M³)) + 600 KB alloc | `src/fpca_variants.rs:488` | M-dominated gram eigendecomposition `DMatrix::from_column_slice(g_dim,g_dim,&gram).symmetric_eigen()`. |
| 8 | `boosting_regression::boost_fosr` | 5.27 ms @ n50_m100 | `src/boosting_regression/boost_fosr.rs:263` | Iterative boosting (mstop=30 fixed); modest growth in N and M. |
| 9 | `inference::t_perm_test` | 1.74 ms @ n200_m50 | `src/inference/permutation.rs:152` | Sequential permutation loop → PERF-03 parallelism candidate (payback-threshold guard needed). |
| 10 | `density_fda::wasserstein_barycenter` | 309 µs @ n200_m101 (cheapest) | `src/density_fda.rs:407` | Lowest priority — already fast. |

### Allocation ranking (PERF-02 targets)

| Rank | Function | total bytes | peak bytes | blocks | anchor |
|------|----------|-------------|-----------|--------|--------|
| 1 | `fts::dpca` | 42 084 568 | 8 637 712 | 17 739 | `src/fts/spectral.rs:203` |
| 2 | `fpca_variants::fsvd` | 600 049 | 410 880 | 275 | `src/fpca_variants.rs:488` |
| 3 | `fpca_variants::ssvd` | 314 416 | 182 384 | 22 | `src/fpca_variants.rs:740` |
| 4 | `fts::long_run_covariance` | 100 400 | 40 400 | 6 | `src/fts/acf.rs:337` |

`fts::dpca` dwarfs the rest by ~70× total bytes and ~2500× block count — the single clearest
allocation-reduction target for PERF-02.

---

## N×M Scaling

| Subsystem::fn | Observed scaling | Evidence |
|---------------|------------------|----------|
| `face_covariance` | **strongly (n · m²)** | 242 ms → 984 ms as n 50→200 (×4 for ×4 n); m 30→60 pushed n50 cell to 3.8 s (~16× for ×2 m → ~m²) |
| `fem_smooth` | **superlinear in node count (~O(nodes^2.5))** | 595 µs → 40 ms → 452 ms for 64 → 256 → 576 nodes |
| `frechet_anova` | **both N and M** (permutation × pairwise variance) | n200_m50 (129 ms) ≈ n50_m200 (133 ms) ≫ n50_m50 (32 ms) |
| `long_run_covariance` | **N-dominated** | n1000_m50 (21.3 ms) ≫ n50_m200 (4.24 ms) |
| `co_cluster` | **N-dominated** (iterative CEM) | n100_m50 (13.3 ms) > n50_m100 (12.2 ms) > n50_m50 (4.66 ms) |
| `fsvd` | **M-dominated O(M³)** | n50_m200 (12.1 ms) ≫ n1000_m50 (2.88 ms); gram is m×m |
| `t_perm_test` | **N-dominated** | n200_m50 (1.74 ms) > n50_m200 (992 µs) |
| `boost_fosr` | mild in N and M (fixed mstop) | 2.69 → 4.61 (n) / 5.27 (m) ms |
| `wasserstein_barycenter` | N-dominated, cheap | n200_m101 (309 µs) > n50_m201 (195 µs) |

---

## Scope Confirmation

Measure-only: all probe benches + the `alloc_audit_new_subsystems.rs` dhat file are removed in Plan 02
Task 3; no permanent `[[bench]]` registered (Phase 51 / BENCH-01 owns those); no new crate dependency
(criterion + dhat already dev-deps); zero `fdars-core/src/` edits. Governor `powersave` caveat applies.

## Phase 47 / 51 Hand-off

**Phase 47 (PERF) top targets:** compute-bound → `face_covariance` (#1), `fem_smooth` (#2),
`frechet_anova` (#4); allocation → `fts::dpca` (42 MB churn, PERF-02 #1). Parallelism candidates
(PERF-03): sequential permutation loops in `frechet_anova` / `t_perm_test`, and `co_cluster` inits.
**Phase 51 (BENCH-01) module list:** all 9 subsystems profiled here need permanent `[[bench]]` coverage.
