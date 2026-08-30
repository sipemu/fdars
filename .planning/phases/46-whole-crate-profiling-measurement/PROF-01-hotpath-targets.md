# PROF-01 — Ranked Hot-Path Optimization Target List

**Phase:** 46 (Whole-Crate Profiling & Measurement) · **Requirement:** PROF-01 · **Consumers:** Phase 47 (PERF), Phase 51 (BENCH-01 module list)
**Method:** throwaway criterion probe benches (copy of `benches/audit_hotpaths.rs`) + feature-gated `dhat-heap` allocation probes over the 9 reuse-first v0.19–v0.29 subsystems. Measure-only — probes removed in Plan 02; zero `src/` edits.
**Ranking metric (locked):** wall-time × representativeness, allocation count secondary.

> **Status:** SKELETON (Plan 01 tracer). Contains the Environment section and the first grounded
> `fpca_variants::fsvd` row. Plan 02 expands the Measured-Cells table to all 9 subsystems and
> completes the Ranked Targets + N×M scaling sections.

---

## Environment

| Property | Value |
|----------|-------|
| CPU governor | `powersave` — **LOW-CONFIDENCE honesty caveat** (unpinned; multi-thread cells not directly comparable across runs — v0.14.0 audit note) |
| Logical cores | 20 (RAYON_NUM_THREADS default = 20 unless overridden) |
| Feature flags | `linalg,parallel` (criterion), `dhat-heap,linalg` (allocation probes) |
| Harness | criterion 0.5 (`harness = false`), dhat 0.3 |
| Date | 2026-08-30 |
| Host tmp | `TMPDIR=/home/simonm/.cache/fdars-bench-tmp` (small /tmp tmpfs avoidance) |

Downstream before/after comparisons (Phase 47) must re-capture this block and, ideally, pin the
governor to `performance` for honest deltas.

---

## Measured Cells

Wall-time is criterion's median [low mid high]. Allocations from `dhat-heap` probes where measured.

| Module | Function | N | M | mean (median) | 95% CI | allocs (blocks / total B / peak B) | anchor |
|--------|----------|---|---|---------------|--------|-----------------------------------|--------|
| fpca_variants | `fsvd` | 50 | 50 | 601 µs | [573 µs, 637 µs] | — | `src/fpca_variants.rs:405` (gram @:488) |
| fpca_variants | `fsvd` | 200 | 50 | 999 µs | [950 µs, 1.05 ms] | 275 / 600 049 / 410 880 | `src/fpca_variants.rs:488` |
| fpca_variants | `fsvd` | 50 | 200 | 12.09 ms | [11.89 ms, 12.37 ms] | — | `src/fpca_variants.rs:488` |
| fpca_variants | `fsvd` | 1000 | 50 | 2.88 ms | [2.77 ms, 2.98 ms] | — | `src/fpca_variants.rs:405` |

_(Plan 02 appends rows for inference, fts, frechet, density_fda, face, boosting_regression, fem_smoothing, coclustering.)_

---

## Ranked Targets

Top target list (ranked by wall-time × representativeness). **Skeleton — completed in Plan 02 to a top-10.**

| Rank | Module::Function | Cost signal | Anchor | Notes |
|------|------------------|-------------|--------|-------|
| 1 (provisional) | `fpca_variants::fsvd` | 12.1 ms @ n50_m200; strongly M-dominated | `src/fpca_variants.rs:488` | gram eigendecomposition `DMatrix::from_column_slice(g_dim,g_dim,&gram).symmetric_eigen()` — O(M³) in the number of eval points; 600 KB / 411 KB peak alloc at n200_m50 |

_(Plan 02 ranks the top-10 across all 9 subsystems.)_

---

## N×M Scaling

**fpca_variants::fsvd** — strongly **M-dominated**: holding N and scaling M 50→200 (×4) raised time
601 µs → 12.09 ms (~20×), while scaling N 50→1000 (×20) at M=50 raised time 601 µs → 2.88 ms (~4.8×).
This matches the m×m gram-matrix `symmetric_eigen()` at `src/fpca_variants.rs:488` being **O(M³)**;
N enters only through the O(N·M²) cross-covariance accumulation. → Phase 47 should target the eval-point
dimension (M) path first for this subsystem.

_(Per-subsystem scaling analysis completed in Plan 02.)_
