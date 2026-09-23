# BEFORE vs AFTER Criterion: perf_face_covariance/n200_m30

## Measurement Context

- **Benchmark:** `perf_hotpaths::perf_face_covariance/n200_m30` (n=200 curves, m=30 grid points)
- **Change:** Added `sym_cov_irreg` (upper-triangle + bit-copy mirror), switched call site in `face_covariance`
- **BEFORE SHA:** 328215e7 (Plan 01 baseline — un-optimized `cov_irreg(ifd, grid, grid, bw)`)
- **AFTER SHA:** c0e84bd8 (Plan 02 — `sym_cov_irreg(ifd, grid, bw)`)

## Results

| Metric      | BEFORE (baseline)           | AFTER (optimized)            |
|-------------|-----------------------------|------------------------------|
| Mean        | 268.166 ms                  | 77.786 ms                    |
| CI lower    | 260.1 ms                    | 76.710 ms                    |
| CI upper    | 275.9 ms                    | 79.372 ms                    |
| CI level    | 95%                         | 95%                          |

## Speedup

- **Percent reduction:** ~71.0% (AFTER mean / BEFORE mean = 77.79 / 268.17 = 0.290 → 71.0% faster)
- **Speedup factor:** ~3.45x (268.17 ms / 77.79 ms)
- Criterion change report: `[-71.921% -70.993% -69.944%]` (p = 0.00)

## Overlap Verdict

**NON-OVERLAPPING — statistically significant speedup.**

AFTER CI upper (79.372 ms) < BEFORE CI lower (260.1 ms) by a margin of ~181 ms.
The improvement is unambiguously measurable at this cell, not within noise.

## Why Faster Than Expected

The RESEARCH predicted ~2x speedup (halving the si×ti outer loop). The observed ~3.45x
reflects an additional benefit: with a SINGLE kernel table (`w` instead of `w_s` + `w_t`),
the total table-build work halves too (from 2 × total_obs × m kernel_gaussian() calls to
1 × total_obs × m), and memory pressure drops. Combined with the triangle shortcut, the
total exp() evaluation and inner-loop work is roughly halved on both axes.

## Numeric Identity

Golden fixture `equivalence_phase115` passes UNCHANGED at ≤1e-10 (debug + release).
Upper-triangle values are bit-identical to `cov_irreg`; the mirrored half is a bit-copy.
