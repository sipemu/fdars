# PERF-RESULTS — Phase 47 Hot-Path & Allocation Performance

Before/after ledger for the PERF-01/PERF-02 optimizations. Feeds Phase 51 (BENCH-02) regression
guards. All numbers from the `perf_hotpaths` criterion bench + `alloc_audit_dpca`/`alloc_audit_fpca`
dhat probes. Before-numbers are the PROF-01 (Phase 46) measurements.

## Environment

| Property | Value |
|----------|-------|
| CPU governor | `powersave` — **LOW-CONFIDENCE** (unpinned; `cpupower` pin needs sudo — see per-OPT notes) |
| Logical cores | 20 (RAYON_NUM_THREADS default = 20) |
| Feature flags | `linalg,parallel` (criterion), `dhat-heap,linalg` (allocation) |
| Harness | criterion 0.5, dhat 0.3 |
| Host tmp | `TMPDIR=/home/simonm/.cache/fdars-bench-tmp` |
| Date | 2026-08-31 |

> Wall-time cells taken at `powersave` are LOW-CONFIDENCE; the primary proof for allocation-bound
> optimizations (OPT-A..D) is the dhat block/byte reduction, not wall-time.
>
> **OPT-A deviation note:** the plan's `<1000` block target was optimistic — it assumed the
> per-eigenvector `Vec` collection was nearly all of dpca's allocation. In fact ~8,000 blocks come
> from `spectral_density` (called inside `dpca`) and nalgebra `SymmetricEigen` internals allocating
> per-frequency, both outside OPT-A's `eigen_at_frequency` scope. OPT-A removed the ~9,600 eigenvector
> blocks it targeted (17,739 → 8,139, −54%), clearing the locked ≥25% bar, output provably-equivalent
> (golden `1e-12`). Further reduction would need a custom/workspace-reusing eigensolver (risky) — out of scope.

---

## Allocation results (PERF-02) — dhat, n200_m50

| OPT | Function | Before (blocks / total B / peak B) | After (blocks / total B / peak B) | Reduction |
|-----|----------|-----------------------------------|-----------------------------------|-----------|
| OPT-A | `fts::dpca` | 17,739 / 42,084,568 / 8,637,712 | 8,139 / 33,782,168 / 8,315,984 | **−54% blocks** (−20% bytes) |
| OPT-B | `fpca_variants::fsvd` | 275 / 600,049 / 410,880 | _TBD (Plan 02)_ | — |
| OPT-C | `fpca_variants::ssvd` | 22 / 314,416 / 182,384 | _TBD (Plan 02)_ | — |
| OPT-D | `fts::functional_acf` | _(measured in Plan 02)_ | _TBD_ | — |

---

## Wall-time results (PERF-01) — criterion median

| OPT | Cell | Before (PROF-01) | After | Δ |
|-----|------|------------------|-------|---|
| OPT-A | `perf_dpca/n200_m50` | (allocation-bound; wall-time secondary) | _TBD_ | _informational_ |
| OPT-E | `perf_face_covariance/n200_m30` | 983.8 ms | _TBD (Plan 03)_ | _target ≥15%_ |
| OPT-F | `perf_fem_smooth/nodes576` | 452.3 ms | _TBD (Plan 04, partial — clone removal only)_ | _informational_ |

---

## Deferred (documented, no safe behavior-preserving win)

- _TBD (Plan 04): `fem_smooth` O(N³) Cholesky + GCV-EDF bottleneck — no constant-factor win without
  sparse solvers (new dep, out of scope) or dropping GCV (breaking API). Clone removal only._
