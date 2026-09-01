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
| OPT-B | `fpca_variants::fsvd` | 275 blocks | 274 blocks | −1 block (gram staging Vec + m×m matrix copy) |
| OPT-C | `fpca_variants::ssvd` | 22 blocks | 21 blocks | −1 block (c_scaled staging Vec + m×m matrix copy) |
| OPT-D | `fts::functional_acf` | (staging Vec + ~m² redundant `sqrt`) | from_fn + m `sqrt` | −1 block + drops ~(m²−m) `sqrt()` calls |

> OPT-B/C/D are pure copy-eliminations: each removes one `Vec` staging buffer + the redundant m×m
> matrix copy into nalgebra (block count drops by 1; the avoided bytes are the m×m copy). OPT-D also
> replaces ~m² `weights[j].sqrt()` recomputations with m precomputed roots. All three proven
> byte-equivalent by golden tests at rel 1e-12.

---

## Wall-time results (PERF-01) — criterion median

| OPT | Cell | Before (PROF-01) | After | Δ |
|-----|------|------------------|-------|---|
| OPT-A | `perf_dpca/n200_m50` | (allocation-bound; wall-time secondary) | _TBD_ | _informational_ |
| OPT-E | `perf_face_covariance/n200_m30` | 983.8 ms | 189.8 ms [167.8, 217.3] | **−80.7%** (non-overlapping CIs; ≫15% bar) |
| OPT-F | `perf_fem_smooth/nodes576` | 452.3 ms | ≈ unchanged (clone removal is an allocation win: one N×N ~2.6 MB copy dropped; compute unchanged) | O(N³) solve DEFERRED |

---

## Deferred (documented, no safe behavior-preserving win)

- **`fem_smooth` O(N³) dense Cholesky + GCV-EDF column solves** (the ~452 ms @ 576-nodes bottleneck,
  `src/fem_smoothing.rs`). No safe behavior-preserving constant-factor win exists without either
  (a) sparse assembly/solvers — requires a new crate dependency, out of scope for this
  no-new-dependency milestone; or (b) skipping the GCV `edf` computation — would change the returned
  `edf`/`gcv` fields, a breaking API change, out of scope. OPT-F landed the safe part (drop the
  `phi_t_phi.clone()` N×N copy). A rustdoc DEFER note records this at the A⁻¹ loop. Revisit in a
  future sparse-linalg or 1.0-readiness milestone.

## Summary (OPT-A..F)

| OPT | Target | Result | Requirement |
|-----|--------|--------|-------------|
| A | `fts::dpca` | −54% alloc blocks (17,739→8,139), behavior-preserving | PERF-02 |
| B | `fpca_variants::fsvd` | copy removal (275→274 blocks + m×m copy) | PERF-02 |
| C | `fpca_variants::ssvd` | copy removal (22→21 blocks + m×m copy) | PERF-02 |
| D | `fts::functional_acf` | copy removal + ~m² fewer `sqrt()` | PERF-02 |
| E | `irreg_fdata::face_covariance` | **−80.7% wall-time** (983.8→189.8 ms) | PERF-01 |
| F | `fem_smoothing::fem_smooth` | clone removal (N×N alloc); O(N³) solve DEFERRED | PERF-01 |

All six behavior-preserving (golden equivalence tests, rel ≤1e-12). Headline wins: **E −80.7% wall-time**
(PERF-01), **A −54% allocations** (PERF-02). The `perf_hotpaths` benches + this ledger feed Phase 51 (BENCH-02).
