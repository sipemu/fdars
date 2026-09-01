# BENCH-RESULTS — v0.30.0 Regression-Guard Ledger (BENCH-02)

Consolidated benchmark ledger for the v0.30.0 Performance & Consolidation Pass. Documents the
Phase 47/48 PERF wins as guarded baselines and records first-run baselines for the 9 modules newly
covered in Phase 51 (BENCH-01).

## Guard model

- **Documented SOFT guards** — the criterion benches (`benches/*.rs`) MEASURE; they do **not** fail CI
  on regression. Regressions of the Phase 47/48 wins are caught by running `cargo bench` and comparing
  against the numbers below (criterion's built-in baseline comparison, `--save-baseline` / `--baseline`).
  This is a deliberate, locked decision: wall-time thresholds are governor-sensitive (see caveat) and
  a hard `assert!` would flake on an unpinned CI runner.
- **Deterministic HARD guards** — `tests/alloc_audit_dpca.rs` and `tests/alloc_audit_fpca.rs`
  (feature `dhat-heap`) hard-assert allocation-block counts. Allocation counts are deterministic and
  governor-independent, so these ARE a real CI failure surface. Unchanged this phase.
- **No wall-time `assert!` regression tests were added** — governor-sensitive; explicitly out of scope.

## Phase 47 wins — Hot-Path & Allocation Performance (PERF-01/02)

Source: `.planning/phases/47-hot-path-allocation-performance/PERF-RESULTS.md`.

| Opt | Path | Cell | Before → After | Win | Guard cell |
|-----|------|------|----------------|-----|-----------|
| OPT-A | `fts::dpca` | n200_m50 | 17,739 → 8,139 alloc blocks (−20% bytes) | **−54% blocks** | `tests/alloc_audit_dpca.rs` (hard) + `perf_hotpaths::perf_dpca` |
| OPT-E | `irreg_fdata::face_covariance` | perf_face_covariance/n200_m30 | 983.8 ms → 189.8 ms | **−80.7% wall** | `perf_hotpaths::perf_face_covariance` (soft) |
| OPT-B | `fsvd` (fpca_variants) | — | 275 → 274 alloc blocks | −1 block | `tests/alloc_audit_fpca.rs` (hard) |
| OPT-C | `ssvd` | — | 22 → 21 alloc blocks | −1 block | `tests/alloc_audit_fpca.rs` (hard) |
| OPT-D | `functional_acf` | — | −1 block + ~(m²−m) sqrt dropped | alloc + compute | (covered by fpca alloc audit) |
| OPT-F | `fem_smoothing::fem_smooth` | perf_fem_smooth/nodes576 | clone removed (alloc win) | alloc; O(N³) solve DEFERRED | `perf_hotpaths::perf_fem_smooth` (soft) |

## Phase 48 wins — Parallelism-Gap Closure (PERF-03)

Source: `.planning/phases/48-parallelism-gap-closure/PERF-PARALLEL-RESULTS.md`. Thread-scaling via the
env-var sweep:

```
RAYON_NUM_THREADS=1  TMPDIR=/home/simonm/.cache/fdars-bench-tmp \
  cargo bench -p fdars-core --features linalg,parallel --bench perf_parallelism -- <cell>
RAYON_NUM_THREADS=20 TMPDIR=/home/simonm/.cache/fdars-bench-tmp \
  cargo bench -p fdars-core --features linalg,parallel --bench perf_parallelism -- <cell>
```

| Path | Cell | 1-thread → 20-thread | Speedup | Guard cell |
|------|------|----------------------|---------|-----------|
| `frechet::frechet_anova` | perf_parallelism_frechet_anova/n24_m81_nperm999 | 322.73 ms → 32.57 ms | **9.9×** (criterion −89.8%, p<0.05) | `perf_parallelism` (soft) |
| `coclustering::co_cluster` | perf_parallelism_co_cluster/n200_m50_ninit8 | 337.34 ms → 52.91 ms | **6.4×** (−84.0%, p<0.05) | `perf_parallelism` (soft) |

## New-module baselines — BENCH-01 (Phase 51)

First-run `--quick` medians (NOT before/after — these establish the baseline for future comparison).
All are `[[bench]] harness=false`, one bench file per module.

| Module | Bench (target fn) | Cell | Baseline median |
|--------|-------------------|------|-----------------|
| `inference` | `inference_benchmarks` (`t_perm_test`) | na30_nb30_m50_nperm999 | ~2.196 ms |
| `fts` | `fts_benchmarks` (`ftsm`) | n200_m50_ncomp3 | ~12.7 ms |
| `frechet` | `frechet_benchmarks` (`frechet_global_reg`, concrete) | n24_m81_xout5 | ~264 µs |
| `boosting_regression` | `boosting_regression_benchmarks` (`boost_fosr`) | n100_m50_p2_mstop100 | ~28.2 ms |
| " | " | n100_m50_p2_mstop50 (knob variant) | ~14.5 ms |
| `coclustering` | `coclustering_benchmarks` (`co_cluster_select`) | n120_m40_grid2x2 | ~232 ms |
| `fem_smoothing` | `fem_smoothing_benchmarks` (`fem_smooth_gcv`) | nodes256_ngrid5 (k=16) | ~305.8 ms |
| `density_fda` | `density_fda_benchmarks` (`lqd_fpca`) | n100_m81_ncomp3 | ~3.04 ms |
| " | " (`wasserstein_barycenter`) | n100_m81 | ~389 µs |
| `fpca_variants` | `fpca_variants_benchmarks` (`fpca_der`) | n200_m50_ncomp5_nderiv1 | ~9.76 ms |
| " | " (`fsvd`) | n200_m50_ncomp5 | ~1.28 ms |
| `face` | `face_benchmarks` (`irreg_fdata::mface_covariance`) | vars2_n100_m30 (bw=0.3) | ~1.232 s |

Populate/refresh any cell with `cargo bench -p fdars-core --bench <name>` (drop `--quick` for full samples).

## Environment / governor caveat

- **CPU governor `powersave`, unpinned** (sudo/cpupower unavailable). → All **wall-time** cells are
  **LOW-CONFIDENCE**: treat absolute medians and ratios as indicative. The thread-scaling *direction*
  (N-thread ≪ 1-thread) remains a valid signal; treat the 9.9×/6.4× as lower bounds. Re-run under a
  `performance` governor to supersede the LOW-CONFIDENCE absolute medians.
- **Allocation wins (OPT-A..D) are governor-INDEPENDENT, HIGH-confidence** dhat block counts — the
  hard `alloc_audit_*` guards protect them regardless of CPU state.
- Host: 20 logical cores; `RAYON_NUM_THREADS` default = 20. Features: `linalg,parallel` (criterion) /
  `dhat-heap,linalg` (alloc audits). criterion 0.5, dhat 0.3 (pre-existing dev-deps; no new dependency).
- `TMPDIR=/home/simonm/.cache/fdars-bench-tmp` required for all bench/doctest builds (/tmp tmpfs
  exhaustion otherwise gives bogus "No space left" at link time).
- Date: 2026-09-01.

## Guard inventory

| Guard | Kind | Status |
|-------|------|--------|
| `benches/perf_hotpaths.rs` (`[[bench]]`, Cargo.toml) | PERMANENT soft (Phase 47) | ✅ registered, unchanged this phase |
| `benches/perf_parallelism.rs` (`[[bench]]`, Cargo.toml) | PERMANENT soft (Phase 48) | ✅ registered, unchanged this phase |
| `tests/alloc_audit_dpca.rs` | deterministic HARD (dhat-heap) | ✅ intact |
| `tests/alloc_audit_fpca.rs` | deterministic HARD (dhat-heap) | ✅ intact |
| 9 new `benches/<module>_benchmarks.rs` | soft coverage baselines (Phase 51) | ✅ added + registered |

No `Cargo.toml` or `src/` edit in the BENCH-02 (this) plan — documentation only. Crate version
unchanged (0.29.0); no `v*` tag pushed (audit/perf milestone — a tag would trigger a phantom
crates.io publish; the version bump + publish is the deferred operator step REL-01).
