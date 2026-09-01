---
phase: 51-benchmark-coverage-regression-guards
plan: 03
subsystem: benchmarks
tags: [benchmark, criterion, bench-coverage, BENCH-01, fem_smoothing, density_fda, fpca_variants, face]
requires:
  - phase: 51
    plan: 01
    provides: "The proven add-bench pipeline (create → Cargo register → build --benches → clippy --all-targets → fmt → commit --no-verify) + the generate_curves generator this plan reuses"
  - phase: 51
    plan: 02
    provides: "Prior 4 BENCH-01 module benches (fts, frechet, boosting_regression, coclustering); serializes the shared Cargo.toml [[bench]] edits (wave 3 after wave 2)"
provides:
  - "fdars-core/benches/fem_smoothing_benchmarks.rs — criterion bench for fem_smoothing::fem_smooth_gcv (256-node mesh, n_grid=5)"
  - "fdars-core/benches/density_fda_benchmarks.rs — lqd_fpca (n100 m81 ncomp3) + wasserstein_barycenter cells"
  - "fdars-core/benches/fpca_variants_benchmarks.rs — fpca_der (n200 m50 ncomp5 nderiv1) + fsvd cross-cov cell"
  - "fdars-core/benches/face_benchmarks.rs — irreg_fdata::mface_covariance (2 vars, n100 m30)"
  - "fdars-core/Cargo.toml — 4 new [[bench]] harness=false entries; all 9 BENCH-01 module benches now registered"
  - "BENCH-01 complete: all 9 module benches build together, lint clean, and run"
affects: []
actuals:
  tokens: 9000
  tasks: 4
  commits: 5
tech-stack:
  added: []
  patterns:
    - "each bench file is a separate compilation unit — deterministic non-RNG generators (grid_mesh, two_group_densities, generate_curves, IrregFdata::from_lists construction) copied verbatim into each bench file"
    - "criterion pattern: build data OUTSIDE b.iter(); black_box on every input AND the returned result; group.sample_size/measurement_time/warm_up_time per cost class"
    - "cost-class tuning: slow (fem_smooth_gcv, mface_covariance) → sample_size 10/15, 60s/30s; medium (lqd_fpca, fpca_der) → 20, 30s; cheap 2nd cells (wasserstein_barycenter, fsvd) → 15-20s"
key-files:
  created:
    - fdars-core/benches/fem_smoothing_benchmarks.rs
    - fdars-core/benches/density_fda_benchmarks.rs
    - fdars-core/benches/fpca_variants_benchmarks.rs
    - fdars-core/benches/face_benchmarks.rs
  modified:
    - fdars-core/Cargo.toml
key-decisions:
  - "fem_smooth_gcv uses a SMALLER 256-node mesh (grid_mesh(16)) than the 576-node fem_smooth already benched — GCV multiplies the O(N^3) solve by n_grid; log_lambda_range=(-4.0,0.0), n_grid=5. Signature VERIFIED at src/fem_smoothing.rs:641."
  - "density_fda: added the optional cheap wasserstein_barycenter 2nd cell (signature (density_matrix, argvals, Option<&[f64]> weights) VERIFIED at src/density_fda.rs:407). lqd_fpca signature VERIFIED at src/density_fda.rs:563."
  - "fpca_variants: added the optional fsvd 2nd cell. fsvd needs two curve sets — added a deterministic phase-shifted generate_curves_shifted (cos-based) so the cross-covariance is non-trivial. fpca_der signature VERIFIED at src/fpca_variants.rs:189, fsvd at src/fpca_variants.rs:405."
  - "face bench used mface_covariance (NOT the face_trajectory fallback) — it ran fine at ~1.23 s with 2 vars n100 m30. Imports fdars_core::irreg_fdata::{mface_covariance, IrregFdata} (re-export VERIFIED at src/irreg_fdata/mod.rs:31); mface_covariance signature VERIFIED at src/irreg_fdata/face.rs:263."
  - "No down-sizing (RESEARCH A1/A2) needed — all cells ran within budget. No #[non_exhaustive] config struct was touched (none of the 4 targets take a config struct), so the default()-then-reassign allow was not needed this plan."
  - "New [[bench]] blocks appended adjacent to the prior Phase-51 blocks; the PERMANENT perf_hotpaths/perf_parallelism blocks + comments left intact. No src/ edit, no new dependency. No v* tag (audit milestone — a tag would trigger a phantom crates.io publish)."
requirements-completed: [BENCH-01]
coverage:
  - id: D1
    description: "4 new module benches (fem_smooth_gcv, lqd_fpca, fpca_der, mface_covariance) registered as [[bench]] harness=false; all 9 BENCH-01 module benches build under --benches and lint clean under clippy --all-targets"
    requirement: BENCH-01
    verification:
      - kind: integration
        ref: "cargo build -p fdars-core --benches --features linalg,parallel => Finished (all 9 module + 2 permanent benches compiled together)"
        status: pass
      - kind: lint
        ref: "cargo clippy --all-targets --features linalg,parallel -- -D warnings => clean (primary gate, lints bench code)"
        status: pass
      - kind: integration
        ref: "each --bench <name> -- --quick ran its cell(s) once without panic (baselines captured below)"
        status: pass
    human_judgment: false
status: complete
---

# Phase 51 Plan 03: Final 4 Module Benches (BENCH-01 Complete) Summary

Added the last 4 criterion module benches of Phase 51, completing BENCH-01. Each covers a
representative sibling entry NOT already benched, mirroring the `perf_hotpaths.rs` pipeline exactly
(deterministic non-RNG data built OUTSIDE `b.iter()`, `black_box` on inputs and result, cost-class
tuning). After this plan all 9 BENCH-01 module benches (fts, frechet, boosting_regression,
coclustering, fem_smoothing, density_fda, inference, fpca_variants, face) build together, lint
clean, and run.

## What shipped

- **`benches/fem_smoothing_benchmarks.rs`** — `fem_smooth_gcv` on a 256-node mesh (`grid_mesh(16)`,
  copied verbatim), `log_lambda_range=(-4.0,0.0)`, `n_grid=5`. Cell `nodes256_ngrid5`.
  sample_size(10), 60s/3s (slow — n_grid × O(N³)). Smaller mesh than the 576-node `fem_smooth`
  already in `perf_hotpaths` (GCV multiplies the solve by n_grid).
- **`benches/density_fda_benchmarks.rs`** — copies `two_group_densities` (strictly-positive rows)
  verbatim. Cell `density_lqd_fpca/n100_m81_ncomp3` (medium, 20/30s) + cheap 2nd cell
  `density_wasserstein_barycenter/n100_m81` (20/15s).
- **`benches/fpca_variants_benchmarks.rs`** — copies `generate_curves` verbatim + a deterministic
  cos-based `generate_curves_shifted` for the cross-covariance input. Cell
  `fpca_der/n200_m50_ncomp5_nderiv1` (medium, 20/30s) + 2nd cell `fpca_fsvd/n200_m50_ncomp5`.
- **`benches/face_benchmarks.rs`** — 2 `IrregFdata` variables (n=100, m=30) built via
  `IrregFdata::from_lists` (mirroring the `perf_hotpaths` construction), differing phase/amp/freq.
  Cell `mface_covariance/vars2_n100_m30`, bandwidth=0.3. sample_size(15), 30s/3s (medium/slow).
  Used `mface_covariance` — the `face_trajectory` fallback was NOT needed.
- **`Cargo.toml`** — 4 new `[[bench]] harness=false` entries appended adjacent to the prior
  Phase-51 blocks; the PERMANENT `perf_hotpaths`/`perf_parallelism` blocks untouched.
- **No src/ edit; no new dependency.**

## Baselines captured (for BENCH-RESULTS.md)

| Bench cell | Inputs | Median (--quick) |
|------------|--------|------------------|
| `fem_smooth_gcv/nodes256_ngrid5` | 256 nodes (k=16), n_grid=5, log_lambda (-4,0) | **~305.8 ms** (303.7–306.3) |
| `density_lqd_fpca/n100_m81_ncomp3` | n=100, m=81, ncomp=3 | **~3.04 ms** (2.97–3.06) |
| `density_wasserstein_barycenter/n100_m81` | n=100, m=81 | **~389 µs** (388.8–390.8) |
| `fpca_der/n200_m50_ncomp5_nderiv1` | n=200, m=50, ncomp=5, nderiv=1 | **~9.76 ms** (9.42–11.12) |
| `fpca_fsvd/n200_m50_ncomp5` | two n=200 m=50 sets, ncomp=5 | **~1.28 ms** (1.22–1.29) |
| `mface_covariance/vars2_n100_m30` | 2 vars, n=100, m=30, bw=0.3 | **~1.232 s** (1.22–1.28) |

All cells ran within budget; NO cell down-sizing was applied (RESEARCH A1/A2 not exercised).

## Optional 2nd cells added

- **wasserstein_barycenter** (density_fda) — added (cheap, ~389 µs).
- **fsvd** (fpca_variants) — added (cheap, ~1.28 ms); required a deterministic phase-shifted second
  curve generator for the cross-covariance input.

## Commit count

5 atomic commits:
- `98b2f89d` feat(51-03): add fem_smoothing_benchmarks bench (fem_smooth_gcv)
- `bdde7412` feat(51-03): add density_fda_benchmarks bench (lqd_fpca + wasserstein_barycenter)
- `5399bb9e` feat(51-03): add fpca_variants_benchmarks bench (fpca_der + fsvd)
- `bf522b3d` feat(51-03): add face_benchmarks bench (mface_covariance)
- (this SUMMARY commit)

## Gate results

| Gate | Result |
|------|--------|
| `cargo build -p fdars-core --benches --features linalg,parallel` | clean — all 9 module + 2 permanent benches compiled together |
| `cargo clippy --all-targets --features linalg,parallel -- -D warnings` | clean after every task (primary gate — lints bench code) |
| `cargo bench --bench <name> -- --quick` (×4 files, 6 cells) | each ran its cell(s) once without panic; baselines captured |
| `cargo fmt -p fdars-core` | applied per commit (reformatted the `make_variable` signature in face_benchmarks — cosmetic) |
| commit | `git commit --no-verify` per task (avoids the slow full hook that stalls executor watchdogs) |

## Deviations from Plan

None substantive — plan executed as written. Every cargo command carried the
`TMPDIR=/home/simonm/.cache/fdars-bench-tmp` prefix; a pre-emptive
`rm -rf target/debug/{incremental,examples}` was run before the first build (31 G free on /home);
no link/disk failure occurred, so no retry was needed. `cargo fmt` reformatted the `make_variable`
signature in `face_benchmarks.rs` (multi-line params) — cosmetic, committed. No `v*` tag created
(crate version stays 0.29.0). No stray `.planning/state.json` appeared.

## Known Stubs

None. Every bench calls the real function (`fem_smooth_gcv`, `lqd_fpca`, `wasserstein_barycenter`,
`fpca_der`, `fsvd`, `mface_covariance`) with concrete deterministic inputs and `.unwrap()`s a real
result. No TODOs, empty returns, or mock data.

## Threat Flags

None. All bench inputs are hard-coded deterministic generators — no external/untrusted input, no
network, no auth/crypto (attack surface nil). Threat T-51-02 (phantom crates.io publish from a `v*`
tag) mitigated exactly: no `v*` tag pushed, crate version unchanged at 0.29.0. T-51-SC (package
install) untouched — no dependency added (criterion 0.5 is a pre-existing dev-dep).

## Self-Check: PASSED

- `fdars-core/benches/fem_smoothing_benchmarks.rs` — FOUND
- `fdars-core/benches/density_fda_benchmarks.rs` — FOUND
- `fdars-core/benches/fpca_variants_benchmarks.rs` — FOUND
- `fdars-core/benches/face_benchmarks.rs` — FOUND
- `fdars-core/Cargo.toml` 4 new `[[bench]]` entries (21 total) — FOUND
- Commits `98b2f89d`, `bdde7412`, `5399bb9e`, `bf522b3d` — FOUND
