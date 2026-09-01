---
phase: 51-benchmark-coverage-regression-guards
plan: 02
subsystem: benchmarks
tags: [benchmark, criterion, bench-coverage, BENCH-01]
requires:
  - phase: 51
    plan: 01
    provides: "benches/inference_benchmarks.rs + the proven create→register→build→clippy→fmt→commit add-bench pipeline this plan repeats 4×"
  - phase: 47
    provides: "benches/perf_hotpaths.rs — generate_curves generator + criterion structure mirrored"
  - phase: 48
    provides: "benches/perf_parallelism.rs — two_group_densities + co_cluster_curves generators mirrored"
provides:
  - "fdars-core/benches/fts_benchmarks.rs — criterion bench for fts::ftsm (n200_m50_ncomp3 cell)"
  - "fdars-core/benches/frechet_benchmarks.rs — criterion bench for frechet::frechet_global_reg (n24_m81_xout5 cell)"
  - "fdars-core/benches/boosting_regression_benchmarks.rs — criterion bench for boosting_regression::boost_fosr (mstop100 + mstop50 cells)"
  - "fdars-core/benches/coclustering_benchmarks.rs — criterion bench for coclustering::co_cluster_select (n120_m40_grid2x2 cell)"
  - "fdars-core/Cargo.toml — 4 new [[bench]] harness=false entries"
affects: [51-03, 51-04]
actuals:
  tokens: 14500
  tasks: 4
  commits: 5
tech-stack:
  added: []
  patterns:
    - "each bench file is a separate compilation unit — the deterministic non-RNG generator (generate_curves / two_group_densities / co_cluster_curves) is copied verbatim into each bench file"
    - "criterion pattern: build data OUTSIDE b.iter(); black_box every input AND the returned result; group.sample_size/measurement_time/warm_up_time per cost class"
    - "config structs marked #[non_exhaustive] (CoClusterConfig) forbid struct-update syntax outside the crate (E0639) → reassign-after-default with a targeted #[allow(clippy::field_reassign_with_default)]; non-non_exhaustive structs (BoostingConfig) use the clean `Config { field, ..Default::default() }` form"
key-files:
  created:
    - fdars-core/benches/fts_benchmarks.rs
    - fdars-core/benches/frechet_benchmarks.rs
    - fdars-core/benches/boosting_regression_benchmarks.rs
    - fdars-core/benches/coclustering_benchmarks.rs
  modified:
    - fdars-core/Cargo.toml
key-decisions:
  - "Import paths VERIFIED live: fts::ftsm (src/fts/mod.rs:27 re-export), frechet::frechet_global_reg (src/frechet/mod.rs:42 re-export), boosting_regression::{boost_fosr, BoostingConfig} (mod.rs:330 + struct at :44), coclustering::{co_cluster_select, CoClusterConfig} (co_cluster_select at coclustering.rs:1119, CoClusterConfig at :153)."
  - "frechet: used the CONCRETE frechet_global_reg (not the generic _space<S: MetricSpace>) to avoid a Sync/object-construction complication. All 4 input dims satisfy validate_reg_input: 24 density rows + 81 monotone argvals (−3..3), 1-col n-row predictor, 5-row 1-col xout."
  - "boost_fosr: kept BOTH the representative default cell (mstop=100, ~28.2 ms) and the lighter mstop=50 variant (~14.5 ms). The default single-shot was fast enough to keep (RESEARCH A2 — mstop is an executor-tunable knob demonstrated by the second cell); no down-sizing of n/m/p was needed."
  - "co_cluster_select: kept the sweep grid tiny (2×2: k=[2,3], l=[2,3]) and bounded per-fit cost (max_iter=20, n_init=3 default) → ~232 ms per iter, not minutes-long."
requirements-completed: [BENCH-01]
coverage:
  - id: D1
    description: "4 module benches (ftsm, frechet_global_reg, boost_fosr, co_cluster_select) registered as [[bench]] harness=false; compile under --benches and lint clean under clippy --all-targets"
    requirement: BENCH-01
    verification:
      - kind: integration
        ref: "cargo build -p fdars-core --benches --features linalg,parallel => Finished (all 4 new benches compiled)"
        status: pass
      - kind: lint
        ref: "cargo clippy --all-targets --features linalg,parallel -- -D warnings => clean (primary gate, lints bench code)"
        status: pass
      - kind: integration
        ref: "each --bench <name> -- --quick ran its cell(s) once without panic; baseline medians captured"
        status: pass
    human_judgment: false
status: complete
---

# Phase 51 Plan 02: Module Benchmark Coverage (BENCH-01, Wave 2) Summary

Added 4 new criterion bench files covering the sibling public entry points of 4 modules whose
headline fn was already benched — `fts::ftsm` (dpca already benched), `frechet::frechet_global_reg`
(frechet_anova already benched), `boosting_regression::boost_fosr` (no prior coverage), and
`coclustering::co_cluster_select` (co_cluster single-fit already benched). Each mirrors the proven
51-01 tracer pipeline: copy a deterministic non-RNG generator into the bench file (benches are
separate compilation units), build data outside `b.iter()`, `black_box` inputs and the result,
register a `[[bench]] harness=false` entry adjacent to the existing blocks, then
build → clippy → fmt → `commit --no-verify`.

## What shipped

- **`benches/fts_benchmarks.rs`** — group `fts_ftsm`, cell `n200_m50_ncomp3`:
  `ftsm(&data, 3, &argvals)`, `generate_curves(200, 50)` copied from perf_hotpaths.rs.
  medium cost class (sample_size=20, measurement_time=30s, warm_up=3s).
- **`benches/frechet_benchmarks.rs`** — group `frechet_global_reg`, cell `n24_m81_xout5`:
  concrete `frechet_global_reg(&predictors, &responses, &argvals, &xout)`. `two_group_densities(12, 81)`
  supplies the 24 density-response rows + 81 monotone argvals; predictors = 1-col n-row deterministic
  scalar covariate; xout = 5-row 1-col query points. medium cost class.
- **`benches/boosting_regression_benchmarks.rs`** — group `boosting_boost_fosr`, TWO cells:
  `n100_m50_p2_mstop100` (BoostingConfig::default) and `n100_m50_p2_mstop50` (mstop knob variant).
  `generate_curves(100, 50)` response + a deterministic n×2 Euclidean predictor. slow cost class
  (sample_size=10, measurement_time=60s, warm_up=3s).
- **`benches/coclustering_benchmarks.rs`** — group `coclustering_co_cluster_select`, cell
  `n120_m40_grid2x2`: `co_cluster_select(&data, &argvals, &[2,3], &[2,3], &config)` with
  `co_cluster_curves(120, 40)`, `n_init=3`, `max_iter=20`. slow cost class.
- **`Cargo.toml`** — 4 new `[[bench]] harness = false` entries appended adjacent to the PERMANENT
  `perf_hotpaths` / `perf_parallelism` blocks (their comments untouched).
- **No src/ edit; no new dependency** (criterion 0.5 is a pre-existing dev-dep).

## Baselines captured (for BENCH-RESULTS.md, plan 51-04)

| Bench cell | Params | Median (--quick) |
|------------|--------|------------------|
| `fts_ftsm/n200_m50_ncomp3` | n=200, m=50, ncomp=3 | **~12.7 ms** (9.96–13.34 ms range) |
| `frechet_global_reg/n24_m81_xout5` | n=24, m=81, xout=5 | **~264 µs** (254.6–266.3 µs) |
| `boosting_boost_fosr/n100_m50_p2_mstop100` | n=100, m=50, p=2, mstop=100 | **~28.2 ms** (28.16–28.30 ms) |
| `boosting_boost_fosr/n100_m50_p2_mstop50` | n=100, m=50, p=2, mstop=50 | **~14.5 ms** (14.42–14.71 ms) |
| `coclustering_co_cluster_select/n120_m40_grid2x2` | n=120, m=40, grid 2×2, max_iter=20 | **~232 ms** (224.4–234.2 ms) |

`--quick` medians are indicative (few samples); plan 51-04 captures governed baselines.

## Commit count

5 atomic commits:
- `de91c0df` feat(51-02): add fts_benchmarks bench (ftsm)
- `07aac82e` feat(51-02): add frechet_benchmarks bench (frechet_global_reg)
- `10f99a08` feat(51-02): add boosting_regression_benchmarks bench (boost_fosr)
- `c3ef720c` feat(51-02): add coclustering_benchmarks bench (co_cluster_select)
- (this SUMMARY commit)

## Gate results

| Gate | Result |
|------|--------|
| `cargo build -p fdars-core --benches --features linalg,parallel` | clean — all 4 new benches compiled |
| `cargo clippy --all-targets --features linalg,parallel -- -D warnings` | clean (primary gate — lints bench code) |
| `--bench <name> -- --quick` (×4 benches, 5 cells) | all ran once without panic; baselines captured above |
| `cargo fmt -p fdars-core --check` | clean (no drift) |
| commit | `git commit --no-verify` per task (avoids the slow full hook that stalls executor watchdogs) |

Every cargo command was prefixed with `TMPDIR=/home/simonm/.cache/fdars-bench-tmp`; a pre-emptive
`rm -rf target/debug/{incremental,examples}` was run before the first build. No link/disk failure
occurred, so no retry was needed. No `v*` tag was created (audit milestone — a tag would trigger a
phantom crates.io publish; crate version stays 0.29.0). No stray `.planning/state.json` appeared.

## Deviations from Plan

- **Task 3 (boost_fosr):** the plan made the `mstop=50` variant conditional on the default cell
  being slow. The default `mstop=100` cell measured ~28 ms/iter — fast enough that it did NOT
  require replacement, but the `mstop=50` variant was ADDED anyway (per the plan's "keep both" note
  and RESEARCH A2) to demonstrate the executor-tunable mstop knob. No n/m/p down-sizing applied.
- **Task 4 (co_cluster_select):** `CoClusterConfig` is `#[derive(...)] #[non_exhaustive]`, so the
  struct-update literal `CoClusterConfig { n_init: 3, max_iter: 20, ..default() }` is rejected
  outside the defining crate (E0639). Resolved with the reassign-after-default pattern behind a
  targeted `#[allow(clippy::field_reassign_with_default)]` (the same pattern the existing
  perf_parallelism.rs co_cluster bench uses). `BoostingConfig` (Task 3) is NOT non_exhaustive, so
  its lighter variant uses the clean `..Default::default()` form.
- No cell was down-sized for bench time; all cells run in ≤~0.25 s/iter.

## Known Stubs

None. Each bench calls the real module entry point (`ftsm`, `frechet_global_reg`, `boost_fosr`,
`co_cluster_select`) with concrete deterministic inputs; no TODOs, empty returns, or mock data.

## Threat Flags

None. Bench inputs are hard-coded deterministic generators — no external/untrusted input, no
network, no auth/crypto (RESEARCH Security Domain: attack surface nil). Threat T-51-02 (phantom
crates.io publish from a `v*` tag) mitigated exactly as prescribed: no `v*` tag pushed, crate
version unchanged at 0.29.0. T-51-SC (package install) untouched — no dependency added.

## Self-Check: PASSED

- `fdars-core/benches/fts_benchmarks.rs` — FOUND
- `fdars-core/benches/frechet_benchmarks.rs` — FOUND
- `fdars-core/benches/boosting_regression_benchmarks.rs` — FOUND
- `fdars-core/benches/coclustering_benchmarks.rs` — FOUND
- `fdars-core/Cargo.toml` 4 new `[[bench]]` entries — FOUND
- Commits `de91c0df`, `07aac82e`, `10f99a08`, `c3ef720c` — FOUND
