# Phase 51: Benchmark Coverage & Regression Guards - Research

**Researched:** 2026-09-01
**Domain:** Rust criterion micro-benchmark harness for a functional-data-analysis library (measurement-only, no `src/` change, no new dependency)
**Confidence:** HIGH (all findings are from in-repo source read this session; `[VERIFIED: <path:lines>]` with verbatim signatures)

## Summary

Phase 51 is a pure measurement/coverage phase. **BENCH-01** adds one `[[bench]]` file per module (`harness = false`) for the 9 currently-unbenchmarked modules, mirroring the established `benches/perf_hotpaths.rs` template (deterministic non-RNG data generators built once outside `b.iter()`, `black_box` on inputs and outputs, `sample_size`/`measurement_time` tuned per cost). **BENCH-02** confirms the two PERMANENT PERF-proof benches (`perf_hotpaths.rs`, `perf_parallelism.rs`) remain registered and lands `BENCH-RESULTS.md` consolidating the Phase 47/48 before/after numbers with the governor caveat. The `alloc_audit_dpca.rs`/`alloc_audit_fpca.rs` deterministic hard-assert guards stay untouched.

The primary gate is `cargo clippy --all-targets --features linalg,parallel -- -D warnings` (it lints bench code — a plain `-p … -D warnings` false-greens per project MEMORY). No module in the 9 targets is `linalg`-gated at the source level `[VERIFIED: src/lib.rs:79-104 + grep for cfg(feature="linalg") returned no matches in target modules]`, so all bench cells compile under both `--no-default-features` and `--features linalg,parallel`. No RNG appears in any recommended data generator — every cell reuses the sinusoid `generate_curves(n,m)` / density generators already proven in `perf_hotpaths.rs` / `perf_parallelism.rs`.

**Primary recommendation:** Use `inference` (`t_perm_test`) as the tracer — it is the cheapest, most self-contained cell (two `FdMatrix` inputs from the existing sinusoid generator, `[VERIFIED: src/inference/permutation.rs:152-158]`), proves the full add-bench → register-in-Cargo.toml → clippy-gate pipeline, and has zero feature or generic-bound complications. Then fan out the remaining 8 modules.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **Documented baselines (soft guards)** — criterion benches MEASURE; they do not fail CI on regression. Regressions caught via `cargo bench` + criterion baseline comparison against committed before/after numbers.
- **Keep the existing DETERMINISTIC hard-assert guards** — `tests/alloc_audit_dpca.rs` + `tests/alloc_audit_fpca.rs` (feature `dhat-heap`) stay as real CI guards.
- **NO wall-time `assert!` regression tests** — governor-sensitive; would flake. Do NOT add them.
- **BENCH-RESULTS.md** — consolidated ledger documenting PERF wins as guarded baselines, referencing PERF-RESULTS.md (Phase 47) + PERF-PARALLEL-RESULTS.md (Phase 48): face_covariance −80.7%, dpca −54% allocations, frechet_anova 9.9×, co_cluster 6.4×. Record environment/governor caveat.
- **One `[[bench]]` file per module** (mirror `*_benchmarks.rs` naming + `harness = false`), all 9 modules; register each in `fdars-core/Cargo.toml`.
- Mirror the criterion pattern (`benches/perf_hotpaths.rs` is the template): build deterministic data OUTSIDE `b.iter()` (NO RNG), `black_box` inputs+outputs, `group.sample_size(N)` + `group.measurement_time(Duration)` tuned per cost.
- Cover the module's OTHER representative entry points — do NOT duplicate the cells already in perf_hotpaths/perf_parallelism (fts::dpca, frechet_anova, fem_smooth, face_covariance, co_cluster). Pick 1–3 representative public fns per module.
- No behavior-changing `src/` edits; no new dependency. `cargo fmt` + `git commit --no-verify` per commit.
- `perf_hotpaths.rs` + `perf_parallelism.rs` remain `[[bench]]`-registered PERMANENT; keep alloc_audit tests as-is.

### Claude's Discretion
- Exact bench-cell sizes (n/m/mesh) per module, tuned to reasonable bench time.
- Whether/how to feature-gate any linalg-only bench cell (match how perf_hotpaths handles it — research finds NO linalg gating is needed for these 9 modules).
- Exact gate mechanics; optionally run each new bench once (short sample) to capture a BENCH-RESULTS.md baseline cell.

### Deferred Ideas (OUT OF SCOPE)
- Hard wall-time `assert!` regression tests — governor-sensitive/flaky.
- Any behavior-changing `src/` optimization — measurement/coverage only.
- CI wiring to auto-run benches on every PR — benches stay `cargo bench`-on-demand.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| BENCH-01 | Add criterion `[[bench]]` coverage for the 9 unbenchmarked modules (`fts`, `frechet`, `boosting_regression`, `coclustering`, `fem_smoothing`, `density_fda`, `inference`, `fpca_variants`, `face`). | Per-module target table below: exact public fn + verbatim signature + non-RNG input construction + size + cost class + sample_size/measurement_time recommendation. |
| BENCH-02 | Commit PERF-proof benches (`perf_hotpaths.rs`, `perf_parallelism.rs`) as regression guards with documented before/after numbers. | Both confirmed already `[[bench]]`-registered PERMANENT `[VERIFIED: Cargo.toml:98-104]`. BENCH-02 Consolidation Plan section extracts the before/after ledger from PERF-RESULTS.md + PERF-PARALLEL-RESULTS.md. alloc_audit guards confirmed present. |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| New bench cells (BENCH-01) | Dev tooling (`benches/`) | — | Criterion benches are dev-dependencies; never linked into the library |
| Regression baselines ledger (BENCH-02) | Planning docs (`.planning/`) | Dev tooling | Documented soft guards live as markdown; the measuring benches live in `benches/` |
| Deterministic alloc hard-asserts | Integration tests (`tests/`) | — | Feature `dhat-heap` guards; real CI failure surface (unchanged this phase) |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| criterion | 0.5 (html_reports) | Statistical micro-benchmark harness | Already the sole bench harness; existing dev-dep `[VERIFIED: Cargo.toml:53]` |
| std::time::Duration | std | `measurement_time`/`warm_up_time` tuning | Used by every existing bench `[VERIFIED: benches/perf_hotpaths.rs:13]` |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| dhat | 0.3 | Allocation-block hard-assert probes | ONLY in existing `tests/alloc_audit_*.rs` (feature `dhat-heap`) — do not touch `[VERIFIED: Cargo.toml:54]` |

**No new dependency.** criterion + dhat are already dev-deps. This is a locked constraint.

### Installation
None — no crate added. New files register in `[[bench]]` blocks of `fdars-core/Cargo.toml`.

## Package Legitimacy Audit

> Not applicable — this phase installs **no external packages** (locked "no new dependency" constraint). criterion 0.5 and dhat 0.3 are pre-existing dev-dependencies `[VERIFIED: Cargo.toml:52-54]`. No legitimacy gate to run.

## Per-Module Benchmark-Target Table (BENCH-01)

All signatures below are `[VERIFIED]` — read verbatim from source this session. All inputs reuse the deterministic non-RNG sinusoid `generate_curves(n,m)` or density generators from `perf_hotpaths.rs` / `perf_parallelism.rs`. Cost classes: **fast** <10ms → `sample_size(50-100)`, `measurement_time(10s)`; **medium** 10–100ms → `sample_size(20-30)`, `measurement_time(20-30s)`; **slow** >100ms → `sample_size(10-15)`, `measurement_time(30-60s)` (matches the tuning in the template `[VERIFIED: benches/perf_hotpaths.rs:35-37, benches/perf_parallelism.rs:57-59]`).

| # | Module | Public fn to bench | Verbatim signature (module path) | Input construction (NO RNG) | Size | Feature | Cost class | sample_size / measurement_time | Already benched? |
|---|--------|--------------------|----------------------------------|-----------------------------|------|---------|-----------|-------------------------------|------------------|
| 1 | `fts` | `ftsm` | `pub fn ftsm(data: &FdMatrix, ncomp: usize, argvals: &[f64]) -> Result<FtsmResult, FdarError>` — `fdars_core::fts::ftsm` `[VERIFIED: src/fts/forecast.rs:267]` | `generate_curves(n,m)` sinusoid (template line 21); `ncomp=3` | n=200, m=50 | none | medium (runs `fdata_to_pc_1d` + per-comp AR fit) | 20 / 30s | **No** — dpca (spectral) is benched; ftsm (forecast) is a distinct entry |
| 1b | `fts` (optional 2nd cell) | `fplsr` | `pub fn fplsr(data: &FdMatrix, ncomp: usize, argvals: &[f64]) -> Result<FplsrResult, FdarError>` — `fdars_core::fts::fplsr` `[VERIFIED: src/fts/forecast.rs:554]` | same `generate_curves(200,50)`; `ncomp=3` | n=200, m=50 | none | medium | 20 / 30s | No |
| 2 | `frechet` | `frechet_global_reg` | `pub fn frechet_global_reg(predictors: &FdMatrix, responses: &FdMatrix, argvals: &[f64], xout: &FdMatrix) -> Result<FrechetGlobalRegResult, FdarError>` — `fdars_core::frechet::frechet_global_reg` `[VERIFIED: src/frechet/regression.rs:236-241]` | responses = `two_group_densities(n_per,m).0` (density rows, template line 29); predictors = a 1-col Euclidean `FdMatrix` of a deterministic scalar per row (e.g. `(i as f64/n)`); `xout` = a small deterministic 1-col `FdMatrix` of query points. Validation: `responses.nrows()==predictors.nrows()`, `argvals.len()==responses.ncols()`, `xout.ncols()==predictors.ncols()` `[VERIFIED: src/frechet/regression.rs:171-213]` | n=24, m=81, xout rows=5 | none | medium (signed_quantile_average per xout row over n_q≥101) | 20 / 30s | **No** — frechet_anova is benched; regression is a distinct entry. Prefer the CONCRETE `frechet_global_reg` over the generic `*_space<S: MetricSpace>` (no `Sync`/object-construction complication) |
| 3 | `boosting_regression` | `boost_fosr` | `pub fn boost_fosr(data: &FdMatrix, predictors: &FdMatrix, argvals: &[f64], config: &BoostingConfig) -> Result<BoostFosrResult, FdarError>` — `fdars_core::boosting_regression::boost_fosr` `[VERIFIED: src/boosting_regression/boost_fosr.rs:263-268]` | `data = generate_curves(n,m).0` (functional response); `predictors` = deterministic Euclidean `FdMatrix` (n × p, e.g. p=2 sinusoid-of-index columns); `argvals` from generator; `config = BoostingConfig::default()` `[VERIFIED: src/boosting_regression/mod.rs:67-83]` — `mstop=100, nu=0.1, nbasis=10, order=4, lfd_order=2, lambda=1.0, ncomp_x=3, seed=0`. Validation: `n>=3, m>0, predictors.nrows()==n, argvals.len()==m, p>=1` `[VERIFIED: src/boosting_regression/boost_fosr.rs:273-292]` | n=100, m=50, p=2 | none | slow (mstop=100 boosting iterations w/ penalized B-spline base-learners) | 10 / 60s | No coverage exists. Consider `mstop=50` cell to keep bench time bounded |
| 4 | `coclustering` | `co_cluster_select` | `pub fn co_cluster_select(data: &FdMatrix, argvals: &[f64], k_range: &[usize], l_range: &[usize], config: &CoClusterConfig) -> Result<CoClusterSelectResult, FdarError>` — `fdars_core::coclustering::co_cluster_select` `[VERIFIED: src/coclustering.rs:1119-1125]` | `co_cluster_curves(n,m)` (template line 80, deterministic two-latent-group sinusoid); `k_range=&[2,3]`, `l_range=&[2,3]`; `config = CoClusterConfig::default()` with small `n_init` (e.g. 3) + `max_iter` (e.g. 20) to bound the grid×restart cost | n=120, m=40 | none | slow (sweeps K×L grid, each calls internally-parallel `co_cluster`) | 10 / 60s | **No** — `co_cluster` (single fit) is benched in perf_parallelism; `co_cluster_select` (model-selection sweep) is a distinct entry. Keep the grid TINY (2×2) so the cell is not minutes-long |
| 5 | `fem_smoothing` | `fem_smooth_gcv` | `pub fn fem_smooth_gcv(nodes: &[[f64;2]], triangles: &[[usize;3]], obs_xy: &[[f64;2]], y: &[f64], log_lambda_range: (f64,f64), n_grid: usize) -> Result<FemSmoothResult, FdarError>` — `fdars_core::fem_smoothing::fem_smooth_gcv` `[VERIFIED: src/fem_smoothing.rs:641-648]` | `grid_mesh(k)` (template line 73) at a SMALLER mesh than the 24 already benched — use `k=16` (256 nodes) so the GCV grid-sweep stays bounded; `y` = deterministic `sin(2πx)·cos(πy)` (template line 98); `log_lambda_range=(-4.0,0.0)`, `n_grid=5` | k=16 (256 nodes), n_grid=5 | none | slow (n_grid × O(N³) FEM solves) | 10 / 60s | **No** — `fem_smooth` (single λ) benched at 576 nodes; `fem_smooth_gcv` (λ-grid) is distinct. Use SMALLER mesh (256) because GCV multiplies the O(N³) cost by n_grid |
| 6 | `density_fda` | `lqd_fpca` | `pub fn lqd_fpca(density_matrix: &FdMatrix, argvals: &[f64], ncomp: usize, n_quantile_pts: Option<usize>) -> Result<LqdFpcaResult, FdarError>` — `fdars_core::density_fda::lqd_fpca` `[VERIFIED: src/density_fda.rs:563-568]` | `two_group_densities(n_per,m).0` (strictly-positive density rows, template line 29) + its `argvals` (range −3..3); `ncomp=3`, `n_quantile_pts=None` | n=100, m=81 | none | medium (LQD transform per row + FPCA) | 20 / 30s | No coverage. Optional 2nd cell: `wasserstein_barycenter(density_matrix, argvals, None)` `[VERIFIED: src/density_fda.rs:407-411]` — cheap (fast) |
| 7 | `inference` | `t_perm_test` **(TRACER)** | `pub fn t_perm_test(data_a: &FdMatrix, data_b: &FdMatrix, argvals: &[f64], n_perm: usize, seed: u64) -> Result<TestResult, FdarError>` — `fdars_core::inference::t_perm_test` `[VERIFIED: src/inference/permutation.rs:152-158]` | `generate_curves(n,m)` twice with different sizes for `data_a`/`data_b` (share `argvals`); `n_perm = DEFAULT_N_PERM (999)` `[VERIFIED: src/inference/permutation.rs:18]`; `seed=42` | n_a=n_b=30, m=50, n_perm=999 | none | medium (999 permutations × integrated-L2 mean-diff) | 20 / 30s | No coverage. Sibling cell: `f_perm_test` (same signature `[VERIFIED: src/inference/permutation.rs:216-222]`) and/or `oneway_anova_vstat(data, groups, argvals)` `[VERIFIED: src/inference/anova.rs:71-74]` (fast, no permutation loop) |
| 8 | `fpca_variants` | `fpca_der` | `pub fn fpca_der(data: &FdMatrix, ncomp: usize, argvals: &[f64], nderiv: usize) -> Result<FpcaResult, FdarError>` — `fdars_core::fpca_variants::fpca_der` `[VERIFIED: src/fpca_variants.rs:189-194]` | `generate_curves(n,m)`; `ncomp=5`, `nderiv=1` | n=200, m=50 | none | medium (derivative + SVD) | 20 / 30s | **No** — `fsvd`/`ssvd` were alloc-profiled in Phase 47 dhat but NOT wall-time benched in perf_hotpaths. `fpca_der` has zero coverage. Optional 2nd cell: `fsvd(x, argvals_x, y, argvals_y, ncomp)` `[VERIFIED: src/fpca_variants.rs:405-411]` with two `generate_curves` sets (cross-covariance SVD) |
| 9 | `face` | `mface_covariance` | `pub fn mface_covariance(variables: &[IrregFdata], grids: &[Vec<f64>], bandwidth: f64) -> Result<MfaceCovResult, FdarError>` — `fdars_core::irreg_fdata::mface_covariance` `[VERIFIED: src/irreg_fdata/face.rs:263-267]` | Build 2 `IrregFdata` via `IrregFdata::from_lists(&argvals_list,&values_list)` (same construction as `bench_face_covariance` template lines 52-65); `grids` = per-variable regular grids; `bandwidth=0.3`. Requires `variables.len()>=2` and `grids.len()==variables.len()` `[VERIFIED: src/irreg_fdata/face.rs:268-283]` | 2 vars, n=100, m=30 each | none | medium/slow (multivariate FACE covariance) | 15 / 30s | **No** — `face_covariance` (univariate) benched in perf_hotpaths; `mface_covariance` (multivariate) is distinct. Alt: `face_trajectory(data, config)` — `pub fn face_trajectory(data: &IrregFdata, config: &PaceFpcaConfig) -> Result<PaceFpcaResult, FdarError>` `[VERIFIED: src/irreg_fdata/face.rs:407-410]`, delegates to `pace_fpca`; `PaceFpcaConfig::default()` gives `ncomp=3, bandwidth=0.1, sigma2=0.01, work_grid=51-pt uniform, alpha=0.05` `[VERIFIED: src/pace_fpca.rs:72-83]` |

### Modules flagged (only-meaningful-entry-already-covered) — none blocking

- **No target module is fully blocked.** Every one of the 9 has at least one representative public entry NOT already in perf_hotpaths/perf_parallelism (confirmed via the per-module fn lists read this session). The only modules whose *headline* fn is already benched (`fts::dpca`, `frechet::frechet_anova`, `fem_smoothing::fem_smooth`, `irreg_fdata::face_covariance`, `coclustering::co_cluster`) each have a clean sibling entry (`ftsm`, `frechet_global_reg`, `fem_smooth_gcv`, `mface_covariance`, `co_cluster_select` respectively) — captured in the table.

### Feature-gating finding (Claude's Discretion resolved)

`[VERIFIED: grep 'cfg(feature = "linalg")' across fts/, frechet/, boosting_regression/, coclustering.rs, fem_smoothing.rs, density_fda.rs, inference/, fpca_variants/, irreg_fdata/ returned NO matches]` and `[VERIFIED: src/lib.rs:79-104 — all 9 modules declared as plain `pub mod`, none behind cfg]`. **None of the 9 target functions is `linalg`-gated.** No bench cell needs a `#[cfg(feature = "linalg")]` guard; every new bench compiles under both `--no-default-features` and `--features linalg,parallel`. Match the template: `perf_hotpaths.rs` uses no cfg guards `[VERIFIED: benches/perf_hotpaths.rs — no cfg attributes present]`, even though it is *run* under `--features linalg,parallel`.

## BENCH-02 Consolidation Plan

### Registration confirmation (already PERMANENT)
`[VERIFIED: Cargo.toml:98-104]`:
```
[[bench]]
name = "perf_hotpaths"   # PERMANENT — Phase 47 PERF proof / Phase 51 BENCH-02 regression guard
harness = false
[[bench]]
name = "perf_parallelism"   # PERMANENT — Phase 48 PERF-03 thread-scaling proof / Phase 51 BENCH-02 regression guard
harness = false
```
Both remain — BENCH-02 requires **no Cargo.toml change** for these two. The 9 new BENCH-01 entries are the only additions.

### Deterministic alloc guards confirmed present
`[VERIFIED: ls tests/alloc_audit_dpca.rs (3640 B) + tests/alloc_audit_fpca.rs (6087 B) both exist]`. These stay as-is (locked decision) — feature `dhat-heap`, hard-assert allocation-block counts.

### Numbers to consolidate into BENCH-RESULTS.md

From **PERF-RESULTS.md (Phase 47)** `[VERIFIED: .planning/phases/47-hot-path-allocation-performance/PERF-RESULTS.md:30-78]`:

| Win | Cell | Before | After | Δ | Guard bench cell |
|-----|------|--------|-------|---|------------------|
| OPT-A dpca allocations | `fts::dpca` n200_m50 | 17,739 blocks / 42,084,568 B / 8,637,712 peak | 8,139 blocks / 33,782,168 B / 8,315,984 peak | **−54% blocks** (−20% bytes) | `alloc_audit_dpca` + `perf_hotpaths::perf_dpca/n200_m50` |
| OPT-E face_covariance wall-time | `perf_face_covariance/n200_m30` | 983.8 ms | 189.8 ms [167.8, 217.3] | **−80.7%** | `perf_hotpaths::perf_face_covariance/n200_m30` |
| OPT-B fsvd | `fpca_variants::fsvd` | 275 blocks | 274 blocks | −1 block + m×m copy | `alloc_audit_fpca` |
| OPT-C ssvd | `fpca_variants::ssvd` | 22 blocks | 21 blocks | −1 block + m×m copy | `alloc_audit_fpca` |
| OPT-D functional_acf | `fts::functional_acf` | staging Vec + ~m² sqrt | from_fn + m sqrt | −1 block + ~(m²−m) sqrt dropped | (covered by dpca alloc probe path) |
| OPT-F fem_smooth | `perf_fem_smooth/nodes576` | 452.3 ms | ≈ unchanged (N×N clone dropped; O(N³) solve DEFERRED) | alloc win only | `perf_hotpaths::perf_fem_smooth/nodes576` |

From **PERF-PARALLEL-RESULTS.md (Phase 48)** `[VERIFIED: .planning/phases/48-parallelism-gap-closure/PERF-PARALLEL-RESULTS.md:25-48]`:

| Win | Cell | 1-thread | 20-thread | Speedup | Guard bench cell |
|-----|------|----------|-----------|---------|------------------|
| frechet_anova parallel | `perf_parallelism_frechet_anova/n24_m81_nperm999` | 322.73 ms | 32.57 ms | **9.9×** (change −89.8%, p<0.05) | `perf_parallelism::bench_frechet_anova` |
| co_cluster parallel | `perf_parallelism_co_cluster/n200_m50_ninit8` | 337.34 ms | 52.91 ms | **6.4×** (change −84.0%, p<0.05) | `perf_parallelism::bench_co_cluster` |

### Environment/governor caveat (carry verbatim into BENCH-RESULTS.md)
`[VERIFIED: PERF-RESULTS.md:9-19, PERF-PARALLEL-RESULTS.md:8-23]`:
- CPU governor **`powersave`** (unpinned; `cpupower` pin needs sudo — unavailable) → all **wall-time** cells are **LOW-CONFIDENCE**; treat exact medians/ratios as indicative, not authoritative. Thread-scaling *direction* (N-thread < 1-thread) is a valid signal.
- Logical cores: 20 (`RAYON_NUM_THREADS` default = 20). Feature flags: `linalg,parallel` (criterion), `dhat-heap,linalg` (alloc). Harness: criterion 0.5, dhat 0.3. Host tmp: `TMPDIR=/home/simonm/.cache/fdars-bench-tmp`. Date: 2026-08-31.
- The **allocation** wins (OPT-A..D) are governor-independent (dhat block/byte counts) → those are the HIGH-confidence guards; wall-time is secondary.
- Re-run under a `performance` governor to supersede LOW-CONFIDENCE absolute medians when a pinned environment is available.

## Architecture Patterns

### Bench file template (mirror `benches/perf_hotpaths.rs`)
**What:** Each new `benches/<module>_benchmarks.rs` follows the exact structure of the existing benches.
**When to use:** All 9 BENCH-01 files.
**Example:**
```rust
// Source: benches/perf_hotpaths.rs (VERIFIED template, this repo)
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::f64::consts::PI;
use std::time::Duration;
use fdars_core::inference::{t_perm_test, DEFAULT_N_PERM};
use fdars_core::matrix::FdMatrix;

// Deterministic sinusoid generator — copied verbatim from perf_hotpaths.rs (NO RNG).
fn generate_curves(n: usize, m: usize) -> (FdMatrix, Vec<f64>) {
    let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
    let mut data = vec![0.0; n * m];
    for i in 0..n {
        let phase = 0.2 * ((i as f64 * 3.7 + 0.5).sin());
        let amp = 1.0 + 0.3 * ((i as f64 * 5.1 + 0.3).sin());
        for j in 0..m {
            data[i + j * n] = amp * (2.0 * PI * (argvals[j] + phase)).sin();
        }
    }
    (FdMatrix::from_column_major(data, n, m).unwrap(), argvals)
}

fn bench_t_perm_test(c: &mut Criterion) {
    let mut group = c.benchmark_group("inference_t_perm_test");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(30));
    group.warm_up_time(Duration::from_secs(3));
    let (a, argvals) = generate_curves(30, 50);
    let (b_data, _) = generate_curves(30, 50); // second sample; shares argvals
    group.bench_function("na30_nb30_m50_nperm999", |b| {
        b.iter(|| {
            black_box(
                t_perm_test(black_box(&a), black_box(&b_data), black_box(&argvals),
                            black_box(DEFAULT_N_PERM), 42).unwrap(),
            )
        })
    });
    group.finish();
}

criterion_group!(benches, bench_t_perm_test);
criterion_main!(benches);
```
Then in `Cargo.toml`:
```toml
[[bench]]
name = "inference_benchmarks"
harness = false
```

### Anti-Patterns to Avoid
- **RNG inside the generator or `b.iter()`** — non-deterministic data breaks reproducibility and criterion baselines. Every generator here is a closed-form sinusoid/gaussian (NO `rand`). `[VERIFIED: benches/perf_hotpaths.rs + perf_parallelism.rs contain no `rand` import]`.
- **Building data inside `b.iter()`** — inflates the measured time with allocation. Build once outside; `black_box(&data)` inside.
- **Duplicating an already-benched cell** — `fts::dpca`, `frechet_anova`, `fem_smooth`, `face_covariance`, `co_cluster` are already covered; picking those exact fns wastes bench time and confuses the ledger. Use the sibling entries in the table.
- **`-p fdars-core -D warnings` without `--all-targets`** — false-greens because it skips bench/test code. Use `cargo clippy --all-targets --features linalg,parallel -- -D warnings` (project MEMORY: `ci-clippy-all-targets-gate`).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Deterministic curve data | A new RNG-seeded generator | Copy `generate_curves`/`co_cluster_curves`/`two_group_densities` from the existing benches | Already proven deterministic + equivalence-tested; avoids RNG in bench data |
| Statistical timing / outlier rejection | Manual `Instant::now()` loops | criterion `group.bench_function` | criterion handles warmup, sampling, outlier detection, HTML report |
| Regression detection | Custom wall-time asserts | criterion baseline compare + BENCH-RESULTS.md documented numbers | Wall-time asserts flake on `powersave` (locked out of scope) |

## Runtime State Inventory

> This is a benchmark-addition phase (new files + Cargo.toml `[[bench]]` blocks + a markdown ledger). It is NOT a rename/refactor/migration. No stored data, live-service config, OS-registered state, secrets, or build artifacts carry any renamed string. **None — verified: no `src/` symbol is renamed and no existing identifier changes.** Section included for completeness only.

## Common Pitfalls

### Pitfall 1: /tmp tmpfs exhaustion during bench link
**What goes wrong:** `cargo bench`/doctest link into a small `/tmp` tmpfs → bogus "No space left on device".
**Why it happens:** Release-mode bench binaries + doctests link temp objects into `/tmp`.
**How to avoid:** Prefix every cargo bench/build with `TMPDIR=/home/simonm/.cache/fdars-bench-tmp` (project MEMORY: `tmp-exhaustion-blocks-precommit`; CONTEXT line 102).
**Warning signs:** "No space left" while `/home` has space.

### Pitfall 2: target/ fills /home partition
**What goes wrong:** `target/` grows to 100+GB; build/link dies with "linking with cc failed" (NOT a code bug).
**How to avoid:** `rm -rf target/debug/{incremental,examples}` frees ~108G (project MEMORY: `target-dir-fills-home-partition`; CONTEXT line 104).
**Warning signs:** "linking with cc failed" mid-build.

### Pitfall 3: clippy false-green
**What goes wrong:** A plain `-p fdars-core -D warnings` skips bench/test code → the new bench's clippy warnings pass silently, CI fails later.
**How to avoid:** Gate with `cargo clippy --all-targets --features linalg,parallel -- -D warnings` (project MEMORY: `ci-clippy-all-targets-gate`; CONTEXT line 19).

### Pitfall 4: --no-verify leaves fmt drift
**What goes wrong:** `git commit --no-verify` (needed to dodge the slow hook) also skips `cargo fmt` → CI fmt-check fails despite green clippy.
**How to avoid:** Run `cargo fmt` per commit + a whole-crate sweep at phase end (project MEMORY: `noverify-commits-leave-fmt-drift`; CONTEXT line 72).

### Pitfall 5: bench compile time (release) is slow
**What goes wrong:** Each `[[bench]]` is a release build; 9 new files = long compile. Running full benches to capture baselines is minutes-to-hours.
**How to avoid:** For the "optionally run each bench once" step, use a short sample (`--sample-size 10 -- --warm-up-time 1 --measurement-time 5`) or `--quick`, and keep the slow cells (boost_fosr, co_cluster_select, fem_smooth_gcv) at bounded sizes (per the table). Verify build with `cargo build -p fdars-core --benches --features linalg,parallel` before running.

### Pitfall 6: no git tag on this milestone
**What goes wrong:** `release.yml` publishes to crates.io on any `v*` tag; this is a measurement-only milestone (crate unchanged) — a phantom version would publish.
**How to avoid:** Do NOT push a `v*` tag for this audit milestone (project MEMORY: `audit-milestone-no-git-tag`).

## Code Examples

Covered inline under **Architecture Patterns → Bench file template** (verified against `benches/perf_hotpaths.rs`). Each of the 9 modules follows that shape, swapping the generator + `bench_function` body per the target table.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Ad-hoc `Instant` timing | criterion 0.5 grouped benches | pre-existing | Statistical, HTML reports, baseline compare |
| Wall-time regression asserts | Documented soft baselines + deterministic alloc hard-asserts | this milestone (locked) | No flaky governor-sensitive CI failures |

**Deprecated/outdated:** none relevant — no library version churn in this phase.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Cost-class estimates (fast/medium/slow) for the new cells are inferred from algorithmic structure (permutation count, mstop, O(N³) solves, grid sweeps), not measured this session. | Per-Module table | LOW — cost class only sets `sample_size`/`measurement_time`; a mis-estimate makes a cell slower/faster to run, not wrong. Planner should treat sizes as starting points and the executor may down-size any cell that runs too long. |
| A2 | The exact bench-time budget of the "slow" cells (boost_fosr mstop=100, co_cluster_select 2×2 grid, fem_smooth_gcv 256-node n_grid=5) fits a reasonable one-shot baseline capture. | Per-Module table | LOW — if a cell is minutes-long, reduce mstop/mesh/grid; the fn + input-construction stay valid. |

**All function signatures, module paths, config defaults, and before/after numbers are `[VERIFIED]` (read from source/ledgers this session). The only `[ASSUMED]` items are the two cost/time estimates above — both are executor-tunable knobs, not correctness claims.**

## Open Questions

1. **Should `frechet_global_reg` predictors be 1-column or multi-column?**
   - What we know: `validate_reg_input` requires `xout.ncols()==predictors.ncols()` and `responses.nrows()==predictors.nrows()` `[VERIFIED: src/frechet/regression.rs:206-212]`; a 1-column Euclidean predictor is the simplest valid shape.
   - What's unclear: whether a multi-covariate predictor better represents the hotspot.
   - Recommendation: Use 1-column predictor (simplest deterministic construction); the signed-quantile-average inner loop over `n_q≥101` per xout row dominates cost regardless of p.

2. **Does `boost_fosr` `mstop=100` make the cell too slow for a single-shot baseline?**
   - What we know: default `mstop=100` `[VERIFIED: src/boosting_regression/mod.rs:73]`; each iteration fits penalized B-spline base-learners.
   - Recommendation: Bench at default `mstop=100` for representativeness, but the executor may add a `mstop=50` variant if the default cell exceeds a few seconds/iter. Keep `n=100, m=50`.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust toolchain | build/bench | ✓ (assumed — project builds) | 1.97 dev / MSRV 1.81 | — |
| criterion | benches | ✓ (dev-dep) | 0.5 `[VERIFIED: Cargo.toml:53]` | — |
| dhat | alloc tests (untouched) | ✓ (dev-dep) | 0.3 `[VERIFIED: Cargo.toml:54]` | — |
| `linalg` feature deps (faer, anofox-regression) | bench run under `--features linalg,parallel` | ✓ (optional deps declared) `[VERIFIED: Cargo.toml:41-43]` | needs Rust 1.84+ | build without `linalg` — none of the 9 target fns need it |
| `TMPDIR=/home/simonm/.cache/fdars-bench-tmp` | bench/doctest link | must be set per-invocation | — | free `/tmp` / `rm -rf target/debug/{incremental,examples}` |

**Missing dependencies with no fallback:** none. **Missing with fallback:** none — all bench deps are pre-existing.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in test harness (`#[test]`) + criterion 0.5 (benches, `harness = false`) |
| Config file | none (Cargo `[[bench]]` blocks) |
| Quick run command | `cargo build -p fdars-core --benches --features linalg,parallel` (compile-check all bench entries) |
| Full suite command | `cargo clippy --all-targets --features linalg,parallel -- -D warnings` (primary gate — lints bench code) |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| BENCH-01 | 9 new bench files compile + register | smoke (compile) | `TMPDIR=… cargo build -p fdars-core --benches --features linalg,parallel` | ❌ Wave 0 (files to be created) |
| BENCH-01 | bench code passes lint gate | lint | `cargo clippy --all-targets --features linalg,parallel -- -D warnings` | ✅ existing gate |
| BENCH-01 | each new bench executes (optional baseline capture) | smoke (run once) | `TMPDIR=… cargo bench -p fdars-core --features linalg,parallel --bench <name> -- --quick` | ❌ Wave 0 |
| BENCH-02 | perf benches stay registered PERMANENT | smoke | `cargo build -p fdars-core --benches --features linalg,parallel` (both compile) | ✅ `[VERIFIED: Cargo.toml:98-104]` |
| BENCH-02 | alloc guards intact | integration | `cargo test -p fdars-core --features dhat-heap,linalg --test alloc_audit_dpca --test alloc_audit_fpca` | ✅ `[VERIFIED: tests/alloc_audit_*.rs exist]` |
| BENCH-02 | BENCH-RESULTS.md consolidation | manual (doc review) | n/a — markdown ledger | ❌ Wave 0 (to be created) |

### Sampling Rate
- **Per task commit:** `cargo build -p fdars-core --benches --features linalg,parallel` + `cargo clippy --all-targets --features linalg,parallel -- -D warnings` + `cargo fmt`.
- **Per wave merge:** full clippy gate green.
- **Phase gate:** all 9 benches compile + lint clean; both PERF benches still registered; alloc tests still pass; BENCH-RESULTS.md landed.

### Wave 0 Gaps
- [ ] `benches/fts_benchmarks.rs` — covers BENCH-01 (fts::ftsm)
- [ ] `benches/frechet_benchmarks.rs` — BENCH-01 (frechet::frechet_global_reg)
- [ ] `benches/boosting_regression_benchmarks.rs` — BENCH-01 (boost_fosr)
- [ ] `benches/coclustering_benchmarks.rs` — BENCH-01 (co_cluster_select)
- [ ] `benches/fem_smoothing_benchmarks.rs` — BENCH-01 (fem_smooth_gcv)
- [ ] `benches/density_fda_benchmarks.rs` — BENCH-01 (lqd_fpca)
- [ ] `benches/inference_benchmarks.rs` — BENCH-01 (t_perm_test) **← tracer**
- [ ] `benches/fpca_variants_benchmarks.rs` — BENCH-01 (fpca_der)
- [ ] `benches/face_benchmarks.rs` — BENCH-01 (mface_covariance) *(name to match module; module path is `irreg_fdata::mface_covariance`)*
- [ ] 9 `[[bench]] name=… harness=false` entries in `fdars-core/Cargo.toml`
- [ ] `.planning/phases/51-…/BENCH-RESULTS.md` — consolidated ledger
- [ ] Framework install: none — criterion pre-installed

*(Naming note: the `face` module's public fn lives at `fdars_core::irreg_fdata::` — the bench filename can be `face_benchmarks.rs` or `irreg_face_benchmarks.rs` at planner discretion; the import path is `use fdars_core::irreg_fdata::{mface_covariance, IrregFdata};`.)*

## Security Domain

> `security_enforcement` is enabled in config. This is a measurement-only, dev-tooling phase: **no external/untrusted input, no network, no auth/session/crypto, no new dependency, no `src/` behavior change.** The attack surface is nil.

### Applicable ASVS Categories
| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — (no auth in a bench) |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | no | Bench inputs are hard-coded deterministic generators (no external input) |
| V6 Cryptography | no | — (no crypto; RNG explicitly forbidden in generators) |
| V14 Config / Dependencies | yes | No new dependency; supply-chain surface unchanged (locked constraint). `--features dhat-heap` never enabled in release/default CI `[VERIFIED: Cargo.toml:30-32 comment]` |

### Known Threat Patterns for a bench-only phase
| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Accidentally enabling `dhat-heap` `#[global_allocator]` in release | Tampering (perf/behavior) | Feature stays dev/test-only; never in default CI (existing comment `[VERIFIED: Cargo.toml:30-32]`) |
| Phantom crates.io publish from a `v*` tag on a no-code milestone | (supply-chain) | Do NOT push a `v*` tag (MEMORY `audit-milestone-no-git-tag`) |

## Sources

### Primary (HIGH confidence — read this session)
- `fdars-core/benches/perf_hotpaths.rs` — template + generate_curves + grid_mesh + already-benched cells (dpca, face_covariance, fem_smooth)
- `fdars-core/benches/perf_parallelism.rs` — two_group_densities + co_cluster_curves + already-benched (frechet_anova, co_cluster)
- `fdars-core/benches/regression_benchmarks.rs` — module-bench naming/structure convention
- `fdars-core/Cargo.toml` — `[[bench]]` registration, dev-deps, features
- `src/fts/{mod,forecast}.rs`, `src/frechet/{mod,regression,mean}.rs`, `src/boosting_regression/{mod,boost_fosr,boost_fofr,gamlss}.rs`, `src/coclustering.rs`, `src/fem_smoothing.rs`, `src/density_fda.rs`, `src/inference/{mod,permutation,anova}.rs`, `src/fpca_variants.rs`, `src/irreg_fdata/{mod,face}.rs`, `src/pace_fpca.rs`, `src/lib.rs` — public signatures, config defaults, module paths, feature-gating
- `.planning/phases/47-hot-path-allocation-performance/PERF-RESULTS.md` + `.planning/phases/48-parallelism-gap-closure/PERF-PARALLEL-RESULTS.md` — BENCH-02 before/after numbers + governor caveat

### Secondary (MEDIUM confidence)
- Project MEMORY.md pointers (tmp exhaustion, target/ growth, clippy --all-targets, --no-verify fmt drift, audit-milestone-no-tag) — operational pitfalls

### Tertiary (LOW confidence)
- none

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — pre-existing dev-deps, no new packages, verified in Cargo.toml.
- Architecture (bench template + per-module targets): HIGH — every signature/path/config default read verbatim from source this session.
- Pitfalls: HIGH — sourced from project MEMORY + CONTEXT (documented incidents), not speculation.
- Cost/time estimates: LOW — inferred, executor-tunable (see Assumptions A1/A2).

**Tracer recommendation:** `inference` (`benches/inference_benchmarks.rs`, `t_perm_test`). Cheapest self-contained cell (two sinusoid `FdMatrix` from the existing generator, no feature gate, no generic bound, no multi-input assembly). Proves the full add-bench → Cargo.toml register → `cargo build --benches` → `clippy --all-targets` → `cargo fmt` → commit `--no-verify` pipeline before fanning out the other 8 modules.

**Research date:** 2026-09-01
**Valid until:** 2026-10-01 (stable — no fast-moving deps; signatures pinned to current source)
