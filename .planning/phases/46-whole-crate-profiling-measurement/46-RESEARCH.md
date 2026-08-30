# Phase 46: Whole-Crate Profiling & Measurement — Research

**Researched:** 2026-08-30
**Domain:** Rust criterion benchmarking, dhat allocation profiling, codebase static analysis
**Confidence:** HIGH (all findings verified against source files this session)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **Profiling methodology:** Reuse the existing harness (`benches/audit_hotpaths.rs` + the 10
  module benches); add **throwaway** probe benches for the 9 currently-unbenchmarked subsystems.
  Permanent `[[bench]]` registration is explicitly Phase 51 (BENCH-01) — do not register them here.
- **Scaling grid:** 2–3 point N×M grid (e.g. n ∈ {50, 200, 1000} curves × m ∈ {50, 200} eval
  points) to expose scaling behavior, not a single mid-size point.
- **Allocation profiling:** Feature-gated `dhat-heap` on the **top candidate** workloads per
  subsystem (not exhaustive) — enough to surface `FdMatrix`↔`DMatrix` copies and per-iteration
  allocs.
- **Environment recording:** CPU governor and `RAYON_NUM_THREADS` must be recorded in the report
  (v0.14.0 audit governor/CPU-pinning caveat).
- **Artifact location:** Three separate ranked inventory docs under
  `.planning/phases/46-whole-crate-profiling-measurement/` (PROF-01/02/03). No committed `src/`
  changes.
- **Item format:** Every inventory item carries a real criterion/allocation number and a `file:line`
  source anchor.
- **Depth bound:** Prioritize the 9 named reuse-first v0.19–v0.29 subsystems (`inference`, `fts`,
  `frechet`, `density_fda`, `fpca_variants`, `face`, `boosting_regression`, `fem_smoothing`,
  `coclustering`); time-box exhaustiveness to "enough to drive Phases 47–50 concretely."
- **Ranking criteria:**
  - Hot-path: wall-time × representativeness; allocation count as secondary signal.
  - Dedup-leverage: (# call sites × complexity/drift-risk) with `file:line` anchors.
  - API-inconsistency: user-facing impact + breadth; proposed canonical form per item.

### Claude's Discretion

_(none specified in CONTEXT.md)_

### Deferred Ideas (OUT OF SCOPE)

- Permanent `[[bench]]` registration for the 9 unbenchmarked modules → Phase 51 (BENCH-01).
- Committing PERF-proof benchmarks as regression guards → Phase 51 (BENCH-02).
- Actually optimizing / dedup'ing / unifying anything surfaced here → Phases 47–50.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| PROF-01 | Whole-crate criterion + allocation profiling pass produces a ranked hot-path optimization target list (N×M-scaled, v0.19–v0.29 subsystems prioritized) | §Profiling Harness Pattern, §9 Subsystems Inventory, §Allocation Hotspots |
| PROF-02 | Duplication/consolidation inventory — machinery repeated across modules with `file:line` anchors, ranked by dedup leverage | §Duplication Candidates |
| PROF-03 | API-inconsistency inventory — config/result patterns and redundant public functions, with proposed canonical form per item | §API Inconsistency Candidates |
</phase_requirements>

---

## Summary

Phase 46 is a pure measurement phase: write throwaway probe benches for the 9 unbenchmarked
subsystems, run the existing + probe harnesses, capture dhat allocation profiles for the highest-
allocation workloads, then produce three ranked inventory documents. Zero changes to `fdars-core/src/`.

The existing benchmarking infrastructure is mature and fully reusable. `benches/audit_hotpaths.rs`
(1,208 lines, `harness = false`) demonstrates every pattern needed — N×M grid cells, sentinel
sizing, `generate_curves()` helper, `generate_weighted_input()` for SVD paths, and `black_box()`
discipline. The dhat integration pattern is established in `tests/alloc_audit_fpca.rs`: a
`#[cfg(feature = "dhat-heap")]` integration test (not inline `#[cfg(test)]`) so dhat's
`#[global_allocator]` occupies its own binary. Both are directly copy-adapt templates for the 9
probe benches and the new dhat probe integration tests.

The three inventories are grounded in real sites identified by static analysis this session.
The duplication map is well-defined (simpsons_weights: 20+ call sites across 9+ files; chi2: two
independent implementations in `spm::chi_squared` and `inference::dist`; permutation loops in 5+
modules). The API-inconsistency map is likewise concrete (4 of 56 Config structs lack `Default`;
`fts`/`frechet`/`boosting_regression` result structs do not share a unifying trait; the
`_1d`/`_2d`/`_nd` suffix family has 30+ variants with no unified dispatch).

**Primary recommendation:** Write probe benches first (one per subsystem, 3-cell grid, sized to
keep each individual cell under 60 s), run them, then write the three inventory docs from the
measurements. Do not run a full `cargo bench` across all 10 registered benches and the 9 probes in
a single session — that would exhaust `/home` (`target/` growth) and the `/tmp` tmpfs. Run
registered benches and probes in named batches per MEMORY.md.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Criterion probe bench authoring | Dev harness (`benches/`) | — | All criterion benches live in `benches/`, registered via `[[bench]]` in Cargo.toml, `harness = false` |
| dhat allocation profiling | Integration tests (`tests/`) | — | dhat requires a separate process / binary; must be an integration test, not inline `#[cfg(test)]` (see `tests/alloc_audit_fpca.rs`) |
| Hot-path inventory (PROF-01) | Phase 46 report doc | Phase 47 consumer | Output doc lives under `.planning/phases/46-*/`; consumed by Phase 47 planner |
| Duplication inventory (PROF-02) | Phase 46 report doc | Phase 49 consumer | Same pattern |
| API-inconsistency inventory (PROF-03) | Phase 46 report doc | Phase 50 consumer | Same pattern |
| Parallelism-gap measurement | Phase 46 (detect gaps) | Phase 48 (fix gaps) | PERF-03 is Phase 48; Phase 46 only identifies which subsystems lack `iter_maybe_parallel!` |

---

## Profiling Harness Pattern

### Existing Bench Structure (Verified)

[VERIFIED: fdars-core/benches/audit_hotpaths.rs:1-1208]

All 10 registered benches (`seasonal_benchmarks`, `depth_benchmarks`, `classification_benchmarks`,
`alignment_benchmarks`, `regression_benchmarks`, `explain_benchmarks`, `smoothing_benchmarks`,
`basis_benchmarks`, `matrix_benchmarks`, `audit_hotpaths`) use `harness = false` in
`fdars-core/Cargo.toml` lines 58–97. The canonical bench pattern is:

```rust
// 1. Generate inputs OUTSIDE b.iter() to exclude allocation cost
let (data, argvals) = generate_curves(n, m);

// 2. Tune sample_size and measurement_time per timing tier:
//    < 1 ms/iter  → sample_size(30), measurement_time(10s)
//    1–20 ms/iter → sample_size(20), measurement_time(20s)
//    > 1 s/iter   → sample_size(10), measurement_time(60s)
group.sample_size(20);
group.measurement_time(std::time::Duration::from_secs(20));
group.warm_up_time(std::time::Duration::from_secs(5));

// 3. bench_function name encodes the cell: "n200_m50"
group.bench_function("n200_m50", |b| {
    b.iter(|| black_box(target_fn(black_box(&data), black_box(&argvals))))
});
```

The `generate_curves()` helper produces deterministic column-major `FdMatrix` (no RNG dependency)
and is **in sync** with `tests/alloc_audit_fpca.rs:generate_test_curves` — the comment in
`audit_hotpaths.rs:43` mandates this sync. Any new probe bench that also needs a dhat companion
must copy the same formula verbatim.

### Running a Single Bench (Without Full cargo bench)

```bash
# Single registered bench — no recompile of others
TMPDIR=/home/simonm/.cache/fdars-bench-tmp \
  cargo bench --bench audit_hotpaths -- audit_fpca

# For throwaway probe benches (not registered — use --manifest-path trick or temp [[bench]] entry):
# Option A: temporarily add a [[bench]] entry in Cargo.toml (remove after measurement)
# Option B: cargo bench --bench <probe_name> works once registered (Phase 51 does permanent reg)
# For Phase 46, throwaway probes need temporary registration to run via cargo bench.
TMPDIR=/home/simonm/.cache/fdars-bench-tmp \
  cargo bench --bench <probe_name> --features linalg,parallel
```

Alternatively, a standalone binary approach (avoids Cargo.toml edits):
```bash
TMPDIR=/home/simonm/.cache/fdars-bench-tmp \
  cargo build --release --features linalg,parallel -p fdars-core
# Then manually compile and run the probe as a bin with Criterion setup.
```

The simplest path for throwaway probes: add a `[[bench]]` entry to `Cargo.toml` temporarily
(clearly commented `# THROWAWAY — Phase 46 only`), run, record numbers, remove the entry. The
`harness = false` pattern requires this registration to invoke `criterion_main!`.

### N×M Scaling Grid (Locked Decision)

The CONTEXT.md locks a 3-point N grid × 2-point M grid = 6 cells. Recommended sizing based on the
O(N·M) complexity of most subsystems:

| Cell | n | m | n×m | Expected tier |
|------|---|---|-----|--------------|
| small | 50 | 50 | 2,500 | < 10 ms |
| medium-n | 200 | 50 | 10,000 | 10–100 ms |
| medium-m | 50 | 200 | 10,000 | 10–100 ms |
| large | 1,000 | 50 | 50,000 | 100 ms–2 s |
| large-m | 200 | 200 | 40,000 | 100 ms–2 s |
| cap-check | 1,000 | 200 | 200,000 | measure once; may be 30–60 s |

For subsystems with O(N²·M²) paths (elastic: `elastic_self_distance_matrix`), the existing cap
(N=100, M=50) established in `audit_hotpaths.rs` must be respected.

For `co_cluster` (N=1,000 × k_blocks × l_blocks inner loops), measure at the small cell first to
estimate whether large cells are tractable.

---

## 9 Unbenchmarked Subsystems — Inventory

[VERIFIED: fdars-core/src/lib.rs:64-138 module declarations; individual mod.rs/file reads this session]

### Module Existence and Status

| Module name | Disk path | Type | Line count | Main public entry points |
|-------------|-----------|------|-----------|--------------------------|
| `inference` | `src/inference/` (dir) | 8 files | ~3,400 total | `two_sample_mean_test`, `oneway_anova_vstat`, `flm_f_test`, `flm_gof_test`, `t_perm_test`, `f_perm_test`, `mean_scb`, `scb_two_sample_test`, `itp_flm`, `itp_one_pop`, `itp_two_pop` |
| `fts` | `src/fts/` (dir) | 4 files | ~3,400 total | `functional_acf`, `functional_pacf`, `functional_difference`, `stationarity_test`, `long_run_covariance`, `ftsm`, `ftsm_forecast`, `ftsm_forecast_multistep`, `ftsm_update`, `fplsr`, `spectral_density`, `dpca`, `dpca_reconstruct` |
| `frechet` | `src/frechet/` (dir) | 6+ files | ~2,200 total | `frechet_mean`, `frechet_variance`, `frechet_anova`, `frechet_anova_space`, `frechet_regression`, `frechet_local_regression`, `frechet_gradient_descent`, `wasserstein2_distance`, space implementations |
| `density_fda` | `src/density_fda.rs` | single file | 1,133 | `normalize_density`, `lqd_transform`, `inverse_lqd`, `wasserstein_barycenter`, `lqd_fpca` |
| `fpca_variants` | `src/fpca_variants.rs` | single file | 1,256 | `cross_covariance`, `fpca_der`, `dynamical_correlation`, `fsvd`, `ssvd` |
| `face` | `src/irreg_fdata/face.rs` | single file (in `irreg_fdata/`) | 797 | `face_covariance`, `mface_covariance`, `face_trajectory` |
| `boosting_regression` | `src/boosting_regression/` (dir) | 6 files | ~3,100 total | `boost_fosr`, `boost_fofr`, `gamlss_fosr`, `bayesian_fosr`, `stability_selection` |
| `fem_smoothing` | `src/fem_smoothing.rs` | single file | 1,147 | `assemble_fem_matrices`, `fem_basis_eval`, `fem_smooth`, `fem_smooth_gcv`, `fem_predict` |
| `coclustering` | `src/coclustering.rs` | single file | 1,756 | `co_cluster`, `co_cluster_select` |

**Note on `face`:** The module is under `src/irreg_fdata/face.rs` and exported from `lib.rs` as
`pub use irreg_fdata::{face_covariance, face_trajectory, mface_covariance, MfaceCovResult}`
[VERIFIED: fdars-core/src/lib.rs:245]. The BENCH-01/PROF-01 lists call it "face" which refers to
this `irreg_fdata::face` submodule, not a separate top-level module.

### Primary Timing Targets per Subsystem

The planner should direct the executor to profile these specific functions first in each subsystem:

**inference:** `itp_one_pop` / `itp_two_pop` (O(p×n) inner parallel loop via `iter_maybe_parallel!`
in `src/inference/itp.rs:78`); permutation tests `t_perm_test` / `f_perm_test` (sequential loop,
n_perm iterations — parallelism gap candidate).

**fts:** `stationarity_test` (bootstrap permutation, seeded RNG, ~600 permutations by default per
`acf.rs:606`); `long_run_covariance` (`DMatrix::from_column_slice` at `acf.rs:337` — allocation
hotspot); `dpca` / `spectral_density` (`DMatrix::from_column_slice` at `spectral.rs:203`).

**frechet:** `frechet_anova` / `frechet_anova_space` (permutation loop, 999 default, sequential at
`anova.rs:171`); `frechet_mean` (outer loop over objects, `.clone()` per object in test data
visible at `mean.rs:79`).

**density_fda:** `wasserstein_barycenter` (N outer iterations, quadratic inner work at `density_fda.rs:474`);
`lqd_fpca` (calls `fdata_to_pc_1d` — existing FPCA allocation profile applies).

**fpca_variants:** `fsvd` (`DMatrix::from_column_slice(g_dim, g_dim, &gram)` at `fpca_variants.rs:488`);
`ssvd` (`DMatrix::from_column_slice(m, m, &c_scaled)` at `fpca_variants.rs:740`); `dynamical_correlation`
(clones `x` and `y` at lines 316–317 before centering).

**face:** `face_covariance` (sparse covariance surface, kernel smoothing inner loop); `mface_covariance`
(block extension — potential for parallel per-variable loop).

**boosting_regression:** `boost_fosr` / `boost_fofr` (mstop boosting iterations, each calls
`cholesky_factor` at `boost_fosr.rs:413` and `boost_fofr.rs:328`; `fdata_to_pc_1d` in `bayesian_fosr`
at `bayesian.rs:167`); `stability_selection` (parallel resampling via `iter_maybe_parallel!` at
`stability.rs:135` — already parallel).

**fem_smoothing:** `fem_smooth` (Cholesky factor + `cholesky_forward_back` at `fem_smoothing.rs:558,568,574`);
`fem_smooth_gcv` (GCV search loop calling `fem_smooth` repeatedly).

**coclustering:** `co_cluster` (nested `for l in 0..l_blocks` / `for k in 0..k_blocks` loop with
sequential inner loop at lines 238–333 — high dedup-leverage and parallelism-gap candidate);
`co_cluster_select` (calls `co_cluster` multiple times).

---

## Allocation Hotspots (PROF-01 Secondary Signal)

[VERIFIED: source files read this session; line numbers below]

### Established Pattern (from prior audit)

The existing dhat probe at `tests/alloc_audit_fpca.rs:75-85` demonstrates the pattern:

```rust
#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

#[test]
#[cfg(feature = "dhat-heap")]
fn count_<workload>_allocations() {
    // Build inputs OUTSIDE profiler scope (setup, not target)
    let (data, argvals) = generate_test_curves(n, m);
    let _profiler = dhat::Profiler::builder().testing().build();
    let _ = target_fn(&data, ...);
    let stats = dhat::HeapStats::get();
    println!("Total heap blocks: {}", stats.total_blocks);
    println!("Total heap bytes:  {}", stats.total_bytes);
    println!("Peak heap bytes:   {}", stats.max_bytes);
    // Do NOT hard-assert — this is a baseline measurement
}
```

Run with: `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features dhat-heap,linalg -- <test_name> --nocapture`

### Known `FdMatrix`↔`DMatrix` Copy Sites in 9 Subsystems

[VERIFIED: source file reads this session]

| Site | File:line | Allocation size (N=200, m=50) | Priority |
|------|-----------|-------------------------------|---------|
| `long_run_covariance` | `src/fts/acf.rs:337` | m×m = 50×50 = 20 KB | MEDIUM |
| `spectral_density` / `dpca` | `src/fts/spectral.rs:203` | m×m = 20 KB | MEDIUM |
| `fsvd` gram matrix | `src/fpca_variants.rs:488` | g_dim×g_dim (up to m×m) | HIGH |
| `ssvd` covariance matrix | `src/fpca_variants.rs:740` | m×m | HIGH |
| `SpdSpace::frechet_mean` | `src/frechet/spaces/spd.rs:154` | d×d per iteration | MEDIUM |
| `fem_smooth` clone for GCV | `src/fem_smoothing.rs:541` | nbasis×nbasis | LOW |

These should each get a dhat integration test in a new `tests/alloc_audit_new_subsystems.rs` file
(same `#[cfg(feature = "dhat-heap")]` gating pattern as the existing file).

### Existing Allocation Baselines (Reference Points)

[VERIFIED: fdars-core/tests/alloc_audit_fpca.rs:66-84]

From the prior audit (Phase 4 baseline, not re-measured this session):
- `fdata_to_pc_1d` (N=500, M=200): ~3 allocations — `zeros(n,m)` 800 KB + `clone()` 800 KB + `to_dmatrix()` 800 KB → ~2.4 MB total, ~1.6 MB peak. [ASSUMED — numbers from doc comment, not re-run]
- `vert_fpca` (N=100, M=50): 1 `to_dmatrix()` at `elastic_fpca.rs:214`. [ASSUMED — not re-run]
- `joint_fpca` (N=100, M=50): 1 `to_dmatrix()` at `elastic_fpca.rs:317`. [ASSUMED — not re-run]

---

## Duplication Candidates (PROF-02)

[VERIFIED: grep results and source file reads this session]

### Category 1 — Simpson/Quadrature Weights (Highest call-site count)

`helpers::simpsons_weights` is already the canonical implementation at `src/helpers.rs` and
imported via `use crate::helpers::simpsons_weights`. It is called in **20+ sites** across 9+ files:

[VERIFIED: grep this session]

| File | Lines (call sites) |
|------|--------------------|
| `src/distance.rs` | :68 |
| `src/alignment/geodesic.rs` | :114, :261 |
| `src/spm/contrib.rs` | :185 |
| `src/clustering.rs` | :91, :583, :690, :772, :848, :956 |
| `src/regression.rs` | :325, :659, :980 |
| `src/utility.rs` | :20, :39, :64 |
| `src/elastic_changepoint.rs` | :98, :148 |
| `src/fof_regression.rs` | :444 |
| `src/scoring.rs` | (uses `simpsons_weights`) |

**Assessment:** This function is already centralized — call sites import from `crate::helpers`. The
dedup concern here is not removing duplication (it's already unified) but rather ensuring the 9 new
subsystems also use this canonical function rather than local reimplementations. The 9 subsystems
DO NOT appear to have local reimplementations — none were found by grep. PROF-02 should verify
this explicitly for all 9 subsystems.

### Category 2 — Chi-Squared Survival Function (Two Independent Implementations)

[VERIFIED: fdars-core/src/spm/chi_squared.rs:164-189 and fdars-core/src/inference/dist.rs:99-132]

Two separate implementations exist, computing the same mathematical function (regularized upper
incomplete gamma):

**Implementation A — `inference::dist`:**
- `chi_square_sf(x: f64, k: usize)` at `src/inference/dist.rs:99` — `pub(crate)`, used by
  `hotelling.rs:143` (Hotelling T² p-value) and `frechet/anova.rs:167,256` (Fréchet ANOVA asymptotic p-value)
- `chi_square_sf_df(x: f64, df: f64)` at `src/inference/dist.rs:118` — float-df variant for
  Satterthwaite/Box approximations

**Implementation B — `spm::chi_squared`:**
- `chi2_cdf(x: f64, k: usize)` at `src/spm/chi_squared.rs:164` — `pub(super)`, CDF direction
- `chi2_quantile(p: f64, k: usize)` at `src/spm/chi_squared.rs:189` — quantile/inverse function
- Used at: `spm/ewma.rs:328`, `spm/control.rs:93,181`, `spm/contrib.rs:391`

These compute the same regularized incomplete gamma but differ in API direction (sf vs cdf/quantile)
and scope (spm-internal vs inference-internal). CONS-01 could consolidate into a single
`pub(crate)` module in `src/helpers.rs` or a new `src/distributions.rs` covering both SF and
quantile. **Dedup leverage: MEDIUM** — 2 implementations, ~8 call sites, moderate drift risk.

### Category 3 — Permutation-Test Loop Scaffolding (CONS-02 Target)

[VERIFIED: grep results this session]

The permutation loop pattern (shuffle labels → recompute statistic → count extremes → p-value)
appears independently in 5+ modules:

| Module | Function | File:line | Notes |
|--------|----------|-----------|-------|
| `inference::permutation` | `t_perm_test` / `f_perm_test` | `src/inference/permutation.rs:175, 238` | Sequential `for _ in 0..n_perm` |
| `frechet::anova` | `frechet_anova` | `src/frechet/anova.rs:171` | Sequential `for perm in 0..n_perm` |
| `function_on_scalar` | `fanova` | `src/function_on_scalar.rs:831,847` | Sequential `for _ in 0..n_perm` |
| `famm` | `permutation_test` | `src/famm.rs:861–899` | Private helper, sequential |
| `explain_generic::importance` | `generic_permutation_importance` | `src/explain_generic/importance.rs:22` | Parallel via `iter_maybe_parallel!` |
| `explain::importance` | `fpc_permutation_importance` | `src/explain/importance.rs:59,127` | Uses `n_perm` sequential |

Common boilerplate in each: seed a `StdRng`, shuffle indices using Fisher-Yates, recompute
statistic, count `n_ge`, return `(n_ge + 1) / (n_perm + 1)`. A `pub(crate) fn run_permutation_test`
in `src/helpers.rs` accepting a generic statistic closure would consolidate this. **Dedup leverage:
HIGH** — 6 sites, identical structure, sequential versions miss a parallelism opportunity.

### Category 4 — Per-Thread Seeded-RNG Pattern (CONS-02 Target)

[VERIFIED: grep results this session]

The pattern `StdRng::seed_from_u64(seed)` for a top-level RNG (or `seed_from_u64(seed + k as u64)`
for per-thread) appears in 8+ files:

| File | Line | Pattern |
|------|------|---------|
| `src/clustering.rs` | :584, :773 | `StdRng::seed_from_u64(seed)` |
| `src/alignment/clustering.rs` | :195 | `StdRng::seed_from_u64(config.seed)` |
| `src/fts/acf.rs` | :143, :606 | `StdRng::seed_from_u64(seed)` (bootstrap permutation) |
| `src/regression.rs` | :1535, :1575 (test code) | `StdRng::seed_from_u64(99)` / `seed_from_u64(42)` |
| `src/explain/helpers/shap_helpers.rs` | :39 | `rng: &mut StdRng` parameter |

The per-thread variant (`StdRng::seed_from_u64(seed + k as u64)`) occurs in several parallel loops
(documented in `CLAUDE.md` as the canonical pattern). A `pub(crate) fn make_thread_rng(seed: u64,
thread_idx: usize) -> StdRng` helper in `src/helpers.rs` would make this idiomatic and searchable.
**Dedup leverage: MEDIUM** — 8 sites, risk of divergence if new modules invent novel seeding
patterns.

### Category 5 — SVD Sign-Fix (`fix_svd_signs`) Duplication

[VERIFIED: fdars-core/src/regression.rs:180 and fdars-core/src/pace_fpca.rs:219]

`fix_svd_signs` is a private function in `src/regression.rs:180` used internally at lines :381
and :991. `src/pace_fpca.rs:219` implements the same sign-convention fix independently ("mirrors
`fix_svd_signs` in regression.rs" per the comment). CONS-01 should promote `fix_svd_signs` to
`pub(crate)` in `src/linalg.rs` (which already contains Cholesky and SVD-adjacent utilities) and
remove the `pace_fpca.rs` copy. **Dedup leverage: LOW** — 2 sites, but any future FPCA variant
that forgets to apply this will produce sign-flipped loadings (silent correctness bug).

### Category 6 — FPCA Scoring Paths

[VERIFIED: grep results this session]

`fdata_to_pc_1d` (canonical FPCA entry point at `src/regression.rs:287`) is correctly used as the
single FPCA engine across all 9 new subsystems — no independent FPCA re-implementation was found.
However, the **result projection** pattern (`fpca.project(&new_data)`) is repeated inline in
several places rather than using the `FpcaResult::project` method. This is a usage pattern
inconsistency, not an algorithmic duplication. PROF-02 inventory should note this as a minor item.

### Category 7 — Cholesky Solve (Already Centralized)

[VERIFIED: fdars-core/src/linalg.rs:85-137 and call sites in boosting_regression/]

`cholesky_factor`, `cholesky_forward_back`, `cholesky_solve` are `pub(crate)` in `src/linalg.rs`
and are correctly used via `use crate::linalg::{...}` in `boosting_regression/bayesian.rs:39`,
`boost_fosr.rs:33`, `boost_fofr.rs:56`, and `fem_smoothing.rs:558,568,574`. No duplicates — this
category is already well-consolidated. PROF-02 should confirm the 9 new subsystems use `linalg`
rather than rolling local Cholesky.

---

## API Inconsistency Candidates (PROF-03)

[VERIFIED: grep and source reads this session]

### Category A — Config Structs Missing `Default` Impl

[VERIFIED: grep results this session — 56 Config structs total, 52 with `impl Default for`]

52 of 56 `pub struct *Config` types have `impl Default for`. The 4 missing:

| Struct | File | Fields with sensible defaults |
|--------|------|------------------------------|
| `BoostingConfig` | `src/boosting_regression/mod.rs:44` | `mstop: 100`, `nu: 0.1`, `nbasis: 10`, `order: 4`, `lfd_order: 2` (FDboost defaults documented in field docs) |
| `BayesianConfig` | `src/boosting_regression/mod.rs:76` | `ncomp`, `n_iter`, `burnin`, `thin`, `seed` — all have FDboost/refund-aligned defaults in field docs |
| `StabilityConfig` | `src/boosting_regression/mod.rs:103` | `b_count: 100`, `cutoff: 0.6`, `subsample_frac: 0.5`, `seed: 42` (documented) |
| `StlConfig` | `src/detrend/stl.rs:49` | `ns`, `np`, `nt`, `nl`, `ni`, `no` — all standard STL parameter defaults exist in R reference |

**Proposed canonical form:** Add `impl Default for BoostingConfig`, `BayesianConfig`,
`StabilityConfig`, `StlConfig` using the documented reference-implementation defaults. Callers
using `BoostingConfig { mstop: 200, ..Default::default() }` is idiomatic Rust; forcing construction
of all fields is a usability friction point. **User-facing impact: HIGH** (all four are public API;
`boosting_regression` was added in v0.19–v0.25, one of the "new subsystems").

### Category B — `_1d` / `_2d` / `_nd` Function Families Without Unified Dispatch

[VERIFIED: grep results this session]

30+ public functions carry a dimensionality suffix. Representative families:

| Family | Functions | Files |
|--------|-----------|-------|
| `mean_1d` / `mean_2d` | `src/fdata.rs:167, 184` | Separate implementations, no common `mean()` |
| `deriv_1d` / `deriv_2d` | `src/fdata.rs:852, 934` | Separate; `deriv_2d` is a 2D tensor extension |
| `geometric_median_1d` / `geometric_median_2d` | `src/fdata.rs:1000, 1025` | Separate |
| `modal_1d` / `modal_2d` | `src/depth/modal.rs:17, 44` | Separate implementations |
| `fdata_to_pc_1d` only | `src/regression.rs:287` | No `_2d` variant (2D uses `function_on_scalar_2d.rs`) |
| `fdata_to_basis_1d` / `basis_to_fdata_1d` | `src/basis/projection.rs:160, 226` | No `_2d` |
| `dtw_self_1d` / `dtw_cross_1d` | `src/metric/dtw.rs:71, 81` | No `_2d` |
| `soft_dtw_*_1d` (4 functions) | `src/metric/soft_dtw.rs:88–134` | No `_2d` |
| `hshift_self_1d` / `hshift_cross_1d` | `src/metric/hshift.rs:45, 59` | No `_2d` |
| `deriv_self_1d` / `deriv_cross_1d` | `src/metric/deriv.rs:100, 159` | No `_2d` |

**Proposed canonical form (API-01 / API-02):** This is a large surface; Phase 50 should not unify
all at once. Prioritize the highest-user-impact families (depth, regression, fdata utilities).
A `FdaOps` trait with `mean(&self) -> Vec<f64>` etc. is overkill; the pragmatic canonical form is
a `mean()` function that dispatches on `FdMatrix` dimensionality via a `dim` parameter or an enum.
Mark the existing `_1d` / `_2d` variants `#[deprecated(since = "0.30.0", note = "Use mean() with dim parameter")]`.

**Note:** Many `_nd` variants in `src/alignment/nd.rs` serve multidimensional curve alignment
(e.g. `elastic_align_pair_nd`, `karcher_mean_nd`, `pca_nd`) which are genuinely different
algorithms — these should NOT be deprecated, only reorganized if naming is confusing.

### Category C — Result Struct Field Naming Inconsistency

[VERIFIED: reading several result struct definitions this session]

The canonical `FpcaResult` pattern (`scores`, `rotation`, `mean`, `weights`, `singular_values`)
is well-established, but newer result structs in the 9 subsystems use different field names for
equivalent concepts:

| Concept | `FpcaResult` field | Variant found |
|---------|-------------------|---------------|
| Principal components | `scores` (n×k matrix) | `FtsmResult` carries `rotation` for what would be `components` |
| Reconstructed fit | `fitted` | `BoostFosrResult` uses `fitted_values`; `FtsmResult` uses `fitted` |
| Residuals | `residuals` | `BoostFosrResult` uses `residuals` (consistent); others inconsistent |

The `FpcPredictor` trait (`src/explain_generic/mod.rs`) already provides a unification layer for
`FregreLmResult`, `FunctionalLogisticResult`, and `ClassifFit`. PROF-03 should recommend extending
`FpcPredictor` or creating a similar trait for the `fts` / `boosting_regression` result types.
**User-facing impact: MEDIUM** (ergonomics, not correctness).

### Category D — Permutation Test Signature Inconsistency

[VERIFIED: comparing function signatures this session]

Public permutation test functions use inconsistent argument ordering for `n_perm` and `seed`:

| Function | Signature pattern |
|----------|-------------------|
| `t_perm_test` | `(..., n_perm: usize, seed: u64)` |
| `f_perm_test` | `(..., n_perm: usize, seed: u64)` |
| `frechet_anova` | `(..., n_perm: usize, seed: u64)` |
| `fanova` | `(data, groups, n_perm)` — no `seed`, uses internal seeding |
| `generic_permutation_importance` | `(..., n_perm: usize, seed: u64)` |
| `fclassif_cv` | `(..., k_folds, ncomp, seed)` — no `n_perm` |

**Proposed canonical form (API-01):** `(data, ..., n_perm: usize, seed: u64)` as the last two
parameters. The `fanova` signature is inconsistent — it should expose `seed`. Phase 50 adds a
`fanova_v2(data, groups, n_perm, seed)` and deprecates the current form.

---

## Parallelism Gap Inventory (Pre-computation for Phase 48)

[VERIFIED: grep of iter_maybe_parallel and rayon usage this session]

### Already Parallelized (in 9 subsystems)

| Module | Site | Macro |
|--------|------|-------|
| `inference::itp` | `src/inference/itp.rs:78` | `iter_maybe_parallel!(0..p)` |
| `boosting_regression::stability` | `src/boosting_regression/stability.rs:135` | `iter_maybe_parallel!(0..b_count)` |

### Sequential Outer Loops (Parallelism Gap Candidates for Phase 48)

| Module | Function | File:line | Loop structure | Phase 48 priority |
|--------|----------|-----------|----------------|-------------------|
| `inference::permutation` | `t_perm_test`, `f_perm_test` | `permutation.rs:175,238` | `for _ in 0..n_perm` | HIGH (permutation is embarrassingly parallel) |
| `frechet::anova` | `frechet_anova` | `anova.rs:171` | `for perm in 0..n_perm` | HIGH |
| `coclustering` | `co_cluster` | `coclustering.rs:238-333` | nested `for l`/`for k` blocks | MEDIUM (data dep between l/k) |
| `density_fda` | `wasserstein_barycenter` | `density_fda.rs:474` | `for i in 0..n` | MEDIUM (per-curve independent) |
| `frechet::anova` | `frechet_anova_space` | `anova.rs:216` | `for perm` loop | HIGH |

---

## Operational Landmines

[VERIFIED: MEMORY.md and STATE.md reads this session]

### TMPDIR Requirement

All benchmark and test builds must use:
```bash
TMPDIR=/home/simonm/.cache/fdars-bench-tmp
```
The `/tmp` tmpfs on this machine is small; doctests and bench linking exhaust it without this
override, producing spurious "No space left on device" errors that look like build failures.
The cache directory must exist before running: `mkdir -p /home/simonm/.cache/fdars-bench-tmp`.

### target/ Disk Exhaustion

`target/` grows to 100+ GB with full builds. Before any session that builds benches + examples:
```bash
rm -rf /home/simonm/projects/rust/fdars/target/debug/{incremental,examples}
```
This frees ~108 GB (MEMORY.md: `target-dir-fills-home-partition.md`).

### Full Clippy Gate

CI lints test and bench code (not just lib code). A plain `-p fdars-core -D warnings` misses
warnings in test/bench compilation units. Always use:
```bash
cargo clippy --all-targets --features linalg,parallel -- -D warnings
```
This is especially relevant for Phase 46 since throwaway bench code is still compiled.

### Criterion Build Times

Criterion bench compiles with `--features linalg,parallel` are long (MEMORY.md: executor
subagents stall at the 600s watchdog). Build the probe bench separately and confirm compile time
before running measurements:
```bash
TMPDIR=/home/simonm/.cache/fdars-bench-tmp \
  cargo bench --bench <probe_name> --features linalg,parallel --no-run
```
Then run with timing:
```bash
TMPDIR=/home/simonm/.cache/fdars-bench-tmp \
  cargo bench --bench <probe_name> --features linalg,parallel
```

### Governor / CPU-Pinning Caveat

Multi-threaded cells measured with an unpinned CPU governor were flagged as LOW-CONFIDENCE in the
v0.14.0 audit. The Phase 46 report MUST record:
- CPU governor at measurement time: `cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor`
- `RAYON_NUM_THREADS` setting (default: number of logical cores)
- Cargo feature combination used (`linalg,parallel` vs `parallel` only)

### dhat-heap Feature Warning

`dhat-heap` feature MUST NOT be enabled in release builds or default CI. It activates
`#[global_allocator]` in the test harness, which is incompatible with parallel library use.
[VERIFIED: fdars-core/Cargo.toml:31 comment]

---

## Three Inventory Document Specifications

The planner must create three tasks, each producing one document:

### Inventory 1: Hot-Path Target List (PROF-01)

**File:** `.planning/phases/46-whole-crate-profiling-measurement/PROF-01-hotpath-targets.md`

**Method:**
1. Write 9 probe bench files (one per subsystem or grouped by similarity), 3–6 cells each.
2. Add temporary `[[bench]]` entries to `fdars-core/Cargo.toml`.
3. Run each bench: `TMPDIR=... cargo bench --bench <probe> --features linalg,parallel`.
4. Collect wall-time numbers from criterion output (mean ± stddev per cell).
5. Run dhat probes for top-3 subsystems by wall-time.
6. Rank by wall-time × representativeness.

**Output document must contain:**
- Environment section (governor, `RAYON_NUM_THREADS`, feature flags, date)
- Table of all measured cells: `[module] [function] [N] [M] [mean ms] [stddev ms] [allocs if measured]`
- Ranked list: top 10 optimization targets with `file:line` anchors
- N×M scaling regression: does wall-time scale as O(N·M), O(N·M²), O(N²·M²)? (plot or table)

### Inventory 2: Duplication/Consolidation Inventory (PROF-02)

**File:** `.planning/phases/46-whole-crate-profiling-measurement/PROF-02-dedup-inventory.md`

**Method:**
1. Static grep analysis (no bench needed — can be done in parallel with bench runs).
2. For each candidate from §Duplication Candidates above: enumerate all call sites, count them,
   assess implementation-drift risk (are the copies identical or have they diverged?).
3. Rank by dedup leverage = (call sites × complexity × drift risk).

**Output document must contain:**
- Table: `[Category] [Call sites] [Files] [Complexity] [Drift risk] [Leverage score] [CONS-0X target]`
- For each high-leverage item: verbatim quote of the duplicated pattern with `file:line` anchors
- Proposed consolidation location (`src/helpers.rs`, `src/linalg.rs`, new `src/distributions.rs`)

### Inventory 3: API Inconsistency Inventory (PROF-03)

**File:** `.planning/phases/46-whole-crate-profiling-measurement/PROF-03-api-inventory.md`

**Method:**
1. Static analysis only (no bench needed).
2. Enumerate all `pub struct *Config` without `Default`, all `_1d`/`_2d` families, result field
   name inconsistencies, permutation test signature patterns.
3. For each item: document current state, user-facing impact, proposed canonical form.

**Output document must contain:**
- Table: `[Item] [Current forms] [Proposed canonical form] [Impact] [API-0X target]`
- For `#[deprecated]` candidates: exact signature of the unified replacement
- Note which items are additive-safe (add new + deprecate old) vs. require breaking changes
  (out of scope for v0.30.0)

---

## Standard Bench Commands Reference

```bash
# Verify bench compiles (dry run — no execution)
TMPDIR=/home/simonm/.cache/fdars-bench-tmp \
  cargo bench --bench audit_hotpaths --features linalg,parallel --no-run

# Run a specific benchmark group within a bench file
TMPDIR=/home/simonm/.cache/fdars-bench-tmp \
  cargo bench --bench audit_hotpaths --features linalg,parallel -- audit_fpca

# Run dhat allocation probe (integration test)
TMPDIR=/home/simonm/.cache/fdars-bench-tmp \
  cargo test -p fdars-core --features dhat-heap,linalg \
  -- count_fpca_allocations_n500_m200 --nocapture

# Full clippy check (must pass before committing any bench code)
TMPDIR=/home/simonm/.cache/fdars-bench-tmp \
  cargo clippy --all-targets --features linalg,parallel -- -D warnings

# Free target/ before long bench sessions
rm -rf /home/simonm/projects/rust/fdars/target/debug/{incremental,examples}
```

---

## Validation Architecture

This is a **measure-only audit phase**. There are no behavior-changing edits to `fdars-core/src/`
and therefore no Nyquist validation test coverage requirements for Phase 46 itself. The full test
suite remains unchanged and must stay green as a baseline check before measurements begin.

**Pre-measurement gate (not a Nyquist requirement, but a sanity check):**
```bash
TMPDIR=/home/simonm/.cache/fdars-bench-tmp \
  cargo test -p fdars-core --features linalg,parallel 2>&1 | tail -5
# Expected: test result: ok. N passed; 0 failed; 0 ignored
```

The `workflow.nyquist_validation` key is absent from `.planning/config.json` (file does not exist
in this repo), which per the RESEARCH.md template rules means "treat as enabled." However, for a
pure-measurement phase with zero `src/` changes, the only applicable validation is confirming the
existing test suite remains green after adding (then removing) throwaway bench entries — not writing
new `#[test]` functions for measurement behavior.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | dhat allocation baseline numbers (2.4 MB total, ~1.6 MB peak for `fdata_to_pc_1d` N=500,M=200) are still accurate for current code | Allocation Hotspots | Low — these are documentation baselines; Phase 46 will remeasure. If wrong, the doc comment in `alloc_audit_fpca.rs:66-71` is stale but Phase 46 will correct it |
| A2 | `/home/simonm/.cache/fdars-bench-tmp` directory exists or can be created | Operational Landmines | Medium — Phase 46 executor must `mkdir -p` before first bench run |
| A3 | 9 probe benches, each 3–6 cells, will not exhaust `target/` in a single session | Operational Landmines | Medium — with prior cleanup (`rm -rf target/debug/{incremental,examples}`), risk is low; monitor disk during runs |
| A4 | `face` in REQUIREMENTS.md refers to `irreg_fdata::face` (FACE sparse covariance) not a top-level `face` module | Subsystems Inventory | LOW — confirmed by `lib.rs:245` re-export and `fpca_variants.rs:590` comment "FACE sparse-covariance path (`irreg_fdata::face`)" |
| A5 | The 4 missing Config `Default` impls (`BoostingConfig`, `BayesianConfig`, `StabilityConfig`, `StlConfig`) are the complete set | API Inconsistency | LOW — grep was exhaustive; count verified (52 Default / 56 total) |

---

## Sources

### Primary (HIGH confidence — code read this session)

- `fdars-core/benches/audit_hotpaths.rs` — full read, pattern extraction
- `fdars-core/tests/alloc_audit_fpca.rs` — full read, dhat pattern verified
- `fdars-core/Cargo.toml` — feature flags, bench registration, dhat dev-dep
- `fdars-core/src/lib.rs:64-138` — module declarations, `face` re-export at :245
- `fdars-core/src/inference/dist.rs:99-132` — `chi_square_sf` implementation
- `fdars-core/src/spm/chi_squared.rs:164-189` — `chi2_cdf` / `chi2_quantile`
- `fdars-core/src/linalg.rs:16-137` — Cholesky centralization verified
- `fdars-core/src/irreg_fdata/face.rs:1-20` — face module identity confirmed
- Grep corpus: `simpsons_weights`, `fix_svd_signs`, `seed_from_u64`, `DMatrix::from_column_slice`,
  `to_dmatrix`, `iter_maybe_parallel`, `pub struct.*Config`, `impl Default for.*Config`,
  `pub fn.*_1d`, `chi_square_sf`, `n_perm` — all run this session

### Secondary (MEDIUM confidence — CONTEXT.md and planning docs)

- `.planning/phases/46-whole-crate-profiling-measurement/46-CONTEXT.md` — locked decisions
- `.planning/REQUIREMENTS.md` — PROF-01/02/03 acceptance criteria
- `.planning/STATE.md` — operational pointers, milestone decisions

### Tertiary (LOW confidence — prior session doc comments, not re-run)

- dhat allocation baseline numbers in `alloc_audit_fpca.rs` doc comments (prior Phase 4 audit
  measurements; Phase 46 will produce fresh numbers)

---

## Metadata

**Confidence breakdown:**
- Profiling harness pattern: HIGH — read `audit_hotpaths.rs` and `alloc_audit_fpca.rs` in full
- 9 subsystem identification: HIGH — verified module paths and public functions from source
- Duplication candidates: HIGH — grep-verified call sites with file:line anchors
- API inconsistency candidates: HIGH — grep + source reads confirmed
- Allocation baseline numbers: LOW — from prior-audit doc comments, not re-measured this session

**Research date:** 2026-08-30
**Valid until:** 2026-12-31 (codebase is stable; no new modules expected until next milestone)
