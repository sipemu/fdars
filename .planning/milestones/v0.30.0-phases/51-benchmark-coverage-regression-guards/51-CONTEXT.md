# Phase 51: Benchmark Coverage & Regression Guards - Context

**Gathered:** 2026-09-01
**Status:** Ready for planning
**Mode:** Smart discuss (autonomous) — benchmark-infrastructure phase (measurement only; NO src behavior change)

<domain>
## Phase Boundary

Two deliverables, both measurement-only (no `src/` behavior changes, no new crate dependency):
- **BENCH-01** — add criterion `[[bench]]` coverage for the 9 currently-unbenchmarked new modules:
  `fts`, `frechet`, `boosting_regression`, `coclustering`, `fem_smoothing`, `density_fda`,
  `inference`, `fpca_variants`, `face`. (Existing benches cover seasonal/depth/classification/
  alignment/regression/explain/smoothing/basis/matrix + audit_hotpaths — these 9 have none.)
- **BENCH-02** — commit the PERF-proof benches (`perf_hotpaths.rs` from Phase 47, `perf_parallelism.rs`
  from Phase 48 — both already registered PERMANENT) as regression guards with the before/after
  numbers documented so future changes can detect regressions.

The full clippy gate (`cargo clippy --all-targets --features linalg,parallel -- -D warnings`, which
lints bench code) must stay green with the new bench entries. No new crate dependency (criterion +
feature-gated dhat-heap are existing dev-deps).

</domain>

<decisions>
## Implementation Decisions

### Regression-guard mechanism (operator-confirmed)
- **Documented baselines (soft guards)** — criterion benches MEASURE; they do not fail CI on
  regression. Regressions are caught via `cargo bench` + criterion's built-in baseline comparison
  against the committed before/after numbers.
- **Keep the existing DETERMINISTIC hard-assert guards** — `tests/alloc_audit_dpca.rs` +
  `tests/alloc_audit_fpca.rs` hard-assert allocation-block counts (deterministic, feature `dhat-heap`).
  These stay as real CI guards.
- **NO wall-time `assert!` regression tests** — wall-time is governor-sensitive (the milestone flagged
  `powersave`-governor numbers LOW-CONFIDENCE); a hard wall-time threshold would flake on the unpinned
  CI runner. Do NOT add them.
- **BENCH-RESULTS.md** — a consolidated ledger documenting the PERF wins as guarded baselines,
  referencing the existing PERF-RESULTS.md (Phase 47) + PERF-PARALLEL-RESULTS.md (Phase 48):
  face_covariance −80.7%, dpca −54% allocations, frechet_anova 9.9×, co_cluster 6.4×. Record the
  environment/governor caveat so future re-runs are comparable.

### BENCH-01 structure
- **One `[[bench]]` file per module** (mirror the existing `*_benchmarks.rs` naming + `harness = false`
  convention), all 9 modules. Register each in `fdars-core/Cargo.toml`.
- Mirror the established criterion pattern (`benches/perf_hotpaths.rs` is the template): build
  deterministic data OUTSIDE `b.iter()` (NO RNG in data generators — reuse the sinusoid/density
  generators already used in perf_hotpaths / equivalence_phase47-49), `black_box` inputs+outputs,
  `group.sample_size(N)` + `group.measurement_time(Duration)` tuned per cost, `criterion_group!` /
  `criterion_main!`.
- Cell sizes: mirror the PROF-01 measurement cells where a module has one (fts::dpca, frechet_anova,
  fem_smooth, face_covariance already in perf_hotpaths/perf_parallelism — the NEW benches cover the
  module's OTHER representative entry points, not duplicate those). Pick 1–3 representative public fns
  per module at a size that runs in reasonable bench time.
- Feature-awareness: some modules need `linalg` (e.g. frechet/face). Benches are built/run under
  `--features linalg,parallel`; gate any linalg-only bench cell appropriately so the crate still builds
  under `--no-default-features` if that matters for the clippy gate (Claude's discretion — match how
  perf_hotpaths handles it).

### BENCH-02
- `perf_hotpaths.rs` + `perf_parallelism.rs` are already `[[bench]]`-registered as PERMANENT (Phase
  47/48). This phase confirms they remain, and lands BENCH-RESULTS.md consolidating the documented
  before/after. Keep `alloc_audit_dpca.rs`/`alloc_audit_fpca.rs` as-is (deterministic guards).

### Gates (Claude's Discretion on exact mechanics)
- `cargo clippy --all-targets --features linalg,parallel -- -D warnings` clean (lints the new bench
  code — the primary gate for this phase).
- `cargo build -p fdars-core --benches --features linalg,parallel` builds all bench entries.
- Optionally run each new bench once (short sample) to confirm it executes and to capture a baseline
  cell for BENCH-RESULTS.md — WATCH DISK/TMP: prefix cargo with `TMPDIR=/home/simonm/.cache/fdars-bench-tmp`;
  `rm -rf target/debug/{incremental,examples}` if space runs low; benches are release builds (slow compile).
- No behavior-changing `src/` edits; no new dependency. `cargo fmt` + `git commit --no-verify` per commit.

</decisions>

<code_context>
## Existing Code Insights

### Bench pattern (template = benches/perf_hotpaths.rs)
- `use criterion::{black_box, criterion_group, criterion_main, Criterion};` + `std::time::Duration`.
- Deterministic `generate_curves(n, m)` sinusoid generator (column-major FdMatrix), data built once
  outside `b.iter()`; `black_box` on inputs and the returned result.
- `group.sample_size(10-20)`, `group.measurement_time(Duration::from_secs(30-60))`,
  `group.warm_up_time(...)` tuned to cost.
- Cargo.toml: `[[bench]] name = "…"  harness = false`.

### Already-benchmarked (do NOT duplicate)
- seasonal, depth, classification, alignment, regression, explain, smoothing, basis, matrix
  (`*_benchmarks.rs`); audit_hotpaths; perf_hotpaths (dpca/face_covariance/fem_smooth); perf_parallelism
  (frechet_anova/co_cluster).

### The 9 target modules + candidate entry points (planner/researcher to confirm exact public fns)
- `fts` (functional time series — dpca is in perf_hotpaths; bench e.g. ftsm/forecast/other fts fns),
  `frechet` (frechet_anova in perf_parallelism; bench frechet regression/mean/other), `boosting_regression`
  (fosr boosting fit), `coclustering` (co_cluster in perf_parallelism; bench co_cluster_select or a
  different cell), `fem_smoothing` (fem_smooth in perf_hotpaths; bench another entry / mesh size),
  `density_fda`, `inference` (t_perm_test/f_perm_test/vstat), `fpca_variants` (pace_fpca/pca variants),
  `face` (face_covariance in perf_hotpaths; bench another face entry).

### Environment / MEMORY pointers (bench-heavy phase)
- `TMPDIR=/home/simonm/.cache/fdars-bench-tmp` REQUIRED (doctest/bench link into /tmp tmpfs → bogus
  "No space left" otherwise).
- `target/` grows to 100+GB and fills /home; `rm -rf target/debug/{incremental,examples}` frees ~108G
  if a build/link fails ("linking with cc failed" — NOT a code bug).
- Full clippy gate lints test/bench code; a plain `-p … -D warnings` false-greens.
- Governor `powersave` unpinned → absolute bench numbers LOW-CONFIDENCE; record the environment.

</code_context>

<specifics>
## Specific Ideas

- 9 new `benches/<module>_benchmarks.rs` files (or a naming the planner prefers), each `harness=false`,
  registered in Cargo.toml, covering the module's representative public entry points at PROF-01-scaled
  cells — WITHOUT duplicating the cells already in perf_hotpaths/perf_parallelism.
- `BENCH-RESULTS.md` consolidating the guarded PERF baselines + the new modules' measured cells + the
  governor/environment caveat.

</specifics>

<deferred>
## Deferred Ideas

- Hard wall-time `assert!` regression tests — governor-sensitive/flaky; explicitly out of scope
  (documented-baseline guards + deterministic alloc asserts only).
- Any behavior-changing `src/` optimization — this phase is measurement/coverage only (optimizations
  were Phases 47/48).
- CI wiring to auto-run benches on every PR — out of scope (benches remain `cargo bench`-on-demand).

</deferred>
