# Stack Research: v0.47.0 Cross-Language Benchmark Harness + Optimization Pass

**Domain:** Cross-language (Rust / Python / R) FDA performance benchmark harness, plus behavior-preserving Rust optimization pass on fdars-core
**Researched:** 2026-09-19
**Confidence:** MEDIUM (web search + direct codebase read; tool versions verified against PyPI/CRAN search results; criterion output format verified via docs; existing validation/ R infrastructure read directly)

---

## Boundary Rule — What Lives Where

The hard constraint from PROJECT.md is absolute:

- The **cross-language harness lives OUTSIDE `fdars-core`** — in `benchmarks/` (external Python/R scripts + an orchestration script). No new entry in `fdars-core/Cargo.toml` of any kind.
- The **Rust optimization pass** touches only `fdars-core/src/`. No new crate dependency (`[dependencies]` or `[dev-dependencies]`) is added to `Cargo.toml`. The existing `criterion 0.5` dev-dependency is sufficient for before/after measurement.
- External CLI tools (`cargo install flamegraph`, system `perf`) are **not** Cargo.toml entries — they are development machine tools installed separately and are allowed.

This boundary appears in every section below under "External vs. Cargo-dep" tags.

---

## Recommended Stack

### Rust Side — Already Present, Nothing New

| Technology | Version | Purpose | Cargo dep? |
|------------|---------|---------|------------|
| criterion | 0.5 (existing dev-dep) | Before/after micro-benchmarks; the existing 24+ bench files cover the operations being optimized | Existing — no change |
| rustc / cargo | 1.97.0 (dev) / MSRV 1.81 | Build + bench; `cargo bench --features linalg,parallel` is the primary benchmark invocation | External toolchain |
| cargo-flamegraph | latest (cargo install) | CPU flamegraph for profiling optimization candidates; wraps `perf` on Linux | External CLI — NOT a Cargo dep |
| perf (Linux) | system | CPU sampling profiler used by cargo-flamegraph | System tool |
| samply | latest (cargo install) | Alternative profiler (Firefox Profiler UI); lighter to set up than perf on some systems | External CLI — NOT a Cargo dep |

**Why criterion 0.5 is sufficient for the Rust side:** The existing benchmark suite already covers FPCA, alignment (elastic_align_pair, karcher_mean), depth, smoothing, regression, and classification. Adding new bench functions for the elastic O(N²·M²) hotspot targets within the existing bench files requires zero Cargo.toml changes. Before/after wall-clock medians from criterion are the authoritative Rust-side numbers.

**Why NOT dhat-heap:** `dhat 0.3.2` would require adding `[dev-dependencies] dhat = "0.3"` to `fdars-core/Cargo.toml`. This violates the no-new-Cargo-dep constraint. If heap allocation profiling is needed for the optimization pass, use `perf` with `--event=cache-misses` or heaptrack (external system tool) instead.

**Why NOT cargo-criterion:** `cargo-criterion` provides stable `--message-format=json` stdout output, but it is a separate binary that requires installing. The orchestration script can instead parse `target/criterion/*/estimates.json` directly — these internal JSON files have been stable across criterion 0.4→0.5 in practice, even though they are technically private API.

### Python Side — External, No Cargo Dependency

| Technology | Version | Purpose | External vs. Cargo-dep |
|------------|---------|---------|------------------------|
| scikit-fda | 0.10.1 (pinned) | The Python baseline; already verified installed in `.planning/research/skfda-verify/venv/` | External — pip install |
| pytest-benchmark | 5.3.0 | Time Python FDA operations; `--benchmark-json` produces structured JSON with all timing stats | External — pip install |
| numpy | ≥1.24 (scikit-fda requirement) | Array operations; controls BLAS backend | External — pip install |
| scipy | ≥1.9 (scikit-fda requirement) | LAPACK/BLAS for SVD, linear algebra | External — pip install |
| uv | ≥0.4 | Lock the Python environment via `uv pip compile` → `benchmarks/requirements.lock` + `uv pip sync` for exact reproducible installs | External — system install |

**Why pytest-benchmark over pyperf:** pytest-benchmark 5.3.0 integrates directly with pytest fixtures, which allows realistic setup (constructing `FDataGrid` objects once outside the timed loop), outputs `--benchmark-json` with full stats (min, median, stddev, rounds, iterations), and requires only one file per FDA operation. pyperf 2.10.0 provides stronger process-isolation guarantees, but the setup overhead for constructing matched N×M inputs across languages makes it harder to keep the datasets identical. Use pytest-benchmark with `--benchmark-disable-gc` and explicit `rounds=10` for the outer comparison runs.

**Why NOT airspeed velocity (asv) 0.6.5:** asv is designed for tracking benchmarks across a project's commit history with its own virtual environment management and git context. It is significantly more complex to set up for a one-shot cross-language comparison. A simple pytest-benchmark script with JSON output is the right tool for this milestone.

**Why NOT plain `timeit`:** timeit has no structured output format, no CLI JSON export, and no built-in statistical aggregation. pytest-benchmark wraps timeit internally but adds all of the above.

### R Side — External, No Cargo Dependency

| Technology | Version | Purpose | External vs. Cargo-dep |
|------------|---------|---------|------------------------|
| bench | 1.1.4 | Time R FDA package calls; returns tibble with min/median/mem_alloc/gc columns; export via `jsonlite::toJSON()` | External — install.packages() |
| microbenchmark | 1.5.0 | Alternative timing (100 iterations, nanosecond precision); simpler than bench for pure timing without memory tracking | External — install.packages() |
| jsonlite | ≥1.8 | Serialize bench/microbenchmark results to JSON for the orchestration collector | External — already in validation/R/install_deps.R |
| renv | ≥1.0 | Lock R package versions via `renv::snapshot()` → `benchmarks/renv.lock`; `renv::restore()` for reproducibility | External — install.packages() |
| fda | ≥6.1 | R FDA baseline — B-spline smoothing, FPCA, regression | External — already installed for validation |
| fda.usc | ≥2.1 | R FDA baseline — depth, kNN classification, kernel regression | External — already installed for validation |
| refund | ≥0.1-38 | R FDA baseline — FPCR, FLR, pfr | External — install.packages() |
| fdapace | ≥0.5.9 | R FDA baseline — PACE FPCA, optimal design | External — install.packages() |
| roahd | ≥1.4 | R FDA baseline — robust depth, band depth | External — already installed for validation |

**Why bench over microbenchmark as the primary R timer:** bench::mark() tracks memory allocation and GC events alongside timing, which is relevant for R FDA operations that allocate large intermediate matrices. microbenchmark is the fallback for R versions < 4.0 or when the tibble output format creates complexity. Use `bench::mark(..., iterations = 50)` and export via `as.data.frame() |> jsonlite::toJSON()`.

**renv lockfile:** The existing `validation/R/install_deps.R` covers `fda`, `fda.usc`, `roahd`, `fdasrvf` (all already present). New additions needed for v0.47.0: `bench`, `refund`, `fdapace`. Add all to `benchmarks/R/install_deps.R` (separate from the validation deps), lock with renv.

### Orchestration — Python Script in `benchmarks/`

| Technology | Version | Purpose | External vs. Cargo-dep |
|------------|---------|---------|------------------------|
| Python 3.x | 3.12+ (system) | Orchestration script that runs all three benchmark tiers and collects results | External — system Python |
| subprocess | stdlib | Run `cargo bench`, Rscript, and pytest-benchmark as subprocesses | stdlib — no install |
| json | stdlib | Parse criterion estimates.json and pytest-benchmark JSON output | stdlib — no install |
| csv | stdlib | Write unified comparison table as CSV | stdlib — no install |
| pathlib | stdlib | Path manipulation for result files | stdlib — no install |

The orchestration script (`benchmarks/run_all.py`) is the integration point. It:
1. Runs `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 cargo bench --features linalg,parallel --bench <bench_name> 2>&1` and captures stdout
2. Parses `target/criterion/*/estimates.json` for median + confidence intervals
3. Runs `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 pytest benchmarks/python/ --benchmark-json=benchmarks/results/python.json -v`
4. Runs `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 Rscript benchmarks/R/run_benchmarks.R benchmarks/results/r.json`
5. Joins all results by `(operation, n, m)` key into `benchmarks/results/comparison.csv` and `comparison.json`

---

## Version Capture — Reproducibility Mechanics

### What to Capture at Benchmark Runtime

Every benchmark run must record these into `benchmarks/results/env.json`:

| Field | How to Capture |
|-------|----------------|
| `rustc_version` | `rustc --version` |
| `cargo_version` | `cargo --version` |
| `fdars_version` | `grep '^version' fdars-core/Cargo.toml` |
| `criterion_version` | `grep 'criterion' fdars-core/Cargo.toml` |
| `python_version` | `python3 --version` |
| `scikit_fda_version` | `python3 -c "import skfda; print(skfda.__version__)"` |
| `numpy_version` | `python3 -c "import numpy; print(numpy.__version__)"` |
| `scipy_version` | `python3 -c "import scipy; print(scipy.__version__)"` |
| `r_version` | `Rscript -e "cat(R.version.string)"` |
| `r_pkg_versions` | `Rscript -e "cat(as.character(packageVersion('fda')), ...)"` for each R package |
| `blas_backend` | `python3 -c "import numpy; numpy.show_config()"` or `Rscript -e "sessionInfo()"` |
| `cpu_model` | `/proc/cpuinfo` or `lscpu` |
| `os_version` | `uname -a` |
| `rayon_num_threads` | Environment variable at time of run |
| `openblas_num_threads` | Environment variable at time of run |
| `timestamp` | ISO 8601 UTC |

### Environment Variables to Set Before Any Run

```bash
# Pin BLAS threading for fair single-threaded comparison
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
export OMP_DYNAMIC=FALSE

# Pin Rayon thread count for the parallel-disabled Rust comparison run
# (run benches twice: once with RAYON_NUM_THREADS=1, once with default)
# Default run: unset RAYON_NUM_THREADS (uses all cores)
# Sequential comparison run: RAYON_NUM_THREADS=1
```

### Python Lock

```bash
uv pip compile benchmarks/python/requirements.in --output-file benchmarks/python/requirements.lock
uv pip sync benchmarks/python/requirements.lock
```

`requirements.in` pins: `scikit-fda==0.10.1`, `pytest-benchmark==5.3.0`, `numpy`, `scipy`.

### R Lock

```r
# In benchmarks/R/ directory
renv::init()
renv::install(c("bench", "microbenchmark", "jsonlite", "fda", "fda.usc", "refund", "fdapace", "roahd"))
renv::snapshot()  # writes benchmarks/renv.lock
```

---

## Unified Output Format

All three tiers produce a JSON record per operation with this schema:

```json
{
  "operation": "fpca",
  "n": 200,
  "m": 100,
  "language": "rust",
  "library": "fdars-core",
  "library_version": "0.47.0",
  "median_ns": 12345678,
  "mean_ns": 12400000,
  "stddev_ns": 150000,
  "min_ns": 12100000,
  "max_ns": 12900000,
  "samples": 20,
  "feature_flags": "linalg,parallel"
}
```

The Python orchestration script merges per-language JSON files into a single `comparison.json` + `comparison.csv` keyed on `(operation, n, m)`.

**Criterion → JSON conversion:** After `cargo bench`, parse `target/criterion/<group>/<bench_id>/new/estimates.json`. Fields: `mean.point_estimate`, `median.point_estimate`, `std_dev.point_estimate` — all in nanoseconds. The `sample.json` in the same directory has the raw iteration values.

**pytest-benchmark → JSON:** `--benchmark-json=benchmarks/results/python.json` writes a file with `benchmarks[].stats.{min,max,mean,median,stddev,rounds,iterations}` all in seconds. Multiply by 1e9 for nanoseconds to match criterion.

**R bench → JSON:** In the R script, `jsonlite::write_json(as.data.frame(result), path)` where `result` is from `bench::mark()`. The `median` column is in seconds (double); multiply by 1e9 for nanoseconds.

---

## Alternatives Considered

| Recommended | Alternative | When Alternative Wins |
|-------------|-------------|----------------------|
| pytest-benchmark 5.3.0 | pyperf 2.10.0 | If you need strong process-isolation guarantees for very fast (<1µs) Python operations. For FDA operations (milliseconds to seconds), pytest-benchmark's within-process isolation is sufficient. |
| bench 1.1.4 | microbenchmark 1.5.0 | microbenchmark is better for R < 4.0, or when the tibble dependency chain is undesirable. Also simpler for quick interactive checks. Use microbenchmark as a fallback validator, bench as the primary. |
| uv for Python locking | pip-tools / pip freeze | uv is faster and produces a cross-platform lock. pip freeze without compile is non-deterministic across Python versions. |
| renv for R locking | Packrat (deprecated) | renv is the current r-lib standard; packrat is superseded. |
| External Python orchestrator | Makefile | Python gives structured JSON/CSV collection and arithmetic (computing speedup ratios). A Makefile can drive the runs but cannot merge the output files without external tools. |
| cargo-flamegraph (external CLI) | dhat crate (dev-dep) | dhat is useful for heap profiling, but it requires adding a dev-dependency to Cargo.toml which violates the no-new-dep constraint. Use heaptrack (system tool, no Cargo dep) if heap profiling is needed. |
| Parse target/criterion/*/estimates.json | cargo-criterion --message-format=json | cargo-criterion requires a separate install step; parsing estimates.json is simpler and has been stable in practice across criterion 0.4→0.5. Use cargo-criterion if the estimates.json format becomes unstable. |

---

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `dhat` as `[dev-dependencies]` | Violates no-new-Cargo-dep constraint; adding it to `fdars-core/Cargo.toml` is forbidden | `heaptrack` (system tool) or `cargo-flamegraph` with allocation events via `perf` |
| `asv` (airspeed velocity) | Over-engineered for a single-milestone comparison; requires git context, per-commit venvs, machine config files | `pytest-benchmark` with `--benchmark-json` |
| Plain `timeit` in Python | No structured output, no statistical aggregation, no JSON export | `pytest-benchmark` which wraps timeit and adds all of these |
| `cargo-criterion` as a Cargo dep | It is a binary, not a library — it is installed via `cargo install cargo-criterion`, NOT added to Cargo.toml | `cargo install cargo-criterion` OR just parse `target/criterion/*/estimates.json` |
| Uncontrolled BLAS threading | OpenBLAS/MKL auto-detect CPU count; this inflates Python/R times on multi-core machines, making the comparison unfair vs. Rust's explicit `RAYON_NUM_THREADS` control | Set `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` for all runs |
| Single-run timing in any language | Single runs are dominated by cache warmup, JIT effects (Python), and garbage collection | Use minimum 20 criterion samples (Rust), minimum 10 bench rounds (R), minimum 5 pytest-benchmark rounds (Python) with at least 3 warmup rounds |
| Comparing debug-build Rust to release Python/R | Criterion already runs at `opt-level=3` under `cargo bench`, but manually invoking `cargo test --bench` can produce debug builds | Always use `cargo bench` (not `cargo test`); verify the output path contains `/release/deps/` not `/debug/deps/` |
| `refund` + `fdapace` without checking they use the same BLAS | These R packages link to the R LAPACK/BLAS layer; thread count must be set before `library()` is called | Set `OPENBLAS_NUM_THREADS=1` as an environment variable before starting the Rscript process, not inside the script |

---

## Directory Layout for the Harness

```
benchmarks/
  run_all.py              # orchestration: runs Rust + Python + R, collects results
  python/
    requirements.in       # pinned: scikit-fda==0.10.1, pytest-benchmark==5.3.0
    requirements.lock     # uv pip compile output (exact, reproducible)
    bench_fpca.py         # pytest-benchmark: FPCA
    bench_smoothing.py    # pytest-benchmark: B-spline smoothing
    bench_depth.py        # pytest-benchmark: depth measures
    bench_elastic.py      # pytest-benchmark: elastic alignment (Fisher-Rao)
    bench_regression.py   # pytest-benchmark: scalar-on-function regression
    shared_data.py        # shared N×M synthetic dataset generation (matches Rust generators)
  R/
    install_deps.R        # install bench, microbenchmark, jsonlite, fda, fda.usc, refund, fdapace, roahd
    renv.lock             # renv::snapshot() output
    run_benchmarks.R      # bench::mark() for all R FDA operations, writes JSON
    shared_data.R         # matching synthetic dataset generation
  results/
    .gitkeep              # empty; results are generated artifacts, not committed
  env_template.json       # schema for the captured environment record
```

The `results/` directory is excluded from git (`.gitignore`). The `python/requirements.lock` and `R/renv.lock` files ARE committed — they are the reproducibility guarantee.

---

## Rust-Side Profiling Workflow (Optimization Pass)

The optimization pass does not add Cargo dependencies. The workflow is:

1. **Identify candidate**: Run `cargo bench --features linalg,parallel` and compare criterion medians against the cross-language baseline. The elastic O(N²·M²) path in `alignment/` is the known anchor target.

2. **Profile**: `cargo install flamegraph` (one-time), then:
   ```bash
   sudo sysctl -w kernel.perf_event_paranoid=1
   TMPDIR=/home/simonm/.cache/fdars-bench-tmp \
     cargo flamegraph --features linalg,parallel \
       --bench alignment_benchmarks -- --bench elastic_align_pair
   ```
   Output: `flamegraph.svg` in the project root.

3. **Optimize**: Edit `fdars-core/src/` only. The existing column-major `FdMatrix` layout, `iter_maybe_parallel!` macro, and `linalg::cholesky_solve` are the reuse targets. No new crate dependency.

4. **Verify**: Full test suite must stay green: `cargo test --features linalg,parallel`. Then `cargo bench` before vs. after within criterion — criterion's built-in comparison mode flags regressions automatically.

5. **Re-run harness**: Run `benchmarks/run_all.py` after the optimization to produce the final before/after cross-language numbers.

---

## Version Compatibility Notes

| Package | Version | Compatibility Notes |
|---------|---------|---------------------|
| criterion | 0.5 (existing) | Uses `html_reports` feature; `estimates.json` schema stable since 0.4; `sample.json` contains raw nanosecond values |
| scikit-fda | 0.10.1 | Requires numpy ≥ 1.24, scipy ≥ 1.9, scikit-learn ≥ 1.2; FisherRaoElasticRegistration API stable since 0.9 |
| pytest-benchmark | 5.3.0 | Requires pytest ≥ 7; JSON schema stable since 3.x; `--benchmark-json` flag present since 3.0 |
| pyperf | 2.10.0 | Optional fallback; JSON schema stable; not used in primary harness |
| bench (R) | 1.1.4 | Requires R ≥ 4.0; tibble ≥ 3.2.1; `mem_alloc` column present since 1.1.0 |
| microbenchmark (R) | 1.5.0 | No R version minimum beyond base; stable since 1.4.x |
| renv (R) | ≥ 1.0 | renv.lock format changed between 0.x and 1.x — use 1.x format only |
| uv | ≥ 0.4 | requirements.lock format stable; `uv pip compile` + `uv pip sync` workflow compatible with standard pip |
| Rust MSRV | 1.81 | No change — optimization pass uses only existing stable APIs |

---

## Sources

- pytest-benchmark readthedocs (5.3.0) — version, `--benchmark-json` format, timing mechanics — web search result — MEDIUM confidence
- pyperf readthedocs (2.10.0) — multi-process isolation, JSON output — web search result — MEDIUM confidence
- asv readthedocs (0.6.5) — JSON result storage, per-machine tracking — web search result — MEDIUM confidence (ruled out for this use case)
- scikit-fda readthedocs (0.10.1) — API surface (FPCA, smoothing, depth, elastic registration) — web search + direct install verification in `.planning/research/skfda-verify/version.txt` — MEDIUM confidence (verified at runtime)
- R microbenchmark CRAN (1.5.0) — timing mechanics, nanosecond precision, OS-specific timers — web search result — MEDIUM confidence
- R bench CRAN (1.1.4) — tibble output format, mem_alloc tracking — web search result — MEDIUM confidence
- Intel/OpenBLAS threading docs — `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `OMP_NUM_THREADS`, `MKL_DYNAMIC` — web search result — MEDIUM confidence
- cargo-flamegraph GitHub (flamegraph-rs/flamegraph) — installation, perf wrapping, `--no-rosegment` requirement for lld/mold — web search result — MEDIUM confidence
- dhat-rs (nnethercote/dhat-rs, 0.3.2) — heap profiling, dev-dependency usage — web search result — MEDIUM confidence (ruled out: violates no-new-Cargo-dep constraint)
- criterion.rs docs / bheisler.github.io — `estimates.json` internal format, `cargo-criterion --message-format=json` stable API — web search result — MEDIUM confidence
- `fdars-core/Cargo.toml` — criterion 0.5 existing dev-dep, MSRV 1.81, feature flags — direct codebase read — HIGH confidence
- `fdars-core/benches/alignment_benchmarks.rs`, `perf_hotpaths.rs` — existing bench structure, TMPDIR workaround, `--features linalg,parallel` invocation — direct codebase read — HIGH confidence
- `validation/R/install_deps.R` — existing R packages (fda, fda.usc, roahd, fdasrvf, jsonlite already installed) — direct codebase read — HIGH confidence
- `.planning/research/skfda-verify/version.txt` — scikit-fda 0.10.1 confirmed installed and working in existing venv — direct file read — HIGH confidence
- uv documentation — `uv pip compile` / `uv pip sync` lock workflow — web search result — MEDIUM confidence
- renv documentation — `renv::snapshot()` lockfile format, 1.x API — web search result — MEDIUM confidence

---

*Stack research for: v0.47.0 cross-language FDA benchmark harness (Rust/Python/R) + behavior-preserving Rust optimization pass*
*Researched: 2026-09-19*
