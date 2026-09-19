# Pitfalls Research

**Domain:** Cross-language numerical benchmarking (fdars-core vs scikit-fda vs R FDA packages) + behavior-preserving Rust optimization pass — v0.47.0
**Researched:** 2026-09-19
**Confidence:** HIGH (project-specific patterns from codebase analysis + MEDIUM-verified web findings cross-checked against fdars constraints)

---

## Critical Pitfalls — Part A: Cross-Language Benchmarking

### Pitfall A1: Measuring Python Cold-Start / Import Overhead Instead of Algorithm Compute

**What goes wrong:**
A Python benchmark script that imports `scikit-fda`, `numpy`, and `scipy` at the top and then times a single call to `FPCA().fit(data)` is measuring: module import time (~50–500ms depending on environment), NumPy BLAS library loading, C-extension initialization, and potentially JIT compilation of Numba/Cython paths — all folded into the "algorithm time". The result looks slow not because scikit-fda's FPCA is slow but because Python startup is slow.

Concretely: `time.perf_counter()` around a single `fit()` call in a fresh process captures all of the above. Reported time may be 10–100x the actual compute time for small-to-medium datasets.

**Why it happens:**
Developers write the benchmark as a script that runs once, matching how they'd time a Rust `cargo run` call. But CPython has no JIT; every import runs interpreter code. The first call to any NumPy/scipy function also triggers BLAS thread pool initialization.

**How to avoid:**
- Use `timeit` or `pytest-benchmark` / `pyperf` with explicit warm-up iterations before recording. Minimum: 3–5 warm-up executions per benchmark.
- Separate import from compute: import all libraries at module load time, then time only the algorithm call.
- Alternatively run each benchmark in a loop of at least 10 iterations and report the minimum (not mean) of the non-first iterations.
- For R: `bench::mark()` already runs a garbage-collect + warm-up loop; `system.time()` on a single call does not — never use bare `system.time()` for cross-language comparison.
- Capture Python and R `VERSION` strings in the environment record; cold-start overhead varies by Python version.

**Warning signs:**
- Python timings for small operations (N=50 curves, M=100 points) are >10ms — almost certainly measuring startup, not compute.
- Running the same benchmark twice in the same process gives times that differ by more than 2x on the first vs second call.
- Total benchmark wallclock time is dominated by a single initial call.

**Phase to address:** Harness-setup phase (Phase 112). Gate: verify first-call vs steady-state difference is < 5% before recording any data point.

---

### Pitfall A2: BLAS / OpenMP Thread-Count Mismatch Making Comparisons Unfair

**What goes wrong:**
`numpy` linked against OpenBLAS or MKL auto-detects the number of CPU cores at startup and uses all of them for BLAS operations (matrix multiply, SVD, Cholesky). A 16-core machine will run `np.linalg.svd()` on 16 threads while fdars-core (using nalgebra, which has no BLAS dependency) runs on 1 thread. The benchmark will show Python FPCA 8–16x faster than Rust, which is not a fair comparison of algorithm efficiency.

The reverse also kills credibility: if fdars-core uses rayon with `--features parallel` but Python is pinned to 1 thread, fdars looks unfairly fast.

**Why it happens:**
BLAS thread-count defaults are invisible — no warning is printed and the environment variable is not documented in the benchmarking README. The benchmark author sees a number, records it, and publishes it.

**How to avoid:**
- Always set all BLAS thread-count environment variables before the Python/R process starts:
  ```bash
  export OMP_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export BLIS_NUM_THREADS=1
  export NUMEXPR_NUM_THREADS=1
  ```
- For fdars: run benchmarks in two modes — (a) with `--features parallel` disabled (single-threaded), (b) with rayon and `RAYON_NUM_THREADS=1`, comparing against Python/R at 1 thread. Optionally add a multi-threaded comparison (all N threads vs all N threads) but label it clearly.
- In R: `RhpcBLASctl::blas_set_num_threads(1)` or set the same env vars before starting R. R itself is single-threaded; the BLAS overhead matters for `fda`, `refund`, etc.
- Record the thread counts used in the environment capture block of every benchmark result file.
- Print `numpy.__config__.show()` and `sessionInfo()` (R) to record which BLAS library is linked.

**Warning signs:**
- Python SVD time scales inversely with core count but Rust time does not — BLAS threading is active.
- Python benchmark time drops sharply when moving from `OMP_NUM_THREADS=1` to default — confirms multi-threading was the source.
- `nproc` shows 16 cores and Python timings are suspiciously fast for SVD/FPCA.

**Phase to address:** Harness-setup phase. Gate: the harness README documents all thread-count env vars; each result JSON file includes a `thread_config` field.

---

### Pitfall A3: Apples-to-Oranges Algorithm Variants (Different Convergence Tolerances, Smoothing, Basis Choices)

**What goes wrong:**
`FPCA` in scikit-fda uses B-spline smoothing with a configurable number of basis functions. `FPCA` in fdars-core uses Simpson's-weights integration on the raw data grid. The R `fda` package's `pca.fd()` uses functional objects with their own basis. All three implementations are called "FPCA" but compute different quantities unless carefully configured to match.

Specific divergences that look like performance differences but are actually algorithm differences:
- Scikit-fda's FPCA with `n_basis=10` B-splines is faster than fdars SVD on the full 200-point grid — but it is also less accurate because it projects onto a low-dimensional basis first.
- R's `elastic_registration()` uses a different tolerance and convergence criterion than fdars `elastic_align_pair()` — they may iterate different numbers of steps, making wall-time comparison meaningless.
- Depth functions: halfspace depth in R's `roahd` uses a different random-projection count than fdars depth measures — more projections = slower but more accurate.

**Why it happens:**
Developers call the "FPCA function" in each library with default parameters and assume defaults are equivalent. Default parameters vary widely across libraries.

**How to avoid:**
- Before timing, verify numeric agreement: run both implementations on the same synthetic dataset and assert results match to within a specified tolerance (e.g., ||scores_A - scores_B|| / ||scores_A|| < 0.01 after sign alignment). If they do not agree, the implementations are not computing the same thing.
- Document the exact parameter configuration used for each library in the benchmark harness:
  ```python
  # scikit-fda: n_basis=None (raw data, no smoothing), n_components=3
  # fdars: default config, ncomp=3
  # R fda: eval.points=same grid, ncomp=3
  ```
- For elastic alignment: pin `max_iter` and `tol` explicitly in all libraries to the same values before timing.
- For depth measures: pin the number of random projections.
- Flag any parameter that had to differ (e.g., because one library does not expose the setting) as a comparison caveat in the report.

**Warning signs:**
- Numeric results differ by >5% between two "equivalent" calls on the same data — algorithm variant mismatch, not timing noise.
- One library converges in 5 iterations and another in 50 iterations for the same task — tolerance mismatch.
- Benchmark results contradict theoretical complexity (e.g., O(N·M²) vs O(N·M) — different algorithm paths chosen by defaults).

**Phase to address:** Harness-setup and baseline-capture phases. Gate: numeric-agreement check (correlation > 0.99 or relative error < 1%) on a reference dataset before any timing is recorded.

---

### Pitfall A4: Not Verifying Libraries Actually Compute the Same Thing (Missing Numeric-Agreement Gate)

**What goes wrong:**
The harness measures time but never checks that the outputs of all three libraries (fdars, scikit-fda, R) agree numerically on the same input. This means:
- A bug in one library produces wrong results at high speed and looks like a "performance win."
- An algorithm variant (see A3) produces a different answer at different speed and the speedup number is meaningless.
- Sign conventions differ (FPCA eigenvectors, see below) — dot products with flipped sign are negative, indicating a comparison failure that goes undetected without a check.

**Why it happens:**
Benchmarking is treated as a timing-only activity; correctness checks are considered separate. They are not separate — a correctness check is a necessary precondition for any timing comparison to be meaningful.

**How to avoid:**
- Implement a `verify_agreement(result_a, result_b, tolerance=0.01)` function that runs before any timing loop:
  1. Align eigenvector signs: for FPCA, multiply each component by `sign(dot(comp_a, comp_b))` before comparing.
  2. Check that the absolute relative difference in the primary output (scores, fitted values, depth values) is < `tolerance`.
  3. If the check fails, abort with a clear message: "Libraries do not agree — cannot benchmark fairly."
- For elastic alignment: check that both the warped curve and the warp function agree (amplitude + phase).
- For depth measures: check that depth ordering is preserved (Spearman rank correlation > 0.95).
- Run the agreement check on at least two datasets: one synthetic (known properties) and the real benchmark dataset.

**Warning signs:**
- Outputs from two libraries have correlations < 0.95 — algorithm disagreement.
- FPCA scores from scikit-fda and fdars have opposite signs systematically — missing sign alignment.
- Depth ordering disagrees on >10% of curves — different projection strategy or depth variant.

**Phase to address:** Harness-setup (implement the checker), baseline-capture (run it). Gate: the agreement gate must pass before any timing JSON is written.

---

### Pitfall A5: FPCA Eigenvector Sign Ambiguity Masking Real Numeric Differences

**What goes wrong:**
Eigenvectors have sign ambiguity — any valid eigenvector can be negated and remain valid. All FPCA implementations must apply a sign-canonicalization convention. scikit-fda uses the convention that the maximum absolute value component is positive; fdars may use a different convention (from the SVD sign in nalgebra); R's `pca.fd()` uses yet another. When comparing loadings or scores across libraries, sign-mismatched eigenvectors produce large apparent errors that are actually just convention differences — and real errors may be hidden behind "we expected sign differences."

More critically: if the agreement check uses absolute values everywhere to "handle" sign ambiguity without explicit alignment, it may accept a wrong result (e.g., opposite sign on all components) as "correct."

**Why it happens:**
Sign ambiguity is known in the FPCA literature and often dismissed with "sign is arbitrary." But if you are computing dot products or checking sign-dependent downstream quantities (regression coefficients, classification boundaries), the wrong sign produces wrong results.

**How to avoid:**
- Align signs explicitly: for each principal component k, compute `s_k = sign(dot(loadings_a_k, loadings_b_k))` and multiply `scores_b[:, k] *= s_k` before any comparison. This is a standard procedure documented in scikit-fda discussions.
- Never use `abs()` as a substitute for sign alignment — it hides real disagreements.
- After alignment, check both loadings and scores separately.
- Document the sign convention used by fdars in the benchmark README so readers understand the alignment step.

**Warning signs:**
- Agreement check passes with `abs()` but fails without it — sign alignment needed.
- Scores from two libraries are strongly negatively correlated (r ≈ -1) rather than positively correlated (r ≈ +1) — sign flip.
- Every other principal component is sign-flipped — indicates a systematic convention difference.

**Phase to address:** Harness-setup (implement sign alignment in agreement checker). Gate: correlation after alignment must be > 0.99 (not just |correlation| > 0.99).

---

### Pitfall A6: Measuring Input Preparation / IO / Conversion Overhead Instead of Algorithm Compute

**What goes wrong:**
A Python benchmark that times the full pipeline including:
- Constructing a `FDataGrid` object from a numpy array (scikit-fda's wrapper)
- Sorting or validating the evaluation points
- Python-to-C conversion of the array layout (row-major in numpy vs Fortran-order needed by LAPACK)
- Writing results to disk for later inspection

...is not measuring FPCA or elastic alignment — it is measuring data-format conversion and Python object construction.

Similarly, an R benchmark that includes `read.csv()` + `Data2fd()` + `pca.fd()` is measuring file IO + basis construction + FPCA, not just FPCA.

**Why it happens:**
Developers write "one function that does the whole workflow" and time that, which is natural but incorrect for algorithm comparison.

**How to avoid:**
- Structure the benchmark as setup + timed region:
  ```python
  # Setup (not timed)
  fd_grid = FDataGrid(data_matrix, grid_points)  # construct object once
  
  # Timed region only
  timer.start()
  result = FPCA(n_components=3).fit(fd_grid)
  timer.stop()
  ```
- In R: construct the `fd` object and basis system outside the `bench::mark()` call.
- In the fdars harness: construct the `FdMatrix` outside the criterion closure.
- For criterion.rs: put all data construction in the `setup` closure, not the benchmark closure.
- Exclude file IO from all timings.

**Warning signs:**
- Python FPCA time grows with file size even for the same matrix dimensions — IO in the loop.
- Benchmark timing varies with number of CPU cores even for single-threaded algorithm — object construction triggering parallel BLAS init.
- Tiny matrices (N=10) take same time as large matrices (N=1000) — overhead dominates.

**Phase to address:** Harness-setup. Gate: verify that setup overhead is < 1% of timed computation for each benchmark by measuring setup separately.

---

### Pitfall A7: Version Drift — Not Pinning Library Versions in the Harness

**What goes wrong:**
The benchmark harness runs against whatever `pip install scikit-fda` or `install.packages("fda")` installs at the time of execution. Different runs on different machines, or the same machine after a `pip upgrade`, use different library versions with different performance characteristics, different algorithms, or different defaults. The published results are not reproducible.

**Why it happens:**
Developers treat the benchmark as a one-time measurement, not a reproducible artifact. No version pinning is set up.

**How to avoid:**
- Commit a `requirements.txt` (or `pyproject.toml` with pinned deps) and `renv.lock` (for R) alongside the harness scripts.
- At the start of each benchmark run, emit a JSON environment block:
  ```json
  {
    "date": "2026-09-19",
    "python_version": "3.12.4",
    "scikit_fda_version": "0.10.1",
    "numpy_version": "1.26.4",
    "r_version": "4.4.1",
    "fda_version": "6.1.4",
    "fdars_core_version": "0.47.0",
    "fdars_git_sha": "abc123",
    "cpu": "AMD Ryzen 9 5900X",
    "os": "Linux 6.18.45 Manjaro",
    "blas_threads": 1
  }
  ```
- Store this environment block alongside every result JSON.
- Document in the harness README: "To reproduce, run `./setup.sh` which installs exact pinned versions."

**Warning signs:**
- Two runs of the "same" benchmark give different results with no code change — version drift.
- `pip show scikit-fda` shows a different version than what was used for the published numbers.
- R `packageVersion("fda")` differs between two machines.

**Phase to address:** Harness-setup. Gate: CI-style reproducibility check: re-run on the same machine 24 hours apart and verify results differ by < 5%.

---

### Pitfall A8: Tiny-Input Noise — Benchmarking N < 100 Curves and Calling It a Performance Story

**What goes wrong:**
Benchmarking FPCA on N=10 curves, M=50 points measures timer overhead and function-call dispatch more than algorithm complexity. For these sizes, a single Python function call overhead (1–5 μs) is comparable to the actual numeric computation. The criterion/timeit measurements have high coefficient of variation (>20%), and the "speedup" numbers have wide confidence intervals that overlap zero.

More importantly, tiny inputs miss the O(N·M²) vs O(N·M·K) vs O(N²·M²) differences that matter in practice. Reporting a 2x speedup at N=10 says nothing about N=1000.

**Why it happens:**
Developers use small inputs for quick iteration during harness development and forget to scale up before publishing.

**How to avoid:**
- Benchmark at multiple input sizes: small (N=50, M=100), medium (N=500, M=200), large (N=2000, M=500) at minimum.
- For the cross-language comparison, use the input sizes that reflect realistic FDA workloads — N=200 curves with M=150 points is a typical functional dataset.
- Report speedup as a function of N and M, not a single number.
- Accept only data points where criterion's relative standard deviation (RSD) < 5% as reliable.
- For elastic alignment (the O(N²·M²) anchor target), use N >= 100 to see the quadratic scaling.

**Warning signs:**
- Criterion reports "outlier fraction > 20%" — input is too small, timer noise dominates.
- Speedup varies by 5x between N=10 and N=100 for no algorithmic reason — noise floor effects.
- Python benchmark gives negative speedup (Rust slower) at N=10 but positive at N=1000 — cold-start overhead (A1) was included.

**Phase to address:** Baseline-capture. Gate: all published benchmark points have RSD < 5% and include at least 3 input sizes.

---

## Critical Pitfalls — Part B: Behavior-Preserving Rust Optimization

### Pitfall B1: Float Reassociation / Loop Reordering Silently Breaking Numeric Identity

**What goes wrong:**
Refactoring a loop from:
```rust
let mut sum = 0.0_f64;
for i in 0..n { sum += a[i] * b[i]; }
```
to:
```rust
let sum: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
```
may seem equivalent but is not bit-identical. The `sum()` call on an iterator uses the same sequential order as the manual loop — BUT if the optimizer later vectorizes the `.map().sum()` chain (e.g., by unrolling or using SIMD), it changes the accumulation order, changing the floating-point result.

Additionally, Rust 1.98 added `f64::algebraic_add()` / `algebraic_mul()`. If any optimization change accidentally uses algebraic ops (e.g., via a library update), results change. Even without algebraic ops, changing the accumulation order of `rayon::par_iter().sum()` from the sequential `.iter().sum()` changes results.

For fdars, the constraint is numeric-identical: the full 2942-test suite includes bit-exact assertions (e.g., `golden_co_cluster_*`, `svd_sign_fpca_*`) that detect any bit-level difference.

**Why it happens:**
The refactoring looks like a pure style change. The developer runs the test suite with a tolerance-based assertion and misses the bit-identity tests. Or the optimization is correct for debug builds but the release build applies LLVM optimizations that reorder operations.

**How to avoid:**
- After any refactoring, run the full test suite in release mode: `cargo test --release -p fdars-core --features linalg`.
- Specifically run the known bit-identity tests: `cargo test golden` to target the golden-output tests.
- If a refactoring changes a bit-identity test's result, it is not behavior-preserving — revert or find an approach that preserves the exact accumulation path.
- Do not change the order of floating-point operations in any hot path without measuring before and after against the test suite.
- Do not introduce `rayon::par_iter().sum()` to replace a sequential sum — use `par_iter().map(...).collect::<Vec<_>>()` then sequential sum if parallelism is needed.

**Warning signs:**
- Any bit-identity test in the suite fails after a refactoring that "shouldn't change behavior."
- `cargo test --release` passes but `cargo test` fails — debug vs release LLVM optimization divergence.
- A function that previously returned `0.0` exactly now returns `1e-17` — accumulation order changed.

**Phase to address:** Optimization phase. Gate: run `cargo test --release --features linalg` as part of every optimization commit's verification; specifically run the golden-output tests.

---

### Pitfall B2: Parallel Reduction Order Changing Results — Parallelizing a Sequential Sum

**What goes wrong:**
fdars uses the `maybe_par_*` macro family (parallel.rs) to gate rayon usage behind the `parallel` feature flag. The existing codebase carefully avoids non-determinism in rayon contexts by using per-thread seeded RNGs (`StdRng::seed_from_u64(seed + k)`). BUT: rayon's `parallel_sum()` over `f64` is non-deterministic in reduction order — the order depends on thread scheduling, which can vary between runs.

If an optimization introduces a `par_iter().sum()` to accelerate an inner product or norm computation, the result will differ by 1–2 ULP from the sequential version. The 2942-test suite will detect this. Additionally, the result may differ between the `parallel` and non-`parallel` feature builds, which the v0.43.0 determinism guardrail CI job is specifically designed to catch.

**Why it happens:**
Developers see that a sequential sum is a hot path, add `par_iter()`, the tests pass with tolerances, but the bit-identity tests fail or the CI determinism job fails. The issue is often not noticed until CI runs the `RAYON_NUM_THREADS=1` variant.

**How to avoid:**
- Do not replace a sequential `iter().sum()` with `par_iter().sum()` in fdars — it changes the bit-identical result.
- If parallelism is wanted in a reduction, use the pattern: parallel map into a `Vec`, then sequential sum: `par_iter().map(|x| ...).collect::<Vec<f64>>().iter().sum()`. This gives the same sequential accumulation path while parallelizing the per-element work.
- Verify after any parallelism change: `cargo test --features parallel` and `cargo test` (without parallel) must produce identical numeric results in all tests.
- The v0.43.0 determinism guardrail CI job (`RAYON_NUM_THREADS=1` repeated run) catches this — do not merge before CI is green.

**Warning signs:**
- `cargo test --features parallel` and `cargo test` produce different results in any test.
- A test that passes locally fails in CI with `RAYON_NUM_THREADS=1`.
- A golden-output test fails when `--features parallel` is added.

**Phase to address:** Optimization phase. Gate: run the determinism guardrail job (repeat + `RAYON_NUM_THREADS=1`) on all optimization commits.

---

### Pitfall B3: Overclaiming Speedups from Noisy Criterion Benchmarks

**What goes wrong:**
A developer runs criterion before and after an optimization and sees "35% faster." criterion reports this as statistically significant (p < 0.05). The developer publishes "35% speedup for elastic alignment." But:
- The machine was under load during one of the runs (background process, CPU frequency scaling).
- The sample size was too small (criterion defaults to 100 samples, which may not be enough for high-variance operations).
- The benchmark included warm-up but the warm-up time was too short for a cache-cold first run.
- The speedup is real but only at the specific input size (N=50) and reverses at N=500.

For the cross-language report to be credible, every claimed speedup must survive a methodology challenge.

**Why it happens:**
Criterion provides a p-value and a "X% faster" number, which look authoritative. Developers stop there and do not check whether the result is robust.

**How to avoid:**
- Run criterion benchmarks at least twice: once with `CARGO_FLAGS=--release` and once with CPU frequency pinned (`sudo cpupower frequency-set -g performance`).
- Check criterion's outlier report: if more than 5% of samples are flagged as outliers, re-run in a quieter environment.
- Report confidence intervals, not point estimates: "35% ± 8% faster" is honest; "35% faster" is not.
- For the before/after comparison: re-run the cross-language harness (not just criterion) to confirm the speedup is visible at the same scale.
- Only claim a speedup in the published report if it is > 10% and reproducible across at least 2 separate runs.

**Warning signs:**
- Criterion reports different speedup magnitudes on re-run (> 10% difference in the claimed change).
- High outlier fraction (> 5%) in criterion output for the benchmarked function.
- Speedup is large at N=50 but vanishes at N=500 — noise floor effect from tiny input (see A8).

**Phase to address:** Optimization phase and report phase. Gate: each claimed speedup must have CI < ±5% of point estimate and be confirmed by the cross-language harness re-run.

---

### Pitfall B4: Accidental Numeric Drift When the Test Suite Uses Tolerances (Not Bit Identity)

**What goes wrong:**
The 2942-test suite includes both bit-identity tests (golden-output, SVD sign) and tolerance-based tests (e.g., `assert_relative_eq!(..., epsilon = 1e-8)`). An optimization that changes results by 1e-12 passes all tolerance-based tests but may still break downstream behavior in ways that are hard to detect — for example, the sign of an eigenvector may flip due to a 1e-12 difference in a singular value, causing `svd_sign_fpca_*` to fail.

More subtly: an optimization that is "numerically equivalent" in double precision may produce different results in the `--features serde` build if serialization round-trips truncate precision.

**Why it happens:**
Developers see all tests pass (tolerance-based) and conclude the change is numeric-preserving. They do not explicitly run the bit-identity subset.

**How to avoid:**
- Identify all bit-identity tests in the suite: `grep -r "bit_identical\|golden\|assert_eq.*f64\|assert_eq.*output" tests/ fdars-core/src/`.
- Run this subset explicitly after every optimization: `cargo test golden && cargo test svd_sign && cargo test equivalence`.
- For any test that changes from "pass" to "fail" — even a tolerance-based test — treat it as a potential sign that the optimization is not behavior-preserving.
- After optimization, compare the before/after outputs of the full test suite at debug level: `cargo test 2>&1 | diff before.log -` to catch any change in output.

**Warning signs:**
- Any test changes from "ok" to "FAILED" after an optimization, even one that was previously tolerance-based.
- `cargo test --release` gives different test results than `cargo test` (debug) — LLVM is changing the result.
- The `--features serde` build gives different serialized values than before the optimization.

**Phase to address:** Optimization phase. Gate: run `cargo test --release --features linalg` AND `cargo test --features linalg` (debug) on the same commit; both must have zero failures.

---

### Pitfall B5: Regressing the `--features serde` / WASM / No-`linalg` Builds

**What goes wrong:**
fdars-core has three significant feature-gated build variants that all public functions must support:
1. Default (no features) — `parallel` enabled by default, no `linalg` (no faer)
2. `--features linalg` — enables faer for Cholesky/ridge regression; requires Rust 1.84+
3. `--features serde` — adds serde derives to all public types
4. `--target wasm32-unknown-unknown` — WASM build (uses `js` feature, no std threading)

An optimization that introduces a new `use faer::...` call without gating it behind `#[cfg(feature = "linalg")]` breaks the default build. An optimization that adds a new field to a public struct (even a private cache) breaks `--features serde` if the type is not `Serialize`. An optimization that uses `std::thread` instead of rayon breaks WASM.

**Why it happens:**
The optimization is developed and tested with `--features linalg` (the developer's usual workflow) and only runs `cargo clippy --all-targets --features linalg,parallel`. The default build and WASM build are not checked.

**How to avoid:**
- After any optimization that touches public types, hot paths, or `use` statements, run all four build variants:
  ```bash
  cargo build -p fdars-core                              # default features
  cargo build -p fdars-core --features linalg
  cargo build -p fdars-core --features serde
  cargo build -p fdars-core --features linalg,serde,parallel
  cargo build -p fdars-core --target wasm32-unknown-unknown --features js
  ```
- Use the project's CI gate: `cargo clippy --all-targets --features linalg,parallel -- -D warnings`. The `--all-targets` flag catches test + bench + example code.
- If a new dependency is needed, gate it behind the appropriate feature. The constraint for v0.47.0 is: **no new `Cargo.toml` dependency** — all optimizations must use existing crate dependencies.

**Warning signs:**
- `cargo build -p fdars-core` (no features) fails after optimization.
- `cargo build --features serde` fails with "the trait `Serialize` is not implemented for `X`".
- Any WASM-related CI step fails.
- `cargo clippy --all-targets` finds warnings not present in `cargo clippy`.

**Phase to address:** Optimization phase. Gate: all five build variants must compile clean before any optimization commit is finalized.

---

### Pitfall B6: Breaking Seeded-RNG Determinism by Changing Iteration Order

**What goes wrong:**
fdars uses the pattern `StdRng::seed_from_u64(seed + k as u64)` in parallel loops (where k is the loop index) to ensure each thread gets a different but deterministic seed. If an optimization reorders the loop (e.g., by changing the chunk size, the rayon scheduling strategy, or the iteration order over a collection), the association between `k` and the work item changes, producing different random sequences and different output — even though the seed was set.

This can manifest as: GMM clustering produces different cluster assignments after an optimization, or elastic bootstrap CIs change slightly.

**Why it happens:**
The developer optimizes the outer loop structure (e.g., changes from `chunks(n/p)` to `chunks_exact(...)`) without realizing it changes which iteration index k corresponds to which work item.

**How to avoid:**
- Do not change the outer loop structure (chunk size, iteration order, collection type) in any code that uses the `seed + k` RNG pattern without re-verifying RNG determinism.
- After any loop-structure change, run the tests that are sensitive to RNG seeding: `cargo test determinism && cargo test gmm && cargo test elastic`.
- The v0.43.0 determinism guardrail (repeat-run CI job) catches this automatically.
- Document which loops use `seed + k` RNG — these are the sensitive points.

**Warning signs:**
- A test involving GMM, elastic registration, or bootstrap CI fails after a loop-structure change.
- `cargo test` gives different results on two runs (non-determinism introduced).
- The determinism guardrail CI job fails (repeat run gives different output).

**Phase to address:** Optimization phase. Gate: the determinism guardrail must be green on all optimization commits.

---

### Pitfall B7: Disk / `/tmp` Pressure on Full Build + Test Cycles

**What goes wrong:**
This is documented in MEMORY.md and remains a hazard for v0.47.0:
1. The optimization phase will add criterion benchmarks (likely adding new `[[bench]]` entries), increasing `target/debug/` artifact size.
2. Full `cargo test --release` (needed to verify bit-identity after optimization) builds both debug and release artifacts simultaneously.
3. `/tmp` exhaustion causes pre-commit hook failures with "No space left on device."
4. `target/` filling `/home` causes link failures at the `cargo test` stage with "linking with cc failed."

The benchmark harness also generates criterion HTML reports under `target/criterion/` which accumulate.

**How to avoid:**
- Before any full build: `df -h /tmp /home` — abort and clean if either is < 2GB.
- Free target incrementals before optimization work: `rm -rf target/debug/{incremental,examples}`.
- Use `cargo test -p fdars-core` (package-scoped) not workspace-wide `cargo test` to limit artifact accumulation.
- Clean criterion HTML reports periodically: `rm -rf target/criterion/`.
- If commit hooks fail with "No space left": free `/tmp` first, then retry.

**Warning signs:**
- "No space left on device" during `cargo test` or pre-commit hooks.
- "linking with cc failed" on a green clippy check (not a code error — disk full).
- `cargo bench` fails partway through on the second or third benchmark.

**Phase to address:** All phases. Gate: disk check before each full build in the phase execution checklist.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Single-threaded-only benchmark comparison (no multi-thread variant) | Simpler harness | Report understates fdars parallelism advantage; stakeholders ask "what about multi-core?" | Acceptable for MVP harness if labeled clearly; add multi-thread comparison in report phase |
| Using `time.time()` instead of `timeit` for Python benchmarks | Easier to write | Includes cold-start, GC, and OS scheduler noise; numbers not reproducible | Never for published numbers |
| Skipping numeric-agreement check and just timing | Saves harness-setup time | Published speedup may be meaningless if algorithms differ | Never — agreement check is mandatory |
| Running criterion at default sample size (100) without checking RSD | Fast iteration | May report spurious speedups with CI > 10% | Acceptable during development; not for published report |
| Not pinning BLAS thread count | Simpler setup | Python may run on 16 threads vs Rust on 1 — comparison is misleading | Never for published numbers |
| Skipping `--features serde` + WASM build check after optimization | Saves 5 minutes | Silent breakage discovered by users after publish | Never — add to the optimization phase checklist |
| Using `debug` build for criterion benchmarks | Faster iteration | LLVM optimizations in release may change numeric results and performance | Never for published criterion results — always use release |

---

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| scikit-fda FPCA vs fdars fdata_to_pc_1d | Comparing raw score matrices without sign alignment | Apply sign alignment: multiply each component by sign(dot(fdars_loading, skfda_loading)) before comparing |
| R `pca.fd()` vs fdars FPCA | Using R's `ncomp` default (all components) vs fdars `ncomp` | Explicitly set `ncomp=3` (or same value) in both; R default computes all components (slow) |
| numpy BLAS threading | Forgetting to set `OMP_NUM_THREADS=1` | Set all four vars: OMP, OPENBLAS, MKL, BLIS before the Python process starts |
| criterion warm-up for large allocations | Warm-up time too short (default 3s) — matrix allocations not yet JIT-optimized | Set `.warm_up_time(Duration::from_secs(10))` for benchmarks involving large matrices |
| R `bench::mark()` vs `microbenchmark()` | Using `microbenchmark()` which does not run GC between trials | Prefer `bench::mark()` which runs GC before timing and reports gc counts |
| fdars `FdMatrix` column-major vs numpy row-major | Copying data into numpy in wrong order | Ensure the benchmark constructs numpy arrays in the same logical order as fdars — transpose if needed |
| Python process isolation | All benchmarks run in one process — GC from one benchmark pollutes the next | Run each benchmark function in a separate process (pyperf process isolation) or at least call `gc.collect()` between runs |

---

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Benchmarking with `debug` build Rust vs optimized Python BLAS | Rust looks slower than it is | Always use `cargo bench` (release) or `cargo test --release` | Every benchmark run if not using `--release` |
| Including `FdMatrix` construction in the timed region | fdars looks slower than Python by 2–5x for small N because object construction is relatively expensive | Put data construction in criterion setup closure, not benchmark closure | Especially bad at N < 100 |
| Reporting single-run speedups from criterion | Reported CI is wide but only point estimate is reported | Always report the full confidence interval from criterion JSON output | When CPU is under variable load |
| Calling `elastic_align_pair` once per benchmark loop (timing includes return value copy) | Timing includes result struct allocation, not just compute | Use `black_box()` on input, not output; or benchmark in-place variant if available | For functions returning large structs |

---

## "Looks Done But Isn't" Checklist

- [ ] **BLAS thread count pinned:** verify `OMP_NUM_THREADS=1` is set and Python/R actually uses 1 thread — print `numpy.linalg.svd` timing at N=1 vs N=10 to confirm linear (not superlinear) scaling.
- [ ] **Numeric agreement gate passes:** the agreement check runs on the reference dataset before any timing JSON is written, not just during development.
- [ ] **Sign alignment implemented:** FPCA and elastic comparison functions align eigenvector signs before computing agreement metrics.
- [ ] **Python warm-up verified:** first-call vs steady-state difference is < 5% for each benchmark function.
- [ ] **Version pinning committed:** `requirements.txt` and `renv.lock` are committed to the harness repo with exact versions.
- [ ] **Environment JSON emitted:** every result file includes CPU, OS, library versions, and thread counts.
- [ ] **Multiple input sizes:** each benchmark covers at least 3 input sizes (small, medium, large).
- [ ] **Criterion release build:** all criterion runs use `cargo bench` or `cargo build --release`.
- [ ] **All feature builds compile:** after optimization, all five build variants (default, linalg, serde, linalg+serde+parallel, WASM) compile without warnings.
- [ ] **Bit-identity tests pass in release mode:** `cargo test --release --features linalg` runs golden + svd_sign + equivalence tests and all pass.
- [ ] **Determinism guardrail green:** optimization commits are verified by the v0.43.0 determinism CI job (repeat + RAYON_NUM_THREADS=1).
- [ ] **Disk check before full build:** `df -h /tmp /home` run before any full `cargo test --release` or `cargo bench`.

---

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Cold-start overhead in Python timings | LOW | Re-run with proper warm-up; discard first-call data points |
| BLAS thread count not pinned (published wrong numbers) | HIGH | Re-run entire baseline with thread count pinned; update report |
| Numeric agreement check not implemented | MEDIUM | Implement checker, re-run baseline, potentially discover algorithm-variant issue requiring harness redesign |
| Bit-identity test fails after optimization | MEDIUM | Revert to prior accumulation order; profile to find a different approach that preserves the exact float path |
| Parallel reduction order change breaks determinism | MEDIUM | Replace `par_iter().sum()` with parallel-map + sequential-sum pattern; re-run test suite |
| `--features serde` build broken by optimization | LOW | Add missing `#[cfg_attr(feature = "serde", derive(...))]` to new types; re-run build |
| WASM build broken by optimization | LOW-MEDIUM | Remove std-only dependency; replace with wasm-compatible alternative already in crate |
| Disk pressure blocks build | MEDIUM | `rm -rf target/debug/{incremental,examples}` + free /tmp; rebuild |
| Criterion speedup not reproducible | LOW | Re-run in controlled environment; widen CI reporting |
| Version drift in results | HIGH | Re-run with pinned versions from requirements.txt/renv.lock |

---

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Cold-start / import overhead (A1) | Harness-setup (Phase 112) | First-call vs steady-state < 5% difference for each Python/R function |
| BLAS thread-count mismatch (A2) | Harness-setup (Phase 112) | Result JSON includes `blas_threads=1`; timing scales linearly with N on both sides |
| Apples-to-oranges algorithm variant (A3) | Harness-setup + baseline-capture | Parameter config documented; algorithm names match the same variant |
| Missing numeric-agreement gate (A4) | Harness-setup (Phase 112) | Agreement gate runs before any timing JSON is written |
| FPCA sign ambiguity (A5) | Harness-setup (Phase 112) | Correlation after sign alignment > 0.99 on reference dataset |
| Input-prep / IO in timed region (A6) | Harness-setup (Phase 112) | Setup overhead measured separately and is < 1% of compute time |
| Version drift (A7) | Harness-setup (Phase 112) | `requirements.txt` and `renv.lock` committed; environment JSON emitted per run |
| Tiny-input noise (A8) | Baseline-capture | RSD < 5%; at least 3 input sizes; elastic target uses N >= 100 |
| Float reassociation drift (B1) | Optimization phase | `cargo test --release --features linalg` passes all golden + bit-identity tests |
| Parallel reduction order change (B2) | Optimization phase | Determinism guardrail CI job green on all optimization commits |
| Noisy criterion speedup claims (B3) | Optimization phase + report | CI < ±5% of point estimate; confirmed by cross-language harness re-run |
| Tolerance-based tests masking drift (B4) | Optimization phase | Explicitly run bit-identity test subset after each optimization commit |
| Feature-build regressions (B5) | Optimization phase | All five build variants compile clean after each optimization commit |
| RNG determinism from loop reordering (B6) | Optimization phase | Determinism guardrail CI job; `cargo test determinism` passes |
| Disk / /tmp pressure (B7) | All phases | `df -h /tmp /home` before each full build; cleanup script documented |

---

## Sources

- `fdars-core/src/parallel.rs` — `maybe_par_*` macro family, per-thread seed pattern
- `fdars-core/src/matrix.rs` — `FdMatrix` column-major layout, row methods
- Project MEMORY.md — disk pressure, executor stall, `--no-verify` fmt drift, `--all-targets` CI gate, determinism guardrail, golden flake root cause
- v0.43.0 milestone — determinism guardrail CI job design (repeat + RAYON_NUM_THREADS=1)
- Rayon docs (docs.rs/rayon) — ParallelIterator::sum does not guarantee reduction order
- Criterion.rs docs (bheisler.github.io/criterion.rs) — warm-up, outlier detection, noise threshold
- Rust RFC 3514 float semantics — algebraic float operators, IEEE-754 strictness default
- SuperFastPython (superfastpython.com) — NumPy BLAS thread control via OMP/OPENBLAS/MKL env vars
- scikit-fda FPCA discussion #512 (github.com/GAA-UAM/scikit-fda) — sign ambiguity acknowledgment
- "Are We Fast Yet?" paper — cross-language benchmark fairness methodology
- "Impacts of floating-point non-associativity on reproducibility" (arxiv 2408.05148) — parallel reduction order effects
- Tweag blog "Improving Criterion.rs" (tweag.io) — statistical pitfalls in Criterion usage

---
*Pitfalls research for: Cross-language numerical benchmarking + behavior-preserving Rust optimization — fdars-core v0.47.0*
*Researched: 2026-09-19*
