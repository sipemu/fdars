# Stack Research: Rust Performance Audit Toolkit for fdars

**Domain:** CPU-bound numerical Rust library — performance profiling and benchmarking
**Researched:** 2026-08-07
**Confidence:** MEDIUM (versions verified against docs.rs and crates.io; tool behaviors confirmed against official docs and Rust Performance Book)

---

## The Core Decision Tree

Before picking tools, pick the question you are answering:

| Question | Right tool |
|----------|-----------|
| "Where does wall-clock time go?" | criterion + flamegraph/samply |
| "Which commit regressed instruction count?" | iai-callgrind |
| "How many allocations does this hot path do?" | dhat-rs (in-process) or heaptrack (external) |
| "Does parallelism actually help?" | criterion with `RAYON_NUM_THREADS` sweep |
| "Is faer faster than nalgebra for this SVD?" | criterion microbenchmark comparing both |

---

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| criterion | 0.5.1 (current in repo) | Statistical wall-clock microbenchmarking | Already in use; 8 existing bench files; API is stable and suitable. The repo uses `criterion = "0.5"` pinned — do not upgrade to 0.7/0.8 mid-audit (breaking `black_box` behavior changed). |
| iai-callgrind | 0.16.1 | Deterministic instruction-count benchmarks for CI regression tracking | Reports Ir (instruction refs), cache misses, branch mispredicts. Produces identical numbers across machines — suitable for tracking in CI where wall-clock is noisy. Requires Valgrind on Linux. |
| flamegraph (cargo-flamegraph) | 0.6.13 | CPU sampling flamegraph from `perf` on Linux | Zero code changes required; run against `cargo bench` or a standalone binary. The fastest way to find the top 3 hot functions in any module. |
| samply | 0.13.x | Interactive CPU profiler (Firefox Profiler UI) | Better interactivity than flamegraph; zoom into specific time windows; useful when the flamegraph is ambiguous. On Linux uses `perf_event_open`. |
| dhat | 0.3.3 | In-process heap allocation profiling | Documents the exact allocation call stack. Key use: confirming the FdMatrix→DMatrix round-trip copies. The `testing()` mode lets you assert `total_blocks == N` in integration tests. |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| heaptrack | system package (KDE/heaptrack) | External heap profiler — no code changes | Use when dhat-rs is inconvenient (e.g. profiling an example binary without modifying it). Requires `debug = true` in release profile. Lower overhead than Valgrind massif. |
| perf (linux-tools) | kernel-matched | Hardware performance counter sampling | Use with `perf stat -d` to see L1/L2 cache misses alongside CPU cycles — critical when optimizing column-major access patterns in FdMatrix. |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| `cargo bench --features linalg` | Run criterion benchmarks with faer enabled | Always benchmark with `linalg` feature to measure realistic paths; faer's Cholesky/SVD are only in-play when `linalg` is active |
| `cargo bench -- --save-baseline <name>` | Save a named baseline for comparison | Compare before/after with `--baseline <name>`; stored in `target/criterion/` |
| `RAYON_NUM_THREADS=1 cargo bench` | Disable parallelism to isolate single-core cost | Essential for separating algorithmic cost from parallelism benefit |
| `RAYON_NUM_THREADS=N cargo bench` | Sweep thread counts to measure scaling | Use N=1,2,4,8 to build a scaling curve per benchmark |

---

## Installation

```bash
# flamegraph (cargo-flamegraph) — system perf + Rust binary
cargo install flamegraph

# samply — interactive sampling profiler
cargo install samply

# iai-callgrind — also requires the matching runner binary
cargo install iai-callgrind-runner --version 0.16.1
# Add to Cargo.toml dev-dependencies:
# iai-callgrind = "0.16.1"

# Valgrind (required for iai-callgrind and dhat on Linux)
# Manjaro/Arch:
sudo pacman -S valgrind

# heaptrack — external allocation profiler
sudo pacman -S heaptrack   # or build from KDE/heaptrack on GitHub

# Linux perf (already present on Manjaro in most configs; check kernel tools)
sudo pacman -S perf
# lower paranoid level if needed:
echo -1 | sudo tee /proc/sys/kernel/perf_event_paranoid
```

Add dhat-rs as a dev-dependency with a feature flag (no production code changes):

```toml
# fdars-core/Cargo.toml
[dev-dependencies]
dhat = "0.3"

[features]
dhat-heap = []   # gate with feature flag so it never enters normal CI
```

---

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| criterion 0.5 (current pin) | criterion 0.7/0.8 | Upgrade after the audit is complete — 0.7 bumps MSRV to 1.80, which is fine, but don't break the bench harness mid-audit |
| iai-callgrind | original `iai` crate | Never — `iai` is unmaintained; iai-callgrind is the maintained successor |
| dhat-rs | Valgrind DHAT tool | DHAT tool is Linux-only; dhat-rs works on all platforms and allows assertion tests |
| flamegraph / samply | cargo-instruments | cargo-instruments is macOS-only (Xcode Instruments). Not applicable on this Linux dev environment |
| heaptrack | Valgrind massif | heaptrack has significantly lower overhead and better GUI; massif is appropriate for long-running processes only |
| perf + flamegraph | VTune, Tracy | VTune requires Intel license; Tracy requires code instrumentation. perf+flamegraph is zero-friction for library code |

---

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `cargo bench` in debug mode | Debug builds have no optimization — numbers are meaningless for hot-path analysis | Always run `cargo bench` (which uses `opt-level = 3` profile) |
| Wall-clock criterion in CI for regression tracking | CI machines vary in load, clock frequency, throttling — false positives are frequent | iai-callgrind for regression gates; criterion only for local understanding |
| Wrapping only the function output in `black_box` | Compiler can still optimize away the input computation | Wrap both input and output: `b.iter(|| f(black_box(&data)))` — inputs that are const-foldable must be black-boxed too |
| Single input-size benchmarks | Hides O(n²) vs O(n log n) behavior | Always sweep at least 3 values of N (curves) and M (grid points) using `BenchmarkId` |
| Benchmark the parallel feature disabled (no `--features parallel`) | Misses rayon overhead and contention cost | Benchmark with `parallel` feature (the default) and disable it separately for comparison |

---

## Criterion Best Practices for FDA Workloads

### black_box and throughput

```rust
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

fn bench_fpca(c: &mut Criterion) {
    let mut group = c.benchmark_group("fdata_to_pc_1d");
    // Set Throughput so criterion reports curves/sec alongside ns/iter
    // Use Elements for FDA: "number of curves" is the natural unit
    for &n in &[50, 200, 500, 1000] {
        for &m in &[50, 100, 200] {
            let (data, argvals) = generate_data(n, m);
            let label = format!("n{n}_m{m}");
            group.throughput(Throughput::Elements(n as u64));
            group.bench_with_input(BenchmarkId::new("params", &label), &(n, m), |b, _| {
                // black_box the input reference — prevents the compiler folding constant data
                b.iter(|| fdata_to_pc_1d(black_box(&data), 5, black_box(&argvals)));
            });
        }
    }
    group.finish();
}
```

### SamplingMode for slow benchmarks

Elastic alignment (`karcher_mean`, `elastic_self_distance_matrix`) at n=50, m=200 can take seconds per iteration. Switch to flat sampling:

```rust
group.sampling_mode(criterion::SamplingMode::Flat);
group.sample_size(20);           // fewer samples, each run counts
group.measurement_time(std::time::Duration::from_secs(30));
```

### iter_batched for allocating setup

When the benchmark itself includes setup allocation (e.g. cloning FdMatrix), use `iter_batched` to keep setup outside the measurement window:

```rust
b.iter_batched(
    || generate_data(n, m),                      // setup: runs outside measurement
    |(data, argvals)| fdata_to_pc_1d(black_box(&data), 5, black_box(&argvals)),
    criterion::BatchSize::LargeInput,
);
```

### Feature flag discipline

Always run benchmarks with the features that match production:

```bash
# Standard: parallel enabled, linalg enabled (faer/anofox paths active)
cargo bench --features linalg -- <bench_name>

# Sequential baseline
RAYON_NUM_THREADS=1 cargo bench --features linalg -- <bench_name>

# Without linalg (nalgebra-only paths)
cargo bench -- <bench_name>
```

---

## Profiling Workflow for Hot-Path Discovery

### Phase 1: Get the flamegraph first (15 min)

Flamegraph gives you the call-stack time distribution with zero setup. On Linux with perf:

```bash
# Profile the regression benchmark binary
CARGO_PROFILE_BENCH_DEBUG=true cargo bench --features linalg \
    --bench regression_benchmarks --no-run
# Find the compiled binary
BENCH_BIN=$(ls target/release/deps/regression_benchmarks-* | head -1)

# Record with perf, then generate flamegraph
cargo flamegraph --bench regression_benchmarks --features linalg \
    -- --bench --profile-time 10
# Opens target/flamegraph.svg
```

Interpret: wide boxes = hot functions, tall stacks = deep call chains. Prioritize boxes that are nalgebra SVD, matrix allocation, or clone.

### Phase 2: samply for interactive drill-down (optional)

When flamegraph shows "nalgebra::linalg::svd" as 60% of runtime but you want to see which call sites invoke it:

```bash
samply record cargo bench --features linalg \
    --bench regression_benchmarks -- --bench --profile-time 10
# Opens Firefox Profiler in browser
```

samply is recommended over perf-only when you need to filter to a specific benchmark iteration or zoom into a specific time window.

### Phase 3: perf stat for cache behavior

Column-major FdMatrix accesses columns sequentially (good), but row-wise access (e.g. `mat.row(i)`) copies — a potential L2 miss source:

```bash
perf stat -d cargo bench --features linalg \
    --bench matrix_benchmarks -- --bench
# Look for: LLC-load-misses, L1-dcache-load-misses
```

---

## Allocation Profiling Workflow

### dhat-rs for specific hot-path confirmation

The documented anti-pattern — `FdMatrix::to_dmatrix()` called inside SVD loops — should be confirmed with dhat before claiming it matters:

```rust
// In an integration test (separate process, required by dhat):
#[test]
fn count_svd_allocations() {
    let _profiler = dhat::Profiler::builder().testing().build();
    let (data, argvals) = generate_data(200, 100);
    let _ = fdata_to_pc_1d(&data, 5, &argvals);
    let stats = dhat::HeapStats::get();
    // This will fail if allocations are excessive — document the baseline, then track
    println!("Total heap blocks: {}", stats.total_blocks);
    println!("Peak heap bytes: {}", stats.peak_bytes);
}
```

Run with: `cargo test --features dhat-heap count_svd_allocations`

View heap profile in DHAT viewer: `npx dhat-viewer dhat-heap.json` (or use the official viewer at nnethercote.github.io/dh_view/dh_view.html).

### heaptrack for external binary profiling

When profiling an example rather than a test:

```bash
heaptrack cargo run --features linalg --example regression_example --release
heaptrack_gui heaptrack.cargo.*.zst   # opens GUI showing allocation flame graph
```

---

## nalgebra vs faer Performance Reasoning

**The verified benchmark data** (from the faer codebase benchmarks, confirmed via web search):

| Operation | Matrix size | nalgebra | faer | Speedup |
|-----------|-------------|----------|------|---------|
| SVD (square) | 1024×1024 | 347 ms | 43 ms | ~8× |
| SVD (rect) | 4096×1024 | 3950 ms | 298 ms | ~13× |
| Cholesky | 1024×1024 | measured | significantly faster | varies |

**What this means for fdars:**

- `fdata_to_pc_1d` uses nalgebra SVD. For n=200 curves, m=100 grid points, the matrix is 200×100 — small enough that the 8× gap may be only ~5 ms absolute. Measure before declaring it a problem.
- The `linalg` feature gates faer's Cholesky for ridge regression. For n=200 this is already the fast path. SVD remains in nalgebra.
- **Interop cost**: converting `FdMatrix` → `nalgebra::DMatrix` (already in-use via `to_dmatrix()`) requires one full allocation + copy. Calling it inside a loop (e.g., cross-validation) is the documented anti-pattern. A faer SVD path would require an equally expensive conversion to `faer::Mat`. The interop cost can negate the faer speedup for small n.
- **Decision rule**: If the flamegraph shows nalgebra SVD consuming >30% of runtime at problem sizes used in real workloads, benchmark a faer-backed SVD path as a separate criterion bench group. Otherwise, the round-trip copy cost may dominate.

---

## Rayon Parallelism Measurement

### Thread scaling curve

Run the same benchmark at 1, 2, 4, N threads to build a scaling curve. If speedup plateaus at 2 threads for an 8-core machine, the bottleneck is either synchronization overhead, shared cache eviction, or tasks too small to parallelize efficiently.

```bash
for T in 1 2 4 8; do
    echo "=== RAYON_NUM_THREADS=$T ===" 
    RAYON_NUM_THREADS=$T cargo bench --features linalg \
        --bench alignment_benchmarks -- elastic_self_distance_matrix
done
```

### Identifying contention vs compute

If `perf stat -d` shows a high `sched:sched_migrate_task` count (via `perf stat -e sched:*`), threads are migrating between CPU cores and invalidating L1/L2 caches — a Rayon work-stealing artifact documented in the 2024 Gendignoux post.

```bash
perf stat -e sched:sched_migrate_task,sched:sched_switch \
    RAYON_NUM_THREADS=8 cargo bench --features linalg \
    --bench alignment_benchmarks -- --bench
```

A high migrate count relative to work items suggests the parallel grain size is too small (Rayon is spending more time on work-stealing than computation). The fix is coarser-grained parallel chunks, not a tool concern — but the measurement here validates the hypothesis.

### Sequential vs parallel comparison as a benchmark

Add an explicit comparison bench for algorithms that have both paths:

```rust
group.bench_function("sequential", |b| {
    b.iter(|| {
        // call with RAYON_NUM_THREADS=1 semantics by using iter directly
        // or expose an internal sequential helper for benchmarking
    })
});
group.bench_function("parallel", |b| {
    b.iter(|| elastic_self_distance_matrix(black_box(&mat), black_box(&argvals), 0.0))
});
```

---

## iai-callgrind for CI Regression Tracking

iai-callgrind is the only tool in this list that produces identical numbers on different machines (because Valgrind simulates execution deterministically). Use it for CI regression gates on the top 3-5 hottest functions discovered by criterion.

### Minimal setup

```toml
# Cargo.toml dev-dependencies
iai-callgrind = "0.16.1"
```

```bash
# Cargo.toml [[bench]] entry
[[bench]]
name = "iai_fpca"
harness = false
```

```rust
// benches/iai_fpca.rs
use iai_callgrind::{library_benchmark, library_benchmark_group, main, LibraryBenchmarkConfig};

#[library_benchmark]
fn fpca_n200_m100() -> fdars_core::regression::FpcaResult {
    let (data, argvals) = generate_data(200, 100);
    fdars_core::regression::fdata_to_pc_1d(&data, 5, &argvals).unwrap()
}

library_benchmark_group!(name = fpca_group; benchmarks = fpca_n200_m100);
main!(library_benchmark_groups = fpca_group);
```

```bash
# Install runner (version must match the library version)
cargo install iai-callgrind-runner --version 0.16.1
cargo bench --bench iai_fpca --features linalg
```

**Key metrics to watch:**
- `Ir` (instruction references) — primary regression signal
- `Dr`/`Dw` (data reads/writes) — memory pressure indicator
- `I1mr`/`DLmr`/`DLmw` — cache miss rates; high values suggest poor memory locality

**When to add an iai bench vs criterion bench:** Add iai benches only for functions confirmed as hot by criterion + flamegraph. iai setup has higher overhead per benchmark (Valgrind is slow) — keep the iai suite small (5-10 functions at most).

---

## Recommended Measurement Plan for This Audit

Priority-ordered — complete each phase before the next:

### Phase A: Establish baselines (1-2 days)

1. Run all 9 existing criterion bench groups with `--features linalg` and save a named baseline:
   ```bash
   cargo bench --features linalg -- --save-baseline pre-audit
   ```
2. For each group, note the highest-latency benchmark (these are the candidates for profiling).
3. Run `RAYON_NUM_THREADS=1` for the same group — compute the parallel speedup ratio for each bench.

### Phase B: Hot-path flamegraphs (1-2 days)

4. For the top 3 slowest bench groups (likely: alignment, regression/FPCA, depth), generate flamegraphs with `cargo flamegraph --bench <name> --features linalg`.
5. Document top-3 hot functions per flamegraph — is it nalgebra SVD? clone? to_dmatrix?

### Phase C: Allocation audit (1 day)

6. Add dhat-rs as a dev-dep (feature-gated). Write one integration test per suspected allocation hotspot:
   - `fdata_to_pc_1d` (SVD path) — counts `total_blocks` and `peak_bytes`
   - `elastic_self_distance_matrix` — likely allocates per-pair distance
7. Compare allocations with and without the `parallel` feature — rayon may introduce per-task clones.

### Phase D: nalgebra vs faer comparison (optional, only if Phase B shows SVD as hot)

8. Write a dedicated bench that calls nalgebra SVD vs faer SVD on the same matrix sizes found in Phase A (n=50,200,500; m=50,100,200). Report absolute ms and the crossover point where faer wins after accounting for conversion cost.

### Phase E: CI regression baseline (after Phase A-C)

9. Add iai-callgrind benches for the top 3 functions identified in Phase B. These become the regression-tracking suite for future milestones.

---

## Stack Patterns by Variant

**If benchmarking a function that involves RNG** (e.g., GMM clustering, random-projection depth):
- Use a fixed seed: `StdRng::seed_from_u64(42)` in the bench setup
- Wrap seed in `black_box` only if seed is a variable in the comparison; otherwise keep it constant

**If benchmarking an O(n²) pairwise operation** (e.g., `elastic_self_distance_matrix`):
- Use `SamplingMode::Flat` and small sample counts (10-20)
- Sweep n from 10 to 50 — extrapolate, do not run n=500 in CI

**If comparing sequential vs parallel paths:**
- Use separate `BenchmarkGroup` entries, not separate functions — groups share configuration and produce comparable HTML charts

**If the bench includes a large allocation in setup** (e.g., generating `FdMatrix` in each iteration):
- Use `iter_batched` with `BatchSize::LargeInput` so setup is outside the measured window

---

## Version Compatibility

| Package | Compatible With | Notes |
|---------|-----------------|-------|
| criterion 0.5.1 | Rust 1.70+ | Current in fdars; `black_box` delegates to `std::hint::black_box` |
| criterion 0.7.0 | Rust 1.80+ | Bumps MSRV; upgrade after audit completes if desired |
| iai-callgrind 0.16.1 | Rust 1.74.1+; Valgrind 3.14+ | Runner binary version must exactly match library version |
| dhat 0.3.3 | Rust 1.60+ | No MSRV conflict with fdars (MSRV 1.81) |
| flamegraph 0.6.13 | Rust stable; Linux kernel 4.1+ for perf | Requires `perf_event_paranoid ≤ 1` or root |
| samply 0.13.x | Linux (perf_event_open) + macOS | No root required on Linux when built with `CAP_PERFMON` |

---

## Sources

- criterion docs.rs (version 0.8.2 as of 2026-06 per fetched docs.rs) — throughput API, SamplingMode, BenchmarkGroup
- [Criterion.rs Advanced Configuration](https://bheisler.github.io/criterion.rs/book/user_guide/advanced_configuration.html) — SamplingMode, throughput usage
- iai-callgrind lib.rs (version 0.16.1, July 2025) — metrics reported, MSRV, setup/teardown API
- [Rust Performance Book — Profiling](https://nnethercote.github.io/perf-book/profiling.html) — tool recommendations per platform
- [Rust Performance Book — Heap Allocations](https://nnethercote.github.io/perf-book/heap-allocations.html) — dhat vs heaptrack guidance
- dhat docs.rs (version 0.3.3, June 2026) — `Profiler::builder().testing()`, `HeapStats::get()`
- flamegraph crates.io (version 0.6.13, June 2026) — install command
- samply crates.io (version 0.13.x) — Linux support confirmation
- faer benchmark data (via web search, nalgebra vs faer SVD: 8-13× faster at n=1024) — MEDIUM confidence (benchmarks from faer repo, not independently replicated)
- [Gendignoux 2024 — Optimizing Rayon workloads](https://gendignoux.com/blog/2024/11/18/rust-rayon-optimized.html) — `sched_migrate_task`, cache invalidation patterns, strace analysis
- [hotpath.rs — sampling profiler comparison](https://hotpath.rs/blog/sampling_comparison) — flamegraph vs samply tradeoffs

---

*Stack research for: Rust performance audit toolkit — CPU-bound numerical library (fdars-core)*
*Researched: 2026-08-07*
