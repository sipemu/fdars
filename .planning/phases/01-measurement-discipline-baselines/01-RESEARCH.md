# Phase 01: Measurement Discipline & Baselines — Research

**Researched:** 2026-08-07
**Domain:** Criterion 0.5 benchmark harness, Rust release-mode measurement discipline, fdars hot-path API surface
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** Author a **new dedicated audit bench file** (`fdars-core/benches/audit_hotpaths.rs`, `harness = false`). Leave existing 9 criterion bench files untouched.
- **D-02:** Wrap both inputs and outputs in `criterion::black_box` (Pitfall 3). Register via `[[bench]]` with `harness = false`.
- **D-03:** Baseline = **one representative sentinel function per hot-path module** (6 modules), run at `release + linalg,parallel`, 2 independent invocations each for ±5% variance check.
- **D-04:** Additionally run **one sentinel target across all 4 feature combos** (`""`, `parallel`, `linalg`, `linalg,parallel`). FPCA/SVD is the recommended sentinel. Proves the feature-flag matrix methodology end-to-end.
- **D-05:** Single growing report at `.planning/research/AUDIT-REPORT.md`. Phase 1 writes the methodology section and workload matrix.
- **D-06:** Raw criterion output under `.planning/research/bench/` with naming `p1_<target>_<features>_run<N>.txt`. Create `bench/` dir in this phase.
- **D-07:** Per-module tailored subsets. Candidate sizes N∈{100,500,1000} × M∈{50,200,500}. Cap expensive modules with documented justification (elastic: N≤500, M≤200). Each module's chosen cells and rationale must be in the workload-matrix table.

### Claude's Discretion

- Sentinel-function selection per module (planner/researcher picks — see Section 6 of this document)
- Machine-state / reproducibility controls beyond ±5% two-run rule (cpupower optional)
- Criterion `sample_size` / `measurement_time` config for large-input cells

### Deferred Ideas (OUT OF SCOPE)

- Allocation profiling (dhat) of FdMatrix→DMatrix SVD copy — Phase 4
- Full N×M sweeps per hot path — Phases 3 and 4
- nalgebra-vs-faer SVD comparison — Phase 6
- RAYON_NUM_THREADS thread-scaling sweep — Phase 5
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| PERF-02 | A representative workload matrix (N × M input sizes) is defined per hot-path module so benchmarks reflect realistic usage, not toy inputs | Section 8 (Workload Matrix), justified by CONCERNS.md complexity notes |
</phase_requirements>

---

## Summary

This phase delivers three analysis artifacts: (1) a benchmark methodology section for `AUDIT-REPORT.md`, (2) a representative N×M workload matrix per hot-path module, and (3) baseline criterion runs for one sentinel per module at `linalg,parallel`. The only code produced is a new `fdars-core/benches/audit_hotpaths.rs` bench file plus its `[[bench]]` entry — measurement infrastructure, not algorithm changes.

The research task is entirely codebase-internal: all APIs, patterns, and constraints are verifiable from the existing source tree. No external packages are added (criterion 0.5 is already a dev-dependency). The primary research questions are: (a) the exact Criterion 0.5 idioms for slow, large-N audit benches; (b) the four `cargo bench` invocations for the feature-flag matrix; (c) how to prove release-mode execution; (d) the linker/bus-error triage rule (Pitfall 8); (e) which public function best represents each of the 6 hot-path modules; and (f) how to generate seeded column-major inputs without leaking `test_helpers.rs` into bench scope.

All answers below are sourced from the in-repo files read in this session. No external packages are proposed; no legitimacy audit is required.

**Primary recommendation:** Mirror the scaffold from `regression_benchmarks.rs` and `alignment_benchmarks.rs` exactly — same imports, same `criterion_group!`/`criterion_main!` macro pattern, same `harness = false` `[[bench]]` entry — but use workload-matrix sizes (N∈{100,500,1000}, M∈{50,200}) and tune `sample_size`/`measurement_time` per group for large cells.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Bench file (audit_hotpaths.rs) | `fdars-core/benches/` | — | Dev-dependency scope; never compiled into the shipped crate |
| Raw artifact storage | `.planning/research/bench/` | — | Analysis output; not part of crate build |
| AUDIT-REPORT.md | `.planning/research/` | — | Audit milestone deliverable; consumed by Phase 9 consolidation |
| Seeded input generator | Inline in `audit_hotpaths.rs` | — | `test_helpers.rs` is `#[cfg(test)]` only; bench scope is not test scope |
| Feature-flag matrix | Cargo CLI (`--no-default-features`, `--features`) | — | Same source, 4 compiled variants; no source-level branching needed in the bench file |

---

## Standard Stack

### Core (already present — no new packages)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| criterion | 0.5 | Benchmarking framework | Already in `[dev-dependencies]`; `html_reports` feature already enabled [VERIFIED: fdars-core/Cargo.toml:50] |
| rand | 0.8 | Seeded RNG for synthetic data | Already in `[dependencies]`; `StdRng::seed_from_u64` used project-wide [VERIFIED: fdars-core/Cargo.toml:33] |

**Installation:** No new packages needed. `cargo bench` picks up criterion from the existing dev-dependency.

### New `[[bench]]` Entry Required

Add to `fdars-core/Cargo.toml` (immediately after the existing `[[bench]]` block for `matrix_benchmarks`):

```toml
[[bench]]
name = "audit_hotpaths"
harness = false
```

[VERIFIED: fdars-core/Cargo.toml:54-89] — all 9 existing bench entries use this exact pattern. The `name` field must match the filename stem (`benches/audit_hotpaths.rs`).

---

## Package Legitimacy Audit

No new external packages are introduced in this phase. Criterion 0.5 and rand 0.8 are existing dev-dependencies already present in `Cargo.lock`. No legitimacy gate required.

---

## Architecture Patterns

### System Architecture Diagram

```
cargo bench --bench audit_hotpaths --features linalg,parallel
            │
            ▼
  fdars-core/benches/audit_hotpaths.rs
            │
            ├── bench_fpca_sentinel()      → fdars_core::regression::fdata_to_pc_1d()
            ├── bench_elastic_sentinel()   → fdars_core::alignment::elastic_self_distance_matrix()
            ├── bench_depth_sentinel()     → fdars_core::depth::fraiman_muniz_1d()
            ├── bench_cv_sentinel()        → fdars_core::classification::fclassif_cv()
            ├── bench_streaming_sentinel() → fdars_core::streaming_depth::StreamingFraimanMuniz
            └── bench_smooth_sentinel()    → fdars_core::smoothing::nadaraya_watson()
                        │
                        ▼
          Criterion writes HTML to target/criterion/
          stdout captured to .planning/research/bench/p1_<target>_<features>_run<N>.txt
```

### Recommended Project Structure (new files only)

```
fdars-core/
└── benches/
    └── audit_hotpaths.rs     ← NEW (measurement infrastructure)
.planning/
└── research/
    ├── AUDIT-REPORT.md       ← NEW (methodology + workload matrix sections)
    └── bench/                ← NEW directory
        ├── p1_fpca_linalg_parallel_run1.txt
        ├── p1_fpca_linalg_parallel_run2.txt
        ├── p1_fpca_none_run1.txt          (4-combo sentinel)
        ├── p1_fpca_parallel_run1.txt      (4-combo sentinel)
        ├── p1_fpca_linalg_run1.txt        (4-combo sentinel)
        ├── p1_elastic_linalg_parallel_run1.txt
        ├── p1_depth_linalg_parallel_run1.txt
        ├── p1_cv_linalg_parallel_run1.txt
        ├── p1_streaming_linalg_parallel_run1.txt
        └── p1_smooth_linalg_parallel_run1.txt
```

### Pattern 1: Criterion Group Setup with `harness = false`

The canonical scaffold copied verbatim from the project's existing bench files:

```rust
// Source: fdars-core/benches/regression_benchmarks.rs (lines 1-11, 142-148)
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use fdars_core::matrix::FdMatrix;
// ... module-specific imports ...

fn bench_fpca_sentinel(c: &mut Criterion) {
    let mut group = c.benchmark_group("audit_fpca");
    // tune for large inputs (see Pattern 3)
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(30));
    group.warm_up_time(std::time::Duration::from_secs(5));

    // one cell per size; input built outside iter() to avoid measuring allocation
    let (data, argvals) = make_fpca_input(500, 200);
    group.bench_function("n500_m200", |b| {
        b.iter(|| fdata_to_pc_1d(black_box(&data), black_box(5), black_box(&argvals)))
    });

    group.finish();
}

criterion_group!(benches, bench_fpca_sentinel, /* ... */);
criterion_main!(benches);
```

[VERIFIED: fdars-core/benches/regression_benchmarks.rs:8,58-74,142-148]

### Pattern 2: `black_box` on Both Input AND Output (Pitfall 3)

```rust
// Source: fdars-core/benches/regression_benchmarks.rs (line 69)
b.iter(|| fdata_to_pc_1d(black_box(&data), black_box(ncomp), black_box(&argvals)));
// ↑ inputs wrapped; return value is `Result<FpcaResult,…>` — Criterion's iter()
// drops it, which is sufficient for a non-Copy result. For functions returning
// primitive f64 or Vec, also wrap the output:
b.iter(|| black_box(fraiman_muniz_1d(black_box(data), black_box(data), true)));
```

**Rule:** Wrap all inputs with `black_box`. For functions returning a primitive or `Vec<f64>`, also wrap the output. For `Result<T, E>` returns where T is a large struct, Criterion's drop at the end of `iter` is sufficient — but wrapping output is always safe.

### Pattern 3: Tuning `sample_size` and `measurement_time` for Large-N Cells

Criterion 0.5 defaults: `sample_size = 100`, `measurement_time = 5s`, `warm_up_time = 3s`. For large-N audit cells these defaults produce unacceptably long runs or fail to converge.

```rust
// Source: fdars-core/benches/smoothing_benchmarks.rs (line 143-144)
group.sample_size(20);  // used by existing bench for expensive optim_bandwidth
```

**Guidance for audit cells:**

| Module / cell | sample_size | measurement_time | warm_up_time | Rationale |
|---------------|-------------|------------------|--------------|-----------|
| FPCA n=100, m=50 | 50 | 10s | 3s | Fast; SVD at small m |
| FPCA n=500, m=200 | 20 | 20s | 5s | SVD O(m³) at m=200 costs ~100ms/iter |
| FPCA n=1000, m=500 | 10 | 30s | 5s | SVD O(m³) at m=500 costs ~1–2s/iter [ASSUMED: estimated from O(m³) scaling] |
| Elastic n=100, m=50 | 20 | 20s | 5s | All-pairs O(n²·m²) |
| Elastic n=500, m=200 | 10 | 60s | 10s | O(500²·200²) ≈ very slow; may need further tuning [ASSUMED] |
| Depth/streaming n=100–1000 | 30 | 15s | 3s | O(n·m) — faster than elastic/FPCA |
| CV loops n=100–500 | 15 | 20s | 5s | K-fold, each fold runs FPCA+classifier |
| Smoothing n=50–1000 | 30 | 10s | 3s | O(n·m) Nadaraya-Watson |

**Key constraint:** `sample_size` minimum is 10 in Criterion 0.5. Going below 10 is not supported and will panic. [ASSUMED — from Criterion 0.5 source behavior; 10 is the documented lower bound]

Use per-group configuration (`group.sample_size()` / `group.measurement_time()`) not global Criterion config, so only the audit bench is affected and CI benches stay unchanged.

### Pattern 4: Seeded Synthetic Column-Major Input Generator

`src/test_helpers.rs` is compiled only under `#[cfg(test)]` — bench scope is not test scope. [VERIFIED: fdars-core/src/lib.rs:71-72]:

```rust
#[cfg(test)]
pub(crate) mod test_helpers;
```

The audit bench must contain its own generator. Mirror the project-wide pattern (visible in `regression_benchmarks.rs` lines 18-44 and `tests/integration_explain_pdp.rs`):

```rust
// Source: adapted from fdars-core/benches/regression_benchmarks.rs:18-44
// and fdars-core/benches/alignment_benchmarks.rs:17-31
use fdars_core::matrix::FdMatrix;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use std::f64::consts::PI;

/// Seeded generator: n curves × m evaluation points, column-major FdMatrix.
/// Uses StdRng::seed_from_u64 consistent with project convention
/// (src/parallel.rs: StdRng::seed_from_u64(seed + k as u64)).
fn make_fd_matrix(n: usize, m: usize, seed: u64) -> (FdMatrix, Vec<f64>) {
    let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();
    // Build column-major: data[i + j*n] = curve i at point j
    let mut data = vec![0.0_f64; n * m];
    for i in 0..n {
        let phase = (i as f64 * 3.7 + 0.5).sin() * 0.3;
        let amp = 1.0 + (i as f64 * 5.1).sin() * 0.2;
        for j in 0..m {
            let t = argvals[j];
            let noise = normal.sample(&mut rng) * 0.05;
            data[i + j * n] = amp * (2.0 * PI * (t + phase)).sin() + noise;
        }
    }
    let mat = FdMatrix::from_column_major(data, n, m).unwrap();
    (mat, argvals)
}
```

**Constructor to use:** `FdMatrix::from_column_major(data: Vec<f64>, nrows: usize, ncols: usize) -> Result<FdMatrix, FdarError>` [VERIFIED: fdars-core/benches/regression_benchmarks.rs:43] — every existing bench calls this exact constructor with `.unwrap()` in non-fallible setup code.

**Column-major layout rule:** `data[i + j*n]` = observation `i` at evaluation point `j`. [VERIFIED: CLAUDE.md and fdars-core/benches/alignment_benchmarks.rs:28]:
```rust
data[i + j * n] = amp * (2.0 * PI * (t + phase)).sin();
```

**rand import note:** `rand` 0.8 and `rand_distr` 0.4 are already in `[dependencies]` (not dev-dependencies) [VERIFIED: fdars-core/Cargo.toml:33-34], so bench files can use them without adding dev-deps.

### Anti-Patterns to Avoid

- **Reusing `test_helpers::uniform_grid`:** It is `#[cfg(test)]` and `pub(crate)` — not accessible from bench code. Write an inline equivalent: `(0..m).map(|j| j as f64 / (m-1) as f64).collect()`.
- **Building inputs inside `b.iter()`:** Measure the hot function, not the allocator. All `FdMatrix` construction must happen before the `bench_function` / `bench_with_input` closure.
- **Using `group.sample_size(0)`:** Panics. Minimum is 10.
- **Omitting `group.finish()`:** Criterion does not flush the group's results without the explicit `finish()` call — seen in every existing bench [VERIFIED: regression_benchmarks.rs:73, alignment_benchmarks.rs:81].

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Statistical noise filtering in bench results | Custom averaging loop | Criterion's built-in outlier detection + `sample_size` / `measurement_time` config | Criterion already does Welch t-test, outlier classification, and HTML report generation |
| Timing in release mode | Manual `std::time::Instant` wrapping | `cargo bench` with `bench` profile (release equivalent) | `cargo bench` compiles with the `bench` profile which mirrors release; Criterion wraps the timer |
| Column-major FdMatrix for bench | Raw Vec indexing | `FdMatrix::from_column_major()` already in the crate | Avoids layout errors; matches what all real callers pass |

---

## Section 1: Criterion 0.5 Harness Recipe

### `harness = false` Registration

Every bench in the project registers via `[[bench]]` with `harness = false` in `fdars-core/Cargo.toml`. [VERIFIED: fdars-core/Cargo.toml:54-89 — verbatim block]:

```toml
[[bench]]
name = "seasonal_benchmarks"
harness = false

[[bench]]
name = "depth_benchmarks"
harness = false
```

The new entry follows the same pattern:
```toml
[[bench]]
name = "audit_hotpaths"
harness = false
```

`harness = false` disables the libtest harness and hands control to Criterion's `criterion_main!` macro, which registers its own entry point. Without this, `cargo bench` would try to run Criterion groups as test cases and fail with a link error.

### Criterion Group + Main Macros

```rust
criterion_group!(
    benches,
    bench_fpca_sentinel,
    bench_elastic_sentinel,
    bench_depth_sentinel,
    bench_cv_sentinel,
    bench_streaming_sentinel,
    bench_smooth_sentinel,
);
criterion_main!(benches);
```

[VERIFIED: fdars-core/benches/regression_benchmarks.rs:142-148 and alignment_benchmarks.rs:149-155] — every existing bench uses this exact two-macro pattern.

### Where Criterion Writes Results

- **HTML reports:** `target/criterion/<benchmark_name>/<bench_id>/report/index.html` — generated when `html_reports` feature is enabled (it is: [VERIFIED: fdars-core/Cargo.toml:50]: `criterion = { version = "0.5", features = ["html_reports"] }`)
- **Stdout:** Wall-clock measurements printed to stdout during the run in the format `<group>/<id>   time: [low  mean  high]`
- **Raw output for `.planning/research/bench/`:** Redirect stdout with `2>&1 | tee`. Example:

```bash
cargo bench -p fdars-core \
  --bench audit_hotpaths \
  --features linalg,parallel \
  -- --nocapture \
  2>&1 | tee .planning/research/bench/p1_fpca_linalg_parallel_run1.txt
```

The `--` passes arguments to the criterion binary. `--nocapture` is not needed for bench (Criterion prints to stdout by default); the `tee` captures it while also displaying to terminal.

---

## Section 2: Feature-Flag Matrix Build/Run Commands

### Feature-Flag Background

[VERIFIED: fdars-core/Cargo.toml:18-29]:
```toml
[features]
default = ["parallel"]
parallel = ["rayon"]
linalg = ["faer", "anofox-regression"]
serde = ["dep:serde", "dep:serde_json"]
js = ["getrandom/js"]
```

Because `default = ["parallel"]`, a plain `cargo bench` implicitly enables `parallel`. To disable it, `--no-default-features` is required. The 4 combos and their invocations:

### The 4 Exact Invocations

```bash
# Combo 1: empty features (WASM / minimal build — no rayon, no faer)
cargo bench -p fdars-core \
  --bench audit_hotpaths \
  --no-default-features \
  2>&1 | tee .planning/research/bench/p1_fpca_none_run1.txt

# Combo 2: parallel only (default user build)
cargo bench -p fdars-core \
  --bench audit_hotpaths \
  --no-default-features --features parallel \
  2>&1 | tee .planning/research/bench/p1_fpca_parallel_run1.txt

# Combo 3: linalg only (ridge/faer paths, sequential)
cargo bench -p fdars-core \
  --bench audit_hotpaths \
  --no-default-features --features linalg \
  2>&1 | tee .planning/research/bench/p1_fpca_linalg_run1.txt

# Combo 4: linalg + parallel (full capability — primary audit build)
cargo bench -p fdars-core \
  --bench audit_hotpaths \
  --features linalg,parallel \
  2>&1 | tee .planning/research/bench/p1_fpca_linalg_parallel_run1.txt
```

**Critical:** Combo 4 does NOT need `--no-default-features` because `parallel` is already in `default`. Writing `--features linalg,parallel` adds `linalg` to the already-active `parallel`. Combos 1–3 need `--no-default-features` first to strip `parallel` from the default set.

### Feature-Flag Compilation Effect (Pitfall 18)

The same `audit_hotpaths.rs` source compiles 4 different binaries. What changes:

| Feature set | `iter_maybe_parallel!` macros | `ridge_regression_fit` | `fdata_to_pc_1d` path |
|-------------|-------------------------------|------------------------|----------------------|
| `""` | sequential (cfg gates rayon out) | absent (compile error if called) | nalgebra SVD, sequential |
| `parallel` | rayon (thread pool active) | absent | nalgebra SVD, parallel where macro is used |
| `linalg` | sequential | present (faer Cholesky) | nalgebra SVD, sequential |
| `linalg,parallel` | rayon | present | nalgebra SVD, parallel |

**Bench file constraint:** The audit bench must ONLY call APIs available in the leanest combo (`""`). If `ridge_regression_fit` (gated by `linalg`) is called unconditionally, Combos 1 and 2 will fail to compile. The 6 sentinel functions proposed in Section 6 below are all available in the empty-features build. FPCA (`fdata_to_pc_1d`) does not gate on `linalg` — it uses nalgebra SVD which is always present. [VERIFIED: fdars-core/src/lib.rs:47: `linalg = ["faer", "anofox-regression"]` — `fdata_to_pc_1d` is in `src/regression.rs` which imports nalgebra, not faer]

### Feature-Flag Table for AUDIT-REPORT.md

Include this table verbatim in the methodology section:

| Feature set | `--no-default-features` required? | What it tests |
|-------------|-----------------------------------|--------------|
| `""` | Yes | Sequential, no linalg (WASM / minimal / CRAN build) |
| `parallel` | Yes + `--features parallel` | Default for most library users; rayon active |
| `linalg` | Yes + `--features linalg` | Ridge/faer paths, sequential |
| `linalg,parallel` | No (parallel is default) | Full capability; primary audit comparison baseline |

---

## Section 3: Release-Mode Confirmation (Pitfall 1)

### How `cargo bench` Compiles

`cargo bench` uses the `bench` profile, which by default has `opt-level = 3` and `debug = false` — equivalent to release. [ASSUMED — standard Cargo behavior; no custom `[profile.bench]` override found in Cargo.toml]. The resulting binary lands in `target/release/deps/`, not `target/debug/deps/`.

### Concrete Verification Check

```bash
# Step 1: Run the bench and capture the Criterion output header
cargo bench -p fdars-core --bench audit_hotpaths --features linalg,parallel \
  2>&1 | head -20

# Step 2: Confirm the binary path in the output
# Criterion 0.5 prints the binary path on the first line, e.g.:
#   Running target/release/deps/audit_hotpaths-a1b2c3d4e5f6

# Step 3: Programmatic check (executor must run this and record output)
ls -la target/release/deps/audit_hotpaths-* 2>/dev/null \
  && echo "RELEASE CONFIRMED" \
  || (ls -la target/debug/deps/audit_hotpaths-* 2>/dev/null \
       && echo "WARNING: DEBUG BUILD DETECTED")
```

**Warning signs from Pitfall 1:** If `/target/debug/` appears, or if `cargo test --bench` was used instead of `cargo bench`, the numbers are 5–50x inflated. Criterion shows `target/release/deps/<bench_name>-<hash>` at run start. The executor must paste this line into the bench artifact file as the first recorded fact.

### Including in Bench Artifacts

The executor captures `rustc --version` and the binary path together:

```bash
{
  echo "=== ENVIRONMENT ==="
  rustc --version
  cargo --version
  echo "=== BINARY ==="
  cargo bench -p fdars-core --bench audit_hotpaths --features linalg,parallel \
    --no-run 2>&1 | grep "Compiling\|Finished\|Running"
} | tee .planning/research/bench/p1_env_info.txt
```

---

## Section 4: rustc / Toolchain Version Capture

```bash
# Exact commands — executor must run and record all three lines
rustc --version        # e.g.: rustc 1.97.0 (2d8144b78 2026-07-07)
cargo --version        # e.g.: cargo 1.97.0 (...)
rustup show active-toolchain  # e.g.: stable-x86_64-unknown-linux-gnu (default)
```

Current environment [VERIFIED: Bash tool output in this session]:
- `rustc 1.97.0 (2d8144b78 2026-07-07)` — stable-x86_64-unknown-linux-gnu

**Runtime requirement for `linalg` feature:** Rust 1.84.0 minimum (faer 0.23 constraint). [VERIFIED: fdars-core/Cargo.toml:38-39]:
```toml
faer = { version = "0.23", optional = true }
```
And from CLAUDE.md: "For `linalg` feature: Rust 1.84.0 or higher." The current runtime (1.97.0) satisfies this.

The methodology section must record the full `rustc --version` string and confirm it meets the 1.84.0 floor before any `linalg` bench results are recorded.

---

## Section 5: Criterion/Doctest Linker-Flakiness and Infra-vs-Code Triage (Pitfall 8)

### What the Failure Looks Like

From PITFALLS.md Pitfall 8 [VERIFIED: .planning/research/PITFALLS.md:156-170]:

> Linker bus errors on Linux can arise from memory-mapped file limits (`vm.max_map_count`), toolchain mismatches between the system linker and the Rust toolchain, or from doctest infrastructure bugs in Criterion 0.5 (a known issue). These are infrastructure failures, not fdars failures.

Symptoms:
- `cargo bench` or `cargo test --doc` exits with SIGBUS (bus error) or SIGSEGV rather than a failed test count
- The failure line says `error: process didn't exit successfully` without a named test underneath it
- The failure disappears on `--test-threads=1` or with `RUSTFLAGS=-C link-arg=-fuse-ld=lld`
- The failing process is the Criterion harness binary itself, not a specific benchmark function

### Triage Rule (the exact rule later phases apply)

```
IF the failure output contains a named test → "FAILED test_name" → code failure → counts in defect list
IF the failure output is "error: process didn't exit successfully" with NO named test → infra/linker failure → does NOT count
```

From PITFALLS.md Pitfall 8 verbatim [VERIFIED: .planning/research/PITFALLS.md:164]:
> run `cargo test -p fdars-core --features linalg -- --test-threads=1 2>&1 | grep -E "^(test |FAILED|error)"` and distinguish: (a) `FAILED` lines naming a specific test = code failure, (b) `error: process didn't exit successfully` without a test name = toolchain/linker failure.

### Integration Gotcha

From PITFALLS.md Integration Gotchas table [VERIFIED: .planning/research/PITFALLS.md:417-418]:
> Criterion 0.5 + doctest harness: Criterion 0.5 has a known linker issue with doctests on some Linux configurations; `cargo test --doc` may bus-error. Run `cargo test -p fdars-core --lib --features linalg` first; isolate doctest failures separately.

### Methodology Section Text (copy into AUDIT-REPORT.md)

The methodology section must include a paragraph reading approximately:

> **Infrastructure vs. Code Failure Triage:** This environment exhibits known Criterion 0.5 / doctest linker bus-error flakiness on Linux (Pitfall 8 in PITFALLS.md). A `cargo bench` or `cargo test --doc` invocation that exits via signal (SIGBUS) without printing a named `FAILED test_name` line is classified as an **infrastructure failure** and does not count as a fdars code defect. Failures of the form `FAILED test_name` are **code failures** and count. All bench artifacts record the full exit status alongside stdout to enable retroactive classification.

---

## Section 6: Sentinel Function Selection (Claude's Discretion)

### Selection Criteria

A sentinel must be: (1) public and reachable from a bench file, (2) representative of the hot-path's dominant cost (the operation that will scale badly with N or M), and (3) callable without `linalg`-gated APIs so all 4 feature combos compile.

All 6 proposed sentinels satisfy criterion (3). [VERIFIED by tracing imports in bench files already read and lib.rs re-exports at lines 127-174.]

### Module 1: Elastic Alignment — `elastic_self_distance_matrix`

**Public path:** `fdars_core::alignment::elastic_self_distance_matrix` [VERIFIED: fdars-core/src/lib.rs:148]

**Signature (from alignment_benchmarks.rs line 95):**
```rust
elastic_self_distance_matrix(&mat, &argvals, 0.0)
// -> Result<Vec<f64>, FdarError>  (flat n×n distance matrix)
```

**Why representative:** This is the all-pairs DP — O(n²·m²) — the exact cost noted in CONCERNS.md [VERIFIED: .planning/codebase/CONCERNS.md:132-134]: "elastic_align_many() computes pairwise alignments, O(n² * m²) DP. For n=1,000, m=500, this becomes 250 million comparisons (~60 sec)." It exercises more of the elastic pipeline than a single-pair alignment.

**Workload cell:** N=100, M=50 (baseline). See Section 8 for size caps.

### Module 2: FPCA/SVD — `fdata_to_pc_1d` (4-combo sentinel per D-04)

**Public path:** `fdars_core::regression::fdata_to_pc_1d` [VERIFIED: fdars-core/benches/regression_benchmarks.rs:11 and fdars-core/src/lib.rs line via `pub mod regression`]

**Signature (from regression_benchmarks.rs line 69):**
```rust
fdata_to_pc_1d(&data, ncomp, &argvals)
// -> Result<FpcaResult, FdarError>
```

**Why representative:** Contains the `FdMatrix → nalgebra::DMatrix` conversion (ARCHITECTURE.md anti-pattern) + full SVD. [VERIFIED: .planning/codebase/ARCHITECTURE.md:197-202]: "Functions convert `FdMatrix` to nalgebra `DMatrix`, compute SVD, convert back... Unnecessary allocations and copies. For large m, overhead is significant." The nalgebra SVD is O(m³) in the number of evaluation points. This function exercises both the `parallel` path (data centering is parallelized via `iter_maybe_parallel!` in `fdata.rs`) and — when `linalg` is active — still uses nalgebra SVD (faer gates only ridge/Cholesky). It is the correct 4-combo sentinel because it exercises the parallel path distinctly between combos 1/3 vs 2/4.

**Workload cells:** N∈{100,500,1000} × M∈{50,200,500} — full grid applies here. See Section 8.

### Module 3: Depth & Distance — `fraiman_muniz_1d`

**Public path:** `fdars_core::depth::fraiman_muniz_1d` [VERIFIED: fdars-core/benches/depth_benchmarks.rs:8]

**Signature (from depth_benchmarks.rs line 40):**
```rust
fraiman_muniz_1d(data, data, true)
// -> Vec<f64>  (depth vector of length n)
```

**Why representative:** The Fraiman-Muniz depth is O(n²·m) — dominant depth cost. The existing bench already uses N∈{50,200,500,1000,2300} at M=200, confirming it is tractable at audit sizes. The `lp_self_1d` distance matrix function is also a candidate but FM depth is more central to the module.

**Workload cells:** N∈{100,500,1000} × M∈{50,200,500}.

### Module 4: CV Loops — `fclassif_cv`

**Public path:** `fdars_core::classification::fclassif_cv` [VERIFIED: fdars-core/src/classification/mod.rs:165: `pub use cv::fclassif_cv;`]

**Signature (from fdars-core/src/classification/cv.rs:45-55):**
```rust
fclassif_cv(
    data: &FdMatrix,
    argvals: &[f64],
    y: &[usize],
    scalar_covariates: Option<&FdMatrix>,
    method: &str,      // "lda", "qda", "knn", etc.
    ncomp: usize,
    nfold: usize,
    seed: u64,
) -> Result<ClassifCvResult, FdarError>
```

**Why representative:** CV loops are the dominant cost for any tuning workflow. `fclassif_cv` runs K-fold cross-validation where each fold includes an FPCA fit + classifier fit + prediction — making it a composition of nearly all hot paths in one call. Use `method = "lda"` (fastest classifier, isolates the CV loop overhead). [VERIFIED: fdars-core/src/classification/mod.rs:165 confirms public re-export]

**Workload cells:** N∈{100,500} × M∈{50,200}, `nfold=5`, `ncomp=5`. N=1000 at M=200 with 5-fold generates 200-sample training sets per fold with full FPCA — include if time permits, cap at N=500 as baseline.

### Module 5: Streaming Depth — `StreamingFraimanMuniz::depth_batch`

**Public path:** `fdars_core::streaming_depth::StreamingFraimanMuniz` + method `depth_batch` [VERIFIED: fdars-core/benches/depth_benchmarks.rs:16-18, and fdars-core/src/streaming_depth/fraiman_muniz.rs:72]

**Pattern (from depth_benchmarks.rs lines 66-79):**
```rust
let state = SortedReferenceState::from_reference(black_box(data));
let fm = StreamingFraimanMuniz::new(state, true);
fm.depth_batch(black_box(data))
```

**Why representative:** The streaming depth module is designed for the incremental-arrival use case. The construct-then-query pattern measures both the reference-state build (sort + index) and the query cost, matching real usage. The existing bench already goes to N=2300 at M=200 — the audit can inherit those sizes for streaming.

**Workload cells:** N∈{100,500,1000} × M∈{50,200,500}. Include `from_reference` + `depth_batch` as a single timed unit (as the existing bench does).

### Module 6: Smoothing — `nadaraya_watson`

**Public path:** `fdars_core::smoothing::nadaraya_watson` [VERIFIED: fdars-core/src/smoothing.rs:72: `pub fn nadaraya_watson`; and benches/smoothing_benchmarks.rs:37-51]

**Signature (from smoothing_benchmarks.rs line 43-49):**
```rust
nadaraya_watson(
    &x,      // &[f64] — input evaluation points
    &y,      // &[f64] — observed values
    &x_new,  // &[f64] — prediction grid
    bandwidth,
    "gaussian",
)
// -> Result<Vec<f64>, FdarError>
```

**Why representative:** Kernel smoothing is the most commonly used pre-processing step. Nadaraya-Watson is O(n·m) and is the base case for the smoothing module. Note that for the smoothing sentinel, N is the number of training observations (not curves) and M is the prediction grid size. The existing bench goes to N=1000, M=200.

**Workload cells:** N∈{100,500,1000} (input observations) × M∈{50,200,500} (prediction grid).

### Sentinel Summary Table

| Module | Sentinel Function | Public Path | Dominant Cost | Feature Requirement |
|--------|------------------|-------------|--------------|---------------------|
| Elastic alignment | `elastic_self_distance_matrix` | `fdars_core::alignment::elastic_self_distance_matrix` | O(n²·m²) DP | none (all 4 combos) |
| FPCA/SVD | `fdata_to_pc_1d` | `fdars_core::regression::fdata_to_pc_1d` | SVD O(m³) + copy | none (4-combo sentinel) |
| Depth & distance | `fraiman_muniz_1d` | `fdars_core::depth::fraiman_muniz_1d` | O(n²·m) rank comparison | none |
| CV loops | `fclassif_cv` | `fdars_core::classification::fclassif_cv` | K × FPCA + fit | none |
| Streaming depth | `StreamingFraimanMuniz::depth_batch` | `fdars_core::streaming_depth::StreamingFraimanMuniz` | O(n·m) sorted lookup | none |
| Smoothing | `nadaraya_watson` | `fdars_core::smoothing::nadaraya_watson` | O(n·m) kernel eval | none |

---

## Section 7: Input Generation Pattern for the Bench

**No seeded generator exists in bench scope.** [VERIFIED: fdars-core/src/lib.rs:71-72 — `test_helpers` is `#[cfg(test)]` only.] The audit bench must inline its own.

**Complete inline generator (copy into `audit_hotpaths.rs`):**

```rust
use fdars_core::matrix::FdMatrix;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::f64::consts::PI;

/// Build a column-major FdMatrix with n curves × m evaluation points.
/// Layout: data[i + j*n] = observation i at evaluation point j.
/// Seed is stable: same (n, m, seed) always produces the same matrix.
fn make_fd_matrix(n: usize, m: usize, seed: u64) -> (FdMatrix, Vec<f64>) {
    let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
    let mut data = vec![0.0_f64; n * m];
    for i in 0..n {
        // Deterministic pseudo-random phase/amplitude — no heap allocation per iteration
        let phase = ((i as f64 * 3.7 + seed as f64 * 0.01).sin()) * 0.3;
        let amp = 1.0 + ((i as f64 * 5.1 + seed as f64 * 0.007).sin()) * 0.2;
        for j in 0..m {
            let t = argvals[j];
            data[i + j * n] = amp * (2.0 * PI * (t + phase)).sin();
        }
    }
    let mat = FdMatrix::from_column_major(data, n, m).unwrap();
    (mat, argvals)
}

/// Build binary class labels for n curves (alternating 0/1).
fn make_class_labels(n: usize) -> Vec<usize> {
    (0..n).map(|i| i % 2).collect()
}
```

**Why no `StdRng` sample in the inner loop:** Sampling a normal distribution inside the per-element loop would require `rand_distr`, adds allocation pressure, and is unnecessary for a sentinel that just needs varied but non-degenerate curves. The deterministic trig formula above is consistent with the approach in `regression_benchmarks.rs` (lines 23-30) and `alignment_benchmarks.rs` (lines 21-29). [VERIFIED: both read in this session]

---

## Section 8: Workload Matrix Justification (PERF-02 / SC1)

### Complexity Sources (from CONCERNS.md)

[VERIFIED: .planning/codebase/CONCERNS.md:119-134]:

| Module | Complexity in N | Complexity in M | Cost note |
|--------|----------------|----------------|-----------|
| Elastic all-pairs DP | O(n²) | O(m²) | n=1000, m=500 ≈ 60s |
| FPCA/SVD | O(n·m) centering | O(m³) SVD | m=500 → several seconds |
| Depth (FM) | O(n²·m) | O(m) per pair | Tractable to n=2300 at m=200 |
| CV loops | O(K · n · m) | O(m³) per fold | K-fold × FPCA cost |
| Streaming depth | O(n·m) | O(m) per query | Very fast; no cap needed |
| Smoothing | O(n·m) | — | n×m kernel evaluations |

### Per-Module Workload Cell Table (D-07)

| Module | N cells | M cells | Cap / Rationale | Full-grid? |
|--------|---------|---------|-----------------|------------|
| Elastic alignment | {100, 500} | {50, 200} | Cap: N≤500, M≤200. O(n²·m²): N=1000×M=500 ≈ 60s per iteration × sample_size=10 = 600s (unacceptable). N=500×M=200 ≈ 3.8s/iter × 10 = 38s (borderline; use `measurement_time=60s`). [VERIFIED: CONCERNS.md:132-134] | No — 4 cells |
| FPCA/SVD | {100, 500, 1000} | {50, 200, 500} | No cap. O(m³) SVD: m=500 costs ~1–2s/iter [ASSUMED]; use sample_size=10. N scaling is cheap (centering is O(n·m)). | Yes — 9 cells |
| Depth & distance | {100, 500, 1000} | {50, 200, 500} | No cap. O(n²·m) but FM depth at n=1000,m=500 is tractable — existing bench goes to n=2300 at m=200 [VERIFIED: depth_benchmarks.rs:37-43]. | Yes — 9 cells |
| CV loops | {100, 500} | {50, 200} | Cap: N≤500, M≤200. Each fold runs FPCA (O(m³)) + classifier fit + predict; K=5 multiplies the cost. N=1000 with 5 folds and m=200 could easily exceed 30s/iter. | No — 4 cells |
| Streaming depth | {100, 500, 1000} | {50, 200, 500} | No cap. O(n·m) build + O(n·m) query. Very fast at all sizes. | Yes — 9 cells |
| Smoothing | {100, 500, 1000} | {50, 200, 500} | No cap. O(n·m) Nadaraya-Watson. Existing bench reaches n=1000,m=200 [VERIFIED: smoothing_benchmarks.rs:34]. | Yes — 9 cells |

**Phase 1 scope (D-03):** Only ONE cell per module is benchmarked as the baseline. The recommended Phase 1 cell is the "medium load" cell that fits the `release+linalg,parallel` 2-run variance check within reasonable time:

| Module | Phase 1 Baseline Cell | Rationale |
|--------|-----------------------|-----------|
| Elastic | N=100, M=50 | O(100²×50²) = 25M ops — completes in < 10s/iter |
| FPCA/SVD | N=500, M=200 | Middle of the grid; SVD at m=200 is the interesting scaling regime |
| Depth | N=500, M=200 | Matches existing bench midpoint |
| CV loops | N=100, M=50 | Fast enough for 2-run check; still real workload |
| Streaming | N=500, M=200 | Matches existing bench midpoint |
| Smoothing | N=500, M=200 | Matches existing bench range |

---

## Common Pitfalls

### Pitfall 1: Debug-mode execution (Pitfall 1 from PITFALLS.md)

**What goes wrong:** `cargo bench` compiles with the `bench` profile (release equivalent) by default, but `cargo test --bench` uses the `test` profile (debug). Numbers are 5–50× inflated.
**How to avoid:** Use `cargo bench`, not `cargo test --bench`. Check output header for `target/release/deps/`. Record binary path in artifact.
**Warning signs:** SVD on a 50×50 matrix takes > 10ms; throughput < 1 MB/s for simple matrix ops. [VERIFIED: .planning/research/PITFALLS.md:14-27]

### Pitfall 2: Missing `black_box` on output (Pitfall 3 from PITFALLS.md)

**What goes wrong:** Compiler eliminates the computation; benchmark measures < 10 ns.
**How to avoid:** Wrap both inputs and outputs. For `Result<T, E>` returns, Criterion's drop is sufficient; for `Vec<f64>` or `f64` returns, explicitly wrap: `black_box(fn_call(...))`.
**Warning signs:** 0% variance; < 10 ns for matrix operations. [VERIFIED: .planning/research/PITFALLS.md:56-70]

### Pitfall 3: Inputs built inside `b.iter()` closure

**What goes wrong:** `FdMatrix::from_column_major` allocates O(n·m) bytes every iteration. The bench measures the allocator, not the function.
**How to avoid:** Build all inputs before `bench_function` / `bench_with_input`. Only the function call goes inside `b.iter()`.

### Pitfall 4: Elastic N=1000 or M=500 without time cap

**What goes wrong:** N=1000×M=500 elastic bench with default `sample_size=100` runs for ~100 minutes, blocks the executor, and may OOM.
**How to avoid:** Use per-group `sample_size(10)` + `measurement_time(Duration::from_secs(60))` for elastic large cells. If a single iteration exceeds 30s, drop the cell from Phase 1 and note it as out-of-bounds in the workload matrix table. [VERIFIED: CONCERNS.md:132-134]

### Pitfall 5: Feature-flag combo omitting `--no-default-features`

**What goes wrong:** Writing `--features ""` does NOT disable `parallel`. `default = ["parallel"]` stays active unless `--no-default-features` is passed.
**How to avoid:** Use the exact invocations in Section 2. [VERIFIED: fdars-core/Cargo.toml:21]

---

## Runtime State Inventory

This is a greenfield phase (new files only). No renaming, refactoring, or migration. Runtime state inventory: SKIPPED — not applicable.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust stable toolchain | All bench compilation | Yes | 1.97.0 [VERIFIED: Bash] | — |
| cargo bench | Criterion invocation | Yes | cargo 1.97.0 | — |
| faer (via `linalg` feature) | Combo 3 and 4 | Yes (1.97 >= 1.84) | 0.23 [VERIFIED: Cargo.toml] | — |
| rayon (via `parallel` feature) | Combo 2 and 4 | Yes (optional dep, in Cargo.lock) | 1.10 [VERIFIED: Cargo.toml] | — |
| criterion 0.5 | Bench harness | Yes [VERIFIED: Cargo.toml dev-deps] | 0.5 | — |
| tee (Linux coreutils) | stdout artifact capture | Yes (standard Linux) | — | `> file` redirect (loses live display) |

**Missing dependencies with no fallback:** None.
**Missing dependencies with fallback:** None.

**Known environment issue (SC4):** Criterion 0.5 / doctest linker bus-error flakiness on this Linux machine. Documented in STATE.md: [VERIFIED: .planning/STATE.md:75-76]: "Environment has known criterion/doctest linker bus-error flakiness — Phase 1 methodology must document infra-vs-code failure triage." Triage rule is in Section 5.

---

## Validation Architecture

> nyquist_validation is enabled (config.json `workflow.nyquist_validation: true`).

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Rust built-in test harness (`cargo test`) |
| Config file | None (no `rustfmt.toml`, no separate test config) |
| Quick run command | `cargo test -p fdars-core --lib --features linalg -- --test-threads=1` |
| Full suite command | `cargo test -p fdars-core --features linalg` |

### Phase 1 Artifacts → Validation Map

Phase 1 produces **non-code artifacts** (a bench file, raw text files, a report section, a directory). "Validation" means confirming these artifacts exist and are correct, not running unit tests.

| SC # | Deliverable | Validation Check | Automated? |
|------|-------------|-----------------|-----------|
| SC1 | Workload matrix in AUDIT-REPORT.md | All 6 module rows present; each has N, M, cap rationale | Manual grep: `grep -c "Elastic\|FPCA\|Depth\|CV\|Streaming\|Smoothing" AUDIT-REPORT.md` |
| SC2 | Methodology section in AUDIT-REPORT.md | Contains: `--release`, feature-flag matrix table, `black_box`, ±5%, rustc version, infra triage rule | Manual review |
| SC3 | Baseline bench runs in `.planning/research/bench/` | 6 `*_linalg_parallel_run1.txt` files + 6 `*_linalg_parallel_run2.txt` files exist and are non-empty | `ls .planning/research/bench/p1_*_run{1,2}.txt | wc -l` (expect ≥ 12) |
| SC4 | Infra triage rule documented | Methodology section contains exact triage rule text (Section 5) | `grep -c "infrastructure failure" .planning/research/AUDIT-REPORT.md` (expect ≥ 1) |
| SC-bench | audit_hotpaths.rs compiles for all 4 combos | `cargo check --bench audit_hotpaths --no-default-features` and `--features linalg,parallel` both succeed | `cargo check` automated |

### Wave 0 Gaps

- [ ] `fdars-core/benches/audit_hotpaths.rs` — does not yet exist; must be created in Wave 1
- [ ] `.planning/research/bench/` directory — does not yet exist; must be created before first bench run
- [ ] `.planning/research/AUDIT-REPORT.md` — does not yet exist; must be seeded with methodology + workload matrix sections

---

## Security Domain

> `security_enforcement: true` in config.json.

This phase introduces no network calls, no user input, no secrets, no authentication, and no data persistence beyond local text files. The new code is a criterion bench file that calls existing in-crate functions on synthetic in-memory data.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | Not applicable (local CLI tool) |
| V3 Session Management | No | Not applicable |
| V4 Access Control | No | Not applicable |
| V5 Input Validation | No | Synthetic data only; no user-supplied input |
| V6 Cryptography | No | `StdRng` used for reproducibility, not security [VERIFIED: CONCERNS.md:62-65 — "context is statistical/simulation, not cryptography"] |

**Security finding:** None. This phase is purely additive analysis infrastructure with no attack surface.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `sample_size(10)` is the minimum allowed by Criterion 0.5 without panic | Section 1 (Pattern 3) | If minimum is lower, executor could use smaller sizes for very slow cells without error |
| A2 | FPCA at N=1000, M=500 costs ~1–2s per iteration (used to justify sample_size=10 and measurement_time=30s) | Section 1 (Pattern 3), Section 8 | If actually < 100ms, could use larger sample_size; if > 10s, would need further reduction |
| A3 | Elastic N=500, M=200 costs ~3.8s per iteration (O(500²×200²) extrapolated from CONCERNS.md n=1000,m=500≈60s) | Section 8 | If faster, medium elastic cell fits in default measurement_time; if slower, may need to drop to N=100,M=50 only |
| A4 | `cargo bench` uses the `bench` profile (opt-level=3, equivalent to release) by default with no custom `[profile.bench]` override | Section 3 | No `[profile.bench]` was found in `fdars-core/Cargo.toml`; if a workspace `Cargo.toml` overrides it, the binary might not be fully optimized |
| A5 | `fdata_to_pc_1d` exercises a `parallel`-gated path in its data centering step | Section 6 (Module 2) | If centering is not wrapped in `iter_maybe_parallel!`, Combos 1/3 vs 2/4 would measure identical code paths — check `src/fdata.rs` before writing the methodology text |

**If this table is empty:** Not empty — 5 assumptions logged above. A5 is the highest-risk and should be verified by the executor before writing the 4-combo comparison methodology.

---

## Open Questions

1. **Does `fdata_to_pc_1d` actually exercise a parallel-gated code path?**
   - What we know: The function calls centering (via `src/fdata.rs`) and then nalgebra SVD. The 5 `iter_maybe_parallel!` macros are in `parallel.rs`.
   - What's unclear: Whether the centering loop in `fdata.rs` uses `iter_maybe_parallel!` or a plain sequential loop.
   - Recommendation: The executor should grep `src/fdata.rs` for `iter_maybe_parallel` before finalizing the methodology. If not present, substitute `karcher_mean` (elastic) as the 4-combo sentinel instead — it uses `maybe_par_chunks_mut!` in the gradient step.

2. **Workspace-level `[profile.bench]` override**
   - What we know: `fdars-core/Cargo.toml` has no custom bench profile. The workspace root `Cargo.toml` was not read in this session.
   - What's unclear: Whether the workspace root overrides the bench profile.
   - Recommendation: Executor reads `/home/simonm/projects/rust/fdars/Cargo.toml` and checks for `[profile.bench]` before the first bench run. If absent, the standard bench-profile-is-release assumption (A4) holds.

3. **Elastic N=500, M=200 single-iteration cost**
   - What we know: CONCERNS.md states n=1000,m=500≈60s. The O(n²·m²) extrapolation to n=500,m=200 gives approximately (500/1000)² × (200/500)² × 60 = 0.25 × 0.16 × 60 ≈ 2.4s per iter.
   - What's unclear: The Sakoe-Chiba band (added in v0.14.0) may reduce this significantly.
   - Recommendation: Run a single `cargo bench --bench audit_hotpaths -p fdars-core --features linalg,parallel -- --warm-up-time 1 --sample-size 1` trial to get a first-pass timing before committing to sample_size/measurement_time config.

---

## Sources

### Primary (HIGH confidence — read directly in this session)

- `fdars-core/Cargo.toml` lines 18-89 — features, dependencies, all 9 existing `[[bench]]` entries
- `fdars-core/benches/regression_benchmarks.rs` lines 1-148 — canonical scaffold: imports, group setup, `criterion_group!`/`criterion_main!`, `black_box` pattern, `FdMatrix::from_column_major`, generator pattern
- `fdars-core/benches/alignment_benchmarks.rs` lines 1-155 — elastic sentinel patterns, column-major data layout in benches
- `fdars-core/benches/depth_benchmarks.rs` lines 1-324 — depth and streaming depth sentinel patterns, `SortedReferenceState` usage
- `fdars-core/benches/smoothing_benchmarks.rs` lines 1-177 — smoothing sentinel patterns; `group.sample_size(20)` for expensive operations (line 144)
- `fdars-core/src/lib.rs` lines 1-174 — public module declarations, `#[cfg(test)]` gating of `test_helpers`, all re-exports confirming sentinel paths
- `fdars-core/src/classification/mod.rs` (grep) — confirms `pub use cv::fclassif_cv` at line 165
- `fdars-core/src/streaming_depth/fraiman_muniz.rs` (grep) — confirms `depth_batch` at line 77
- `.planning/research/PITFALLS.md` lines 1-503 — all relevant pitfalls (1,2,3,4,5,6,7,8,17,18) and integration gotchas table
- `.planning/codebase/CONCERNS.md` lines 73-134 — performance bottlenecks and scaling limits used to justify size caps
- `.planning/codebase/ARCHITECTURE.md` lines 196-202 — `FdMatrix→DMatrix` anti-pattern
- `.planning/codebase/TESTING.md` lines 56-57, 119-130 — bench file locations, `generate_regression_data` pattern
- `.planning/phases/01-measurement-discipline-baselines/01-CONTEXT.md` — all locked decisions D-01 through D-07
- `.planning/STATE.md` line 75-76 — known linker bus-error flakiness in this environment
- `rustc --version` (Bash) — confirmed 1.97.0 stable on x86_64-unknown-linux-gnu

### Tertiary (LOW confidence — training knowledge, not verified this session)

- Criterion 0.5 minimum sample_size floor of 10 [A1 — ASSUMED]
- `cargo bench` bench profile = release by default (no custom profile override confirmed at workspace level) [A4 — ASSUMED pending workspace Cargo.toml check]

---

## Metadata

**Confidence breakdown:**
- Criterion harness recipe: HIGH — all patterns directly read from existing bench files
- Feature-flag matrix invocations: HIGH — derived from verified Cargo.toml features block
- Sentinel function selection: HIGH — all 6 paths confirmed via lib.rs re-exports and existing bench imports
- Workload size caps: HIGH for elastic (CONCERNS.md verbatim), MEDIUM for other modules (extrapolated)
- Timing estimates for `sample_size` guidance: LOW (A2, A3 are assumed from O() extrapolation)
- Linker triage rule: HIGH — copied verbatim from PITFALLS.md Pitfall 8

**Research date:** 2026-08-07
**Valid until:** Stable — fdars-core is a published Rust crate and these patterns don't change between bench runs. Recheck if criterion version is updated or if `linalg`/`parallel` feature semantics change.
