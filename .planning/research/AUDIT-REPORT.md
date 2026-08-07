# fdars Performance & Functionality Audit Report

**Crate:** fdars-core v0.14.0
**Audit milestone:** v0.14.0 — audit-only, no production code changes
**Started:** 2026-08-07
**Status:** In progress (Phase 1 of 9)

---

## Phase 1 — Measurement Discipline & Baselines

Phase 1 establishes the benchmark measurement apparatus and records one sentinel per hot-path module.

### Methodology

- All benchmarks use the existing criterion 0.5 harness (`harness = false` `[[bench]]` entries).
- Benchmarks run under `cargo bench` (bench profile = release).  The binary path `target/release/deps/` is confirmed in the criterion output header before recording any numbers.
- Both inputs and outputs are wrapped in `criterion::black_box` (Pitfall 3 guard).
- Raw criterion stdout is saved under `.planning/research/bench/` using the naming convention `p1_<target>_<features>_run<N>.txt` (D-06).
- The feature-flag matrix (`""`, `parallel`, `linalg`, `linalg,parallel`) is exercised on one sentinel that genuinely differs across combos (D-04).

### 4-combo sentinel selection (D-04, Open Question A5 resolved)

**Original candidate:** `fdata_to_pc_1d` (FPCA/SVD module baseline sentinel, D-03).

**Finding (A5):** `fdata_to_pc_1d` was examined as the 4-combo feature-matrix sentinel but was found unsuitable:
- `center_columns` (`src/regression.rs` lines 167–181) uses plain sequential `for` loops.
- nalgebra SVD (`nalgebra::SVD`) is always sequential regardless of the `parallel` feature flag.
- Therefore `fdata_to_pc_1d` produces near-identical timings for the `parallel` vs non-`parallel` combos and cannot discriminate between them.

**Substituted sentinel:** `karcher_mean` (`fdars_core::alignment::karcher_mean`).
- `karcher_mean` uses `iter_maybe_parallel!` in its inner N-loop (`src/alignment/karcher.rs:185`).
- With `parallel` feature active the loop runs via rayon; without it, sequential.
- This produces genuinely different timings across the 4 combos, making it a valid D-04 discriminator.
- Cell: N=100, M=50 (keeps the 4 combo runs fast).

`fdata_to_pc_1d` remains as the D-03 module baseline sentinel for FPCA/SVD and is run at `linalg,parallel` for the module baseline record.

### Artifacts produced in Phase 1

| Artifact | Path | Status |
|----------|------|--------|
| Audit bench file | `fdars-core/benches/audit_hotpaths.rs` | Created |
| Cargo bench entry | `fdars-core/Cargo.toml` → `[[bench]] name = "audit_hotpaths"` | Added |
| Raw artifact directory | `.planning/research/bench/` | Created |
| FPCA sentinel run | `.planning/research/bench/p1_fpca_linalg,parallel_run1.txt` | Recorded |
| Karcher 4-combo runs | `.planning/research/bench/p1_karcher_*_run1.txt` (4 files) | Recorded |

### Phase 1 findings

Raw criterion results are in the `.planning/research/bench/` directory.

See §Methodology and §Workload Matrix below for the full discipline rules and per-module baseline numbers.

---

## §Methodology

This section documents the measurement discipline applied across all phases. Each later phase inherits these rules and must not deviate without a documented justification.

### Release-Mode Discipline

All benchmarks use `cargo bench` (not `cargo test --bench`). `cargo bench` compiles with the `bench` profile which has `opt-level = 3` and `debug = false` — equivalent to release.

**To confirm release mode:** The Criterion 0.5 binary path printed on the first output line must contain `/release/deps/`, e.g.:
```
Running benches/audit_hotpaths.rs (target/release/deps/audit_hotpaths-aea52eeb0c35d5bd)
```
If the path shows `/debug/deps/` — which happens when `cargo test --bench` is used instead of `cargo bench` — the numbers are 5–50× inflated (Pitfall 1) and must be discarded.

**This is recorded as the first fact in every bench artifact.**

### Feature-Flag Matrix

Four feature combinations are required. Because `default = ["parallel"]` in `fdars-core/Cargo.toml`, `--no-default-features` must be passed explicitly to disable rayon. The `linalg` feature enables faer 0.23 (requires Rust ≥ 1.84.0) and anofox-regression.

| Feature set | Command flag | What it tests |
|-------------|-------------|---------------|
| `""` | `--no-default-features` | Sequential, no linalg (WASM / minimal / CRAN build) |
| `parallel` | `--no-default-features --features parallel` | Default for most library users; rayon active |
| `linalg` | `--no-default-features --features linalg` | Ridge/faer paths, sequential |
| `linalg,parallel` | `--features linalg,parallel` | Full capability; primary audit comparison baseline |

**Primary audit build:** `linalg,parallel` — all phase baselines use this combo unless specified otherwise.

**4-combo sentinel requirement (D-04):** One sentinel per phase must be run across all 4 combos and show measurable differences to confirm the feature-flag matrix methodology is exercised end-to-end. In Phase 1 this is `karcher_mean` (see §4-combo sentinel selection above).

**Bench file constraint:** The audit bench (`audit_hotpaths.rs`) must call only APIs available under the leanest combo (`""`). No `linalg`-gated function (e.g., `ridge_regression_fit`) may be called unconditionally or Combos 1 and 2 will fail to compile (Pitfall 18).

### black_box on Inputs and Outputs

All benchmark inputs and outputs must be wrapped in `criterion::black_box` (Pitfall 3) to prevent the compiler from eliminating computations:

```rust
// For Vec<f64> / primitive returns — wrap both input and output:
b.iter(|| black_box(fraiman_muniz_1d(black_box(&data), black_box(&data), true)));

// For Result<T, E> where T is a large struct — wrapping inputs is sufficient:
b.iter(|| fdata_to_pc_1d(black_box(&data), black_box(5usize), black_box(&argvals)));

// For Result<Vec<f64>, _> — unwrap inside iter, wrap output if desired:
b.iter(|| nadaraya_watson(black_box(&x), black_box(&y), black_box(&x_new),
                           black_box(bandwidth), black_box("gaussian")).unwrap());
```

**Warning sign for missing black_box:** benchmark reports < 10 ns or 0% variance for a matrix operation.

### Toolchain Version Capture

The Rust toolchain version must be recorded as the first fact in every bench artifact. Current environment:

- `rustc 1.97.0 (2d8144b78 2026-07-07)` — stable-x86_64-unknown-linux-gnu
- `cargo 1.97.0 (c980f4866 2026-06-30)`
- **linalg feature floor:** Rust ≥ 1.84.0 (faer 0.23 requirement). The current 1.97.0 satisfies this.

Every raw artifact under `.planning/research/bench/` begins with:
```
=== ENVIRONMENT ===
rustc rustc 1.97.0 (2d8144b78 2026-07-07)
cargo cargo 1.97.0 (c980f4866 2026-06-30)
```
followed immediately by the criterion binary-path line confirming `/release/`.

### ±5% Two-Run Variance Rule

Each module sentinel is run **twice independently** under the same `linalg,parallel` build. The two-run median times are compared:

- **Variance ≤ 10%:** ACCEPTABLE — numbers are stable enough for the Phase 1 baseline.
- **Variance > 10%:** Tag the module's artifact `CONFIDENCE: LOW` and document in the report. The baseline is still recorded but marked as subject to re-measurement in later phases. It does NOT block Phase 1 completion.

**This threshold is conservative.** The ±5% figure in the heading refers to the acceptable noise band per measurement; a two-run spread > 10% indicates OS/scheduler interference beyond normal measurement noise.

### Criterion sample_size / measurement_time Configuration

Per-group settings are used (not global) so only the audit bench is affected. Recommended settings by module:

| Sentinel | sample_size | measurement_time | warm_up_time |
|----------|-------------|------------------|--------------|
| Elastic N=100, M=50 | 20 | 20s | 5s |
| FPCA N=500, M=200 | 20 | 20s | 5s |
| Depth N=500, M=200 | 30 | 15s | 3s |
| CV N=100, M=50 | 15 | 20s | 5s |
| Streaming N=500, M=200 | 30 | 15s | 3s |
| Smoothing N=500, M=200 | 30 | 10s | 3s |

Criterion 0.5 minimum `sample_size` is 10; going below panics. For very slow cells reduce `sample_size` to 10 and increase `measurement_time` instead.

### Artifact Naming Convention (D-06)

All raw criterion stdout artifacts are stored under `.planning/research/bench/` using the naming scheme:

```
p1_<target>_<features>_run<N>.txt
```

where:
- `<target>` ∈ {fpca, elastic, depth, cv, streaming, smooth, karcher, …}
- `<features>` ∈ {none, parallel, linalg, `linalg,parallel`}
- `<N>` ∈ {1, 2, …}

Example: `.planning/research/bench/p1_fpca_linalg,parallel_run1.txt`

Every finding in later phases must link to its artifact. A finding without a raw artifact backing it is not a valid finding (Pitfall 17).

### Infrastructure vs. Code Failure Triage (SC4)

**Infrastructure vs. Code Failure Triage:** This environment exhibits known Criterion 0.5 / doctest linker bus-error flakiness on Linux (Pitfall 8 in PITFALLS.md). A `cargo bench` or `cargo test --doc` invocation that exits via signal (SIGBUS) without printing a named `FAILED test_name` line is classified as an **infrastructure failure** and does not count as a fdars code defect. Failures of the form `FAILED test_name` are **code failures** and count. All bench artifacts record the full exit status alongside stdout to enable retroactive classification.

**Triage decision tree:**
```
IF output contains "FAILED test_name"   → code failure  → counts in defect list
IF output is "error: process didn't exit successfully"
   with NO named test_name               → infra failure → does NOT count
```

**Known environment cause:** `/tmp` tmpfs is at ~94% capacity. Criterion and doctest harnesses link their binaries in `/tmp`; a full tmpfs produces `LLVM ERROR: IO failure on output stream: No space left on device` (manifests as SIGBUS or SIGSEGV in the harness). This is an infrastructure failure per the triage rule.

**Mitigation:** `mkdir -p /home/simonm/.cache/fdars-bench-tmp && export TMPDIR=/home/simonm/.cache/fdars-bench-tmp` before running any bench or doctest command. Do not record the link failure as a benchmark result.

---

## §Workload Matrix

This section defines the per-module N×M candidate sizes for the full audit (PERF-02, D-07). All phases 3–9 benchmark against this shared size contract. Each module's cap is justified by its computational complexity.

**Candidate sizes (shared contract):** N ∈ {100, 500, 1000} × M ∈ {50, 200, 500}

### Per-Module N×M Cell Table

| Module | N cells | M cells | Cap | Cap rationale | Full-grid? |
|--------|---------|---------|-----|---------------|------------|
| Elastic alignment | {100, 500} | {50, 200} | N≤500, M≤200 | O(n²·m²) DP. N=1000×M=500: O(1000²×500²) = 250B ops ≈ 60s/iter (CONCERNS.md). N=500×M=200 ≈ 3.8s/iter (borderline; use measurement_time=60s). | No — 4 cells |
| FPCA/SVD | {100, 500, 1000} | {50, 200, 500} | None | O(m³) SVD. m=500 → ~1–2s/iter; use sample_size=10. N scaling is cheap (centering is O(n·m)). | Yes — 9 cells |
| Depth & distance | {100, 500, 1000} | {50, 200, 500} | None | O(n²·m) FM depth. Existing bench runs to N=2300 at M=200 (depth_benchmarks.rs). | Yes — 9 cells |
| CV loops | {100, 500} | {50, 200} | N≤500, M≤200 | Each fold runs FPCA O(m³) + classifier fit + predict; K=5 multiplies cost. N=1000×5 folds at M=200 ≈ 30s+/iter. | No — 4 cells |
| Streaming depth | {100, 500, 1000} | {50, 200, 500} | None | O(n·m) build + O(n·m) query. Very fast; existing bench reaches N=2300 at M=200. | Yes — 9 cells |
| Smoothing | {100, 500, 1000} | {50, 200, 500} | None | O(n·m) Nadaraya-Watson kernel eval. Existing bench reaches N=1000 at M=200. | Yes — 9 cells |

**Cap complexity sources:**
- Elastic cap: CONCERNS.md — "elastic_align_many() computes pairwise alignments, O(n² * m²) DP. For n=1,000, m=500, this becomes 250 million comparisons (~60 sec)."
- CV cap: Each cross-validation fold runs FPCA (O(m³)) + LDA/QDA fit + predict. At K=5, N=1000, M=200: 5 folds × (SVD at m=200 ≈ 100ms + fit) ≈ 500ms+ per iteration; at M=500 this scales to several seconds per iteration.

### Phase 1 Baseline Cells (D-03)

One cell per module was benchmarked at `linalg,parallel` with 2 independent runs:

| Module | Phase 1 Baseline Cell | Sentinel | Run 1 mean | Run 2 mean | Variance | Confidence | Artifact |
|--------|----------------------|----------|-----------|-----------|----------|------------|---------|
| FPCA/SVD | N=500, M=200 | `fdata_to_pc_1d` | 16.207 ms | 16.454 ms | 1.5% | OK | [run1](bench/p1_fpca_linalg,parallel_run1.txt) [run2](bench/p1_fpca_linalg,parallel_run2.txt) |
| Elastic | N=100, M=50 (capped) | `elastic_self_distance_matrix` | 789.80 ms | 816.80 ms | 3.4% | OK | [run1](bench/p1_elastic_linalg,parallel_run1.txt) [run2](bench/p1_elastic_linalg,parallel_run2.txt) |
| Depth | N=500, M=200 | `fraiman_muniz_1d` | 474.18 µs | 474.35 µs | 0.0% | OK | [run1](bench/p1_depth_linalg,parallel_run1.txt) [run2](bench/p1_depth_linalg,parallel_run2.txt) |
| CV loops | N=100, M=50 (capped) | `fclassif_cv` (lda, 5-fold) | 947.99 µs | 952.41 µs | 0.5% | OK | [run1](bench/p1_cv_linalg,parallel_run1.txt) [run2](bench/p1_cv_linalg,parallel_run2.txt) |
| Streaming depth | N=500, M=200 | `StreamingFraimanMuniz::depth_batch` | 491.23 µs | 545.90 µs | 11.1% | **LOW** | [run1](bench/p1_streaming_linalg,parallel_run1.txt) [run2](bench/p1_streaming_linalg,parallel_run2.txt) |
| Smoothing | N=500, M=200 | `nadaraya_watson` | 125.80 µs | 121.46 µs | 3.4% | OK | [run1](bench/p1_smooth_linalg,parallel_run1.txt) [run2](bench/p1_smooth_linalg,parallel_run2.txt) |

**Streaming depth LOW CONFIDENCE note:** Run2 had 3/30 high-severe outliers (OS scheduler jitter at sub-ms scale). The algorithm itself is O(n·m) and fast; the variance reflects measurement noise rather than algorithm instability. Later phases should re-run the streaming sentinel under `taskset`/`cpupower` for a stable baseline.

**Karcher mean 4-combo baseline (D-04):**

| Feature combo | Time | Notes |
|---------------|------|-------|
| `""` (sequential) | ~1555 ms | Baseline: sequential rayon-free |
| `parallel` | ~162 ms | 10× speedup — rayon active |
| `linalg` (sequential) | ~1555 ms | faer adds ridge path; loop still sequential |
| `linalg,parallel` | ~167 ms | 10× speedup — rayon + faer |

Artifacts: [karcher/none](bench/p1_karcher_none_run1.txt) [karcher/parallel](bench/p1_karcher_parallel_run1.txt) [karcher/linalg](bench/p1_karcher_linalg_run1.txt) [karcher/linalg,parallel](bench/p1_karcher_linalg,parallel_run1.txt)

---

---

## Phase 2 — Static Hot-Path Analysis

Phase 2 is a zero-runtime static analysis: no fdars-core source files are changed, no benchmarks are run, and no code is compiled. All findings are derived by reading source files under `fdars-core/src/` and recording file:line citations directly. The deliverable is the three sub-sections below, appended to this report (decision D-05: single growing report).

**SVD-copy site count:** The ROADMAP entry claims "8 FdMatrix→DMatrix SVD-copy sites." The verified count across all production source is **8 production `SVD::new(x.to_dmatrix(), …)` call sites** — this is correct. A ninth `to_dmatrix()` call exists at `matrix.rs:682` but is wrapped in `#[cfg(test)]` and compiled only into the test binary; it is excluded from the allocation hotspot list below (RESEARCH §2 / Pitfall 2).

### Complexity Table

Each row gives the dominant Big-O in N (number of curves) and M (number of evaluation points) separately — per RESEARCH Pitfall 4, N-scaling and M-scaling must not be conflated. The feature-gate tag records whether the complexity applies at the `[always]` level or is reduced only when a feature flag is active.

| Module | Primary function (file:line) | N complexity | M complexity | Feature gate | Fragile flag |
|--------|------------------------------|-------------|--------------|--------------|--------------|

### Allocation Hotspot List

All sites below allocate a new `DMatrix<f64>` of the stated size, pass it into `SVD::new`, and discard the `DMatrix` after the SVD completes. None of these sites cache or reuse the intermediate `DMatrix`. The `[always]` tag means the allocation occurs regardless of which Cargo feature flags are active. Phase 4 (dhat heap profiling) will measure each site's contribution to total heap traffic.

| Site (file:line) | Category | Enclosing fn | Alloc size | Feature gate | Phase target |
|------------------|----------|--------------|------------|--------------|--------------|

### Parallelism Gap List

Status values: **ALREADY PARALLEL** = the loop is wrapped in one of the five `iter_maybe_parallel!` / `slice_maybe_parallel!` family macros defined in `parallel.rs` and is feature-gated on the `parallel` Cargo feature. **SEQUENTIAL** = plain `for` loop with no parallelism macro; a gap candidate for Phase 5 parallelism work. The banding note records the opt-in banding behaviour that affects M-scaling without changing N-scaling.

| Loop (file:line) | Status | Parallelism macro | Feature gate tag | Gap candidate? |
|------------------|--------|-------------------|------------------|----------------|

*Full report sections (hot-path analysis, scikit-fda gap analysis, consolidated findings, prioritized backlog) to be written across Phases 2–9.*
