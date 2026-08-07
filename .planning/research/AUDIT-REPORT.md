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

**SVD-copy site count:** The ROADMAP entry claims "8 FdMatrix→DMatrix SVD-copy sites." The verified count across all production source is **8 production `SVD::new(x.to_dmatrix(), …)` call sites** — this is correct. One additional `to_dmatrix()` call exists in a `#[cfg(test)]` block (a round-trip test helper in `matrix.rs`) and is compiled only into the test binary; it is excluded from the allocation hotspot list below (RESEARCH §2 / Pitfall 2).

### Complexity Table

Each row gives the dominant Big-O in N (number of curves) and M (number of evaluation points) separately — per RESEARCH Pitfall 4, N-scaling and M-scaling must not be conflated. The feature-gate tag records whether the complexity applies at the `[always]` level or is reduced only when a feature flag is active.

| Module | Primary function (file:line) | N complexity | M complexity | Feature gate | Fragile flag |
|--------|------------------------------|-------------|--------------|--------------|--------------|
| Elastic alignment | `karcher_mean` → `karcher_mean_impl` (`alignment/karcher.rs:323`); secondary `elastic_self_distance_matrix` → `self_distance_matrix_impl` (`alignment/pairwise.rs:194`) | O(max_iter · N) — N loop uses `iter_maybe_parallel!`; N-scaling is **unchanged by banding** | Unbanded: O(m²) per pair (full DP table). Banded: O(m · band) per pair. Total: O(max_iter · N · m²) unbanded / O(max_iter · N · m · band) banded | `[parallel-gated]` — inner N-loop at `karcher.rs:185` uses `iter_maybe_parallel!`; M-loop (DP core) is always sequential | Banding is opt-in: default `karcher_mean()` passes `band_frac=0.0` → `None` (full DP). User must call `karcher_mean_banded()` to enable O(m · band) DP. |
| FPCA / SVD | `fdata_to_pc_1d` (`regression.rs:249`); centering via `center_columns` (`regression.rs:167`), weighted SVD at `regression.rs:298` | Centering: O(N · M) sequential double loop (`regression.rs:171–178`). Weighted scale: O(N · M) sequential loop (`regression.rs:292–295`). N-cost is linear and cheap. | Centering: O(M) per column — outer loop is over M columns, inner over N rows. SVD: O(min(N,M)² · max(N,M)) — dominated by O(M³) when M < N, O(N² · M) when N < M. Phase 1 baseline: M=200 → 16 ms, dominated by the nalgebra SVD step. | `[sequential]` — `center_columns` is a plain double `for` loop (`regression.rs:171`, `regression.rs:176`); nalgebra SVD is always sequential regardless of the `parallel` feature flag. The parallel `fdata.rs:center_1d` is a **different function** not called here (RESEARCH Pitfall 1). | No fragile flag. SVD is the M-scaling bottleneck; centering is secondary. |
| Depth & distance | `fraiman_muniz_1d` (`depth/fraiman_muniz.rs:32`); FM depth delegates immediately to `StreamingFraimanMuniz::depth_batch` (`streaming_depth/fraiman_muniz.rs:77`). Distance matrix via `lp_cross_1d` (`metric/lp.rs:40`, `iter_maybe_parallel!` at `metric/lp.rs:57`) and `lp_self_1d` (`metric/lp.rs:94`, `iter_maybe_parallel!` at `metric/lp.rs:105`). | FM depth: O(N_obj · N_ref) — outer loop over N_obj is `iter_maybe_parallel!` (`streaming_depth/fraiman_muniz.rs:82`); inner loop iterates over N_ref reference curves. Distance matrix: O(N²) upper-triangular pairs, each pair O(M). | FM depth: O(M) per (obj, ref) pair — integration over M evaluation points. Distance matrix: O(M) per pair for the Lp weighted integral (`metric/lp.rs:lp_weighted_distance`). Total FM: O(N_obj · N_ref · M). | `[parallel-gated]` — Open Question 1 resolved: `fraiman_muniz_1d` delegates to `depth_batch` which uses `iter_maybe_parallel!(0..nobj)` at `streaming_depth/fraiman_muniz.rs:82`. Distance loops also use `iter_maybe_parallel!` at `metric/lp.rs:57,105`. | No fragile flag. The static FM depth is fully covered by the streaming implementation; no separate FM-specific parallel path exists. |
| CV loops | `fclassif_cv` (`classification/cv.rs:45`); fold loop at `cv.rs:76` | O(nfold · N_train · cost_per_fold) where cost_per_fold = FPCA (`fdata_to_pc_1d`, itself sequential SVD) + classifier fit + predict. Each fold is independent but the outer `for fold in 0..nfold` is a plain sequential loop. N-scaling enters via the per-fold FPCA centering and SVD steps. | Per-fold cost dominated by FPCA SVD: O(min(N_train, M)² · max(N_train, M)) → O(M³) for typical M < N. Total: O(nfold · M³). M-scaling is the bottleneck. | `[sequential]` — the outer fold loop at `cv.rs:76` is a plain `for fold in 0..nfold` with no parallelism macro. Each fold also invokes the sequential FPCA path (`fdata_to_pc_1d` → sequential `center_columns` + nalgebra SVD). No RNG seeding concern for the fold loop itself (RNG used only in `assign_folds`, not in the fold body). | No fragile flag. Independent folds make this a safe Phase 5 `iter_maybe_parallel!` candidate; Phase 1 baseline: N=100, M=50 → 948 µs/iter. |
| Streaming depth | `StreamingFraimanMuniz::depth_batch` (`streaming_depth/fraiman_muniz.rs:77`); also `StreamingMBD::depth_batch` (`streaming_depth/mbd.rs:71`). Build cost in `SortedReferenceState::from_reference` (sort N_ref · M values). | O(N_obj · N_ref · M) for FM batch query — outer `iter_maybe_parallel!(0..nobj)` at `:82`, inner O(N_ref · M) per-curve depth evaluation. Build: O(N_ref · M · log N_ref) due to per-column sort in `from_reference`. N_obj-scaling is parallelized. | O(N_ref · M) per object curve for the depth evaluation (iterate over M grid points, for each do a binary search over N_ref sorted values → O(M · log N_ref)). Build: O(M · N_ref · log N_ref). | `[parallel-gated]` — `iter_maybe_parallel!(0..nobj)` at `streaming_depth/fraiman_muniz.rs:82` and `streaming_depth/mbd.rs:76`. | Phase 1 streaming sentinel tagged LOW CONFIDENCE (11.1% two-run variance, OS scheduler jitter at sub-ms scale). Re-measure under `taskset`/`cpupower` in later phases. |
| Smoothing | `nadaraya_watson` (`smoothing.rs:72`); outer parallel loop over prediction points via `slice_maybe_parallel!(x_new)` at `smoothing.rs:110`; inner sequential `for i in 0..n` over training points at `smoothing.rs:115`. | O(N_pred · N_train) kernel evaluations — N_pred-loop is `[parallel-gated]`, N_train-loop is always sequential. N-scaling for the training set enters the inner sequential loop only. | O(1) per kernel evaluation (fixed kernel function). Total: O(N_pred · N_train) kernel calls. M-scaling does not enter `nadaraya_watson` directly (it operates on scalar x/y, not on functional curves); the functional-data caller applies it per evaluation point. | `[parallel-gated]` — outer `slice_maybe_parallel!(x_new)` at `smoothing.rs:110` is feature-gated. Inner `for i in 0..n` at `smoothing.rs:115` is always sequential; this inner training-point loop is O(N_train) and low-value to parallelize (workload per iteration is a kernel evaluation, not matrix arithmetic). | No fragile flag. |

### Allocation Hotspot List

All sites below allocate a new `DMatrix<f64>` of the stated size, pass it into `SVD::new`, and discard the `DMatrix` after the SVD completes. None of these sites cache or reuse the intermediate `DMatrix`. The `[always]` tag means the allocation occurs regardless of which Cargo feature flags are active. Phase 4 (dhat heap profiling) will measure each site's contribution to total heap traffic.

| Site (file:line) | Category | Enclosing fn | Alloc size | Feature gate | Phase target |
|------------------|----------|--------------|------------|--------------|--------------|
| `elastic_fpca.rs:214` | to_dmatrix() SVD copy | `horiz_fpca` | n × m `DMatrix<f64>` (shooting vectors matrix) | `[always]` | Phase 4 dhat |
| `elastic_fpca.rs:317` | to_dmatrix() SVD copy | `joint_fpca` | n × (m+1+m) `DMatrix<f64>` (combined augmented+shooting matrix) | `[always]` | Phase 4 dhat |
| `elastic_fpca.rs:483` | to_dmatrix() SVD copy | `horiz_fpca_from_alignment` | n × m `DMatrix<f64>` (shooting vectors matrix) | `[always]` | Phase 4 dhat |
| `elastic_fpca.rs:584` | to_dmatrix() SVD copy | `joint_fpca_from_alignment` | n × (m+1+m) `DMatrix<f64>` (combined augmented+shooting matrix) | `[always]` | Phase 4 dhat |
| `elastic_fpca.rs:930` | to_dmatrix() SVD copy | `optimize_balance_c_raw` (inside `eval_c` closure) | n × (m_aug+m) `DMatrix<f64>` — allocated on **every golden-section iteration** (up to 20×) | `[always]` | Phase 4 dhat |
| `alignment/nd.rs:705` | to_dmatrix() SVD copy | ND elastic FPCA Gram matrix computation | m × m `DMatrix<f64>` (Gram matrix for ND phase FPCA) | `[always]` | Phase 4 dhat |
| `regression.rs:298` | to_dmatrix() SVD copy | `fdata_to_pc_1d` (core FPCA — always executed on every FPCA call) | n × m `DMatrix<f64>` (weighted centered data for SVD) | `[always]` | Phase 4 dhat |
| `spm/mfpca.rs:336` | to_dmatrix() SVD copy | `mfpca` (multivariate FPCA — stacked multi-variable matrix) | n × (sum of m_p across variables) `DMatrix<f64>` (column-stacked centered variables) | `[always]` | Phase 4 dhat |

**8 production `to_dmatrix()` SVD sites confirmed** (ROADMAP's "8" is correct). A 9th `to_dmatrix()` call exists at `matrix.rs` in a `#[cfg(test)]` block (round-trip test helper) and is compiled only into the test binary; it is excluded from this list (RESEARCH §2 / Pitfall 2).

#### DMatrix::from_column_slice — basis construction (NOT SVD copies)

These 14 sites construct `DMatrix<f64>` from flat basis-function buffers for least-squares fitting (B-spline, Fourier, P-spline, SSA trajectory matrix). They are a **distinct category** from the SVD-copy sites above: the optimization path here is solving a normal-equations or QR system, not a full SVD decomposition. Phase 4 dhat profiling should treat them separately (RESEARCH Pitfall 5 — do not conflate with the 8 to_dmatrix() SVD copies).

| Site (file:line) | Category | Enclosing fn | Alloc size | Feature gate | Phase target |
|------------------|----------|--------------|------------|--------------|--------------|
| `smooth_basis.rs:198` | from_column_slice basis | B-spline basis matrix construction (`compute_bspline_basis` or equivalent) | m × k `DMatrix<f64>` (evaluation grid × basis functions) | `[always]` | Phase 4 (secondary) |
| `smooth_basis.rs:199` | from_column_slice basis | Penalty matrix for B-spline roughness (companion to :198) | k × k `DMatrix<f64>` (basis × basis penalty) | `[always]` | Phase 4 (secondary) |
| `smooth_basis.rs:695` | from_column_slice basis | Full-grid basis matrix (adaptive-k path) | m × actual_k `DMatrix<f64>` | `[always]` | Phase 4 (secondary) |
| `smooth_basis.rs:696` | from_column_slice basis | Penalty matrix (adaptive-k path, companion to :695) | actual_k × actual_k `DMatrix<f64>` | `[always]` | Phase 4 (secondary) |
| `seasonal/ssa.rs:178` | from_column_slice basis | SSA trajectory matrix construction | L × K `DMatrix<f64>` (window × number-of-windows) | `[always]` | Phase 4 (secondary) |
| `basis/auto_select.rs:95` | from_column_slice basis | Basis matrix for automatic basis selection (candidate basis) | m × actual_nbasis `DMatrix<f64>` | `[always]` | Phase 4 (secondary) |
| `basis/auto_select.rs:128` | from_column_slice basis | Basis matrix for second candidate basis in selection | m × actual_nbasis `DMatrix<f64>` | `[always]` | Phase 4 (secondary) |
| `basis/fourier_fit.rs:68` | from_column_slice basis | Fourier basis matrix for least-squares smoothing | m × actual_nbasis `DMatrix<f64>` | `[always]` | Phase 4 (secondary) |
| `basis/projection.rs:113` | from_column_slice basis | Projection basis matrix (generic functional projection) | m × actual_nbasis `DMatrix<f64>` | `[always]` | Phase 4 (secondary) |
| `basis/projection.rs:117` | from_column_slice basis | Second basis matrix slot (companion or alternative basis) | m × nbasis `DMatrix<f64>` | `[always]` | Phase 4 (secondary) |
| `basis/projection.rs:119` | from_column_slice basis | Penalty matrix for projection basis | nbasis × nbasis `DMatrix<f64>` | `[always]` | Phase 4 (secondary) |
| `elastic_regression/regression.rs:274` | from_column_slice basis | Basis matrix in elastic regression (shape-on-scalar predictor) | m × actual_nbasis `DMatrix<f64>` | `[always]` | Phase 4 (secondary) |
| `elastic_regression/regression.rs:278` | from_column_slice basis | Penalty matrix for elastic regression basis (companion to :274) | penalty_k × penalty_k `DMatrix<f64>` | `[always]` | Phase 4 (secondary) |
| `elastic_regression/scalar_on_shape.rs:117` | from_column_slice basis | Basis matrix for scalar-on-shape regression | m × nbasis `DMatrix<f64>` | `[always]` | Phase 4 (secondary) |

Note: `elastic_regression/scalar_on_shape.rs:119` constructs the companion penalty matrix (`DMatrix::from_column_slice(nbasis, nbasis, &penalty_flat)`); it is included in the count above as the 14th site alongside :117.

#### Redundant clone — double n×m allocation before SVD

| Site (file:line) | Category | Enclosing fn | Alloc size | Feature gate | Phase target |
|------------------|----------|--------------|------------|--------------|--------------|
| `regression.rs:291` | redundant clone | `fdata_to_pc_1d` — `weighted = centered.clone()` followed immediately by in-place scale then SVD | n × m `FdMatrix` (a second full copy of the centered data, scaled in-place to produce the weighted matrix) | `[always]` | Phase 4 / Phase 6 zero-copy candidate |

At `regression.rs:284` the centering step produces `centered` (n×m). At `regression.rs:291` `weighted = centered.clone()` creates a second n×m allocation, which is then scaled in-place at lines 292–295 and passed to SVD at line 298. Both `centered` and `weighted` are live simultaneously. This is a double-size heap allocation: the `FpcaResult` retains `centered` for downstream use, so the clone cannot be trivially elided, but `weighted` could be stack-allocated or written into a pre-allocated buffer to avoid the heap copy.`

### Parallelism Gap List

Status values: **ALREADY PARALLEL** = the loop is wrapped in one of the five `iter_maybe_parallel!` / `slice_maybe_parallel!` family macros defined in `parallel.rs` and is feature-gated on the `parallel` Cargo feature. **SEQUENTIAL** = plain `for` loop with no parallelism macro; a gap candidate for Phase 5 parallelism work. The banding note records the opt-in banding behaviour that affects M-scaling without changing N-scaling.

| Loop (file:line) | Status | Parallelism macro | Feature gate tag | Gap candidate? |
|------------------|--------|-------------------|------------------|----------------|
| `karcher.rs:185` (`pre_center_template` — per-curve alignment in Karcher iteration) | ALREADY PARALLEL | `iter_maybe_parallel!(0..n)` | `[parallel-gated]` | No |
| `pairwise.rs:227` (`self_distance_matrix_impl` — upper-triangular pairwise distance loop) | ALREADY PARALLEL | `iter_maybe_parallel!(0..n)` | `[parallel-gated]` | No |
| `elastic_fpca.rs:701` (`shooting_vectors_from_psis` — `for i in 0..n` computing per-curve `inv_exp_map_sphere`) | SEQUENTIAL | none | `[sequential]` | Yes — Phase 5 candidate |
| `elastic_fpca.rs:720` (`build_augmented_srsfs` — `for i in 0..n` constructing augmented SRSF rows) | SEQUENTIAL | none | `[sequential]` | Yes — Phase 5 candidate |
| `elastic_fpca.rs:764` (`svd_scores_and_eigenvalues` — inner `for i in 0..n` score extraction per component) | SEQUENTIAL | none | `[sequential]` | Yes — Phase 5 candidate |
| Banding note: `karcher.rs:~300` (`karcher_mean()` passes `band_frac=0.0` → `None`) | BANDING OPT-IN | n/a — API design note | `[always]` | N/A — requires `karcher_mean_banded()` to enable O(m·band) DP |

*Full report sections (hot-path analysis, scikit-fda gap analysis, consolidated findings, prioritized backlog) to be written across Phases 2–9.*
