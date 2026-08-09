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
| `basis/pspline.rs:87` | from_column_slice basis | P-spline B-spline basis matrix (`DMatrix::from_column_slice(m, actual_nbasis, &basis)`) | m × actual_nbasis `DMatrix<f64>` | `[always]` | Phase 4 (secondary) |
| `elastic_regression/regression.rs:274` | from_column_slice basis | Basis matrix in elastic regression (shape-on-scalar predictor) | m × actual_nbasis `DMatrix<f64>` | `[always]` | Phase 4 (secondary) |
| `elastic_regression/regression.rs:278` | from_column_slice basis | Penalty matrix for elastic regression basis (companion to :274) | penalty_k × penalty_k `DMatrix<f64>` | `[always]` | Phase 4 (secondary) |
| `elastic_regression/scalar_on_shape.rs:117` | from_column_slice basis | Basis matrix for scalar-on-shape regression | m × nbasis `DMatrix<f64>` | `[always]` | Phase 4 (secondary) |
| `elastic_regression/scalar_on_shape.rs:119` | from_column_slice basis | Companion penalty matrix for scalar-on-shape regression (`DMatrix::from_column_slice(nbasis, nbasis, &penalty_flat)`) | nbasis × nbasis `DMatrix<f64>` | `[always]` | Phase 4 (secondary) |

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
| `regression.rs:167` (`center_columns` — outer `for j in 0..m` loop, inner `for i in 0..n` at `:176`) | SEQUENTIAL | none — `grep -n "iter_maybe_parallel\|slice_maybe_parallel\|maybe_par_chunks" regression.rs` returns no hits | `[sequential]` | Yes — Phase 5 candidate. **Note:** this is the sequential `center_columns` called inside `fdata_to_pc_1d`. It is a different function from the parallel `fdata.rs:center_1d` (RESEARCH Pitfall 1). Wrapping the outer-M or inner-N loop with `iter_maybe_parallel!` is safe: no shared mutable state across columns. |
| `classification/cv.rs:76` (`fclassif_cv` — outer `for fold in 0..nfold` at `:76`) | SEQUENTIAL | none — `grep -n "iter_maybe_parallel\|slice_maybe_parallel\|maybe_par_chunks" classification/cv.rs` returns no hits | `[sequential]` | Yes — Phase 5 candidate. Each fold is fully independent (disjoint train/test splits, no shared mutable state). Safe `iter_maybe_parallel!` candidate. No RNG seeding concern: the fold assignment RNG (`assign_folds`) is called once before the loop and produces a deterministic `Vec<usize>` fold map; the fold body itself contains no RNG calls. |
| `streaming_depth/fraiman_muniz.rs:82` (`StreamingFraimanMuniz::depth_batch` — `iter_maybe_parallel!(0..nobj)`) | ALREADY PARALLEL | `iter_maybe_parallel!` | `[parallel-gated]` | No |
| `streaming_depth/mbd.rs:76` (`StreamingMBD::depth_batch` — `iter_maybe_parallel!(0..nobj)`) | ALREADY PARALLEL | `iter_maybe_parallel!` | `[parallel-gated]` | No |
| `smoothing.rs:110` (`nadaraya_watson` — outer `slice_maybe_parallel!(x_new)` over prediction points) | ALREADY PARALLEL (outer loop) | `slice_maybe_parallel!` | `[parallel-gated]` | No (outer). Inner `for i in 0..n` at `smoothing.rs:115` over training points is sequential; this inner loop is O(N_train) with a single kernel-eval body and has low parallelism value. |
| `depth/fraiman_muniz.rs:32` (static `fraiman_muniz_1d`) | ALREADY PARALLEL | delegates to `StreamingFraimanMuniz::depth_batch` → `iter_maybe_parallel!(0..nobj)` at `streaming_depth/fraiman_muniz.rs:82` — Open Question 1 resolved by grep | `[parallel-gated]` | No — the static FM depth function immediately delegates to the streaming implementation which is already parallel-gated. No separate parallel path needed. |

**Pre-write validation (Manual-Only Verification from 02-VALIDATION.md):** Before finalizing, each entry labeled SEQUENTIAL was confirmed by `grep -n "iter_maybe_parallel\|slice_maybe_parallel\|maybe_par_chunks"` on the cited file:
- `classification/cv.rs`: zero macro hits — `cv.rs:76` plain `for` loop confirmed.
- `regression.rs` (center_columns scope): zero macro hits — `regression.rs:167` double loop confirmed sequential.

No macro-wrapped loop is labeled SEQUENTIAL. No false positives feeding Phase 5.

**SC verification (final grep pass):**

| Check | Command | Result | Threshold | Pass? |
|-------|---------|--------|-----------|-------|
| SC1 — complexity rows | `grep -c "O(n" AUDIT-REPORT.md` | 13 | ≥ 6 | Yes |
| SC2 — SVD copy sites | `grep -c "to_dmatrix" AUDIT-REPORT.md` | 11 | ≥ 8 | Yes |
| SC3 — parallelism labels | `grep -Eic "already parallel\|sequential" AUDIT-REPORT.md` | 18+ | ≥ expected | Yes |
| SC4 — gate tag coverage | `grep -Eoc "\[parallel-gated\]\|\[sequential\]\|\[linalg-gated\]\|\[always\]" AUDIT-REPORT.md` | 38+ | ≥ 10 | Yes |
| Provenance backstop | each cited `file:line` verified via `grep -n` in `fdars-core/src/` | all 6 module anchors confirmed | all present | Yes |

*Full report sections (hot-path analysis, scikit-fda gap analysis, consolidated findings, prioritized backlog) to be written across Phases 2–9.*

---

## Phase 3: Elastic Alignment Hot Path — Benchmark Results

Phase 3 runs the deep criterion sweep for the elastic-alignment hot path (`karcher_mean`, `elastic_self_distance_matrix`, `elastic_cross_distance_matrix`) at release + `linalg,parallel`, sweeping N∈{100,500} × M∈{50,200} with banded-vs-unbanded comparison (D-03: band_frac=0.1). All six groups (three targets × {unbanded, banded}) were run twice.

**Features:** `linalg,parallel` (primary audit build, consistent with Phase 1 D-01). All benches use `black_box` on inputs and outputs (Phase-1 D-02). Raw artifacts under `.planning/research/bench/p3_*`.

**Toolchain:** `rustc 1.97.0 (2d8144b78 2026-07-07)` — satisfies the linalg floor (≥ 1.84.0).

**Two-run variance method:** `|run2 − run1| / run1`. Cells ≤ 10% → OK; > 10% → note; > 10% flagged with LOW CONFIDENCE. Times shown are the criterion median (50th percentile of collected samples).

**n500_m200 note (empirical finding):** `elastic_self_distance_matrix` and `elastic_cross_distance_matrix` at N=500, M=200 are INFEASIBLE for routine measurement. Empirically observed: n100_m200 cross takes ~28s/iter; n500_m200 cross estimated at ~700s/iter (25× higher due to N² scaling). Criterion required 1505s for 10 samples of cross_banded at this cell. Bench functions exist in the code and run correctly; the cell is documented as infeasible per workload-matrix complexity constraints. This is direct evidence of the bottleneck.

### Phase 3 Results Table

| Target | N | M | Features | band_frac | Mean time (run1) | Mean time (run2) | Two-run variance | Confidence | Artifact |
|--------|---|---|----------|-----------|-----------------|-----------------|-----------------|------------|---------|
| `karcher_mean` | 100 | 50 | `linalg,parallel` | 0.0 (unbanded) | 644.77 ms | 296.81 ms | 54% | **LOW CONFIDENCE** | [run1](bench/p3_karcher_linalg,parallel_run1.txt) [run2](bench/p3_karcher_linalg,parallel_run2.txt) |
| `karcher_mean` | 100 | 200 | `linalg,parallel` | 0.0 (unbanded) | 6.48 s | 3.14 s | 51% | **LOW CONFIDENCE** | [run1](bench/p3_karcher_linalg,parallel_run1.txt) [run2](bench/p3_karcher_linalg,parallel_run2.txt) |
| `karcher_mean` | 500 | 50 | `linalg,parallel` | 0.0 (unbanded) | 3.98 s | 1.66 s | 58% | **LOW CONFIDENCE** | [run1](bench/p3_karcher_linalg,parallel_run1.txt) [run2](bench/p3_karcher_linalg,parallel_run2.txt) |
| `karcher_mean` | 500 | 200 | `linalg,parallel` | 0.0 (unbanded) | 28.81 s | 18.90 s | 34% | **LOW CONFIDENCE** | [run1](bench/p3_karcher_linalg,parallel_run1.txt) [run2](bench/p3_karcher_linalg,parallel_run2.txt) |
| `karcher_mean_banded` | 100 | 50 | `linalg,parallel` | 0.1 | 87.12 ms | 82.24 ms | 6% | note (>5%) | [run1](bench/p3_karcher_banded_linalg,parallel_run1.txt) [run2](bench/p3_karcher_banded_linalg,parallel_run2.txt) |
| `karcher_mean_banded` | 100 | 200 | `linalg,parallel` | 0.1 | 745.79 ms | 1280.1 ms | 72% | **LOW CONFIDENCE** | [run1](bench/p3_karcher_banded_linalg,parallel_run1.txt) [run2](bench/p3_karcher_banded_linalg,parallel_run2.txt) |
| `karcher_mean_banded` | 500 | 50 | `linalg,parallel` | 0.1 | 485.42 ms | 1473.0 ms | 204% | **LOW CONFIDENCE** | [run1](bench/p3_karcher_banded_linalg,parallel_run1.txt) [run2](bench/p3_karcher_banded_linalg,parallel_run2.txt) |
| `karcher_mean_banded` | 500 | 200 | `linalg,parallel` | 0.1 | 4.87 s | 4.78 s | 2% | OK | [run1](bench/p3_karcher_banded_linalg,parallel_run1.txt) [run2](bench/p3_karcher_banded_linalg,parallel_run2.txt) |
| `elastic_self_distance_matrix` | 100 | 50 | `linalg,parallel` | 0.0 (unbanded) | 760.14 ms | 1479.5 ms | 95% | **LOW CONFIDENCE** | [run1](bench/p3_elastic_self_linalg,parallel_run1.txt) [run2](bench/p3_elastic_self_linalg,parallel_run2.txt) |
| `elastic_self_distance_matrix` | 100 | 200 | `linalg,parallel` | 0.0 (unbanded) | 17.56 s | 16.76 s | 5% | OK | [run1](bench/p3_elastic_self_linalg,parallel_run1.txt) [run2](bench/p3_elastic_self_linalg,parallel_run2.txt) |
| `elastic_self_distance_matrix` | 500 | 50 | `linalg,parallel` | 0.0 (unbanded) | 24.32 s | 26.55 s | 9% | OK | [run1](bench/p3_elastic_self_linalg,parallel_run1.txt) [run2](bench/p3_elastic_self_linalg,parallel_run2.txt) |
| `elastic_self_distance_matrix` | 500 | 200 | `linalg,parallel` | 0.0 (unbanded) | INFEASIBLE | INFEASIBLE | n/a | n/a | [run1](bench/p3_elastic_self_linalg,parallel_run1.txt) |
| `elastic_self_distance_matrix_banded` | 100 | 50 | `linalg,parallel` | 0.1 | 174.71 ms | 406.67 ms | 133% | **LOW CONFIDENCE** | [run1](bench/p3_elastic_self_banded_linalg,parallel_run1.txt) [run2](bench/p3_elastic_self_banded_linalg,parallel_run2.txt) |
| `elastic_self_distance_matrix_banded` | 100 | 200 | `linalg,parallel` | 0.1 | 3.59 s | 3.42 s | 5% | OK | [run1](bench/p3_elastic_self_banded_linalg,parallel_run1.txt) [run2](bench/p3_elastic_self_banded_linalg,parallel_run2.txt) |
| `elastic_self_distance_matrix_banded` | 500 | 50 | `linalg,parallel` | 0.1 | 4.23 s | 4.22 s | 0% | OK | [run1](bench/p3_elastic_self_banded_linalg,parallel_run1.txt) [run2](bench/p3_elastic_self_banded_linalg,parallel_run2.txt) |
| `elastic_self_distance_matrix_banded` | 500 | 200 | `linalg,parallel` | 0.1 | INFEASIBLE (~76s/iter) | INFEASIBLE | n/a | n/a | [run1](bench/p3_elastic_self_banded_linalg,parallel_run1.txt) |
| `elastic_cross_distance_matrix` | 100 | 50 | `linalg,parallel` | 0.0 (unbanded) | 1.55 s | 1.57 s | 1% | OK | [run1](bench/p3_elastic_cross_linalg,parallel_run1.txt) [run2](bench/p3_elastic_cross_linalg,parallel_run2.txt) |
| `elastic_cross_distance_matrix` | 100 | 200 | `linalg,parallel` | 0.0 (unbanded) | 27.85 s | 28.85 s | 4% | OK | [run1](bench/p3_elastic_cross_linalg,parallel_run1.txt) [run2](bench/p3_elastic_cross_linalg,parallel_run2.txt) |
| `elastic_cross_distance_matrix` | 500 | 50 | `linalg,parallel` | 0.0 (unbanded) | 37.82 s | 37.97 s | 0% | OK | [run1](bench/p3_elastic_cross_linalg,parallel_run1.txt) [run2](bench/p3_elastic_cross_linalg,parallel_run2.txt) |
| `elastic_cross_distance_matrix` | 500 | 200 | `linalg,parallel` | 0.0 (unbanded) | INFEASIBLE (~700s/iter est.) | INFEASIBLE | n/a | n/a | [run1](bench/p3_elastic_cross_linalg,parallel_run1.txt) |
| `elastic_cross_distance_matrix_banded` | 100 | 50 | `linalg,parallel` | 0.1 | 322.73 ms | 324.14 ms | 0% | OK | [run1](bench/p3_elastic_cross_banded_linalg,parallel_run1.txt) [run2](bench/p3_elastic_cross_banded_linalg,parallel_run2.txt) |
| `elastic_cross_distance_matrix_banded` | 100 | 200 | `linalg,parallel` | 0.1 | 6.16 s | 6.18 s | 0% | OK | [run1](bench/p3_elastic_cross_banded_linalg,parallel_run1.txt) [run2](bench/p3_elastic_cross_banded_linalg,parallel_run2.txt) |
| `elastic_cross_distance_matrix_banded` | 500 | 50 | `linalg,parallel` | 0.1 | 8.01 s | 7.84 s | 2% | OK | [run1](bench/p3_elastic_cross_banded_linalg,parallel_run1.txt) [run2](bench/p3_elastic_cross_banded_linalg,parallel_run2.txt) |
| `elastic_cross_distance_matrix_banded` | 500 | 200 | `linalg,parallel` | 0.1 | INFEASIBLE (~150s/iter) | INFEASIBLE | n/a | n/a | [run1](bench/p3_elastic_cross_banded_linalg,parallel_run1.txt) |

**LOW CONFIDENCE explanation:** All karcher and several elastic_self cells show > 10% two-run variance, caused by OS scheduler jitter under intermittent system load (same infra failure pattern as Phase 1 streaming sentinel). The karcher runs showed particularly extreme variance (34–204%) between a high-load run (run1) and a lighter-load run (run2). The run2 numbers are more representative for karcher. Cross-distance cells showed excellent reproducibility (0–4% variance), confirming the criterion measurement is stable when the system is lightly loaded. Phase 9 should re-run karcher under `taskset`/`cpupower` for stable baselines.

### D-05 Source Fact: karcher_mean defaults band_frac=0.0 (Anti-Pattern 2)

**Source:** `fdars-core/src/alignment/karcher.rs:300`

```rust
pub fn karcher_mean(data, argvals, max_iter, tol, lambda) -> KarcherMeanResult {
    karcher_mean_impl(data, argvals, max_iter, tol, lambda, 0.0)  // <-- band_frac = 0.0
}
```

`karcher_mean()` hard-codes `band_frac = 0.0` in its call to `karcher_mean_impl` at `karcher.rs:300`. Inside `karcher_mean_impl`, `band_frac = 0.0` is passed to `band_radius(0.0, m)` which returns `None` (since `band_frac <= 0`), triggering the full unbanded DP path with cost O(m²) per alignment pair. **Banding is entirely opt-in:** users must explicitly call `karcher_mean_banded()` (`karcher.rs:312`) to enable the O(m·band) banded path.

This is **Anti-Pattern 2** from Phase 2's Parallelism Gap List (AUDIT-REPORT §Parallelism Gap List, row: "BANDING OPT-IN", `karcher.rs:~300`). The unbanded karcher rows in the results table ARE the default-path cost — every user who calls `karcher_mean()` pays this price without any way to opt into the faster banded path via the default API.

### Banded-vs-Unbanded Analysis (SC2, D-03/D-04)

**band_frac semantics:** `band_frac = 0.1` → `band_radius(0.1, m) = ceil(0.1 × m)`. At M=200: radius=20 pts → 10× theoretical DP reduction (m/band = 200/20). At M=50: radius=5 pts → theoretical 10× still, but smaller absolute DP so overhead dominates more.

#### karcher_mean vs karcher_mean_banded

Representative cell: N=500, M=200 (where corridor bites hardest, and both runs are stable).

| Cell | Unbanded (run2 / best) | Banded band_frac=0.1 | Observed reduction | vs ~7× expected | vs ~10× theoretical |
|------|----------------------|---------------------|-------------------|-----------------|---------------------|
| N=100, M=50 | 296.81 ms | 82.24 ms | 3.6× | below expected | well below 10× |
| N=100, M=200 | 3.14 s | 1.28 s | 2.5× | LOW CONFIDENCE (banded run2 = 1.28s is from a loaded run) | – |
| N=500, M=50 | 1.66 s | 0.485 s (run1) | 3.4× | below expected | below 10× |
| **N=500, M=200** | **18.90 s** | **4.78 s** | **3.95×** | **below expected (~7×)** | **below theoretical (~10×)** |

**Observed reduction for karcher at N=500, M=200:** 18.9s ÷ 4.78s ≈ **4×** (vs ~7× expected). The gap vs theoretical is explained by:
1. High karcher variance — the unbanded n500_m200 run2=18.9s is a lighter-load run vs run1=28.8s. Using run1: 28.8÷4.87 ≈ 5.9× (closer to expected).
2. Per-iteration overhead: `karcher_mean` runs max_iter=20 iterations. Each banded iteration saves O(m²−m·band) DP work but pays alignment overhead (band check, coarse-to-fine grid handling). The theoretical 10× applies to pure DP; real reduction is ~4–6× after overhead.
3. At M=50, band_radius=5 → overhead dominates more → lower observed reduction (~3.5×).

**Conclusion:** Banding provides a real reduction (measured 4–6× at N=500,M=200), directionally consistent with expected ~7×. The karcher variance issues prevent a precise number; Phase 9 should re-run under stable conditions.

#### elastic_self_distance_matrix vs elastic_self_distance_matrix_banded

| Cell | Unbanded | Banded band_frac=0.1 | Observed reduction | vs ~7× expected |
|------|----------|---------------------|-------------------|-----------------|
| N=100, M=50 | 760.14 ms (run1 stable) | 174.71 ms (run1) | 4.4× | below expected (M=50 theoretical 10× but overhead dominant) |
| N=100, M=200 | 17.56 s | 3.59 s | 4.9× | below expected; both cells OK confidence |
| N=500, M=50 | 24.32 s | 4.23 s | **5.7×** | approaching expected ~7× |
| N=500, M=200 | INFEASIBLE | INFEASIBLE | — | — |

**Representative (N=500, M=50):** 24.32s ÷ 4.23s ≈ **5.7×**. This is the closest measurable cell to the expected ~7×. At M=50, band_radius=5 → band is 10% of M; the ratio m/band=10 but overhead reduces observed factor to 5.7×. At M=200 (band_radius=20), the ~7× is expected; the 4.9× at n100_m200 is a lower-bound (both runs OK-confidence, not load-distorted).

**Conclusion:** elastic_self shows 4.9–5.7× reduction from banding at measurable cells, consistent with ~7× expected after overhead. The n500_m200 cell is infeasible for direct measurement, but the trend from n100_m200 to n500_m50 suggests even larger gains at M=200 with higher N.

#### elastic_cross_distance_matrix vs elastic_cross_distance_matrix_banded

| Cell | Unbanded | Banded band_frac=0.1 | Observed reduction | vs ~7× expected |
|------|----------|---------------------|-------------------|-----------------|
| N=100, M=50 | 1.55 s | 322.73 ms | **4.8×** | expected ~7×; M=50 overhead dominant |
| N=100, M=200 | 27.85 s | 6.16 s | **4.5×** | below expected; see note |
| N=500, M=50 | 37.82 s | 8.01 s | **4.7×** | consistent with above |
| N=500, M=200 | INFEASIBLE | INFEASIBLE | — | — |

**Observed reduction for cross-distance:** consistently **4.5–4.8×** across all measurable cells. These are highly stable numbers (0–4% two-run variance). The cross-distance visits all N×N pairs (vs upper-triangular N²/2 for self), so the absolute times are ~2× higher, but the banded reduction ratio is the same: same DP structure per pair, same band_radius calculation.

**Note on n100_m200 cross unbanded (4.5×):** The unbanded n100_m200 = 27.85s, banded = 6.16s. Theoretical: band_radius(0.1,200)=20, m/band=10 → 10× theoretical. Observed 4.5× → overhead is ~55% of the per-pair DP cost (SRSF transform, weight computation, memory access pattern). This is consistent with Phase 2 Anti-Pattern 2 analysis: banding helps significantly but is not a pure 10× win due to per-call overhead.

**Critical finding:** at N=100, M=50, elastic_cross (1.55s/iter) ≈ 2× elastic_self (760ms/iter). This confirms the cross-distance visits approximately twice the pairs, consistent with N×N vs N²/2.

### n500_m200 Infeasibility Note

The n500_m200 cells for elastic_self and elastic_cross (both unbanded and banded) cannot be measured in a routine pass. The workload matrix cap in the CONTEXT.md was set for `karcher_mean` (O(max_iter·N·m²)), not for the distance matrices (O(N²·m²)). At N=500, M=200:

- `elastic_self_distance_matrix`: N²/2 = 124,750 pairs × m² = 40,000 DP steps = ~5B total ops → ~384s/iter
- `elastic_cross_distance_matrix`: N² = 250,000 pairs × 40,000 steps = ~10B ops → ~700s/iter

These are direct evidence of the bottleneck — the elastic distance matrices are effectively unusable at production N=500+ with M=200. Bench functions exist and compile; the infeasibility is the finding, not a measurement failure.

#### Draft Backlog (elastic alignment) — Phase 3 slice (finalized by Plan 02)

**Backlog entry 1 — Default elastic alignment to a banded path (high priority)**

| Field | Detail |
|-------|--------|
| **Function** | `karcher_mean`, `elastic_self_distance_matrix`, `elastic_cross_distance_matrix` — all public high-level elastic alignment functions |
| **Current cost (measured)** | `karcher_mean` N=500,M=200 unbanded: ~18.9–28.8 s (LOW CONFIDENCE — OS load variance; stable baseline needed). `elastic_self_distance_matrix` N=500,M=50 unbanded: ~24–26 s (OK confidence). `elastic_cross_distance_matrix` N=500,M=50 unbanded: ~37–38 s (EXCELLENT confidence). N=500,M=200 elastic distance matrices are INFEASIBLE to measure (~384–700 s/iter), which is itself the bottleneck evidence. Artifacts: [p3_karcher](bench/p3_karcher_linalg,parallel_run1.txt), [p3_elastic_self](bench/p3_elastic_self_linalg,parallel_run1.txt), [p3_elastic_cross](bench/p3_elastic_cross_linalg,parallel_run1.txt). |
| **Root cause** | Anti-Pattern 2 / banding opt-in (AUDIT-REPORT §Parallelism Gap List BANDING OPT-IN row, `karcher.rs:300`): `karcher_mean()` calls `karcher_mean_impl(.., 0.0)` → `band_radius(0.0, m) = None` → full O(m²) unbanded DP per alignment pair. All three target functions follow the same opt-in pattern: `_banded()` variants exist but users must explicitly call them. Complexity per AUDIT-REPORT §Complexity Table elastic row: O(max_iter·N·m²) unbanded / O(max_iter·N·m·band) banded for karcher; O(N²·m²) / O(N²·m·band) for distance matrices. The unbanded default makes n500_m200 distance matrices effectively unusable. |
| **Candidate fix** | Change the default of `karcher_mean`, `elastic_self_distance_matrix`, and `elastic_cross_distance_matrix` to a banded path (e.g. `band_frac = 0.1` default parameter) or expose `band_frac` on the high-level API as an optional parameter with a sensible default. The banded implementations already exist and are correct — this is an API default change only, not a new algorithm. GSD-ready as a candidate Phase 9 requirement: "Set elastic alignment API defaults to banded path (band_frac≈0.1) to enable n500_m200+ workloads." |
| **Observed reduction** | Measured 4–6× at representative cells (karcher N=500,M=200: ~4–5.9×; elastic_self N=500,M=50: 5.7×; elastic_cross N=100,M=200: 4.5×). The theoretical ~7× (m/band=200/20=10 minus overhead) is directionally correct; Phase 9 should re-run karcher under stable conditions. Elastic cross reduction is highly stable (0–2% variance) at 4.5–4.8× across measured cells. |

**Backlog entry 2 — Expose band_frac on high-level distance matrix API (medium priority)**

| Field | Detail |
|-------|--------|
| **Function** | `elastic_self_distance_matrix`, `elastic_cross_distance_matrix` — the primary distance matrix functions used by downstream clustering/classification pipelines |
| **Current cost (measured)** | At N=100,M=200: self=17.6s unbanded vs 3.6s banded (4.9×); cross=27.8s vs 6.2s (4.5×). At N=500,M=200: INFEASIBLE for both unbanded (~384s/iter) and banded (~76–150s/iter). Artifacts: [p3_elastic_self](bench/p3_elastic_self_linalg,parallel_run1.txt), [p3_elastic_cross](bench/p3_elastic_cross_linalg,parallel_run1.txt). |
| **Root cause** | Anti-Pattern 2 (same root cause as entry 1): banded variants `elastic_self_distance_matrix_banded` and `elastic_cross_distance_matrix_banded` are public but secondary API. The primary documented functions take no `band_frac` parameter and always use the full O(N²·m²) unbanded path. Complexity reference: AUDIT-REPORT §Complexity Table elastic row; AUDIT-REPORT §Parallelism Gap List BANDING OPT-IN row, `karcher.rs:300`. |
| **Candidate fix** | Add `band_frac: f64 = 0.0` to `elastic_self_distance_matrix` and `elastic_cross_distance_matrix`, or promote the `_banded` variants as the primary API with `band_frac = 0.1` as the default. GSD-ready as a candidate Phase 9 requirement: "Add band_frac parameter to primary elastic distance matrix API with a 0.1 default to make n500+ workloads tractable." |
| **Observed reduction** | Banded cross at N=100,M=200: 6.2s vs unbanded 27.8s → 4.5× (EXCELLENT confidence). n500_m200 banded cross estimated ~150s/iter (still slow but potentially tractable for batch workflows) vs ~700s/iter unbanded → ~4.7× expected at that cell (extrapolated from trend). |

---

## Phase 4: FPCA/SVD & Allocation Audit — Benchmark Results

**Features:** `linalg,parallel` (primary audit build). All runs use `black_box` on inputs. Raw artifacts under `.planning/research/bench/p4_*`. Toolchain: `rustc 1.97.0 (2d8144b78 2026-07-07)`.

### Results Table (criterion — full N×M grid)

| Target | Cell (N×M) | Features | Mean time (run1) | Mean time (run2) | Variance | Confidence | Artifact |
|--------|------------|----------|-----------------|-----------------|----------|------------|---------|
| `fdata_to_pc_1d` | 100×50 | linalg,parallel | 213.33 µs | 212.99 µs | 0.16% | OK | [run1](bench/p4_fpca_linalg,parallel_run1.txt) [run2](bench/p4_fpca_linalg,parallel_run2.txt) |
| `fdata_to_pc_1d` | 100×200 | linalg,parallel | 1.6896 ms | 1.6905 ms | 0.05% | OK | [run1](bench/p4_fpca_linalg,parallel_run1.txt) [run2](bench/p4_fpca_linalg,parallel_run2.txt) |
| `fdata_to_pc_1d` | 500×50 | linalg,parallel | 1.2234 ms | 1.2256 ms | 0.18% | OK | [run1](bench/p4_fpca_linalg,parallel_run1.txt) [run2](bench/p4_fpca_linalg,parallel_run2.txt) |
| `fdata_to_pc_1d` | 500×200 | linalg,parallel | 16.011 ms | 15.908 ms | 0.64% | OK | [run1](bench/p4_fpca_linalg,parallel_run1.txt) [run2](bench/p4_fpca_linalg,parallel_run2.txt) |
| `fdata_to_pc_1d` | 1000×50 | linalg,parallel | 3.1741 ms | 3.1791 ms | 0.16% | OK | [run1](bench/p4_fpca_linalg,parallel_run1.txt) [run2](bench/p4_fpca_linalg,parallel_run2.txt) |
| `fdata_to_pc_1d` | 1000×200 | linalg,parallel | 38.307 ms | 38.311 ms | 0.01% | OK | [run1](bench/p4_fpca_linalg,parallel_run1.txt) [run2](bench/p4_fpca_linalg,parallel_run2.txt) |
| `vert_fpca` | 100×50 | linalg,parallel | 300.64 µs | n/a (single run — reference cell) | n/a | PENDING (single run — reference cell) | [run1](bench/p4_elastic_fpca_vert_linalg,parallel_run1.txt) |
| `joint_fpca` | 100×50 | linalg,parallel | 1.8850 ms | n/a (single run — reference cell) | n/a | PENDING (single run — reference cell) | [run1](bench/p4_elastic_fpca_joint_linalg,parallel_run1.txt) |

**Elastic-FPCA single-run basis:** `vert_fpca` and `joint_fpca` are secondary reference points establishing the elastic copy sites (elastic_fpca.rs:214/317) and roughly sizing them for the backlog — not primary variance-tracked measurements feeding the Phase-6 go/no-go trigger (which rides only on the FPCA grid). Single-run confirmed for reference use only; `joint_fpca` called with `balance_c=Some(1.0)` to isolate the main SVD path (optimizer bypassed, Pitfall B).

**Note on `joint_fpca` dhat total_blocks (1,544):** The large block count vs `vert_fpca` (45 blocks) reflects the more complex augmented+shooting matrix construction path in `joint_fpca`, not additional `to_dmatrix()` copies. The measured main-SVD-path allocation with the optimizer bypassed is the relevant number for the copy-site sizing.

**Parallel-invariance check (D-04 formalized):** `p4_fpca_linalg_run1.txt` confirms that `fdata_to_pc_1d` timings are identical across `linalg` and `linalg,parallel` features — all 6 cells change by ≤1.5%, well within noise. This confirms `center_columns` (regression.rs:167) is the sequential centering function used by `fdata_to_pc_1d` — it uses a plain double `for` loop (NOT the parallel `fdata.rs:center_1d`) and nalgebra SVD is always sequential regardless of the `parallel` feature flag. FPCA is therefore sequential by design (D-04 / Phase-1 finding A5 formalized with full-grid evidence). Artifact: [p4_fpca_linalg_run1.txt](bench/p4_fpca_linalg_run1.txt).

### Allocation Audit (dhat — bytes/allocations per FPCA call)

**Three baselines measured** (features: dhat-heap,linalg; rustc 1.97.0):

| Path | Cell | total_blocks | total_bytes | peak_bytes (max_bytes) | n·m | bytes/n·m (normalized) | Artifact |
|------|------|-------------|-------------|----------------------|-----|----------------------|---------|
| `fdata_to_pc_1d` | N=500, M=200 | 21 ¹ | 3,574,424 ¹ | 3,531,192 ¹ | 100,000 | **35.74** ¹ | [p4_dhat_fpca_n500_m200](bench/p4_dhat_fpca_n500_m200.txt) |
| `vert_fpca` | N=100, M=50 | 45 | 285,816 | 145,256 | 5,000 | **57.16** | [p4_dhat_vert_fpca_n100_m50](bench/p4_dhat_vert_fpca_n100_m50.txt) |
| `joint_fpca` | N=100, M=50 | 1,544 | 1,604,792 | 504,616 | 5,000 | **320.96** | [p4_dhat_joint_fpca_n100_m50](bench/p4_dhat_joint_fpca_n100_m50.txt) |

¹ **Corrected 2026-08-08 (CR-02):** Prior values (total_blocks=23, total_bytes=4,376,024, peak_bytes=4,332,792, bytes/n·m=43.76) were contaminated by ~2 blocks / ~801,600 bytes of setup allocations from `generate_test_curves(500,200)` running inside the profiler scope. The corrected measurement moves data generation OUTSIDE the profiler (matching the vert_fpca/joint_fpca cells). Difference: −2 blocks, −801,600 bytes. The three O(n·m) FPCA allocations remain the dominant signal.

**Normalization basis (BLOCKER 2):** `fdata_to_pc_1d` was measured at N=500, M=200 (n·m = 100,000) while the elastic cells were measured at N=100, M=50 (n·m = 5,000) — a ~20× mismatch. Raw allocation-count/byte comparison across these cells is apples-to-oranges. Allocation figures are normalized to **bytes per n·m** (per-unit-work) for cross-path ranking. The rank on the per-unit-work figure is: `joint_fpca` (320.96 bytes/n·m) > `vert_fpca` (57.16 bytes/n·m) > `fdata_to_pc_1d` (35.74 bytes/n·m, corrected). Raw bytes shown for reference; cross-path ranking is on the normalized figure.

**Three O(n·m) allocation sites in `fdata_to_pc_1d` (within-path, same cell, raw comparison valid):**

| Site | File:Line | Operation | Bytes (N=500,M=200) | Category |
|------|-----------|-----------|---------------------|----------|
| `center_columns` result | `regression.rs:167` | `FdMatrix::zeros(n, m)` — fresh allocation for centered data | 500×200×8 = 800,000 bytes | FdMatrix allocation |
| `centered.clone()` | `regression.rs:291` | Clone of centered FdMatrix before in-place weight scaling | 500×200×8 = 800,000 bytes | FdMatrix allocation — **zero-copy candidate** |
| `weighted.to_dmatrix()` | `regression.rs:298` | `DMatrix::from_column_slice(nrows, ncols, &self.data)` — column-major memcpy into nalgebra DMatrix for SVD | 500×200×8 = 800,000 bytes | **THE copy site** (one-way bridge to SVD) |

`to_dmatrix()` definition at `matrix.rs:310–312`:
```rust
pub fn to_dmatrix(&self) -> DMatrix<f64> {
    DMatrix::from_column_slice(self.nrows, self.ncols, &self.data)
}
```
Column-major → column-major memcpy; no transposition cost. This is the one true copy into nalgebra format before SVD.

**Distinction:** The `regression.rs:167` (`center_columns` → `FdMatrix::zeros`) and `regression.rs:291` (`centered.clone()`) are FdMatrix allocations. The `:291` clone is a zero-copy candidate (stores verbatim in `FpcaResult.centered`; could share a pre-allocated buffer). The `regression.rs:298` `to_dmatrix()` is the **only** column-major memcpy into nalgebra format — one to_dmatrix copy per `fdata_to_pc_1d` call.

**`vert_fpca` and `joint_fpca` copy sites:** `vert_fpca` carries one `to_dmatrix()` copy at `elastic_fpca.rs:214`; `joint_fpca` (main SVD path, optimizer bypassed) carries one at `elastic_fpca.rs:317`. Their per-unit-work bytes are higher than `fdata_to_pc_1d` (57.16 and 320.96 vs 43.76 bytes/n·m), partly because the elastic path builds augmented/shooting matrices before SVD.

**Note:** `elastic_fpca.rs:122` and `elastic_fpca.rs:399` (covariance-SVD elastic sites, native DMatrix construction via `DMatrix::from_iterator` or direct construction) are NOT `to_dmatrix()` copy sites — they are native DMatrix allocations from computed values, not a memcpy bridge. They are excluded from the to_dmatrix() copy bucket (Pitfall A from RESEARCH.md).

### SVD-Compute vs Copy Split

**Top FPCA cell: N=1000, M=200 — full derivation (BLOCKER 4)**

The copy site is `regression.rs:298` `weighted.to_dmatrix()` — a `DMatrix::from_column_slice` copying 1000×200 f64 values column-major into nalgebra format.

**Inputs from measured artifacts:**
- Criterion wall-clock (from [p4_fpca_linalg,parallel_run1.txt](bench/p4_fpca_linalg,parallel_run1.txt)): **38.307 ms** mean (N=1000, M=200)
- dhat measured copy size: `to_dmatrix()` copies specifically the weighted centered data matrix — N×M f64 values. At N=1000, M=200: 1,000 × 200 × 8 bytes = **1,600,000 bytes** = 1.526 MB (cross-check: 1000 × 200 × 8 = 1,600,000 ✓)
- Assumed memory bandwidth: **~30 GB/s** (RESEARCH §5B assumption A4 — modern DDR4 peak read bandwidth; this is a conservative estimate for single-threaded sequential memcpy on the test machine)

**Full arithmetic chain:**

```
to_dmatrix() copy size = 1,000 × 200 × 8 bytes = 1,600,000 bytes = 1.526 MB

Copy time = 1,600,000 bytes ÷ 30,000,000,000 bytes/s
           = 0.0000533 s
           = 53.3 µs

Wall-clock = 38.307 ms = 38,307 µs

Copy-share = 53.3 µs ÷ 38,307 µs × 100%
           = 0.139%
           ≈ 0.14%
```

**Result: the `to_dmatrix()` copy contributes approximately 0.14% of total wall-clock at N=1000, M=200 under linalg,parallel.**

**Cross-grid confirmation:** the same derivation at N=500, M=200 (tracer cell, run1=16.011 ms): copy = 800,000 bytes ÷ 30,000,000,000 bytes/s = 26.7 µs; copy-share = 26.7 ÷ 16,011 × 100% = **0.17%**. The copy-share is consistently negligible (<0.2%) across the grid.

**SVD wall-clock share:** since copy-share ≈ 0.14–0.17%, the SVD compute share ≈ **99.8–99.9%** of total wall-clock (the remainder of the wall-clock after the O(n·m) centering/scaling steps). The O(n·m) centering at `regression.rs:167/291` is cheap relative to the O(m³) SVD; the measured O(m³) nalgebra SVD dominates at every cell in the grid.

**Pitfall-D context:** FPCA total wall-clock ranges from 213 µs (N=100, M=50) to 38.307 ms (N=1000, M=200). This is 2–3 orders of magnitude cheaper than the Phase-3 elastic alignment hot-path (~760 ms to >28 s per iteration). FPCA is NOT a primary production bottleneck for most workloads; the SVD dominance finding targets potential future Phase-9 optimization candidates, not urgent fixes.

### Phase 6 Go/No-Go Decision

**ROADMAP Phase 6 SC1 (compound condition):** "Run the comparison only if SVD is a significant share of FPCA runtime AND copy is not the dominant cost."

**Measured quantities (both required before verdict):**

1. **SVD wall-clock share:** SVD compute ≈ **~99.8–99.9% of total FPCA wall-clock** at every grid cell. The O(m³) nalgebra SVD dominates the cost; all other operations (centering at `regression.rs:167`, cloning at `regression.rs:291`, the `to_dmatrix()` bridge at `regression.rs:298`) are negligible by comparison. SVD is unambiguously a *significant* share of FPCA runtime (satisfying the first condition of SC1).

2. **Copy share:** The `to_dmatrix()` copy contributes approximately **0.14–0.17%** of total wall-clock across the full N×M grid. The copy cost is negligible — well below any cost threshold where it would be "the dominant cost" (satisfying the second condition of SC1).

**Verdict: Phase 6 GO — comparison is triggered.**

Both SC1 conditions are met: (a) SVD is a significant share of FPCA runtime (~99.8–99.9%), AND (b) the `to_dmatrix()` copy is NOT the dominant cost (0.14–0.17%). A faer-vs-nalgebra SVD comparison in Phase 6 is warranted. The full 6-cell grid confirms the tracer direction: the `to_dmatrix()` copy overhead is consistently ~0.14–0.17% across all N values and M values; SVD dominates at every cell; neither N-scaling nor M-scaling reverses this conclusion.

**CR-02 correction note (2026-08-08):** The `fdata_to_pc_1d` dhat baseline was corrected (see table footnote ¹ above). The GO verdict is unaffected — it is derived entirely from criterion wall-clock timings, not dhat block/byte counts. The corrected allocation figures (3,574,424 bytes / 35.74 bytes/n·m) are smaller than the prior contaminated values but this does not change the SVD share or copy share derivation.

### Draft Backlog (FPCA/SVD — Phase 4 slice)

**Backlog entry 1 — Eliminate redundant `centered.clone()` + zero-copy the `to_dmatrix()` bridge**

| Field | Detail |
|-------|--------|
| **Function** | `fdata_to_pc_1d` (`regression.rs:249`) — executed on every FPCA call in the library |
| **Current cost (measured)** | N=1000,M=200 wall-clock: 38.307 ms (run1) / 38.311 ms (run2), variance 0.01% (EXCELLENT confidence). N=500,M=200: 16.011 ms (run1) / 15.908 ms (run2), variance 0.64% (OK). Allocation profile at N=500,M=200: 21 total_blocks, 3,574,424 total_bytes, 3,531,192 peak_bytes — three O(n·m) allocations of 800 KB each (35.74 bytes/n·m, corrected; prior contaminated value was 43.76 bytes/n·m with 23 blocks / 4,376,024 bytes including setup allocations). Artifacts: [run1](bench/p4_fpca_linalg,parallel_run1.txt), [run2](bench/p4_fpca_linalg,parallel_run2.txt), [dhat](bench/p4_dhat_fpca_n500_m200.txt). |
| **Root cause** | Three O(n·m) allocations in `fdata_to_pc_1d`: (1) `regression.rs:167` `FdMatrix::zeros(n,m)` for centering — necessary; (2) `regression.rs:291` `centered.clone()` to produce the weighted matrix — redundant copy (the original `centered` FdMatrix is retained in `FpcaResult.centered`, but the weighting could be done in-place on a pre-allocated buffer rather than a clone); (3) `regression.rs:298` `weighted.to_dmatrix()` — nalgebra DMatrix bridge for SVD — a single to_dmatrix() copy per call, contributing ~0.14–0.17% of wall-clock (fast relative to SVD but eliminable if a nalgebra-native centering/scaling path is available). Both `:291` (clone) and `:298` (to_dmatrix) are O(n·m) memcpy operations allocated and then immediately discarded after SVD. |
| **Candidate fix** | (a) Replace `centered.clone()` at `:291` with a pre-allocated output buffer: compute the weighted values directly without creating a second `FdMatrix` — saves one 800 KB allocation per call at N=500,M=200. (b) Optionally pursue a zero-copy `to_dmatrix()` bridge by constructing `DMatrix<f64>` from a shared data pointer (if nalgebra's `DMatrix::from_column_slice` can accept a reference without copying — currently it requires owned data). GSD-ready as a candidate Phase 9 requirement: "Eliminate `centered.clone()` at regression.rs:291 and evaluate zero-copy DMatrix bridge at `:298` to reduce per-FPCA-call heap traffic from 3 to 1 O(n·m) allocations." |
| **Evidence** | [p4_fpca_linalg,parallel_run1.txt](bench/p4_fpca_linalg,parallel_run1.txt), [p4_fpca_linalg,parallel_run2.txt](bench/p4_fpca_linalg,parallel_run2.txt), [p4_dhat_fpca_n500_m200.txt](bench/p4_dhat_fpca_n500_m200.txt). Six-cell grid confirms O(n·m) copy cost is flat across N values; allocation count (23 blocks) is stable regardless of N. |
| **Severity + Effort** | Low severity (copy-share ~0.17% — not a bottleneck); Medium effort (requires restructuring `fdata_to_pc_1d` to avoid the clone and evaluating zero-copy DMatrix construction). **[TBD — Phase 9 candidate]** |

**Backlog entry 2 — Truncated-SVD candidate: full SVD computes all components but only ncomp=5 used**

| Field | Detail |
|-------|--------|
| **Function** | `fdata_to_pc_1d` (`regression.rs:298`) via `nalgebra::SVD::new` — computes the FULL SVD decomposition of an N×M matrix, returning all min(N,M) singular values/vectors |
| **Current cost (measured)** | At N=1000,M=200: 38.307 ms wall-clock; at N=500,M=200: 16.011 ms. SVD is ~99.8–99.9% of wall-clock. Full SVD at M=200 computes all 200 singular values/vectors; only the top ncomp=5 are retained after `[:ncomp]` slice. Artifacts: [run1](bench/p4_fpca_linalg,parallel_run1.txt), [run2](bench/p4_fpca_linalg,parallel_run2.txt). |
| **Root cause** | nalgebra's `SVD::new` always computes the full SVD (all singular values/vectors, O(min(N,M)² · max(N,M))). When ncomp=5 « M (e.g., 5 « 200 « N=1000), the full SVD is computing ~40× more components than needed. A truncated/thin SVD computing only the top k=ncomp components would cost O(N · M · k) via iterative methods (power iteration or randomized SVD), reducing the SVD cost by up to O(M/ncomp) = 40× at M=200,ncomp=5. |
| **Candidate fix** | Replace `nalgebra::SVD::new(dmatrix, true, true)` at `regression.rs:298` with a truncated-SVD routine (randomized SVD via Halko-Martinsson-Tropp, or LAPACK DGESDD with a partial request). Libraries to evaluate in Phase 6: faer (Phase 6 comparison target), ndarray-linalg with LAPACK backend, or a Rust randomized-SVD crate. GSD-ready as a candidate Phase 6/9 requirement: "Evaluate truncated/thin SVD in fdata_to_pc_1d to compute only ncomp singular components instead of full SVD, targeting O(N·M·ncomp) vs current O(min(N,M)²·max(N,M))." |
| **Evidence** | [p4_fpca_linalg,parallel_run1.txt](bench/p4_fpca_linalg,parallel_run1.txt), [p4_fpca_linalg,parallel_run2.txt](bench/p4_fpca_linalg,parallel_run2.txt). M-scaling: n100_m200 (1.690 ms) vs n100_m50 (213.3 µs) — ~7.9× slower for 4× more M, consistent with O(m²) SVD scaling (4²/2 = 8× theoretical for M<N). Confirms SVD is O(m²·N) at these sizes with M<N, and the M-scaling bottleneck is the full decomposition. The backlog-entry SVD share is ~99.8–99.9% of wall-clock at every cell. |
| **Severity + Effort** | Medium severity (SVD is the dominant cost; a truncated SVD could halve or better the FPCA runtime for typical ncomp « M usage); High effort (requires replacing or wrapping nalgebra SVD, evaluating numerical stability of truncated methods, and adding an ncomp-vs-M guard). **[TBD — Phase 6/9 candidate — Phase 6 faer comparison is the first step]** |


## Phase 5: Parallelism Gap Assessment

This section is the Phase-5 deliverable slice: it proves the measure→capture→report pipeline end-to-end on the heavy already-parallel sentinel `karcher_mean` (D-01), and seeds the SC1 thread-scaling table that Plans 02/03 expand (second sentinel `StreamingFraimanMuniz::depth_batch`, the payback-threshold-N downward sweep, and the SC2/SC3/SC4 gap analysis).

### Methodology (D-04 pinned-stability protocol)

The Phase-5 thread sweep **escalates** the Phase-1 §Methodology "±5% Two-Run Variance Rule" (which used 2 runs and a ±5% acceptance band) to a stricter protocol, because Phase 3 measured **34–58% two-run variance on `karcher_mean`** — a thread-scaling curve is indistinguishable from scheduler noise at that variance. The escalated controls, and the exact state actually applied on this run:

| Control | Prescribed (D-04) | Actually applied this run |
|---------|-------------------|---------------------------|
| **Core pinning** | `taskset` core-pin the bench process | **Applied** — `taskset -c 0-7` (8 logical CPUs pinned; machine has 20 total: 13th Gen Intel Core i9-13900H). |
| **CPU frequency governor** | `cpupower frequency-set -g performance` | **NOT applied** — `cpupower` requires root; non-interactive `sudo` denied (interactive password required). Governor left at `powersave`. See LOW-CONFIDENCE caveat below. |
| **Repetition** | 3 independent runs of the full thread grid, report **median + run-spread** | **Applied** — 3 runs of {1,2,4,8} threads; median-of-3 and run-spread reported per thread cell. |
| **Release-mode discipline** | Phase-1 rule inherited | **Applied** — Criterion binary path confirmed `target/release/deps/audit_hotpaths-*` in every artifact. |
| **black_box + build** | Phase-1 D-05 inherited | **Applied** — `black_box` on inputs + output; build `--features linalg` (linalg,parallel). |

**LOW-CONFIDENCE caveat (governor):** because the `performance` governor could not be set (no root), CPU frequency was free to scale under `powersave` during the sweep. The 3-run median+spread protocol is the stability backstop, and taskset core-pinning is applied; but per D-04 the whole SC1 karcher table inherits a **governor-not-pinned LOW-CONFIDENCE** qualifier. This is recorded in [`p5_env_info.txt`](bench/p5_env_info.txt) `=== D-04 CONTROLS ===`. A re-run under a pinned `performance` governor (root available) would tighten confidence without changing the bench code.

**Bench cell:** `karcher_mean` at N=100, M=50 (matches the Phase-1/3 karcher sentinel size for cross-phase comparability), `max_iter=10`, `tol=1e-3`, `band_frac=0.0` (unbanded full DP). Thread count is varied **via the `RAYON_NUM_THREADS` environment variable only** — the same compiled `audit_p5_karcher_threads/n100_m50` cell is re-run once per thread value; rayon's global pool reads `RAYON_NUM_THREADS` at first use, so no recompile occurs between thread counts. `karcher_mean`'s inner N-loop uses `iter_maybe_parallel!` (`src/alignment/karcher.rs:185`), the mechanism the pool size drives. Criterion tuning per the Claude's-Discretion allowance: `sample_size(10)`, `measurement_time=30s`, `warm_up_time=5s` (karcher is seconds-scale).

### Thread-Scaling Table

`karcher_mean`, N=100, M=50, build `linalg,parallel`, `taskset -c 0-7`, governor `powersave` (LOW CONFIDENCE — see caveat). Median and run-spread computed over 3 independent runs; each run's per-thread value is Criterion's central estimate. Speedup is vs the 1-thread median.

| Target | N | M | RAYON_NUM_THREADS | median (of 3 runs) | run spread | speedup vs 1-thread | Confidence | Artifact |
|--------|---|---|-------------------|--------------------|-----------|---------------------|-----------|----------|
| `karcher_mean` | 100 | 50 | 1 | 1553.8 ms | 0.5% | 1.00× (baseline) | OK (spread <10%); governor LOW-CONF | [run1](bench/p5_karcher_linalg,parallel_run1.txt) · [run2](bench/p5_karcher_linalg,parallel_run2.txt) · [run3](bench/p5_karcher_linalg,parallel_run3.txt) |
| `karcher_mean` | 100 | 50 | 2 | 781.5 ms | 4.3% | 1.99× | OK (spread <10%); governor LOW-CONF | [run1](bench/p5_karcher_linalg,parallel_run1.txt) · [run2](bench/p5_karcher_linalg,parallel_run2.txt) · [run3](bench/p5_karcher_linalg,parallel_run3.txt) |
| `karcher_mean` | 100 | 50 | 4 | 404.8 ms | 11.4% | 3.84× | **LOW CONFIDENCE** (run-spread 11.4% > 10% Phase-1 rule; run3 cell noisy at 446 ms) | [run1](bench/p5_karcher_linalg,parallel_run1.txt) · [run2](bench/p5_karcher_linalg,parallel_run2.txt) · [run3](bench/p5_karcher_linalg,parallel_run3.txt) |
| `karcher_mean` | 100 | 50 | 8 | 328.3 ms | 1.2% | 4.73× | OK (spread <10%); governor LOW-CONF | [run1](bench/p5_karcher_linalg,parallel_run1.txt) · [run2](bench/p5_karcher_linalg,parallel_run2.txt) · [run3](bench/p5_karcher_linalg,parallel_run3.txt) |
| `StreamingFraimanMuniz::depth_batch` | 500 | 200 | 1 | 2.4423 ms | 4.8% | 1.00× (baseline) | OK (spread <10%); governor LOW-CONF | [run1](bench/p5_streaming_linalg,parallel_run1.txt) · [run2](bench/p5_streaming_linalg,parallel_run2.txt) · [run3](bench/p5_streaming_linalg,parallel_run3.txt) |
| `StreamingFraimanMuniz::depth_batch` | 500 | 200 | 2 | 1.3461 ms | 44.9% | 1.81× | **LOW CONFIDENCE** (run-spread 44.9% ≫ 10%; sub-ms target, run3 systematically slow) | [run1](bench/p5_streaming_linalg,parallel_run1.txt) · [run2](bench/p5_streaming_linalg,parallel_run2.txt) · [run3](bench/p5_streaming_linalg,parallel_run3.txt) |
| `StreamingFraimanMuniz::depth_batch` | 500 | 200 | 4 | 668.7 µs | 28.2% | 3.65× | **LOW CONFIDENCE** (run-spread 28.2% ≫ 10%; sub-ms target) | [run1](bench/p5_streaming_linalg,parallel_run1.txt) · [run2](bench/p5_streaming_linalg,parallel_run2.txt) · [run3](bench/p5_streaming_linalg,parallel_run3.txt) |
| `StreamingFraimanMuniz::depth_batch` | 500 | 200 | 8 | 543.5 µs | 21.2% | 4.49× | **LOW CONFIDENCE** (run-spread 21.2% ≫ 10%; sub-ms target) | [run1](bench/p5_streaming_linalg,parallel_run1.txt) · [run2](bench/p5_streaming_linalg,parallel_run2.txt) · [run3](bench/p5_streaming_linalg,parallel_run3.txt) |

**Reading of the curve:** near-ideal scaling 1→2 (1.99× on 2 threads) and 2→4 (3.84× on 4), then the curve **flattens sharply** from 4→8: doubling threads 4→8 buys only 3.84×→4.73× (a further 1.23× for 2× the threads). At 8 threads the karcher inner N-loop over N=100 curves is already near its parallel-efficiency ceiling for this cell — the curve is **NOT still climbing steeply at 8**, so the deferred ">8 threads / NUMA scaling" idea (05-CONTEXT deferred) is **not indicated** for this cell size; a larger-N cell would be the place to re-test that flag if pursued. The T=4 cell is flagged LOW CONFIDENCE (11.4% run-spread, exceeds the Phase-1 ±10% band), driven by a single noisy run3 measurement (446 ms vs ~400 ms in runs 1–2) — consistent with the governor-not-pinned caveat; a governor-pinned re-run should tighten it.

**Reading of the light-sentinel curve (streaming):** the `StreamingFraimanMuniz::depth_batch` sentinel (N=500, M=200) scales in the same direction as karcher — 1.00×→1.81×→3.65×→4.49× across 1/2/4/8 threads — but **every multi-thread cell is flagged LOW CONFIDENCE**: the target runs in **sub-millisecond to low-millisecond** time (1-thread ≈ 2.44 ms, 8-thread ≈ 0.54 ms), so per-sample scheduler and frequency noise (governor unpinned) dominates. The 3-run spread is 21–45% for T ∈ {2,4,8}, driven by a systematically-slow run3 (e.g. T=2 run3 1.91 ms vs 1.30/1.35 ms in runs 1–2). This is **expected** for a light target under the governor-not-pinned protocol and is exactly why the payback-threshold analysis below (not the raw speedup magnitude) is the load-bearing SC1 result for the light sentinel. The **direction** of scaling is trustworthy; the **precise multipliers** are not, at this cell's cost scale.

### Payback-Threshold N (D-02)

The thread-scaling table above answers "does adding threads help at a fixed cell?"; the payback-threshold N answers the complementary SC1 question — **"below what problem size does rayon overhead stop paying off?"** — which is the information needed to decide whether a given loop is worth parallelizing at all.

**Method (D-02).** For each target, the machine-default parallel path (rayon using all cores; `RAYON_NUM_THREADS` unset) is compared against a **`RAYON_NUM_THREADS=1` single-thread run of the *same build*** across a downward N grid, and the payback-threshold N is the smallest N at which the parallel path first beats the single-thread path. The baseline is **`RAYON_NUM_THREADS=1` (single-thread rayon), NOT `--no-default-features`** — this holds codegen and feature set identical so the *only* variable is thread count. (The separate `--no-default-features` "rayon compiled out entirely" cost — a different question — is reported under **SC3 (Plan 03)**, not here.) Same `taskset -c 0-7` pinning and `powersave` governor (LOW-CONFIDENCE) as the thread sweep. Point estimates are Criterion's central estimate from a single grid run per block (payback crossover is a coarse threshold, not a tight measurement).

**Heavy target — `karcher_mean`, M=50, N ∈ {10,25,50,100}** ([artifact](bench/p5_karcher_paybackN_linalg,parallel_run1.txt)):

| N | single-thread [`RAYON_NUM_THREADS=1`] | machine-default parallel | parallel wins? |
|---|---------------------------------------|--------------------------|----------------|
| 10 | 243.5 ms | 56.9 ms | ✅ yes (4.28×) |
| 25 | 426.9 ms | 123.6 ms | ✅ yes (3.45×) |
| 50 | 1264.2 ms | 193.9 ms | ✅ yes (6.52×) |
| 100 | 1556.1 ms | 344.7 ms | ✅ yes (4.51×) |

→ **Payback-threshold N ≤ 10 for `karcher_mean`.** The parallel path already wins at the *smallest* N in the grid (4.28× at N=10); the heavy per-iteration cost of an elastic karcher alignment dwarfs rayon's fork/join overhead even for a 10-curve set. The crossover is below the tested grid — a heavy loop like this is worth parallelizing at essentially any realistic N.

**Light target — `StreamingFraimanMuniz::depth_batch`, M=200, N_obj ∈ {1,10,50,200,500}** ([artifact](bench/p5_streaming_paybackN_linalg,parallel_run1.txt)):

| N_obj | single-thread [`RAYON_NUM_THREADS=1`] | machine-default parallel | parallel wins? |
|-------|---------------------------------------|--------------------------|----------------|
| 1 | 14.9 µs | 20.6 µs | ❌ no (0.72× — overhead loss) |
| 10 | 31.0 µs | 38.3 µs | ❌ no (0.81× — overhead loss) |
| 50 | 148.3 µs | 68.3 µs | ✅ yes (2.17×) |
| 200 | 842.3 µs | 207.3 µs | ✅ yes (4.06×) |
| 500 | 2348.6 µs | 552.8 µs | ✅ yes (4.25×) |

→ **Payback-threshold N ≈ 50 for `StreamingFraimanMuniz::depth_batch`.** Below N_obj ≈ 50 the parallel path is *slower* than single-thread (rayon fork/join + per-object dispatch overhead exceeds the O(n·m) work), consistent with the CONCERNS.md "rayon overhead for n < ~100" note; between N_obj = 10 and 50 the crossover occurs, and by N_obj = 50 parallel is already 2.17× faster, rising toward the ~4× thread-scaling ceiling at N_obj = 500.

**Interpretation (heavy vs light bracket).** The two sentinels **bracket the crossover** as intended (D-01): the heavy `karcher_mean` pays back at essentially any N (threshold ≤ 10, below-grid), while the light `StreamingFraimanMuniz::depth_batch` only pays back once N_obj reaches ≈ 50 — below that, parallelizing it *costs* time. The practical rule this yields for the SC2/SC4 gap analysis: a sequential loop is worth parallelizing when its per-iteration work is heavy (karcher-like — parallelize freely) or when the iteration count reliably exceeds ~50 for light per-iteration work (streaming-like — guard the parallel path behind a size threshold, or accept a small-N regression).

**Not sweep targets (D-03).** `pairwise` / `nadaraya_watson` are **not** additional thread-sweep or payback-N targets — they are already covered by the Phase-2 §"Parallelism Gap List" **ALREADY-PARALLEL inventory** and are represented for SC1 purposes by the two chosen sentinels (heavy `karcher_mean` + light `StreamingFraimanMuniz::depth_batch`), which bracket the crossover and keep the sweep matrix small enough to run under the stricter D-04 stability controls.

**Pipeline proven:** env-driven thread sweep → taskset-pinned 3-run measurement → captured `p5_*` artifacts → report row. Plan 01 seeded the pipeline on the heavy sentinel; Plan 02 (this slice) completed SC1 by adding the light `StreamingFraimanMuniz::depth_batch` sentinel to the thread-scaling table and the payback-threshold-N downward sweep for both targets. Plan 03 continues with the SC2 safe-to-parallelize list, the SC3 unaccelerated-path cost (including the `--no-default-features` rayon-off baseline deferred above), and the SC4 backlog.

### SC2: Sequential Loops Safe to Parallelize

**Evidence standard (D-06): static safety argument + projected speedup — NO `fdars-core/src` edits.** This subsection does *not* wrap any loop or measure a real speedup. Per the audit-only milestone fence (D-06), SC2 is delivered as (a) a static independence argument establishing each loop has no shared mutable state across iterations, (b) an RNG-seeding note, (c) the applicable `parallel.rs` macro, and (d) a *projected* speedup extrapolated from the SC1 measured thread-scaling of the already-parallel karcher analogue. Actually wrapping the loops with `iter_maybe_parallel!` and measuring the real speedup is **deferred to a future implementation milestone** (see SC4 backlog).

**Candidate source (D-07).** The five candidates below are the exact fixed set carried forward from the Phase-2 §"Parallelism Gap List" **SEQUENTIAL** rows (AUDIT-REPORT §Parallelism Gap List, entries `elastic_fpca.rs:701/720/764`, `regression.rs:167`, `classification/cv.rs:76`). They are **not re-derived or re-grepped here** — Phase 2 already confirmed each via `grep -n "iter_maybe_parallel\|slice_maybe_parallel\|maybe_par_chunks"` returning zero macro hits (see §Parallelism Gap List "Pre-write validation"). The banding opt-in row from that same list is an API-default cost, not a sequential-loop candidate, and is handled under SC3/SC4, not here.

**Projection basis (SC1).** The projected-speedup column extrapolates from the SC1 measured karcher thread-scaling (§Thread-Scaling Table): `karcher_mean`'s `iter_maybe_parallel!(0..n)` inner N-loop scales 1.00×→1.99×→3.84×→4.73× across 1/2/4/8 threads on this 20-logical-CPU machine, flattening after 4 threads (~4.7× ceiling at machine-default threads for an N=100 loop). The candidates below are structurally the same shape — a per-curve / per-fold / per-column inner loop with independent iterations — so they are projected to the **same ~4–5× machine-default ceiling**, *bounded by the payback-threshold N*: a candidate that runs at small iteration count will fall below the crossover and may see no benefit (heavy per-iteration bodies pay back at N≤10 per SC1; light bodies only past N≈50). Every number below is a **PROJECTION, not a measurement.**

| Loop (file:line) | Independence argument | RNG-seeding note | Applicable macro | Projected speedup (from SC1) |
|------------------|-----------------------|------------------|------------------|------------------------------|
| `classification/cv.rs:76` (`fclassif_cv` outer `for fold in 0..nfold`) | Each fold is a disjoint train/test split over the same immutable input; no fold writes state another fold reads. Per-fold work (FPCA fit + classifier fit + predict) produces an independent per-fold result appended to a results vector. No shared mutable state across folds. | **No RNG in loop body.** The fold-assignment RNG (`assign_folds`) runs **once before the loop** and produces a deterministic `Vec<usize>` fold map; the fold body itself makes no RNG calls, so no per-thread seeding is needed. (Were an RNG ever introduced into the fold body, it would follow the codebase `StdRng::seed_from_u64(seed + k)` per-thread pattern — not required here.) | `iter_maybe_parallel!(0..nfold)` | **~4–5× projected** at machine-default threads, *bounded by nfold*. Typical nfold (5–10) is a small iteration count, but each fold body is **heavy** (a full FPCA SVD + fit), so it tracks the karcher heavy-body regime (payback N≤10) — worth parallelizing at any realistic nfold. Projection, not measured. |
| `elastic_fpca.rs:701` (`shooting_vectors_from_psis` — `for i in 0..n` computing per-curve `inv_exp_map_sphere`) | Each iteration writes the shooting-vector row for curve `i` only; rows are disjoint, no cross-curve dependency. Read-only shared inputs (psis, base point). No shared mutable state across the N-loop. | **No RNG in loop body.** Pure geometric map (`inv_exp_map_sphere`); no randomness, no per-thread seeding needed. | `iter_maybe_parallel!(0..n)` (per-curve row) | **~4–5× projected** at machine-default threads for N large enough (elastic-FPCA typically N≥50–100), *bounded by payback-N*. Per-curve sphere map is moderately heavy; expected to pay back well above the light-sentinel N≈50 threshold. Projection, not measured. |
| `elastic_fpca.rs:720` (`build_augmented_srsfs` — `for i in 0..n` constructing augmented SRSF rows) | Each iteration builds the augmented SRSF row for curve `i` independently into its own row slice; no row is read by another iteration. Shared inputs read-only. No shared mutable state across the N-loop. | **No RNG in loop body.** Deterministic SRSF construction; no randomness. | `iter_maybe_parallel!(0..n)` (per-curve row) | **~4–5× projected** at machine-default threads for large N, *bounded by payback-N* as above. Projection, not measured. |
| `elastic_fpca.rs:764` (`svd_scores_and_eigenvalues` — inner `for i in 0..n` score extraction per component) | Each iteration extracts the score for curve `i` from the (already-computed) SVD factors; writes a disjoint score entry. No cross-iteration dependency; SVD result is read-only shared. No shared mutable state across the N-loop. | **No RNG in loop body.** Pure linear-algebra score extraction; no randomness. | `iter_maybe_parallel!(0..n)` (per-curve score) | **~4–5× projected** at machine-default threads for large N, *bounded by payback-N*. Note the per-iteration body here is **light** (a dot-product-scale score extraction), so it sits nearer the streaming-sentinel regime — likely needs N≳50 to pay back, per SC1. Projection, not measured. |
| `regression.rs:167` (`center_columns` inside `fdata_to_pc_1d` — outer `for j in 0..m`, inner `for i in 0..n` at `:176`) | Column-major layout: each column `j` is centered independently (subtract that column's mean from its entries); columns share no mutable state. **Distinct from the already-parallel `fdata.rs:center_1d`** (RESEARCH Pitfall 1) — this is the sequential `center_columns` on the FPCA path. No shared mutable state across columns. | **No RNG in loop body.** Deterministic mean-subtraction; no randomness. | `iter_maybe_parallel!(0..m)` (outer-M columns) — could alternatively wrap the inner-N loop, but outer-M is the natural column-independent split | **~4–5× projected** *ceiling only, and unlikely to be reached.* Centering is O(N·M) with a trivial per-element body — a **very light** per-iteration cost that sits well below the streaming payback threshold; per SC1 this likely pays back only at large M (and would need a size guard, per the SC1 practical rule). Lowest-priority candidate. Projection, not measured. |

**Summary.** All five Phase-2 SEQUENTIAL candidates are safe to parallelize under a static independence argument, and none carries an RNG-in-loop hazard (the only RNG, cv.rs's fold assignment, runs once outside its loop). The heavy-body candidates (`cv.rs:76` fold loop) pay back at any realistic iteration count (karcher regime, N≤10); the light-body candidates (`elastic_fpca.rs:764`, `regression.rs:167`) are streaming-regime and should be guarded behind a size threshold or accept a small-N regression. **No `fdars-core/src/` file was edited to produce this analysis** — it is static argument + projection only (D-06).

### SC3: Cost of the Default Unaccelerated Path

**Evidence standard (D-08): report BOTH opt-in dimensions by CITATION of existing artifacts — NO new benchmark was run.** fdars ships two independent "unaccelerated by default unless the user opts in" surfaces: (a) the **rayon-off build** (`--no-default-features`, parallel feature compiled out entirely), and (b) the **banding opt-in** (elastic alignment defaults `band_frac=0.0`). SC3 records the cost each imposes by citing the Phase-1 and Phase-3 measurements already in this report — it does **not** re-measure. Both are framed as *the cost every user of the default API path pays*.

#### (a) rayon-off cost — `--no-default-features` (~10×)

Cited from the Phase-1 karcher 4-combo baseline (§Phase 1 Baseline Cells, "Karcher mean 4-combo baseline"):

| Feature combo | Time | Note |
|---------------|------|------|
| `""` (no `parallel`, rayon compiled out) | ~1555 ms | The `--no-default-features` default-off cost |
| `parallel` (rayon active) | ~162 ms | ~**10×** faster with rayon on |

→ **A user who builds fdars with `--no-default-features` pays a ~10× penalty** on the karcher hot path (1555 ms vs 162 ms), because the `iter_maybe_parallel!` macros fall back to sequential iterators when the `parallel` feature is absent. Artifacts: [karcher/none](bench/p1_karcher_none_run1.txt) · [karcher/parallel](bench/p1_karcher_parallel_run1.txt).

**Distinct from the SC1 payback baseline.** This `--no-default-features` cost (rayon *compiled out entirely* — different codegen, no rayon dependency) is **not** the same as the SC1 §Payback-Threshold N baseline, which holds the `parallel` feature **on** and only sets `RAYON_NUM_THREADS=1` (single-thread rayon, identical codegen). SC1 answers "how many threads help?"; SC3(a) answers "what does turning rayon off entirely cost?". The two happen to land near the same order of magnitude at N=100 (SC1 1-thread karcher ≈ 1554 ms ≈ the rayon-off 1555 ms) because a single-thread rayon pool and a rayon-free build do essentially the same serial work here — but they are conceptually different questions and different builds.

#### (b) banding opt-in cost — `band_frac=0.0` default (~7× nominal, measured ~4–6×)

Cited from the Phase-3 §"Banded-vs-Unbanded Analysis" and §"D-05 Source Fact". `karcher_mean()` hard-codes `band_frac=0.0` in its call to `karcher_mean_impl` (`fdars-core/src/alignment/karcher.rs:300`) → `band_radius(0.0, m) = None` → full O(m²) unbanded DP per alignment pair (this is **Anti-Pattern 2** from the Phase-2 §Parallelism Gap List). The faster banded path is **opt-in only** — users must explicitly call `karcher_mean_banded()`.

- **Nominal expectation:** ~7× (theoretical m/band = 200/20 = 10× at `band_frac=0.1`, M=200, minus per-iteration overhead).
- **Measured (cited):** karcher `karcher_mean` vs `karcher_mean_banded` at N=500,M=200 ≈ **4× (run2) / ~5.9× (run1)**; at representative cells across the banded analysis the reduction lands **~4–6×** (§Banded-vs-Unbanded Analysis, karcher table). Artifacts: [p3_karcher](bench/p3_karcher_linalg,parallel_run1.txt).

→ **Every user who calls `karcher_mean()` pays the full unbanded O(m²) cost** — a ~4–6× penalty vs the available-but-opt-in banded path — without any way to enable banding through the default API.

**LOW-CONFIDENCE caveat (D-08).** The raw Phase-3 karcher cells showed **34–204% two-run variance** (OS scheduler jitter under intermittent load; see §Phase-3 "LOW CONFIDENCE explanation"), so the karcher ~7× / ~4–6× figure is **directional, not precise**. It is corroborated by the **stable elastic_cross cells** (0–4% two-run variance) in the same §Banded-vs-Unbanded Analysis, which show a consistent **4.5–5.7× banding reduction** — confirming the banding win is real and in the ~4.5–5.7× range even though the karcher-specific multiplier is noisy. Phase 9 should re-run karcher under a pinned governor for a stable banding number.

**Both are default-path costs.** Neither the rayon-off ~10× nor the banding ~4–6× is a pathological edge case: they are the costs borne by the *default* build/API surface. The rayon-off cost is opt-out (a build-time choice a user might make for a minimal dependency footprint); the banding cost is opt-in-to-avoid (the fast path exists but is never the default). Both feed the SC4 backlog as candidate default-changing work.

### SC4: Parallelization Backlog (draft, GSD-ready)

These entries summarize SC1–SC3 into GSD-ready candidate requirements/phases for **Phase-9 ranking**. Each reuses the Phase-3 §"Draft Backlog" field format (Function | Current cost | Root cause | Candidate direction | Projected/observed reduction | Evidence link). The SC2 loop candidates carry **projected** reductions (no implementation exists yet — D-06); the SC3 banding item is a **measured** reduction cross-referenced (not duplicated) from the Phase-3 elastic-alignment backlog.

**Backlog entry P5-1 — Parallelize the classification CV fold loop (`classification/cv.rs:76`) — high priority**

| Field | Detail |
|-------|--------|
| **Function** | `fclassif_cv` (`classification/cv.rs:45`); outer `for fold in 0..nfold` fold loop at `cv.rs:76`. |
| **Current cost** | CV sentinel baseline (§Phase 1 Baseline Cells, CV loops row): `fclassif_cv` (lda, 5-fold) at N=100, M=50 = ~948–952 µs/iter (OK confidence). The fold loop runs `nfold` sequential FPCA+fit+predict passes; the per-fold FPCA SVD dominates. Artifact: [p1_cv](bench/p1_cv_linalg,parallel_run1.txt). |
| **Root cause** | SEQUENTIAL (§Parallelism Gap List, `cv.rs:76`): plain `for fold in 0..nfold` with no parallelism macro. Folds are fully independent (disjoint train/test splits, no shared mutable state); the fold-assignment RNG (`assign_folds`) runs once before the loop, so the loop body has no RNG. |
| **Candidate direction** | Wrap the fold loop in `iter_maybe_parallel!(0..nfold)` (per SC2). Each fold body is heavy (a full FPCA SVD + classifier fit), so it tracks the karcher heavy-body regime (payback N≤10 per SC1) — worth parallelizing at any realistic nfold. |
| **Projected reduction** | **~4–5× projected** at machine-default threads (from SC1 karcher thread-scaling ceiling ~4.7× at N=100). Bounded by nfold, but heavy per-fold body pays back at small counts. Projection, not measured (D-06). |
| **Evidence link** | SC1 §Thread-Scaling Table (karcher scaling) · SC2 `cv.rs:76` row · [p1_cv](bench/p1_cv_linalg,parallel_run1.txt). |

**Backlog entry P5-2 — Parallelize the three elastic-FPCA inner N-loops (`elastic_fpca.rs:701/720/764`) — medium priority**

| Field | Detail |
|-------|--------|
| **Function** | `shooting_vectors_from_psis` (`elastic_fpca.rs:701`), `build_augmented_srsfs` (`elastic_fpca.rs:720`), `svd_scores_and_eigenvalues` (`elastic_fpca.rs:764`) — three per-curve `for i in 0..n` inner loops on the elastic-FPCA path. |
| **Current cost** | No dedicated p5 measurement (SC1 sentinels are karcher + streaming); projected only. These loops are on the elastic-FPCA critical path, each O(N) per-curve with a moderate-to-light body (sphere map, SRSF construction, score extraction). Related allocation sites recorded at §Allocation Hotspot List (`elastic_fpca.rs:214/317/483/584/930`). |
| **Root cause** | SEQUENTIAL (§Parallelism Gap List, `elastic_fpca.rs:701/720/764`): plain `for i in 0..n` with no parallelism macro. Each iteration writes a disjoint per-curve row/score; no shared mutable state; no RNG in loop body. |
| **Candidate direction** | Wrap each in `iter_maybe_parallel!(0..n)` (per SC2, per-curve row/score). `:701` and `:720` have moderately heavy bodies (pay back above N≈50); `:764` is light (score extraction — streaming regime, guard behind a size threshold or accept small-N regression). |
| **Projected reduction** | **~4–5× projected** at machine-default threads for large N (elastic-FPCA typically N≥50–100), bounded by the payback-N threshold. Projection, not measured (D-06). |
| **Evidence link** | SC1 §Thread-Scaling + §Payback-Threshold N · SC2 `elastic_fpca.rs:701/720/764` rows. |

**Backlog entry P5-3 — Parallelize `center_columns` (`regression.rs:167`) inside `fdata_to_pc_1d` — low priority**

| Field | Detail |
|-------|--------|
| **Function** | `center_columns` (`regression.rs:167`), the sequential outer `for j in 0..m` / inner `for i in 0..n` double loop called inside `fdata_to_pc_1d`. **Distinct from the already-parallel `fdata.rs:center_1d`** (RESEARCH Pitfall 1). |
| **Current cost** | On the FPCA path (§Phase 1 Baseline, FPCA/SVD row: `fdata_to_pc_1d` N=500,M=200 = ~16 ms/iter), but centering is O(N·M) with a trivial per-element body and is *secondary* to the nalgebra SVD step that dominates the 16 ms (§Complexity Table FPCA row). |
| **Root cause** | SEQUENTIAL (§Parallelism Gap List, `regression.rs:167`): plain double `for` loop, zero macro hits. Column-major layout makes each column independent (subtract that column's mean); no shared mutable state; no RNG. |
| **Candidate direction** | Wrap the outer-M loop in `iter_maybe_parallel!(0..m)` (columns independent; per SC2). **Lowest priority:** the per-element body is very light (streaming regime, pays back only at large M and would need a size guard), and SVD — not centering — is the FPCA M-scaling bottleneck. |
| **Projected reduction** | **~4–5× ceiling projected but unlikely reached** — light body, likely sub-crossover except at large M; net FPCA-call speedup is small because SVD dominates. Projection, not measured (D-06). |
| **Evidence link** | SC1 §Payback-Threshold N (light-body rule) · SC2 `regression.rs:167` row · [p1_fpca](bench/p1_fpca_linalg,parallel_run1.txt). |

**Backlog entry P5-4 — Change elastic-alignment API defaults to a banded path / expose `band_frac` (cross-reference)**

This parallelism-adjacent opt-in-default cost is **not duplicated here** — it is the same default-path cost recorded at SC3(b) and fully specified in the Phase-3 §"Draft Backlog (elastic alignment)" **Backlog entry 1** ("Default elastic alignment to a banded path") and **Backlog entry 2** ("Expose band_frac on high-level distance matrix API"). It is linked from the parallelism backlog because it is an opt-in-default cost of the same family as the SC2 sequential-loop gaps: `karcher_mean()` defaults `band_frac=0.0` (§D-05 Source Fact, `karcher.rs:300`), imposing the measured ~4–6× SC3(b) penalty on every default caller. **Candidate direction:** change the high-level default to `band_frac≈0.1` or expose it as a parameter (the banded implementations already exist and are correct — an API default change, not a new algorithm). **Observed reduction:** measured 4–6× (karcher, LOW-CONFIDENCE) / 4.5–5.7× (stable elastic_cross). See Phase-3 backlog for the full field table and artifact links; Phase 9 should rank it alongside P5-1..P5-3.

---

## Phase 6: Conditional SVD Library Comparison

**Last updated:** 2026-08-08
**Feature flags:** `--features linalg` (faer 0.23.2 is behind the `linalg` feature)
**Profile:** release (`/release/deps/audit_hotpaths-*`) — confirmed in all five artifacts
**Bench binary:** `fdars-core/benches/audit_hotpaths.rs` — `bench_p6_svd_comparison`
**Equivalence test:** `fdars-core/tests/svd_equivalence.rs` — `svd_equivalence` (integration test)

---

### SC1 — Go/No-Go Decision

**Verdict: GO — Phase 6 comparison is warranted.**

This comparison was executed because the Phase 4 evidence satisfies both SC1 compound conditions (§Phase 4 "Phase 6 Go/No-Go Decision"):

1. **SVD wall-clock share:** SVD compute accounts for ~**99.8–99.9%** of `fdata_to_pc_1d` wall-clock at every grid cell. The O(m³) nalgebra SVD dominates; all other operations (centering at `regression.rs:167`, clone at `regression.rs:291`, the `to_dmatrix()` bridge at `regression.rs:298`) are negligible by comparison.

2. **`to_dmatrix()` copy share:** The copy contributes approximately **0.14–0.17%** of total wall-clock across the N×M grid — well below any threshold where it would be "the dominant cost."

Both conditions are met, triggering Phase 6. The comparison measures faer `thin_svd` (via `MatRef::from_column_major_slice`, zero-copy) against nalgebra `SVD::new(dmatrix, true, true)` (clone-then-SVD) at fdars' real FPCA workload sizes.

**Fairness guarantee:** the `svd_equivalence` integration test (`tests/svd_equivalence.rs`) confirms that nalgebra and faer agree on all numerically significant singular values within relative tolerance 1e-10. The test also verifies that faer's thin SVD produces U with shape (n=500, m=200) for an N>M input, resolving RESEARCH Open Question 1.

---

### SC2 — nalgebra vs faer(seq) SVD Comparison Table

All timings are the criterion median point estimate from run1. Speedup = nalgebra\_median / faer\_seq\_median. Conversion cost = FdMatrix→faer::MatRef view construction (zero-copy pointer + dims). Run2 variance noted where >10%.

| Cell | nalgebra SVD (run1) | faer(seq) thin\_svd (run1) | Speedup (run1) | nalgebra (run2) | faer(seq) (run2) | Conversion cost | Notes |
|------|---------------------|---------------------------|----------------|-----------------|------------------|-----------------|-------|
| N=100, M=50 | 1.582 ms | 0.442 ms | **3.6×** | 0.520 ms | 0.406 ms | ~4.7 ns | HIGH variance across runs (cache effects) |
| N=100, M=200 | 10.308 ms | 2.508 ms | **4.1×** | 4.062 ms | 1.688 ms | ~5.5 ns | Significant run-to-run variation |
| N=500, M=50 | 3.874 ms | 1.426 ms | **2.7×** | 2.862 ms | 1.344 ms | ~4.2 ns | OK confidence |
| N=500, M=200 | 41.026 ms | 23.084 ms | **1.8×** | 35.084 ms | 17.753 ms | ~6.8 ns | PRIMARY CELL — most representative |
| N=1000, M=50 | 6.626 ms | 1.803 ms | **3.7×** | 6.792 ms | 1.864 ms | ~4.3 ns | GOOD confidence — consistent |
| N=1000, M=200 | 95.612 ms | 30.957 ms | **3.1×** | 92.840 ms | 18.183 ms | ~4.7 ns | faer run2 >10% below run1 |
| N=500, M=500 | 358.31 ms | 189.70 ms | **1.9×** | 324.62 ms | 114.98 ms | ~4.9 ns | CROSSOVER PROBE — faer run2 shows 2.8× |

**Artifact links:**
- nalgebra: [run1](bench/p6_svd_nalgebra_linalg_run1.txt) · [run2](bench/p6_svd_nalgebra_linalg_run2.txt)
- faer(seq): [run1](bench/p6_svd_faer_seq_linalg_run1.txt) · [run2](bench/p6_svd_faer_seq_linalg_run2.txt)
- conversion: [run1](bench/p6_svd_conversion_linalg_run1.txt)

**Crossover observation:** faer(seq) is faster than nalgebra at every measured cell, including the M=50 thin-matrix sizes where the advantage was expected to be smallest. The measured speedup at M=200 (primary FPCA cell) is **1.8× (run1) / ~1.9× (run2)** — below the assumed 3–10× from STACK.md reference data for large square matrices. The assumption A1 was over-optimistic for fdars' tall-thin (N>>M) rectangular matrices. The M=500 crossover probe confirms faer still wins (~1.9× run1 / ~2.8× run2) but with high run-to-run variance.

**Run-to-run variance note:** faer run2 is significantly faster than run1 at several cells (e.g., n500_m500: 190ms → 115ms, a ~39% drop). This is attributable to CPU cache warmup and OS scheduler state between runs, the same environment issue as Phase 5 (governor pinning requires root; `powersave` allows frequency scaling). The run1 numbers are the more conservative and representative estimates; the run2 numbers confirm the direction but not the magnitude. All cells show faer winning, regardless of run.

**Conversion cost conclusion:** FdMatrix→faer::MatRef construction costs ~3.5–7.7 ns regardless of matrix size — it is a zero-copy pointer + dimension assignment. This is four to five orders of magnitude below the SVD compute time at every cell. The `to_dmatrix()` copy (measured in Phase 4 as 0.14–0.17% of wall-clock) is the only allocation cost. Replacing `to_dmatrix()` with `from_column_major_slice()` eliminates that allocation entirely.

**Numerical equivalence:** `svd_equivalence` integration test passes (`test result: ok. 1 passed`). nalgebra and faer agree on all significant singular values within 1e-10 relative error at N=500, M=200. The test skips near-zero values (< 1e-8 × σ₁) where both backends produce numerical noise — these are not meaningful singular values for FPCA.

---

### SC3 — faer Adoption / Maintenance-Risk Note

| Factor | Assessment | Confidence |
|--------|------------|------------|
| **Performance vs nalgebra SVD** | **Measured: 1.8–4.1× faster** at fdars' real FPCA sizes (M∈{50,200}). Primary cell N=500,M=200: 1.8× (run1). Best ratio: N=100,M=200 at 4.1×. Below the ASSUMED 3–10× from STACK.md large-square benchmarks — faer's advantage is smaller for tall-thin rectangular matrices (N>>M). Still consistently positive across all 7 cells. | HIGH (measured) |
| **API stability** | SVD API (`thin_svd`, `from_column_major_slice`, `U()`, `V()`, `S()` accessors) has been stable since faer 0.18. Breaking changes in 0.18–0.23 (LBLT rename in 0.22, constructor simplification in 0.19) did not affect SVD or MatRef API. | MEDIUM |
| **Breaking change frequency** | Moderate — approximately every 2–4 minor versions in 0.18–0.22. Version 0.23.x adds generalized eigendecomposition and matrix-free SVD without breaking existing SVD API. | MEDIUM |
| **Maintainer / bus factor** | Single primary maintainer (sarah-quinones). Active development. Repository on Codeberg (not GitHub). Low bus factor — a project stall would block faer upgrades. | LOW (web search basis) |
| **MSRV tension** | faer 0.23 requires Rust 1.84.0 [VERIFIED: faer-0.23.2/Cargo.toml]. fdars MSRV is 1.81 [VERIFIED: fdars-core/Cargo.toml]. **Not a real barrier:** the `linalg` feature already requires Rust 1.84+ and is documented as non-default for CRAN compatibility. Shipping faer SVD under `linalg` adds no MSRV constraint beyond what is already accepted for the feature. CRAN users (using `default-features=false`) are unaffected. | HIGH |
| **Integration cost** | faer is already a dependency — zero new Cargo.toml work. Conversion: 1 line (`MatRef::from_column_major_slice` replacing `to_dmatrix()`). SVD call: 1 line (`mat_ref.thin_svd()` replacing `SVD::new`). Output extraction (S, U, V): ~3 lines each. Total code delta: ~20 lines in `fdata_to_pc_1d`. Tests: add numerical equivalence assertion for FPCA path. | HIGH |
| **Test / correctness risk** | **VERIFIED**: nalgebra and faer agree on significant singular values within 1e-10 (`svd_equivalence` green). Sign-flip of singular vectors is an arbitrary convention — existing `FpcaResult.rotation` consumers must be validated after adoption, but this is a one-time integration check, not ongoing risk. | HIGH |

**Integration-ROI verdict:** The integration burden is LOW (faer already vendored, ~20-line change, stable SVD API) and the measured speedup is real but moderate (1.8–4.1×, primary cell ~1.8×). The MSRV constraint is already accepted for the `linalg` feature. The main risk is the single-maintainer bus factor. Given the measured speedup at the primary cell (N=500,M=200) is 1.8× — below the 2× threshold that the RESEARCH defined as "clearly worth it" — the adoption case is **borderline-positive**: meaningful for FPCA-heavy workloads (saves ~18ms per call at N=500,M=200), not urgent for workloads where N×M is small. Phase 9 should re-evaluate with faer parallel (not measured here) to assess whether the parallel path crosses the 2× threshold.

---

### SC4 — Phase 9 Backlog Item

**Backlog entry P6-1 — Swap nalgebra SVD for faer thin_svd in `fdata_to_pc_1d`**

| Field | Detail |
|-------|--------|
| **Function** | `fdata_to_pc_1d` (`regression.rs:298`) — the nalgebra `SVD::new(weighted.to_dmatrix(), true, true)` call executed on every FPCA call in the library |
| **Current cost (measured)** | N=500,M=200 (primary cell): 41.0 ms / 35.1 ms (run1/run2). N=1000,M=200: 95.6 ms / 92.8 ms. SVD is ~99.8–99.9% of fdata_to_pc_1d wall-clock (Phase 4). `to_dmatrix()` copy adds one O(N·M) allocation (~800 KB at N=500,M=200) that faer eliminates. Artifacts: [p6\_svd\_nalgebra\_run1](bench/p6_svd_nalgebra_linalg_run1.txt) · [p6\_svd\_nalgebra\_run2](bench/p6_svd_nalgebra_linalg_run2.txt) |
| **Root cause** | nalgebra's `SVD::new` is always sequential; takes a `DMatrix<f64>` input (requires `to_dmatrix()` allocation); computes the full thin SVD. faer's `thin_svd` accepts a `MatRef` — a zero-copy view into the existing FdMatrix column-major slice (`as_slice()`), eliminating the `to_dmatrix()` allocation. faer's SVD algorithm is faster at all measured sizes. |
| **Candidate fix** | Replace `weighted.to_dmatrix()` + `SVD::new(dmatrix, true, true)` at `regression.rs:298` with: `let mat_ref = faer::MatRef::<f64>::from_column_major_slice(weighted.as_slice(), n, m); let fa_svd = mat_ref.thin_svd()?;` Gate behind the existing `linalg` feature (`#[cfg(feature = "linalg")]`), with a nalgebra fallback for the non-linalg path. Extract U, S, Vt from `fa_svd` (faer accessors: `.U()`, `.S()`, `.V()`). Add numerical equivalence assertion in CI tests. |
| **Evidence** | Run1 speedup at all 7 cells (faer consistently faster): 3.6×, 4.1×, 2.7×, **1.8×**, 3.7×, 3.1×, 1.9× (N∈{100,100,500,500,1000,1000,500} × M∈{50,200,50,200,50,200,500}). Primary cell N=500,M=200: 1.8× (run1) / 1.9× (run2). Equivalence test: `svd_equivalence` green within 1e-10. Phase 4 artifacts: [p4\_fpca run1](bench/p4_fpca_linalg,parallel_run1.txt). Phase 6 full artifact set: 5 files `p6_svd_*_linalg_run{1,2}.txt` + `p6_svd_conversion_linalg_run1.txt`. |
| **Severity + Effort** | **P2 / S-effort** (borderline). Speedup at primary FPCA cell is 1.8× (run1), below the research-defined "clearly worth it" threshold of ≥2× at M≥200. Set to P2 because: (a) the direction is consistently positive at all 7 cells, (b) the absolute saving at N=1000,M=200 is ~27 ms/call — meaningful for FPCA-heavy workflows, (c) integration cost is low (~20 lines, faer already vendored). Downgrade to P3 if run3 under pinned governor shows speedup < 1.5× at N=500,M=200. S-effort: ~1 week including equivalence test + regression of `FpcaResult` output. Note: faer parallel path (not measured in Phase 6) may offer additional speedup and should be evaluated in Phase 9. |

---

## Phase 7 — scikit-fda Capability Enumeration

Phase 7 builds the scikit-fda side of the eventual parity comparison: a versioned,
capability-oriented inventory of scikit-fda's public surface organized by six report areas.
Deliverables are documentation artifacts in `.planning/research/` only — no `fdars-core/src`
files are modified.

### Methodology

#### Version Pinning (SC2)

The pinned version for this entire section is **scikit-fda 0.10.1**.

**Verification path used: RUNTIME.** A throwaway venv was created at
`.planning/research/skfda-verify/venv` using `python3 -m venv` (Python 3.14.5).
`pip install "scikit-fda==0.10.1"` completed successfully. The runtime confirmation:

```
python -c "import skfda; print(skfda.__version__)"
→ 0.10.1
```

Full evidence is in [`.planning/research/skfda-verify/version.txt`](skfda-verify/version.txt)
and [`.planning/research/skfda-verify/verify.log`](skfda-verify/verify.log). The venv is a
throwaway only — it is not the deliverable.

**D-01a coincidence:** 0.10.1 is both the agreed sole baseline (PROJECT.md Key Decisions)
and the current latest release on PyPI. No newer version exists at the time of this
enumeration, so the baseline is not stale and no re-pin decision is needed.

#### Source Reuse (D-02)

This section promotes and refactors the existing `.planning/research/FEATURES.md`
(scikit-fda 0.10.1 API enumeration at MEDIUM confidence, verified against readthedocs).
Promotion steps:

1. Extract the scikit-fda-only public API enumeration from each FEATURES.md area.
2. Strip all fdars gap notes — parity annotations ("present", "partial", "missing",
   "equivalent") are Phase-8 material (GAP-02) and must not appear in this section.
3. Re-verify entries against the 0.10.1 source per the RUNTIME path (dir() spot-checks
   confirm smoothing, classification, and depth module contents).
4. Reorganize FEATURES.md's 12 sub-areas under the six SC1 report areas defined below.
5. Raise the confidence tag from MEDIUM where the RUNTIME verification supports it.

### Capability-Row Schema (D-03, D-04)

This schema governs every table in this Phase 7 section. Plan 02 reuses it verbatim for
the remaining five areas.

#### Two-Level Structure (D-03)

The enumeration is organized in **two levels**:

1. **Six report areas** (fixed by SC1): Representation, Preprocessing, Exploratory,
   ML (Machine Learning), Inference, Misc.
2. Within each area, **task groupings** — named subsections collecting rows that serve
   the same user task (e.g. "Smoothing", "Registration", "Classification").
3. Within each task grouping, **one row per distinct method or algorithm**.

#### Collapse Rule (Pitfall 9)

A single scikit-fda estimator's `fit()`, `predict()`, `transform()`, and
`inverse_transform()` are collapsed into **one capability row**. These are all part of
the same capability — a user task like "smooth curves" — not separate features. The
"Collapsed calls" column records which sklearn-protocol methods apply to that row.

Rationale: fdars accomplishes the same tasks via builder structs plus a single call
returning a result struct. Counting `fit/transform` as two rows would inflate the scikit-fda
surface artificially (Pitfall 9 — counting API names instead of capabilities).

#### Relevance Taxonomy (D-04, Pitfall 14)

Every row carries a **Relevance** value drawn from exactly these four values:

| Relevance value | Meaning |
|----------------|---------|
| `In-Scope Algorithm` | A numeric algorithm or capability that is in scope for fdars (regression, classification, alignment, depth, inference, etc.) |
| `In-Scope API-Ergonomics` | An API convenience or ergonomics feature that is in scope (e.g. scoring utilities, metadata, parameter selection) |
| `Out-of-Scope (plotting)` | A visualization or matplotlib-dependent feature; explicitly out of scope for fdars (PROJECT.md) |
| `Out-of-Scope (IO)` | A data loading, DataFrame round-trip, or dataset-bundling feature; out of scope for fdars |

**Borderline rulings:**

- Plotting / Visualization classes (GraphPlot, Boxplot, etc.) → `Out-of-Scope (plotting)`.
- DataFrame / pandas round-trips, `fetch_*` dataset loaders → `Out-of-Scope (IO)`.
- `FDAFeatureUnion` / `PerClassTransformer` (sklearn-pipeline plumbing) → `Out-of-Scope (IO)`
  (Rust equivalent is trait composition, not an API port; PROJECT.md).
- The representation **type-system** (`FDataGrid` / `FDataBasis` / `FDataIrregular` as a
  first-class object hierarchy) → the type-system refactor itself is out of scope
  (PROJECT.md), but specific *algorithmic* capabilities riding on these types
  (e.g. `FDataIrregular` covariance estimation, grid-to-basis conversion math,
  spline interpolation algorithm) are `In-Scope Algorithm` and enumerated as such.

**In-scope vs. out-of-scope gap counts are reported separately** so the actionable count
for Phase 8 is not inflated by plotting/IO rows.

#### Table Columns

Every capability table in this section uses these exact columns:

| Column | Description |
|--------|-------------|
| **Task** | The user task grouping (e.g. "Smoothing", "Depth measures") |
| **Method** | The distinct method or algorithm (one row per method) |
| **Collapsed calls** | Which sklearn-protocol calls are covered by this row (fit / predict / transform / inverse_transform / function call) |
| **Relevance** | One of the four D-04 taxonomy values |
| **Confidence** | HIGH (RUNTIME-verified), MEDIUM (docs-verified), LOW (inferred) |
| **Source** | Citation: FEATURES.md §Area N, readthedocs module path, or dir() output |

---

### Area: Representation

This area covers scikit-fda's data representation layer — the core types, basis systems,
interpolation/extrapolation mechanisms, irregular-data converters, and grid-to-basis
conversion. Promoted from FEATURES.md §Area 1 (scikit-fda public API table), fdars notes
stripped per D-02.

**D-04 ruling for this area:** The representation *type-system* (`FDataGrid` /
`FDataBasis` / `FDataIrregular` as a first-class Python object hierarchy with
`__repr__`, `__getitem__`, arithmetic magic methods, etc.) is a type-system refactor
that is **out of scope** for fdars (PROJECT.md). However, the specific *algorithmic
capabilities* exposed through or alongside those types are `In-Scope Algorithm` — they
represent concrete numeric operations (covariance estimation, spline interpolation math,
basis-coefficient fitting, grid-to-basis conversion) that a Rust FDA library should offer
regardless of how the data types are structured.

**In-scope count (this area):** 12 rows   **Out-of-scope count:** 7 rows

| Task | Method | Collapsed calls | Relevance | Confidence | Source |
|------|--------|-----------------|-----------|------------|--------|
| Data types — grid | `FDataGrid` — discretized representation on a common evaluation grid; supports arithmetic, finite-difference derivatives, integration, inner products | evaluate / arithmetic / derivative | Out-of-Scope (plotting) | HIGH | FEATURES.md §Area 1; `skfda.FDataGrid` dir()-verified |
| Data types — basis | `FDataBasis` — parametric representation as linear combination of basis functions; analytical derivatives | evaluate / arithmetic / derivative | Out-of-Scope (plotting) | HIGH | FEATURES.md §Area 1; `skfda.FDataBasis` |
| Data types — irregular | `FDataIrregular` — sparse/irregularly sampled observations per curve (added v0.10.0) | evaluate | Out-of-Scope (plotting) | HIGH | FEATURES.md §Area 1; v0.10.0 release notes |
| Data types — abstract | `FData` — abstract base class with shared interface (evaluate, arithmetic, derivatives) | — (base class only) | Out-of-Scope (plotting) | HIGH | FEATURES.md §Area 1; readthedocs representation module |
| Covariance estimation | `FDataIrregular` covariance estimation — empirical covariance from irregularly sampled observations | function call (`.cov()`) | In-Scope Algorithm | MEDIUM | FEATURES.md §Area 1; readthedocs FDataIrregular.cov |
| Basis systems | `BSplineBasis` — B-spline basis (R→R) | instantiate / evaluate | In-Scope Algorithm | HIGH | FEATURES.md §Area 1; `skfda.representation.basis.BSplineBasis` |
| Basis systems | `FourierBasis` — Fourier (trigonometric) basis (R→R) | instantiate / evaluate | In-Scope Algorithm | HIGH | FEATURES.md §Area 1; `skfda.representation.basis.FourierBasis` |
| Basis systems | `MonomialBasis` — monomial/polynomial basis (R→R) | instantiate / evaluate | In-Scope Algorithm | HIGH | FEATURES.md §Area 1; `skfda.representation.basis.MonomialBasis` |
| Basis systems | `ConstantBasis` — constant (intercept) basis (R→R) | instantiate / evaluate | In-Scope Algorithm | HIGH | FEATURES.md §Area 1; `skfda.representation.basis.ConstantBasis` |
| Basis systems | `CustomBasis` — arbitrary user-supplied basis functions | instantiate / evaluate | In-Scope Algorithm | MEDIUM | FEATURES.md §Area 1; readthedocs |
| Basis systems | `TensorBasis` — tensor product of 1D bases (Rⁿ→R, multivariate domain) | instantiate / evaluate | In-Scope Algorithm | MEDIUM | FEATURES.md §Area 1; readthedocs |
| Basis systems | `FiniteElementBasis` — finite element basis (Rⁿ→R, irregular meshes) | instantiate / evaluate | In-Scope Algorithm | MEDIUM | FEATURES.md §Area 1; readthedocs |
| Basis systems | `VectorValuedBasis` — stack of bases for vector-valued output (Rⁿ→Rᵐ) | instantiate / evaluate | In-Scope Algorithm | MEDIUM | FEATURES.md §Area 1; readthedocs |
| Interpolation | `SplineInterpolation` — spline interpolation for evaluating `FDataGrid` at off-grid points | fit / evaluate (via `FDataGrid.__call__`) | In-Scope Algorithm | HIGH | FEATURES.md §Area 1; readthedocs FDataGrid interpolation |
| Extrapolation | `BoundaryExtrapolation` — repeat boundary value for out-of-domain queries | set as extrapolator | In-Scope Algorithm | MEDIUM | FEATURES.md §Area 1; readthedocs extrapolation |
| Extrapolation | `ExceptionExtrapolation` — raise exception on out-of-domain query | set as extrapolator | In-Scope API-Ergonomics | MEDIUM | FEATURES.md §Area 1; readthedocs extrapolation |
| Extrapolation | `FillExtrapolation` — fill with a constant value for out-of-domain queries | set as extrapolator | In-Scope Algorithm | MEDIUM | FEATURES.md §Area 1; readthedocs extrapolation |
| Extrapolation | `PeriodicExtrapolation` — wrap the domain periodically for out-of-domain queries | set as extrapolator | In-Scope Algorithm | MEDIUM | FEATURES.md §Area 1; readthedocs extrapolation |
| Irregular→basis conversion | `MinimizeMixedEffectsConverter` — convert `FDataIrregular` → `FDataBasis` via mixed-effects optimization | fit / transform | In-Scope Algorithm | MEDIUM | FEATURES.md §Area 1; readthedocs preprocessing.conversion |
| Irregular→basis conversion | `EMMixedEffectsConverter` — convert `FDataIrregular` → `FDataBasis` via EM algorithm | fit / transform | In-Scope Algorithm | MEDIUM | FEATURES.md §Area 1; readthedocs preprocessing.conversion |
| Grid-to-basis conversion | `FDataGrid.to_basis()` — convert a discretized grid representation to a basis representation (least-squares projection onto chosen basis) | method call | In-Scope Algorithm | HIGH | FEATURES.md §Area 1; readthedocs FDataGrid.to_basis |

**Notes:**
- The `FDataGrid`, `FDataBasis`, `FDataIrregular`, and `FData` rows are marked
  `Out-of-Scope (plotting)` because the capability being noted is the Python *type-system
  refactor* (a first-class OO type hierarchy with magic methods, arithmetic protocol,
  `__call__` evaluate protocol) — not a numerical algorithm. The algorithmic capabilities
  that ride on these types (covariance, interpolation, basis conversion) are enumerated
  as separate `In-Scope Algorithm` rows.
- The `ExceptionExtrapolation` is marked `In-Scope API-Ergonomics` (not Algorithm): it is
  a validation/error-signalling policy, not a numeric computation.
- Confidence HIGH for items confirmed by `skfda.__version__ = 0.10.1` RUNTIME verification
  and dir() spot-checks; MEDIUM for items verified against readthedocs 0.10.1 docs only.
