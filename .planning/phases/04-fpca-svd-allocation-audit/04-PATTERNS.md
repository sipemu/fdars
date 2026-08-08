# Phase 4: FPCA/SVD & Allocation Audit — Pattern Map

**Mapped:** 2026-08-08
**Files analyzed:** 5 (2 code modifications + 1 manifest edit + 1 report append + 1 coverage stub)
**Analogs found:** 5 / 5

This is an **audit phase**: no `fdars-core/src` algorithm changes. All new code is measurement scaffolding (criterion bench extension, dhat integration test harness) plus report/backlog markdown appended to existing files. Every pattern needed already exists in the Phase 1/3 bench harness and in AUDIT-REPORT.md.

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `fdars-core/benches/audit_hotpaths.rs` (MODIFY — extend) | benchmark harness | batch/measurement | itself: `bench_fpca_sentinel` (lines 82–96), `bench_p3_karcher` (lines 278–345) | exact |
| `fdars-core/tests/alloc_audit_fpca.rs` (NEW) | test harness (dhat integration test) | request-response | `fdars-core/benches/audit_hotpaths.rs` generate_curves + bench cell discipline; dhat wiring from STACK.md | role-match |
| `fdars-core/Cargo.toml` (MODIFY — dev-dep + feature) | config/manifest | build-registration | itself: `[dev-dependencies]` at line 49, `[features]` (implicit default feature) | exact |
| `.planning/research/AUDIT-REPORT.md` (MODIFY — append) | report section | transform/accumulate | itself: `## Phase 3` section shape; `## Phase 2` backlog fields | exact |
| `.planning/phases/04-fpca-svd-allocation-audit/04-COVERAGE.md` (NEW) | coverage stub | — | `.planning/phases/03-elastic-alignment-hot-path/03-COVERAGE.md` | exact |

---

## Pattern Assignments

### `fdars-core/benches/audit_hotpaths.rs` — `bench_p4_fpca` and `bench_p4_elastic_fpca` (MODIFY)

**Analog:** the same file's existing sentinel functions and Phase-3 grid sweeps.

**Imports pattern** (lines 20–32) — extend the existing use block to add `elastic_fpca` module imports:
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use fdars_core::alignment::{
    elastic_cross_distance_matrix, elastic_cross_distance_matrix_banded,
    elastic_self_distance_matrix, elastic_self_distance_matrix_banded,
    karcher_mean, karcher_mean_banded,
};
use fdars_core::classification::fclassif_cv;
use fdars_core::depth::fraiman_muniz_1d;
use fdars_core::matrix::FdMatrix;
use fdars_core::regression::fdata_to_pc_1d;
// NEW for Phase 4:
use fdars_core::elastic_fpca::{joint_fpca, vert_fpca};
use std::f64::consts::PI;
```

**Seeded N×M generator — reuse verbatim** (lines 38–52):
```rust
fn generate_curves(n: usize, m: usize) -> (FdMatrix, Vec<f64>) {
    let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
    let mut data = vec![0.0; n * m];
    for i in 0..n {
        let phase = 0.2 * ((i as f64 * 3.7 + 0.5).sin());
        let amp = 1.0 + 0.3 * ((i as f64 * 5.1 + 0.3).sin());
        for j in 0..m {
            let t = argvals[j];
            data[i + j * n] = amp * (2.0 * PI * (t + phase)).sin();
        }
    }
    let mat = FdMatrix::from_column_major(data, n, m).unwrap();
    (mat, argvals)
}
```

**Core bench-cell pattern — `bench_fpca_sentinel` (lines 82–96)** — exact template for `bench_p4_fpca`:
```rust
fn bench_fpca_sentinel(c: &mut Criterion) {
    let mut group = c.benchmark_group("audit_fpca");
    group.sample_size(20);
    group.measurement_time(std::time::Duration::from_secs(20));
    group.warm_up_time(std::time::Duration::from_secs(5));

    // Build input OUTSIDE b.iter() to avoid measuring the allocator
    let (data, argvals) = generate_curves(500, 200);
    group.bench_function("n500_m200", |b| {
        b.iter(|| fdata_to_pc_1d(black_box(&data), black_box(5usize), black_box(&argvals)))
    });

    group.finish();
}
```

**Multi-cell grid pattern — `bench_p3_karcher` (lines 278–345)** — exact template for the 6-cell `bench_p4_fpca` grid (re-declare data variables per cell, adjust `sample_size` per timing tier):
```rust
fn bench_p3_karcher(c: &mut Criterion) {
    let mut group = c.benchmark_group("audit_p3_karcher");

    // Per-cell tuning — small cells use sentinel defaults
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(20));
    group.warm_up_time(std::time::Duration::from_secs(5));
    let (data100_50, argvals50) = generate_curves(100, 50);
    group.bench_function("n100_m50", |b| {
        b.iter(|| {
            black_box(karcher_mean(
                black_box(&data100_50),
                black_box(&argvals50),
                black_box(20usize),
                black_box(1e-4),
                black_box(0.0),
            ))
        })
    });
    // ... additional cells follow with their own let bindings ...
    group.finish();
}
```

**Phase-4 FPCA grid — timing tier map** (from RESEARCH.md §3D):

| Cell | Estimated iter time | sample_size | measurement_time |
|------|--------------------|----|---|
| N=100, M=50 | <10 ms | 20 | 20s |
| N=100, M=200 | ~1–2 ms | 20 | 20s |
| N=500, M=50 | ~2–4 ms | 20 | 20s |
| N=500, M=200 | ~16 ms (Phase 1 baseline) | 20 | 20s |
| N=1000, M=50 | ~8–16 ms | 20 | 20s |
| N=1000, M=200 | ~64–256 ms | 10 | 20s |

**Elastic-FPCA bench cell with `iter_batched` / pre-computed setup** (from RESEARCH.md §Section 2B + §Pattern 2). `vert_fpca` and `joint_fpca` require a `KarcherMeanResult` — build it outside `b.iter()`:
```rust
// Pre-compute Karcher result OUTSIDE b.iter() — karcher_mean is setup, not the target
let (data, argvals) = generate_curves(100, 50);
let karcher = karcher_mean(&data, &argvals, 10, 1e-3, 0.0).unwrap(); // or expect()
group.bench_function("vert_fpca_n100_m50", |b| {
    b.iter(|| vert_fpca(black_box(&karcher), black_box(&argvals), black_box(5usize)))
});
```
For `joint_fpca`, pass `balance_c = Some(1.0)` to bypass `optimize_balance_c_raw` in the "main SVD path" cell (Pitfall B from RESEARCH.md).

**Registration pattern — `criterion_group!` macro (lines 736–751)** — append Phase 4 fns:
```rust
criterion_group!(
    benches,
    bench_fpca_sentinel,
    bench_matrix_sentinel,
    // ... existing Phase 1/3 groups ...
    bench_p3_karcher,
    bench_p3_karcher_banded,
    bench_p3_elastic_self,
    bench_p3_elastic_self_banded,
    bench_p3_elastic_cross,
    bench_p3_elastic_cross_banded,
    bench_p4_fpca,           // NEW — Phase 4
    bench_p4_elastic_fpca,   // NEW — Phase 4
);
criterion_main!(benches);
```

---

### `fdars-core/tests/alloc_audit_fpca.rs` (NEW — dhat integration test harness)

**Analog:** `fdars-core/benches/audit_hotpaths.rs` for the `generate_curves` helper and measurement discipline; dhat wiring pattern from `.planning/research/STACK.md:228–250` (verified HIGH confidence).

**Full dhat harness pattern** (RESEARCH.md §4B, verbatim from STACK.md):
```rust
// fdars-core/tests/alloc_audit_fpca.rs
#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

#[test]
#[cfg(feature = "dhat-heap")]
fn count_fpca_allocations_n500_m200() {
    let _profiler = dhat::Profiler::builder().testing().build();
    let (data, argvals) = generate_test_curves(500, 200);
    let _ = fdata_to_pc_1d(&data, 5, &argvals);
    let stats = dhat::HeapStats::get();
    println!("Total heap blocks: {}", stats.total_blocks);
    println!("Total heap bytes: {}", stats.total_bytes);
    println!("Peak heap bytes: {}", stats.peak_bytes);
    // Record — do not hard-assert a specific count (baseline, not regression gate)
}
```

**Critical constraint:** `#[global_allocator]` must be at file scope in an integration test under `fdars-core/tests/` — NOT inside a `#[cfg(test)]` module in `src/`. Using it inside a `src/` test module contaminates the global allocator for all tests in that compilation unit.

**`generate_test_curves` helper** — replicate the same deterministic generator from `audit_hotpaths.rs` lines 38–52. In the integration test, it is a private `fn generate_test_curves(n, m)` returning `(FdMatrix, Vec<f64>)` with identical logic (column-major `data[i + j * n]`, deterministic phase/amp, `FdMatrix::from_column_major`).

**Import pattern for the test file:**
```rust
use fdars_core::elastic_fpca::{joint_fpca, vert_fpca};
use fdars_core::matrix::FdMatrix;
use fdars_core::regression::fdata_to_pc_1d;
use std::f64::consts::PI;
```

**Run command** (RESEARCH.md §4B):
```bash
TMPDIR=/home/simonm/.cache/fdars-bench-tmp \
    cargo test -p fdars-core --features dhat-heap,linalg \
    -- count_fpca_allocations_n500_m200 --nocapture
```

**Three test cells to add:**
1. `count_fpca_allocations_n500_m200` — `fdata_to_pc_1d` (primary target, regression.rs:298 copy site)
2. `count_vert_fpca_allocations_n100_m50` — `vert_fpca` (elastic_fpca.rs:214 copy site)
3. `count_joint_fpca_allocations_n100_m50` — `joint_fpca` with `balance_c = Some(1.0)` (elastic_fpca.rs:317 copy site, optimizer bypassed)

---

### `fdars-core/Cargo.toml` (MODIFY — dev-dep + feature flag)

**Analog:** itself — existing `[dev-dependencies]` block at line 49 and implicit features.

**Existing dev-dependencies block (lines 49–52):**
```toml
[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
```

**Change: add `dhat` dev-dep** (RESEARCH.md §4A):
```toml
[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }
dhat = "0.3"          # NEW — allocation profiling for Phase 4 dhat integration tests
serde = { version = "1", features = ["derive"] }
serde_json = "1"
```

**Change: add `dhat-heap` feature** — add to the `[features]` section (check Cargo.toml for existing features section; if features are currently only `default = ["parallel"]` and `linalg`/`js`, add alongside):
```toml
[features]
default = ["parallel"]
parallel = ["rayon", "rayon-core"]
linalg = ["faer", "anofox-regression"]
js = ["getrandom/js"]
dhat-heap = []          # NEW — gates dhat allocation profiling; never enabled in release builds
```

**IMPORTANT:** Verify dhat version via `cargo search dhat` before committing — RESEARCH.md flags `"0.3.3"` as ASSUMED (Assumption A1). The `"0.3"` semver range is safe if 0.3.x is current.

---

### `.planning/research/AUDIT-REPORT.md` (MODIFY — append Phase 4 section)

**Analog:** the same file's existing `## Phase 2` and `## Phase 3` sections. The report is append-only per D-05.

**Phase 3 results table schema** (AUDIT-REPORT.md §Phase 3 results table — 8-column minimum):
```markdown
| Target | Cell (N×M) | Features | Mean time (run1) | Mean time (run2) | Variance | Confidence | Artifact |
|--------|-----------|---------|-----------------|-----------------|----------|------------|---------|
| `fdata_to_pc_1d` | 100×50 | linalg,parallel | [fill] ms | [fill] ms | [fill]% | OK/LOW | [run1](bench/p4_fpca_linalg,parallel_run1.txt) [run2](bench/..._run2.txt) |
```

**Backlog entry field shape** (RESEARCH.md §6C, D-07 from Phase 3 CONTEXT.md + REQUIREMENTS.md RPT-03):
```markdown
#### BACKLOG: `fdata_to_pc_1d` — eliminate redundant clone + to_dmatrix copy

| Field | Value |
|-------|-------|
| **Function** | `fdata_to_pc_1d` (`regression.rs:249`) |
| **Current cost** | ~16.2 ms at N=500, M=200 (`linalg,parallel`); full grid from Phase 4 |
| **Root cause** | `regression.rs:291` redundant `centered.clone()` — clones n×m matrix only to apply sqrt-weight scaling in-place, while `centered` is stored verbatim in `FpcaResult`. Plus `regression.rs:298` `weighted.to_dmatrix()` — plain memcpy of n×m `f64` values into `DMatrix<f64>` before SVD. Three allocations of O(n·m) bytes per call. |
| **Candidate fix** | Eliminate `centered.clone()` via in-place weight-scaling before storing to `FpcaResult`; consider zero-copy nalgebra construction if layout permits. |
| **Severity** | [TBD — Phase 9] |
| **Effort** | [TBD — Phase 9] |
| **Evidence** | Artifact `p4_fpca_linalg,parallel_run1.txt`; dhat baseline `p4_dhat_fpca_n500_m200.txt` |
```

**Phase 4 section header and sub-section structure** (RESEARCH.md §6A):
```markdown
---

## Phase 4: FPCA/SVD & Allocation Audit — Benchmark Results

### Results Table (criterion — full N×M grid)

### Allocation Audit (dhat — bytes/allocations per FPCA call)

### SVD-Compute vs Copy Split

### Phase 6 Go/No-Go Decision

### Draft Backlog (FPCA/SVD slice)
```

**Phase 6 go/no-go template** (RESEARCH.md §5C — planner must fill one of these three statements):
```markdown
### Phase 6 Go/No-Go Decision

Based on the Phase 4 measurements:

- [ ] "Copy is the dominant cost (≥50% of wall-clock) → Phase 6 **not warranted**"
- [ ] "SVD is the dominant cost (≥50% of wall-clock) → Phase 6 **triggered**"
- [ ] "Both costs are comparable (split is unclear) → Phase 6 **triggered with caveat**"

Supporting numbers: [criterion wall-clock at N=500,M=200] ms wall-clock; [dhat total_bytes] bytes allocated; estimated copy time [X µs] = [Y]% of wall-clock.
```

**Wall-clock share calculation method** (RESEARCH.md §5B):
- Wall-clock from criterion `bench_p4_fpca` at N=500, M=200 (Phase 1 baseline: ~16.2 ms).
- Copy time estimate: `total_bytes_from_dhat ÷ memory_bandwidth_30GBps` → expected ~26 µs for the `to_dmatrix()` copy at 800 KB.
- Report the fraction: `copy_time_µs / wall_clock_µs × 100%`.

---

### `.planning/phases/04-fpca-svd-allocation-audit/04-COVERAGE.md` (NEW)

**Analog:** `.planning/phases/03-elastic-alignment-hot-path/03-COVERAGE.md` — minimal one-line stub.

**Exact content to copy:**
```markdown
No external API integration: benchmark-only phase measuring internal fdars-core FPCA/SVD functions plus dhat allocation profiling.
```

---

## Shared Patterns

### `generate_curves` helper (column-major, deterministic)
**Source:** `fdars-core/benches/audit_hotpaths.rs` lines 38–52
**Apply to:** `bench_p4_fpca`, `bench_p4_elastic_fpca` (bench file), and `alloc_audit_fpca.rs` (test file — replicate as `generate_test_curves`)

The generator uses no RNG — deterministic sine with per-curve phase/amplitude variation. `data[i + j * n]` is column-major. Both the bench and the integration test must use this exact layout to match `FdMatrix::from_column_major` expectations.

### Criterion cell discipline (inherited from Phase 1/3)
**Source:** every `bench_*_sentinel` fn in `audit_hotpaths.rs`, confirmed in Phase 3 PATTERNS.md
**Apply to:** all new `bench_p4_fpca` and `bench_p4_elastic_fpca` cells
- Build inputs **outside** `b.iter()` (comment: `// Build input OUTSIDE b.iter() to avoid measuring the allocator`)
- Wrap **both** inputs and return value in `black_box(...)` (Pitfall 3)
- `Result<T,E>` returns: wrapping inputs is sufficient (`bench_fpca_sentinel` line 92 precedent)
- Per-cell `sample_size` / `measurement_time` / `warm_up_time` — use the tier map above

### TMPDIR workaround (mandatory)
**Source:** AUDIT-REPORT.md §Infrastructure vs. Code Failure Triage; MEMORY.md
**Apply to:** every `cargo bench` and `cargo test` invocation in Phase 4 plan tasks
```bash
TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo bench -p fdars-core \
    --features linalg,parallel \
    --bench audit_hotpaths -- audit_p4_fpca
```
Omitting `TMPDIR` causes SIGBUS linker failures from `/tmp` at ~94% capacity — classified as infrastructure failure, not a code defect.

### Artifact naming and environment header
**Source:** AUDIT-REPORT.md §Artifact Naming Convention (D-06); §Toolchain Version Capture
**Apply to:** all `.planning/research/bench/p4_*` artifacts
```
=== ENVIRONMENT ===
rustc rustc 1.97.0 (2d8144b78 2026-07-07)
cargo cargo 1.97.0 (c980f4866 2026-06-30)
Running benches/audit_hotpaths.rs (target/release/deps/...)
```
Phase 4 artifact names: `p4_fpca_linalg,parallel_run{1,2}.txt`, `p4_fpca_linalg_run1.txt`, `p4_elastic_fpca_vert_linalg,parallel_run{1,2}.txt`, `p4_elastic_fpca_joint_linalg,parallel_run{1,2}.txt`, `p4_dhat_fpca_n500_m200.txt`, `p4_dhat_fpca_n500_m200.json`.

### dhat feature-gate discipline
**Source:** RESEARCH.md §4B, §4A
**Apply to:** all dhat code in `alloc_audit_fpca.rs` and references in `Cargo.toml`

Every dhat symbol must be wrapped in `#[cfg(feature = "dhat-heap")]`. The `#[global_allocator]` attribute AND each `#[test]` function must both be guarded. This ensures `dhat` is never activated in normal CI (`cargo test`) or release builds — only when explicitly `--features dhat-heap` is passed.

### Key SVD copy site reference (for report / backlog)
**Source:** `fdars-core/src/regression.rs` lines 291 and 298 (verified HIGH confidence in RESEARCH.md §1A)

| Line | Operation | Bytes (N=500, M=200) |
|------|-----------|----------------------|
| 291 | `centered.clone()` — redundant copy before in-place weight scaling | 500×200×8 = 800 KB |
| 298 | `weighted.to_dmatrix()` — `DMatrix::from_column_slice(nrows, ncols, &self.data)` | 500×200×8 = 800 KB |

`to_dmatrix()` definition at `fdars-core/src/matrix.rs:310–312`:
```rust
pub fn to_dmatrix(&self) -> DMatrix<f64> {
    DMatrix::from_column_slice(self.nrows, self.ncols, &self.data)
}
```
Column-major → column-major memcpy; no transposition cost. Both layouts identical.

---

## No Analog Found

None. All five files/modifications have exact or role-match analogs in the codebase or in prior phase artifacts. No RESEARCH.md fallback patterns are required.

---

## Metadata

**Analog search scope:** `fdars-core/benches/audit_hotpaths.rs`, `fdars-core/Cargo.toml`, `.planning/research/AUDIT-REPORT.md`, `.planning/phases/03-elastic-alignment-hot-path/03-PATTERNS.md`, `.planning/research/STACK.md` (dhat wiring via RESEARCH.md citations)
**Files scanned:** 6
**Pattern extraction date:** 2026-08-08
