# Phase 04: FPCA/SVD & Allocation Audit — Research

**Researched:** 2026-08-08
**Domain:** fdars-core FPCA/SVD hot path — criterion benchmarking, dhat allocation profiling, SVD-copy overhead separation
**Confidence:** HIGH (all findings source-verified from fdars-core/src/ and prior phase summaries read in this session)

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| PERF-03 (FPCA share) | Criterion benchmarks measure the top hot-path suspects in release build with correct feature flags and `black_box`, producing a results table tagged with feature set and toolchain version | Section 2 enumerates the bench grid targets; Section 3 provides the exact invocation convention from Phase 1/3; reuse `bench_fpca_sentinel` group in `audit_hotpaths.rs` |
| PERF-04 | An allocation audit (dhat) quantifies the documented `FdMatrix→DMatrix` SVD-copy overhead (and other allocation hotspots) with a reproducible baseline | Section 4 details the dhat wiring strategy (feature-gated dev-dep, testing() mode, integration-test harness); Section 5 covers wall-clock-share method and Phase 6 trigger |
</phase_requirements>

---

## Summary

Phase 4 is the FPCA/SVD slice of the criterion bench grid (PERF-03) plus the dhat allocation audit (PERF-04). It separates SVD compute cost from copy/allocation overhead so Phase 6 (conditional faer comparison) can make a go/no-go decision on solid evidence.

The Phase 1 FPCA sentinel already measured `fdata_to_pc_1d` at N=500, M=200 as 16.2 ms (two-run variance 1.5%, EXCELLENT confidence) [VERIFIED: .planning/research/AUDIT-REPORT.md §Workload Matrix]. Phase 4 expands this to the full N∈{100,500,1000}×M∈{50,200} grid for `fdata_to_pc_1d` and adds the four elastic-FPCA functions with `to_dmatrix()` SVD sites. The dhat audit then isolates what fraction of the ~16 ms wall-clock is allocation overhead vs pure SVD compute.

The **Phase 6 go/no-go trigger** is stated in ROADMAP.md §Phase 6: "the comparison is performed only if SVD is a significant share of FPCA runtime **and** copy is not the dominant cost" [VERIFIED: .planning/ROADMAP.md:131]. Phase 4 must produce the numbers that answer exactly this: if allocation (FdMatrix→DMatrix copy) dominates wall-clock, nalgebra-vs-faer comparison is unwarranted; if pure SVD dominates, it is warranted.

**Primary recommendation:** Extend `audit_hotpaths.rs` with a `bench_p4_fpca` criterion group (full N×M grid), run both `linalg,parallel` and `linalg` (no-parallel) since FPCA is sequential, add a dhat integration test under `--features dhat-heap`, save all artifacts under `.planning/research/bench/p4_*`, and append the Phase 4 section to `AUDIT-REPORT.md`.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Criterion bench grid (FPCA) | `fdars-core/benches/audit_hotpaths.rs` | `.planning/research/bench/p4_*` (artifacts) | Extend existing audit bench; no new file needed |
| dhat allocation profiling | `fdars-core/tests/` (integration test harness) | `fdars-core/Cargo.toml` (feature flag) | dhat requires a separate process; integration tests are the canonical mechanism |
| Wall-clock-share analysis | `.planning/research/AUDIT-REPORT.md` | — | Manual calculation: dhat peak_bytes / wall-clock from criterion; documented in report |
| AUDIT-REPORT Phase 4 section | `.planning/research/AUDIT-REPORT.md` | — | Append-only per D-05 (single growing report) |
| GSD-ready backlog | `.planning/research/AUDIT-REPORT.md` | — | Backlog entries appended to same file, matching Phase 3 D-07 field shape |

---

## Section 1: SVD Call Sites — Complete Inventory

All call sites below were verified by reading source files in this session.

### 1A: `fdata_to_pc_1d` — Primary FPCA Target

[VERIFIED: fdars-core/src/regression.rs:249-322]

The full allocation chain per `fdata_to_pc_1d` call (verbatim from source):

```
regression.rs:167  center_columns(data)        → allocates FdMatrix::zeros(n, m)   — O(n·m) bytes
regression.rs:291  centered.clone()            → allocates FdMatrix (n×m copy)     — O(n·m) bytes
regression.rs:298  weighted.to_dmatrix()       → allocates DMatrix<f64> (n×m copy) — O(n·m) bytes
```

Three distinct heap allocations of size n×m per call. The `weighted.to_dmatrix()` call at line 298 is the copy documented in CONCERNS.md — it uses `DMatrix::from_column_slice(self.nrows, self.ncols, &self.data)` [VERIFIED: fdars-core/src/matrix.rs:310-312] which is a plain memcpy into nalgebra's storage. Both layouts are column-major so the copy is O(n·m) bytes with no transposition cost.

The `centered.clone()` at regression.rs:291 is a redundant copy: it clones the centered matrix to `weighted` only to apply sqrt-weight scaling in-place, while `centered` itself is also stored verbatim in `FpcaResult.centered`. A pre-allocated buffer strategy could eliminate this allocation.

**Feature gate:** `[sequential]` — `center_columns` (regression.rs:167) is a plain `for j in 0..m` loop; nalgebra SVD is always sequential regardless of `parallel` feature. Running with or without `--features parallel` produces identical FPCA timings (confirmed in Phase 1 audit_hotpaths.rs doc comment lines 7-14) [VERIFIED: fdars-core/benches/audit_hotpaths.rs:7-14].

### 1B: Elastic-FPCA SVD Sites — `to_dmatrix()` calls

[VERIFIED: grep on fdars-core/src/elastic_fpca.rs in this session]

| File:Line | Enclosing function | Matrix dimensions | Is `to_dmatrix()` copy | Feature gate |
|-----------|-------------------|------------------|------------------------|--------------|
| `elastic_fpca.rs:214` | `horiz_fpca` (line 182) | n×m shooting vectors → centered n×m | Yes — `centered.to_dmatrix()` | `[always]` |
| `elastic_fpca.rs:317` | `joint_fpca` (line 274) | n×(m_aug+m) combined | Yes — `combined.to_dmatrix()` | `[always]` |
| `elastic_fpca.rs:483` | `vert_fpca_from_alignment` (line 366) | n×m shooting vectors → centered | Yes — `centered.to_dmatrix()` | `[always]` |
| `elastic_fpca.rs:584` | `joint_fpca_from_alignment` (line 541) | n×(m_aug+m) combined | Yes — `combined.to_dmatrix()` | `[always]` |
| `elastic_fpca.rs:930` | `optimize_balance_c_raw` (line 905) | n×(m_aug+m) combined | Yes — `combined.to_dmatrix()` inside `eval_c` closure; called ≤20× per golden-section loop (STATE.md decision) | `[always]` |

**Note on elastic_fpca.rs:122 and :399:** These two `SVD::new` calls at lines 122 and 399 do NOT use `to_dmatrix()`. They operate on a `nalgebra::DMatrix<f64>` built directly by `build_symmetric_covariance` (which constructs a `nalgebra::DMatrix::zeros(d, d)` natively at line 799) [VERIFIED: fdars-core/src/elastic_fpca.rs:792-816]. These are **covariance-matrix SVDs** (m×m, not n×m), not FdMatrix→DMatrix copy sites. They are allocation hotspots of a different kind and are secondary targets for Phase 4 dhat.

### 1C: Other In-Scope SVD Sites

| File:Line | Function | Matrix | Phase 4 scope? |
|-----------|----------|--------|----------------|
| `alignment/nd.rs:705` | (ND elastic FPCA Gram matrix) | m×m Gram | Yes — `gram.to_dmatrix()` copy |
| `spm/mfpca.rs:336` | (multivariate FPCA) | stacked n×m | Yes — `stacked.to_dmatrix()` copy |

**Sites excluded from Phase 4:** `elastic_fpca.rs:122` and `elastic_fpca.rs:399` (covariance-matrix SVD, no `to_dmatrix()` copy), `matrix.rs:682` (`#[cfg(test)]` only). [VERIFIED: fdars-core/src/elastic_fpca.rs:122, 399, 792-816; fdars-core/src/matrix.rs:682]

---

## Section 2: Criterion Bench Grid — Targets and Cells

### 2A: Primary target — `fdata_to_pc_1d`

**Workload matrix cells** (from AUDIT-REPORT.md §Workload Matrix, FPCA/SVD row): N∈{100, 500, 1000} × M∈{50, 200}. Full grid = 6 cells. The M=500 cells are omitted per the ROADMAP Phase 4 success criteria which specifies only `N∈{100,500,1000}×M∈{50,200}`. [VERIFIED: .planning/ROADMAP.md:101]

M-scaling note: SVD cost is O(min(n,m)²·max(n,m)). Since n < m is typical for FPCA workloads (more evaluation points than curves), the dominant cost is O(n²·m). For n=1000, m=200: O(10⁶×200)=200M ops — tractable. The Phase 1 sentinel at N=500, M=200 measured ~16 ms [VERIFIED: .planning/research/AUDIT-REPORT.md §Phase 1 Baseline Cells]. N=1000 will approximately quadruple this (O(n²) scaling) to ~64 ms.

**Bench function to add:** Extend `audit_hotpaths.rs` with a new criterion group `audit_p4_fpca` sweeping all 6 cells. The sentinel `bench_fpca_sentinel` (group `audit_fpca`) already covers N=500, M=200 [VERIFIED: fdars-core/benches/audit_hotpaths.rs:76-96] — the new Phase 4 group is a full 6-cell sweep, not a replacement.

**Feature flag:** Run at `linalg,parallel` (primary) AND at `linalg` (no-parallel) as a secondary run to confirm FPCA timings are identical across parallel/no-parallel (formalizing the Phase 1 D-04 finding). Since center_columns and nalgebra SVD are both sequential, the two combos should produce identical numbers — recording this explicitly addresses ROADMAP SC1's "tagged with feature set" requirement.

### 2B: Elastic-FPCA SVD targets (secondary)

The four public elastic-FPCA functions each contain at least one `to_dmatrix()` SVD call:

| Function | File | Lines (pub fn) | Primary SVD site |
|----------|------|----------------|------------------|
| `vert_fpca` | elastic_fpca.rs | :89 | :214 |
| `joint_fpca` | elastic_fpca.rs | :274 | :317 |
| `horiz_fpca` | elastic_fpca.rs | :182 | but NOT `to_dmatrix()` — uses `build_symmetric_covariance` then SVD; see 1B note |
| `vert_fpca_from_alignment` | elastic_fpca.rs | :366 | :483 |
| `joint_fpca_from_alignment` | elastic_fpca.rs | :541 | :584 |

[VERIFIED: fdars-core/src/elastic_fpca.rs:89, 182, 214, 274, 317, 366, 399, 452, 483, 541, 584]

**Phase 4 bench scope:** Add bench cells for `vert_fpca` and `joint_fpca` (the two that use `to_dmatrix()` as primary SVD) at representative sizes N∈{100,500}×M∈{50,200}. These functions depend on `KarcherMeanResult` (require running `karcher_mean` first as setup), so their bench cells must build the Karcher result outside `b.iter()` or use `iter_batched`.

**Important:** The ROADMAP Phase 4 SC1 specifies "fdata_to_pc_1d (and elastic-FPCA SVD sites)" — so including at least one elastic-FPCA function in the bench grid is a success criterion, not optional. [VERIFIED: .planning/ROADMAP.md:101]

---

## Section 3: Established Bench Harness — Reuse Conventions

### 3A: File and Registration

`fdars-core/benches/audit_hotpaths.rs` is the audit bench file established in Phase 1 [VERIFIED: fdars-core/benches/audit_hotpaths.rs:1]. The `[[bench]] name = "audit_hotpaths"` entry exists in `fdars-core/Cargo.toml` at line 91 [VERIFIED: fdars-core/Cargo.toml:90-92]. **Do not create a new file** — extend `audit_hotpaths.rs` following the Phase 3 precedent (append new bench functions, register in the `criterion_group!` macro at line 736).

### 3B: Invocation Convention

From Phase 3 summaries [VERIFIED: .planning/phases/03-elastic-alignment-hot-path/03-01-SUMMARY.md:65] and AUDIT-REPORT.md §Methodology:

```bash
TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo bench -p fdars-core \
    --features linalg,parallel \
    --bench audit_hotpaths -- audit_p4_fpca
```

**Mandatory elements:**
- `TMPDIR=...` — required because `/tmp` tmpfs is at ~94% capacity; doctest-linker bus errors occur without it (Pitfall 8 / MEMORY.md)
- `--features linalg,parallel` — primary audit build per AUDIT-REPORT §Methodology
- `--bench audit_hotpaths` — target specific bench file (avoids running all 9 bench files)
- Group filter `-- audit_p4_fpca` — target only Phase 4 bench groups

### 3C: `black_box` Pattern

Both inputs and outputs wrapped per AUDIT-REPORT §Methodology:

```rust
// Source: fdars-core/benches/audit_hotpaths.rs:91-93
group.bench_function("n500_m200", |b| {
    b.iter(|| fdata_to_pc_1d(black_box(&data), black_box(5usize), black_box(&argvals)))
});
```

For `Result<T, E>` returns, wrapping inputs is sufficient (verified via Phase 1 pattern). [VERIFIED: fdars-core/benches/audit_hotpaths.rs:82-96]

### 3D: Timing Tuning Guidance

From Phase 3 precedent, per-cell tuning [VERIFIED: .planning/phases/03-elastic-alignment-hot-path/03-02-SUMMARY.md:88-90]:

| Cell size | Estimated iter time | Recommended settings |
|-----------|---------------------|----------------------|
| N≤100, M≤50 | <100 ms | `sample_size(20)`, `measurement_time(20s)` |
| N≤500, M≤200 (FPCA) | ~16–64 ms | `sample_size(20)`, `measurement_time(20s)` |
| N=1000, M=200 (FPCA) | ~64–256 ms (O(n²) growth) | `sample_size(10)`, `measurement_time(20s)` |

FPCA cells are fast (sub-second) even at large N because SVD is O(n²·m) not O(n²·m²) — all 6 grid cells should complete with standard settings. No `measurement_time(60s)` extension needed (contrast with elastic O(n²·m²) cells).

### 3E: Artifact Naming Convention

Phase 4 artifacts follow the same scheme as Phases 1 and 3 [VERIFIED: AUDIT-REPORT.md §Artifact Naming Convention]:

```
p4_fpca_linalg,parallel_run1.txt
p4_fpca_linalg,parallel_run2.txt
p4_fpca_linalg_run1.txt          ← secondary: no-parallel comparison
p4_elastic_fpca_<func>_linalg,parallel_run1.txt
```

Save under `.planning/research/bench/` (directory already exists). Each artifact starts with `=== ENVIRONMENT ===` header (rustc version + cargo version) followed by the criterion binary path confirming `/release/deps/`.

### 3F: generate_curves Helper

The deterministic input generator already exists in `audit_hotpaths.rs` [VERIFIED: fdars-core/benches/audit_hotpaths.rs:38-52]:

```rust
fn generate_curves(n: usize, m: usize) -> (FdMatrix, Vec<f64>) { ... }
```

Reuse directly. For elastic-FPCA bench cells that need a `KarcherMeanResult`, call `karcher_mean` on the generated curves outside `b.iter()` using `iter_batched` or build the result in the bench setup.

---

## Section 4: dhat Allocation Audit

### 4A: dhat Is Not Yet Wired

`dhat` is NOT currently a dev-dependency in `fdars-core/Cargo.toml` [VERIFIED: fdars-core/Cargo.toml:49-53 — only `criterion` and `serde`/`serde_json` are dev-deps]. It must be added in Wave 0 or the first plan task.

The STACK.md research (from Phase 1 research cycle) specifies the wiring pattern [VERIFIED: .planning/research/STACK.md:80-90]:

```toml
# fdars-core/Cargo.toml — add to [dev-dependencies]:
dhat = "0.3"

# fdars-core/Cargo.toml — add to [features]:
dhat-heap = []
```

The `dhat-heap` feature flag gates allocation profiling so it never activates in normal CI or release builds.

**Important:** The `dhat` version in STACK.md is `0.3.3`. STACK.md is tagged MEDIUM confidence for tool versions. The planner must verify current `dhat` version on crates.io before emitting the `Cargo.toml` change. [ASSUMED — version not re-verified in this session; use `cargo search dhat` or `cargo add dhat --dev` to get current version]

### 4B: dhat Integration Test Harness

The STACK.md pattern [VERIFIED: .planning/research/STACK.md:234-248]:

```rust
// In fdars-core/tests/alloc_audit_fpca.rs (new file for Phase 4):
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

Run with:
```bash
TMPDIR=/home/simonm/.cache/fdars-bench-tmp \
    cargo test -p fdars-core --features dhat-heap,linalg \
    -- count_fpca_allocations_n500_m200 --nocapture
```

**Critical constraint:** dhat requires a **separate process** (integration test, not inline `#[test]` in src/). The `#[global_allocator]` attribute sets the global allocator for the test process, so it must be in a separate integration test file under `fdars-core/tests/`. Using it inside the library's `#[cfg(test)]` modules would affect all tests in that compilation unit.

### 4C: What to Measure per FPCA Call

For each cell measured, record:
1. `stats.total_blocks` — number of heap allocations (expect ≈ 3 major ones: `center_columns` FdMatrix, `centered.clone()`, `weighted.to_dmatrix()`)
2. `stats.total_bytes` — total bytes allocated across all blocks
3. `stats.peak_bytes` — peak live heap bytes (tells you the memory cost at peak)

For N=500, M=200: each FdMatrix allocation is 500×200×8 bytes = 800 KB. Three allocations → ~2.4 MB total, ~1.6 MB peak (two FdMatrix live simultaneously + DMatrix).

**For ranking hotspots:** run the same test over `vert_fpca` and `joint_fpca` to compare their allocation profiles against `fdata_to_pc_1d`. The `optimize_balance_c_raw` closure at `elastic_fpca.rs:930` calls `SVD::new(combined.to_dmatrix(), ...)` inside a golden-section loop of ≤20 iterations [VERIFIED: STATE.md decision, elastic_fpca.rs:942] — it will show as a repeating allocator hotspot in the dhat output.

### 4D: Baseline Artifact Location

Save dhat output to `.planning/research/bench/p4_dhat_fpca_n500_m200.txt` (print `--nocapture` output) following the D-06 naming convention. For the dhat JSON heap profile (written to `dhat-heap.json` by default), save as `.planning/research/bench/p4_dhat_fpca_n500_m200.json`.

---

## Section 5: Wall-Clock Share Method — SVD-Compute vs Copy Split

### 5A: Pitfall 5 Definition

Pitfall 5 in PITFALLS.md [VERIFIED: .planning/research/PITFALLS.md:93-113]:

> "The audit reports 'SVD takes 12 ms' but the majority of that time is memory allocation: the `FdMatrix → nalgebra::DMatrix` copy... A fix that removes the copy would halve the time, but the audit missed this because it only measured wall-clock time without heap profiling."

The Phase 4 success criterion SC3 requires: "The report states allocation cost as a share of wall-clock for the top FPCA path, so the SVD-compute vs copy split is explicit."

### 5B: Method

The wall-clock-share calculation uses three numbers all produced within this phase:

1. **Wall-clock per call** — from criterion `bench_p4_fpca` at N=500, M=200, `linalg,parallel`. Phase 1 sentinel baseline: ~16.2 ms [VERIFIED: AUDIT-REPORT.md §Phase 1 Baseline Cells].

2. **Allocation time estimate** — from dhat `total_bytes` at N=500, M=200, estimate copy time using memory bandwidth. The `to_dmatrix()` copy at regression.rs:312 is a `DMatrix::from_column_slice(...)` which is a contiguous memcpy of n×m f64 values. At 500×200×8 = 800 KB, on a modern Linux machine with ~30 GB/s memory bandwidth, the raw copy takes ~26 µs. Against ~16 ms wall-clock, this is ~0.16% — if true, copy is negligible and SVD compute dominates.

   However, nalgebra's DMatrix allocation also incurs heap allocator overhead. At 800 KB, this is a single `malloc(800_000)` — typically <10 µs on glibc. Still negligible against 16 ms.

3. **Cross-check using iai-callgrind** (optional): If criterion numbers are noisy, iai-callgrind reports instruction counts deterministically. The ratio (instructions for the `to_dmatrix` call / total instructions for `fdata_to_pc_1d`) is the instruction-level share.

### 5C: Phase 6 Trigger

From ROADMAP.md §Phase 6 SC1 [VERIFIED: .planning/ROADMAP.md:131-134]:

> "the comparison is performed only if SVD is a significant share of FPCA runtime **and** copy is not the dominant cost; otherwise the report records 'not warranted' with the supporting numbers"

The Phase 4 report must explicitly state one of:
- "Copy is the dominant cost (≥50% of wall-clock) → Phase 6 not warranted"
- "SVD is the dominant cost (≥50% of wall-clock) → Phase 6 triggered"
- "Both costs are comparable (split is unclear) → Phase 6 triggered with caveat"

Given the rough estimate above (~0.16% copy vs wall-clock), the Phase 6 trigger is **likely to fire** — SVD compute will dominate. But this must be confirmed with measured numbers, not estimates.

---

## Section 6: Report + Backlog Conventions

### 6A: Report Structure

Phase 4 appends a new section to `.planning/research/AUDIT-REPORT.md` (append-only per D-05) [VERIFIED: .planning/research/AUDIT-REPORT.md lines 1-10, single-file pattern]. The section header:

```markdown
## Phase 4: FPCA/SVD & Allocation Audit — Benchmark Results
```

Sub-sections to include (matching Phase 3 structure):

```markdown
### Results Table (criterion — full N×M grid)
### Allocation Audit (dhat — bytes/allocations per FPCA call)
### SVD-Compute vs Copy Split
### Phase 6 Go/No-Go Decision
### Draft Backlog (FPCA/SVD slice)
```

### 6B: Results Table Schema

From Phase 3 precedent [VERIFIED: .planning/research/AUDIT-REPORT.md §Phase 3 results table]:

| Target | Cell (N×M) | Features | Mean time (run1) | Mean time (run2) | Variance | Confidence | Artifact |
|--------|-----------|---------|-----------------|-----------------|----------|------------|---------|

8 columns minimum. Every row links to its artifact (Pitfall 17).

### 6C: Backlog Entry Fields (D-07 shape from Phase 3)

From Phase 3 CONTEXT.md §D-07 [VERIFIED: .planning/phases/03-elastic-alignment-hot-path/03-CONTEXT.md:39]:

> "Each backlog entry carries: function, current-cost (Phase-4 measured numbers), root-cause citing the relevant AUDIT-REPORT anti-pattern / complexity row, plus a one-line candidate fix"

The REQUIREMENTS.md RPT-03 completeness checklist adds [VERIFIED: .planning/REQUIREMENTS.md:32]:

> "location/area, current cost or gap, root cause, proposed direction, severity, effort estimate, and evidence link"

Minimal field set for Phase 4 backlog entries:

| Field | Example |
|-------|---------|
| **Function** | `fdata_to_pc_1d` |
| **Current cost** | ~16.2 ms at N=500,M=200 (Phase 1 sentinel); full grid numbers from this phase |
| **Root cause** | `regression.rs:291` redundant clone + `regression.rs:298` to_dmatrix copy; 3 allocations of n×m per call |
| **Candidate fix** | Eliminate `centered.clone()` via in-place weight-scaling before storing to FpcaResult; use truncated SVD |
| **Evidence link** | Artifact `p4_fpca_linalg,parallel_run1.txt`, dhat baseline `p4_dhat_fpca_n500_m200.txt` |

Phase 9 will add severity (P1/P2/P3) and effort (S/M/L) rankings when all phase findings are cross-ranked. For Phase 4 backlog stubs, leave severity/effort as `[TBD — Phase 9]` or draft a preliminary estimate.

### 6D: COVERAGE.md

Create `.planning/phases/04-fpca-svd-allocation-audit/04-COVERAGE.md` with the same minimal format as Phase 3 [VERIFIED: .planning/phases/03-elastic-alignment-hot-path/03-COVERAGE.md]:

```markdown
No external API integration: benchmark-only phase measuring internal fdars-core FPCA/SVD functions plus dhat allocation profiling.
```

---

## Architecture Patterns

### System Architecture Diagram

```
audit_hotpaths.rs (extended with Phase 4 groups)
    │
    ├── bench_p4_fpca (NEW: 6-cell grid, linalg,parallel + linalg)
    │       N∈{100,500,1000} × M∈{50,200}
    │       → artifacts: p4_fpca_*_run{1,2}.txt
    │
    └── bench_p4_elastic_fpca (NEW: vert_fpca + joint_fpca, representative cells)
            → artifacts: p4_elastic_fpca_*_run{1,2}.txt

fdars-core/tests/alloc_audit_fpca.rs (NEW: dhat integration tests)
    │
    ├── count_fpca_allocations_n500_m200       (fdata_to_pc_1d)
    ├── count_vert_fpca_allocations_n100_m50   (vert_fpca)
    └── count_joint_fpca_allocations_n100_m50  (joint_fpca)
    → artifacts: p4_dhat_fpca_*.txt, p4_dhat_fpca_*.json

.planning/research/AUDIT-REPORT.md
    └── ## Phase 4: FPCA/SVD & Allocation Audit
            ├── Results Table
            ├── Allocation Audit
            ├── SVD-Compute vs Copy Split
            ├── Phase 6 Go/No-Go Decision
            └── Draft Backlog (SVD-copy elimination, truncated-SVD candidates)
```

### Recommended Project Structure

No new source files in `fdars-core/src/`. New files:

```
fdars-core/
├── Cargo.toml                            ← add dhat dev-dep + dhat-heap feature
├── benches/
│   └── audit_hotpaths.rs                 ← extend with bench_p4_fpca, bench_p4_elastic_fpca
└── tests/
    └── alloc_audit_fpca.rs               ← new: dhat integration tests

.planning/
└── research/
    ├── AUDIT-REPORT.md                   ← append Phase 4 section
    └── bench/
        ├── p4_fpca_linalg,parallel_run1.txt
        ├── p4_fpca_linalg,parallel_run2.txt
        ├── p4_fpca_linalg_run1.txt
        ├── p4_elastic_fpca_vert_linalg,parallel_run1.txt
        ├── p4_elastic_fpca_vert_linalg,parallel_run2.txt
        ├── p4_elastic_fpca_joint_linalg,parallel_run1.txt
        ├── p4_elastic_fpca_joint_linalg,parallel_run2.txt
        ├── p4_dhat_fpca_n500_m200.txt
        └── p4_dhat_fpca_n500_m200.json
```

### Pattern 1: Extend `criterion_group!` macro

After adding bench functions, add them to the macro at audit_hotpaths.rs:736 [VERIFIED: fdars-core/benches/audit_hotpaths.rs:736-751]:

```rust
criterion_group!(
    benches,
    bench_fpca_sentinel,
    // ... existing Phase 1/3 groups ...
    bench_p4_fpca,           // NEW — Phase 4
    bench_p4_elastic_fpca,   // NEW — Phase 4
);
```

### Pattern 2: `iter_batched` for Elastic-FPCA Setup

Elastic-FPCA bench cells require pre-computed `KarcherMeanResult`. Use `iter_batched` if the setup itself is expensive, or pre-compute outside `b.iter()` for stable setup:

```rust
// Build KarcherMeanResult OUTSIDE b.iter() — karcher_mean is expensive but
// is setup, not the measurement target.
let (data, argvals) = generate_curves(100, 50);
let karcher = karcher_mean(&data, &argvals, 10, 1e-3, 0.0);
group.bench_function("vert_fpca_n100_m50", |b| {
    b.iter(|| vert_fpca(black_box(&karcher), black_box(&argvals), black_box(5usize)))
});
```

This matches the Phase 1 "Build input OUTSIDE b.iter()" discipline [VERIFIED: fdars-core/benches/audit_hotpaths.rs:89-90].

### Anti-Patterns to Avoid

- **Using bench_fpca_sentinel as Phase 4's grid:** The existing `bench_fpca_sentinel` (group `audit_fpca`) covers only N=500,M=200. Phase 4 requires a 6-cell grid in a new group `audit_p4_fpca` — do not replace or modify the sentinel.
- **Confusing `center_1d` with `center_columns`:** `fdata.rs:center_1d` uses `iter_maybe_parallel!` and IS parallelized. `regression.rs:center_columns` (called by `fdata_to_pc_1d`) is plain sequential. These are different functions [VERIFIED: fdars-core/src/regression.rs:167; Phase 2 RESEARCH Pitfall 1].
- **Running dhat inside a `#[cfg(test)]` module in src/:** The `#[global_allocator]` must be in a separate integration test file; inline tests share the global allocator with other tests in the same compilation unit, producing incorrect counts.
- **Treating elastic_fpca.rs:122 and :399 as `to_dmatrix()` copy sites:** These two `SVD::new` calls operate on `nalgebra::DMatrix::zeros(d, d)` built natively by `build_symmetric_covariance` — no FdMatrix→DMatrix copy involved [VERIFIED: fdars-core/src/elastic_fpca.rs:792-816].

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Allocation profiling | Manual instrumented counters or printf allocation counts | `dhat = "0.3"` dev-dependency with `testing()` mode | dhat counts every heap block precisely; manual approaches miss small allocations |
| Estimating allocation cost | Calculate from memory-bandwidth theory | Run dhat and report measured `total_bytes` + elapsed time from a separate timing loop | Theory gives ~26 µs; actual glibc malloc/free overhead is workload-dependent |
| Counting `to_dmatrix()` sites | Manual code inspection | `grep -n "to_dmatrix()" fdars-core/src/` | Already executed in Phase 2 — 8 production sites confirmed |
| Expressing "copy cost as % of wall-clock" | Writing a custom profiler | Criterion wall-clock ÷ (dhat total_bytes ÷ memory_bandwidth) | Phase 4 SC3 only requires the report to state the fraction; rough calculation from measured numbers is sufficient |

---

## Common Pitfalls

### Pitfall A: Including elastic_fpca.rs:122/:399 in the "SVD-copy hotspot" count

**What goes wrong:** Lines 122 and 399 contain `SVD::new(k_mat, ...)` where `k_mat` is a `nalgebra::DMatrix<f64>` built natively by `build_symmetric_covariance`. There is no `to_dmatrix()` call; no FdMatrix→DMatrix copy is made. Including them in the "copy hotspot" category inflates the count and suggests eliminating a copy that doesn't exist.

**How to avoid:** The Phase 4 dhat targets are the `to_dmatrix()` SVD sites only. Verified sites: `regression.rs:298`, `elastic_fpca.rs:214/317/483/584/930`, `alignment/nd.rs:705`, `spm/mfpca.rs:336`. [VERIFIED: this session]

**Warning signs:** If dhat shows 0 new `DMatrix::from_column_slice` allocations for the covariance-SVD path, that confirms build_symmetric_covariance constructs the DMatrix directly.

### Pitfall B: Measuring `optimize_balance_c_raw` (elastic_fpca.rs:905) as a main FPCA path

**What goes wrong:** `optimize_balance_c_raw` is an internal helper called from `joint_fpca` to optimize the balance parameter `c`. It calls `SVD::new(combined.to_dmatrix(), ...)` inside a closure that iterates ≤20 times (golden-section search). If a bench cell exercises `joint_fpca` with `balance_c = None`, the dhat output will show 20 allocations from this helper — which is a legitimate cost of `joint_fpca` but a separate optimization concern from the main SVD path.

**How to avoid:** When benching `joint_fpca`, use `balance_c = Some(1.0)` (bypass optimizer) for the "main SVD path" cell, and `balance_c = None` (trigger optimizer) for a separate "with optimizer" cell. This separates the two allocation profiles.

### Pitfall C: Applying Phase 3's INFEASIBILITY pattern to FPCA cells

**What goes wrong:** Phase 3 found N=500,M=200 elastic distance cells infeasible (~384s/iter). The analyst applies the same caution to FPCA and under-sizes the grid. FPCA at N=1000,M=200 is ~64–256 ms/iter — entirely feasible. The O(n²·m) SVD complexity is vastly cheaper than O(N²·m²) elastic alignment.

**How to avoid:** FPCA grid uses the full N∈{100,500,1000}×M∈{50,200} contract from ROADMAP SC1. No cell needs to be declared infeasible; the maximum expected iteration time is ~256 ms (N=1000,M=200) which is well within standard criterion settings.

### Pitfall D: Reporting Phase 4 FPCA results as comparable to Phase 3 elastic numbers

**What goes wrong:** FPCA at N=500,M=200 takes ~16 ms; elastic_self_distance_matrix at N=500,M=50 takes ~24 s — a 1500× difference. If the report presents these side-by-side without context, it can mislead about relative bottleneck severity.

**How to avoid:** The Phase 4 AUDIT-REPORT section must explicitly note that FPCA is NOT the primary bottleneck at these sizes — elastic alignment (Phase 3) is 2–3 orders of magnitude more expensive. This context is important for Phase 9 backlog prioritization.

---

## Runtime State Inventory

SKIPPED — analysis-only phase. No fdars-core source changes, no renaming, no migration. All work is read-only from `fdars-core/src/` and write-only to markdown report + bench artifacts.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust toolchain 1.97.0 | All benchmark compilation | Yes | 1.97.0 | — |
| `linalg` feature (faer, Rust ≥1.84) | `--features linalg,parallel` bench runs | Yes (1.97.0 > 1.84) | — | — |
| TMPDIR workaround | Bench linking (Pitfall 8) | Yes | `/home/simonm/.cache/fdars-bench-tmp` | None — required |
| `dhat` crate | Allocation profiling | Not yet in Cargo.toml | Need to add (~0.3.x) | heaptrack (not installed) |
| `valgrind` | iai-callgrind optional cross-check | Not installed | — | Use criterion only for Phase 4 |
| `heaptrack` | Alternative allocator profiler | Not installed | — | Use dhat-rs instead |

**Missing dependencies with no fallback:**
- `dhat` dev-dependency — must be added to `fdars-core/Cargo.toml` before running the allocation tests.

**Missing dependencies with fallback:**
- `valgrind` / `iai-callgrind` — not needed for Phase 4; criterion is sufficient for wall-clock; dhat-rs works without Valgrind.
- `heaptrack` — not needed; dhat-rs is the primary tool.

---

## Validation Architecture

> `workflow.nyquist_validation = true` in `.planning/config.json` [VERIFIED: .planning/config.json:24]

### Test Framework

| Property | Value |
|----------|-------|
| Framework | criterion 0.5 (wall-clock); dhat `testing()` mode (allocation counts) |
| Config file | `fdars-core/Cargo.toml` — `[[bench]] name = "audit_hotpaths"` + new `dhat-heap` feature |
| Quick run command | `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo bench -p fdars-core --features linalg,parallel --bench audit_hotpaths -- audit_p4_fpca` |
| Full suite command | Same as above plus dhat test: `cargo test -p fdars-core --features dhat-heap,linalg -- count_fpca_allocations_n500_m200 --nocapture` |

### Phase Requirements → Verification Map

| SC # | Behavior | Verification Type | Automated Command |
|------|----------|-----------|-------------------|
| SC1 | Criterion results table for `fdata_to_pc_1d` at N∈{100,500,1000}×M∈{50,200}, release + `linalg,parallel` + `black_box` | Manual grep | `grep -c "n.*_m.*" .planning/research/AUDIT-REPORT.md` (expect ≥6 Phase-4 rows) |
| SC2 | dhat allocation audit quantifies `FdMatrix→DMatrix` overhead, bytes/allocations per call, reproducible baseline | Manual check | `ls .planning/research/bench/p4_dhat_*.txt` (file must exist with `total_blocks` value) |
| SC3 | Report states allocation cost as % of wall-clock, SVD-compute vs copy split explicit | Manual review | `grep -c "copy.*%\|allocation.*share\|SVD.*dominates\|copy.*dominates" .planning/research/AUDIT-REPORT.md` |
| SC4 | Backlog entries with function/current-cost/root-cause drafted | Manual review | `grep -c "Current cost\|Root cause\|Candidate fix" .planning/research/AUDIT-REPORT.md` (expect ≥6 for 2+ entries × 3 fields) |

### Wave 0 Gaps

- [ ] `fdars-core/Cargo.toml` — add `dhat = "0.3"` to `[dev-dependencies]` and `dhat-heap = []` to `[features]`
- [ ] `fdars-core/tests/alloc_audit_fpca.rs` — create new integration test file with dhat harness and `#[global_allocator]`

*(criterion bench extension and AUDIT-REPORT append are Wave 1+ tasks — not Wave 0 gaps)*

---

## Security Domain

> `security_enforcement: true` in config.json [VERIFIED: .planning/config.json:47]. This phase writes only to local markdown files and bench artifacts. No network calls, no user input, no secrets, no production code changes.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | Local bench-only phase |
| V3 Session Management | No | Local bench-only phase |
| V4 Access Control | No | Local bench-only phase |
| V5 Input Validation | No | Synthetic generated inputs, no user input |
| V6 Cryptography | No | No cryptographic operations |

**Security finding:** None. Phase 4 is read-only measurement of local source code + write-only to local markdown and artifact files. The `dhat-heap` feature flag must never be activated in release/production builds (gated behind `#[cfg(feature = "dhat-heap")]` throughout).

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `dhat` version `0.3.3` is current and compatible | Section 4A | Low — dhat API stable; use `cargo search dhat` to verify before adding to Cargo.toml |
| A2 | `optimize_balance_c_raw` is called ≤20 times per `joint_fpca` invocation (golden-section search bound) | Section 4C, Pitfall B | If loop bound > 20, dhat allocation count will be higher than estimated; still a valid measurement, just more allocations |
| A3 | FPCA at N=1000,M=200 takes ~64–256 ms/iter (O(n²) scaling from Phase 1 baseline) | Section 2A | Low — Phase 1 measured ~16 ms at N=500,M=200; scaling by (1000/500)²=4 gives ~64 ms; may be lower if SVD uses sub-quadratic path for n<<m |
| A4 | `to_dmatrix()` copy cost is ~26 µs at N=500,M=200 (memory bandwidth calculation) | Section 5B | Medium — theory-based estimate; measure confirms or refutes; does not affect which measurements to run, only the pre-analysis prediction |
| A5 | `horiz_fpca` does NOT use `to_dmatrix()` as its primary SVD site — it uses `build_symmetric_covariance` directly | Sections 1B, 2B | Low — verified by reading elastic_fpca.rs:182-270 in this session; if wrong, horiz_fpca would also be a `to_dmatrix()` copy site |

---

## Open Questions

1. **Does `vert_fpca` (not `vert_fpca_from_alignment`) call `to_dmatrix()` before its SVD?**
   - What we know: `vert_fpca` (line 89) calls `shooting_vectors_from_psis` then `center_matrix` then `SVD::new(centered.to_dmatrix(), ...)` at line 214.
   - What's unclear: `vert_fpca` takes a `&KarcherMeanResult` directly — its benchmark setup needs a `KarcherMeanResult` object. The bench cell must pre-compute the Karcher result.
   - Recommendation: Planner should set up bench cell for `vert_fpca` using `karcher_mean` result at N=100,M=50 (keeping setup fast). The measurement target is `vert_fpca` only; `karcher_mean` cost is setup, not measured.

2. **What does `horiz_fpca` call instead of `to_dmatrix()` for its SVD?**
   - What we know: From grep result in this session, `elastic_fpca.rs:122` contains `SVD::new(k_mat, true, true)` where `k_mat` is built by `build_symmetric_covariance` (returns `nalgebra::DMatrix<f64>` directly). This is NOT a `to_dmatrix()` copy.
   - What's unclear: Whether the m×m covariance matrix construction itself has significant allocation cost at M=200 (200×200×8 = 320 KB).
   - Recommendation: Include `horiz_fpca` in the dhat audit to compare its allocation profile against `vert_fpca` (which does use `to_dmatrix()`). The difference in allocation profiles directly illustrates the copy-vs-no-copy contrast.

---

## Sources

### Primary (HIGH confidence — read directly in this session)

- `fdars-core/src/regression.rs:167-322` — `center_columns` (sequential, line 167), `centered.clone()` (line 291), `SVD::new(weighted.to_dmatrix(), ...)` (line 298), full `fdata_to_pc_1d` function [VERIFIED]
- `fdars-core/src/matrix.rs:310-312` — `to_dmatrix()` definition: `DMatrix::from_column_slice(self.nrows, self.ncols, &self.data)` — plain column-major memcpy [VERIFIED]
- `fdars-core/src/elastic_fpca.rs:89-930` — all five `to_dmatrix()` SVD sites at lines 214, 317, 483, 584, 930; `build_symmetric_covariance` at line 792; covariance-SVD sites at lines 122, 399 (NOT `to_dmatrix()`) [VERIFIED]
- `fdars-core/src/spm/mfpca.rs:336` — `SVD::new(stacked.to_dmatrix(), ...)` [VERIFIED]
- `fdars-core/src/alignment/nd.rs:705` — `SVD::new(gram.to_dmatrix(), ...)` [VERIFIED]
- `fdars-core/benches/audit_hotpaths.rs:1-753` — full audit bench file; `bench_fpca_sentinel` (lines 76-96), `generate_curves` (lines 38-52), `criterion_group!` (lines 736-751) [VERIFIED]
- `fdars-core/Cargo.toml:49-92` — dev-dependencies (no dhat yet), existing `[[bench]]` entries [VERIFIED]
- `.planning/research/AUDIT-REPORT.md:1-250` — Phase 1 sentinel baseline (~16 ms at N=500,M=200), methodology (append-only D-05, artifact naming D-06), workload matrix (FPCA/SVD row: 9 cells, no cap) [VERIFIED]
- `.planning/research/PITFALLS.md:93-113` — Pitfall 5 definition (allocation vs CPU cost) [VERIFIED]
- `.planning/research/STACK.md:80-90, 228-250` — dhat wiring pattern (Cargo.toml feature flag, testing() mode, integration test harness) [VERIFIED]
- `.planning/ROADMAP.md:93-135` — Phase 4 success criteria (SC1-SC4), Phase 6 go/no-go trigger (SC1) [VERIFIED]
- `.planning/phases/03-elastic-alignment-hot-path/03-CONTEXT.md` — Phase 3 bench conventions (D-01 through D-07), `band_frac`, karcher params [VERIFIED]
- `.planning/phases/03-elastic-alignment-hot-path/03-01-SUMMARY.md` — Phase 3 tracer pipeline (bench fn → artifact → report row → backlog stub) [VERIFIED]
- `.planning/phases/03-elastic-alignment-hot-path/03-02-SUMMARY.md` — Phase 3 full grid conventions (artifact naming, timing tuning, backlog fields) [VERIFIED]
- `.planning/phases/02-static-hot-path-analysis/02-RESEARCH.md:76-131` — Pitfall 1 (confusing `center_1d` with `center_columns`), Pitfall 2 (test-only `to_dmatrix()`), allocation chain for `fdata_to_pc_1d` [VERIFIED]
- `.planning/research/AUDIT-REPORT.md §Phase 3` — backlog entry field shape (D-07), results table schema [VERIFIED]
- `STATE.md` — decision: `elastic_fpca.rs:930` enclosing fn is `optimize_balance_c_raw`, called ≤20× per golden-section search [VERIFIED]
- `.planning/config.json:24,47` — `nyquist_validation: true`, `security_enforcement: true` [VERIFIED]

### Tertiary (LOW confidence — training knowledge / estimates)

- dhat version `0.3.3` current on crates.io — [ASSUMED: A1]
- Memory bandwidth estimate (~30 GB/s) for `to_dmatrix()` copy time calculation — [ASSUMED: A4]
- O(n²·m) SVD scaling at N=1000 extrapolated from Phase 1 N=500 baseline — [ASSUMED: A3]

---

## Metadata

**Confidence breakdown:**
- SVD call site inventory: HIGH — all sites read directly from source in this session
- Bench harness conventions: HIGH — read from audit_hotpaths.rs and Phase 3 summaries
- dhat wiring pattern: MEDIUM — read from STACK.md (itself rated MEDIUM confidence on versions)
- Wall-clock-share estimate: LOW — theoretical calculation; confirmed numbers come from Phase 4 execution

**Research date:** 2026-08-08
**Valid until:** Stable — fdars-core v0.14.0 source is the audit target; recheck only if source files change before Phase 4 executes.
