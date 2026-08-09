---
phase: 04-fpca-svd-allocation-audit
verified: 2026-08-08T10:16:14Z
status: passed
score: 4/4 must-haves verified
behavior_unverified: 0
overrides_applied: 0
---

# Phase 4: FPCA/SVD & Allocation Audit Verification Report

**Phase Goal:** Separate SVD compute cost from copy/allocation overhead in FPCA and produce its report + backlog slice.
**Verified:** 2026-08-08T10:16:14Z
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths (ROADMAP Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| SC1 | A criterion results table for `fdata_to_pc_1d` (and elastic-FPCA SVD sites) at N∈{100,500,1000}×M∈{50,200}, release + linalg,parallel + black_box, tagged with feature set and toolchain, exists in the report | ✓ VERIFIED | AUDIT-REPORT.md lines 483–492: 6 FPCA rows (n100_m50, n100_m200, n500_m50, n500_m200, n1000_m50, n1000_m200) + 2 elastic rows (vert_fpca, joint_fpca at 100×50), all with `linalg,parallel` feature tag and `rustc 1.97.0` toolchain. All 6 FPCA cells ran twice with run artifacts saved. `/release/` binary path confirmed in each artifact. `black_box` wraps all inputs in `bench_p4_fpca` (verified in source lines 770–774). |
| SC2 | A dhat allocation audit quantifies the FdMatrix→DMatrix SVD-copy overhead (bytes/allocations per FPCA call) and ranks other allocation hotspots, with a reproducible baseline saved under `.planning/research/bench/` | ✓ VERIFIED | Three dhat baselines on disk: `p4_dhat_fpca_n500_m200.txt` (21 blocks / 3,574,424 bytes — corrected per CR-02), `p4_dhat_vert_fpca_n100_m50.txt` (45 blocks / 285,816 bytes), `p4_dhat_joint_fpca_n100_m50.txt` (1,544 blocks / 1,604,792 bytes). All feature-tagged `dhat-heap,linalg` with toolchain. AUDIT-REPORT ranks hotspots normalized to bytes/n·m: joint_fpca 320.96 > vert_fpca 57.16 > fdata_to_pc_1d 35.74. Three `regression.rs` allocation sites (`:167`, `:291`, `:298`) cited with per-site byte sizes. |
| SC3 | The report states allocation cost as a share of wall-clock for the top FPCA path, so the SVD-compute vs copy split is explicit and directly informs the Phase 6 go/no-go trigger | ✓ VERIFIED | AUDIT-REPORT "SVD-Compute vs Copy Split" section gives full arithmetic chain at N=1000,M=200: 1,600,000 bytes ÷ 30 GB/s = 53.3 µs; ÷ 38,307 µs × 100% = 0.14%. Cross-grid confirmation at N=500,M=200: 800,000 bytes ÷ 30 GB/s = 26.7 µs; ÷ 16,011 µs × 100% = 0.17%. Both figures independently re-derived and verified correct. All three required literals present: measured byte count, 30 GB/s figure, and resulting %. |
| SC4 | This slice's backlog entries (SVD-copy elimination, truncated-SVD candidates) are drafted with function/current-cost/root-cause fields | ✓ VERIFIED | AUDIT-REPORT "Draft Backlog" section has two entries: (1) eliminate `centered.clone()` at `regression.rs:291` + zero-copy `to_dmatrix()` bridge; (2) truncated-SVD candidate replacing `nalgebra::SVD::new`. Both entries carry Function, Current cost (measured grid numbers + dhat figures), Root cause (cite regression.rs line numbers), Candidate fix, Evidence (artifact links), and Severity+Effort `[TBD — Phase 9]`. Field count: 13 occurrences of required fields (well above 2×3=6 minimum). |

**Score:** 4/4 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `fdars-core/Cargo.toml` | dhat dev-dep + dhat-heap feature gate | ✓ VERIFIED | `dhat = "0.3"` in `[dev-dependencies]`; `dhat-heap = []` in `[features]`; absent from `default = ["parallel"]`. Comment explicitly warns "NEVER enable in release builds or default CI". |
| `fdars-core/tests/alloc_audit_fpca.rs` | dhat integration harness with 3 gated cells | ✓ VERIFIED | 133-line file with `#[cfg(feature = "dhat-heap")] #[global_allocator]`, `generate_test_curves()`, and three test cells: `count_fpca_allocations_n500_m200`, `count_vert_fpca_allocations_n100_m50`, `count_joint_fpca_allocations_n100_m50`. All dhat symbols gated. Module doc updated (IN-01 fix). `generate_test_curves` is correctly outside profiler scope in primary cell (CR-02 fix). |
| `fdars-core/benches/audit_hotpaths.rs` | `bench_p4_fpca` (6-cell grid) + `bench_p4_elastic_fpca`, both registered in `criterion_group!` | ✓ VERIFIED | `bench_p4_fpca` at lines 760–840 has all 6 cells with correct sample_size tiers. `bench_p4_elastic_fpca` at lines 861–892 benchmarks vert_fpca and joint_fpca (balance_c=Some(1.0), karcher outside b.iter()). Both registered at criterion_group! lines 909–910. |
| `.planning/research/AUDIT-REPORT.md` Phase-4 section | Complete phase section with criterion table, dhat audit, SVD-vs-copy split, go/no-go verdict, backlog | ✓ VERIFIED | Section at lines 477–609. 8-row results table, allocation audit with normalized ranking, full arithmetic chain, Phase-6 GO verdict citing both SVD-share (99.8–99.9%) and copy-share (0.14–0.17%). 2 backlog entries with all required fields. |
| `.planning/research/bench/p4_fpca_linalg,parallel_run1.txt` | 6-cell criterion run1 artifact | ✓ VERIFIED | All 6 cells present (n100_m50 through n1000_m200). Feature-tagged linalg,parallel. `/release/` binary path confirmed. |
| `.planning/research/bench/p4_fpca_linalg,parallel_run2.txt` | 6-cell criterion run2 artifact (variance) | ✓ VERIFIED | All 6 cells. Variance 0.01–0.64% across cells (all OK confidence). |
| `.planning/research/bench/p4_fpca_linalg_run1.txt` | No-parallel invariance run | ✓ VERIFIED | Feature-tagged `linalg (no parallel)`. All 6 cells within ≤1.5% of linalg,parallel timings. |
| `.planning/research/bench/p4_elastic_fpca_vert_linalg,parallel_run1.txt` | vert_fpca 100×50 single-run artifact | ✓ VERIFIED | 300.64 µs mean, linalg,parallel, /release/ binary confirmed. |
| `.planning/research/bench/p4_elastic_fpca_joint_linalg,parallel_run1.txt` | joint_fpca 100×50 single-run artifact | ✓ VERIFIED | 1.8850 ms mean, linalg,parallel, /release/ binary confirmed. |
| `.planning/research/bench/p4_dhat_fpca_n500_m200.txt` | Primary FPCA dhat baseline (corrected) | ✓ VERIFIED | Shows corrected figures: 21 total_blocks / 3,574,424 total_bytes / 3,531,192 peak_bytes (CR-02 fix applied). Correction note embedded in artifact. `generate_test_curves` is called BEFORE profiler scope in source. |
| `.planning/research/bench/p4_dhat_vert_fpca_n100_m50.txt` | vert_fpca dhat baseline | ✓ VERIFIED | 45 blocks / 285,816 bytes / 145,256 peak_bytes. Feature-tagged dhat-heap,linalg. |
| `.planning/research/bench/p4_dhat_joint_fpca_n100_m50.txt` | joint_fpca dhat baseline (optimizer bypassed) | ✓ VERIFIED | 1,544 blocks / 1,604,792 bytes / 504,616 peak_bytes. Feature-tagged dhat-heap,linalg. |
| `.planning/phases/04-fpca-svd-allocation-audit/04-COVERAGE.md` | Benchmark-only stub | ✓ VERIFIED | One-line stub: "No external API integration: benchmark-only phase measuring internal fdars-core FPCA/SVD functions plus dhat allocation profiling." |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `dhat-heap` feature in Cargo.toml | `#[cfg(feature = "dhat-heap")]` symbols in `alloc_audit_fpca.rs` | Empty feature flag gates all dhat symbols | ✓ WIRED | All 7 use-level imports and all 3 test functions in `alloc_audit_fpca.rs` are under `#[cfg(feature = "dhat-heap")]`. `#[global_allocator]` is also gated. Feature absent from `default` array. |
| `bench_p4_fpca` in `audit_hotpaths.rs` | `criterion_group!` macro | Listed at line 909 | ✓ WIRED | Both `bench_p4_fpca` and `bench_p4_elastic_fpca` appear in the `criterion_group!(benches, ...)` call. |
| `p4_dhat_fpca_n500_m200.txt` (corrected 21 blocks / 3,574,424 bytes) | AUDIT-REPORT.md allocation table | Transcribed in Phase-4 section with CR-02 footnote | ✓ WIRED | AUDIT-REPORT line 506 shows corrected values (21 / 3,574,424 / 3,531,192). Footnote ¹ explains the correction. Backlog entry 1 references corrected numbers. |
| Criterion timings + dhat total_bytes | Copy-share % in AUDIT-REPORT | Full arithmetic chain inline | ✓ WIRED | Report derives 0.14% and 0.17% using N×M×8 bytes from the specific to_dmatrix() call (not total dhat bytes), 30 GB/s bandwidth, and measured wall-clock from artifacts. Independent re-derivation confirms both figures correct. |
| SVD-share + copy-share | Phase-6 GO verdict | Both quantities cited before verdict call | ✓ WIRED | AUDIT-REPORT "Phase 6 Go/No-Go Decision" section explicitly names: (1) SVD share ~99.8–99.9%, (2) copy-share ~0.14–0.17%, then issues "Phase 6 GO" verdict. |
| CI `.github/workflows/rust-ci.yml` | `dhat-heap` feature NOT activated | Explicit feature list excludes dhat-heap on test, clippy, coverage steps | ✓ WIRED (CR-01 fix applied) | CI test step uses `cargo test --features linalg,parallel,serde` (no `dhat-heap`). Clippy and coverage steps likewise use explicit lists excluding `dhat-heap`. Comments in CI file explain the parallel-profiler panic risk. |

---

### Data-Flow Trace (Level 4)

Not applicable — this is an audit/measurement phase. The deliverables are documentation artifacts (AUDIT-REPORT.md) and measurement scaffolding code (bench file, dhat harness). Data flows from bench runs and dhat test runs into `.planning/research/bench/` artifact files, which are then transcribed into AUDIT-REPORT.md. All three links verified above.

---

### Behavioral Spot-Checks

Step 7b: Not applicable to this phase as an audit milestone. The "behaviors" that matter are: (a) the dhat harness compiles under the feature gate, (b) the criterion bench cells register correctly, and (c) the bench artifacts contain real timing data. All three are verified by direct inspection of source files and artifact files above. Running the full bench suite would re-execute measurements — outside scope of a read-only verification pass.

---

### Probe Execution

No probes declared in PLAN files. Phase is a local-only audit producing static artifacts. Skipped.

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| PERF-03 | Plans 02, 03 | Criterion benchmarks in release build with correct feature flags (linalg, parallel) and black_box, producing a results table tagged with feature set and toolchain version | ✓ SATISFIED | 8-row results table in AUDIT-REPORT.md with linalg,parallel tag, rustc 1.97.0 toolchain, /release/ binary confirmation, and black_box on all inputs. Two runs per FPCA cell for variance. |
| PERF-04 | Plans 01, 02, 03 | Allocation audit (dhat) quantifies FdMatrix→DMatrix SVD-copy overhead with reproducible baseline | ✓ SATISFIED | Three dhat baselines under `.planning/research/bench/p4_dhat_*.txt`, each with toolchain+feature tags. Allocation audit section in AUDIT-REPORT with per-unit-work normalized ranking and three regression.rs copy sites cited. |

REQUIREMENTS.md traceability: PERF-03 marked Complete for Phase 3 + Phase 4; PERF-04 marked Complete for Phase 4. Both statuses match phase deliverables.

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| AUDIT-REPORT.md | 594 | `[TBD — Phase 9]` in backlog entries | ℹ Info | Intentional — backlog items explicitly deferred to Phase 9 with a formal phase reference. Not an unresolvable TBD; the phase reference makes it auditable. |
| `alloc_audit_fpca.rs` | 29 (removed) | `use dhat;` bare import (WR-01 from 04-REVIEW.md) | ℹ Info | CR/WR-01 was flagged in code review. Module doc updated (IN-01). The bare `use dhat;` is absent from the current file — the fix was applied per REVIEW.md. |

No unresolved `TBD`, `FIXME`, or `XXX` markers found in phase-modified source files (`alloc_audit_fpca.rs`, `audit_hotpaths.rs`). The `[TBD — Phase 9]` markers in AUDIT-REPORT.md reference formal future phases — not unresolvable debt markers per the debt-marker gate rule.

---

### Code Review Status

04-REVIEW.md records `status: resolved` with 2 Critical findings (CR-01, CR-02) and 1 Warning (WR-01):

- **CR-01 (CI --all-features activates dhat-heap):** FIXED. `.github/workflows/rust-ci.yml` test, clippy, and coverage steps all use explicit feature lists excluding `dhat-heap`, with explanatory comments about the parallel-profiler panic risk.
- **CR-02 (generate_test_curves inside profiler scope in primary FPCA cell):** FIXED. In `alloc_audit_fpca.rs` lines 77-79, `generate_test_curves(500, 200)` is called BEFORE `dhat::Profiler::builder().testing().build()`. The corrected baseline (21 blocks / 3,574,424 bytes) is reflected in both the artifact file and in every AUDIT-REPORT.md reference. Correction footnote ¹ in the allocation table explicitly records the prior (contaminated) values and the delta.
- **WR-01 (redundant `use dhat;` import):** Fixed — not present in current file.
- **IN-01 (stale module doc):** Fixed — doc now describes all 3 cells.
- **IN-02 (duplicated generate_test_curves):** Addressed with sync comment.
- **IN-03 (stale timing comment):** Fixed — `bench_p4_fpca` doc comment at line 756 reads "~38 ms/iter (measured)" matching actual 38.307 ms.

---

### Corrected Baseline Consistency Check

The CR-02 correction changes `fdata_to_pc_1d` baseline from 23 blocks/4,376,024 bytes to 21 blocks/3,574,424 bytes.

The Phase-6 GO verdict is derived entirely from criterion wall-clock timings (not dhat byte counts), so it is unaffected by the correction. The AUDIT-REPORT explicitly notes this (line 585: "The GO verdict is unaffected — it is derived entirely from criterion wall-clock timings, not dhat block/byte counts.").

The copy-share arithmetic uses `N×M×8` bytes (the known `to_dmatrix()` copy size), not the total dhat `total_bytes`. The corrected `total_bytes` (3,574,424) is smaller than the prior contaminated value (4,376,024), which means the normalization figure (bytes/n·m) decreased from 43.76 to 35.74 — a more accurate picture of FPCA's true allocation footprint. The GO verdict and copy-share % are internally consistent with the corrected artifact.

---

### Human Verification Required

None — all success criteria are verifiable from static artifacts (source files, bench output files, AUDIT-REPORT.md). No runtime UI, real-time behavior, or external service integration is involved.

---

## Gaps Summary

No gaps. All four ROADMAP success criteria are fully satisfied by on-disk artifacts. The code review findings (CR-01, CR-02, WR-01, IN-01, IN-02, IN-03) are all resolved in the current state of the files. REQUIREMENTS.md traceability for PERF-03 and PERF-04 is complete.

---

_Verified: 2026-08-08T10:16:14Z_
_Verifier: Claude (gsd-verifier)_
