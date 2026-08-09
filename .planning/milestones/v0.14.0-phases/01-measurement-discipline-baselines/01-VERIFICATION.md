---
phase: 01-measurement-discipline-baselines
verified: 2026-08-07T19:28:02Z
status: passed
score: 4/4 must-haves verified
behavior_unverified: 0
overrides_applied: 0
re_verification: false
---

# Phase 1: Measurement Discipline & Baselines — Verification Report

**Phase Goal:** Lock in build-mode/feature-flag guardrails, define the N×M workload matrix, and record baseline benchmark numbers.
**Verified:** 2026-08-07T19:28:02Z
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths (ROADMAP Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| SC1 | A representative workload matrix (N×M input sizes per hot-path module: elastic alignment, FPCA/SVD, depth & distance, CV loops, streaming depth, smoothing) exists in the report, with realistic sizes (N∈{100,500,1000}, M∈{50,200,500}) justified against Pitfall 4 | ✓ VERIFIED | `AUDIT-REPORT.md §Workload Matrix` contains a 6-row table with all 6 modules, N and M cells, cap rationale citing complexity sources from CONCERNS.md. Elastic cap cites O(n²·m²); CV cap cites K×FPCA O(m³)+fit+predict. Candidate sizes explicitly stated as N∈{100,500,1000}×M∈{50,200,500}. |
| SC2 | A benchmark methodology section documents the mandatory `--release` build check, the feature-flag matrix (`""`, `parallel`, `linalg`, `linalg,parallel`), `black_box` requirement, rustc version capture, and ±5% two-run variance threshold | ✓ VERIFIED | `AUDIT-REPORT.md §Methodology` contains all 6 items: release-mode discipline with `/release/deps/` confirmation check; 4-combo feature-flag matrix table with exact command flags; `black_box` requirement with code examples and warning sign; toolchain capture (rustc 1.97.0; 1.84.0 linalg floor); ±5% two-run variance rule with >10% = LOW CONFIDENCE; artifact naming convention. |
| SC3 | A baseline benchmark run for at least one target per hot-path module is recorded (release + `linalg,parallel`), with binary-path `/release/` confirmed and results saved under `.planning/research/bench/` | ✓ VERIFIED | 12 baseline artifacts exist (6 modules × 2 independent runs). All 6 `_run1.txt` files contain `Running benches/audit_hotpaths.rs (target/release/deps/audit_hotpaths-aea52eeb0c35d5bd)`. All contain criterion `time:` measurement lines. Real numbers recorded: FPCA 16.207 ms, Elastic 789.80 ms, Depth 474.18 µs, CV 947.99 µs, Streaming 491.23 µs, Smoothing 125.80 µs. |
| SC4 | The methodology section explicitly documents the criterion/doctest linker-flakiness issue and the "infrastructure failure vs code failure" triage rule | ✓ VERIFIED | `AUDIT-REPORT.md §Methodology` contains the verbatim triage rule with the literal phrase "infrastructure failure", a decision tree, and the known environment cause (/tmp tmpfs at 94%) with mitigation. |

**Score:** 4/4 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `fdars-core/benches/audit_hotpaths.rs` | Bench file with seeded generator + 7 sentinels + criterion macros | ✓ VERIFIED | File exists, 271 lines. Contains `generate_curves`, `make_class_labels`, `generate_smoothing_data`, and all 7 sentinel functions. `criterion_group!` registers all 7. `criterion_main!` present. |
| `fdars-core/Cargo.toml` `[[bench]]` entry | `name = "audit_hotpaths"` with `harness = false` | ✓ VERIFIED | Entry confirmed: `[[bench]] / name = "audit_hotpaths" / harness = false`. |
| `.planning/research/bench/` | Directory with raw criterion stdout artifacts | ✓ VERIFIED | Directory exists with 17 files total, including 12 module baseline files + 4 karcher 4-combo files + 1 env_info file. |
| `.planning/research/bench/p1_fpca_linalg,parallel_run{1,2}.txt` | FPCA baseline, 2 runs | ✓ VERIFIED | Both files exist. run1: `time: [16.114 ms 16.155 ms 16.207 ms]`, `/release/` confirmed. run2: has rustc env header + timing data (16.454 ms). |
| `.planning/research/bench/p1_elastic_linalg,parallel_run{1,2}.txt` | Elastic baseline, 2 runs | ✓ VERIFIED | Both files exist with release path and `time:` lines. |
| `.planning/research/bench/p1_depth_linalg,parallel_run{1,2}.txt` | Depth baseline, 2 runs | ✓ VERIFIED | Both files exist with release path and `time:` lines. |
| `.planning/research/bench/p1_cv_linalg,parallel_run{1,2}.txt` | CV baseline, 2 runs | ✓ VERIFIED | Both files exist with release path and `time:` lines. |
| `.planning/research/bench/p1_streaming_linalg,parallel_run{1,2}.txt` | Streaming baseline, 2 runs | ✓ VERIFIED | Both files exist. Tagged `CONFIDENCE: LOW` (11.1% variance). Release path confirmed. |
| `.planning/research/bench/p1_smooth_linalg,parallel_run{1,2}.txt` | Smoothing baseline, 2 runs | ✓ VERIFIED | Both files exist with release path and `time:` lines. |
| `.planning/research/bench/p1_karcher_*_run1.txt` | 4-combo karcher artifacts | ✓ VERIFIED | All 4 files present: `none`, `parallel`, `linalg`, `linalg,parallel`. Confirms 10× speedup (parallel vs sequential). |
| `.planning/research/AUDIT-REPORT.md` | Report with §Methodology + §Workload Matrix | ✓ VERIFIED | File exists with all required sections. |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `audit_hotpaths.rs` | `fdars_core::regression::fdata_to_pc_1d` | direct call in `bench_fpca_sentinel` | ✓ WIRED | Import and call confirmed at lines 25, 88. |
| `audit_hotpaths.rs` | `fdars_core::alignment::karcher_mean` | direct call in `bench_matrix_sentinel` | ✓ WIRED | Import and call confirmed at lines 21, 114. |
| `audit_hotpaths.rs` | `fdars_core::alignment::elastic_self_distance_matrix` | direct call in `bench_elastic_sentinel` | ✓ WIRED | Import at line 21; call at line 143. |
| `audit_hotpaths.rs` | `fdars_core::depth::fraiman_muniz_1d` | direct call in `bench_depth_sentinel` | ✓ WIRED | Import at line 23; call at line 168. |
| `audit_hotpaths.rs` | `fdars_core::classification::fclassif_cv` | direct call in `bench_cv_sentinel` | ✓ WIRED | Import at line 22; call at lines 191–200. |
| `audit_hotpaths.rs` | `fdars_core::streaming_depth::{SortedReferenceState, StreamingFraimanMuniz}` | direct call in `bench_streaming_sentinel` | ✓ WIRED | Import at line 27; calls at lines 222–224. |
| `audit_hotpaths.rs` | `fdars_core::smoothing::nadaraya_watson` | direct call in `bench_smooth_sentinel` | ✓ WIRED | Import at line 26; call at lines 248–253. |
| `cargo bench` invocation | `target/release/deps/` binary | confirmed via artifact content | ✓ WIRED | All run1 files contain `target/release/deps/audit_hotpaths-aea52eeb0c35d5bd`. |
| `AUDIT-REPORT.md §Workload Matrix` | `.planning/research/bench/` artifacts | per-module artifact links in baseline table | ✓ WIRED | Report table for each module links to both run1 and run2 artifact paths. |

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Bench compiles under leanest combo (`--no-default-features`) — proves no `linalg`-gated call | `cargo bench -p fdars-core --bench audit_hotpaths --no-default-features -- --test` | All 7 sentinels: `Success` | ✓ PASS |
| Bench compiles under full combo (`--features linalg,parallel`) | `cargo bench -p fdars-core --bench audit_hotpaths --features linalg,parallel -- --test` | All 7 sentinels: `Success` | ✓ PASS |
| Release binary path confirmed in FPCA run1 artifact | `grep "release/deps" p1_fpca_linalg,parallel_run1.txt` | `Running benches/audit_hotpaths.rs (target/release/deps/audit_hotpaths-aea52eeb0c35d5bd)` | ✓ PASS |
| 12 baseline artifacts present with criterion `time:` lines | `ls bench/p1_{fpca,elastic,depth,cv,streaming,smooth}_linalg,parallel_run{1,2}.txt \| wc -l` | 12 | ✓ PASS |
| Streaming LOW CONFIDENCE tagged in artifacts | `grep "CONFIDENCE.*LOW" streaming_run{1,2}.txt` | `CONFIDENCE: LOW (>10% two-run variance...)` in both | ✓ PASS |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| PERF-02 | 01-01-PLAN.md, 01-02-PLAN.md | A representative workload matrix (N × M input sizes) is defined per hot-path module so benchmarks reflect realistic usage | ✓ SATISFIED | REQUIREMENTS.md marks PERF-02 `[x]` Complete. `AUDIT-REPORT.md §Workload Matrix` contains the full 6-module table with justified caps. Traceability table maps PERF-02 to Phase 1 with status Complete. |

No other requirement IDs are declared in either plan's frontmatter. No orphaned Phase 1 requirements found in REQUIREMENTS.md — all other requirements are mapped to Phases 2–9.

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| — | — | — | — | No anti-patterns found. No TBD/FIXME/XXX/TODO/HACK markers in any modified file. |

---

### Notable Observations (non-blocking)

**fpca run1 missing rustc header:** `p1_fpca_linalg,parallel_run1.txt` was captured in Plan 01 before the env-info convention was established in Plan 02. It lacks the `=== ENVIRONMENT ===` header with `rustc` version. However: (a) `p1_fpca_linalg,parallel_run2.txt` does have the rustc header; (b) `AUDIT-REPORT.md §Methodology` documents the toolchain version (rustc 1.97.0); (c) the release binary path is confirmed in run1. SC2 requires "rustc version capture" which is satisfied by run2 and the report. Not a blocker.

**Streaming LOW CONFIDENCE (11.1% variance):** Correctly identified, tagged in both artifacts and the report baseline table. The plan and report document this as OS/scheduler jitter (sub-ms scale), not algorithm instability, and recommend re-measurement under `taskset` in later phases. Proper application of the ±5% variance rule.

**7 sentinels vs 6 modules:** The bench registers 7 sentinel functions (FPCA + karcher D-04 discriminator + 5 module sentinels). The D-04 karcher sentinel is supplementary to the FPCA D-03 module baseline — both were required. The plan explicitly accounts for this. Correct.

---

### Human Verification Required

None. All must-haves are verifiable programmatically for this audit-only phase. The deliverables are files (bench code, raw artifacts, report sections) fully verifiable by existence, content, and compilation checks.

---

## Gaps Summary

No gaps. All 4 ROADMAP success criteria are verified in the codebase with direct evidence:

- SC1: `AUDIT-REPORT.md §Workload Matrix` — 6 modules, N/M cells, justified caps.
- SC2: `AUDIT-REPORT.md §Methodology` — all 6 required discipline items documented.
- SC3: 12 release baseline artifacts with `/release/deps/` confirmed and criterion `time:` lines.
- SC4: Literal "infrastructure failure" phrase and triage decision tree in §Methodology.

PERF-02 requirement is checked off in REQUIREMENTS.md with traceability to Phase 1 Complete.

The bench harness compiles and smoke-runs under both the leanest (`""`) and full (`linalg,parallel`) feature combos, confirming no `linalg`-gated API was accidentally called unconditionally (Pitfall 18 guard).

---

_Verified: 2026-08-07T19:28:02Z_
_Verifier: Claude (gsd-verifier)_
