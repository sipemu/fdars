# Phase 1: Measurement Discipline & Baselines - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-08-07
**Phase:** 1-Measurement Discipline & Baselines
**Areas discussed:** Benchmark harness strategy, Baseline breadth, Report/artifact structure, Workload matrix sizing

---

## Benchmark Harness Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| New dedicated audit benches | Author new `benches/audit_*.rs` at N×M workload sizes; leave existing 9 untouched | ✓ |
| Extend existing benches | Add large-input variants into the existing 9 files inline | |
| Hybrid | Audit existing for black_box, reuse where adequate, add new benches only where inputs too small | |

**User's choice:** New dedicated audit benches
**Notes:** Keeps existing benches CI-fast; audit measurement stays explicit, feature-tagged, reproducible without mixing CI concerns. Addresses Pitfall 4 (small CI inputs hide scaling).

---

## Baseline Breadth (this phase)

| Option | Description | Selected |
|--------|-------------|----------|
| One sentinel per module, one combo | 6 targets at linalg,parallel only; deep sweeps in Phases 3–6 | |
| One sentinel per module, all 4 combos | 6 targets × 4 feature combos | |
| You decide | Let planner/researcher pick minimal set satisfying SC3 + Pitfall 18 | ✓ |

**User's choice:** You decide
**Claude's resolution:** One sentinel per hot-path module (6 targets) at `release + linalg,parallel`, 2 runs each for ±5% variance (satisfies SC3 minimally) — PLUS one sentinel (FPCA/SVD) run across all 4 feature combos to validate the feature-flag matrix methodology (Pitfall 18) without over-investing before Phases 3–6.

---

## Report & Artifact Structure

| Option | Description | Selected |
|--------|-------------|----------|
| One growing report + bench/ dir | Single `AUDIT-REPORT.md` appended per phase; raw dumps under `research/bench/` | ✓ |
| Per-phase section files | Each phase writes its own fragment; Phase 9 concatenates | |
| You decide | Let planner choose | |

**User's choice:** One growing report + bench/ dir
**Notes:** Methodology + workload matrix written into `AUDIT-REPORT.md` now; later phases append; Phase 9 finalizes. Bench artifacts named `p1_<target>_<features>_run<N>.txt` under `.planning/research/bench/` (Pitfall 17).

---

## Workload Matrix Sizing

| Option | Description | Selected |
|--------|-------------|----------|
| Per-module tailored subsets | Full grid where feasible; cap expensive modules (elastic N≤500) with documented justification | ✓ |
| Uniform grid for all | Apply {100,500,1000}×{50,200,500} to every module identically | |

**User's choice:** Per-module tailored subsets
**Notes:** Candidate sizes N∈{100,500,1000}×M∈{50,200,500}. Elastic alignment capped (O(n²·m²) ≈ 60s at n=1000×m=500 per CONCERNS.md). Each module's cells + cap reason documented in the workload-matrix table.

---

## Claude's Discretion

- Sentinel-function selection per module (e.g. `karcher_mean` for elastic, `fdata_to_pc_1d` for FPCA/SVD; FPCA/SVD is the 4-combo sentinel).
- Machine-state / reproducibility controls beyond Pitfall 7 baseline (optional `cpupower`/frequency-scaling notes).
- Criterion `sample_size`/`measurement_time` tuning for slow large-input audit benches.

## Deferred Ideas

- dhat allocation profiling of FdMatrix→DMatrix SVD copy → Phase 4 (PERF-04).
- Full per-size N×M sweeps per hot path → Phases 3 & 4 (PERF-03).
- nalgebra-vs-faer SVD comparison → Phase 6 (PERF-06).
- RAYON_NUM_THREADS thread-scaling sweep → Phase 5 (PERF-05).
