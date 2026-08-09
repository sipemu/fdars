# Phase 1: Measurement Discipline & Baselines - Context

**Gathered:** 2026-08-07
**Status:** Ready for planning

<domain>
## Phase Boundary

Establish the measurement guardrails and workload definitions that make every downstream benchmark (Phases 3–6) valid and reproducible. This phase delivers **analysis artifacts only** — no code changes to `fdars-core` algorithms:

1. A **benchmark methodology section** documenting build-mode (`--release`) discipline, the feature-flag matrix (`""` / `parallel` / `linalg` / `linalg,parallel`), `black_box` requirement, rustc version capture, ±5% two-run variance threshold (>10% = LOW CONFIDENCE), and the criterion/doctest linker-flakiness "infra failure vs code failure" triage rule.
2. A **representative workload matrix** (N × M input sizes per hot-path module) justified against realistic usage (Pitfall 4).
3. **Baseline criterion runs** for at least one target per hot-path module, release + `linalg,parallel`, with `/release/` binary path confirmed and raw output saved under `.planning/research/bench/`.

New criterion bench code (a dedicated audit bench file) IS in scope — it is measurement infrastructure, not a change to the library's algorithms.

**Not in scope:** benchmarking the full N×M sweep of every module (Phases 3–6 own the deep sweeps), fixing any bottleneck found, or touching `fdars-core/src`.

</domain>

<decisions>
## Implementation Decisions

### Benchmark Harness Strategy
- **D-01:** Author a **new dedicated audit bench file** (e.g. `fdars-core/benches/audit_hotpaths.rs`, `harness = false`) at the workload-matrix sizes. Leave the existing 9 criterion bench files untouched so they stay CI-fast. Rationale: the existing benches use small CI inputs that hide O(n²)/O(m³) scaling (Pitfall 4); a separate file keeps audit measurement explicit, feature-tagged, and reproducible without polluting CI concerns. — **Reversibility:** reversible — new bench file, no impact on shipped library.
- **D-02:** The new audit bench(es) must wrap both inputs and outputs in `criterion::black_box` (Pitfall 3), and register via a `[[bench]]` entry with `harness = false` matching the existing convention.

### Baseline Breadth (this phase)
- **D-03:** Baseline = **one representative sentinel function per hot-path module** (6 modules: elastic alignment, FPCA/SVD, depth & distance, CV loops, streaming depth, smoothing) run at `release + linalg,parallel`, 2 independent `cargo bench` invocations each for the ±5% variance check. This satisfies SC3 ("at least one target per module") minimally and keeps Phase 1 lean — deep per-size sweeps belong to Phases 3–6.
- **D-04:** Additionally run **one sentinel target across all 4 feature combos** (`""`, `parallel`, `linalg`, `linalg,parallel`) to validate the feature-flag matrix methodology end-to-end (Pitfall 18). Pick a target that exercises both `linalg` and `parallel` paths — FPCA/SVD is the recommended sentinel. This proves the whole measurement apparatus works without over-investing before Phases 3–6.
- (Sentinel-function selection per module is left to the planner/researcher — see Claude's Discretion.)

### Report & Artifact Structure
- **D-05:** Maintain a **single growing report** at `.planning/research/AUDIT-REPORT.md`. Phase 1 writes the **methodology section** and the **workload matrix** into it; each later phase appends its own section; Phase 9 (RPT-01) finalizes/consolidates rather than assembling from scratch. — **Reversibility:** costly — later phases and Phase 9's consolidation step depend on this single-file append convention; switching to per-phase fragment files mid-milestone would require re-threading every phase's output.
- **D-06:** Save raw criterion output under `.planning/research/bench/` using a documented naming convention keyed by phase, target, feature set, and run number (e.g. `p1_<target>_<features>_run<N>.txt`). Every finding/backlog item links to its artifact (Pitfall 17). Create the `bench/` directory in this phase.

### Workload Matrix Sizing
- **D-07:** Use **per-module tailored subsets**, not a uniform grid. Candidate sizes are N∈{100,500,1000} × M∈{50,200,500}. Apply the full grid where feasible (e.g. FPCA/SVD), but **cap expensive modules with a documented per-module justification** — notably elastic alignment (O(n²·m²); CONCERNS.md notes n=1000×m=500 ≈ 60s) capped at N≤500, M≤200. Each module's chosen cells and the reason for any cap must be written into the workload-matrix table. — **Reversibility:** costly — the workload matrix is the shared size contract that Phases 3–6 benchmark against; changing sizes later invalidates cross-phase comparability.

### Claude's Discretion
- **Sentinel-function selection:** which specific public function represents each hot-path module for the baseline (e.g. `karcher_mean` or an elastic distance-matrix fn for elastic; `fdata_to_pc_1d` for FPCA/SVD). Planner/researcher picks the most representative, respecting that FPCA/SVD is the 4-combo sentinel (D-04).
- **Machine-state / reproducibility controls:** how much to mandate beyond Pitfall 7's baseline (close non-essential processes, 2 runs within ±5%). Optional `cpupower`/frequency-scaling notes are at the planner's discretion — document whatever controls are actually applied.
- **Criterion sample-size / measurement-time config** for the large-input audit benches (large N cells are slow; tuning `sample_size`/`measurement_time` to keep runs tractable is a planner call, documented in methodology).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase definition & requirements
- `.planning/ROADMAP.md` §"Phase 1: Measurement Discipline & Baselines" — the 4 success criteria this phase must satisfy (workload matrix, methodology section, baseline run, infra-vs-code triage).
- `.planning/REQUIREMENTS.md` — PERF-02 (this phase's requirement) and PERF-03 (the downstream benchmark requirement this phase's discipline governs).

### Measurement discipline (MANDATORY — governs the methodology section)
- `.planning/research/PITFALLS.md` — the authoritative source for this phase. Directly relevant: Pitfall 1 (debug-mode benchmarking / confirm `/release/` binary path), Pitfall 2 & 18 (feature-flag matrix), Pitfall 3 (`black_box`), Pitfall 4 (unrepresentative input sizes → workload matrix), Pitfall 5 (allocation vs CPU cost — flag for Phase 4), Pitfall 6 (warm/cold cache column), Pitfall 7 (noisy machine / ±5% variance / 2-run rule), Pitfall 8 (linker/toolchain bus-error = infra failure, not code failure), Pitfall 17 (reproducible evidence under `.planning/research/bench/`). See also the "Integration Gotchas" and "Performance Traps" tables.

### fdars-side inputs (workload matrix justification)
- `.planning/codebase/CONCERNS.md` §"Performance Bottlenecks", §"Scaling Limits" — algorithmic complexity and empirical cost notes (elastic O(n²·m²) ≈ 60s at n=1000×m=500; SVD O(m³); parallel overhead below n≈100) that justify per-module size caps (D-07).
- `.planning/codebase/TESTING.md` §"Criterion benchmarks" — the existing 9 bench files, `harness = false` convention, and run commands the new audit bench must mirror.
- `.planning/codebase/ARCHITECTURE.md` — the `FdMatrix ↔ nalgebra::DMatrix` round-trip anti-pattern (relevant to the FPCA/SVD sentinel and the Phase 4 allocation audit this phase seeds).

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **Existing 9 criterion benches** (`fdars-core/benches/{alignment,basis,classification,depth,explain,matrix,regression,seasonal,smoothing}_benchmarks.rs`): scaffolding to copy for the new `audit_hotpaths.rs` (criterion group setup, `harness = false` registration). NOT reused directly — inputs are too small (Pitfall 4).
- **Deterministic test-data generators** (e.g. `generate_regression_data(n, m, seed)` in `tests/`, `uniform_grid(n)` in `src/test_helpers.rs`): patterns for building reproducible N×M synthetic inputs at workload sizes. `test_helpers.rs` is test-only, so audit benches will likely need their own seeded generator.
- **Criterion 0.5** already a dev-dependency — no new tooling to add for wall-clock benches.

### Established Patterns
- `[[bench]]` entries with `harness = false` in `fdars-core/Cargo.toml` — the new audit bench must add a matching entry.
- Feature gates: `linalg` (requires Rust 1.84+, gates faer ridge/Cholesky), `parallel` (gates the 5 `iter_maybe_parallel!` macros in `src/parallel.rs`). The feature matrix runs must pass these explicitly; the same source compiles differently per combo (Pitfall 18).
- Column-major `FdMatrix` (`src/matrix.rs`) — synthetic inputs must be built via `FdMatrix::from_column_major` respecting the layout.

### Integration Points
- New `fdars-core/benches/audit_hotpaths.rs` + a `[[bench]]` entry in `fdars-core/Cargo.toml`.
- New artifact dir `.planning/research/bench/` (raw criterion output).
- New/seed `.planning/research/AUDIT-REPORT.md` (methodology + workload matrix sections).

</code_context>

<specifics>
## Specific Ideas

- Bench artifact naming convention: `p1_<target>_<features>_run<N>.txt` under `.planning/research/bench/` (user-approved preview).
- FPCA/SVD is the recommended 4-combo feature-matrix sentinel because it exercises both `linalg` and `parallel` code paths.
- Elastic alignment size cap example: N≤500, M≤200 (from O(n²·m²) cost in CONCERNS.md).

</specifics>

<deferred>
## Deferred Ideas

- **Allocation profiling (dhat) of the FdMatrix→DMatrix SVD copy** — Phase 4 (PERF-04). Phase 1 only records wall-clock baselines; the CPU-vs-allocation split (Pitfall 5) is Phase 4's job. Flag the FPCA/SVD sentinel as an allocation-audit candidate but do not measure allocations here.
- **Full N×M sweeps per hot path** — Phases 3 (elastic) and 4 (FPCA/SVD) own the deep, per-size criterion tables (PERF-03). Phase 1 records only one sentinel per module.
- **nalgebra-vs-faer SVD comparison** — Phase 6 (PERF-06), conditional on Phase 4 evidence.
- **RAYON_NUM_THREADS thread-scaling sweep** — Phase 5 (PERF-05). Phase 1's feature-combo run is not a thread sweep.

None of these are scope creep — they are downstream phases already in the roadmap; noted here only to keep Phase 1's baseline lean.

</deferred>

---

*Phase: 1-Measurement Discipline & Baselines*
*Context gathered: 2026-08-07*
