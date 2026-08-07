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

Full methodology section, workload matrix, and per-module baseline numbers will be written in Plan 02.

---

*Full report sections (hot-path analysis, scikit-fda gap analysis, consolidated findings, prioritized backlog) to be written across Phases 2–9.*
