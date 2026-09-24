---
phase: 116-cross-language-re-run-published-report
plan: "02"
subsystem: benchmarks
tags: [benchmark-harness, criterion, crosslang-grid, op-map, tdd, functional_boxplot-noise]

# Dependency graph
requires:
  - phase: 116-cross-language-re-run-published-report
    plan: "01"
    provides: crosslang_grid.rs's grid-aligned scaffold (GRID_CELLS, data_dir(), require_csv(), read_csv_rows(), load_curves(), StdRng(42) fallback, bench_crosslang_fpca) that this plan extends with the remaining five op groups
provides:
  - fdars-core/benches/crosslang_grid.rs — six crosslang_<op> criterion groups (fpca, basis_smoothing, depth, functional_boxplot, fpcr, elastic) over the full 6-cell N x M grid, parameter-matched to the Python/R runners
  - benchmarks/collect_rust.py OP_MAP — crosslang_<op> -> <op> entries so rust.json / comparison.csv label the new rows with the shared op names
  - fdars-core/benches/depth_benchmarks.rs functional_boxplot group — raised sample_size/measurement_time/warm_up_time to address the Phase 113 RSD 24-33% noise
affects: [116-03, 116-04, 116-06]

# Actuals (#2632)
actuals:
  tokens: 4900
  tasks: 2
  commits: 4
  plan_head_before: b255e7f447891e918db58ee246c6bb7826b420ec

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Per-cell lazy Option cache filled via get_or_insert_with inside a criterion bench_function closure, reused verbatim across all six crosslang_<op> groups"
    - "CSV-with-deterministic-fallback loader pair (load_response/load_elastic_pair) mirroring load_curves's FDARS_CROSSLANG_REQUIRE_CSV hard-fail / StdRng(seed) fallback contract, seeded off the harness's seed+100/seed+200 convention (StdRng 142/242)"

key-files:
  created: []
  modified:
    - fdars-core/benches/crosslang_grid.rs
    - benchmarks/collect_rust.py
    - benchmarks/test_collect_rust.py
    - fdars-core/benches/depth_benchmarks.rs

key-decisions:
  - "load_response/load_elastic_pair use StdRng::seed_from_u64(142)/seed_from_u64(242) as the deterministic fallback (harness base seed 42 + 100/+200), mirroring generate_response's seed+100 and generate_elastic_pair's seed+200 convention — internal reproducibility, not bit-for-bit parity with NumPy's PCG64 (StdRng and PCG64 are different RNGs by construction; the CSV path is what guarantees cross-language input parity)."
  - "crosslang_elastic benches all six grid cells (not just the two distinct m values) even though the pairwise computation depends only on m — intentional per the plan, so the report can empirically show the fdars N-scaling exponent is approximately 0 for this op (D-13)."
  - "Task 2's TDD cycle (RED/GREEN) covers only the OP_MAP mapping feature; the functional_boxplot measurement-settings change (plan step 3) is not test-driven behavior and was committed as its own atomic chore commit rather than folded into GREEN or labeled REFACTOR, keeping each commit's diff scoped to one concern."

requirements-completed: [RPT-01]

coverage:
  - id: D1
    description: "crosslang_grid.rs gains crosslang_basis_smoothing, crosslang_depth, crosslang_functional_boxplot, crosslang_fpcr, crosslang_elastic groups, each with 6 n{N}_m{M} ids matching the harness grid and parameters matched to the Python/R runners"
    requirement: RPT-01
    verification:
      - kind: unit
        ref: "cargo clippy -p fdars-core --benches --features linalg,parallel -- -D warnings"
        status: pass
      - kind: integration
        ref: "cargo bench -p fdars-core --features linalg,parallel --bench crosslang_grid -- --profile-time 1 '/n100_m50$' (6/6 distinct crosslang_<op>/n100_m50 ids observed)"
        status: pass
    human_judgment: false
  - id: D2
    description: "collect_rust.py OP_MAP maps all six crosslang_<op> groups to their shared op labels, covered by TestCrosslangOpMap (RED confirmed failing on the target assertions, then GREEN)"
    requirement: RPT-01
    verification:
      - kind: unit
        ref: "benchmarks/test_collect_rust.py::TestCrosslangOpMap (both test methods) — pytest -q reports 9 passed"
        status: pass
    human_judgment: false
  - id: D3
    description: "depth_benchmarks.rs functional_boxplot group runs with raised sample_size(50)/measurement_time(20s)/warm_up_time(5s) and a documented D-10 rationale"
    requirement: RPT-01
    verification:
      - kind: unit
        ref: "awk '/fn bench_functional_boxplot/,/group.finish/' fdars-core/benches/depth_benchmarks.rs | grep -c 'sample_size(50)' == 1; grep -c 'Phase 116 D-10' == 1"
        status: pass
    human_judgment: false

duration: 15min
completed: 2026-09-24
status: complete
---

# Phase 116 Plan 02: Grid-Aligned Cross-Language Benches — All Six Ops Summary

Expanded `crosslang_grid.rs` from the tracer's single `fpca` group to all six overlapping ops (basis_smoothing, depth, functional_boxplot, fpcr, elastic), each parameter-matched to the Python/R runners over the full 6-cell N×M grid; taught `collect_rust.py` to label the new rows with the shared op names via a TDD RED/GREEN cycle; and raised the noisy `functional_boxplot` group's criterion measurement settings per the Phase 113 baseline's RSD 24-33% finding (D-10).

## Performance

- **Duration:** ~15 min
- **Started:** 2026-09-24T21:20:00Z (approx.)
- **Completed:** 2026-09-24T21:35:20Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- `fdars-core/benches/crosslang_grid.rs` gained `crosslang_basis_smoothing`, `crosslang_depth`, `crosslang_functional_boxplot`, `crosslang_fpcr`, and `crosslang_elastic` — each a `n{N}_m{M}`-ided group over the full 6-cell grid, matching the Python/R runners' exact parameters (B-spline order 4/nbasis 10/lambda 0; Modified Band Depth; MBD boxplot factor 1.5; 5-component FPCR OLS; lambda-0 pairwise Fisher-Rao alignment).
- Two new CSV loaders, `load_response(n)` and `load_elastic_pair(m)`, follow `load_curves`'s exact contract: fail-closed under `FDARS_CROSSLANG_REQUIRE_CSV=1`, deterministic `StdRng` fallback (seeds 142/242, mirroring the harness's seed+100/seed+200 convention) otherwise.
- `benchmarks/collect_rust.py`'s `OP_MAP` now maps all six `crosslang_<op>` groups to their shared op labels, proven by a new `TestCrosslangOpMap` unit test class (TDD: RED confirmed failing on the exact target assertions, then GREEN with 9/9 `test_collect_rust.py` tests passing).
- `fdars-core/benches/depth_benchmarks.rs`'s `functional_boxplot` group now runs with `sample_size(50)`, `measurement_time(20s)`, `warm_up_time(5s)` and an inline doc comment documenting the Phase 113 RSD/CI-width noise finding and the rationale (Phase 116 D-10).
- All 36 crosslang criterion ids (6 ops × 6 cells) build and execute cleanly under `cargo clippy --benches --features linalg,parallel -- -D warnings` and a `--profile-time 1` smoke run at `n100_m50`; no `fdars-core/src` change, no new crate dependency.

## Task Commits

Each task was committed atomically (Task 2 followed the TDD RED → GREEN cycle plus one non-TDD config commit):

1. **Task 1: Add crosslang_basis_smoothing, crosslang_depth, crosslang_functional_boxplot, crosslang_fpcr, crosslang_elastic groups** - `78af2c99` (feat)
2. **Task 2 (RED): add failing test for crosslang op-map labels** - `ba797b4f` (test)
3. **Task 2 (GREEN): map crosslang_<op> criterion groups to shared op labels** - `cce28d10` (feat)
4. **Task 2 (config): raise functional_boxplot criterion measurement settings (D-10)** - `b73af348` (chore)

**Plan metadata:** (this commit)

## Files Created/Modified

- `fdars-core/benches/crosslang_grid.rs` - five new `crosslang_<op>` groups + `load_response`/`load_elastic_pair` loaders
- `benchmarks/collect_rust.py` - six `crosslang_<op> -> <op>` `OP_MAP` entries + docstring update
- `benchmarks/test_collect_rust.py` - new `TestCrosslangOpMap` unittest.TestCase class (2 test methods)
- `fdars-core/benches/depth_benchmarks.rs` - `functional_boxplot` group settings bump + D-10 doc comment

## Decisions Made

See `key-decisions` in the frontmatter above (StdRng seed convention for the two new fallback loaders, benching all six cells for `crosslang_elastic` despite N-independence, and splitting the non-TDD `functional_boxplot` settings change into its own commit rather than folding it into GREEN).

## Deviations from Plan

None - plan executed exactly as written. The plan's Task 2 step 4 phrasing ("run cargo fmt, then commit") reads as a single commit, but Task 2 carries `tdd="true"` — per the executor's TDD execution flow this was expanded into the canonical RED → GREEN commit-scope contract (`test(116-02):` then `feat(116-02):`) plus a separate `chore(116-02):` commit for the non-test-driven `functional_boxplot` settings change, keeping each commit scoped to one concern. This is a clarification of commit granularity, not a change to what was built.

## Issues Encountered

None.

## Known Stubs

None introduced by this plan. `documentation/PERFORMANCE.md`'s headline table still shows em dashes for `basis_smoothing`, `depth`, `functional_boxplot`, `fpcr`, and `elastic` (tracked from 116-01's Known Stubs) — this plan makes the fdars side of those rows measurable and correctly labeled, but populating the report table itself is 116-03/116-04's scope, not this plan's.

## Threat Flags

None beyond the plan's own `<threat_model>` (T-116-06, T-116-07), both disposed as `accept`/`mitigate` and implemented as specified: the new CSV loaders build file names only from integer grid constants and are read-only (T-116-06), and `collect_rust.py`'s existing path-escape guard (T-112-01) is unchanged and exercised by the new test's temp-dir tree (T-116-07).

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Ready for 116-03. Every one of the six overlapping ops now has a complete, correctly-labeled, per-cell fdars criterion number, so the head-to-head report and the fdars-side N/M scaling-exponent fits (D-13) have no structural gaps left to fill. `functional_boxplot`'s raised measurement settings should be re-checked against the 5% noise gate during the real Phase 116 re-run (116-06); if it still fails, the plan's own rationale says to flag it in the report, never headline it.

## Self-Check: PASSED

- Modified files verified present with the expected content: `fdars-core/benches/crosslang_grid.rs` (6 `bench_crosslang_*` functions, 6 `benchmark_group("crosslang_*")` calls), `benchmarks/collect_rust.py` (6 `"crosslang_` OP_MAP entries), `benchmarks/test_collect_rust.py` (`class TestCrosslangOpMap` present), `fdars-core/benches/depth_benchmarks.rs` (`sample_size(50)` + `Phase 116 D-10` both present) — all confirmed via grep/awk above.
- Commits verified: `git log --oneline` shows `78af2c99`, `ba797b4f`, `cce28d10`, `b73af348` all present on `main`; `git rev-list --count b255e7f4..HEAD` = 4 (matches `actuals.commits`).
- All plan-level `<verify>` and task `<verify>`/`<acceptance_criteria>` commands re-run and passing: `cargo clippy --all-targets --features linalg,parallel -- -D warnings` (0 warnings), `cargo fmt -p fdars-core -- --check` (exit 0), the crosslang_grid profile-time smoke run (6/6 ops at n100_m50), `benchmarks/.venv/bin/python -m pytest benchmarks/ -q` (69 passed), and the `fdars-core/Cargo.toml` no-dependency-change diff (empty).

---
*Phase: 116-cross-language-re-run-published-report*
*Completed: 2026-09-24*
