---
phase: 116-cross-language-re-run-published-report
plan: "01"
subsystem: benchmarks
tags: [benchmark-harness, criterion, tracer, crosslang-grid, headline-report, ab-evidence, gitignore-carveout]

# Dependency graph
requires:
  - phase: 113-cross-language-baseline-hotspot-ranking
    provides: the committed Phase 113 before-baseline (comparison.csv @ 597f857d) this plan freezes byte-identically, plus run_all.py/gate.py/assemble_comparison.py/rank_hotspots.py the new scripts extend or reuse verbatim
  - phase: 114-elastic-alignment-optimization
    provides: committed before/after criterion estimates.json (dp_fill_grid banded refactor, within-noise verdict) this plan copies into benchmarks/results/ab/
  - phase: 115-svd-fpca-covariance-optimization
    provides: committed before/after criterion estimates.json (sym_cov_irreg, ~3.45x speedup) and the SVD re-benchmark estimates (near-optimal verdict) this plan copies into benchmarks/results/ab/
provides:
  - fdars-core/benches/crosslang_grid.rs — grid-aligned fdars criterion bench scaffold (crosslang_fpca group; 116-02 adds the other five op groups on this same scaffold)
  - benchmarks/run_all.py --cell/--rust-bench/--rust-filter/--timeout-s CLI + merge-not-overwrite per-cell persistence, for single-cell/single-op re-runs
  - benchmarks/render_report.py — the generated-table renderer (splice_generated/select_cell/ratio_label/emit_headline/main), reused verbatim by 116-04's remaining report blocks
  - documentation/PERFORMANCE.md skeleton with a populated, gate-passed fpca headline row
  - benchmarks/results/comparison_baseline.csv + env_baseline.json — the frozen Phase 113 before-evidence 116-04's before/after tables diff against
  - benchmarks/results/ab/ — the Phase 114/115 A/B + SVD-rebench evidence, with a manifest disclosing SHAs and unpinned thread config
affects: [116-02, 116-03, 116-04, 116-07]

# Actuals (#2632)
actuals:
  tokens: 25000
  tasks: 2
  commits: 2

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Grid-aligned criterion bench scaffold: crosslang_<op> group, n{N}_m{M} bench ids, seeded-CSV-with-deterministic-fallback data loading, gated behind FDARS_CROSSLANG_REQUIRE_CSV for real harness runs"
    - "Generated-block splicing: BEGIN/END GENERATED marker pairs, idempotent replace-between-markers, tables generated / prose hand-written"
    - "Process-group timeout kill (start_new_session + os.killpg) for a bounded single-bench cargo invocation, mirroring the harness's existing subprocess-only-list-form convention"
    - "gitignore directory-exclusion carve-out: results/* + explicit !negations (a bare directory pattern cannot be selectively re-included)"

key-files:
  created:
    - fdars-core/benches/crosslang_grid.rs
    - benchmarks/render_report.py
    - documentation/PERFORMANCE.md
    - benchmarks/results/comparison_baseline.csv
    - benchmarks/results/env_baseline.json
    - benchmarks/results/ab/manifest.json
    - benchmarks/results/ab/phase114_elastic_cross_banded_n500_m50/before.json
    - benchmarks/results/ab/phase114_elastic_cross_banded_n500_m50/after.json
    - benchmarks/results/ab/phase115_face_covariance_n200_m30/before.json
    - benchmarks/results/ab/phase115_face_covariance_n200_m30/after.json
    - benchmarks/results/ab/phase115_svd_rebench/svd_n500_m50.json
    - benchmarks/results/ab/phase115_svd_rebench/svd_n1000_m200.json
    - benchmarks/results/ab/phase115_svd_rebench/svd_n500_m500.json
  modified:
    - fdars-core/Cargo.toml
    - benchmarks/run_all.py
    - benchmarks/.gitignore

key-decisions:
  - "crosslang_grid.rs's criterion group is named crosslang_fpca (not fpca) so collect_rust.py's OP_MAP falls back to the raw group name as the op label in comparison.csv; no OP_MAP change was needed because render_report.py's select_cell() matches the fdars row by the criterion `source` string (criterion:crosslang_fpca/n500_m200), not by the op field."
  - "The svd_rebench manifest entry has no before/after pair (it is a tightened-CI re-measurement, not an A/B) — recorded with a `files` list + a single `sha` (74b9ab0d, the Phase 115 Plan 02 Task 2 commit) instead of sha_before/sha_after."
  - "Both Phase 114 and Phase 115 A/B runs used the default (unpinned) rayon pool, not thread-pinned like the Phase 116 harness — disclosed verbatim in each manifest entry's thread_config so the report never implies apples-to-apples thread configuration across the two evidence sources."
  - "gitignore rule changed from `results/` to `results/*` plus explicit `!` negations per file/pattern — git cannot selectively re-include a path whose parent directory itself is ignored, confirmed by testing a nested benchmarks/results/ab/testdir/sample.json against git check-ignore before committing the real evidence."

requirements-completed: [RPT-01]

coverage:
  - id: D1
    description: "Grid-aligned fdars criterion bench scaffold (crosslang_fpca group, 6-cell N x M ids, seeded-CSV loading with deterministic StdRng(42) fallback)"
    verification:
      - kind: unit
        ref: "cargo clippy -p fdars-core --benches --features linalg,parallel -- -D warnings"
        status: pass
      - kind: integration
        ref: "cargo bench -p fdars-core --features linalg,parallel --bench crosslang_grid -- '^crosslang_fpca/n500_m200$' (RUST_BENCH_STATUS=ok)"
        status: pass
    human_judgment: false
  - id: D2
    description: "run_all.py --cell/--op grid subsetting, --rust-bench/--rust-filter/--timeout-s single-bench invocation with process-group timeout kill, merge-not-overwrite per-cell persistence"
    verification:
      - kind: unit
        ref: "run_all._parse_cell('500x200') == (500, 200); raises ValueError on '500x75'"
        status: pass
      - kind: integration
        ref: "benchmarks/run_all.py --grid --cell 500x200 --op fpca -> [GATE PASS]; python_n500_m200.json/r_n500_m200.json correctly merged (6/7 records, not truncated to 1)"
        status: pass
    human_judgment: false
  - id: D3
    description: "render_report.py generator: splice_generated, select_cell, ratio_label, emit_headline, main"
    verification:
      - kind: unit
        ref: "splice_generated idempotency check (plan acceptance criterion): old content removed, new content present, re-splicing is a no-op"
        status: pass
      - kind: integration
        ref: "render_report.py --after tracer_comparison.csv --doc documentation/PERFORMANCE.md --blocks headline; re-run leaves git diff --quiet true"
        status: pass
    human_judgment: false
  - id: D4
    description: "documentation/PERFORMANCE.md skeleton with a gate-passed, fully-populated fpca headline row at N=500, M=200"
    verification:
      - kind: other
        ref: "awk '/BEGIN GENERATED: headline/,/END GENERATED: headline/' documentation/PERFORMANCE.md | grep -Eq '^\\| fpca \\| 500 \\| 200 \\| [0-9]+\\.[0-9]{3}[^|]* \\| [0-9]+\\.[0-9]{3} \\| [0-9]+\\.[0-9]{3} \\|'"
        status: pass
    human_judgment: false
  - id: D5
    description: "Phase 113 before-baseline frozen byte-identically (comparison_baseline.csv, env_baseline.json), taken from git objects at 597f857d, before any Phase 116 re-run"
    verification:
      - kind: other
        ref: "cmp against `git show 597f857d:benchmarks/results/comparison.csv` and `git show HEAD:benchmarks/results/comparison.csv`"
        status: pass
    human_judgment: false
  - id: D6
    description: "Phase 114/115 A/B + SVD-rebench criterion evidence committed under benchmarks/results/ab/ with a manifest disclosing SHAs and thread config"
    verification:
      - kind: other
        ref: "python assertion: 3 manifest entry ids, 7 total before/after/files paths, every referenced estimates.json has mean.point_estimate > 0"
        status: pass
    human_judgment: false
  - id: D7
    description: "benchmarks/.gitignore carve-out re-includes every committed evidence artifact while scratch outputs stay ignored"
    verification:
      - kind: other
        ref: "git check-ignore --no-index checks for comparison.csv/comparison_baseline.csv/env.json/env_baseline.json/hotspot_ranking.md/rust.json/rerun_log.json/python_n*/r_n*/ab/** (all NOT ignored) vs tracer_comparison.csv/python.json (still ignored)"
        status: pass
    human_judgment: false

duration: 20min
completed: 2026-09-24
status: complete
---

# Phase 116 Plan 01: Tracer Slice — Cross-Language Re-Run & Published Report Summary

Proved the whole measure -> gate -> assemble -> render -> publish path for one op (fpca) at one grid cell (N=500, M=200) — thread-pinned fdars criterion, scikit-fda, and R (SVD-matched + native pca.fd) all agree on the unchanged fdars-vs-R hard gate, producing a fully populated, generated headline row in `documentation/PERFORMANCE.md` — then froze the Phase 113 before-baseline and the Phase 114/115 A/B evidence so no later re-run can silently overwrite the reference numbers.

## Performance

- **Duration:** ~20 min
- **Started:** 2026-09-24T19:09:00Z (approx.)
- **Completed:** 2026-09-24T19:24:12Z
- **Tasks:** 2
- **Files modified:** 16 (13 created, 3 modified)

## Accomplishments

- New `fdars-core/benches/crosslang_grid.rs` — grid-aligned `crosslang_fpca` criterion group reading the exact same seeded CSVs the Python/R runners read, with a `FDARS_CROSSLANG_REQUIRE_CSV=1` hard-fail for real harness runs and a deterministic `StdRng(42)` fallback for standalone `cargo bench` runs.
- `benchmarks/run_all.py` gained `--cell NxM` (repeatable), grid-mode `--op` filtering with merge-not-overwrite per-cell persistence, and `--rust-bench`/`--rust-filter`/`--timeout-s` (`_run_with_timeout` + `run_rust_bench`) for a single thread-pinned criterion invocation with process-group SIGKILL on timeout.
- New `benchmarks/render_report.py` — the report-table generator (`splice_generated`, `select_cell`, `ratio_label`, `emit_headline`, `main`) that later plans (116-04) extend with the remaining report blocks.
- `documentation/PERFORMANCE.md` created with a real, gate-passed fpca headline row: fdars 22.026 ms (noise-flagged, RSD 38.9%) vs scikit-fda 18.154 ms vs R (SVD-matched) 79.436 ms.
- Committed the Phase 113 before-baseline (`comparison_baseline.csv`, `env_baseline.json`) byte-identical to the git objects at `597f857d`, and the Phase 114/115 A/B + SVD-rebench criterion evidence under `benchmarks/results/ab/` with a manifest disclosing SHAs and (unpinned) thread configuration.
- `benchmarks/.gitignore` switched from a bare `results/` exclusion to `results/*` + explicit negations, so the frozen evidence can be tracked without `git add -f` while scratch outputs stay ignored.

## Task Commits

Each task was committed atomically:

1. **Task 1: End-to-end FPCA slice — grid-aligned fdars bench -> scikit-fda/R -> gate -> assembled CSV -> generated headline row** - `200358cd` (feat)
2. **Task 2: Freeze the Phase 113 baseline + env and the Phase 114/115 A/B evidence; carve committed artifacts out of the results ignore rule** - `6af0a7c2` (docs)

**Plan metadata:** (this commit)

## Files Created/Modified

- `fdars-core/benches/crosslang_grid.rs` - grid-aligned `crosslang_fpca` criterion bench, 6-cell scaffold (only the fpca group wired this plan)
- `fdars-core/Cargo.toml` - registered the `crosslang_grid` bench (`harness = false`, no dependency change)
- `benchmarks/run_all.py` - `_parse_cell`, `_run_with_timeout`, `run_rust_bench`, `_merge_cell_records`, plus `--cell`/`--rust-bench`/`--rust-filter`/`--timeout-s` CLI
- `benchmarks/render_report.py` - new generated-table renderer
- `documentation/PERFORMANCE.md` - new report, headline block populated
- `benchmarks/results/comparison_baseline.csv`, `benchmarks/results/env_baseline.json` - frozen Phase 113 before-evidence
- `benchmarks/results/ab/manifest.json` + 7 `estimates.json` copies - frozen Phase 114/115 A/B + SVD-rebench evidence
- `benchmarks/.gitignore` - `results/*` + negation carve-out

## Decisions Made

See `key-decisions` in the frontmatter above (op-label matching via `source` string, svd_rebench manifest shape, unpinned-thread disclosure, gitignore carve-out mechanics).

## Deviations from Plan

None - plan executed exactly as written. The tracer's `<verify>` carried only `<automated>` checks and `HUMAN_VERIFY_MODE` is the default `end-of-phase`, so per the executor's tracer-feedback-gate protocol the automated `<verify>` was re-run after Task 1 and passed — no checkpoint was synthesized, and Task 2 proceeded directly.

## Issues Encountered

None. The `run_all.py --grid --cell 500x200 --op fpca` run correctly merged into the pre-existing Phase 113 per-cell JSON files (6/7 records retained, not truncated to the single re-run op) — this was the intended `_merge_cell_records` behavior, not a bug. All tracked timing files it touched (`rust.json`, `env.json`, `python_n500_m200.json`, `r_n500_m200.json`) were reverted with `git restore benchmarks/results/` before committing, per the plan's step 6(f).

## Known Stubs

- `documentation/PERFORMANCE.md`'s headline table shows an em dash for `basis_smoothing`, `depth`, `functional_boxplot`, `fpcr`, and `elastic` — only `fpca` has a grid-aligned fdars criterion bench so far (this plan's tracer scope was deliberately one op). Resolved by 116-02, which adds the other five `crosslang_<op>` groups to `crosslang_grid.rs` on this same scaffold. Logged to `.planning/WINDOWS.md` (kind: todo, phase 116).
- The Methodology and Reproduction sections referenced in `PERFORMANCE.md`'s intro paragraph do not exist yet — intentional, authored by 116-07 per the plan's task list.

## Threat Flags

None beyond the plan's own `<threat_model>` (T-116-01..05), which are all disposed as `mitigate`/`accept` and implemented as specified: list-form-only subprocess calls with bench-file existence validation, process-group timeout kill, git-object-sourced baseline freeze, and numeric-only headline cells.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Ready for 116-02 (expand `crosslang_grid.rs` to the other five ops on this proven scaffold). The `run_all.py` grid-subsetting/merge machinery, `render_report.py`'s splice/select/ratio-label primitives, and the frozen baseline + A/B evidence are all reusable as-is by the remaining six plans in this phase.

## Self-Check: PASSED

- Created files verified present: `fdars-core/benches/crosslang_grid.rs`, `benchmarks/render_report.py`, `documentation/PERFORMANCE.md`, `benchmarks/results/comparison_baseline.csv`, `benchmarks/results/env_baseline.json`, `benchmarks/results/ab/manifest.json` (+ 7 estimates.json) — all present on disk.
- Commits verified: `git log --oneline --all | grep -q 200358cd` and `6af0a7c2` both found.
- All plan-level `<verify>` and task `<verify>`/`<acceptance_criteria>` commands re-run and passing (see coverage block above).

---
*Phase: 116-cross-language-re-run-published-report*
*Completed: 2026-09-24*
