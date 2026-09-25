---
phase: 116-cross-language-re-run-published-report
fixed_at: 2026-09-25T00:00:00Z
review_path: .planning/phases/116-cross-language-re-run-published-report/116-REVIEW.md
iteration: 1
findings_in_scope: 5
fixed: 5
skipped: 0
status: all_fixed
---

# Phase 116: Code Review Fix Report

**Fixed at:** 2026-09-25T00:00:00Z
**Source review:** .planning/phases/116-cross-language-re-run-published-report/116-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 5 (fix_scope=critical_warning: 1 critical + 4 warnings)
- Fixed: 5
- Skipped: 0

All fixes were verified with `benchmarks/.venv/bin/python -m pytest benchmarks/ -q`
(209 passed, 0 failed, throughout) run in the main checkout (`workflow.use_worktrees`
is `false` in `.planning/config.json`, so no isolated worktree was created —
edits and commits happened directly in the main working tree on `main`, per the
documented opt-out). `benchmarks/render_report.py --check` and `--check-prose`
were re-run after every doc-affecting fix and exit 0.

## Fixed Issues

### CR-01: `step_run_gate` silently no-ops (fail-open) when an op's records are missing

**Files modified:** `benchmarks/run_all.py`
**Commit:** `a0a9bae6`
**Applied fix:** `step_run_gate` now accepts an explicit `ops: list[str] | None`
parameter naming the ops this call actually requested. Both call sites
(`run_grid`'s per-cell gate call and the single-op CLI path in `main()`) now
pass it explicitly. The FPCA hard-gate check and the step-6b per-op loop both
now print a `GATE FAIL: <op> requested but python or r record missing` message
and set the corresponding pass flag to `False` — instead of silently leaving
`fpca_passed`/`all_passed_py_r` at their initial `True` value — whenever a
requested op has no counterpart record on one or both sides. When `ops` is
omitted (legacy direct callers), it falls back to inferring requested-ness
from the union of ops present in either record set, preserving prior behavior
for full-op calls.

### WR-04: `step_run_gate` has zero direct unit-test coverage

**Files modified:** `benchmarks/test_run_all.py`
**Commit:** `0384c2b9`
**Applied fix:** Added `TestStepRunGate` with 6 tests: all-ops-agree passes;
fdars-vs-r hard-gate disagreement aborts (`SystemExit(1)`); the CR-01
regression case on the FPCA-specific path (`ops=["fpca"]`, r.json missing the
fpca record) fails closed and never even calls the (mocked) comparison
function; the CR-01 regression case on the generic step-6b loop
(`ops=["depth"]`, r.json missing) fails closed and never calls `run_gate()`
with a manufactured pass; a healthy present-on-both-sides subset passes; and
the `ops=None` legacy-inference fallback still works. Committed together with
CR-01 as a closely-related but separate atomic commit (fix code, then its
direct test coverage).

### WR-01: `_match_rust_rec`'s "exact n/m match" can never succeed

**Files modified:** `benchmarks/run_all.py`, `benchmarks/test_run_all.py`
**Commit:** `a91e4aa4`
**Applied fix:** Added `_parse_bench_n_m()`, which parses n/m out of a
criterion bench-ID string using the same regex style already used by
`assemble_comparison.py::_build_rust_rows` (`[_/]?n(\d+)` / `[_/]?m(\d+)`).
`_match_rust_rec` now scans `rust_records` for an op match whose *parsed*
bench-ID n/m equals the requested cell, and returns `{}` (no rust timing for
this exact cell) when none is found — replacing the previous "first record
whose op matches, regardless of size" fallback that could silently substitute
e.g. a `functional_boxplot` N=50 hardcoded sweep point for a requested N=500
cell. The caller (`write_timing_comparison`) already treats an empty dict as
"N/A", so this is a pure fail-safe improvement with no schema change to
`rust.json`/`collect_rust.py`. Added `TestMatchRustRec` (5 tests) and
`TestParseBenchNM` (4 tests) covering exact match, no-match-returns-empty,
prefixed bench IDs, wrong-op rejection, and the regex's correct non-match on
slash-separated sweep params (`"N/50"`, `"m/100"`).

### WR-02: `classify_missing` ignores the timed-out cell's `ops` list

**Files modified:** `benchmarks/render_report.py`, `benchmarks/test_render_report.py`
**Commit:** `c3ff49ec`
**Applied fix:** The python/r branch of `classify_missing` now additionally
checks that the missing pair's `op` is a member of the matching `grid_cell`
record's `ops` list before classifying it as `"timeout"` (falling through to
`"not-run"` otherwise). A `grid_cell` record with no `"ops"` key at all
(pre-existing/minimal schema) is still treated as covering any op, for
backward compatibility. Updated the existing
`test_grid_cell_timeout_for_python` fixture to include `"ops": ["fpca"]` (it
previously omitted the field, which is not representative of any real
committed `rerun_log.json` record — all 6 real `grid_cell` records already
carry a full `ops` list) and added two new regression tests: a wrong-op case
that must now classify as `"not-run"`, and a missing-`"ops"`-field
backward-compatibility case. Regenerated `documentation/PERFORMANCE.md` and
`README.md` — **no diff**, confirming the fix only changes behavior for
future partial (`--grid --op <subset>`) re-runs, not the currently committed
report data.

### WR-03: `emit_before_after_crosslang_fdars` conflates comparator and internal-only benches

**Files modified:** `benchmarks/render_report.py`
**Commit:** `c4113039`
**Applied fix:** Investigated both fix options in the review. Restricting to
criterion IDs starting with `crosslang_` would make the table permanently
empty (the true `crosslang_*` benches are always classified `"new-after"` by
`_paired_rust_rows`/`_is_new_crosslang_row`, since they don't exist in the
Phase 113 `comparison_baseline.csv`); a generic `(op, n, m)`-existence
"comparable" column would still mislabel e.g. `audit_p3_elastic_cross` (a
self/cross distance-matrix study) as "comparable" whenever its `(n, m)`
happens to coincide with a grid cell, since it shares the `elastic` op label
by coincidence of `collect_rust.py`'s `OP_MAP`, not because it measures the
same sub-operation Python/R measured. Applied the safer, non-lossy fix:
`emit_before_after_crosslang_fdars` now passes an explicit disclosure title
(mirroring `emit_before_after_internal`'s existing pattern) into
`_emit_before_after_table`, stating plainly that the table is a Rust-only
before/after regression check (same criterion ID across two runs) rather than
itself a Python/R comparison, and naming the specific row families
(`audit_p3_elastic_*`/`audit_karcher`/`karcher_mean`) that have no direct
Python/R equivalent at all. No rows were removed — all pre-existing timing
data is preserved, only the surrounding disclosure changed. Regenerated
`documentation/PERFORMANCE.md` (a 2-line addition: the new title sentence);
`README.md` is unaffected (this generated block isn't rendered there).
`render_report.py --check` and `--check-prose` both exit 0 after
regeneration.

## Skipped Issues

None — all 5 in-scope findings (1 critical, 4 warnings) were fixed.

---

_Fixed: 2026-09-25T00:00:00Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
