---
phase: 116-cross-language-re-run-published-report
reviewed: 2026-09-25T00:00:00Z
depth: standard
files_reviewed: 10
files_reviewed_list:
  - benchmarks/run_all.py
  - benchmarks/render_report.py
  - benchmarks/collect_rust.py
  - benchmarks/test_run_all.py
  - benchmarks/test_render_report.py
  - benchmarks/test_collect_rust.py
  - benchmarks/.gitignore
  - fdars-core/Cargo.toml
  - fdars-core/benches/crosslang_grid.rs
  - fdars-core/benches/depth_benchmarks.rs
findings:
  critical: 0
  warning: 0
  info: 6
  total: 6
status: clean
---

# Phase 116: Code Review Report (re-review after fix pass)

**Reviewed:** 2026-09-25T00:00:00Z
**Depth:** standard
**Files Reviewed:** 10
**Status:** clean

## Summary

This is a re-review after the fix pass documented in `116-REVIEW-FIX.md`
(commits `a0a9bae6`, `0384c2b9`, `a91e4aa4`, `c3ff49ec`, `c4113039`), which
addressed CR-01 and WR-01..WR-04 from the prior `116-REVIEW.md`. All five
fixes were verified directly against the current source (not just the fixer's
self-report):

- **CR-01** (`step_run_gate` fail-open on missing op records): confirmed
  fixed. `step_run_gate` now takes an explicit `ops` parameter; both call
  sites (`run_grid` at `run_all.py:1013` and the single-op CLI path at
  `run_all.py:1959`) pass it. The FPCA hard-gate branch and the step-6b
  per-op loop both now print `GATE FAIL: <op> requested but python or r
  record missing` and set the corresponding pass flag to `False` — verified
  by reading `run_all.py:536-620` directly and by running the new
  `TestStepRunGate` suite (`test_ops_subset_missing_counterpart_record_fails_closed`,
  `test_ops_subset_missing_non_fpca_counterpart_fails_closed`), both of which
  assert `SystemExit(1)` and that the mocked comparison function is never
  called with a manufactured pass.
- **WR-01** (`_match_rust_rec` exact-match dead code): confirmed fixed. New
  `_parse_bench_n_m()` parses n/m from the bench ID string (mirroring the
  existing `assemble_comparison.py::_build_rust_rows` regex — not a novel,
  untested pattern), and `_match_rust_rec` now returns `{}` when no record's
  parsed n/m matches the requested cell, instead of falling back to the first
  op-matching record regardless of size. Verified against
  `fdars-core/benches/crosslang_grid.rs`'s actual bench IDs
  (`format!("n{n}_m{m}")`), which parse cleanly and unambiguously.
- **WR-02** (`classify_missing` ignoring `ops`): confirmed fixed. The
  python/r branch of `classify_missing` (`render_report.py:548-568`) now
  requires `op in (rec.get("ops") or [])` (or absence of the `ops` key, for
  backward compatibility) before classifying a missing pair as `"timeout"`.
- **WR-03** (crosslang-fdars table conflates comparator and internal-only
  benches): confirmed addressed via the disclosure-title approach (option
  investigated and documented as safer than either literal fix suggestion,
  both of which had their own problems — see `116-REVIEW-FIX.md`). The
  generated `documentation/PERFORMANCE.md` now carries the explicit
  disclosure sentence; `render_report.py --check` confirms it is not stale.
- **WR-04** (no direct test coverage for `step_run_gate`): confirmed fixed.
  `TestStepRunGate` (6 tests) directly exercises the pass, hard-gate-fail,
  and both CR-01 regression paths (FPCA-specific and generic step-6b loop).

**Verification performed independently of the fixer's self-report:**
`benchmarks/.venv/bin/python -m pytest benchmarks/ -q` from the repo root —
**209 passed, 0 failed**. `render_report.py --check` and `--check-prose` both
exit 0 (re-run directly, not just trusted from the fix log). Diffs for all
five fix commits were read in full; no file outside
`run_all.py`/`render_report.py`/`test_run_all.py`/`test_render_report.py`/
`documentation/PERFORMANCE.md` was touched, matching the fix report's stated
scope.

No new Critical or Warning issues were introduced by the fix pass. One new
minor diagnostic-completeness observation is recorded below as Info (IN-06).
The five pre-existing Info items from the original review are carried
forward unchanged (none of the fix commits touched their surrounding code).

## Info

### IN-06 (new): short-circuited `run_gate()` call skips diagnostics for other requested ops once one op's record is missing

**File:** `benchmarks/run_all.py:618` (`all_passed_py_r = all_passed_py_r and run_gate(gate_records)`)

**Issue:** The CR-01 fix changed this line from an unconditional
`all_passed_py_r = run_gate(gate_records)` to
`all_passed_py_r = all_passed_py_r and run_gate(gate_records)`. Python's
`and` short-circuits, so if an earlier op in the step-6b loop was requested
but missing a counterpart record (`all_passed_py_r` already `False`),
`run_gate(gate_records)` is never called at all — even if `gate_records`
contains other, fully-present ops that were appended to it before the
missing-record `continue`. This is correct for the pass/fail *result* (the
overall gate already fails regardless — verified by
`test_ops_subset_missing_non_fpca_counterpart_fails_closed` asserting
`run_gate` is never called), but it means the per-op `GATE FAIL [<lib>]
<op>` / pass diagnostics for those other, fully-present ops are silently
skipped on the console — a reader debugging "why did my `--grid --op
depth,elastic` gate fail" would see only the missing-record message for
`depth`, with no information about whether `elastic` (present on both
sides) would itself have passed or failed. Purely a diagnostics-completeness
regression in an already-failing path; the fail-closed correctness is
unaffected.

**Fix:** Call `run_gate(gate_records)` unconditionally (drop the
short-circuit) and combine with `all_passed_py_r` via a non-short-circuiting
combination, e.g. `gate_ok = run_gate(gate_records) if gate_records else
True; all_passed_py_r = all_passed_py_r and gate_ok`, so the diagnostics for
present ops still print even when another requested op's record is missing.

### IN-01 (carried forward): `smoke` parameter is dead code — no CLI flag ever sets it `True`

**File:** `benchmarks/run_all.py:326, 396` (`step_python_runner`, `step_r_runner`)

**Issue:** Unchanged by the fix pass. Both functions accept
`smoke: bool = False` and branch on it, but `main()`'s argparse defines no
`--smoke` flag and no call site passes it. The `if smoke:` branches are
unreachable from the CLI.

**Fix:** Either wire up a `--smoke` flag or remove the dead branches/parameter.

### IN-02 (carried forward): Falsy-check hides a legitimate `0.0` timing as "N/A"

**File:** `benchmarks/run_all.py:733`

**Issue:** Unchanged by the fix pass.
`py_t = f"{row['python_mean_ns']:.0f}" if row["python_mean_ns"] else "N/A"`
uses truthiness rather than `is not None`; a `mean_ns` of exactly `0.0`
would print as `"N/A"`. Console-diagnostics cosmetic only (the underlying
JSON is written correctly).

**Fix:** `if row["python_mean_ns"] is not None else "N/A"`.

### IN-03 (carried forward): `parse_criterion_settings` silently keeps only the last occurrence when a group name repeats

**File:** `benchmarks/render_report.py:163, 765` (`_GROUP_BLOCK_RE`, its use)

**Issue:** Unchanged by the fix pass. A later `benchmark_group("<name>")`
match with a repeated group name silently overwrites an earlier one. Not
triggered by any bench file currently reviewed, but fragile against a future
bench file splitting the same group literal across two functions.

**Fix:** Merge (or assert-uniqueness) instead of blind overwrite.

### IN-04 (carried forward): `check_prose`'s number-detection regex misses numbers followed directly by punctuation

**File:** `benchmarks/render_report.py:1635`

**Issue:** Unchanged by the fix pass.
`_NUMBER_UNIT_RE = re.compile(r"(\d[\d,]*\.?\d*)\s*(?:×|[xX]\s|ms\b)")`
requires whitespace immediately after a bare `x`/`X`; `"10.00x,"` (comma
directly after `x`) would not be flagged as an unbacked number.

**Fix:** Use a word-boundary pattern, e.g. `[xX]\b`, instead of requiring a
literal following whitespace character.

### IN-05 (carried forward): `emit_scaling`'s noise-dagger applies to every exponent for an op even when only one unrelated cell is noisy

**File:** `benchmarks/render_report.py:1341-1351`

**Issue:** Unchanged by the fix pass. `noisy_any` is computed once per op
over the entire 3×2 grid and applied uniformly to all five reported
exponents, over-flagging (safe direction) rather than precisely attributing
noise to the specific cells that feed each exponent.

**Fix:** Compute per-exponent noise flags from only the cells that feed that
specific exponent's `fit_scaling_exponent` call.

---

_Reviewed: 2026-09-25T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
