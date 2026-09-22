---
phase: 112-benchmark-harness-setup
fixed_at: 2026-09-22T00:00:00Z
review_path: .planning/phases/112-benchmark-harness-setup/112-REVIEW.md
iteration: 1
findings_in_scope: 6
fixed: 6
skipped: 0
status: all_fixed
---

# Phase 112: Code Review Fix Report

**Fixed at:** 2026-09-22
**Source review:** `.planning/phases/112-benchmark-harness-setup/112-REVIEW.md`
**Iteration:** 1

**Summary:**
- Findings in scope: 6 (CR-01, CR-02, WR-01, WR-02, WR-03, WR-04)
- Fixed: 6
- Skipped: 0

**Verification environment:** main checkout (workflow.use_worktrees=false — no isolated worktree).
Syntax verified via `python3 -c "import ast; ast.parse(...)"` for all .py files.
Tests run: `benchmarks/.venv/bin/python -m pytest benchmarks/test_gate.py benchmarks/test_collect_rust.py -q` — 25 passed.

---

## Fixed Issues

### CR-01: R timing values are in seconds, not nanoseconds

**Files modified:** `benchmarks/r_runner.R`
**Commit:** `f3dc1610`
**Applied fix:** Corrected the comment on `time_fn` (was "as.numeric() gives nanoseconds" — wrong)
and multiplied `as.numeric(timing$median)` by `1e9` to convert seconds to nanoseconds. The fix
covers the single timing helper used by ALL op branches — every R timing value was ~1e9x too small.

---

### CR-02: Broken command-list in `step_python_runner` when `op=None, smoke=True`

**Files modified:** `benchmarks/run_all.py`
**Commit:** `0ee33360`
**Applied fix:** Replaced the inline conditional `"--smoke" if smoke else "--op", "fpca"` with an
explicit `if smoke: ... else: ...` branch. The smoke path now emits `["python_runner.py", "--smoke",
"--n", ...]` only (no spurious `"fpca"` positional argument that caused argparse exit code 2).
Combined with WR-04 in the same commit since both affect the same functions.

---

### WR-01: Path-traversal guard in `collect_rust.py` defeated by symlinks on Python <= 3.12

**Files modified:** `benchmarks/collect_rust.py`
**Commit:** `9f283585`
**Applied fix:** Changed `est_path.relative_to(criterion_dir)` to
`est_path.resolve().relative_to(criterion_dir)` so symlinks that point outside `criterion_dir` are
caught even on Python <= 3.12 (where `Path.rglob()` follows directory symlinks). Added a comment
explaining the Python-version dependency. The existing `test_out_of_tree_path_rejected` test
continues to pass (it creates a symlink and verifies the path is rejected).

---

### WR-02: Python `run_functional_boxplot` times two depth computations

**Files modified:** `benchmarks/python_runner.py`
**Commit:** `4315eff6`
**Applied fix:** Removed the redundant `depth_fn(fd)` call from inside the timed `_run()` closure.
`_run()` now only calls `Boxplot(fd, depth_method=depth_fn, factor=1.5)` which already invokes MBD
internally — one MBD computation per iteration, matching R's single `fbplot()` call. The
`depth_fn(fd)` call is now made once, outside the timing loop, solely to produce the result
fingerprint for the gate.

Note: scikit-fda's `Boxplot` does not expose pre-computed depths as a public attribute (confirmed by
reading `_boxplot.py` in the installed package), so extracting depths from `bp` is not possible.
Moving the extra call outside the timed loop is the cleanest fix.

---

### WR-03: `check_agreement` returns `True` for unknown ops

**Files modified:** `benchmarks/gate.py`
**Commit:** `fac02ee4`
**Applied fix:** Replaced the fail-open `else` branch (which printed a warning and returned
`(True, 0.0, ...)`) with `raise ValueError(...)` naming all six valid ops. Fail-closed behavior
prevents op-name typos in future callers from silently passing the gate. `run_gate` already has
`if op not in _GATED_OPS: continue` before calling `check_agreement`, so the `ValueError` is
unreachable from the normal execution path — only direct callers with an unrecognised op string
hit it. All 25 existing tests continue to pass.

---

### WR-04: All-ops path in `run_all.py` always produces zero Python and R timings

**Files modified:** `benchmarks/run_all.py`
**Commit:** `0ee33360` (combined with CR-02)
**Applied fix:** Both `_run_all_ops_python` and the `step_r_runner(op=None, smoke=False)` path
previously always passed `--smoke` to their respective runners, setting `mean_ns=0.0` for every
op. Fixed both to loop over `ALL_OPS` and call each runner with `--op <single_op>` (no `--smoke`).
Since both runners overwrite their output JSON file on each invocation, the loop accumulates
records after each per-op run and writes the merged all-ops list at the end, so downstream
consumers (gate, `write_timing_comparison`) see all 6 op records with real timing values.

---

## Skipped Issues

None.

---

_Fixed: 2026-09-22_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
