---
phase: 112-benchmark-harness-setup
reviewed: 2026-09-22T10:00:00Z
depth: standard
files_reviewed: 10
files_reviewed_list:
  - benchmarks/gate.py
  - benchmarks/collect_rust.py
  - benchmarks/run_all.py
  - benchmarks/generate_data.py
  - benchmarks/python_runner.py
  - benchmarks/r_runner.R
  - benchmarks/capture_env.py
  - benchmarks/verify_inputs.py
  - benchmarks/fpca_dump/src/main.rs
  - fdars-core/benches/depth_benchmarks.rs
findings:
  critical: 0
  warning: 0
  info: 4
  total: 4
status: clean
---

# Phase 112: Code Review Report (Post-Fix Re-review)

**Reviewed:** 2026-09-22
**Depth:** standard
**Files Reviewed:** 10
**Status:** clean

## Summary

Re-review after the fix pass. All six prior findings are confirmed resolved, with no
regressions introduced. No new Critical or Warning findings were discovered.

**Fix verification — all six fixes hold:**

- **CR-01 (R timing unit):** `r_runner.R:83` now reads
  `mean_ns <- as.numeric(timing$median) * 1e9`. The comment on line 74 was also
  corrected ("as.numeric() gives seconds"). Every op branch routes through `time_fn`,
  and the smoke path correctly returns `0.0` without touching the timing multiplier.

- **CR-02 (broken smoke command construction):** `run_all.py:step_python_runner` now
  branches cleanly — the smoke sub-path calls `_run(...)` with `["--smoke", ...]`
  (no op value spliced), and the non-smoke path delegates to `_run_all_ops_python`.
  The broken `["--smoke", "fpca", ...]` construction is gone.

- **WR-04 (all-ops zero timings):** Both `_run_all_ops_python` (lines 126–152) and
  the non-smoke branch of `step_r_runner` (lines 168–188) now loop over `ALL_OPS`
  with individual `--op <name>` invocations and accumulate the per-op JSON records
  into a merged file. Real timings will be written to `python.json` and `r.json`
  for all 6 ops on a default `--op all` run.

- **WR-01 (symlink guard):** `collect_rust.py:103` calls
  `est_path.resolve().relative_to(criterion_dir)`, resolving symlinks before the
  path-containment test.

- **WR-02 (double MBD computation):** `python_runner.py:244` — `_run()` now contains
  only the single `Boxplot(fd, depth_method=depth_fn, factor=1.5)` call. The
  fingerprint depth call `depth_fn(fd)` (line 250) is outside the timed closure and
  is not counted in `mean_ns`.

- **WR-03 (fail-open unknown op):** `gate.py:244–250` — the `else` branch now raises
  `ValueError` with the full list of valid ops.

**Regression check — none found.** The `_run_all_ops_python` / `_run_all_ops_r`
accumulator pattern correctly reads each per-op 1-element JSON list back after the
subprocess exits 0 (`check=True`), extends the running list, and writes the merged
6-element list once. The gate in `step_run_gate` reads the merged file and gates before
`write_timing_comparison` is called — the BENCH-05 measure-first invariant is
preserved for the all-ops path.

---

## Narrative Findings (AI reviewer)

No Critical or Warning findings. Four Info items are noted below (IN-01 through IN-03
are carry-forwards from the prior review; IN-04 is new).

---

## Info

### IN-01: R script in `verify_inputs.py` inlines the CSV path via f-string

**File:** `benchmarks/verify_inputs.py:68,86`

**Issue:** Both the primary and fallback R scripts in `content_hash_r` embed the file
path by f-string interpolation into a double-quoted R string literal. A path
containing `"` or `\` (common on Windows) would produce an R parse error or silently
read the wrong path. The harness is Linux-only in practice and paths are
auto-generated, so risk is low. No change made by the fix pass.

**Fix:** Escape the path before interpolation:
```python
r_path = str(path).replace("\\", "\\\\").replace('"', '\\"')
# then use r_path in the f-string
```

---

### IN-02: `fpca_dump/src/main.rs` panics on `m=1` (integer underflow in argvals)

**File:** `benchmarks/fpca_dump/src/main.rs:100`

**Issue:**
```rust
let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
```
When `m=1`, `m - 1` is `0usize` and the integer division panics in both debug and
release builds. The harness always passes `m=50`, so this is unreachable in normal
use. No change made by the fix pass.

**Fix:**
```rust
if m < 2 {
    eprintln!("fpca_dump: m must be >= 2, got {}", m);
    std::process::exit(1);
}
```

---

### IN-03: `capture_env.py` uses deprecated `datetime.datetime.utcnow()`

**File:** `benchmarks/capture_env.py:171`

**Issue:**
```python
"timestamp": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
```
`datetime.utcnow()` is deprecated in Python 3.12 and scheduled for removal. No change
made by the fix pass.

**Fix:**
```python
"timestamp": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
```

---

### IN-04: `_run_all_ops_python` and R accumulator silently skip a record if the result file is absent after a successful subprocess exit

**File:** `benchmarks/run_all.py:145,182`

**Issue:** In `_run_all_ops_python` (line 145) and the analogous R loop (line 182),
the per-op record is only accumulated when `py_out.exists()` / `r_out.exists()` is
true after the subprocess returns. If a subprocess exits 0 but does not create the
output file (e.g. due to a disk-full condition at the runner side), the record is
silently dropped from `accumulated`. The final write would then produce a JSON with
fewer than 6 entries; the gate would skip the missing op (line 299 in `gate.py`
returns True for missing fdars reference), and `write_timing_comparison` would emit
`None` for that op's `python_mean_ns` or `r_mean_ns` without error.

In practice, `check=True` on `_run` catches non-zero exits and the runners always
write the output file before exiting 0. The risk is low, but a missing-file assertion
after `_run` would make this fail loudly:

```python
_run([_PYTHON, ..., "--op", single_op, ...], env=env)
if not py_out.exists():
    raise RuntimeError(
        f"python_runner.py exited 0 but did not write {py_out} for op={single_op!r}"
    )
with py_out.open() as f:
    records = json.load(f)
accumulated.extend(records)
```

---

_Reviewed: 2026-09-22_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
