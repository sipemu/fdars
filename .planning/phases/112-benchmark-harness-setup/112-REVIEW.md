---
phase: 112-benchmark-harness-setup
reviewed: 2026-09-22T00:00:00Z
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
  critical: 2
  warning: 4
  info: 3
  total: 9
status: issues_found
---

# Phase 112: Code Review Report

**Reviewed:** 2026-09-22
**Depth:** standard
**Files Reviewed:** 10
**Status:** issues_found

## Summary

Phase 112 delivers the full cross-language benchmark harness (Python/R/Rust, six shared ops,
numeric-agreement gate, criterion collector, environment capture). The architecture is sound:
the gate fails closed on mismatch, sign-alignment logic is correct, Simpson's weights are
faithfully ported to R, list-form subprocesses are used throughout, and the new
`bench_functional_boxplot` group is properly registered.

Two blockers are present. The more severe is a unit error in the R runner that makes every R
timing value ~1e9x too small (seconds stored in a field named `mean_ns`). The second is a
broken command-list construction in `step_python_runner` that causes an argparse error whenever
the smoke path is taken for the all-ops case. Four warnings cover the path-traversal guard's
Python-version dependency, a double-computation in the Python functional_boxplot timing loop,
a fail-open branch in `check_agreement`, and zero-timing data silently written to the comparison
JSON. Three info items cover code quality.

Known/accepted items from the phase specification (depth/fpcr/functional_boxplot tolerance
calibration deferred to Phase 113; venv multimethod patch; R hand-rolled FPCA) are not flagged.

---

## Critical Issues

### CR-01: R timing values are in seconds, not nanoseconds — all R timing comparisons are ~1e9x wrong

**File:** `benchmarks/r_runner.R:74,82`

**Issue:** The comment on line 74 states `"as.numeric() gives nanoseconds"` but this is
factually wrong. `bench::mark()` returns a `bench_time` object; calling `as.numeric()` on it
yields the duration in **seconds** (confirmed: `Sys.sleep(0.001)` → `as.numeric(median) ≈
0.001`). The field is stored as `mean_ns` and consumed by `run_all.py::write_timing_comparison`
without any unit conversion (lines 335, 348, 361 of `run_all.py`). Every R timing entry in
`timing_comparison.json` and `fpca_timing.json` will be `~1e-9` of the correct nanosecond
value, making every R vs Python/Rust speed comparison completely wrong.

```r
# CURRENT (wrong):
mean_ns <- as.numeric(timing$median)   # returns seconds, e.g. 0.001 for a 1 ms op

# FIX: multiply by 1e9 to convert to nanoseconds
mean_ns <- as.numeric(timing$median) * 1e9   # now correctly ~1000000 ns for 1 ms op
```

---

### CR-02: Broken command-list in `step_python_runner` when `op=None, smoke=True` — argparse error

**File:** `benchmarks/run_all.py:105-110`

**Issue:** The argument list construction on lines 105–110 conditionally replaces `"--op"` with
`"--smoke"` but unconditionally appends the string `"fpca"` as the next element:

```python
[_PYTHON, str(_SCRIPT_DIR / "python_runner.py"),
 "--smoke" if smoke else "--op", "fpca",    # "fpca" always appended
 "--n", str(n), "--m", str(m), "--seed", str(seed)]
```

When `smoke=True`, the resulting command is:
```
python python_runner.py --smoke fpca --n 60 --m 50 --seed 42
```

`python_runner.py` uses `--smoke` as a `store_true` flag; argparse then sees `"fpca"` as an
unrecognized positional argument and exits with code 2. The function is not called with
`smoke=True` from the current `main()`, but the function signature and branch exist, and the
broken path would fail silently if a future caller sets `smoke=True`.

```python
# FIX: build the list conditionally rather than splicing a conditional in mid-list
if smoke:
    cmd = [_PYTHON, str(_SCRIPT_DIR / "python_runner.py"),
           "--smoke",
           "--n", str(n), "--m", str(m), "--seed", str(seed)]
else:
    _run_all_ops_python(n, m, seed, env)
    return
_run(cmd, env=env)
```

---

## Warnings

### WR-01: Path-traversal guard in `collect_rust.py` is defeated by symlinks on Python ≤ 3.12

**File:** `benchmarks/collect_rust.py:93-101`

**Issue:** The guard resolves `criterion_dir` (line 93) but does **not** resolve `est_path`
before calling `est_path.relative_to(criterion_dir)` (line 98). A symlink inside
`criterion_dir` pointing to a directory outside it would pass the textual `relative_to` check
because the symlink path is lexically under `criterion_dir`. On Python ≥ 3.13, `Path.rglob()`
no longer follows directory symlinks by default, so the test `test_out_of_tree_path_rejected`
passes in the dev environment (Python 3.14). On Python ≤ 3.12, `rglob` **does** follow
symlinks, and the guard fails — the malicious file is collected.

```python
# FIX: resolve est_path before the relative_to check
try:
    est_path.resolve().relative_to(criterion_dir)   # resolves symlinks
except ValueError:
    continue
```

Also add a comment that the guard's secondary defence for Python ≤ 3.12 requires resolution.

---

### WR-02: Python `run_functional_boxplot` times two depth computations — Python timing is ~2x inflated

**File:** `benchmarks/python_runner.py:239-244`

**Issue:** The timed closure `_run()` inside `run_functional_boxplot` calls **both**
`Boxplot(fd, depth_method=depth_fn, ...)` (which internally computes MBD) **and**
`depth_fn(fd)` separately to extract the fingerprint:

```python
def _run():
    bp = Boxplot(fd, depth_method=depth_fn, factor=1.5)  # computes MBD internally
    d = depth_fn(fd)                                      # computes MBD again
    return bp, d
```

The R side (`run_functional_boxplot`) calls `fbplot(fdata_obj, Depths="MBD", ...)` once and
extracts `bp_result$Depth` from the returned object — a single MBD computation. The Python
benchmark therefore includes approximately twice the depth work, making Python appear ~2x
slower than the equivalent R operation.

```python
# FIX: compute depth once and reuse
def _run():
    d = depth_fn(fd)
    bp = Boxplot(fd, depth_method=depth_fn, factor=1.5)
    return bp, d
# Or: use bp.depth_values if the API exposes pre-computed depths to avoid double work.
```

---

### WR-03: `check_agreement` returns `True` for unknown ops — fail-open for any future op typo

**File:** `benchmarks/gate.py:244-250`

**Issue:** The `else` branch in `check_agreement` silently returns `(True, 0.0, ...)` for any
`op` string that does not match the six known ops. While `run_gate` guards with
`if op not in _GATED_OPS: continue` before calling `check_agreement`, the `--dry-run` path in
`gate.py::main()` iterates `_GATED_OPS` directly and calls `check_agreement` — so the unknown
branch is unreachable today. However, `check_agreement` is an exported public function;
external callers or future code with an op-name typo would silently get `passed=True`. A
fail-closed approach is safer for a correctness gate.

```python
# FIX: raise an error for unknown ops
else:
    raise ValueError(
        f"check_agreement: unknown op={op!r}. "
        f"Valid ops: fpca, basis_smoothing, depth, functional_boxplot, fpcr, elastic"
    )
```

---

### WR-04: All-ops path in `run_all.py` always produces zero Python and R timings

**File:** `benchmarks/run_all.py:103-127, 130-140`

**Issue:** When `op=None` (i.e., `--op all`), both `step_python_runner` and `step_r_runner`
unconditionally pass `--smoke` to their respective runners regardless of the `smoke` parameter
value. In smoke mode `mean_ns=0.0` for every op. The `main()` call path:

```
main() → step_python_runner(op=None, smoke=False) → _run_all_ops_python() → [--smoke]
main() → step_r_runner(op=None, smoke=False) → [--smoke, ...]
```

Both always emit `--smoke`. The result is `timing_comparison.json` with `0.0` for all
`python_mean_ns` and `r_mean_ns` fields when running the default `--op all` mode. The display
formatter hides this by printing `"N/A"` for falsy values, but `0.0` is stored in the JSON and
will mislead downstream consumers.

The fix is either (a) loop over each op individually for the non-smoke all-ops path, or
(b) document that `--op all` is fingerprint-only (gate verification) and that individual
`--op <op>` runs are required for real timing:

```python
# FIX option A: in step_python_runner when op is None and not smoke:
for single_op in ALL_OPS:
    _run([_PYTHON, str(_SCRIPT_DIR / "python_runner.py"),
          "--op", single_op, "--n", str(n), "--m", str(m), "--seed", str(seed)],
         env=env)
```

---

## Info

### IN-01: R script in `verify_inputs.py` inlines the CSV path via f-string — injection risk on unusual paths

**File:** `benchmarks/verify_inputs.py:66-69`

**Issue:** `content_hash_r` builds an R script with the CSV path inlined via an f-string:

```python
r_script = f"""
df <- read.csv("{path}", check.names=FALSE)
```

If `path` contains a double-quote character or a backslash (common on Windows), the R script
will have a syntax error or parse a different path. On the current Linux-only environment with
auto-generated paths this is low-risk, but the harness claims cross-platform intent.

```python
# FIX: use shlex.quote to escape the path for R, or pass it via a temp env variable
import shlex
r_path_escaped = str(path).replace("\\", "\\\\").replace('"', '\\"')
r_script = f'df <- read.csv("{r_path_escaped}", check.names=FALSE)\n...'
```

---

### IN-02: `fpca_dump/src/main.rs` panics on `m=1` (integer division by zero in argvals)

**File:** `benchmarks/fpca_dump/src/main.rs:100`

**Issue:**

```rust
let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
```

When `m=1`, `(m-1)` is `0usize` and integer division triggers a panic in both debug and
release builds. The harness always passes `m=50` via the generator, so this is unreachable in
normal use, but defensive code would check `m >= 2` and either error or return `[0.0]`.

```rust
// FIX: guard before argvals computation
if m < 2 {
    eprintln!("fpca_dump: m must be >= 2, got {}", m);
    std::process::exit(1);
}
let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
```

---

### IN-03: `capture_env.py` uses deprecated `datetime.datetime.utcnow()`

**File:** `benchmarks/capture_env.py:171`

**Issue:**

```python
"timestamp": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
```

`datetime.datetime.utcnow()` is deprecated in Python 3.12 and will be removed in a future
release. The modern replacement is `datetime.datetime.now(datetime.timezone.utc)`.

```python
# FIX:
"timestamp": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
```

---

_Reviewed: 2026-09-22_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
