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
  critical: 1
  warning: 4
  info: 5
  total: 10
status: issues_found
---

# Phase 116: Code Review Report

**Reviewed:** 2026-09-25T00:00:00Z
**Depth:** standard
**Files Reviewed:** 10
**Status:** issues_found

## Summary

This phase adds `render_report.py` (new, ~1840 lines), a resumable/budgeted
chunk scheduler + grid-cell hardening layer in `run_all.py` (+1342 lines), the
`crosslang_grid.rs` bench file (new), and a handful of `collect_rust.py`
OP_MAP entries plus a noise-remediation tweak to `depth_benchmarks.rs`. The
new code is generally careful about fail-closed behavior (atomic rerun-log
writes, path-traversal guards in `load_estimates`/`iter_estimates`, explicit
`sys.exit` on malformed logs, list-form subprocess calls). However, one
pre-existing gate function that is now exercised through a *new* code path
has a genuine fail-open hole that can silently mark the FPCA/BENCH-05
"measure-first" numeric-agreement gate as passed when it never actually ran,
and the new `render_report.py` mixes non-comparable internal-only benchmarks
into a table that reads as cross-language evidence. Neither issue is
hypothetical: both are reachable through officially-documented Phase 116
workflows (`--grid --op <op>` and the standard rendering pipeline).

## Critical Issues

### CR-01: `step_run_gate` silently no-ops (fail-open) when an op's records are missing — newly reachable via `--grid --op <op>`

**File:** `benchmarks/run_all.py:518-523` and `benchmarks/run_all.py:557-585`

**Issue:**

```python
print("[6a] Agreement gate: fdars FPCA vs Python + R...")
py_fpca = next((r for r in py_records if r["op"] == "fpca"), None)
r_fpca = next((r for r in r_records if r["op"] == "fpca"), None)

fpca_passed = True
if py_fpca and r_fpca:
    ...
    if not p2:
        fpca_passed = False
```

`fpca_passed` is initialized to `True` *before* the guard. If either
`py_fpca` or `r_fpca` is `None` (i.e. no `"fpca"` record exists in
`python.json`/`r.json` for this call), the entire FPCA hard-gate block is
skipped — `fpca_passed` is never set to `False`, it just stays `True`. The
same pattern repeats for the other five ops at the step-6b loop:

```python
for op in ["basis_smoothing", "depth", "functional_boxplot", "fpcr", "elastic"]:
    py_rec = py_by_op.get(op)
    r_rec = r_by_op.get(op)
    if py_rec and r_rec:
        ...
        gate_records.append(ref_rec)
        gate_records.append(r_rec)
```

If a record for an op is absent, that op is simply never added to
`gate_records`, so `run_gate(gate_records)` never sees it — not "sees it and
fails", but literally never checks it. `gate_all_passed = fpca_passed and
all_passed_py_r` can therefore be `True` even though zero ops were actually
verified this call.

This was previously reachable only via the ad-hoc single-op path
(`python run_all.py --op depth`), but Phase 116 Plan 01 explicitly adds
`ops`-subset support to `run_grid()` (`run_all.py:434` diff,
`_run_all_ops_python(..., ops=ops, ...)`, `_run_r_with_native_fpca(...,
ops=ops, ...)`), which is documented as a supported workflow ("a single-op
re-run (e.g. `--op fpca`) does not re-time the other five ops"). Running
`python run_all.py --grid --op depth --cell 100x50` writes only a `"depth"`
record into `python.json`/`r.json` for that call, so `py_fpca`/`r_fpca` are
both `None`. `step_run_gate` then returns "passed" without ever running the
fdars-vs-R hard gate (`_FPCA_CORR_THRESHOLD`) that is supposed to prove fdars'
FPCA output is numerically correct — and `_write_cell_results` /
`_record_cell_status(..., "ok", ...)` persist the cell as gate-passed. This
directly threatens the BENCH-05 "measure-first" invariant the whole harness
is built around: a `--grid --op <single-op>` re-run can silently disable
agreement-checking for every op *not* in that call, not just skip it —
`gate_all_passed` is computed as if it had passed.

**Fix:** Track which ops were actually verifiable and fail closed if the
requested op set (or any op with data present) could not be checked:

```python
fpca_requested = any(r["op"] == "fpca" for r in py_records) or any(r["op"] == "fpca" for r in r_records)
fpca_passed = True
if fpca_requested:
    if not (py_fpca and r_fpca):
        print("GATE FAIL: fpca requested but python or r record missing", file=sys.stderr)
        fpca_passed = False
    else:
        ...

for op in [...]:
    py_rec, r_rec = py_by_op.get(op), r_by_op.get(op)
    op_requested = op in {r["op"] for r in py_records} or op in {r["op"] for r in r_records}
    if op_requested and not (py_rec and r_rec):
        print(f"GATE FAIL: {op} requested but python or r record missing", file=sys.stderr)
        all_passed_py_r = False
        continue
    if py_rec and r_rec:
        ...
```

At minimum, add a unit test for `step_run_gate` (there is currently none in
`test_run_all.py`) that passes an `ops` subset excluding `"fpca"` and asserts
the function does *not* report an unconditional pass.

## Warnings

### WR-01: `_match_rust_rec`'s "exact n/m match" can never succeed — rust.json carries no n/m fields

**File:** `benchmarks/run_all.py:600-617` (function unchanged by this diff, but directly relevant to the phase's "grid-aligned" timing-pairing goal)

**Issue:** `_match_rust_rec` documents itself as preferring "an exact n/m
match (parsed from the record's bench field) so that
`write_timing_comparison` does not associate Python/R timings for one grid
cell with Rust timings from a different cell":

```python
exact = [
    r for r in rust_records
    if r.get("op") == op and r.get("n") == n and r.get("m") == m
]
if exact:
    return exact[0]
return next((r for r in rust_records if r.get("op") == op), {})
```

But `collect_rust.py::collect()` never emits `n`/`m` keys on a record — its
schema is `{op, library, group, bench, mean_ns, ci_lower_ns, ci_upper_ns,
std_dev_ns, std_err_ns}` (see `collect_rust.py:174-186`). `r.get("n")` is
therefore always `None`, so `exact` is always empty and the function always
falls through to "first record whose op matches, regardless of size". Since
several bench files being collected (e.g. `depth_benchmarks.rs`'s
`functional_boxplot` group) sweep their own hardcoded `N` values ([50, 200,
500, 1000, 2300]) unrelated to the caller's `--n`/`--m`, the single-op path's
printed `rust=<ns>` line and the legacy `results/fpca_timing.json` can
silently report a completely different cell's fdars timing than the one the
user asked for.

**Fix:** Either parse `n`/`m` out of the criterion `bench` id (e.g.
`n100_m50` or `N/100`) in `collect_rust.py` and attach them to each record, or
have `_match_rust_rec` fall back to `{}` (an explicit "no rust timing for
this exact cell") instead of silently substituting an unrelated size.

### WR-02: `classify_missing` attributes a python/r missing row to "timeout" without checking the timed-out cell's `ops`

**File:** `benchmarks/render_report.py:548-557`

**Issue:**

```python
n, m = pair.get("n"), pair.get("m")
for rec in records:
    if (
        rec.get("kind") == "grid_cell"
        and _num_eq(rec.get("n"), n)
        and _num_eq(rec.get("m"), m)
        and rec.get("status") == "timeout"
    ):
        return {"status": "timeout", "detail": f"grid cell n={n}, m={m} timed out"}
```

`grid_cell` records carry an `ops` list (`run_all.py::_record_cell_status`),
but this lookup ignores it. If two separate `--grid --op <subset>` calls at
the same `(n, m)` produce one `ok` record for `ops=["fpca"]` and a later
`timeout` record for `ops=["elastic"]`, a missing-after row for an unrelated
op (say `"depth"`, never attempted in either call) will be classified as
"timeout" even though the actual reason it is missing is "not-run". This
mislabels the disclosure table (`not-rerun` block) that readers rely on to
understand why a number is absent from the published report.

**Fix:** Only match a `grid_cell` timeout record when `pair.get("op")` is a
member of `rec.get("ops", [])`.

### WR-03: `emit_before_after_crosslang_fdars` conflates true comparator benches with internal-only benchmarks sharing the same op label

**File:** `benchmarks/render_report.py:1194-1196`, `benchmarks/collect_rust.py:71-83`

**Issue:** `_paired_rust_rows` filters only on `pair.get("op") in
CROSSLANG_OPS`; it does not distinguish "this criterion id is the
grid-aligned comparator bench" from "this criterion id merely happens to
share the same shared op label". `collect_rust.py`'s `OP_MAP` maps a long
list of internal-only, N×M-swept perf-hotpath benchmarks —
`audit_p3_elastic_cross`, `audit_p3_elastic_cross_banded`,
`audit_p3_elastic_self`, `audit_p3_elastic_self_banded`, `audit_elastic`,
`audit_p5_karcher_paybackN`, `audit_karcher`, `audit_p3_karcher`,
`audit_p3_karcher_banded`, `karcher_mean`, `elastic_align_pair`,
`audit_p4_elastic_fpca` — to the same `"elastic"` op used by the new
grid-aligned `crosslang_elastic` bench. None of those internal-only
benchmarks (self/cross distance-matrix computations, Karcher-mean payback
studies, etc.) have any scikit-fda/R equivalent, yet because `op == "elastic"
in CROSSLANG_OPS`, their before/after rows land in
`emit_before_after_crosslang_fdars`'s table with `title=None` — unlike
`emit_before_after_internal`, which is explicitly titled "fdars-internal
before/after (no scikit-fda/R comparator)". A reader of the generated
PERFORMANCE.md section could reasonably interpret every row in the
"crosslang-fdars" table as being part of the head-to-head evidence, when most
of the `elastic`-labeled rows in the committed baseline are not.

**Fix:** Restrict `emit_before_after_crosslang_fdars` to rows whose criterion
id actually starts with `crosslang_` (the true grid-aligned comparator
benches), or add a per-row "comparable" column driven by whether a
Python/R counterpart exists at the same `(op, n, m)`.

### WR-04: `step_run_gate` has zero direct unit-test coverage

**File:** `benchmarks/test_run_all.py` (whole file)

**Issue:** `test_run_all.py` has extensive coverage of the chunk manifest,
rerun log, scheduler, grid-cell hardening, and `--full` sequencing, but no
`TestStepRunGate`-equivalent class exists. `TestGridCellHardening` mocks
`step_run_gate` out entirely rather than exercising its logic. Given
`step_run_gate` is the single function implementing the BENCH-05
"measure-first" invariant this whole harness is designed around (see CR-01),
its complete absence from the test suite is a significant gap — the
fail-open bug in CR-01 would have been caught by a single test asserting that
an `ops` subset excluding `"fpca"` does not report an unconditional gate
pass.

**Fix:** Add unit tests for `step_run_gate` covering: (a) all ops present and
agreeing → pass; (b) FPCA disagreement → fail; (c) an `ops` subset missing a
record for one op → must not silently report an overall pass.

## Info

### IN-01: `smoke` parameter is dead code — no CLI flag ever sets it `True`

**File:** `benchmarks/run_all.py:324-434` (`step_python_runner`, `step_r_runner`)

**Issue:** Both functions accept `smoke: bool = False` and branch on it, but
`main()`'s `argparse` setup defines no `--smoke` flag, and every call site
(`step_python_runner(op=op, n=args.n, m=args.m, seed=args.seed, env=env)` and
the analogous R call) omits the `smoke` argument. The `if smoke:` branches in
both functions are therefore unreachable from the CLI.

**Fix:** Either wire up a `--smoke` flag in `main()`'s argparse, or remove the
dead branches and the `smoke` parameter.

### IN-02: Falsy-check hides a legitimate `0.0` timing as "N/A"

**File:** `benchmarks/run_all.py:672-676`

**Issue:**

```python
py_t = f"{row['python_mean_ns']:.0f}" if row["python_mean_ns"] else "N/A"
```

Uses truthiness rather than `is not None`. A `mean_ns` of exactly `0.0` (the
value the now-dead `--smoke` path used to write, per the WR-04-adjacent
comment "the previous implementation passed --smoke, which sets mean_ns=0.0
for every op") would print as `"N/A"` instead of `"0"`. Purely a console
diagnostics cosmetic issue (the underlying JSON is written correctly), but
worth fixing for consistency with the `is not None` checks used everywhere
else in this file (e.g. `_safe_float`).

**Fix:** `if row["python_mean_ns"] is not None else "N/A"`.

### IN-03: `parse_criterion_settings` silently keeps only the last occurrence when a group name repeats

**File:** `benchmarks/render_report.py:745-764`

**Issue:** `_GROUP_BLOCK_RE.finditer(text)` matches every
`benchmark_group("<name>")` occurrence in a bench file and assigns
`result[group] = {...}` per match — a later match with the same group name
silently overwrites an earlier one. Not triggered by the two files currently
reviewed (each group name appears once), but fragile: if a future bench file
defines the same `benchmark_group` literal from two different functions (a
common criterion pattern when splitting a large group across helper
functions), `emit_criterion_settings` would silently report only the last
one's `sample_size`/`measurement_time`/`warm_up_time`.

**Fix:** Merge (or assert-uniqueness) instead of blind overwrite.

### IN-04: `check_prose`'s number-detection regex misses numbers followed directly by punctuation

**File:** `benchmarks/render_report.py:1599`

**Issue:** `_NUMBER_UNIT_RE = re.compile(r"(\d[\d,]*\.?\d*)\s*(?:×|[xX]\s|ms\b)")`
requires whitespace immediately after a bare `x`/`X` (`[xX]\s`). A sentence
like `"...a 10.00x, unverified figure"` (comma directly after `x`, no space)
would not be flagged as an unbacked number, silently defeating the
"prose can only quote numbers that exist in a generated table" guard for that
phrasing.

**Fix:** Use a word-boundary based pattern, e.g. `[xX]\b`, instead of
requiring a literal following whitespace character.

### IN-05: `emit_scaling`'s noise-dagger applies to every exponent for an op even when only one unrelated cell is noisy

**File:** `benchmarks/render_report.py:1303-1315`

**Issue:**

```python
noisy_any = any(
    rust_grid[(n, m)][1] != "PASS"
    for n in GRID_N for m in GRID_M
    if (n, m) in rust_grid
)
...
def _fmt_exp(value):
    ...
    return text + _NOISE_DAGGER if noisy_any else text
```

`noisy_any` is computed once per op over the *entire* 3×2 grid and then
applied uniformly to every one of the five reported exponents (`N@M=50`,
`N@M=200`, `M@N=100`, `M@N=300`, `M@N=500`). An exponent fit from two clean
cells will still be daggered as "possibly noisy" merely because a third,
unrelated cell for the same op failed the noise gate. This over-flags rather
than under-flags (safe direction), but reduces the report's ability to tell
readers which specific numbers are actually suspect.

**Fix:** Compute per-exponent noise flags from only the cells that feed that
specific exponent's `fit_scaling_exponent` call.

---

_Reviewed: 2026-09-25T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
