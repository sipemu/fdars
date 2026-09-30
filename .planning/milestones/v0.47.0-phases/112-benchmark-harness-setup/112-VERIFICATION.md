---
phase: 112-benchmark-harness-setup
verified: 2026-09-22T00:00:00Z
status: passed
score: 5/5 must-haves verified
covered_files:
  - .planning/REQUIREMENTS.md
  - .planning/phases/112-benchmark-harness-setup/112-01-PLAN.md
  - .planning/phases/112-benchmark-harness-setup/112-01-SUMMARY.md
  - .planning/phases/112-benchmark-harness-setup/112-02-PLAN.md
  - .planning/phases/112-benchmark-harness-setup/112-02-SUMMARY.md
  - .planning/phases/112-benchmark-harness-setup/112-03-PLAN.md
  - .planning/phases/112-benchmark-harness-setup/112-03-SUMMARY.md
  - benchmarks/capture_env.py
  - benchmarks/collect_rust.py
  - benchmarks/gate.py
  - benchmarks/generate_data.py
  - benchmarks/python_runner.py
  - benchmarks/r_runner.R
  - benchmarks/renv.lock
  - benchmarks/requirements.lock
  - benchmarks/run_all.py
  - benchmarks/test_collect_rust.py
  - benchmarks/test_gate.py
  - benchmarks/verify_inputs.py
  - fdars-core/benches/depth_benchmarks.rs
covered_digest: "v1:sha256:9531f21b6c18e9f7ad007c0fba86eb0c3b81d0be7d725a65d7ccd34fe11258d6"
behavior_unverified: 0
overrides_applied: 0
re_verification:
  reason: "Two re-blessings of the 5/5-passed verification, neither a regression: (1) 2026-09-22 — post-verification code-review fix pass (CR-01/CR-02/WR-01..WR-04) strengthened correctness (R timing ns units, non-zero all-ops timings, fail-closed gate), re-reviewed clean. (2) 2026-09-23 — Phase 113 legitimately evolved the shared benchmarks/ harness (gate.py recalibration, run_all.py --grid mode, collect_rust.py std_dev fix, r_runner.R native pca.fd), which are in 112's covered_files, so the digest was recomputed to the current tree. Phase 112's goal (a reproducible external harness + numeric-agreement gate) remains satisfied; 5/5 must-haves still hold."
  digest_updated: "2026-09-23"
---

# Phase 112: Benchmark Harness Setup Verification Report

**Phase Goal:** A reproducible external harness exists that times fdars, scikit-fda, and the R FDA ecosystem on identical inputs, with fairness controls and a numeric-agreement gate wired in before the first measurement.
**Verified:** 2026-09-22
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | The seeded generator emits CSVs that the Python, R, and Rust sides all read as bit-identical bytes (BENCH-01) | VERIFIED | `verify_inputs.py` exits 0; all three fixtures produce matching SHA-256 hashes across regeneration and cross-runner loading. Confirmed live. |
| 2 | The Python (scikit-fda 0.10.1) and R runners execute all six overlapping ops (FPCA, basis smoothing, depth, functional boxplot, FPCR, elastic) and emit structured timing+result records (BENCH-02) | VERIFIED | `python_runner.py` defines `ALL_OPS = ["fpca","basis_smoothing","depth","functional_boxplot","fpcr","elastic"]` with a branch per op. `r_runner.R` mirrors the same six. Both files are substantive (15 KB+ each). |
| 3 | fdars timings are parsed from existing criterion `new/estimates.json` into the shared comparison format with no new dev-dependency (BENCH-03) | VERIFIED | `collect_rust.py` uses `OP_MAP` mapping all criterion group names (e.g. `audit_fpca`, `fregre_lm`, `elastic_align_pair`) to shared op labels. No new entry in `fdars-core/Cargo.toml` dev-dependencies. `test_collect_rust.py` — 7 tests pass. |
| 4 | env.json captures all six thread-env vars pinned to 1 plus non-empty CPU/BLAS/version metadata (BENCH-04) | VERIFIED | `run_all.py --env-only` exits 0. `env.json` asserted: `OMP_NUM_THREADS=1`, `RAYON_NUM_THREADS=1`, `scikit-fda=0.10.1`, `r.blas=/usr/lib/libblas.so.3.12.0`, `cpu.model=13th Gen Intel Core i9-13900H`. All six thread-env keys confirmed "1". |
| 5 | The numeric-agreement gate verifies every overlapping op with its per-op tolerance and ABORTS before any timing is recorded on mismatch (BENCH-05) | VERIFIED | `gate.py --self-test` exits 0: pass-on-agree AND fail-on-mismatch proven for all ops. `run_all.py` calls `step_run_gate` (line 432) before `write_timing_comparison` (line 435). `test_gate.py` — 18 tests pass. Three ops (depth, functional_boxplot, fpcr) currently fail their tolerances due to cross-library convention differences; user approved deferring tolerance calibration to Phase 113 per verified-context deferral. The gate exists, runs first, and discriminates — BENCH-05 satisfied. |

**Score:** 5/5 truths verified (0 present, behavior-unverified)

### Deferred Items

Items not yet met but explicitly addressed in later milestone phases.

| # | Item | Addressed In | Evidence |
|---|------|-------------|----------|
| 1 | Per-op tolerance calibration for depth, functional_boxplot, and fpcr | Phase 113 | User-approved deferral 2026-09-22; documented in 112-03-SUMMARY.md "Phase 113 hand-off" and .planning/STATE.md Pending Todos. Gate is fail-closed; Phase 113 must calibrate before those three ops produce a committed baseline. |
| 2 | BENCH-01 checkbox in REQUIREMENTS.md remains `[ ]` (unchecked) | Phase 112 closeout | BENCH-01 implementation is verified (no Cargo.toml in benchmarks/, seeded generator, bit-identical CSVs). The planning-doc checkbox was never ticked. This is a documentation inconsistency, not an implementation gap. |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `benchmarks/generate_data.py` | Seeded CSV generator | VERIFIED | 5,890 bytes; PCG-64 RNG, %.17g precision, sinusoidal curves |
| `benchmarks/python_runner.py` | scikit-fda 0.10.1 runner, 6 ops | VERIFIED | 15,762 bytes; all six ops wired with `ALL_OPS` list |
| `benchmarks/r_runner.R` | R ecosystem runner, 6 ops | VERIFIED | 15,562 bytes; all six ops wired |
| `benchmarks/collect_rust.py` | Criterion output parser | VERIFIED | 8,614 bytes; `OP_MAP` covers all overlapping groups; path-traversal guard present |
| `benchmarks/gate.py` | Per-op numeric agreement gate | VERIFIED | 20,301 bytes; per-op dispatch for all 6 ops; `run_gate` fails closed |
| `benchmarks/capture_env.py` | env.json fairness capture | VERIFIED | 6,732 bytes; captures CPU/BLAS/thread-env/versions |
| `benchmarks/run_all.py` | Orchestrator with gate-before-timing | VERIFIED | 16,434 bytes; `step_run_gate` (line 432) before `write_timing_comparison` (line 435) |
| `benchmarks/verify_inputs.py` | Bit-identical CSV spot-check | VERIFIED | 8,232 bytes; checks all 3 fixture types |
| `benchmarks/test_gate.py` | Pytest gate unit tests | VERIFIED | 15,514 bytes; 18 tests, all pass |
| `benchmarks/test_collect_rust.py` | Pytest collector unit tests | VERIFIED | 7,361 bytes; 7 tests, all pass |
| `benchmarks/requirements.lock` | Pinned Python deps | VERIFIED | Contains `scikit-fda==0.10.1` |
| `benchmarks/renv.lock` | Pinned R deps | VERIFIED | Contains fda, fda.usc, roahd, fdasrvf, bench, renv, refund, fdapace |
| `fdars-core/benches/depth_benchmarks.rs` | `bench_functional_boxplot` group | VERIFIED | Function defined at line 348; registered in `criterion_group!` at line 380; compiles clean (`--no-run` exit 0) |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `generate_data.py` CSV | `python_runner.py`, `r_runner.R`, Rust runners | `%.17g` precision CSV; `verify_inputs.py` SHA-256 hash equality | VERIFIED | All three fixture files confirmed bit-identical across runners by live `verify_inputs.py` run |
| `gate.py` FPCA sign-alignment | `fdars fix_svd_signs` convention | `np.dot(l1, l2) < 0 → flip` mirroring `regression.rs:288-301` | VERIFIED | `gate.py` line 88: `if np.dot(l1, l2) < 0: l2 = -l2`. Matches the `dominant_sign_negative` dot-product test in `regression.rs:266-276`. |
| `run_all.py` thread pinning | All subprocess calls | `THREAD_ENV` dict with `OMP/OPENBLAS/MKL/RAYON/NUMEXPR/VECLIB=1` passed via `_run(cmd, env=env)` | VERIFIED | `_run()` helper at line 82 passes `env=env` to all `subprocess.run` calls; no `shell=True` present |
| `run_all.py` gate ordering | No timing written on gate failure | `step_run_gate` (line 432) before `write_timing_comparison` (line 435); `sys.exit(1)` on gate failure | VERIFIED | Line-number ordering confirmed; gate failure prints `GATE FAIL — aborting. No timing comparison written.` |
| `collect_rust.py` | `target/criterion/*/new/estimates.json` | `rglob("new/estimates.json")` with `relative_to(CRITERION_DIR)` guard | VERIFIED | Path-traversal guard at line 98; `base/` excluded by glob pattern targeting only `new/` |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|-------------------|--------|
| `gate.py` `check_fpca_agreement` | `fdars_loadings`, `other_loadings` | Loaded from `results/python.json`, `results/r.json`, `fpca_dump` binary output | Yes — runtime data from runners | FLOWING |
| `run_all.py` `write_timing_comparison` | `timing_rows` | Reads `python.json`, `r.json`, `rust.json` produced by runners and collector | Yes — only written post-gate-pass | FLOWING |
| `collect_rust.py` `iter_estimates` | `mean_ns` | `data["mean"]["point_estimate"]` from parsed `estimates.json` | Yes — real criterion output | FLOWING |
| `capture_env.py` | `thread_env` | `os.environ.get("OMP_NUM_THREADS", "1")` etc. | Yes — live env at capture time | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Pytest gate unit tests (18 tests) | `benchmarks/.venv/bin/python -m pytest benchmarks/test_gate.py benchmarks/test_collect_rust.py -q` | 25 passed in 0.85s | PASS |
| Gate self-test: pass-on-agree AND fail-on-mismatch | `benchmarks/.venv/bin/python benchmarks/gate.py --self-test` | `GATE SELF-TEST: PASSED (all ops: pass-on-agree and fail-on-mismatch verified)` | PASS |
| Bit-identical CSV spot-check | `benchmarks/.venv/bin/python benchmarks/verify_inputs.py` | `BIT-IDENTITY CHECK: PASSED` for all 3 fixtures | PASS |
| env.json correctness assertion | Python one-liner checking thread_env, cpu.model, scikit-fda==0.10.1, r.blas | `env.json assertions: PASS` | PASS |
| bench_functional_boxplot compiles | `cargo bench -p fdars-core --features linalg,parallel --bench depth_benchmarks --no-run` | `Finished bench profile [optimized]` | PASS |

### Requirements Coverage

| Requirement | Source Plan(s) | Description | Status | Evidence |
|-------------|---------------|-------------|--------|----------|
| BENCH-01 | 112-01-PLAN | External benchmarks/ harness, seeded generator, bit-identical CSVs | SATISFIED | benchmarks/ has no Cargo.toml; verify_inputs.py PASS; no fdars-core dep in Python/R lockfiles. Note: REQUIREMENTS.md checkbox still `[ ]` — documentation inconsistency, not implementation gap. |
| BENCH-02 | 112-01-PLAN, 112-02-PLAN | Python + R runners covering 6 overlapping ops | SATISFIED | Both runners define all 6 ops; python_runner.py `ALL_OPS` list; r_runner.R `ALL_OPS` vector confirmed |
| BENCH-03 | 112-01-PLAN, 112-03-PLAN | fdars timings from existing criterion harness, no new dep | SATISFIED | `OP_MAP` in collect_rust.py; criterion 0.5 remains the only dev-dep; test_collect_rust.py 7 tests pass |
| BENCH-04 | 112-01-PLAN, 112-03-PLAN | Fairness controls: thread pinning + env.json | SATISFIED | All six thread-env vars = "1"; env.json complete with CPU/BLAS/version; run_all.py --env-only exits 0 |
| BENCH-05 | 112-01-PLAN, 112-03-PLAN | Numeric-agreement gate before timing; fail-closed | SATISFIED | gate.py --self-test PASS; 18 pytest tests pass; run_all.py orchestration gate-before-timing confirmed by line numbers; three-op deferral user-approved per verification context |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `benchmarks/capture_env.py` | 171 | `datetime.datetime.utcnow()` deprecated in Python 3.14 | Info | Deprecation warning only; no correctness impact; env.json written correctly |

No TBD, FIXME, or XXX markers found in any file modified by this phase.

### Constraint Verification

| Constraint | Status | Evidence |
|------------|--------|----------|
| No change to `fdars-core/src/` | VERIFIED | `git log --since=2026-09-19 -- fdars-core/src/` returns no commits |
| No change to `fdars-core/Cargo.toml` dependencies | VERIFIED | `git log --since=2026-09-19 -- fdars-core/Cargo.toml` returns no commits; dev-deps unchanged (criterion 0.5, dhat, serde, serde_json) |
| `benchmarks/` NOT in Cargo workspace members | VERIFIED | Root `Cargo.toml` `[workspace] members = ["fdars-core"]` only; no Cargo.toml in benchmarks/ |
| No new crate dependency | VERIFIED | No additions to fdars-core/Cargo.toml; `fpca_dump` is a standalone crate with its own `[workspace]` opt-out, not a workspace member |

### Human Verification (Already Approved)

Both blocking-human gates (Plan 01 Task 4 and Plan 03 Task 4) were explicitly approved by the user:

- **Plan 01 Task 4** — approved 2026-09-20: CSV bit-identity PASS, FPCA gate min|corr|=0.998271 (fdars-vs-python) and 1.000000 (fdars-vs-R) both > 0.99, env.json fairness fields accepted.
- **Plan 03 Task 4** — approved 2026-09-22: Full-op verify_inputs PASS, per-op gate metrics reviewed, three op failures (depth/functional_boxplot/fpcr) accepted as cross-library convention differences with tolerance calibration deferred to Phase 113, env.json accepted.

---

_Verified: 2026-09-22_
_Verifier: Claude (gsd-verifier)_

## Re-fingerprint 2026-09-25

Covered-file digest recomputed by the v0.47.0 autonomous orchestrator before the milestone audit. Cause of staleness: harness files evolved by design in Phases 113/116 (gate recalibration, crosslang ops, scheduler); full benchmarks pytest suite green (209). No verified truth was invalidated.

## Re-fingerprint 2026-09-25 (post-Phase-117)

Covered-file digest recomputed before the milestone audit. Changed covered files: only .planning/REQUIREMENTS.md changed (milestone requirements marked complete) — false-stale. No verified truth was invalidated.
