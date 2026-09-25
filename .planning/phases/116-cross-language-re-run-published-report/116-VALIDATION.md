---
phase: "116"
slug: "cross-language-re-run-published-report"
status: validated
nyquist_compliant: true
wave_0_complete: true
created: "2026-09-25"
---

# Phase 116 — Validation Strategy

> Per-phase validation contract (reconstructed post-execution by validate-phase, State B).

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (benchmarks/.venv, Python 3.14.7) + cargo clippy/fmt for bench code |
| **Config file** | none (pytest default discovery in `benchmarks/`) |
| **Quick run command** | `benchmarks/.venv/bin/python -m pytest benchmarks/ -q` |
| **Full suite command** | `benchmarks/.venv/bin/python -m pytest benchmarks/ -q && benchmarks/.venv/bin/python benchmarks/render_report.py --check && benchmarks/.venv/bin/python benchmarks/render_report.py --check-prose && benchmarks/.venv/bin/python benchmarks/render_report.py --ab-check && benchmarks/.venv/bin/python benchmarks/render_report.py --noise-report && cargo clippy --all-targets --features linalg,parallel -- -D warnings` |
| **Estimated runtime** | ~5 seconds (pytest + report checks); clippy incremental ~1 s |

---

## Sampling Rate

- **After every task commit:** Run the quick run command
- **After every plan wave:** Run the full suite command
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 116-01-01 | 01 | 1 | RPT-01 | T-116 (tracer) | fail-closed gate before timing | integration | `pytest benchmarks/test_collect_rust.py benchmarks/test_render_report.py -q` | ✅ | ✅ green |
| 116-01-02 | 01 | 1 | RPT-01 | baseline freeze | baseline byte-identical to Phase 113 | integrity | `cmp <(git show 597f857d:benchmarks/results/comparison.csv) benchmarks/results/comparison_baseline.csv` | ✅ | ✅ green |
| 116-02-01 | 02 | 2 | RPT-01 | — | N/A | build/lint | `cargo clippy --all-targets --features linalg,parallel -- -D warnings` | ✅ | ✅ green |
| 116-02-02 | 02 | 2 | RPT-01 | — | N/A | unit (TDD) | `pytest benchmarks/test_collect_rust.py -q` (9 passed) | ✅ | ✅ green |
| 116-03-01 | 03 | 3 | RPT-01 | T-116 scheduler | atomic fail-closed run log; timeouts never re-queued | unit (TDD) | `pytest benchmarks/test_run_all.py -q` (71 passed) | ✅ | ✅ green |
| 116-03-02 | 03 | 3 | RPT-01 | — | N/A | unit (TDD) | `pytest benchmarks/test_run_all.py -q` | ✅ | ✅ green |
| 116-04-01 | 04 | 2 | RPT-01 | — | N/A | unit (TDD) | `pytest benchmarks/test_render_report.py -q` (69 passed) | ✅ | ✅ green |
| 116-04-02 | 04 | 2 | RPT-01 | — | N/A | unit (TDD) + CLI | `render_report.py --check` / `--check-prose` exit 0 | ✅ | ✅ green |
| 116-05-01 | 05 | 4 | RPT-01 | T-116-18/19 | locked deps; all rounds recorded | evidence check | `render_report.py --ab-check` exit 0 (phase116 entry eligible) | ✅ | ✅ green |
| 116-05-02 | 05 | 4 | RPT-01 | T-116-17 | scratch worktrees removed; main checkout untouched | integrity | `git diff --quiet HEAD -- fdars-core/src fdars-core/benches/perf_hotpaths.rs` | ✅ | ✅ green |
| 116-06-01 | 06 | 5 | RPT-01 | T-116-21 | every chunk has final status | CLI | `run_all.py --rust-status` exit 0 | ✅ | ✅ green |
| 116-06-02 | 06 | 5 | RPT-01 | T-116-22 | gate unchanged + passed before timing | integrity | `git diff --name-only ec57fa01 HEAD -- benchmarks/gate.py` empty | ✅ | ✅ green |
| 116-06-03 | 06 | 5 | RPT-01 | T-116-20 | only fresh measurements | CLI | freshness verify (36 crosslang rows, no orphans/stale) + `render_report.py --noise-report` exit 0 | ✅ | ✅ green |
| 116-07-01 | 07 | 6 | RPT-01 | — | N/A | CLI | `render_report.py --check` + `--check-prose` exit 0 (also in fresh clone) | ✅ | ✅ green |
| 116-07-02 | 07 | 6 | RPT-01 | — | N/A | CLI | `pytest benchmarks/ -q` (209 passed) | ✅ | ✅ green |
| review-fix | — | — | RPT-01 | CR-01 | missing hard-gated record fails closed | unit | `pytest benchmarks/test_run_all.py -k StepRunGate -q` | ✅ | ✅ green |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

Existing infrastructure covers all phase requirements (pytest suites in `benchmarks/` pre-existed from Phases 112–113; this phase added `test_render_report.py` and `test_run_all.py`).

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Physical re-measurement of all timings | RPT-01 | ~3 h wall-clock, needs a quiet machine + pinned Python/R toolchains; timing values are inherently non-deterministic | Follow `documentation/PERFORMANCE.md` → Reproduction (`benchmarks/run_all.py --full`), then `render_report.py`; compare within reported CIs |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 10s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-09-25

## Validation Audit 2026-09-25
| Metric | Count |
|--------|-------|
| Gaps found | 0 |
| Resolved | 0 |
| Escalated | 0 |
