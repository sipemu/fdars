---
phase: "112"
slug: "benchmark-harness-setup"
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
# audit-milestone §5.5 distinguishes NOT-VALIDATED (draft) from PARTIAL (validated + nyquist_compliant: false) (#2117)
status: draft
nyquist_compliant: false
wave_0_complete: false
created: "2026-09-20"
---

# Phase 112 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> This is an EXTERNAL tooling phase (workspace-root `benchmarks/`, NOT a Cargo member): "tests"
> are the numeric-agreement gate, the bit-identical-input spot-check, and structural sanity checks
> on generated artifacts. `fdars-core` is not modified except for adding external-agnostic criterion
> bench functions (`fregre_lm` output group, `functional_boxplot`), which are covered by the existing
> `cargo test` / `cargo bench --no-run` gates.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Harness self-checks: Python `pytest`-optional + a `verify_agreement` gate script; Rust criterion 0.5 (existing); shell spot-check |
| **Config file** | `benchmarks/requirements.lock`, `benchmarks/renv.lock` (Wave 0 installs) |
| **Quick run command** | `python benchmarks/verify_inputs.py` (bit-identical CSV spot-check across the 3 runners) |
| **Full suite command** | `python benchmarks/run_agreement_gate.py` (numeric-agreement gate over the overlapping op set) |
| **Estimated runtime** | ~60–120 seconds (agreement gate on small N×M grid) |

---

## Sampling Rate

- **After every task commit:** Run the relevant quick check (input spot-check or a single-op agreement check)
- **After every plan wave:** Run the full numeric-agreement gate
- **Before `/gsd-verify-work`:** Agreement gate must pass (or explicitly flag) for every overlapping op; `env.json` populated
- **Max feedback latency:** ~120 seconds

---

## Per-Task Verification Map

> Seeded by plan-phase; the planner refines Task IDs to match the emitted PLAN.md waves. Every
> BENCH requirement has an observable automated check below.

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 112-01-01 | 01 | 1 | BENCH-01 | — | N/A | integration | `python benchmarks/verify_inputs.py` | ❌ W0 | ⬜ pending |
| 112-02-01 | 02 | 2 | BENCH-02 | — | N/A | integration | `python benchmarks/runners/py_runner.py --smoke && Rscript benchmarks/runners/r_runner.R --smoke` | ❌ W0 | ⬜ pending |
| 112-03-01 | 03 | 2 | BENCH-03 | — | N/A | integration | `python benchmarks/collect_criterion.py --check` | ❌ W0 | ⬜ pending |
| 112-04-01 | 04 | 3 | BENCH-04 | — | N/A | integration | `test -s benchmarks/results/env.json && python -c "import json;d=json.load(open('benchmarks/results/env.json'));assert d['threads']=='1'"` | ❌ W0 | ⬜ pending |
| 112-05-01 | 05 | 3 | BENCH-05 | — | N/A | integration | `python benchmarks/run_agreement_gate.py` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `benchmarks/requirements.lock` — pin scikit-fda 0.10.1 + numpy/scipy (Python venv)
- [ ] `benchmarks/renv.lock` — pin `fda`/`fda.usc`/`refund`/`fdapace`/`roahd`/`fdasrvf` (install missing: `renv`, `bench`, `refund`, `fdapace`)
- [ ] Shared CSV schema + seeded generator (single source of truth) before any runner consumes data
- [ ] Add criterion bench functions for `fregre_lm` (ensure `target/criterion` output) and `functional_boxplot` in existing `fdars-core/benches/*.rs`

*If none: "Existing infrastructure covers all phase requirements."*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| BLAS-backend detection in `env.json` | BENCH-04 | Backend string depends on host NumPy/R build; cannot assert a fixed value in CI | Inspect `benchmarks/results/env.json` — confirm `blas_backend`, `cpu`, versions, and `*_NUM_THREADS=1` are all captured and non-empty |

*If none: "All phase behaviors have automated verification."*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 120s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
