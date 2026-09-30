---
phase: "113"
slug: "cross-language-baseline-hotspot-ranking"
status: draft
nyquist_compliant: false
wave_0_complete: false
created: "2026-09-22"
---

# Phase 113 — Validation Strategy

> Per-phase validation contract. EXTERNAL measurement/ranking phase (reuses Phase 112's
> `benchmarks/` harness). "Tests" = the recalibrated agreement gate passing for all six ops, the
> noise gate correctly flagging high-RSD entries, a committed `comparison.csv`, and a deterministic
> ranked hotspot list. No `fdars-core/src` change (only criterion bench runs / external-agnostic
> parameterization).

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (gate + ranking unit tests) + the harness scripts + criterion 0.5 (existing) |
| **Config file** | reuse `benchmarks/requirements.lock` / `renv.lock` |
| **Quick run command** | `benchmarks/.venv/bin/python -m pytest benchmarks/test_gate.py -q` |
| **Full suite command** | `benchmarks/.venv/bin/python benchmarks/run_all.py --op all` (gate must pass for all 6 ops, then produce comparison.csv) |
| **Estimated runtime** | full baseline run is minutes (criterion benches at N≥100) |

---

## Sampling Rate

- **After every task commit:** run the relevant gate/ranking pytest.
- **After every plan wave:** run the recalibrated gate over all six ops (must pass) + regenerate the ranking.
- **Before `/gsd-verify-work`:** `comparison.csv` committed, ranked hotspot list committed, noise gate applied, all six ops pass the (recalibrated) agreement gate.
- **Max feedback latency:** ~120 s for the gate/ranking unit tests (the full timed baseline is the heavier, wave-level run).

---

## Per-Task Verification Map

> Seeded by plan-phase; planner refines Task IDs to its waves.

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 113-01-01 | 01 | 1 | BASE-01 | unit | `benchmarks/.venv/bin/python -m pytest benchmarks/test_gate.py -q` (recalibrated depth/boxplot/fpcr → correlation) | ❌ W0 | ⬜ pending |
| 113-02-01 | 02 | 2 | BASE-01 | integration | `test -s benchmarks/results/comparison.csv && head -1 benchmarks/results/comparison.csv | grep -q rsd_pct` | ❌ W0 | ⬜ pending |
| 113-03-01 | 03 | 3 | BASE-02 | unit+integration | ranked hotspot list is deterministic from comparison.csv and confirms the elastic anchor | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] Generate the two missing criterion outputs: `cargo bench -p fdars-core --features linalg,parallel --bench regression_benchmarks -- fregre_lm` and `--bench depth_benchmarks -- functional_boxplot`.
- [ ] Recalibrate `benchmarks/gate.py` depth/functional_boxplot/FPCR → Pearson/rank correlation ≥0.99 (FPCR threshold per planner per RESEARCH A2) + update `benchmarks/test_gate.py`.

*If none: "Existing infrastructure covers all phase requirements."*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| The ranked hotspot list is a *fair, honest* reflection of the numbers | BASE-02 | "Fair/honest" is a judgment on methodology disclosure, not a fixed assertion | Human-verify: the ranking confirms/refutes the elastic anchor with fresh N≥100 numbers, noise-gated entries are disclosed, and the leverage formula is stated |

*If none: "All phase behaviors have automated verification."*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 120s (unit tests)
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
