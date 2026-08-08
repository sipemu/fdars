---
phase: 6
slug: conditional-svd-library-comparison
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
# audit-milestone §5.5 distinguishes NOT-VALIDATED (draft) from PARTIAL (validated + nyquist_compliant: false) (#2117)
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-08-08
---

# Phase 6 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> **Audit-milestone note:** This phase produces measurement artifacts + report text, not shipped `fdars-core` behavior. "Validation" here means: the comparison bench compiles under `linalg`, runs on the `/release/` binary, produces feature-tagged reproducible artifacts under `.planning/research/bench/`, and the numerical-equivalence check passes so the nalgebra-vs-faer comparison is apples-to-apples.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | criterion 0.5 (bench harness) + built-in `#[test]` (numerical-equivalence check) |
| **Config file** | `fdars-core/Cargo.toml` (`[[bench]] audit_hotpaths, harness = false`) |
| **Quick run command** | `cargo test -p fdars-core --features linalg svd_equivalence` |
| **Full suite command** | `cargo bench -p fdars-core --features linalg --bench audit_hotpaths -- bench_p6_svd_comparison` |
| **Estimated runtime** | ~2–6 min for the bench group (M=500 cell dominates; sample_size reduced per Pitfall F) |

---

## Sampling Rate

- **After every task commit:** Run the equivalence unit test (`cargo test ... svd_equivalence`) — fast (<10s).
- **After the bench task:** Run `bench_p6_svd_comparison` and confirm artifacts land under `.planning/research/bench/p6_*` with two runs for variance (±5% rule, Pitfall 7).
- **Before verification:** Numerical-equivalence test green; both `p6_*_run1.txt` and `p6_*_run2.txt` present and feature-tagged; AUDIT-REPORT Phase-6 section transcribes on-disk numbers.
- **Max feedback latency:** ~10 s (equivalence test); bench artifacts are the slow, deliberate signal.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 6-01-01 | 01 | 0 | PERF-06 | — | N/A (audit) | unit | `cargo test -p fdars-core --features linalg svd_equivalence` | ❌ W0 | ⬜ pending |
| 6-01-02 | 01 | 1 | PERF-06 | — | N/A (audit) | bench | `cargo bench -p fdars-core --features linalg --bench audit_hotpaths -- bench_p6_svd_comparison` | ❌ W0 | ⬜ pending |
| 6-01-03 | 01 | 1 | PERF-06 | — | N/A (audit) | manual | Inspect `.planning/research/bench/p6_*` artifacts feature-tagged + `/release/` path | ✅ | ⬜ pending |
| 6-01-04 | 01 | 2 | PERF-06 | — | N/A (audit) | manual | AUDIT-REPORT Phase-6 section: table + crossover + faer risk note + Phase-9 backlog draft | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*
*Wave/task IDs are indicative; the planner sets the authoritative decomposition.*

---

## Wave 0 Requirements

- [ ] `bench_p6_svd_comparison` in `fdars-core/benches/audit_hotpaths.rs` — nalgebra vs faer(seq) SVD + conversion sub-groups, registered in `criterion_group!`
- [ ] `svd_equivalence` unit test (singular-value agreement within tolerance; sign-ambiguity-safe) — numerical apples-to-apples guard
- [ ] Existing criterion 0.5 harness + `linalg` feature already installed — no new dependency (faer already vendored)

*Existing infrastructure (criterion, faer under `linalg`) covers the phase; only the new bench group + equivalence test are added.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Benchmark artifacts reproduce within ±5% across two runs | PERF-06 | Timing measurement is inherently observational; variance is judged, not asserted by CI | Compare `p6_*_run1.txt` vs `p6_*_run2.txt` per cell; flag any cell >5% variance |
| faer adoption / maintenance-risk note is evidence-backed | PERF-06 / SC3 | Qualitative ROI judgement (bus-factor, API churn, MSRV) | Confirm AUDIT-REPORT Phase-6 "faer Adoption Note" cites RESEARCH.md sources, not opinion |
| Go/no-go decision cites Phase 4 evidence | PERF-06 / SC1 | Cross-phase reasoning traceability | Confirm the report references SVD-share (~99.8–99.9%) and copy-share (0.14–0.17%) from Phase 4 |

---

## Validation Sign-Off

- [ ] Bench task has a runnable `cargo bench` verify command
- [ ] Equivalence unit test gates the comparison's fairness
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify (measurement tasks are inherently manual-artifact-checked — documented above)
- [ ] Wave 0 covers the new bench group + equivalence test
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s (equivalence test)
- [ ] `nyquist_compliant: true` set in frontmatter (by `/gsd-validate-phase`)

**Approval:** pending
