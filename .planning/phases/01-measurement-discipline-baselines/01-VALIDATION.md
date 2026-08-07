---
phase: 1
slug: measurement-discipline-baselines
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
# audit-milestone §5.5 distinguishes NOT-VALIDATED (draft) from PARTIAL (validated + nyquist_compliant: false) (#2117)
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-08-07
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
>
> **Audit-milestone note:** this phase produces analysis artifacts (methodology section,
> workload matrix, baseline runs) plus one piece of measurement infrastructure
> (`fdars-core/benches/audit_hotpaths.rs`). "Validation" here means the bench
> **compiles and runs** across the feature-flag matrix and the artifacts satisfy the
> four ROADMAP success criteria — not application unit tests.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in test harness + Criterion 0.5 (bench harness = false) |
| **Config file** | `fdars-core/Cargo.toml` (`[[bench]]` entry for `audit_hotpaths`) |
| **Quick run command** | `cargo bench -p fdars-core --bench audit_hotpaths --no-default-features --features linalg,parallel -- --test` |
| **Full suite command** | 4-combo compile + baseline run (see Per-Task map) |
| **Estimated runtime** | Compile check ~seconds; a full baseline sweep is minutes (elastic cell dominates, capped per D-07) |

*`-- --test` runs Criterion benches once as a smoke test (no measurement) — the fast "does it compile and execute" gate. Full measurement runs drop `--test`.*

---

## Sampling Rate

- **After every task commit:** Run the quick `cargo bench ... -- --test` smoke check for the touched bench/target.
- **After the bench-authoring wave:** Compile the bench under all 4 feature combos (`""`, `parallel`, `linalg`, `linalg,parallel`).
- **Before `/gsd-verify-work`:** At least one baseline target per module has recorded raw output under `.planning/research/bench/`, and the AUDIT-REPORT.md methodology + workload-matrix sections exist.
- **Max feedback latency:** ~60s for the compile/smoke gate (measurement runs are intentionally longer).

---

## Per-Task Verification Map

> Filled concretely by the planner. Skeleton rows below map the phase's success criteria to checks.

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 1-01-01 | 01 | 1 | PERF-02 | — | N/A (no runtime surface) | compile | `cargo bench -p fdars-core --bench audit_hotpaths --no-default-features --features linalg,parallel -- --test` | ❌ W0 | ⬜ pending |
| 1-01-02 | 01 | 1 | PERF-02 | — | N/A | matrix-compile | 4-combo compile (`""`/`parallel`/`linalg`/`linalg,parallel`) | ❌ W0 | ⬜ pending |
| 1-01-03 | 01 | 2 | PERF-02 | — | N/A | artifact | `/release/` path confirmed in criterion output; raw file saved under `.planning/research/bench/` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `fdars-core/benches/audit_hotpaths.rs` — new audit bench (seeded synthetic N×M `FdMatrix` generator + sentinel targets)
- [ ] `[[bench]]` entry with `harness = false` in `fdars-core/Cargo.toml`
- [ ] `.planning/research/bench/` directory created for raw criterion output

*Criterion 0.5 and rand 0.8 are already dependencies — no framework install needed.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| `/release/` binary-path confirmation | PERF-02 (SC3) | Requires inspecting the criterion bench binary path in stdout of a real run | Run `cargo bench ... --features linalg,parallel`, confirm the executed binary is under `target/release/deps/`, record the path in AUDIT-REPORT.md |
| ±5% two-run variance check | PERF-02 (SC2) | Requires two independent `cargo bench` invocations and human comparison | Run the sentinel twice; if the two medians differ >10% mark LOW CONFIDENCE per methodology |
| Infra-vs-code triage (bus error / linker flakiness) | PERF-02 (SC4) | A SIGBUS/linker failure is a toolchain signal, not a measurable behavior | Document the triage rule in the methodology section; classify any bench/doctest bus error as infra, re-run |

---

## Validation Sign-Off

- [ ] Audit bench compiles under all 4 feature combos
- [ ] Bench smoke-runs (`-- --test`) green under `linalg,parallel`
- [ ] At least one baseline target per module recorded with `/release/` confirmed
- [ ] Methodology + workload-matrix sections written to AUDIT-REPORT.md
- [ ] Sampling continuity: no 3 consecutive tasks without an automated compile/smoke check
- [ ] `nyquist_compliant: true` set in frontmatter (by validate-phase)

**Approval:** pending
