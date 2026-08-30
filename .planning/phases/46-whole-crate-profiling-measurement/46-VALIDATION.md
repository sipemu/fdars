---
phase: 46
slug: whole-crate-profiling-measurement
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
# audit-milestone §5.5 distinguishes NOT-VALIDATED (draft) from PARTIAL (validated + nyquist_compliant: false) (#2117)
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-08-30
---

# Phase 46 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

**Measure-only audit phase.** Phase 46 produces three inventory documents and makes ZERO
behavior-changing edits to `fdars-core/src/`. There are therefore **no new Nyquist test-coverage
requirements** — no new `#[test]` functions verify measurement behavior. The only applicable
validation is a **baseline-green sanity check**: the existing suite must stay green after any
throwaway probe benches are added and removed. Bench probes are `harness = false` criterion
binaries and are discarded (not registered as permanent `[[bench]]` entries — that is Phase 51).

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in test harness (`cargo test`) + criterion 0.5 (`harness = false`) + dhat 0.3 (`dhat-heap` feature) |
| **Config file** | none — dev-deps already present in `fdars-core/Cargo.toml` |
| **Quick run command** | `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel --lib` |
| **Full suite command** | `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel` |
| **Estimated runtime** | ~120–300 seconds (full suite; 2583 lib tests + integration/doc) |

---

## Sampling Rate

- **After every task commit:** confirm `cargo build -p fdars-core --features linalg,parallel` still compiles (probe benches must not break the build).
- **After every plan wave:** Run the full suite command — must be green (proves zero behavior change).
- **Before `/gsd-verify-work`:** Full suite green AND `cargo clippy --all-targets --features linalg,parallel -- -D warnings` clean.
- **Max feedback latency:** ~300 seconds.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| baseline | — | 0 | PROF-01/02/03 | — | Suite green before + after measurement (no `src/` change) | integration | `cargo test -p fdars-core --features linalg,parallel` | ✅ | ⬜ pending |

---

## Wave 0 Requirements

- [ ] Free disk headroom for bench builds: `rm -rf target/debug/{incremental,examples}` (MEMORY.md pointer — example LINK dies when `target/` fills /home).
- [ ] Confirm baseline suite green under `TMPDIR=/home/simonm/.cache/fdars-bench-tmp` before any measurement.

*Existing infrastructure (cargo test + criterion + dhat) covers all Phase 46 measurement needs — no new test framework required.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Ranked inventory completeness / groundedness | PROF-01/02/03 | The deliverables are *report documents* (rankings, anchors, canonical-form proposals), not executable code — correctness is a human judgement about whether the inventories are concrete enough to drive Phases 47–51 | Review each of the three inventory docs: every item carries a real criterion/allocation number (PROF-01) or a `file:line` anchor (PROF-02/03) and a proposed canonical form (PROF-03) |

*Automated verification covers "suite stays green"; inventory quality is verified by the phase verifier / human review.*

---

## Validation Sign-Off

- [ ] Full suite green before measurement (baseline)
- [ ] Full suite green after probe benches removed (zero behavior change confirmed)
- [ ] No throwaway probe bench left registered as a permanent `[[bench]]` (Phase 51 owns those)
- [ ] Three inventory documents exist, each with the required anchors/numbers
- [ ] `nyquist_compliant: true` set in frontmatter (N/A test coverage — measure-only; set once baseline-green + inventories confirmed)

**Approval:** pending
