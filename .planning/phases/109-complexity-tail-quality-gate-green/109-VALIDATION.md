---
phase: "109"
slug: "complexity-tail-quality-gate-green"
status: draft
nyquist_compliant: false
wave_0_complete: false
created: "2026-09-14"
---

# Phase 109 — Validation Strategy

> Behavior-preserving decomposition of the full complexity tail (97 functions).
> Signal: existing suite stays green (numeric-identical) AND the curated
> `pmat quality-gate --checks complexity,satd,dead-code,duplicates` reports 0.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in test harness + pmat 3.18.0 |
| **Quick run command** | `cargo test -p fdars-core --features linalg,parallel --lib <module>` |
| **Full suite command** | `cargo test -p fdars-core --features linalg,parallel` |
| **Per-file complexity** | `pmat analyze complexity` (confirm function + helpers ≤ cog 23 / cyc 30) |
| **Phase gate** | `pmat quality-gate --checks complexity,satd,dead-code,duplicates --fail-on-violation` → 0 |
| **Estimated runtime** | full suite ~120–300s |

---

## Sampling Rate

- **After every task commit:** touched-module tests + `pmat analyze complexity` on the touched file (function + new helpers ≤23/≤30).
- **After every plan wave:** full suite + `cargo clippy --all-targets --features linalg,parallel -- -D warnings`.
- **Before `/gsd-verify-work`:** full suite green + `cargo fmt --check` clean + `pmat quality-gate --checks complexity,satd,dead-code,duplicates` → **0**.
- **Max feedback latency:** ~300 seconds.

---

## Per-Task Verification Map

No new behavior added → existing tests are the non-regression + numeric-identity
oracle. Each of the 97 flagged functions gets: (1) its module tests stay green,
(2) pmat confirms the drop to ≤ cog 23 / cyc 30 (function AND helpers).

| Group | Requirement | Test Type | Automated Command | Status |
|-------|-------------|-----------|-------------------|--------|
| Each plan's file cluster | QUAL-03 | non-regression + complexity | `--lib <module>` + `pmat analyze complexity` | ⬜ pending |
| Whole phase (final) | QUAL-03 | gate | `pmat quality-gate --checks complexity,satd,dead-code,duplicates` → 0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

Existing infrastructure covers all requirements — refactor phase. The 2942 lib
tests + 215 doctests exercise every flagged function; they are the behavior
oracle. No new test files required.

---

## Manual-Only Verifications

*All phase behaviors have automated verification.* Per-function numeric/RNG
hazards (seeded clustering, iterative EM, DP/gradient loops, FP-accumulation
order) are proven preserved by the existing seeded/deterministic tests staying
bit-green — see the hazard map in 109-RESEARCH.md.

---

## Validation Sign-Off

- [ ] Every flagged function's module tests stay green (numeric/behavior identical)
- [ ] `pmat analyze complexity` shows each Phase-109 function + all new helpers ≤ cog 23 / cyc 30
- [ ] `pmat quality-gate --checks complexity,satd,dead-code,duplicates` → **0 violations**
- [ ] Full suite green, clippy `--all-targets` clean, `cargo fmt --check` clean
- [ ] No public API / signature / visibility change (git diff shows only added private helpers)
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
