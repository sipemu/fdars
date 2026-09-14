---
phase: "108"
slug: "entropy-extraction-satd-unwrap-cleanup"
status: draft
nyquist_compliant: false
wave_0_complete: false
created: "2026-09-14"
---

# Phase 108 — Validation Strategy

> Behavior-preserving cleanup of three pmat sub-metrics (entropy, SATD, defect).
> The signal is the existing suite staying green (numeric/behavior identical)
> AND the three pmat sub-metrics reaching 0.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in test harness + pmat 3.18.0 |
| **Config file** | none — existing suite (2942 lib + 215 doctests + 28 examples) |
| **Quick run command** | `cargo test -p fdars-core --features linalg,parallel --lib <module>` |
| **Full suite command** | `cargo test -p fdars-core --features linalg,parallel` |
| **Entropy check** | `pmat analyze entropy` — **run 3× consecutively; all must read 0** (pmat's entropy sampler is NON-DETERMINISTIC) |
| **SATD check** | `pmat analyze satd` → 0 |
| **Defect check** | `pmat analyze defects` → the `depth/rpd.rs` `.unwrap()` gone |
| **Gate** | `pmat quality-gate` (counts ALL violations incl. warnings; no warnings-only pass) |
| **Estimated runtime** | full suite ~120–300s; pmat runs ~10–60s each |

---

## Sampling Rate

- **After every task commit:** touched-module tests + the relevant pmat sub-metric (`satd` / `defects` / `entropy`).
- **After entropy work:** run `pmat analyze entropy` **3× in a row** — the stochastic sampler can flag a different file subset each run, so a single 0 is not proof. All 3 must read 0 before the entropy task is done.
- **After every plan wave:** full suite + `cargo clippy --all-targets --features linalg,parallel -- -D warnings`.
- **Before `/gsd-verify-work`:** full suite green + `cargo fmt --check` clean + `pmat analyze entropy`(×3), `pmat analyze satd`, `pmat analyze defects` all 0.
- **Max feedback latency:** ~300 seconds.

---

## Per-Task Verification Map

No new behavior added → existing tests are the non-regression + numeric-identity oracle. SATD is prose-only (no code change). Unwrap keeps the same successful-path result.

| Workstream | Requirement | Test Type | Automated Command | Status |
|------------|-------------|-----------|-------------------|--------|
| SATD reword (dispatch.rs:281, metric/lp.rs:175) | QUAL-05 | prose-only + gate | `pmat analyze satd` → 0; docs still compile | ⬜ pending |
| Unwrap fix (depth/rpd.rs:229) | QUAL-05 | non-regression + defect | `--lib depth` tests + `pmat analyze defects` | ⬜ pending |
| Entropy extraction (test-helper + production + doctest sites) | QUAL-04 | non-regression + entropy | full suite + `pmat analyze entropy` ×3 → 0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

Existing infrastructure covers all phase requirements. Entropy extraction that
touches `#[cfg(test)]` modules must keep those tests green; doctest-example
extraction must keep the doctests passing. No new test files required — the
existing suite is the behavior oracle.

---

## Manual-Only Verifications

*All phase behaviors have automated verification.* Behavior-preservation via
the existing suite; the three pmat sub-metrics via pmat. **Known risk (from
RESEARCH.md):** pmat entropy sampling is non-deterministic — a single 0 run is
not proof; require 3 consecutive 0 runs. If entropy cannot be stabilized to 0
after all sites are addressed, that is a milestone-level decision (escalate:
pmat quality-gate config vs. accept a documented entropy threshold) — it is NOT
resolved by loosening the behavior-preservation contract.

---

## Validation Sign-Off

- [ ] `pmat analyze satd` → 0 (both doc-comment false-positives reworded; docs still compile)
- [ ] `pmat analyze defects` → the `depth/rpd.rs:229` `.unwrap()` replaced; no new defect
- [ ] `pmat analyze entropy` → 0 on **3 consecutive runs**
- [ ] Full suite green, clippy `--all-targets` clean, `cargo fmt --check` clean (numeric-identical)
- [ ] No public API / signature / visibility change (`build_augmented` return type unchanged; new helpers private / `#[cfg(test)]`)
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
