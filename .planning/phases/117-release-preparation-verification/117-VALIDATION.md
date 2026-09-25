---
phase: "117"
slug: "release-preparation-verification"
status: validated
nyquist_compliant: true
wave_0_complete: true
created: "2026-09-25"
---

# Phase 117 — Validation Strategy

> Reconstructed post-execution by validate-phase (State B).

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | cargo test (lib + integration + doctests), cargo clippy/fmt, pmat 3.18.0, pytest (benchmarks harness) |
| **Config file** | fdars-core/Cargo.toml; .github/workflows/rust-ci.yml (pmat gate definition) |
| **Quick run command** | `cargo fmt --check && (cd fdars-core && pmat quality-gate --checks complexity,satd,dead-code,duplicates --fail-on-violation) && cargo test -p fdars-core --features linalg,parallel --test equivalence_phase114` |
| **Full suite command** | the 16-gate battery recorded in `117-GATE-LOG.md` (run one gate per foreground call, 570 s bound) |
| **Estimated runtime** | quick ~60 s; full battery ~15 min |

---

## Sampling Rate

- **After every task commit:** quick run command
- **Before `/gsd-verify-work`:** full 16-gate battery green
- **Max feedback latency:** 60 seconds (quick)

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 117-01-01 | 01 | 1 | REL-01 | T-117-01 | version-only manifest diff | diff gate | `grep '^version = "0.47.0"' fdars-core/Cargo.toml` + single-line diff check | ✅ | ✅ green |
| 117-01-02 | 01 | 1 | REL-01 | T-117-02/04 | caches ignored; honest claims | package list + grep | `cargo package -p fdars-core --list` clean; [0.47.0] cites 1.19x, not 3.45x | ✅ | ✅ green |
| 117-02-01 | 02 | 2 | REL-01 | T-117-05 | decision before edit | checkpoint | `pmat_decision: refactor` recorded in 117-02-SUMMARY.md | ✅ | ✅ green |
| 117-02-02 | 02 | 2 | REL-01 | T-117-06 | numeric identity | golden (debug+release) | `cargo test --test equivalence_phase114 --test equivalence_phase47` (+ `--release`) | ✅ | ✅ green |
| 117-02-03 | 02 | 2 | REL-01 | T-117-07 | gate not weakened | CLI | `cd fdars-core && pmat quality-gate ... --fail-on-violation` → 0 violations | ✅ | ✅ green |
| 117-03-01 | 03 | 3 | REL-01 | T-117-09 | — | lint/build | fmt, clippy --all-targets, serde build, 28 examples | ✅ | ✅ green |
| 117-03-02 | 03 | 3 | REL-01 | T-117-08 | exit codes not masked | test suite | lib 2942 / integration 613 (21 bins) / doctests 215 | ✅ | ✅ green |
| 117-04-01 | 04 | 4 | REL-01 | T-117-02/11 | clean, non-dirty package; hermetic fallback | package | `cargo package -p fdars-core` + tarball bench `--test n100_m50` | ✅ | ✅ green |
| 117-04-02 | 04 | 4 | REL-01 | T-117-03/10 | no tag/publish; verdict tied to rows | CLI | pmat, pytest (209), render --check/--check-prose, `git tag -l 'v0.47*'` empty | ✅ | ✅ green |

---

## Wave 0 Requirements

Existing infrastructure covers all phase requirements.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Tag + crates.io publish | REL-01 (deferred operator step) | irreversible, outward-facing; intentionally out of phase scope | see `117-GATE-LOG.md` → "Deferred operator step" |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 60s (quick)
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-09-25

## Validation Audit 2026-09-25
| Metric | Count |
|--------|-------|
| Gaps found | 0 |
| Resolved | 0 |
| Escalated | 0 |
