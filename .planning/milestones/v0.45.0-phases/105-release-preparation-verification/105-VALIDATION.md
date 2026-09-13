---
phase: "105"
slug: "release-preparation-verification"
status: draft
nyquist_compliant: false
wave_0_complete: false
created: "2026-09-13"
---

# Phase 105 — Validation Strategy

> Per-phase validation contract — the milestone's whole-crate gate battery.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust toolchain: fmt, clippy, test, package |
| **Config file** | `fdars-core/Cargo.toml`, root + crate `CHANGELOG.md` |
| **Quick run command** | `cargo test -p fdars-core --features linalg,parallel soft_dtw` |
| **Full suite command** | `cargo test -p fdars-core --features linalg,parallel` |
| **Estimated runtime** | ~2–5 minutes (full gate battery) |

---

## Sampling Rate

- **After version/CHANGELOG edits:** `cargo package -p fdars-core --no-verify --allow-dirty` (metadata sanity)
- **Before completion:** the full gate battery below must all be green.
- **Max feedback latency:** ~5 minutes

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 105-01-01 | 01 | 0 | REL-01 | — | N/A | gate | `cargo test -p fdars-core --features linalg,parallel` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] Whole-crate gate battery all green:
  - `cargo fmt --check`
  - `cargo clippy --all-targets --features linalg,parallel -- -D warnings`
  - full `cargo test` (the milestone end-to-end proof)
  - `--features serde` build guard (`cargo build -p fdars-core --features serde`)
  - all 28 examples + doctests compile/run
  - `cargo package` (metadata + packaging sanity)
- [ ] Version bumped 0.44.0 → 0.45.0 in `fdars-core/Cargo.toml`.
- [ ] `[0.45.0]` entry in BOTH root and crate CHANGELOG.
- [ ] `SDTW-O1` checked off on `documentation/ROADMAP-TO-1.0.md`.
- [ ] NO git tag / publish (deferred operator step).

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| — | — | — | — |

*All release gates are automated.*

---

## Validation Sign-Off

- [ ] All gates green
- [ ] Version + CHANGELOGs + checklist updated
- [ ] No tag/publish performed
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
