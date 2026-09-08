---
phase: "86"
slug: "surface-sealing"
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
status: draft
nyquist_compliant: false
wave_0_complete: false
created: "2026-09-08"
---

# Phase 86 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

This is an **API-shape-only** refactor (seal `wire`, add `#[non_exhaustive]` to 38 config structs). No numeric or behavioral change. Validation is **compile-time + existing-test non-regression**: no new test logic is authored. The existing test suite is the behavioral non-regression proof; the 28 examples + doctests are the compile-time proof that the breaking changes are complete and self-consistent.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in test harness (`cargo test`) + doctests |
| **Config file** | `fdars-core/Cargo.toml` (features: `default=["parallel"]`, `linalg`, `serde`) |
| **Quick run command** | `cargo build -p fdars-core --features linalg,parallel` |
| **Full suite command** | `cargo test -p fdars-core --features linalg,parallel` |
| **Estimated runtime** | ~120–300 seconds (full build + 2857 tests) |

---

## Sampling Rate

- **After every task commit:** Run `cargo build -p fdars-core --features linalg,parallel` (compile proof)
- **After every plan wave:** Run `cargo test -p fdars-core --features linalg,parallel`
- **Before `/gsd-verify-work`:** Full gate set green (fmt, clippy `--all-targets`, test, `--features serde` build, examples, doctests)
- **Max feedback latency:** ~300 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 86-01-A | 01 | 1 | SEAL-01 | — / — | N/A (visibility only) | compile | `cargo build -p fdars-core --features linalg,parallel && cargo test -p fdars-core --doc --features linalg,parallel` | ✅ | ⬜ pending |
| 86-01-B | 01 | 1 | SEAL-02, SEAL-03 | — / — | N/A (attribute only) | compile | `cargo build -p fdars-core --features linalg,parallel` | ✅ | ⬜ pending |
| 86-01-C | 01 | 1 | SEAL-01/02/03 | — / — | N/A | gate | `cargo fmt --check && cargo clippy --all-targets --features linalg,parallel -- -D warnings && cargo test --features linalg,parallel && cargo build --features serde && cargo build --examples` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

*Existing infrastructure covers all phase requirements.* The 2857-test suite + doctests + 28 examples already exist and provide the full compile-time + non-regression coverage this phase needs. No new test files or fixtures are authored.

---

## Manual-Only Verifications

*All phase behaviors have automated verification.* Sealing is proven by compilation of the crate + examples + doctests; non-regression is proven by the existing test suite. Nothing requires manual inspection beyond the gate commands.

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references (none — existing infra suffices)
- [ ] No watch-mode flags
- [ ] Feedback latency < 300s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
