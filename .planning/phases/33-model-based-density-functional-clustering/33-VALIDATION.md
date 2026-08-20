---
phase: 33
slug: model-based-density-functional-clustering
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-08-20
---

# Phase 33 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in test harness (`#[cfg(test)]` inline modules) |
| **Config file** | none — Cargo workspace |
| **Quick run command** | `cargo test -p fdars-core --features linalg,parallel --lib clustering_advanced gmm::subspace` |
| **Full suite command** | `cargo test -p fdars-core --features linalg,parallel` |
| **Estimated runtime** | ~90 seconds full suite; ~5s for clusterer-only |

---

## Sampling Rate

- **After every task commit:** Run the quick clusterer-filtered test
- **After every plan wave:** Run `cargo test -p fdars-core --features linalg,parallel --lib`
- **Before `/gsd-verify-work`:** Full suite (incl. doctests) + `cargo clippy --all-targets --features linalg,parallel -- -D warnings` must be green
- **Max feedback latency:** 90 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 33-00-01 | 00 | 0 | CLUS-01 | — / — | test-only adjusted_rand_index helper | unit | `cargo test -p fdars-core --features linalg,parallel --lib adjusted_rand` | ❌ W0 | ⬜ pending |
| 33-01-01 | 01 | 1 | CLUS-01 | — / — | funHDDC recovers groups (ARI ≥ threshold); invalid input → FdarError | unit | `cargo test -p fdars-core --features linalg,parallel --lib gmm::subspace` | ❌ W0 | ⬜ pending |
| 33-02-01 | 02 | 2 | CLUS-01 | — / — | funFEM discriminative-subspace recovery | unit | `cargo test -p fdars-core --features linalg,parallel --lib clustering_advanced::tests::funfem` | ❌ W0 | ⬜ pending |
| 33-02-02 | 02 | 2 | CLUS-01 | — / — | DBSCAN recovers groups + flags noise (None) | unit | `cargo test -p fdars-core --features linalg,parallel --lib clustering_advanced::tests::dbscan` | ❌ W0 | ⬜ pending |
| 33-02-03 | 02 | 2 | CLUS-01 | — / — | kCFC subspace-embedding recovery | unit | `cargo test -p fdars-core --features linalg,parallel --lib clustering_advanced::tests::kcfc` | ❌ W0 | ⬜ pending |
| 33-02-04 | 02 | 2 | CLUS-01 | — / — | align-and-cluster recovers shape-shifted groups | unit | `cargo test -p fdars-core --features linalg,parallel --lib clustering_advanced::tests::align_cluster` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

*Task IDs are indicative — the planner sets the authoritative task breakdown.*

---

## Wave 0 Requirements

- [ ] Add test-only `adjusted_rand_index` helper (~20 lines, Hubert & Arabie 1985) — no ARI helper exists in repo
- [ ] New `fdars-core/src/gmm/subspace.rs` (funHDDC) + new `fdars-core/src/clustering_advanced.rs` (funFEM/DBSCAN/kCFC/align-cluster) with inline `#[cfg(test)]` tests
- [ ] No new test framework — Rust built-in harness

*Existing infrastructure covers all phase requirements.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| — | — | — | — |

*All phase behaviors have automated verification (synthetic-recovery ARI/accuracy + noise-flagging + invalid-input tests).*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references (ARI helper)
- [ ] No watch-mode flags
- [ ] Feedback latency < 90s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
