---
phase: 45
slug: functional-co-clustering-funlbm-latent-block
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-08-24
---

# Phase 45 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in test harness (`#[cfg(test)]` inline modules) |
| **Config file** | none — inline unit tests per source file (project convention) |
| **Quick run command** | `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel --lib coclustering` |
| **Full suite command** | `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel` |
| **Estimated runtime** | ~30–180 seconds |

---

## Sampling Rate

- **After every task commit:** module-scoped quick command (`coclustering`)
- **After every plan wave:** full suite command
- **Before `/gsd-verify-work`:** full suite green + `cargo clippy --all-targets --features linalg,parallel -- -D warnings`
- **Max feedback latency:** ~180 seconds

---

## Per-Task Verification Map

*Seeded pre-planning — the planner/validate-phase fills exact task IDs after PLAN.md exists.*

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| TBD | 01 | 1 | CLUS-02-01 (funLBM CEM fit) | — | N/A (pure numeric) | unit | `cargo test -p fdars-core --features linalg coclustering` | ❌ W0 | ⬜ pending |
| TBD | — | — | CLUS-02-02 (result: labels, block params, log-lik/ICL) | — | N/A | unit | `cargo test -p fdars-core --features linalg coclustering::result` | ❌ W0 | ⬜ pending |
| TBD | — | — | CLUS-02-03 (slope-heuristic (K,L) selection) | — | N/A | unit | `cargo test -p fdars-core --features linalg slope` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- Existing Rust test harness covers all phase requirements — no framework install needed.
- New inline `#[cfg(test)] mod tests` blocks authored alongside `coclustering.rs`.

**Test oracles (from CONTEXT + RESEARCH):**
- On synthetic data with a known (K,L) block structure, CEM recovers row AND column labels up to permutation with high `adjusted_rand_index` (compare via ARI, never raw labels — label switching).
- Classification log-likelihood is non-decreasing across CEM iterations.
- ICL is finite; determinism — same seed → identical labels / log-lik / ICL.
- Slope heuristic selects the true (K,L) (or near it) on well-separated synthetic data.
- Column-clusters range over the m argument points (`col_labels.len() == m`).
- Error paths: K > n, L > m, ncomp invalid, dimension mismatch → `FdarError` (no panic).

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| — | — | All phase behaviors have automated verification (numeric library) | — |

*All phase behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 180s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
