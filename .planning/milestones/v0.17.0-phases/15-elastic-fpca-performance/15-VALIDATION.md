---
phase: 15
slug: elastic-fpca-performance
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-08-12
---

# Phase 15 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in `#[test]` harness (inline `#[cfg(test)] mod tests`) |
| **Config file** | `fdars-core/Cargo.toml` |
| **Quick run command** | `cargo test -p fdars-core --features linalg -- elastic_fpca` |
| **Full suite (parallel on)** | `cargo test -p fdars-core --features linalg,parallel` |
| **Full suite (parallel off)** | `cargo test -p fdars-core --features linalg` (no `parallel`) |
| **Estimated runtime** | ~90 seconds |

**Note:** `TMPDIR=/home/simonm/.cache/fdars-bench-tmp` may be needed for link steps. Equivalence must hold with the `parallel` feature ON and OFF.

---

## Sampling Rate

- **After every task commit:** `cargo test -p fdars-core --features linalg -- elastic_fpca`
- **After the wave:** run the suite BOTH with and without `parallel` (equivalence must hold in both).
- **Before `/gsd-verify-work`:** full suite green under `linalg` and `linalg,parallel`.

---

## Per-Task Verification Map

| Req | Behavior | Test Type | Automated Command | File Exists | Status |
|-----|----------|-----------|-------------------|-------------|--------|
| PERF-04-A | `:701` `shooting_vectors_from_psis` uses `iter_maybe_parallel!`; output identical to a sequential reference | unit | `cargo test -p fdars-core --features linalg -- elastic_fpca::tests::test_shooting_vectors_parallel_equiv` | ❌ W0 | ⬜ pending |
| PERF-04-B | `:720` `build_augmented_srsfs` uses `iter_maybe_parallel!`; output identical to sequential reference | unit | `cargo test -p fdars-core --features linalg -- elastic_fpca::tests::test_augmented_srsfs_parallel_equiv` | ❌ W0 | ⬜ pending |
| PERF-04-C | `:764` `svd_scores_and_eigenvalues` inner loop guarded by N≥50 threshold; parallel above, sequential below | unit | `cargo test -p fdars-core --features linalg -- elastic_fpca::tests::test_scores_threshold` | ❌ W0 | ⬜ pending |
| PERF-04-D | `vert_fpca` scores + eigenvalues equal sequential reference (bit-identical / within 1e-12) at N≥50 | unit | `cargo test -p fdars-core --features linalg -- elastic_fpca::tests::test_vert_fpca_parallel_equiv` | ❌ W0 | ⬜ pending |
| PERF-04-E | `joint_fpca` scores + eigenvalues equal sequential reference (bit-identical / within 1e-12) at N≥50 | unit | `cargo test -p fdars-core --features linalg -- elastic_fpca::tests::test_joint_fpca_parallel_equiv` | ❌ W0 | ⬜ pending |
| PERF-04-F | Suite green with `parallel` OFF (macro compiles to sequential; no behavior change) | build+test | `cargo test -p fdars-core --features linalg -- elastic_fpca` | ✅ existing | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red*

---

## Wave 0 Requirements

- [ ] `fdars-core/src/elastic_fpca.rs` — inline `#[cfg(test)] mod tests` (extend if present) with PERF-04-A…E equivalence + threshold tests, using a small synthetic elastic dataset (N≥50 to exercise the `:764` threshold path). A sequential reference is computed inline (or via a pre-parallelization snapshot of expected values).

*Built-in `#[test]` harness already configured — no framework install needed.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Actual wall-clock speedup (~4–5× at N≥50) | PERF-04 | Governor unpinned → LOW-CONFIDENCE per audit; not asserted as a pinned number | Optional: `cargo bench` thread-scaling; feasibility+equivalence is the gate, not a speedup number |

---

## Security Domain (ASVS L1)

Pure internal compute refactor — no I/O, no new public surface, no input-validation change (the three fns are `pub(crate)`/private, called by already-validated `vert_fpca`/`joint_fpca`). No new threat surface; V5/V7 unchanged from the existing entry points.

---

## Validation Sign-Off

- [ ] All tasks have automated verify or Wave 0 dependencies
- [ ] Equivalence holds with `parallel` ON and OFF
- [ ] No watch-mode flags
- [ ] `nyquist_compliant: true` set by `/gsd-validate-phase`

**Approval:** pending
