---
phase: 4
slug: fpca-svd-allocation-audit
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
# audit-milestone §5.5 distinguishes NOT-VALIDATED (draft) from PARTIAL (validated + nyquist_compliant: false) (#2117)
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-08-08
---

# Phase 4 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> **Audit phase:** the deliverables are (a) criterion bench + dhat measurement scaffolding under `fdars-core/` (measurement code only — no `fdars-core` algorithm changes) and (b) a report slice appended to `.planning/research/AUDIT-REPORT.md` plus a baseline under `.planning/research/bench/`. Verification is a mix of `cargo bench` / `cargo test` runs and provenance greps against the report.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | criterion 0.5 (wall-clock timing) + dhat 0.3 `testing()` mode (allocation counts, `#[global_allocator]` in a separate integration-test process) |
| **Config file** | `fdars-core/Cargo.toml` — extend `[[bench]] name = "audit_hotpaths"`; add `dhat` to `[dev-dependencies]` and `dhat-heap` to `[features]` |
| **Quick run command** | `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo bench -p fdars-core --features linalg,parallel --bench audit_hotpaths -- bench_p4_fpca` |
| **Full suite command** | Quick run above **plus** dhat audit: `cargo test -p fdars-core --features dhat-heap,linalg --test alloc_audit_fpca -- --nocapture` **plus** the SC1–SC4 provenance greps below |
| **Estimated runtime** | ~2–4 min (6-cell FPCA grid + elastic cells + dhat test; FPCA cells are sub-second even at N=1000) |

---

## Sampling Rate

- **After every task commit:** Run the task's provenance command from the Per-Task Verification Map (bench cell produces a criterion estimate, or the report grep confirms the section text landed with real `file:line` anchors / measured numbers).
- **After every plan wave:** Run all SC1–SC4 checks against `AUDIT-REPORT.md` + confirm the `.planning/research/bench/` baseline files exist.
- **Before `/gsd-verify-work`:** All four SC checks pass and each cited `file:line` still exists in `fdars-core/src/`.
- **Max feedback latency:** bench cell ~< 30 s per cell; grep checks < 5 s.

---

## Per-Task Verification Map

> Task IDs are provisional (finalized by the planner). Each row maps a success criterion to a deterministic command.

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 4-01-xx (Wave 0: dhat wiring) | 01 | 0 | PERF-04 | — | N/A (measurement scaffolding) | build | `cargo build -p fdars-core --features dhat-heap,linalg` exits 0 | ❌ W0 (Cargo.toml edit + `tests/alloc_audit_fpca.rs`) | ⬜ pending |
| 4-01-xx (SC1: bench grid) | 01 | 1 | PERF-03 | — | N/A | bench | `TMPDIR=… cargo bench -p fdars-core --features linalg,parallel --bench audit_hotpaths -- bench_p4_fpca` produces 6 criterion estimates | ✅ (audit_hotpaths.rs exists) | ⬜ pending |
| 4-01-xx (SC1: report table) | 01 | 1 | PERF-03 | — | N/A | grep | `grep -Ec "n(100|500|1000)_m(50|200)" .planning/research/AUDIT-REPORT.md` (≥ 6 Phase-4 FPCA rows) | ✅ | ⬜ pending |
| 4-01-xx (SC1: elastic sites) | 01 | 1 | PERF-03 | — | N/A | bench | `… --bench audit_hotpaths -- bench_p4_elastic_fpca` produces vert_fpca + joint_fpca estimates | ✅ | ⬜ pending |
| 4-02-xx (SC2: dhat baseline) | 02 | 1 | PERF-04 | — | N/A | test+file | `cargo test -p fdars-core --features dhat-heap,linalg --test alloc_audit_fpca -- --nocapture` then `ls .planning/research/bench/p4_dhat_*.txt` (baseline file with `total_blocks`/`total_bytes`) | ❌ W0 | ⬜ pending |
| 4-02-xx (SC2: hotspot ranking) | 02 | 1 | PERF-04 | — | N/A | grep | `grep -Ec "regression.rs:(167|291|298)" .planning/research/AUDIT-REPORT.md` (the 3 FPCA alloc sites ranked) | ✅ | ⬜ pending |
| 4-02-xx (SC3: wall-clock share) | 02 | 2 | PERF-03, PERF-04 | — | N/A | grep | `grep -Eic "copy.*%|allocation.*share|SVD.*dominat|copy.*dominat" .planning/research/AUDIT-REPORT.md` (≥ 1) | ✅ | ⬜ pending |
| 4-02-xx (SC4: backlog fields) | 02 | 2 | PERF-03, PERF-04 | — | N/A | grep | `grep -Ec "Current cost|Root cause|Candidate" .planning/research/AUDIT-REPORT.md` (≥ 6 = 2 entries × 3 fields) | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `fdars-core/Cargo.toml` — add `dhat = "0.3"` (confirm exact version via `cargo search dhat` first — RESEARCH open question A1) to `[dev-dependencies]`, add `dhat-heap = []` to `[features]`, register `[[test]] name = "alloc_audit_fpca"` if needed
- [ ] `fdars-core/tests/alloc_audit_fpca.rs` — create dhat integration-test harness with `#[global_allocator]` and `dhat::Profiler::builder().testing().build()`, exercising `fdata_to_pc_1d` at the audit cell(s)

*The criterion bench extension (`bench_p4_fpca`, `bench_p4_elastic_fpca` in `audit_hotpaths.rs`) and the `AUDIT-REPORT.md` / `.planning/research/bench/` appends are Wave 1+ tasks — not Wave 0 gaps.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| The report's SVD-compute-vs-copy split is stated as a share of wall-clock for the top FPCA path, and the number is consistent with the measured dhat bytes ÷ measured criterion time (not a theoretical estimate) | PERF-03, PERF-04 (SC3) | A grep can confirm the phrase exists but not that the ratio was computed from the two measured numbers; a human/verifier must cross-check the dhat baseline file and the criterion estimate feed the stated % | Open `.planning/research/bench/p4_dhat_*.txt` for measured bytes/allocs and the criterion estimate for the same cell; confirm the report's stated copy-share % is derived from both, and that the SVD-vs-copy conclusion matches |
| The report's SVD-vs-copy conclusion explicitly states whether the Phase 6 go/no-go trigger fires (comparison only if "SVD is a significant share of FPCA runtime AND copy is not the dominant cost" — ROADMAP §Phase 6 SC1) | PERF-03, PERF-04 (SC3) | The trigger is a compound boolean over two measured quantities; only a reader cross-referencing both numbers against the ROADMAP condition can confirm the go/no-go call is stated and justified | Read the report's Phase-6-trigger sentence; confirm it names both the SVD share and the copy dominance, and states a fires/does-not-fire verdict backed by the measured numbers |
| The dhat audit distinguishes the copy site (`regression.rs:298` `to_dmatrix()`) from the two `FdMatrix` allocations (`regression.rs:167` `center_columns`, `:291` `centered.clone()`), and does NOT attribute the covariance-SVD elastic sites (elastic_fpca.rs:122/399, which build a native `DMatrix` with no `to_dmatrix()` copy) to the copy-overhead bucket | PERF-04 (SC2) | RESEARCH flagged that conflating `to_dmatrix()` copy sites with native-covariance SVD sites would misreport the copy overhead; requires cross-referencing each cited line against source | For each cited `file:line`, `grep -n` the source; confirm the copy-overhead ranking lists only true `to_dmatrix()` sites and the `centered.clone()` zero-copy candidate is called out separately |

---

## Validation Sign-Off

- [ ] Wave 0 (dhat wiring) builds green before Wave 1 measurement tasks run
- [ ] All tasks have an automated bench/test/grep verify or a documented manual verification
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references (dhat dev-dep + `alloc_audit_fpca.rs`)
- [ ] No watch-mode flags
- [ ] Feedback latency within stated bounds
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
