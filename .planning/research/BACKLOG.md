# fdars Prioritized Backlog

**Crate:** fdars-core v0.14.0
**Audit milestone:** v0.14.0 — audit-only deliverable; no production code changes included
**Source report:** [AUDIT-REPORT.md](AUDIT-REPORT.md) (Phases 1–9)
**Produced by:** Phase 9 consolidation (Plans 01–03)

This file is the standalone prioritized backlog for the v0.14.0 audit milestone.
It is intended to be consumed directly by `/gsd-new-milestone` to promote items into
future milestone requirements. Each item is phrased as a GSD-ready candidate
requirement or phase.

---

## Ranking Methodology

### Formula

```
score = value / sqrt(effort)
```

A higher score means more user value delivered per unit of effort. Items are ordered by
descending score in the Ranked Backlog table. The formula rewards high-value items and
penalizes high-effort items non-linearly (large efforts are more than proportionally
expensive to deliver).

### Value Scale (1–5)

Value measures **user value**, not ease of implementation.

| Value | Anchor |
|-------|--------|
| 5 | Table-stakes capability blocking real workloads — absent capability that scikit-fda users rely on daily, or P1 default-path performance cost affecting every caller |
| 4 | High-value capability widely used in practice; present partial implementation needs significant work; or P1 hot-path saving >2× at common workload sizes |
| 3 | Meaningful capability or performance improvement; important but not blocking; commonly requested in FDA toolkits |
| 2 | Useful addition or moderate performance gain; niche use-case or limited to uncommon workload sizes |
| 1 | Niche differentiator, cosmetic improvement, or very minor performance gain with limited real-world impact |

### Effort Map (S / M / L)

| Effort | Numeric | sqrt(effort) | Definition |
|--------|---------|--------------|------------|
| S | 1 | 1.000 | Small — approximately 1 week of implementation including tests |
| M | 3 | 1.732 | Medium — approximately 2–4 weeks including integration and validation |
| L | 9 | 3.000 | Large — approximately 1–3 months or cross-cutting architectural change |

### Severity Scale

| Severity | Meaning |
|----------|---------|
| P1 | Default-path performance cost affecting every caller of a common function, or a table-stakes capability gap blocking real workloads |
| P2 | Meaningful but not blocking — measurable performance win or useful missing capability that sophisticated users notice |
| P3 | Niche or cosmetic-adjacent — minor gain limited to uncommon workload sizes or rare use-cases |

**Note:** Severity (P1/P2/P3) and Value (1–5) are correlated but independent. Severity
describes the category of impact; Value quantifies user benefit for ranking purposes.
A P2 item can have Value=4 if the improvement is significant for a moderately common
workload. A P1 item with wide reach but low absolute gain may have Value=3.

---

## Ranked Backlog

Items ordered by descending `score = value / sqrt(effort)`. Computed score shown in Score column.

| Rank | ID | Title | Severity | Value (1–5) | Effort (S/M/L) | Score (value/sqrt(effort)) | Area / Location |
|------|----|-------|----------|------------|----------------|---------------------------|-----------------|
| 1 | P6-1 | Swap nalgebra SVD for faer thin_svd in `fdata_to_pc_1d` | P2 | 3 | S | 3.00 | FPCA / `regression.rs:298` |

*Rows will be appended by Plans 02 and 03. Final sort by Plans 02/03.*

---

## Backlog Items

### P6-1 — Swap nalgebra SVD for faer thin_svd in `fdata_to_pc_1d`

**Candidate requirement / phase phrasing:** "Replace the nalgebra `SVD::new(weighted.to_dmatrix(), true, true)` call in `fdata_to_pc_1d` with faer `thin_svd` on a zero-copy `MatRef` view, gated behind the existing `linalg` feature."

- **Location / area:** `fdars-core/src/regression.rs`, line 298. The `fdata_to_pc_1d` function is the primary FPCA entry point called by scalar-on-function regression, functional logistic regression, and classification CV loops — every FPCA-backed computation in the library routes through this site.

- **Current cost or gap:** nalgebra SVD at the primary audit cell (N=500, M=200): **41.026 ms** (run1 median). This SVD step accounts for approximately **99.8–99.9%** of total `fdata_to_pc_1d` wall-clock. The `to_dmatrix()` bridge at the same line allocates an ~800 KB DMatrix copy (N×M×8 bytes) on every call; its copy-share is ~0.17% of wall-clock (negligible, but eliminable). At N=1000, M=200: 95.6 ms per FPCA call.

- **Root cause:** `nalgebra::SVD::new` requires a `DMatrix<f64>` input, which forces a `to_dmatrix()` column-major memcopy from `FdMatrix` before the SVD can begin. nalgebra's SVD implementation is always sequential regardless of the `parallel` feature flag. faer's `thin_svd` accepts a `MatRef` — a zero-copy view constructed directly from the `FdMatrix` column-major slice via `MatRef::from_column_major_slice` — and executes a faster SVD algorithm that consistently outperforms nalgebra at fdars' tall-thin (N >> M) rectangular matrix sizes.

- **Proposed direction:** Under `#[cfg(feature = "linalg")]`, replace the `weighted.to_dmatrix()` + `SVD::new` block with:
  ```
  let mat_ref = faer::MatRef::<f64>::from_column_major_slice(weighted.as_slice(), n, m);
  let fa_svd = mat_ref.thin_svd();
  ```
  Extract U, S, Vt using faer accessors (`.U()`, `.S()`, `.V()`). Retain the existing nalgebra path under `#[cfg(not(feature = "linalg"))]` so the `""` and `parallel` builds are unaffected. Add a CI regression test verifying `FpcaResult` output agrees with the nalgebra path within numerical tolerance. Evaluate faer parallel SVD (not measured in Phase 6) — if it offers additional speedup at M≥200, surface as a follow-on candidate.

- **Severity (P1/P2/P3):** **P2** — The measured speedup at the primary cell (N=500, M=200) is **1.8×** (run1) / **1.9×** (run2), below the research-defined "clearly worth it" threshold of ≥2×. The speedup is consistently positive at all 7 measured cells (3.6×, 4.1×, 2.7×, 1.8×, 3.7×, 3.1×, 1.9× — N∈{100,100,500,500,1000,1000,500} × M∈{50,200,50,200,50,200,500}). The absolute saving at N=1000, M=200 is ~27 ms/call — meaningful for FPCA-heavy workflows (pipeline loops, cross-validation grids). Downgrade to P3 if a pinned-governor re-run shows speedup < 1.5× at the primary cell.

- **Effort estimate (S/M/L):** **S** — approximately 1 week. faer is already a dependency of `fdars-core` (no new Cargo.toml additions). Code change is ~20 lines in `fdata_to_pc_1d`. Output extraction requires mapping faer accessors to the existing `FpcaResult` fields (singular values, U, Vt). Singular vector sign conventions may differ from nalgebra — a one-time equivalence check is required. Numerical equivalence already confirmed by the `svd_equivalence` integration test.

- **Evidence link:** [bench/p6_svd_nalgebra_linalg_run1.txt](bench/p6_svd_nalgebra_linalg_run1.txt) (N=500, M=200: 41.026 ms) · [bench/p6_svd_faer_seq_linalg_run1.txt](bench/p6_svd_faer_seq_linalg_run1.txt) (N=500, M=200: 23.084 ms) · speedup: **1.8×**. Wall-clock source for copy-share derivation: [bench/p4_fpca_linalg,parallel_run1.txt](bench/p4_fpca_linalg,parallel_run1.txt) (N=1000, M=200: 38.307 ms; copy-share derived as 0.14% from 53.3 µs / 38,307 µs). Full comparison grid and narrative in AUDIT-REPORT.md §Phase 6 SC2.

---

## Completeness Gate

This section documents the checklist every backlog item MUST pass before a plan is marked
complete, and records the three phase-level assertions Plan 03 will finalize over the full
item set.

### 7-Field Item Checklist

Every item under `## Backlog Items` must carry all seven of the following fields, each with
substantive content (not a placeholder):

1. **Location / area** — file, function, and/or module path; characterizes scope
2. **Current cost or gap** — a real measurement (benchmark number, allocation count) or a
   documented capability absence; no invented figures
3. **Root cause** — the algorithmic or architectural reason the cost or gap exists
4. **Proposed direction** — a concrete, GSD-ready candidate fix or feature description
5. **Severity (P1/P2/P3)** — severity classification with a brief rationale
6. **Effort estimate (S/M/L)** — effort classification with a brief rationale
7. **Evidence link** — a Markdown link to a real file under `.planning/research/bench/` or
   a phase SUMMARY / AUDIT-REPORT section; must be resolvable

### Tracer Item Status

**P6-1** passes the 7-field checklist as of Plan 01:
- Location / area: `regression.rs:298` (fdata_to_pc_1d) — present
- Current cost: 41.026 ms at N=500,M=200; 99.8–99.9% SVD share — present (real benchmark number)
- Root cause: nalgebra requires DMatrix allocation; always sequential — present
- Proposed direction: faer MatRef zero-copy + thin_svd, linalg-gated — present (GSD-ready wording)
- Severity: P2 with rationale (1.8× at primary cell, below 2× threshold) — present
- Effort: S (~1 week, faer already vendored, ~20 lines) — present
- Evidence link: two bench artifacts with real numbers linked — present

Computed score in Ranked Backlog: value=3, effort=S(1), score=3/sqrt(1)=**3.00** — present.

### Phase-Level Assertions (Deferred to Plan 03)

The following three assertions require the full item set (all performance + gap backlog items)
and are explicitly deferred to Plan 03, which performs the final sort and completeness sweep:

1. **P1-existence:** At least one P1 item exists in the backlog. (Cannot be asserted with
   a single P6-1 tracer item; deferred until Plans 02/03 add all performance and gap items.)

2. **No top-10 cosmetic items:** No item in the top 10 ranked rows is a cosmetic
   convenience-only entry (i.e., all top-10 items affect correctness, performance on a real
   workload, or a documented scikit-fda capability gap). (Deferred to Plan 03 final-sort pass.)

3. **Descending-score order:** The `## Ranked Backlog` table rows are ordered by descending
   `score = value / sqrt(effort)` after the final sort. (Deferred to Plan 03; Plan 01 seeds
   one row and Plans 02/03 append rows before Plan 03 performs the final sort.)

Plan 03 will confirm all three phase-level assertions and mark the gate as PASSED or flag
any remaining open items.
