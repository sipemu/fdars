# Requirements: fdars — v0.18.0 R-Ecosystem Gap Audit

**Defined:** 2026-08-13
**Core Value:** A comprehensive, fast Rust functional-data-analysis library that closes the highest-leverage capability and performance gaps against the reference FDA ecosystems — this milestone maps fdars against the R FDA package ecosystem to produce the next evidence-backed, prioritized backlog.

Audit-only milestone. Deliverables are a report + backlog; **zero `fdars-core/src/` code changes** (mirrors v0.14.0). The R FDA ecosystem replaces scikit-fda as the sole comparison yardstick, now that the actionable scikit-fda backlog is exhausted (v0.15.0–v0.17.0).

## v1 Requirements

Requirements for this milestone. Each maps to a roadmap phase.

### R Ecosystem Inventory

- [ ] **INV-01**: Produce a versioned, area-organized capability inventory of the R FDA ecosystem — enumerating capabilities **capability-first** (not API-name-first; fit/predict/transform collapsed per capability) across the core packages (`fda`, `fda.usc`, `refund`, `fdapace`, `roahd`, `fdaoutlier`, `ftsa`, `MFPCA`/`funData`, `fdasrvf`, `fdatest`/`fdANOVA`, `frechet`/`fdadensity`, `funHDDC`/`FDboost`, and any further packages surfaced during research), each capability tagged with its source package and package version.
- [ ] **INV-02**: Apply a design-goal filter classifying every inventoried capability as **in-scope** (numeric algorithm or API-ergonomics) or **out-of-scope** (plotting/visualization or data/IO), with per-area counts — establishing the actionable comparison surface for the parity matrix.

### Gap Analysis

- [ ] **GAP-01**: Produce an fdars-vs-R capability **parity matrix** — a per-capability verdict (**present / partial / absent**) mapping each in-scope R capability to fdars, **matched by capability not API name**, each row carrying a "searched fdars for:" evidence note and closest-match reference.
- [ ] **GAP-02**: **Categorize** every absent/partial gap as **table-stakes / differentiator / out-of-scope**, with rationale, to drive value ranking.
- [ ] **GAP-03**: Produce a **reverse-parity strengths sweep** — catalog fdars capabilities with no R equivalent, or where fdars is ahead of its closest R analog (e.g. elastic/shape vs `fdasrvf`; SPM, explainability, streaming depth, conformal, tolerance bands with no R counterpart), from a full module-map walk.

### Report & Backlog

- [ ] **RPT-01**: Produce a **consolidated R-ecosystem gap report** — methodology (packages + versions surveyed, in/out-of-scope rule, verdict and category rubrics) plus consolidated findings (gap counts by area and category, and the fdars-strengths summary).
- [ ] **RPT-02**: Produce a **GSD-ready value-ranked backlog** — `score = value / √effort` methodology (value 1–5, effort S/M/L, severity P1/P2/P3), a master ranked table (strictly non-increasing), and a 7-field promotion-ready block per candidate item, ready to promote via `/gsd-new-milestone`.

## v2 Requirements

Deferred to future milestones.

### Implementation

- **IMPL-\***: Implementation of any R-parity gap surfaced by this audit — the whole point of the backlog is to feed subsequent implementation milestones top-first. Not actionable until the audit completes.

### Accuracy Validation

- **ACC-VALIDATE** (carried from v0.14.0): Comparative fdars-vs-reference numerical-accuracy validation — could be extended to R references (e.g. `fda`/`fdapace`/`fdasrvf`) once the parity picture identifies the fragile areas.

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Implementing any R-parity gap found | Audit-only milestone — deliverables are report + backlog; implementation is deferred to future milestones to keep scope bounded and decisions evidence-driven |
| Plotting/visualization parity with R (`rainbow`, `roahd`/`fdaoutlier` graphics, `fda` plot methods) | A numeric Rust library needs the underlying statistics, not base-R/ggplot rendering; the numeric underpinnings of graphical diagnostics (e.g. outliergram/MS-plot statistics) may be in-scope but the plots are not |
| Data/IO parity (R dataset loaders, read/write round-trips) | Out of scope, consistent with the v0.14.0 audit fence |
| Runtime performance benchmark against R | This is a capability/API gap comparison, not a cross-language speed contest; R is interpreted and not a meaningful perf baseline |
| Re-auditing scikit-fda | Its actionable backlog is exhausted (v0.15.0–v0.17.0); R is the sole yardstick for this milestone |

## Traceability

Which phases cover which requirements. Populated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| INV-01 | Phase [N] | Pending |
| INV-02 | Phase [N] | Pending |
| GAP-01 | Phase [N] | Pending |
| GAP-02 | Phase [N] | Pending |
| GAP-03 | Phase [N] | Pending |
| RPT-01 | Phase [N] | Pending |
| RPT-02 | Phase [N] | Pending |

**Coverage:**
- v1 requirements: 7 total
- Mapped to phases: 0 (pending roadmap)
- Unmapped: 7 ⚠️

---
*Requirements defined: 2026-08-13*
*Last updated: 2026-08-13 after initial definition (v0.18.0 R-Ecosystem Gap Audit)*
