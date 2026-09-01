# Requirements: fdars — v0.31.0 Multi-Ecosystem Gap Audit

**Defined:** 2026-09-01
**Core Value:** Produce an evidence-backed picture of what fdars is missing relative to four fresh reference ecosystems (MATLAB FDA, Julia FDA, tidyfun/refund, Python-beyond-scikit-fda), turned into a single prioritized, de-duplicated backlog — so future milestones target the highest-leverage net-new capability work first.

## Milestone v0.31.0 Requirements

Audit-only. Every requirement is a **deliverable document**, not `fdars-core/src/` code. Each survey requirement bundles the same four sub-tasks: **enumerate** the reference ecosystem's capability surface, **map** fdars present/partial/absent against it, **de-duplicate** against shipped fdars capabilities and the prior `BACKLOG.md` (v0.14.0) + `R-BACKLOG.md` (v0.18.0), and emit a **net-new gap list**.

### Ecosystem Surveys

- [ ] **MAT-01**: Survey MATLAB FDA — enumerate the Ramsay `fda` MATLAB toolbox + PACE (MATLAB) capability surface (versioned), map fdars present/partial/absent against it, and produce a de-duplicated net-new gap list for MATLAB FDA.
- [ ] **JUL-01**: Survey Julia FDA — enumerate the JuliaStats / functional-data package capability surface (versioned), map fdars present/partial/absent against it, and produce a de-duplicated net-new gap list for Julia FDA (capturing modern/performance-oriented patterns as candidate gaps).
- [ ] **TDY-01**: Survey tidyfun/refund (R) — enumerate the tidyfun data-representation & workflow slice plus refund methods **not already captured in v0.18.0** (versioned), map fdars present/partial/absent against it, and produce a de-duplicated net-new gap list.
- [ ] **PYX-01**: Survey Python-beyond-scikit-fda — enumerate FDApy / tslearn / sktime functional components + other Python FDA/ML libs (versioned), map fdars present/partial/absent against them, and produce a de-duplicated net-new gap list (excluding scikit-fda, covered by v0.14.0).

### Consolidation & Backlog

- [ ] **RPT-01**: Consolidated multi-ecosystem gap report (`GAP-AUDIT-REPORT.md`) — methodology, per-ecosystem findings, cross-ecosystem overlap/convergence analysis (which gaps recur across ecosystems), and a reverse-parity strengths sweep (where fdars leads these ecosystems).
- [ ] **RPT-02**: Ranked, GSD-ready backlog (`GAP-BACKLOG.md`) — every net-new gap scored by value/effort (consistent with the v0.14.0/v0.18.0 `value/√effort` convention), sorted, and expressed as a promotion-ready item block (candidate requirement/phase, effort estimate, reference baseline, rationale).
- [ ] **RPT-03**: De-dup & completeness gate — verify every backlog item is genuinely net-new (not shipped in fdars, not already in `BACKLOG.md` or `R-BACKLOG.md`) and that every surveyed capability gap is either ranked in `GAP-BACKLOG.md` or explicitly recorded as out-of-scope with reasoning. Deliverables written with **zero `fdars-core/src/` edits**.

## Future Requirements

Deferred to future milestones — not in this roadmap.

- Implementing any gap found in this audit — drawn top-first from `GAP-BACKLOG.md` in later implementation milestones.
- Crate-release-hardening / 1.0-readiness pass — candidate for the milestone after this one.
- APIB-01 (breaking removal of the 6 `#[deprecated]` forms from v0.30.0) — needs a version willing to break bindings.

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Any `fdars-core/src/` code change | Audit-only milestone (report + backlog), same fence as v0.14.0/v0.18.0 |
| Git tag / crate publish | Crate is unchanged; a `v*` tag would publish a phantom version (project convention) |
| Re-auditing scikit-fda | Actionable backlog exhausted (v0.15.0–v0.17.0); measure against fresh yardsticks only |
| Re-auditing the core R FDA ecosystem (`fda`/`fda.usc`/`refund`/`fdapace`/`roahd`/`ftsa`/`frechet`) | Actionable backlog exhausted (v0.19.0–v0.29.0); tidyfun/refund pass touches refund **only** where not captured in v0.18.0 |
| Re-listing gaps already in `BACKLOG.md` / `R-BACKLOG.md` or capabilities fdars already ships | Hard de-dup rule — value is in net-new gaps only |
| Plotting/visualization parity (MATLAB/R/Python graphical output) | Numeric Rust library needs the statistics under a plot, not the rendering; the numeric underpinnings may be gaps, the plots are not |
| Data/IO parity (dataset loaders, read/write round-trips) | Consistent with prior audit fences |

## Traceability

Which phases cover which requirements. Populated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| MAT-01 | Phase 52 | Pending |
| JUL-01 | Phase 52 | Pending |
| TDY-01 | Phase 52 | Pending |
| PYX-01 | Phase 52 | Pending |
| RPT-01 | Phase 53 | Pending |
| RPT-02 | Phase 53 | Pending |
| RPT-03 | Phase 53 | Pending |

**Coverage:**
- v0.31.0 requirements: 7 total
- Mapped to phases: 7 (Phase 52: 4 surveys · Phase 53: 3 consolidation)
- Unmapped: 0 ✓

---
*Requirements defined: 2026-09-01*
*Last updated: 2026-09-02 after roadmap creation (phases 52–53 mapped)*
