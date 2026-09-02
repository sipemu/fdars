# Phase 53: Consolidation & Backlog - Context

**Gathered:** 2026-09-02
**Status:** Ready for planning

<domain>
## Phase Boundary

Merge the four per-ecosystem survey gap lists (`survey-matlab.md`, `survey-julia.md`, `survey-tidyfun.md`, `survey-pyx.md` from Phase 52) into a single cross-ecosystem gap report (`GAP-AUDIT-REPORT.md`, RPT-01), produce a value-ranked GSD-ready backlog (`GAP-BACKLOG.md`, RPT-02), and run a de-dup + completeness gate (RPT-03, LAST) confirming every backlog item is genuinely net-new and every surveyed gap is either ranked or explicitly recorded out-of-scope. Audit-only: zero `fdars-core/src/` edits. Deliverables are markdown under `.planning/research/`.

Internal order is fixed: RPT-01 → RPT-02 → RPT-03 (gate last).

</domain>

<decisions>
## Implementation Decisions

### Scoring & Backlog Format
- **Ranking formula**: `value/√effort`, consistent with the v0.14.0 `BACKLOG.md` and v0.18.0 `R-BACKLOG.md` conventions.
- **Scales**: reuse prior scales — value 1–5, effort S/M/L (map S=1, M=2, L=3 for the √effort denominator).
- **Backlog item block (promotion-ready)**: each `GAP-BACKLOG.md` row/block carries a candidate requirement/phase ID, effort estimate, reference baseline (`pkg@version`), rationale, and source ecosystem(s).
- **Cross-ecosystem convergence**: gaps recurring across ≥2 ecosystems are flagged in RPT-01 and get a priority/score boost in RPT-02 (recurrence is evidence of value).

### Completeness Gate (RPT-03) Disposition
- **Flagged out-of-scope candidates** from Phase 52 (Julia GPU kernels; Python SAX/PAA/bag-of-patterns symbolic representations) are recorded EXPLICITLY as out-of-scope with reasoning in RPT-03 — satisfying "every surveyed gap is either ranked or explicitly recorded out-of-scope". They do NOT become backlog rows.
- **Gate ordering**: RPT-01 → RPT-02 → RPT-03, gate LAST.
- **Independent de-dup re-verification**: RPT-03 re-asserts each backlog row absent from shipped fdars AND both `BACKLOG.md` and `R-BACKLOG.md` (a second independent pass, not merely trusting the survey de-dup).
- **Deliverable filenames**: `GAP-AUDIT-REPORT.md` + `GAP-BACKLOG.md`. NEVER overwrite existing `AUDIT-REPORT.md`/`BACKLOG.md` (v0.14.0) or `R-AUDIT-REPORT.md`/`R-BACKLOG.md` (v0.18.0).

### Claude's Discretion
- Exact section ordering within `GAP-AUDIT-REPORT.md`, tie-breaking among equal scores, and prose framing are at Claude's discretion, guided by the prior audit reports' shape.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- **Phase 52 survey files** (the inputs): `.planning/research/survey-{matlab,julia,tidyfun,pyx}.md`, each with a standardized six-column Net-New Gap List (`Capability | Reference (pkg@ver) | fdars status | Searched-for | Net-new? | Notes`) — the mechanical merge contract.
- **Prior audit reports/backlogs as templates**: `AUDIT-REPORT.md`/`BACKLOG.md` (v0.14.0), `R-AUDIT-REPORT.md`/`R-BACKLOG.md` (v0.18.0) — reuse table shape, scoring methodology section, and completeness-gate assertion style.
- **PROJECT.md Validated section** + `fdars-core/src/` — the de-dup baseline for the RPT-03 independent re-verification.

### Candidate net-new gaps carried from Phase 52 (raw material)
- MATLAB (1): FOptDes optimal experimental design.
- Julia (2): differentiable/autodiff FDA core (solid); GPU kernels (flagged out-of-scope).
- tidyfun/refund (2): PEER/lpeer; wcr/wnet wavelet functional regression.
- Python (4): shapelets; k-Shape (SBD); GAK kernel; multi-domain MFPCA (partial). Plus SAX/symbolic (flagged out-of-scope).
- **Total: 7 solid net-new + 2 flagged out-of-scope = 9 surveyed candidates.**

### Integration Points
- RPT-01/02/03 are the terminal milestone deliverables; RPT-03 is the completeness gate that closes the milestone.

</code_context>

<specifics>
## Specific Ideas

- Cross-ecosystem convergence to check explicitly: shapelets/shape-based methods (Python), differentiable/perf architecture (Julia) — note whether any gap independently appears in more than one ecosystem survey.
- No git tag / no crate publish (crate unchanged — audit-milestone convention).

</specifics>

<deferred>
## Deferred Ideas

- Implementing any ranked gap — drawn top-first from `GAP-BACKLOG.md` in a future implementation milestone.

</deferred>
