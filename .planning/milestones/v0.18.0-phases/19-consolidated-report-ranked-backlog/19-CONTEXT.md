# Phase 19: Consolidated Report & Ranked Backlog - Context

**Gathered:** 2026-08-15
**Status:** Ready for planning
**Mode:** Smart discuss (autonomous) — methodology reused verbatim from the v0.14.0 audit; final synthesis phase.

<domain>
## Phase Boundary

Synthesize Phases 16–18 into the two milestone deliverables:
- **RPT-01:** a consolidated R-ecosystem gap report — append `§Methodology (Consolidated)` + `§Consolidated Findings` to `.planning/research/R-AUDIT-REPORT.md`.
- **RPT-02:** a GSD-ready, value-ranked backlog in a NEW file `.planning/research/R-BACKLOG.md` (distinct from the archived scikit-fda `BACKLOG.md`).

This is the last phase of milestone v0.18.0. Consumes the R inventory (16), parity matrix + categorization (17), and strengths sweep (18). No new gap discovery.
</domain>

<decisions>
## Implementation Decisions

### Ranking methodology (reuse v0.14.0 verbatim)
- **Formula:** `score = value / sqrt(effort)`.
- **Value 1–5:** 5 = table-stakes blocking real workloads; 4 = high-value widely used; 3 = meaningful; 2 = useful/niche; 1 = cosmetic.
- **Effort S/M/L → sqrt:** S=1 (1.000, ~1 wk), M=3 (1.732, ~2–4 wk), L=9 (3.000, ~1–3 mo).
- **Severity P1/P2/P3.** Ties in score broken by severity (P1 before P2 before P3).
- Master ranked table **strictly non-increasing** by score.

### Backlog item granularity (cluster the 162 actionable gaps)
- Do NOT emit 162 one-line items. Cluster the 162 actionable gaps (18 table-stakes + 144 differentiator, from Phase 17 §Categorization) into ~20–35 **coherent, GSD-ready candidate requirements/phases**, each milestone-promotable. Table-stakes gaps rank high; big differentiator clusters become milestone-sized items.
- Natural cluster candidates (from Phase 17 findings): **functional inference suite** (Area 5 — the dominant table-stakes deficit, 0 present: two-sample/ANOVA/permutation/SCB/FLM-GoF tests); **Fréchet / object-data & density regression** (Area 7 — 0 present, `frechet`/`fdadensity`); **functional time series forecasting** (Area 6 — 2/25, `ftsa`: FTS forecasting, functional ACF, spectral DPCA, FARMA); **sparse/PACE FPCA + conditional-expectation scores** (Area 9, `fdapace`); **function-on-function regression** (verify vs existing `fof_regression.rs` — may be partial not absent); **concurrent/varying-coefficient regression + GLM families** (Area 4); plus smaller table-stakes items (constant basis, AIC smoothing selection, depth dispatcher, functional boxplot fences, FEM/PDE smoothing).

### 7-field promotion blocks (mirror v0.14.0 BACKLOG.md)
Each ranked item carries: (1) **candidate requirement / phase phrasing**; (2) **R-side reference** (which R package(s)/capability it maps to + the parity-matrix rows it covers); (3) **fdars current gap** (absent vs partial, and what exists today); (4) **proposed direction** (where in `fdars-core/src/` + sketch of approach); (5) **value + effort + severity + category** (table-stakes/differentiator); (6) **score**; (7) **notes/dependencies**. Keep blocks concrete so items promote straight into `/gsd-new-milestone`.

### Honesty carry-forwards
- **Do NOT propose building fdars' existing strengths** (Phase 18: explainability, streaming depth, SPM breadth, conformal breadth, robust SoF, 2D-FOSR, signal toolkit).
- **Note the `Rfssa` caveat** (functional SSA exists in R, missed by the Phase-16 35-package survey) in §Methodology (Consolidated) as a known inventory-completeness limitation.
- Consolidated Findings must include: gap counts by area + category, the strengths summary (12 R-honest strengths), and the headline numbers (35 pkgs, 275 caps, 248 in-scope, 162 actionable gaps, 18 table-stakes).

### Completeness gate (reuse v0.14.0)
- Verify: ≥N P1/table-stakes items present, top items non-cosmetic, master table strictly descending, every ranked item has a matching 7-field block.

### Claude's Discretion
- Exact number of backlog items, cluster boundaries, and value/effort assignments per item, provided the methodology is applied consistently and the completeness gate passes.
</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `.planning/research/R-AUDIT-REPORT.md` — Phases 16 (inventory), 17 (parity matrix + categorization, the 162 actionable gaps), 18 (strengths). The whole input.
- `.planning/research/BACKLOG.md` (v0.14.0) — the 7-field block format + ranking methodology to mirror exactly (new file `R-BACKLOG.md`, do NOT overwrite).
- `.planning/research/AUDIT-REPORT.md` §Phase 9 §Consolidated Findings — the RPT-01 structure to mirror.

### Established Patterns
- Deliverables in `.planning/research/`: append RPT-01 to `R-AUDIT-REPORT.md`; create `R-BACKLOG.md` for RPT-02. Do NOT touch the archived scikit-fda `AUDIT-REPORT.md`/`BACKLOG.md`. Audit-only: zero `fdars-core/src/` edits.

### Integration Points
- `R-BACKLOG.md` is consumed by the NEXT `/gsd-new-milestone` to promote top items into implementation milestones — must be promotion-ready.
</code_context>

<specifics>
## Specific Ideas

- Report anchors: `## Phase 19 — Consolidated Report` with `### §Methodology (Consolidated)` + `### §Consolidated Findings`.
- `R-BACKLOG.md`: header + ranking methodology + master ranked table (strictly non-increasing) + per-item 7-field blocks + a completeness-gate statement.
</specifics>

<deferred>
## Deferred Ideas

None — this is the final phase. Implementation of ranked items is deferred to future milestones (out of scope for this audit).
</deferred>
