# Phase 18: Reverse-Parity Strengths Sweep - Context

**Gathered:** 2026-08-15
**Status:** Ready for planning
**Mode:** Smart discuss (autonomous) — approach reused from v0.14.0 precedent; independent of the R-side enumeration (may run alongside 16–17, must precede 19).

<domain>
## Phase Boundary

A full module-map walk of `fdars-core/src/` cataloguing where fdars is **unique** (no R equivalent) or **ahead** of its closest R analog. Append `§Reverse-Parity-Strengths` to `.planning/research/R-AUDIT-REPORT.md`. Covers GAP-03. Does NOT re-map R→fdars gaps (Phase 17) or produce the ranked backlog (Phase 19).
</domain>

<decisions>
## Implementation Decisions

### Honesty against R specifically (NOT scikit-fda)
- The v0.14.0 reverse-parity sweep was vs scikit-fda and found 30 fdars-only capabilities. **R is much broader than scikit-fda**, so several of those are NOT fdars-unique against R: e.g. **SPM** exists in R (`funcharts`), **conformal prediction** exists (`conformalInference.fd`), **elastic/shape** exists (`fdasrvf`, broader than fdars per Phase 17). Do NOT copy the scikit-fda strengths list uncritically — re-check each candidate against the Phase-16 R inventory / Phase-17 parity matrix before claiming fdars-unique.
- Two claim types, kept distinct:
  - **fdars-unique (no R equivalent):** no R FDA package delivers the capability. Each row names the closest R analog or "none found" with the search that established it.
  - **fdars-ahead (leads closest R analog):** R has an analog but fdars' version is broader/more integrated; name the R analog and state the nature of the lead. Be fair — where R leads (e.g. `fdasrvf` elastic breadth, `fdapace` sparse FPCA), that belongs in Phase 17 gaps, not here.

### Genuine fdars-unique-vs-R candidates to verify (from module list + Phase 17 present-heavy areas)
- **Model explainability for functional models** (`explain/`, `explain_generic/` — PDP/SHAP/LIME/ALE/importance via the `FpcPredictor` trait): no R FDA package offers model-agnostic explainability for functional models — strong fdars-unique candidate.
- **Streaming / online depth** (`streaming_depth/` — incremental Fraiman-Muniz / MBD): no R streaming-depth equivalent found — verify.
- **WASM/JS bindings** (capability-adjacent; note but likely infra, not a numeric capability).
- Check `andrews.rs`, `wire.rs`, `metric/`, `detrend/`, `elastic_explain.rs`, `elastic_changepoint.rs` for uniqueness vs R.

### Method
- Walk EVERY module in `fdars-core/src/` (per-module coverage documented, so the catalogue is demonstrably exhaustive, not cherry-picked). Use `.planning/codebase/` maps + PROJECT.md "Validated" list + direct `ls`/grep.
- For each candidate strength, cross-check the Phase-16 R inventory + Phase-17 parity matrix to confirm no R equivalent (or a genuine lead). Cite the R search.

### Claude's Discretion
- Table layout and grouping; how many "ahead" vs "unique" rows to include, provided the module walk is exhaustive and every claim is R-checked.
</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `.planning/research/AUDIT-REPORT.md` §Phase 8 reverse-parity sweep (30 fdars-only-vs-scikit-fda capabilities) — a candidate list to RE-VET against R (many will NOT survive: SPM, conformal, elastic).
- `.planning/research/R-AUDIT-REPORT.md` §R-Inventory (Phase 16) + §Parity-Matrix (Phase 17) — the R capability surface to check "no R equivalent" claims against; Phase 17's present-heavy areas (SPM 9/10, Preprocessing 16/22, Representation 20/38) hint where fdars leads.
- `fdars-core/src/` module tree (walk target) + `.planning/codebase/` maps + PROJECT.md "Validated" list.

### Established Patterns
- Append to `.planning/research/R-AUDIT-REPORT.md` (do NOT overwrite Phases 16/17; do NOT touch the archived scikit-fda `AUDIT-REPORT.md`). Audit-only: zero `fdars-core/src/` edits.

### Integration Points
- §Reverse-Parity-Strengths feeds Phase 19's consolidated findings (keeps the backlog honest: don't propose building what fdars already leads on).
</code_context>

<specifics>
## Specific Ideas

- Report anchors: `## Phase 18 — Reverse-Parity Strengths Sweep`, with a per-module coverage table (proving exhaustiveness) and two catalogues: fdars-unique (no R equivalent) and fdars-ahead (leads closest R analog).
</specifics>

<deferred>
## Deferred Ideas

- Consolidated findings + value-ranked backlog → Phase 19.
</deferred>
