# Phase 8: Capability Parity Matrix & Categorization - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-08-09
**Phase:** 8-capability-parity-matrix-categorization
**Areas discussed:** Accuracy check, Verdict rubric, Gap categories, Strengths depth

---

## Accuracy check (SC4, Pitfall 12)

| Option | Description | Selected |
|--------|-------------|----------|
| Flag-only (annotate) | Add "accuracy verified?" column; mark fragile areas "present, accuracy not verified" + cite CONCERNS.md; no numeric runs | ✓ |
| Smoke-test top fragile areas | Run fdars vs skfda on ~3-5 fragile/high-use capabilities on one shared dataset, compare within tolerance | |
| Smoke-test all present capabilities | Comparative numeric checks across every "present" capability | |

**User's choice:** Flag-only (annotate)
**Notes:** Keeps Phase 8 pure-documentation; captured a drafted backlog item (D-02a) recommending a real comparative-accuracy validation pass so the deferred verification isn't lost. The Phase-7 scikit-fda venv stays available but unused.

---

## Verdict rubric (SC1, GAP-02, Pitfalls 9/11)

| Option | Description | Selected |
|--------|-------------|----------|
| Defined rubric + source-grep | Explicit present/partial/absent rule; verify each verdict by grepping/reading fdars source | ✓ |
| Defined rubric, STRUCTURE.md only | Same rubric sourced from module map + prior knowledge; spot-check source only for uncertain rows | |
| Binary present/absent only | Drop the "partial" bucket | |

**User's choice:** Defined rubric + source-grep
**Notes:** "partial" bucket retained (Pitfall 11 wants partial ≠ missing). STRUCTURE.md points the search; source confirms each row → higher per-row confidence.

---

## Gap categories (SC2, GAP-03, Pitfall 14)

| Option | Description | Selected |
|--------|-------------|----------|
| Explicit rubric, Claude applies | Define table-stakes vs differentiator; Claude classifies each in-scope gap, reviewer checks section | ✓ |
| Rubric, but you confirm borderline calls | Same rubric; surface ambiguous gaps for a ruling before finalizing | |
| Value-band instead of table-stakes/diff | Replace two-way split with High/Med/Low value band | |

**User's choice:** Explicit rubric, Claude applies
**Notes:** Two-way table-stakes/differentiator split kept as roadmap words it; value ranking stays in Phase 9. Out-of-scope carried from Phase 7 Relevance taxonomy, counted separately.

---

## Strengths depth (SC3, GAP-04)

| Option | Description | Selected |
|--------|-------------|----------|
| Reverse-parity sweep | Full reverse table: every fdars capability with no scikit-fda equivalent | ✓ |
| Four headline areas + brief list | Narrative on the four named strengths + short bullet list of other fdars-only modules | |
| Four headline areas only | Just the four SC3 strengths | |

**User's choice:** Reverse-parity sweep
**Notes:** Walk the STRUCTURE.md module list for fdars-only capabilities (conformal, tolerance bands, GMM, matrix profile, SSA, Hilbert, WIRE, FAMM, changepoint, robust regression, etc.) beyond the four headline areas.

---

## Claude's Discretion

- Matrix layout (master vs per-area tables), column ordering, row confidence tagging.
- Granularity of drafted backlog entries (one per gap vs grouped clusters), kept unranked.
- Ordering within the strengths sweep.

## Deferred Ideas

- Final value-ranking of the gap backlog → Phase 9 (RPT-02).
- Comparative numerical-accuracy smoke-tests → drafted as backlog item (D-02a), not executed this milestone.
- Severity/effort estimates + evidence links per backlog item → Phase 9 (RPT-03).
- Implementing any gap/strength → future milestone seeded by RPT-02 backlog.
