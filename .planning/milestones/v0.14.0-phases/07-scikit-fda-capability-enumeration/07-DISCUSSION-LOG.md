# Phase 7: scikit-fda Capability Enumeration - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-08-09
**Phase:** 7-scikit-fda Capability Enumeration
**Areas discussed:** Version pin & verification, Existing research reuse, Capability unit/grouping, Design-goal filter

---

## Version pin & verification (SC2)

| Option | Description | Selected |
|--------|-------------|----------|
| Hybrid w/ fallback | Attempt throwaway venv install of scikit-fda==0.10.1 to capture `skfda.__version__` + spot-check `dir()`; fall back to PyPI metadata + readthedocs docs if Py3.14 install fails; record which path was used | ✓ |
| Live install required | Mandate working venv install + programmatic introspection as evidence base | |
| Docs + PyPI metadata only | Verify 0.10.1 is latest PyPI release + cite readthedocs, no install | |

**User's choice:** Hybrid w/ fallback (recommended)
**Notes:** 0.10.1 confirmed latest on PyPI (network/pip verified working); scikit-fda not currently installed; Python 3.14.5 present (may break a clean install of a 2024-era release). Hybrid captures the literal `__version__` evidence when possible without blocking the phase on install compat. → CONTEXT D-01/D-01a.

---

## Existing research reuse (FEATURES.md)

| Option | Description | Selected |
|--------|-------------|----------|
| Promote & refactor | Extract scikit-fda-only enumeration, strip fdars gap notes (Phase 8), re-verify against 0.10.1, formalize as authoritative report section | ✓ |
| Fresh enumeration | Re-enumerate from scratch into AUDIT-REPORT.md, FEATURES.md as cross-check only | |
| Use as-is + pin | Keep FEATURES.md content, add version pin + design filter only | |

**User's choice:** Promote & refactor (recommended)
**Notes:** A 37KB `.planning/research/FEATURES.md` already enumerates the full 0.10.1 surface across 12 sub-areas at MEDIUM confidence but mixes in fdars gap notes. Promotion reuses verified work; the strip step keeps the scikit-fda-only deliverable clean and defers fdars mapping to Phase 8. → CONTEXT D-02.

---

## Capability unit / grouping (Pitfall 9, SC3)

| Option | Description | Selected |
|--------|-------------|----------|
| Task-grouped, method-family rows | Two-level: task-area headers + one row per distinct method, collapsing fit/predict/transform/inverse into one row | ✓ |
| User-task granularity only | One row per user task, all method variants collapsed (~20-30 rows) | |
| Method/API granularity | One row per public class/estimator | |

**User's choice:** Task-grouped, method-family rows (recommended)
**Notes:** Gives Phase 8's parity matrix enough granularity to mark parity per method (e.g. "Nadaraya-Watson: absent") without drifting into the API-name counting Pitfall 9 warns against. → CONTEXT D-03.

---

## Design-goal filter (SC4, Pitfall 14)

| Option | Description | Selected |
|--------|-------------|----------|
| 4-value Relevance taxonomy | In-Scope Algorithm / In-Scope API-Ergonomics / Out-of-Scope (plotting) / Out-of-Scope (IO); explicit borderline rulings; separate in/out counts | ✓ |
| Binary in/out-of-scope | Each capability just in- or out-of-scope with one-line reason | |

**User's choice:** 4-value Relevance taxonomy (recommended)
**Notes:** Directly consumed by Phase 8's GAP-03. Borderline rulings decided: plotting/Visualization → Out (plotting); DataFrame/dataset loaders → Out (IO); sklearn-pipeline plumbing → Out; representation type-system refactor → Out, but algorithmic capabilities riding on it (e.g. irregular-data covariance) → In-Scope Algorithm. → CONTEXT D-04.

---

## Claude's Discretion

- Exact table column layout (Area | Task | Method | collapsed-note | Relevance | Confidence | Source).
- venv mechanics (`python -m venv`+`pip` vs `uv`/`pipx` ephemeral env); document exact command + outcome.
- Granularity of `__version__`-verification evidence recorded, depending on which D-01 path succeeds.
- Ordering/merging of the 12 FEATURES.md sub-areas under the six SC1 report areas.

## Deferred Ideas

- fdars-side parity mapping (present/partial/absent) — Phase 8 (GAP-02).
- Gap categorization (table-stakes/differentiator/out-of-scope) + fdars strengths — Phase 8 (GAP-03/04).
- Numerical-accuracy spot-checks vs scikit-fda reference datasets — Phase 8 CONCERNS handling.
- Re-pinning to a newer scikit-fda if released — keep 0.10.1 for milestone consistency.
