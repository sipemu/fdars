# Phase 109: Complexity Tail & Quality-Gate Green - Context

**Gathered:** 2026-09-14
**Status:** Ready for planning
**Mode:** Auto-generated (infrastructure/refactor phase — discuss skipped)

<domain>
## Phase Boundary

Clear **every remaining complexity violation** in `fdars-core` so the curated
milestone quality-gate reports **0 findings**:

```
pmat quality-gate --checks complexity,satd,dead-code,duplicates --fail-on-violation
```

(Entropy is EXCLUDED per the operator decision — it is stochastic; see STATE.md
"Decisions" / [[v0460-gate-definition-decision]].)

**Scope: 97 complexity violations across 56 files** (full list saved at
`.planning/phases/109-complexity-tail-quality-gate-green/109-complexity-violations.txt`,
each line = `./src/<file>:<line> - <function>: cognitive|cyclomatic-complexity ...
(complexity: N, threshold: T)`). This is the largest phase of the milestone —
bigger than 106+107 combined.

**CRITICAL — the gate's threshold is STRICTER than `pmat analyze complexity`:**
- `pmat quality-gate` flags **cognitive > 23** ("recommended") AND > 25
  ("maximum"), plus cyclomatic > (its threshold). Both severities COUNT toward
  the gate. So the internal target for EVERY function is **cognitive ≤ 23 and
  cyclomatic ≤ 30** — NOT the ≤25 that phases 106/107 used.
- Two residual functions from prior phases are in the list and must be finished:
  `funfem_cluster` (clustering_advanced.rs, cog 25) and
  `pace_fpca_validate_config` (pace_fpca.rs, cog 24).

Concentrations (files with the most violations): `peer.rs` (5),
`clustering_advanced.rs` (5), `helpers.rs` (4), `coclustering.rs` (4);
many files have 1–3. `alignment/*` is heavily represented.

Out of scope: the CI guardrail (Phase 110), release (Phase 111). Depends on
106+107+108 having landed (they have).

</domain>

<decisions>
## Implementation Decisions

### Claude's Discretion
All implementation choices are at Claude's discretion — pure refactor phase.
Constraints (non-negotiable):

- **Strictly non-breaking + numeric-preserving.** Pure private helper extraction;
  no public API / signature / visibility change (protects R + WASM bindings).
  No FP-accumulation reorder — lift loop bodies verbatim. Preserve any RNG draw
  order and sequential recurrences exactly (per-function hazard check).
- **Internal target = cognitive ≤ 23, cyclomatic ≤ 30.** Verify EACH extracted
  helper is itself under threshold (pmat counts patterns inside helpers too —
  a helper with 4 similar fold lines can still flag; collapse when needed).
  Use the proven split-validate pattern (`_validate_dims` + `_validate_config`)
  when a single validator exceeds threshold.
- **`fdars-core` only. No new crate dependency.**
- Proof = existing suite green (2942 lib + 215 doctests + 28 examples,
  numeric-identical) + `pmat quality-gate --checks complexity,satd,dead-code,duplicates`
  reporting 0.

### Proven pattern (106/107/108)
Extract private helper → run touched-module tests → `pmat analyze complexity`
confirms the function AND its new helpers are under threshold → `--no-verify`
commit → out-of-band foreground gates. Group by file/module proximity, zero
file-overlap between plans.

</decisions>

<code_context>
## Existing Code Insights

- 97 functions, 56 files — spread across nearly every module (alignment/,
  scalar_on_function/, boosting_regression/, gmm/, conformal/, inference/,
  explain_generic/, shapelet/, fts/, frechet/, irreg_fdata/, plus many
  top-level files like peer.rs, coclustering.rs, helpers.rs, famm.rs,
  fpca_variants.rs, fof_regression.rs, density_fda.rs).
- `helpers.rs` (4 violations) is shared infrastructure — extra care that
  helper extraction there stays private and doesn't perturb callers.
- The exact per-function score + threshold is in the saved violations file;
  the planner/researcher must group these into a manageable set of plans
  (each ~4–6 functions / a few proximate files) — NOT one giant plan
  (executor-stall risk) and NOT 97 thin plans.

</code_context>

<specifics>
## Specific Ideas

No specifics beyond the saved 97-violation list and the cog ≤23 / cyc ≤30
target. After this phase, `pmat quality-gate --checks complexity,satd,dead-code,duplicates`
must report **0 violations** — the milestone's central verifiable exit.

</specifics>

<deferred>
## Deferred Ideas

Entropy (stochastic, excluded from the gate) — already reduced in Phase 108;
not re-addressed here.

</deferred>
