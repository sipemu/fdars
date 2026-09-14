# Phase 107: SPM Complexity Cluster - Context

**Gathered:** 2026-09-14
**Status:** Ready for planning
**Mode:** Auto-generated (infrastructure/refactor phase — discuss skipped)

<domain>
## Phase Boundary

Decompose the concentrated complexity cluster inside `fdars-core/src/spm/`
(statistical process monitoring) into smaller **private** helper functions so
each falls under pmat's cognitive (≤25) / cyclomatic (≤30) thresholds —
**behavior and numeric output unchanged**. Pure internal refactor: no public
API / signature / visibility changes.

Roadmap-named SPM hotspots (verify + expand via live pmat during research):
`mfpca`, `spm_phase1_iterative`, `spm_mewma_monitor`, `select_ncomp`,
`conditional_expectation` / partial variants, `spm_monitor_from_fields` /
`spm_phase*`, `profile_phase1`, `spe_multivariate` / stats. Post-Phase-106
pmat showed `spm/amewma.rs`, `spm/mfpca.rs`, `spm/iterative.rs` among the top
files — all `src/spm/`.

Out of scope: the top-level hotspots already done in Phase 106, entropy/SATD/
unwrap cleanup (Phase 108), and the remaining complexity tail (Phase 109).

</domain>

<decisions>
## Implementation Decisions

### Claude's Discretion
All implementation choices are at Claude's discretion — pure refactor phase.
Constraints (non-negotiable, from PROJECT.md / STATE.md):

- **Strictly non-breaking + numeric-preserving.** Pure private helper extraction.
  No public API / signature / visibility change (protects R + WASM bindings).
- **Reduce complexity by helper extraction only** — never by changing signatures.
- **Preserve numeric identity** — do NOT reorder floating-point accumulations;
  lift loop bodies verbatim. SPM has control charts / MEWMA / MFPCA with
  sequential recurrences and possible RNG — preserve update order exactly.
- **`fdars-core` only. No new crate dependency** (pmat is external dev/CI tooling).
- Proof = existing suite green (2942 lib + 215 doctests + 28 examples,
  numeric-identical) + `pmat analyze complexity` confirming each SPM target
  drops below threshold and every new helper is itself under threshold.

### Proven pattern (from Phase 106)
Follow the 106 tracer loop: extract private helper → run touched-module tests →
`pmat analyze complexity` confirms drop → `--no-verify` commit → out-of-band
foreground gates. Split over-large validators into `_validate_dims` +
`_validate_config` sub-helpers when a single validator itself exceeds cog 25.

</decisions>

<code_context>
## Existing Code Insights

- `src/spm/` is split into many submodule files (per project structure:
  `phase.rs`, `amewma.rs`, `mfpca.rs`, `iterative.rs`, etc.).
- Column-major `FdMatrix` layout; use existing row methods where relevant.
- SPM update loops (EWMA/MEWMA recurrences, phase-1 iterative estimation) are
  order-sensitive — helpers must preserve the exact sequential update order.
- Precise file:line spans + live pmat scores to be pinned during plan-phase
  research (pmat identifies the offending functions).

</code_context>

<specifics>
## Specific Ideas

No specific requirements beyond the named SPM functions and pmat thresholds.
The pmat complexity report (restricted to `src/spm/`) is the authoritative
target list; each flagged SPM function must fall under pmat's cyclomatic +
cognitive thresholds after decomposition.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>
