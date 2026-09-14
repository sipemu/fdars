# Phase 108: Entropy Extraction & SATD/Unwrap Cleanup - Context

**Gathered:** 2026-09-14
**Status:** Ready for planning
**Mode:** Auto-generated (infrastructure/cleanup phase — discuss skipped)

<domain>
## Phase Boundary

Three distinct **behavior-preserving** cleanups that clear the non-complexity
`pmat quality-gate` findings (requirements QUAL-04 + QUAL-05):

1. **Entropy (QUAL-04):** Extract the repeated `DataTransformation` patterns
   pmat entropy flags (baseline: 11) into shared helpers so pmat entropy → 0.
   Post-107 `pmat quality-gate` still reports DataTransformation patterns in
   (at least): `elastic_regression/scalar_on_shape.rs` (7×), `fdata.rs` (7×),
   `regression.rs` (6×, a `///` doctest example line), `covariance.rs` (6×),
   `explain_generic/tests.rs` (6×), `scalar_on_function/tests.rs` (6×),
   `detrend/tests.rs` (6×), plus a project-level "Low pattern diversity 76.9%".
   Several are in TEST modules and doc-comments — handle each in place.
2. **SATD (QUAL-05a):** Reword the 2 doc-comment FALSE POSITIVES so pmat's
   keyword matcher stops flagging them (NO code/behavior change — prose only):
   - `depth/dispatch.rs:281` — "Defect: Deepest 50% of rows …" (the word
     "Defect:" in a `///` triggers SATD).
   - `metric/lp.rs:175` — "Performance: to ensure LLVM can fully optimize …"
     (the word "Performance:" in a `///` triggers SATD).
3. **Unwrap (QUAL-05b):** Replace the 1 critical `.unwrap()` at
   `depth/rpd.rs:229` with proper error handling (`?` or `.expect()` with a
   descriptive message) — behavior-preserving on valid input (crate norm is
   `Result<T, FdarError>`, no panics on validated input).

Out of scope: the complexity tail (Phase 109 drives `pmat quality-gate` fully
green), CI guardrail (110), release (111). Independent of the 106/107
complexity refactors — could run in any order.

</domain>

<decisions>
## Implementation Decisions

### Claude's Discretion
All implementation choices are at Claude's discretion — pure cleanup phase.
Constraints (non-negotiable, from PROJECT.md / STATE.md):

- **Strictly behavior-preserving + numeric-preserving.** Entropy extraction =
  factor repeated `.map(...)`/transformation closures into a shared private
  helper (or a test-helper fn), CALLED identically — same computation, same
  numeric result. Do NOT reorder floating-point accumulations. SATD is prose-
  only (zero code change). The unwrap fix keeps the same result on valid input.
- **No public API / signature / visibility change** where it can be avoided;
  new entropy helpers are private `fn` (or `#[cfg(test)]` test helpers). If a
  shared helper must be crate-visible to be reused across modules, prefer a
  private module-local helper per file, or a `pub(crate)` helper only if truly
  shared — but never widen an existing PUBLIC signature.
- **`fdars-core` only. No new crate dependency.**
- Proof = existing suite green (2942 lib + 215 doctests + 28 examples,
  numeric-identical) + `pmat analyze entropy` → 0, `pmat analyze satd` → 0,
  `pmat analyze defects` (the unwrap) → 0.

### SATD rewording guidance
Keep the doc-comment's MEANING; just drop the leading keyword token pmat
matches. E.g. "Defect: Deepest 50% …" → "Deepest 50% …" or "Note: deepest
50% …"; "Performance: to ensure LLVM …" → "Ensures LLVM …". Verify with
`pmat analyze satd` after each edit that the file no longer flags.

### Unwrap fix guidance
`depth/rpd.rs:229` — inspect what the `.unwrap()` unwraps. If it is on a value
that is guaranteed non-None/Ok by prior validation, `.expect("<invariant>")`
documents the invariant; if it can legitimately fail, bridge to
`FdarError::ComputationFailed { operation, detail }` via `?`/`.ok_or_else()`.
Keep the same successful-path result; add no new failure on currently-valid
inputs (guard with the existing tests).

</decisions>

<code_context>
## Existing Code Insights

- pmat entropy flags REPEATED transformation patterns (e.g. the same
  `(0..n).map(|i| ...).collect()` grid-building idiom recurring in a file).
  The crate already has grid helpers (`test_helpers::uniform_grid(n)`,
  `helpers.rs`); prefer reusing/extending an existing helper over inventing a
  new one. Several flagged files are `*/tests.rs` — a shared `#[cfg(test)]`
  helper is the right tool there.
- SATD findings are pure `///` doc-comment text — grep the exact lines.
- `depth/rpd.rs` is random-projection depth; the `.unwrap()` is likely on an
  SVD/Option/partial-cmp result. Follow the crate's `Result<T, FdarError>` +
  `.ok_or_else()` bridge norm.
- Exact entropy pattern locations + the shared-helper design to be pinned
  during plan-phase research (`pmat analyze entropy` is authoritative).

</code_context>

<specifics>
## Specific Ideas

No specific requirements beyond the pmat entropy/satd/defect findings. After
this phase, `pmat analyze entropy`, `pmat analyze satd`, and the critical
`.unwrap()` defect must all read 0 — leaving only the complexity tail for
Phase 109 to close the full `pmat quality-gate`.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>
