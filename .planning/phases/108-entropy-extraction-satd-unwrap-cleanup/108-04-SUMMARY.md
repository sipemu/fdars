---
plan: "108-04"
phase: "108"
title: "Doc-comment entropy simplification & stable-gate — DESCOPED"
status: complete
completed: "2026-09-14"
requirements: [QUAL-04]
key_files:
  created: []
  modified: []
---

# Plan 108-04 Summary — DESCOPED (no code change)

## Outcome

**Descoped per operator decision (2026-09-14).** This plan's goal was to
DRY-factor the remaining `///` doc-comment example idioms crate-wide and drive
`pmat analyze entropy` to a STABLE 0 (verified over 3 consecutive runs), with an
escalation path if entropy could not be stabilized.

During execution the escalation condition triggered: `pmat analyze entropy` is
**non-deterministic** — three identical runs on the same tree returned 2, 5, and
3 violations respectively, and pmat's output does not name the offending files.
The residual is widespread doctest `(0..n).map(...)` idioms. A stable 0 is
impractical and would make the Phase-110 CI gate flaky.

Per the recorded operator decision, the milestone's quality-gate is redefined to
**exclude the stochastic entropy analyzer**:

```
pmat quality-gate --checks complexity,satd,dead-code,duplicates --fail-on-violation
```

Entropy is documented as an untracked stochastic metric (same treatment as the
already-deferred PVLINT-01 composite). The substantial entropy reduction already
delivered in plans 108-02 and 108-03 (11–13 → ~3; 71 test call-sites + 5
production files de-duplicated, all numeric-identical, full suite green) is kept
as a genuine quality improvement.

## Why no code change here

The chosen path was "exclude entropy from the gate", NOT "grind entropy toward 0"
— so further DRY-factoring of doc-comment examples was explicitly not wanted
(it would harm doc-example readability/self-containedness and could not
guarantee a stable 0 anyway). No source files were modified by this plan.

## Requirement disposition

- **QUAL-04 (entropy):** SATISFIED-AS-REDEFINED. Entropy materially reduced
  (11–13 → ~3) and removed from the deterministic milestone gate as a stochastic
  metric. Not a stable-0 (impossible given pmat's sampler).

## Verification

- No code change → existing suite unaffected (2942 lib + 215 doctests remained
  green as of plan 108-03).
- Gate redefinition recorded in STATE.md decisions + project memory.
