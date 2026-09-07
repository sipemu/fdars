---
schema: summary
plan: 81-01
phase: 81
requirements: [AUDIT-01]
status: complete
completed: 2026-09-07
---

# Plan 81-01 Summary — Whole-Surface Audit → Approved Breaking-Change Inventory

## What was built

`81-AUDIT-INVENTORY.md` — a ranked, ID-addressable (`AUD-01`–`AUD-23`) breaking-change inventory across all four scopes (deprecated-form removal, accidental `pub` exposure, `#[non_exhaustive]` gaps, naming), every entry grounded to real `file:line` with proposed change, blast radius (callers/examples/doctests counted via ripgrep), and value/risk rating. Analysis only — `fdars-core/src/` untouched (verified `git diff --quiet`).

## Approval outcome (human-verify checkpoint)

Presented to the user at the Phase 81 checkpoint; approved at **reduced scope** on 2026-09-07.

**APPROVED (feeds Phases 82/83):**
- Scope B (API-02): `AUD-07`, `AUD-08` (pub → pub(crate) seals).
- Scope C (API-03): `AUD-10` (10 enums), `AUD-11` (2 result structs) get `#[non_exhaustive]`.
- Scope D (API-04): `AUD-14`, `AUD-15`, `AUD-16`, `AUD-17`, `AUD-18` (snake_case fix, 2 result-name clashes, deriv/lp Dim collapses).

**DEFERRED to STAB-03 (Phase 84 checklist):** `AUD-09`/`AUD-13` (`wire` kept public), `AUD-12` (config-struct non_exhaustive — needs builder path), `AUD-19`–`AUD-23` (optional naming incl. the large `AUD-22` lone-suffix batch).

## Consumption contract

- **Phase 82** draws its change set from approved `AUD-07`, `AUD-08`, `AUD-10`, `AUD-11`.
- **Phase 83** draws its rename/dispatcher set from approved `AUD-14`–`AUD-18` only.
- **Phase 84 (STAB-03)** absorbs all DEFERRED entries into the 1.0 gap checklist.
- IDs are stable and never renumbered.

## Artifacts produced
- `.planning/phases/81-api-audit-deprecated-form-removal/81-AUDIT-INVENTORY.md` (inventory + `## Approval` section filled)

## Verification
- All four success-criteria scopes covered in the inventory; each entry carries location + proposed change + blast radius + value/risk. ✓
- Inventory presented to and approved by the user (reduced scope on naming, as allowed). ✓
- No source modified by this plan (`git diff --quiet -- fdars-core/src/`). ✓

## Commits
- `36403d5f` — inventory (Tasks 1–2)
- (this commit) — filled `## Approval` section + SUMMARY
