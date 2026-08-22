---
quick_id: 260822-pvk
title: "Update README: link Python package pyfda, note R package outdated"
created: 2026-08-22
area: docs
files:
  - README.md
---

# Quick Plan: Link pyfda in README, mark R package outdated

## Description

Add the Python package [`pyfda`](https://github.com/sipemu/pyfda) to the top-level
`README.md`, and reflect that the R package (`fdars` on CRAN / `sipemu/fdars-r`) is
currently outdated / lagging the Rust core. Documentation-only change — no source,
no build, no tests.

## Task 1: Surface pyfda and de-emphasize the outdated R package in README.md

**Files:** `README.md`

**Actions:**
1. Tagline (top): mention Python bindings alongside R.
2. **Packages** table: add a `pyfda` (Python) row linking to
   `https://github.com/sipemu/pyfda`; append an "⚠️ Outdated" marker to the R row's
   Status cell.
3. **Installation**: add a Python (`pyfda`) subsection pointing at the GitHub repo;
   add an inline note on the R subsection that it currently lags the Rust core.
4. **Documentation** section: add a Python package link.

**Verify:** `grep -c "pyfda" README.md` ≥ 3; R rows/sections carry an outdated note;
Markdown tables remain well-formed (column counts consistent).

**Done:** README links to pyfda in the Packages table, Installation, and
Documentation sections, and the R package is clearly flagged as outdated.

## Decision notes

- Non-destructive: R references are kept but flagged outdated (reversible), rather
  than removed — the user said "outdated at the moment", implying temporary.
- `pyfda` has no confirmed PyPI release, so Registry is shown as GitHub, and the
  install instructions use the GitHub source.
