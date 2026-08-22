---
quick_id: 260822-pvk
title: "Update README: link Python package pyfda, note R package outdated"
date: 2026-08-22
status: complete
files:
  - README.md
---

# Summary: Link pyfda in README, mark R package outdated

Documentation-only change to `README.md`. No source, build, or tests affected.

## Changes

1. **Tagline** — now reads "implemented in Rust, with Python and R bindings."
2. **Packages table** — added a `pyfda` (Python → [sipemu/pyfda](https://github.com/sipemu/pyfda))
   row; reordered to Rust → Python → R; appended "⚠️ outdated" to the R row's Status.
3. **Installation** — reordered to Rust → Python → R; added a **Python (pyfda)** subsection
   with a `pip install git+…` instruction; added an outdated-warning callout to the **R (fdars)**
   subsection steering readers to the Rust core or pyfda.
4. **Documentation** — added a **Python Package** link and flagged the R package docs as
   outdated.

## Decisions

- **Non-destructive:** R references kept but flagged outdated (reversible) rather than removed —
  the user framed it as "outdated at the moment" (temporary).
- **Registry for pyfda:** shown as GitHub with a `git+https` pip install, since pyfda has no
  confirmed PyPI release.

## Verification

- `grep -c pyfda README.md` → 6 (≥ 3 target met): Packages table, Installation, Documentation.
- Markdown tables remain well-formed (5 columns throughout the Packages table).
- R package flagged outdated in all three surfaces (table, install, docs).
