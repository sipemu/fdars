---
task: remove-beads-issue-tracker
date: 2026-08-23
mode: quick
---

# Quick Task: Remove the `.beads` issue tracker (old, unused artifact)

Remove the git-tracked `.beads/` beads (`bd`) issue-tracker directory and the
beads-only `AGENTS.md` instructions file. The project uses GSD (`.planning/`) for
tracking; beads is no longer used.

## Conservation check (done before removal)
- 15 beads issues: 14 closed (R-vs-fdars validation tasks, already reflected in the
  test suite), 1 open (`fdars-j75` — migrate `fdars-r` R wrapper to the `FdMatrix`
  API). The open item is preserved in STATE.md → Pending Todos.
- Full historical `issues.jsonl` remains recoverable from git history after removal.
- `AGENTS.md` is 100% beads-workflow instructions (`bd onboard`/`bd sync`) → stale
  once beads is gone; removed alongside `.beads/`.
