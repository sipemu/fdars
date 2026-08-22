---
task: remove-beads-issue-tracker
date: 2026-08-23
status: complete
commit: 198b5566
---

# Summary: Removed the `.beads` issue tracker

## What was done
- Removed the git-tracked `.beads/` beads (`bd`) issue-tracker directory (6 files:
  `issues.jsonl`, `interactions.jsonl`, `config.yaml`, `metadata.json`, `README.md`,
  `.gitignore`).
- Removed `AGENTS.md` — it was entirely beads-workflow instructions (`bd onboard`,
  `bd sync`, "Landing the Plane"), stale once beads is gone.

## Conservation
- **Open work preserved:** the only OPEN issue, `fdars-j75` (migrate `fdars-r` R
  wrapper to the `FdMatrix` API), moved to STATE.md → Pending Todos.
- **14 closed issues** were R-vs-fdars validation tasks already reflected in the test
  suite — historical only.
- **Full record recoverable:** `.beads/issues.jsonl` remains in git history
  (`git show <pre-removal-commit>:.beads/issues.jsonl`).

## Not in scope
- `.claude/worktrees/*/.beads` copies live inside leftover agent worktrees (transient,
  under `.claude/`), untouched.

## Verification
- `.beads/` and `AGENTS.md` absent from the working tree and no longer git-tracked.
