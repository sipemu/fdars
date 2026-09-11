---
wave: 1
depends_on: []
autonomous: true
requirements: [REL-01]
files_modified:
  - fdars-core/Cargo.toml
  - CHANGELOG.md
  - fdars-core/CHANGELOG.md
  - documentation/ROADMAP-TO-1.0.md
---

# Phase 100 Plan 01 — Release Preparation & Verification (REL-01)

<objective>
Make fdars-core release-ready at 0.44.0: version bump, [0.44.0] CHANGELOG (root + crate), DIF-F1/F2/F3 checked off on ROADMAP-TO-1.0.md, all whole-crate gates green. The git tag → crates.io publish is the DEFERRED operator step. Executed inline.
</objective>

<tasks>
<task><name>Task 1: version + CHANGELOG + 1.0-checklist</name>
<action>Bump fdars-core/Cargo.toml 0.43.0→0.44.0; add [0.44.0] entry to root + crate CHANGELOG; check off DIF-F1/F2/F3 on documentation/ROADMAP-TO-1.0.md.</action>
<verify><automated>cd /home/simonm/projects/rust/fdars && grep -m1 '^version = "0.44.0"' fdars-core/Cargo.toml && grep -c '0.44.0' CHANGELOG.md fdars-core/CHANGELOG.md && grep -c '\[x\] \*\*`DIF-F' documentation/ROADMAP-TO-1.0.md</automated>
<fails_when>version not 0.44.0, or [0.44.0] missing from either CHANGELOG, or fewer than 3 DIF items checked</fails_when></verify>
<acceptance_criteria>version 0.44.0; [0.44.0] in both CHANGELOGs; DIF-F1/F2/F3 all `[x]`.</acceptance_criteria>
</task>

<task><name>Task 2: full-suite release gate</name>
<action>Run all whole-crate gates: cargo fmt --check, clippy --all-targets --features linalg,parallel -D warnings, full cargo test, --features serde build, 28 examples + doctests, cargo package. Do NOT git tag.</action>
<verify><automated>cd /home/simonm/projects/rust/fdars && cargo fmt -p fdars-core --check && echo GATE_FMT_OK</automated>
<fails_when>any gate fails</fails_when></verify>
<acceptance_criteria>All gates green; no git tag/publish performed (deferred operator step).</acceptance_criteria>
</task>
</tasks>

<must_haves>
truths:
  - "fdars-core bumped 0.43.0 → 0.44.0 with a [0.44.0] entry in both CHANGELOGs (root + crate)."
  - "DIF-F1 / DIF-F2 / DIF-F3 are checked off on documentation/ROADMAP-TO-1.0.md."
  - "All whole-crate gates pass: fmt, clippy --all-targets, full test, --features serde build, 28 examples + doctests, cargo package."
  - "git tag v0.44.0 → crates.io publish left as the deferred operator step (not performed in-phase)."
prohibitions:
  - statement: "No git tag created and no crates.io publish performed in-phase"
    status: enforced
  - statement: "No new crate dependency added to Cargo.toml"
    status: enforced

# Artifacts this phase produces
# - fdars-core/Cargo.toml version 0.44.0; [0.44.0] CHANGELOG entries; DIF-F1/F2/F3 checkoffs
</must_haves>

<threat_model>
No attack surface — version/docs edits + gate runs; no code behavior change, no I/O/network/privilege boundary.
</threat_model>
