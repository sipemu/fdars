# Phase 93: Release Preparation & Readiness Verification - Context

**Gathered:** 2026-09-09
**Status:** Ready for planning

<domain>
## Phase Boundary

Make fdars-core release-ready at **0.43.0**: bump the version, write CHANGELOG `[0.43.0]` entries (BOTH the root `CHANGELOG.md` and the crate-shipped `fdars-core/CHANGELOG.md`), refresh docs minimally, check off the **Quality** (golden-flake) item on the 1.0 checklist, and run every whole-crate release gate green — with the full-suite `cargo test` as the proof the flake fix holds end-to-end. Requirements REL-01, REL-02. Depends on Phases 90, 91, 92. MUST land last.

**The `git tag v0.43.0` push → crates.io publish is the DEFERRED OPERATOR step** — NOT performed in this (or any) phase. Per project convention `release.yml` couples a `v*` tag-push to `cargo publish`, so GSD `git.create_tag` is OFF and tagging/publishing is done by the operator on a disk-healthy machine after a clean full run.

Out of scope: the actual tag/publish; any code/algorithm change (this is release prep only).
</domain>

<decisions>
## Implementation Decisions

### Docs-refresh scope — minimal, targeted
- Bump `fdars-core/Cargo.toml` version `0.42.0` → `0.43.0`.
- Add a `## [0.43.0]` entry to BOTH changelogs (they DIFFER — root `CHANGELOG.md` and `fdars-core/CHANGELOG.md` are not identical; update each in its own style).
- Mark the golden-flake **Quality** item `[x]` in `documentation/ROADMAP-TO-1.0.md` (lines ~49–54) AND correct its description: it currently says "pass per-binary but flake under full parallel `cargo test` (env/BLAS/disk-pressure dependent)" — Phase 90 **overturned** that hypothesis. The true cause is a deterministic faer-vs-nalgebra SVD backend divergence when compiled without `linalg`; the fix was a test-side cfg-guard (not tolerance, not serialization). Update the item text to reflect the real diagnosis + fix and reference Phases 90–92.
- Touch README / guides ONLY where they state a version or the determinism/flake status. Concretely: README line ~84 `fdars-core = { version = "0.41", … }` is stale → update to `"0.43"`. No broad guide rewrite (nothing else changed this milestone).

### CHANGELOG content (both files)
- `[0.43.0]` frames the milestone: **Test Determinism & Release Hardening** — the long-standing co_cluster/svd_sign golden flake is root-caused (faer/nalgebra SVD backend divergence, NOT an environmental flake) and fixed with a least-invasive test-side `#[cfg_attr(not(feature = "linalg"), ignore)]` guard (no new dependency, no tolerance relaxation, no src/ change); a suite-wide robustness audit found zero further fragile tests; a new `determinism-guardrail` CI job (repeat + single-threaded + anti-silent-skip) gates the suite and the crates.io publish.
- **Supersession note:** 0.43.0 supersedes the unpublished 0.41.0 and 0.42.0 (registry is still at 0.40.0 — one publish catches all three up). State this explicitly, and document that the operator-driven `git tag v0.43.0` → crates.io publish is the remaining step (not performed in any phase).

### Release gates — all 6 must pass green (REL-02)
1. `cargo fmt --check`
2. `cargo clippy --all-targets --features linalg,parallel -- -D warnings` (CI also adds `serde` + explicit `-A` allows; the plan may mirror CI's exact clippy invocation)
3. Full `cargo test` (the determinism proof — run with `--features linalg,parallel,serde`, i.e. the CI config)
4. `--features serde` build guard (must stay green — repaired in v0.40.0, do not regress)
5. All 28 examples + doctests
6. `cargo package`

### Claude's Discretion
- Exact CHANGELOG prose wording (following each file's existing style), and whether the full-test gate is the `linalg,parallel,serde` config or also the `--no-default-features --features linalg` config (both are green per Phases 90–92; running the primary serde config satisfies the determinism proof, the sequential one is a bonus).
- Whether `cargo package` runs with `--no-verify` or full (default) — pick what completes on this machine given disk/tmp hazards; document the choice.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets / Patterns
- **Prior release phases** (v0.41.0 Phase 85, v0.42.0 Phase 89) are the template: version bump → CHANGELOG entries → docs tick → all-gates-green verification → operator tag/publish deferred. Follow their structure.
- `fdars-core/Cargo.toml` version line: `version = "0.42.0"` → `0.43.0`.
- Two CHANGELOGs: root `CHANGELOG.md` (top entry `## [0.42.0] - 2026-09-08`) and `fdars-core/CHANGELOG.md` — they DIFFER; update both.
- `documentation/ROADMAP-TO-1.0.md`: API section already all `[x]` (v0.42.0). The golden-flake item under `## Test & quality debt` (lines ~49–54) is the Quality item to clear. There is a `**Update (v0.42.0):**` note pattern at line ~17 — add an analogous `**Update (v0.43.0):**` note for the Quality clearance.
- Phase artifacts to cite in the CHANGELOG: `90-DIAGNOSIS.md`, `91-AUDIT.md`, the `determinism-guardrail` CI job.

### Integration Points
- The full-suite gate is REL-02's determinism proof AND what the new Phase-92 CI guardrail enforces going forward.
- README `fdars-core = { version = "0.41" }` (line ~84) is the one stale version reference to bump.

### Build/Disk Hazards (MEMORY.md — critical for the gate run)
- **Run `cargo fmt` per commit** — `--no-verify` commits otherwise leave fmt drift → CI fmt-check fails.
- Pre-commit hook runs the full cargo gate and TIMES OUT → run gates FOREGROUND, per-gate, with explicit 600s tool timeout, then `git commit --no-verify`. Do NOT run the combined gate as one background job (gets killed mid-run).
- Free `target/debug/{incremental,examples}` before the long gate run (`target/` fills `/home`); doctests + `cargo package` link in a small `/tmp` tmpfs — watch space.
- Keep the `--features serde` build green (repaired v0.40.0).
- Execute inline — fdars executor subagents stall on long cargo.
- Local `main` is ~24 commits ahead of `origin/main` (unpushed) — expected; do NOT push or tag (operator step).

</code_context>

<specifics>
## Specific Ideas

- When checking off the Quality item, do NOT just flip `[ ]`→`[x]`: also rewrite the stale "env/BLAS/disk-pressure dependent" clause to the correct diagnosis (feature-config / faer-vs-nalgebra backend divergence) and the actual fix (cfg-guard), citing Phases 90–92. Leaving the wrong root-cause text checked off would be misleading.
- The CHANGELOG must explicitly say 0.43.0 supersedes unpublished 0.41.0/0.42.0 and that tag/publish is the deferred operator step — this is Success Criterion 4.
- The `cargo package` gate can be sensitive to uncommitted files / VCS state; run it after commits are in place, and be ready for the `/tmp` link hazard.

</specifics>

<deferred>
## Deferred Ideas

- The operator `git tag v0.43.0` → push → crates.io publish (auto via release.yml). Explicitly NOT done in this phase.
- 1.0-CUT (bump to 1.0.0) — remains deferred until every ROADMAP-TO-1.0.md item clears; after this milestone the Quality item clears, but algorithm/differentiable-core/fdars-r items remain.

</deferred>
