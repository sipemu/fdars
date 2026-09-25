# Phase 117: Release Preparation & Verification - Context

**Gathered:** 2026-09-25
**Status:** Ready for planning

<domain>
## Phase Boundary

Make `fdars-core` release-ready at 0.47.0 (REL-01): bump 0.46.0 → 0.47.0, write the `[0.47.0]` CHANGELOG entry at root and crate level, and land every whole-crate gate green (plus the pmat quality-gate and the benchmark-harness checks). No `git tag` and no crates.io publish inside the phase — `git tag v0.47.0` → crates.io publish is documented as the deferred operator step.

</domain>

<decisions>
## Implementation Decisions

### CHANGELOG content & framing
- `[0.47.0]` entry in Keep-a-Changelog style matching `[0.46.0]`: `### Changed (internal only)` — the `face_covariance` `sym_cov_irreg` upper-triangle optimization (numeric-identical, non-breaking, no new dep); `### Added (tooling & docs)` — the new `crosslang_grid` criterion bench, `documentation/PERFORMANCE.md` + README speed story, and the external `benchmarks/` cross-language harness (explicitly NOT part of the published crate).
- Perf numbers: only the honest, pinned, noise-clean figure — face_covariance **1.19×** single-threaded; elastic alignment and SVD documented as already near-optimal (no change). Link `documentation/PERFORMANCE.md`. NEVER cite Phase 115's unpinned 3.45×.
- Root and crate CHANGELOG share the headline + Changed section; the crate-level entry reduces the external harness to one line.
- One head-to-head sentence: single-threaded, fdars is faster than scikit-fda 0.10.1 and R on all six overlapping ops at N=500, M=200 (link the report).

### Release gates & packaging
- Gate battery (all green): `cargo fmt --check`; `cargo clippy --all-targets --features linalg,parallel -- -D warnings`; full `cargo test`; `--features serde` build; all 28 examples + doctests; `cargo package`; `pmat quality-gate --checks complexity,satd,dead-code,duplicates` (0 violations); PLUS harness checks `benchmarks/.venv/bin/python -m pytest benchmarks/ -q` and `render_report.py --check` / `--check-prose`.
- Packaging: verify `cargo package --list` does not ship `benchmarks/`; verify the new `crosslang_grid` bench compiles from the packaged tarball via its deterministic synthetic-data fallback (it reads `../benchmarks/data` only when present).
- Known golden flake (co_cluster ×2 / svd_sign golden tests fail only under full `cargo test`, pass per-binary — MEMORY): if it appears, re-run the affected binaries in isolation, record the result in the gate log, and do NOT treat it as a regression.
- Gates run INLINE by the orchestrator, one gate per FOREGROUND call with a 600 s timeout; free `target/debug/{incremental,examples}` first; route the serde build / doctests away from the small /tmp tmpfs if needed; commit with `--no-verify` after out-of-band gates (pre-commit hook times out).
- Do NOT `git tag` or publish — document the operator step (`git tag v0.47.0` → push → release.yml publishes).

### Claude's Discretion
- Exact CHANGELOG wording, gate-log format/location, and whether `Cargo.lock` refresh needs `cargo update -p fdars-core` or just a build — within the constraints above.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- v0.46.0 release phase (`.planning/milestones/v0.46.0-phases/111-release-preparation-verification/`) — same gate battery, same no-tag rule; reuse its plan shape.
- `CHANGELOG.md` (root) + `fdars-core/CHANGELOG.md` — `[0.46.0] - 2026-09-14` is the style template.
- `fdars-core/Cargo.toml` `version = "0.46.0"`.

### Established Patterns
- Release = bump `fdars-core/Cargo.toml` → commit → (operator) `git tag vX.Y.Z` → push tag → `release.yml` runs `cargo publish`. `git.create_tag` is OFF.
- crates.io currently has 0.40.0 / 0.42.0 / 0.44.0 (0.45.0/0.46.0 unpublished — 0.47.0 supersedes them, like prior skipped versions).

### Integration Points
- `README.md` may reference the version in install snippets — check and bump if present.
- `documentation/` guides may mention the current version.

</code_context>

<specifics>
## Specific Ideas

- Keep the changelog perf claims consistent with `documentation/PERFORMANCE.md` (same numbers, same honesty caveats).

</specifics>

<deferred>
## Deferred Ideas

- Repairing stale verification fingerprints for phases 112–115 (caused by Phase 116 touching harness files they cover) — handled by the orchestrator before the milestone audit, not a release-phase task.

</deferred>
