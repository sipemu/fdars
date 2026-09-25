---
phase: "117"
slug: "release-preparation-verification"
status: verified
threats_open: 0
asvs_level: 1
created: "2026-09-25"
---

# Phase 117 — Security

> Threat register, accepted risks, and audit trail.

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| repo → crates.io (deferred) | tag push triggers release.yml publish | the packaged crate (irreversible) |
| working tree → packaged tarball | cargo package selects files | source files; must exclude caches/planning/harness |
| orchestrator → fdars-core/src | release-prep phase touching library code | alignment/mod.rs (user-approved refactor) |

---

## Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation | Status |
|-----------|----------|-----------|----------|-------------|------------|--------|
| T-117-01 | Tampering | `fdars-core/Cargo.toml` version / deps | low | mitigate | Diff gate asserts exactly one +/- line pair (version only) vs `0c7dc99a`; `cargo package --no-verify` re-validates the manifest and emits `fdars-core-0.47.0.crate`. | closed |
| T-117-02 | Information Disclosure | crate tarball contents (`fdars-core/.pmat*` caches) | medium | mitigate | `.gitignore` entries for `fdars-core/.pmat/` + `fdars-core/.pmat-metrics/`; `cargo package --list` gate asserts zero `.pmat*`/`benchmarks/`/`.planning/`/`target/`/`.claude/` entries. | closed |
| T-117-03 | Elevation of Privilege | release.yml tag -> publish trigger | high | mitigate | Plan never runs `git tag` or a publish command; SCOPE_OK gate asserts `git tag -l 'v0.47*'` is empty. Publish stays the deferred operator step (D-09). | closed |
| T-117-04 | Repudiation | CHANGELOG perf claims | medium | mitigate | Region-scoped grep asserts the [0.47.0] sections cite the pinned 1.19x (present in PERFORMANCE.md's generated key-findings block) and never the unpinned Phase 115 figure (D-02). | closed |
| T-117-SC | Tampering | npm/pip/cargo installs | low | accept | No package installs and no new crate dependency; `cargo tree --offline` only re-resolves the existing lock (no `cargo update`/`generate-lockfile`). | closed |
| T-117-05 | Elevation of Privilege | fdars-core/src edit without consent | high | mitigate | Blocking `checkpoint:decision` (Task 1) before any edit; decision persisted as `pmat_decision` in the SUMMARY; every Task 2/3 verify reads it and asserts an untouched src tree under `defer`. | closed |
| T-117-06 | Tampering | dp_fill_grid numeric output | medium | mitigate | Characterization goldens run BEFORE and after the edit (equivalence_phase114 banded + unbanded at 1e-10, equivalence_phase47), release-profile golden run, alignment lib tests; diff gate forbids any `pub` change and any other src file. | closed |
| T-117-07 | Repudiation | pmat quality gate integrity | medium | mitigate | Gate run exactly as CI runs it; verify asserts no `.github/workflows` diff and no lint-allow attribute added; action forbids threshold/config/suppression/exclusion changes. | closed |
| T-117-SC (117-02) | Tampering | npm/pip/cargo installs | low | accept | No installs; pmat 3.18.0 already installed locally (dev tool, never a crate dependency). | closed |
| T-117-08 | Repudiation | gate verdicts (false green) | high | mitigate | Exit codes captured before any `tail`/`grep` (no pipe masks the status); full logs kept; lib/doc results must match `test result: ok. N passed; 0 failed`; integration uses `--no-fail-fast` so no failing binary goes unreported; flake acceptance limited by code to equivalence_phase48/49 with a mandatory isolated green re-run. | closed |
| T-117-09 | Denial of Service | disk exhaustion during builds/doctests | low | mitigate | D-08 pre-flight frees target/debug/{incremental,examples} and records df; serde build + doctests use TMPDIR=/home/simonm/.cache/fdars-bench-tmp. | closed |
| T-117-03 (117-03) | Elevation of Privilege | release.yml tag -> publish trigger | high | mitigate | No tag/publish command anywhere in this plan; final verify asserts `git tag -l 'v0.47*'` is empty. | closed |
| T-117-SC (117-03) | Tampering | npm/pip/cargo installs | low | accept | No installs; builds use the existing lockfile. | closed |
| T-117-03 (117-04) | Elevation of Privilege | release.yml tag -> publish trigger | high | mitigate | Operator step is written as documentation only, explicitly "DO NOT RUN in-phase"; verify blocks assert `git tag -l 'v0.47*'` is empty in two places; no push/publish command appears in any executed step. | closed |
| T-117-02 (117-04) | Information Disclosure | crate tarball contents | medium | mitigate | `cargo package` run without --allow-dirty (VCS info must not be dirty) + list gate rejecting benchmarks/, .pmat*, .planning/, target/, .claude/. | closed |
| T-117-10 | Repudiation | gate log verdict | medium | mitigate | Every row requires PASS except pmat, which may only be a user-approved DEVIATION tied to `pmat_decision: defer`; verdict is `PARTIAL` in that case, never `SATISFIED`; gated-sha equality check proves the cargo gates cover the packaged source. | closed |
| T-117-11 | Tampering | packaged bench data path | low | mitigate | Tarball extracted outside the repo with the two env overrides unset and `../benchmarks` asserted absent, so the run genuinely exercises the synthetic fallback. | closed |
| T-117-SC (117-04) | Tampering | npm/pip/cargo installs | low | accept | No installs; the tarball build uses `--offline --locked` against the packaged lockfile; the Python venv is the existing pinned harness venv. | closed |

---

## Accepted Risks Log

| Risk ID | Threat Ref | Rationale | Accepted By | Date |
|---------|------------|-----------|-------------|------|
| AR-117-SC | T-117-SC | No package installs and no new crate dependency; `cargo tree --offline` only re-resolves the existing lock (no `cargo update`/`generate-lockfile`). | plan-time disposition (117 planner) | 2026-09-25 |
| AR-117-SC | T-117-SC (117-02) | No installs; pmat 3.18.0 already installed locally (dev tool, never a crate dependency). | plan-time disposition (117 planner) | 2026-09-25 |
| AR-117-SC | T-117-SC (117-03) | No installs; builds use the existing lockfile. | plan-time disposition (117 planner) | 2026-09-25 |
| AR-117-SC | T-117-SC (117-04) | No installs; the tarball build uses `--offline --locked` against the packaged lockfile; the Python venv is the existing pinned harness venv. | plan-time disposition (117 planner) | 2026-09-25 |

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-09-25 | 18 | 18 | 0 | orchestrator (L1; auditor skipped per short-circuit: threats_open 0, plan-time register, ASVS 1) |

Evidence (L1, from 117-GATE-LOG.md rows 1-16 + re-checks): `git tag -l 'v0.47*'` empty; no cargo publish run; `cargo package` without --allow-dirty, vcs sha1 not dirty, 0 entries under benchmarks/.pmat*/.planning/target/.claude; pmat 0 violations with no .github/workflows diff and no lint-allow added; equivalence_phase114/47 green before + after the refactor (debug + release); pmat_decision recorded before any src edit; gate exit codes captured before tail/grep.

---

## Sign-Off

- [x] All threats have a disposition
- [x] Accepted risks documented
- [x] `threats_open: 0` confirmed

**Approval:** verified 2026-09-25
