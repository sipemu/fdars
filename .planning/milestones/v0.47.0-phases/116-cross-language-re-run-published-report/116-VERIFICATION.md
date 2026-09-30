---
phase: 116-cross-language-re-run-published-report
verified: 2026-09-25T12:00:00Z
status: passed
score: 8/8 must-haves verified
covered_files:
  - ".planning/REQUIREMENTS.md"
  - ".planning/phases/116-cross-language-re-run-published-report/116-01-PLAN.md"
  - ".planning/phases/116-cross-language-re-run-published-report/116-01-SUMMARY.md"
  - ".planning/phases/116-cross-language-re-run-published-report/116-02-PLAN.md"
  - ".planning/phases/116-cross-language-re-run-published-report/116-02-SUMMARY.md"
  - ".planning/phases/116-cross-language-re-run-published-report/116-03-PLAN.md"
  - ".planning/phases/116-cross-language-re-run-published-report/116-03-SUMMARY.md"
  - ".planning/phases/116-cross-language-re-run-published-report/116-04-PLAN.md"
  - ".planning/phases/116-cross-language-re-run-published-report/116-04-SUMMARY.md"
  - ".planning/phases/116-cross-language-re-run-published-report/116-05-PLAN.md"
  - ".planning/phases/116-cross-language-re-run-published-report/116-05-SUMMARY.md"
  - ".planning/phases/116-cross-language-re-run-published-report/116-06-PLAN.md"
  - ".planning/phases/116-cross-language-re-run-published-report/116-06-SUMMARY.md"
  - ".planning/phases/116-cross-language-re-run-published-report/116-07-PLAN.md"
  - ".planning/phases/116-cross-language-re-run-published-report/116-07-SUMMARY.md"
  - "README.md"
  - "benchmarks/README.md"
  - "benchmarks/collect_rust.py"
  - "benchmarks/render_report.py"
  - "benchmarks/run_all.py"
  - "documentation/PERFORMANCE.md"
  - "fdars-core/Cargo.toml"
  - "fdars-core/benches/crosslang_grid.rs"
  - "fdars-core/benches/depth_benchmarks.rs"
covered_digest: "v1:sha256:057fcba1c7c54f921e153734aca7ec2fc5c7f7ce9dba575754a479c9e7d32152"
behavior_unverified: 0
overrides_applied: 0
re_verification: null
---

# Phase 116: Cross-Language Re-Run & Published Report Verification Report

**Phase Goal:** The optimization gains are shown to be visible across all three ecosystems and turned into the milestone's proof artifact — a published head-to-head perf report a reader can reproduce.
**Verified:** 2026-09-25T12:00:00Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Framing note (honesty clause on the goal wording)

The goal's literal phrase — "gains visible across all three ecosystems" — cannot
be taken literally. Confirmed by inspecting both the optimized code paths and the
harness bench code:

- The Phase 114 `dp_fill_grid` banded-DP optimization is on the *banded* elastic
  arm; the cross-language harness `elastic` op calls the unbanded pairwise
  `elastic_align_pair` (`fdars-core/benches/crosslang_grid.rs`), so it was never
  on the optimized path.
- `face_covariance` (Phase 115's real win) has no scikit-fda/R comparator at all
  — it cannot appear in any head-to-head table by construction.
- SVD/FPCA was found near-optimal in Phase 115 (no change made), so there is no
  "after" delta to show even qualitatively.

Given this, "gains visible across ecosystems" is unachievable as literally
worded, and the report says so explicitly rather than fabricating cross-language
movement (documentation/PERFORMANCE.md lines ~72-76: "The Phase 114-115
optimizations are not on any harness op's code path"). The fdars-internal win is
instead proven via a controlled, thread-pinned A/B (`phase116_face_covariance_n200_m30`,
1.19× pinned, noise-gate-passing, CI-disjoint), clearly distinguished from
Phase 115's larger-but-unpinned-and-noisy 3.45× figure. **Judgment: this
satisfies the intent of the goal** — the milestone's actual deliverable (an
honest, reproducible proof artifact showing what changed and what didn't) is
present and the impossibility of the literal wording is disclosed, not
papered over. This is not scored as a gap; see Truth 4 below where it is
verified as a truth in its own right (the report's handling of the
irreconcilable wording, not the literal wording itself).

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Full cross-language harness re-run post-optimization; `comparison.csv` updated with after-optimization numbers (ROADMAP SC1) | VERIFIED | `benchmarks/results/comparison.csv` (36129 bytes, mtime 25 Sep 02:19) contains 36 fresh `crosslang_*` rust rows (6 ops x 6 cells) plus refreshed scikit-fda/R rows for all 6 grid cells (`GRID_CELL_STATUS=ok`, gate passed on every cell per 116-06-SUMMARY). `comparison_baseline.csv` was committed first (frozen Phase 113 snapshot) before the overwrite, confirmed present and distinct in content and mtime (24 Sep 21:21 vs 25 Sep 02:19). |
| 2 | Published perf report exists with head-to-head before/after tables, N×M scaling analysis, full methodology disclosure (ROADMAP SC2) | VERIFIED | `documentation/PERFORMANCE.md` (60496 bytes, 10 sections: Summary, Headline, Head-to-head grid, Optimization A/B, Before/after-vs-baseline x4 subsections, N×M scaling, Methodology, Reproduction, Known limitations). Directly inspected the scaling section: per-op per-library N×M grids with log-log N/M slopes and theoretical-complexity citations to real source functions (e.g. `fdars-core/src/regression.rs::fdata_to_pc`). Methodology section discloses thread pinning, environment (generated), gate tolerances (generated), timing-statistic definitions per ecosystem, noise-gate definition, criterion settings, and the R `pca.fd`/Simpson-SVD footnote. |
| 3 | Reproduction instructions a reader can follow end-to-end to regenerate the numbers from scratch (ROADMAP SC3) | VERIFIED | `## Reproduction` section: 7 numbered steps from clean clone (toolchain install, venv + `requirements.lock`, multimethod patch, `renv::restore()`, single `run_all.py --full` command, `render_report.py` + `--check`/`--check-prose`). Measured runtime (2.93h) and disk footprint (57.83 GiB target/) are generated from `rerun_log.json`, not hand-typed. Independently re-ran `render_report.py --check` and `--check-prose` from the repo root myself: both exit 0 against the current committed tree (see Behavioral Spot-Checks). 116-07-SUMMARY also records a fresh-clone `--check`/`--check-prose` pass, independently corroborated here. |
| 4 | The report handles the "gains visible across ecosystems" impossibility honestly (proves the fdars-internal win via a controlled A/B rather than fabricating cross-language movement) | VERIFIED | Summary section states plainly: "The Phase 114-115 optimizations are not on any harness op's code path" (elastic op untouched, face_covariance has no comparator). The A/B table headlines only `phase116_face_covariance_n200_m30` (1.19x, PASS/PASS noise gate, disjoint CIs) and explicitly shows the disqualified 3.45x Phase 115 figure (FAIL-NOISY before-cell) side by side, labeled "not reproduced under single-threaded, noise-gated conditions and is never claimed as this report's headline." |
| 5 | Background-load disclosure present (measurement conditions were not uniformly quiet) | VERIFIED | `documentation/PERFORMANCE.md` line ~712: "Background load disclosure. A quiet-machine check (load average < 2) was [...] returned partway through the ~2.9-hour run: the 1-minute load average [...]" and repeated in Known Limitations ("1-minute load average 2.8-15.0 at scheduler-call starts"). Matches 116-06-SUMMARY's disclosed deviation verbatim. |
| 6 | "Not run (timeout)" disclosure for the 7 elastic/Karcher chunks that could not complete | VERIFIED | Before/after-vs-baseline "not-rerun" subtable lists all 7 by exact chunk id with explicit reasons (`timeout at 540s (single-threaded)` x4, `inferred-timeout: cheaper sibling ... timed out` x3): `audit_p3_elastic_cross/n100_m200`, `audit_p3_elastic_cross/n500_m50`, `audit_p3_elastic_cross_banded/n100_m200`, `audit_p3_elastic_cross_banded/n500_m50`, `audit_p3_elastic_self/n100_m200`, `audit_p3_elastic_self/n500_m50`, `audit_p3_karcher/n500_m200`. Key-findings summary table also surfaces the aggregate counts (`timeout: 4`, `inferred-timeout: 3`). Matches `rerun_log.json`'s `rust_status` counts (60 ok / 3 skipped-inferred-timeout / 4 timeout) reproduced independently below. |
| 7 | Every generated table byte-for-byte reproducible from committed inputs; no fabricated/hand-edited numbers | VERIFIED | Independently re-ran (not trusting SUMMARY claims): `render_report.py --check` -> "check OK — all blocks up to date" (exit 0); `render_report.py --check-prose` -> "check-prose OK" (exit 0); `render_report.py --noise-report` -> all 6 headline cells PASS; `render_report.py --ab-check` -> "AB-CHECK: PASS (phase116_face_covariance_n200_m30 eligible)", matching the report's claimed 1.19x figure exactly. `pytest benchmarks/ -q` -> 209 passed, 0 failed. |
| 8 | Scope prohibitions honored: no `fdars-core/src` changes, no `gate.py` changes, no committed image files | VERIFIED | `git diff --name-only ec57fa01 HEAD -- fdars-core/src benchmarks/gate.py` is empty (independently re-run). `git diff --name-only ec57fa01 HEAD \| grep -E '\.png\|\.svg\|\.jpg\|\.gif'` returns nothing. |

**Score:** 8/8 truths verified (0 present-but-behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `documentation/PERFORMANCE.md` | Full published performance report | VERIFIED | 60496 bytes, 10 sections, 13 `BEGIN/END GENERATED` blocks all populated with real data (spot-checked headline, ab-evidence, scaling, runtime blocks) |
| `README.md` | Short honest speed story + Guides link | VERIFIED | `## Performance` section (line 153) has the generated headline table, 2 takeaway sentences, link to `documentation/PERFORMANCE.md`, qualitative rayon/WASM sentence retained; Guides list has a `Performance` bullet |
| `benchmarks/render_report.py` | Report generator (13 emitters + CLI) | VERIFIED | 74581 bytes; `--check`/`--check-prose`/`--noise-report`/`--ab-check` all independently re-run and pass/report correctly |
| `benchmarks/results/comparison.csv` | After-optimization comparison data | VERIFIED | Overwritten post-baseline-freeze; 36 fresh crosslang rows confirmed |
| `benchmarks/results/comparison_baseline.csv` | Frozen Phase 113 before-baseline | VERIFIED | Present, distinct mtime/content from comparison.csv |
| `benchmarks/results/rerun_log.json` | Run log with per-chunk/per-cell status | VERIFIED | 31670 bytes; `rust_chunk` status counts (60 ok/3 skipped-inferred-timeout/4 timeout) match report's runtime table exactly |
| `benchmarks/results/ab/manifest.json` | A/B evidence manifest incl. phase116 entry | VERIFIED | `phase116_face_covariance_n200_m30` entry present, `--ab-check` confirms `headline_eligible=True` |
| `benchmarks/patches/multimethod-2.1-py314.patch` | Reproduction patch artifact | VERIFIED | Present on disk; provenance verification claimed via RECORD-hash match (not independently re-derived here, low risk — mechanical file-diff artifact, not a load-bearing correctness claim) |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| `benchmarks/run_all.py` | `fdars-core/benches/crosslang_grid.rs` | `run_rust_bench()` invokes `cargo bench --bench crosslang_grid` under pinned thread env | VERIFIED | 36 crosslang rows present in comparison.csv confirms the bench actually ran and was collected |
| `benchmarks/render_report.py` | `benchmarks/rank_hotspots.py` | imports `apply_noise_gate` and its thresholds (never re-declared) | VERIFIED | `--noise-report` output uses identical RSD/CI-width thresholds; grep confirms `from rank_hotspots import` in render_report.py |
| `benchmarks/render_report.py` | `benchmarks/gate.py` | imports gate tolerance constants for the methodology block | VERIFIED | Methodology "agreement gate" block cites gate.py thresholds; `gate.py` itself untouched (diff empty) |
| `README.md` | `documentation/PERFORMANCE.md` | markdown link in Performance section + Guides list | VERIFIED | Both links present and correctly pointed |
| `documentation/PERFORMANCE.md` | `benchmarks/render_report.py` | 13 generated blocks regenerate via `--check` | VERIFIED | `--check` exit 0 (independently re-run from repo root) |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Report tables reproduce from committed data | `benchmarks/.venv/bin/python benchmarks/render_report.py --check` (from repo root) | `check OK — all blocks up to date`, exit 0 | PASS |
| Prose only quotes generated numbers | `benchmarks/.venv/bin/python benchmarks/render_report.py --check-prose` (from repo root) | `check-prose OK`, exit 0 | PASS |
| Headline cells pass the noise gate | `benchmarks/.venv/bin/python benchmarks/render_report.py --noise-report` | all 6 `crosslang_*/n500_m200` cells PASS (RSD 0.77-4.72%, CIW 0.54-3.30%) | PASS |
| A/B headline eligibility is code-derived, not hand-picked | `benchmarks/.venv/bin/python benchmarks/render_report.py --ab-check` | `phase116_face_covariance_n200_m30 speedup=1.192 ... headline_eligible=True`; `phase115_face_covariance` (3.45x) correctly `headline_eligible=False` (before-cell FAIL-NOISY) | PASS |
| Test suite green | `benchmarks/.venv/bin/python -m pytest benchmarks/ -q` | 209 passed, 0 failed, 2 pre-existing unrelated warnings (constant-input pearsonr) | PASS |
| No source/image scope creep | `git diff --name-only ec57fa01 HEAD -- fdars-core/src benchmarks/gate.py` / image-extension grep | both empty | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| RPT-01 | 116-01..07 | Re-run the full harness post-optimization and publish a perf report with head-to-head before/after tables, N×M scaling analysis, methodology disclosure, and reproduction instructions | SATISFIED | All 7 plans' `requirements: [RPT-01]` map to the artifacts/truths verified above; REQUIREMENTS.md already marks RPT-01 `[x]` / `Complete`, corroborated independently (not just trusted) via the direct codebase checks in this report |

No orphaned requirements: REQUIREMENTS.md maps only RPT-01 to Phase 116, and it is claimed by all 7 plans.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| — | — | No `TBD`/`FIXME`/`XXX`/`TODO`/`HACK`/`PLACEHOLDER` markers found in any phase-modified file (`benchmarks/run_all.py`, `render_report.py`, `collect_rust.py`, `crosslang_grid.rs`, `depth_benchmarks.rs`, `PERFORMANCE.md`, `README.md`, `benchmarks/README.md`) | — | None |

Code review (`116-REVIEW.md`, re-review after `116-REVIEW-FIX.md`) independently found 0 critical / 0 warning / 6 info issues, all info-level and non-blocking (dead `smoke` param, a falsy-vs-`is not None` cosmetic check, a group-name-collision edge case, a diagnostics-completeness short-circuit). None of these affect the phase goal or the published report's correctness. Verified the review's "clean" verdict independently: re-ran `pytest benchmarks/ -q` (209/209) and `--check`/`--check-prose` myself rather than trusting the review's self-report.

### Human Verification Required

None. All must-haves resolve to file-system/command-level evidence; no visual, real-time, or external-service behavior is in scope for a documentation/benchmark-report phase.

### Gaps Summary

No gaps. All ROADMAP success criteria are met with direct evidence (not SUMMARY claims): the harness was re-run and `comparison.csv` updated, the published report exists with all required sections and honest disclosures (background load, timeouts, headline-eligibility gating, non-wins stated plainly), and reproduction instructions were independently exercised (the `--check`/`--check-prose`/`--noise-report`/`--ab-check`/pytest commands the report specifies were re-run standalone by this verifier and passed). The one interpretive risk — the goal's literal "gains visible across all three ecosystems" wording being structurally impossible given what Phase 114/115 actually touched — is judged satisfied in spirit: the report discloses this plainly rather than fabricating a cross-language effect, and substitutes the honest, code-gated internal A/B (1.19×) as the milestone's proof point, exactly as the phase's CONTEXT.md decisions anticipated.

---

_Verified: 2026-09-25T12:00:00Z_
_Verifier: Claude (gsd-verifier)_

## Re-fingerprint 2026-09-25 (post-Phase-117)

Covered-file digest recomputed before the milestone audit. Changed covered files: REQUIREMENTS.md (milestone completion), README.md (Phase 117-01 install-snippet bump to 0.47; README Performance section unchanged, `render_report.py --check` still exit 0 per 117 gate 14) and fdars-core/Cargo.toml (Phase 117 version bump 0.46.0 -> 0.47.0). No verified truth was invalidated.
