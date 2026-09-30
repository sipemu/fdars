---
phase: 116-cross-language-re-run-published-report
plan: "07"
subsystem: benchmarks
tags: [performance-report, render-report, readme, reproducibility, honesty-clause, multimethod-patch]

# Dependency graph
requires:
  - phase: 116-cross-language-re-run-published-report
    plan: "01"
    provides: documentation/PERFORMANCE.md skeleton, render_report.py scaffold
  - phase: 116-cross-language-re-run-published-report
    plan: "04"
    provides: render_report.py's full analysis core + all 13 emit_* functions + --check/--check-prose CLI
  - phase: 116-cross-language-re-run-published-report
    plan: "05"
    provides: the pinned, noise-clean phase116_face_covariance_n200_m30 A/B entry (1.19x, headline-eligible)
  - phase: 116-cross-language-re-run-published-report
    plan: "06"
    provides: the after-optimization comparison.csv, rerun_log.json, and refreshed env.json this report renders
provides:
  - documentation/PERFORMANCE.md — the published performance report (ROADMAP SC2/SC3): 10 sections, all 13 generated blocks, full methodology and reproduction instructions
  - README.md "## Performance" speed story + Guides link to the full report
  - benchmarks/README.md full-re-run documentation, extended Result Schema, corrected R FPCA note, multimethod patch subsection
  - benchmarks/patches/multimethod-2.1-py314.patch — RECORD-hash-verified reproduction artifact for the Python 3.14 scikit-fda import fix
affects: []

# Actuals (#2632)
actuals:
  tokens: 18664
  tasks: 2
  commits: 2
  plan_head_before: b660693341d6718ee83e758a9dcaa6e55c5ebf0c

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Generated-block-then-prose authoring: write the full doc skeleton (headings + empty BEGIN/END GENERATED marker pairs + hand-written prose), then run render_report.py once to fill every block from committed CSVs — prose is written to reference numbers already known to exist in a generated block, never transcribed by hand"
    - "check-prose honesty gate: any <num>x/<num>ms in hand-written prose must appear verbatim inside a generated block; percentages and unitless numbers are unconstrained, which is what allows honest narrative framing (e.g. '-2.0% (not significant)') without needing every number pre-rendered"
    - "Supply-chain patch provenance: verify a downloaded pristine package's file hash against the installed venv's own dist-info/RECORD entry before diffing, then prove the patch equals the venv's modification via a reverse dry-run apply (patch --dry-run -R)"

key-files:
  created:
    - benchmarks/patches/multimethod-2.1-py314.patch
  modified:
    - documentation/PERFORMANCE.md
    - README.md
    - benchmarks/README.md

key-decisions:
  - "Headlined face_covariance at the pinned, noise-clean 1.19x (phase116_face_covariance_n200_m30), not the larger but unpinned/noisy 3.45x Phase 115 figure — both are shown in the ab-evidence table, but only the eligible one is claimed as a speedup, per the milestone's honesty clause."
  - "The before/after-vs-Phase-113-baseline section frames every table as a traceability record, not an optimization measurement — the Phase 113 baseline mixed live re-runs with mixed-vintage, non-thread-pinned on-disk criterion output, so massive deltas on elastic/karcher rows (some +800%) reflect losing rayon parallelism between an unpinned baseline and the Phase 116 single-threaded re-run, not a regression."
  - "Added an explicit background-load disclosure (1-min load average 2.8-15.0 during part of the fdars re-run, despite a passing pre-run quiet check) to the Methodology and Known Limitations sections — not required by any automated check, but required by the plan's must_haves and the milestone's honesty clause; framed correctly as bias-against-fdars, never in its favor."
  - "R renv restore documented but not re-executed in this phase (R packages were already installed on the machine from the original Phase 112 setup) — benchmarks/README.md states this explicitly rather than implying the restore was verified fresh."

requirements-completed: [RPT-01]

coverage:
  - id: D1
    description: "documentation/PERFORMANCE.md: 10-section published report (how-to-read legend, key-findings summary, headline table, head-to-head grid, controlled A/B evidence, before/after-vs-baseline with 4 subsections, N x M scaling, methodology, reproduction, known limitations) with all 13 generated blocks rendered and current"
    requirement: RPT-01
    verification:
      - kind: other
        ref: "benchmarks/.venv/bin/python benchmarks/render_report.py --check"
        status: pass
      - kind: other
        ref: "benchmarks/.venv/bin/python benchmarks/render_report.py --check-prose"
        status: pass
      - kind: other
        ref: "grep -c '!\\[' documentation/PERFORMANCE.md == 0; thread_env rows == 6, all '1 | 1'; 'within noise|not significant' and 'near-optimal' each >=1; 'pca.fd' and 'Simpson' each >=1; 'run_all.py --full'/'requirements.lock'/'renv' each >=1"
        status: pass
    human_judgment: false
  - id: D2
    description: "README.md ## Performance rewritten into a short honest speed story (generated headline table + 2-3 takeaway lines + link to the full report + qualitative parallel/rayon/WASM sentence retained); Guides list gains a Performance bullet"
    requirement: RPT-01
    verification:
      - kind: other
        ref: "awk-scoped README Performance section contains the headline marker, the PERFORMANCE.md link, zero image embeds, and 'rayon'; Guides bullet present"
        status: pass
    human_judgment: false
  - id: D3
    description: "benchmarks/README.md documents the full Phase 116 re-run (--full plus the inline resumable sequence), extends Result Schema with comparison_baseline.csv/rerun_log.json/ab-manifest.json, corrects the outdated primary-R-FPCA-path claim, documents the renv restore invocation and the results/* gitignore carve-out, and adds a multimethod patch subsection"
    requirement: RPT-01
    verification:
      - kind: other
        ref: "grep -c fpca_native_r >=1; grep -c 'primary R FPCA path uses' == 0; grep -c 'run_all.py --full' >=1; grep -c rerun_log.json >=1"
        status: pass
    human_judgment: false
  - id: D4
    description: "benchmarks/patches/multimethod-2.1-py314.patch: a verified, reproducible patch derived from a downloaded pristine wheel whose file hash matches the venv's own dist-info/RECORD entry"
    requirement: RPT-01
    verification:
      - kind: other
        ref: "sha256/base64-urlsafe hash of the downloaded pristine multimethod/__init__.py equals the RECORD entry; patch --dry-run -R -p1 -d benchmarks/.venv/lib/python3.14/site-packages < the patch exits 0"
        status: pass
    human_judgment: false
  - id: D5
    description: "Every published table regenerates byte-for-byte in a fresh clone (reproducibility proof), and the phase touched only measurement/docs (no fdars-core/src or gate.py changes, no image files)"
    requirement: RPT-01
    verification:
      - kind: other
        ref: "git clone -q . <tmp>; cd <tmp> && render_report.py --check && --check-prose both exit 0"
        status: pass
      - kind: other
        ref: "git diff --name-only ec57fa01 HEAD -- fdars-core/src benchmarks/gate.py is empty; no .png/.svg/.jpg/.gif/.pdf in the diff"
        status: pass
      - kind: unit
        ref: "benchmarks/.venv/bin/python -m pytest benchmarks/ -q"
        status: pass
  - id: D6
    description: "Honesty framing (face_covariance headlined only if eligible, elastic/SVD stated plainly as non-wins, baseline caveats visible, no implied speedup the tables don't show)"
    requirement: RPT-01
    verification: []
    human_judgment: true
    rationale: "Whether the narrative framing reads as honest and non-misleading (vs. technically-true-but-slanted) is a judgment call automated grep checks cannot fully make, even though the specific honesty-clause facts (1.19x vs 3.45x, within-noise, near-optimal) were verified present and correctly attributed by grep."

duration: ~55min
completed: 2026-09-25
status: complete
---

# Phase 116 Plan 07: Published Performance Report Summary

Authored the milestone's proof artifact: `documentation/PERFORMANCE.md` (10 sections, all 13 generated blocks, full methodology and reproduction instructions), rewrote the README `## Performance` section into a short honest speed story, extended `benchmarks/README.md` with the full re-run procedure and corrected R-FPCA documentation, and shipped a RECORD-hash-verified `multimethod` patch — then proved every table regenerates byte-for-byte in a fresh clone with the prose backed entirely by generated numbers.

## Performance

- **Duration:** ~55 min
- **Started:** 2026-09-25T00:20:00Z (approx.)
- **Completed:** 2026-09-25T01:15:00Z (approx.)
- **Tasks:** 2
- **Files modified:** 4 (1 created, 3 modified)

## Accomplishments

- `documentation/PERFORMANCE.md` authored around all 13 generated blocks in the plan's exact heading order: How to read this report, Summary (key-findings), Headline (N=500,M=200), Head-to-head across the grid, Optimization results (controlled A/B), Before/after vs the Phase 113 baseline (4 subsections: crosslang-fdars, internal, control, not-rerun), N×M scaling, Methodology (environment, agreement gate, timing statistics, criterion settings, data parity, threading — including a required background-load disclosure), Reproduction (numbered steps + measured runtime/footprint), Known limitations and disclosures.
- Honesty clause enforced end-to-end: `face_covariance` is headlined at the pinned, noise-clean **1.19×** (`phase116_face_covariance_n200_m30`), explicitly distinguished from the larger but unpinned/noisy **3.45×** Phase 115 figure; elastic alignment is stated plainly as numeric-identical and within noise (CIs overlap); SVD is stated as already near-optimal with no change made.
- README `## Performance` rewritten: intro sentence, the generated headline table, 2 takeaway lines quoting only generated numbers, a link to the full report, and the existing rayon/WASM sentence kept as qualitative context. Guides list gains a `Performance` bullet.
- `benchmarks/README.md` gained a "Full re-run (Phase 116)" section (the `--full` command plus the inline resumable sequence actually used to produce the published numbers), an extended Result Schema (`comparison_baseline.csv`, `rerun_log.json`, `ab/manifest.json`), a corrected R FPCA note (the gated path is the Simpson-weighted `r_svd` port, not `fda::pca.fd`; native `pca.fd` is timed separately as `fpca_native_r`), the `renv::restore()` clean-clone invocation (documented, not re-executed this phase), the real `results/*` + negations gitignore carve-out (replacing the stale "results/ are gitignored" note), and a new "Python 3.14 multimethod patch" subsection.
- `benchmarks/patches/multimethod-2.1-py314.patch` produced by downloading the pristine `multimethod==2.1` wheel, verifying its `__init__.py` sha256 (base64-urlsafe) matches the venv's own `dist-info/RECORD` entry (proving provenance), then diffing it against the venv's patched copy. A reverse dry-run apply (`patch --dry-run -R -p1`) confirms the patch equals the venv's actual modification.
- Proved reproducibility: cloned the repo to a scratch directory after committing and ran `render_report.py --check` + `--check-prose` inside the clone using only committed inputs — both passed, proving every published table regenerates byte-for-byte from committed data alone.

## Task Commits

Each task was committed atomically:

1. **Task 1: Author documentation/PERFORMANCE.md around all 13 generated blocks and render it** - `a7c0006d` (feat)
2. **Task 2: README speed story + Guides link, benchmarks/README.md update, multimethod reproduction patch, clean-clone render check** - `f05c0f42` (feat)

**Plan metadata:** (this commit)

## Files Created/Modified

- `documentation/PERFORMANCE.md` - full 10-section published report, all 13 generated blocks rendered
- `README.md` - `## Performance` rewritten into a speed story + Guides bullet
- `benchmarks/README.md` - full re-run section, extended Result Schema, corrected R FPCA note, renv/gitignore/multimethod documentation
- `benchmarks/patches/multimethod-2.1-py314.patch` - new reproduction artifact, RECORD-hash-verified

## Decisions Made

See `key-decisions` in the frontmatter above (headline A/B selection honesty, before/after-as-traceability framing, background-load disclosure, renv restore documented-not-executed).

## Deviations from Plan

None — plan executed exactly as written. Both tasks' `<read_first>` files were read before editing; the plan's specific heading order, block placement, and prose content requirements (D-01 through D-15) were all followed. The one addition beyond the plan's literal action list — an explicit background-load disclosure sentence in Methodology and Known Limitations — was mandated by the plan's `must_haves.truths` list ("Methodology discloses... the noise-gate definition... orphan groups and timeouts (D-15)") combined with the executor prompt's `<critical_facts_for_the_prose>` DISCLOSURE REQUIRED instruction; not a deviation from the plan, just a fact this SUMMARY calls out explicitly since it required independent judgment about placement rather than being a literal plan action-list item.

## Issues Encountered

None. The initial full document skeleton (headings + empty marker pairs + prose) was written first, then `render_report.py` was run once to fill all 13 blocks against the real committed `comparison.csv`/`comparison_baseline.csv`/`ab/manifest.json`/`rerun_log.json`/`env.json` — this is the "generated-block-then-prose" pattern noted in `tech-stack.patterns`, chosen because the plan's literal instruction ("First, run render_report.py ... so that you write the prose against the real numbers") cannot execute against a doc that doesn't yet carry all 13 marker pairs; previewing each `emit_*` function's output directly via `render_report` module calls (read-only, no doc writes) served the same "write against real numbers" purpose before the skeleton was committed to disk.

## Known Stubs

None. All 13 generated blocks are populated with real data from the committed Phase 116 re-run; no placeholder or stub content remains in `documentation/PERFORMANCE.md`.

## Threat Flags

None beyond the plan's own `<threat_model>` (T-116-24, T-116-SC, T-116-25), all implemented as specified: `--check-prose` enforces the no-unsupported-claims mitigation (verified passing); the multimethod patch is download-only (nothing installed) with RECORD-hash-verified provenance and a reverse-dry-run-apply proof; the environment block discloses only already-public CPU/library/version metadata.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Phase 116 is now fully executed (7/7 plans complete). Ready for `/gsd-verify-work 116` and Phase 117 (Release Preparation & Verification) — the terminal phase that bumps 0.46.0 → 0.47.0, updates CHANGELOGs, and runs the full whole-crate gate battery. No blockers carried forward from this plan.

## Self-Check: PASSED

- Created/modified files verified present: `documentation/PERFORMANCE.md`, `README.md`, `benchmarks/README.md`, `benchmarks/patches/multimethod-2.1-py314.patch` — all present on disk.
- Commits verified: `git log --oneline --all | grep -q a7c0006d` and `f05c0f42` both found; `git rev-list --count b6606933..HEAD` = 2 (matches `actuals.commits`).
- All plan-level `<verification>` and task `<verify>`/`<acceptance_criteria>` commands re-run and passing: `render_report.py --check` (pass), `--check-prose` (pass) both in the working tree AND in a fresh clone; `pytest benchmarks/ -q` (192 passed); `git diff --name-only ec57fa01 HEAD -- fdars-core/src benchmarks/gate.py` (empty); no image files in the diff.

---
*Phase: 116-cross-language-re-run-published-report*
*Completed: 2026-09-25*
