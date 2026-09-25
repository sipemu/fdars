---
phase: "116"
slug: "cross-language-re-run-published-report"
status: verified
threats_open: 0
asvs_level: 1
created: "2026-09-25"
---

# Phase 116 — Security

> Per-phase security contract: threat register, accepted risks, and audit trail.

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| harness orchestrator → subprocesses (cargo bench, Python, Rscript) | run_all.py spawns local toolchains | argv lists only (no shell); bench names validated against existing files |
| committed result files → render_report.py → published docs | report generated from CSV/JSON on disk | numeric timings + fixed op names; free text escaped |
| scratch A/B worktrees (target/ab116) → committed evidence | historical commits built locally | criterion estimates.json copied; worktrees removed |
| gate output → decision to write comparison.csv | fail-closed agreement boundary | pass/fail verdict |

---

## Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation | Status |
|-----------|----------|-----------|----------|-------------|------------|--------|
| T-116-01 | Elevation of privilege | run_rust_bench / _run_with_timeout | low | mitigate | list-form argv only (no shell=True); bench name validated against an existing fdars-core/benches/<name>.rs file before spawning | closed |
| T-116-02 | Denial of service / result integrity | _run_with_timeout | medium | mitigate | child started with start_new_session=True; whole process group SIGKILLed on timeout so no orphaned bench binary skews later timings | closed |
| T-116-03 | Tampering | comparison_baseline.csv | medium | mitigate | snapshot taken from git object 597f857d and cmp-verified against it; committed in this plan before any re-run | closed |
| T-116-04 | Tampering | emit_headline markdown | low | mitigate | headline cells are numeric values and fixed op names only; free-text columns are never emitted by the tracer (plan 116-04 adds a general cell escaper) | closed |
| T-116-05 | Information disclosure | env_baseline.json | low | accept | contains CPU model / library paths only, already public since the Phase 113 commit; no secrets | closed |
| T-116-06 | Tampering | crosslang_grid.rs CSV loaders | low | accept | the file names are built only from the integer grid constants (curves_n{n}_m{m}.csv, y_n{n}.csv, elastic_n2_m{m}.csv); read-only; malformed input panics inside a dev-only bench binary, never library code | closed |
| T-116-07 | Tampering | collect_rust.py | low | mitigate | the existing resolve()+relative_to() path-escape guard (T-112-01) is unchanged; the new test uses a temp dir under the same guard | closed |
| T-116-08 | Tampering | _reset_cell_files | medium | mitigate | deletes only `python_n{n}_m{m}.json` / `r_n{n}_m{m}.json` whose n, m passed `_parse_cell` validation against GRID_N x GRID_M; no user-supplied path fragments | closed |
| T-116-09 | Tampering | rerun_init criterion archive | low | mitigate | renames (never deletes) target/criterion to a timestamped sibling; refuses to run when a log already exists, so a resumed run never discards fresh results | closed |
| T-116-10 | Denial of service / result integrity | timed-out children | medium | mitigate | process-group SIGKILL (start_new_session + killpg) is unit-tested with a grandchild; prevents orphaned CPU load that would bias later timings | closed |
| T-116-11 | Tampering | rerun_log.json parsing | low | mitigate | malformed log → SystemExit with a clear message (fail-closed); atomic writes via temp file + os.replace | closed |
| T-116-12 | Elevation of privilege | subprocess construction | low | mitigate | list-form argv only; acceptance greps that no call site enables a shell | closed |
| T-116-13 | Tampering | emitters (notes/source columns) | low | mitigate | `_md_cell` escapes pipe and strips CR/LF for every free-text cell; unit-tested | closed |
| T-116-14 | Tampering / info disclosure | load_estimates | medium | mitigate | rejects absolute paths and any path that resolves outside benchmarks/results/ab (resolve + relative_to); unit-tested | closed |
| T-116-15 | Repudiation (unsupported claims) | key findings / headline | medium | mitigate | claims computed only from inputs; noisy or non-significant results cannot be headlined (select_headline_ab eligibility, dagger exclusion); --check detects hand edits | closed |
| T-116-16 | Denial of service | malformed input files | low | mitigate | fail-closed: missing or unparsable required input -> stderr message + exit 1, no partial doc write | closed |
| T-116-17 | Tampering | scratch worktrees | low | mitigate | worktrees are detached under target/ (gitignored); no commits are made from them; bench edits happen only there and are verified absent from the main checkout; removed + pruned at the end | closed |
| T-116-18 | Tampering (supply chain) | dependency versions across A/B sides | medium | mitigate | the same main-checkout Cargo.lock in both worktrees + `--locked` (stop on failure, never drop the flag) | closed |
| T-116-19 | Repudiation (cherry-picking) | bump rounds | medium | mitigate | every round's raw estimates committed and listed in `rounds`; the final round is authoritative; the gate threshold is never relaxed | closed |
| T-116-SC | Tampering | package installs | low | accept | no new packages are installed; cargo resolves only crates already in the committed-equivalent Cargo.lock | closed |
| T-116-20 | Tampering (data integrity) | comparison.csv freshness | high | mitigate | criterion output archived before the run (rerun_init); freshness verify asserts no orphan rows and no bit-identical Python/R rows; head_sha recorded in the log | closed |
| T-116-21 | Repudiation | timeouts / errors | medium | mitigate | every chunk/cell status is logged; timeouts are final (requeue refuses them); the report lists them | closed |
| T-116-22 | Tampering | agreement gate | high | mitigate | gate.py verified unchanged (git diff against ec57fa01); a gate failure stops the plan before assembly | closed |
| T-116-23 | Denial of service / integrity | concurrent CPU load | medium | mitigate | quiet-machine check (no rustc/cargo bench/Rscript, load average < 2.0) before each measurement task; inline sequential execution only — pre-run quiet check passed; background load recurred mid-run and is disclosed in PERFORMANCE.md (Methodology + Known limitations); headline cells pass the 5% noise gate | closed |
| T-116-24 | Repudiation (unsupported claims) | README / PERFORMANCE prose | medium | mitigate | `--check-prose` fails on any × / ms number absent from generated blocks; `--check` + clean-clone check prove tables regenerate from committed inputs; human-check on framing | closed |
| T-116-SC (116-07) | Tampering (supply chain) | multimethod 2.1 wheel download | low | mitigate | download only (never installed), version pinned in requirements.lock and already installed since Phase 112; the pristine file hash is verified against the venv's dist-info RECORD before diffing; the reverse-apply dry run proves the patch equals the venv modification | closed |
| T-116-25 | Information disclosure | environment block | low | accept | CPU model / library paths / versions only, already public since Phase 113; no secrets or usernames beyond paths already committed | closed |

*Status: open · closed · open — below high threshold (non-blocking)*

---

## Accepted Risks Log

| Risk ID | Threat Ref | Rationale | Accepted By | Date |
|---------|------------|-----------|-------------|------|
| AR-116-05 | T-116-05 | contains CPU model / library paths only, already public since the Phase 113 commit; no secrets | plan-time disposition (116 planner) | 2026-09-24 |
| AR-116-06 | T-116-06 | the file names are built only from the integer grid constants (curves_n{n}_m{m}.csv, y_n{n}.csv, elastic_n2_m{m}.csv); read-only; malformed input panics inside a dev-only bench binary, never library code | plan-time disposition (116 planner) | 2026-09-24 |
| AR-116-SC | T-116-SC | no new packages are installed; cargo resolves only crates already in the committed-equivalent Cargo.lock | plan-time disposition (116 planner) | 2026-09-24 |
| AR-116-25 | T-116-25 | CPU model / library paths / versions only, already public since Phase 113; no secrets or usernames beyond paths already committed | plan-time disposition (116 planner) | 2026-09-24 |

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-09-25 | 27 | 27 | 0 | orchestrator (L1 grep-depth; auditor skipped per short-circuit: threats_open 0, plan-time register, ASVS 1) |

Evidence (L1): no `shell=True` call sites in benchmarks/*.py (3 hits are comments); `start_new_session=True` + `killpg` present in run_all.py; atomic `os.replace` log writes; `_md_cell` escaping in render_report.py; `resolve()`+`relative_to()` path guards in render_report.py/collect_rust.py; `rust_requeue` refuses timeout/skipped-inferred-timeout; `git diff ec57fa01 HEAD -- benchmarks/gate.py fdars-core/src` empty; `target/` gitignored; A/B built `--locked` (manifest). Code review CR-01 (fail-open gate path) fixed + unit-tested before this audit.

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed

**Approval:** verified 2026-09-25
