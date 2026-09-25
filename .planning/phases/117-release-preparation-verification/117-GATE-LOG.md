# Phase 117 — Release Gate Log (fdars-core 0.47.0)

- Gated source commit: `384381a466348c73dc56b9ca152ef10ddbcfb5f9`
- Date: 2026-09-25
- Toolchain: rustc 1.97.0 (2d8144b78 2026-07-07), cargo 1.97.0 (c980f4866 2026-06-30); pmat 3.18.0
- Disk (/home): 49G free before freeing target/debug/{incremental,examples}; 50G free after. /tmp tmpfs 16G free (doctests + serde routed to TMPDIR=/home/simonm/.cache/fdars-bench-tmp)
- pmat_decision: refactor
- Raw logs: `/home/simonm/.cache/fdars-bench-tmp/117-gates/*.log` (local, not committed)
- Executed inline by the orchestrator, one gate per foreground call (570 s bound).

| # | Gate | Command | Exit | Evidence | Verdict |
|---|------|---------|------|----------|---------|
| 1 | fmt | `cargo fmt --check` | 0 | no diff | PASS |
| 2 | clippy | `cargo clippy -p fdars-core --all-targets --features linalg,parallel -- -D warnings` | 0 | Finished, 0 warnings | PASS |
| 3 | serde-build | `TMPDIR=~/.cache/fdars-bench-tmp cargo build -p fdars-core --features serde` | 0 | Finished | PASS |
| 4 | examples-build | `cargo build -p fdars-core --examples --features linalg,parallel` | 0 | 28/28 binaries present | PASS |
| 5 | test-compile | `cargo test -p fdars-core --features linalg,parallel --no-run` | 0 | all test targets compiled | PASS |
| 6 | test-lib | `cargo test -p fdars-core --features linalg,parallel --lib` | 0 | test result: ok. 2942 passed; 0 failed; 0 ignored | PASS |
| 7 | test-integration | `cargo test -p fdars-core --features linalg,parallel --no-fail-fast --test '*'` | 0 | 21/21 binaries ok, 613 passed, 0 failed (known co_cluster/svd_sign golden flake did not occur) | PASS |
| 8 | test-doc | `TMPDIR=~/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel --doc` | 0 | test result: ok. 215 passed; 0 failed; 5 ignored | PASS |
| 9 | package-verify | `TMPDIR=~/.cache/fdars-bench-tmp cargo package -p fdars-core` (no --allow-dirty) | 0 | Packaged 401 files, 6.7MiB (1.4MiB compressed); verify build Finished; tarball Cargo.toml version = "0.47.0"; .cargo_vcs_info.json sha1 976a1c84, not dirty | PASS |
| 10 | package-list | `tar -tzf target/package/fdars-core-0.47.0.crate` | 0 | 401 files; 0 entries under benchmarks/ .pmat/ .pmat-metrics/ .planning/ target/ .claude/; benches/crosslang_grid.rs, CHANGELOG.md, README.md present | PASS |
| 11 | package-bench-fallback | extracted tarball, `cargo bench --offline --locked --bench crosslang_grid -- --test n100_m50` (no ../benchmarks, FDARS_CROSSLANG_* unset) | 0 | hermetic bench build Finished (53 s); `Testing crosslang_<op>/n100_m50` for all 6 ops via synthetic-data fallback | PASS |
| 12 | pmat | `cd fdars-core && pmat quality-gate --checks complexity,satd,dead-code,duplicates --fail-on-violation` | 0 | Quality Gate: PASSED; Total violations: 0 (dp_fill_grid refactored in 117-02) | PASS |
| 13 | harness-pytest | `benchmarks/.venv/bin/python -m pytest benchmarks/ -q` | 0 | 209 passed | PASS |
| 14 | harness-render-check | `render_report.py --check` + `--check-prose` | 0 | both exit 0 (check-prose OK) | PASS |
| 15 | changelog-claims | [0.47.0] sections of CHANGELOG.md + fdars-core/CHANGELOG.md | 0 | 1.19x present in both; unpinned Phase 115 figure absent in both | PASS |
| 16 | no-tag-no-publish | `git tag -l 'v0.47*'`; `git diff --name-only 0c7dc99a HEAD -- fdars-core/src` | 0 | no v0.47* tag; only fdars-core/src/alignment/mod.rs changed (pmat_decision: refactor); no cargo publish run | PASS |

cargo half of the gate battery: 8/8 PASS
Full gate battery: 16/16 PASS

## Deferred operator step (NOT performed in Phase 117 — DO NOT RUN in-phase)

For the human operator, after reviewing the release commit:

1. Push `main` so CI (including the `pmat-quality-gate` job) runs on the release commit, and confirm it is green.
2. Tag the Phase 117 release commit and push the tag: `git tag v0.47.0 && git push origin v0.47.0`. `release.yml` then runs `cargo publish --package fdars-core`.
3. The publish step is `continue-on-error` — confirm success in the Actions log and on crates.io (fdars-core 0.47.0 listed; docs.rs build green).
4. Notes: publishing is irreversible (yank-only). crates.io currently lists 0.40.0 / 0.42.0 / 0.44.0; 0.47.0 supersedes the never-published 0.45.0 / 0.46.0. The README / GETTING-STARTED `"0.47"` install snippets resolve only after this publish.

## REL-01 verdict

REL-01: SATISFIED — all 16 gates PASS
