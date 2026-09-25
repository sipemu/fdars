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

cargo half of the gate battery: 8/8 PASS
