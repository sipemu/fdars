---
quick_id: 261126-cxk
slug: fix-rustdoc-dwarnings-errors-so-ci-docum
type: quick
---
# Fix rustdoc -Dwarnings errors so CI Documentation job passes

Goal: `RUSTDOCFLAGS=-Dwarnings cargo doc --no-deps --features linalg,parallel,serde` (from fdars-core/) exits 0.

Tasks (doc comments only, no code/visibility changes, no rustdoc allow attrs):
1. Private-item links -> public re-export path if any, else plain inline code; `[module documentation](self|super)` -> plain text.
2. Unresolved links -> correct `crate::` path or escaped brackets / inline code.
3. Ambiguous fn/mod -> `fn@` / `mod@` disambiguators.
4. Redundant explicit targets -> `[name]`.
5. Verify: CI doc command, cargo fmt, clippy --all-targets -D warnings, filtered doctests.
