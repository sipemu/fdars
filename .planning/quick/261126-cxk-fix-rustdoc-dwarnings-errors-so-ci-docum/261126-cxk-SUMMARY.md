---
quick_id: 261126-cxk
slug: fix-rustdoc-dwarnings-errors-so-ci-docum
status: complete
commit: fef69806
---
# Summary: rustdoc -Dwarnings fixes

- Errors before: 99 (rustdoc) -> after: 0; CI doc command exits 0.
- 67 private-item links (`*_1d` forwarders in depth/metric/fdata, wavelet/frechet private helpers, `crate::linalg`, etc.) converted to plain inline code (script-driven from the rustdoc log).
- 12 `[module documentation](self|super)` links -> plain text.
- Unresolved: `monomial_basis`, `cov_irreg`, `fpc_permutation_importance`, `FdMatrix` -> public `crate::` paths; `Vec[j]`, `π̂[j]`, `[i,j]`, `[0,1]`, `[ASSUMED]` escaped/inline-code; `WaveletFamily::Daubechies(4)` -> link to variant + `(4)`.
- Ambiguous: `fn@boost_fosr`, `fn@boost_fofr`, `mod@crate::pace_fpca`.
- Redundant explicit targets simplified (clustering_advanced, conformal/partial).
- cargo fmt clean; clippy --all-targets -D warnings clean; doctests (wavelet, depth, boost filters) pass.
