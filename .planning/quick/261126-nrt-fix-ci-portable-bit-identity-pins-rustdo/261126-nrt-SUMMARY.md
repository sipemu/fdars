---
quick_id: 261126-nrt
status: complete
commits: [6f0cb570, 6147fdf4]
---
# Summary: fix "Rust CI" — rustdoc errors + portable bit-identity pins

## Problem 1 — Documentation job (6f0cb570)
- `metric/dba.rs`: `[`dtw_path`]`, `[`dtw_distance_windowed`]` unresolved → `super::dtw::…` paths.
- `metric/elastic_family/mod.rs:776`: `[module docs](self)` in the `*_multivariate` macro expands inside
  the private per-distance submodules (9 "links to private item" errors) → `crate::metric::elastic_family`.
- `RUSTDOCFLAGS=-Dwarnings cargo doc --no-deps --features linalg,parallel,serde` exits 0.

## Problem 2 — CI-only test failures (6147fdf4)
**Evidence:**
- Same rustc (1.100.0) locally and on CI; CI runner image is ubuntu-26.04 (glibc 2.43), not 24.04.
- The same test binary passes under ubuntu:24.04 (glibc 2.39) and ubuntu:26.04 (2.43) containers on the
  local i9-13900H, and with CI's freshly resolved dependency versions (Cargo.lock is gitignored; CI got
  nalgebra 0.33.3, matrixmultiply 0.3.11 with new AVX-512 kernels).
- gdb: 794 breakpoints on matrixmultiply/faer/gemm/nano-gemm/private-gemm/rustfft/std_detect — none hit
  on these paths. The faer/gemm/AVX-512 hypothesis is refuted; there is no Rust-side runtime dispatch.
- The only CPU-dispatched code reached is glibc libm (IFUNC: `__exp/__log/__pow/__atan2/__asin/__acos
  _finite`, `sin`, `cos`, `tan`, `atan`). Resolvers test FMA/AVX2/FMA4/AVX only, with no AVX-512 bit.
  Forcing the SSE2 kernels (`GLIBC_TUNABLES=glibc.cpu.hwcaps=-AVX2,-FMA`) flips 12 lib tests, including all
  4 CI failures, and gives exactly CI's closed-Karcher seeds `[20, 0, 39]`.
- Not reproduced: CI's exact hashes. CI fails 4 tests where SSE2 fails 12, so CI hits a different
  last-ulp perturbation that this AVX2-only host cannot produce. hypot (shared by the 4) is not dispatched.

**Fix:** `ValuePin` (per-field `[Σx, Σ|x|, Σx², Σwx]` at rtol) + `libm_canary()` in `test_helpers.rs`.
| Test | Conversion | Measured FMA↔SSE2 spread |
|---|---|---|
| continuous `default_results_bit_identical_to_pre_periodic_pin` | value pin 1e-10 | 1.2e-15 |
| curve_shape `default_warp_optim_bit_identical_to_pre_change` | value pin 1e-10 | 2.6e-14 |
| curve_shape `explicit_default_warp_optim_and_generic_dp_match_classic` | in-process bit identity vs default | — |
| curve_shape `karcher_closed_elastic_one_half_matches_srvf` | seed mismatch allowed only as proven exact symmetry tie (ellipse half-turn restart, rigid residual < 1e-12; discriminating control on `asym`) | — |
| hygiene `outlier_default_is_bit_identical_to_pre_change` | value pin 1e-10 | 2.2e-15 |
| optim `rbfgs_1d_bit_identical_to_pre_multichannel` | value pin 1e-10 | 2.3e-14 |
| optim_coverage `touched_entry_points_bit_identical_to_pre_change` | value pin, 1e-10 DP block / 1e-6 RBFGS block | 2.3e-8 (RBFGS) |
| conformal `default_results_bit_identical_to_pre_72_pin` | value pin 1e-10 | 0 |
| bayesian `default_pcn_sampler_is_bit_identical_to_pcn_release` (3 pins) | value pins 1e-10 | 9e-16 |
| bayesian `default_infhmc_sampler_is_bit_identical_to_pre_mode_jump_release` | chaotic (O(1) spread): value pin enforced only when libm canary matches the recording machine; length always | O(1) |

All references were recorded on 925ef514, which reproduced every old hash bit-for-bit locally.
Verified: all pass under FMA and SSE2 kernels, locally (glibc 2.44) and in ubuntu:26.04 (glibc 2.43).

**Left unchanged (fragile, not hash pins, pass on CI):** `bayesian_group::parallel_tempering_preserves_the_posterior`
and `sparse::tests_closed::closed_mean_recovers_curve_from_different_grids_and_starts` (fail under the SSE2 kernels).
Other literal pins (prune_default, hygiene closed hoist, optim classic/default, classification, spm, …)
are robust to the SSE2 kernels and were left unchanged.

**Gates:** fmt ✓, clippy --all-targets -D warnings ✓, CI doc command ✓, `--lib alignment::` 631 passed ✓,
`conformal::partial` 24 passed ✓.
