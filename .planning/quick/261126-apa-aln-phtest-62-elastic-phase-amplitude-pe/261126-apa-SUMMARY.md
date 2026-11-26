---
quick_id: 261126-apa
status: complete
issue: 62
backlog_id: ALN-PHTEST
commits:
  - 1d4d5ce1
---
# 261126-apa SUMMARY — ALN-PHTEST (GH #62): elastic amplitude/phase permutation test

## Commits
- `1d4d5ce1` feat(quick-261126-apa): elastic amplitude/phase permutation test with PERMANOVA (closes #62)

## Public API added (additive, non-breaking)
- `alignment::elastic_phase_amplitude_test(data, argvals, labels, &PhaseAmplitudeTestConfig)
  -> Result<PhaseAmplitudeTestResult, FdarError>`
  - `PhaseAmplitudeTestConfig` (sealed, `Default`: n_perm 999, seed 42, lambda 0, max_iter 20,
    tol 1e-4, amplitude_metric Pairwise, joint true)
  - `PhaseAmplitudeTestResult { amplitude, phase, joint: Option, amplitude_distances,
    phase_distances, alignment: KarcherMeanResult, group_labels, group_sizes }`
  - `AmplitudeMetric { Pairwise (default), TemplateAligned }`
- `inference::permanova(dist, labels, n_perm, seed) -> Result<PermanovaResult, FdarError>`,
  `PermanovaResult` (serde-gated): statistic, p_value, ss_between/within/total, r_squared, dfs, n_perm.
- Re-exported at crate root; `PhaseAmplitudeTestResult` in prelude.

## Design
- Label-free joint Karcher alignment of all curves (labels never reach the alignment) ⇒ distance
  matrices are functions of the unlabelled sample ⇒ label permutation with fixed matrices is an
  exact test under exchangeability (documented in module docs).
- Phase distance: Fisher–Rao geodesic arccos⟨ψᵢ,ψⱼ⟩ between Karcher warps.
- Joint distance: √(d_A²/mean d_A² + d_P²/mean d_P²) (all-zero part dropped).
- Permutations through the existing `permutation_test::permutation_pvalue` scaffold (seed + k reseed,
  parallel ≥ 200 perms, bit-identical seq/parallel); vegan tie tolerance F* ≥ F − √ε passed as the
  observed threshold. Same seed ⇒ the amplitude/phase/joint tests share permutations.
- Arbitrary `usize` labels mapped to contiguous ids.

## Validation
- R vegan 2.7.8 `adonis2` (installed into ~/.cache/R-lib; script `golden_vegan_adonis2.R` in this
  dir): 3 cases (Euclidean 3 balanced groups; Manhattan 2 unbalanced groups; Euclidean 4 unbalanced
  groups under null). pseudo-F, SS_A/SS_W/SS_T, R² match to 1e-10 relative; p-values (9999 perms each
  side) within 4 combined MC standard errors.
- Hand-computed 1-D two-cluster case (F = 150); label-renaming and observation-reordering invariance;
  all-zero distances (F = 0, p = 1); input validation.
- PERMANOVA size under H0 (400 reps, α = .05) within 4σ binomial band.
- Elastic test: MC size under full H0 (100 reps, n = 12, m = 30) for pairwise amplitude,
  template-aligned amplitude, phase and joint, all within 3σ binomial band; power: phase-only shift
  → phase p ≤ .01, amplitude p > .05, joint ≤ .05; amplitude-only scaling → amplitude p ≤ .01,
  phase p > .05; template metric detects scaling; closed-form phase distance (γ = t² vs id:
  arccos(2√2/3)); determinism; alignment and distances independent of labels; statistic equals
  standalone `permanova` on returned matrices.

## Deviation / finding
- The issue suggested amplitude distance = SRSF distance after template alignment. Exploratory MC
  (60 reps, phase-only shift, 10+10 curves) showed that template-aligned SRSF distances leak
  group-structured residual misalignment into the amplitude test when only phase differs (6/60
  rejections, mean p ≈ 0.23–0.33 at m = 60–100; 4/60, mean p ≈ 0.40 at m = 120), while exact pairwise
  elastic distances were calibrated (3/60, mean p ≈ 0.56). Pairwise is therefore the default;
  TemplateAligned is kept as the faster option with the caveat documented on `AmplitudeMetric`.
  Under the full H0 both are exact (size test).

## Gates
- `cargo fmt --all` clean.
- `cargo test -p fdars-core --lib -- permanova phase_amplitude_test permutation_test`: 20 passed.
- `cargo test -p fdars-core --doc -- permanova phase_amplitude`: 2 passed.
- `cargo clippy -p fdars-core --all-targets --features linalg,parallel -- -D warnings`: clean.
- `cargo check -p fdars-core --features serde`: clean; `--no-default-features`: clean.
- pmat complexity: max cognitive 20 (phase_amplitude_test.rs), 11 (permanova.rs).
- No full-suite run (per runbook).

## Remaining
- None for #62. Possible follow-ups: energy-distance statistic, PERMDISP (dispersion) companion test,
  stratified permutations.
