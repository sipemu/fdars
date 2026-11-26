# 261126-apa — ALN-PHTEST (GH #62): elastic amplitude/phase permutation test

## Goal
Permutation test of group differences in amplitude and in phase in the elastic quotient space,
2+ groups, seeded + deterministic + parallel over permutations.

## Public API
- `inference::permanova(dist: &FdMatrix, labels: &[usize], n_perm, seed) -> Result<PermanovaResult>`
  — one-way PERMANOVA (Anderson 2001) pseudo-F on an arbitrary symmetric distance matrix,
  vegan `adonis2` semantics (SS on squared distances, p = (#{F* >= F - sqrt(eps)} + 1)/(n_perm + 1)).
- `alignment::elastic_phase_amplitude_test(data, argvals, labels, &PhaseAmplitudeTestConfig)
  -> Result<PhaseAmplitudeTestResult>`; `PhaseAmplitudeTestConfig` (sealed, Default),
  `AmplitudeMetric { TemplateAligned, Pairwise }`.

## Algorithm
1. Label-free joint alignment: `karcher_mean(data, argvals, max_iter, tol, lambda)` on ALL curves.
2. Amplitude distances: TemplateAligned = L2 between SRSFs of curves aligned to the common template
   (default); Pairwise = exact pairwise elastic distance matrix (O(n^2) DP).
3. Phase distances: Fisher–Rao geodesic arccos<psi_i, psi_j> between the Karcher warps (psi = sqrt(gamma')).
4. Joint (optional): d_J^2 = d_A^2 / mean(d_A^2) + d_P^2 / mean(d_P^2) (scale-free).
5. PERMANOVA pseudo-F + permutation p-value per distance via `permutation_test::permutation_pvalue`
   (seed + perm reseeding, parallel above threshold, bit-identical across thread counts); same seed
   ⇒ the three tests use identical permutations.
Validity: distances never see labels, so under H0 (exchangeability) the label permutation leaves the
distance matrices unchanged → exact permutation test.

## Validation
- pseudo-F, SS, R² vs R vegan 2.7.8 `adonis2` on deterministic Euclidean and non-Euclidean (L1)
  distance matrices to 1e-10; p-values within MC error.
- Hand-computed tiny case; invariances (label renaming, permutation of observations).
- MC size test under H0 (binomial tolerance); power: phase-only → phase rejects, amplitude not; and
  amplitude-only (scaling) → amplitude rejects, phase not.

## Tasks
1. inference/permanova.rs + tests + vegan golden. 2. alignment/phase_amplitude_test.rs + tests.
3. Re-exports, CHANGELOGs, gates, commits.
