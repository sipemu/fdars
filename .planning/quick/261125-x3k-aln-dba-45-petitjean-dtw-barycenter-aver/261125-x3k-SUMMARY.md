---
quick_id: 261125-x3k
issue: 45
status: complete
---
# Summary — ALN-DBA (GH #45): Petitjean DTW Barycenter Averaging

## Commits
- `f47893d4` chore: fix needless_range_loop clippy lints in tests (cargo 1.100) — pre-step, 9 sites in 5
  files under `fdars-core/tests/` (2 iterator rewrites where the length is asserted just before, 7
  narrowly scoped `#[allow(clippy::needless_range_loop)]` where the index also feeds messages/other data).
- `7bf0c063` feat: Petitjean DTW Barycenter Averaging with mean/medoid init (refs #45)
- `2c1759aa` feat: DTW k-means with DBA centroids + CHANGELOG (closes #45)

## Public API added (all additive; re-exported from `metric`, crate root and prelude)
- `fdars-core/src/metric/dba.rs`: `dtw_barycenter_averaging(&FdMatrix, &DbaConfig) -> Result<DbaResult>`;
  `DbaConfig { max_iter=30, tol=1e-5, window=DtwWindow::None, init=DbaInit::Mean, barycenter_size=None, weights=None }`;
  `DbaInit { Mean, Medoid, Custom(Vec<f64>) }`; `DbaResult { barycenter, cost, n_iter, converged }`.
- `fdars-core/src/dtw_kmeans.rs`: `dtw_kmeans(&FdMatrix, &DtwKMeansConfig) -> Result<DtwKMeansResult>`;
  `DtwKMeansConfig { n_clusters=2, n_init=10, max_iter=50, tol=1e-6, seed=0, window, init, dba_max_iter=30, dba_tol=1e-5 }`;
  `DtwKMeansInit { KMeansPlusPlus, Centroids(FdMatrix) }`; `DtwKMeansResult { centroids, cluster, inertia, iter, converged, n_init_best, window }` + `predict`.
- Reuses `dtw_path` (tslearn tie-breaking), `dtw_distance_windowed` and `DtwWindow` from GH #44.
  No DTW/soft-DTW k-means existed before, so `dtw_kmeans` is new rather than an option on an existing path.

## Validation (tslearn 0.10.0, dtaidistance 2.5.1 in ~/.cache/fdars-golden-venv; script: `golden_dba.py`)
- tslearn docstring examples (exact) + golden barycenters/costs at 1e-10: mean init, Sakoe-Chiba r=1,
  Itakura 2.0, barycenter_size=12, weights, custom init, medoid init (medoid index 4 cross-checked by brute force).
- dtaidistance `dba` (1 step) and `dba_loop(max_it=10, thr=None)` from a fixed init at 1e-12.
- tslearn `TimeSeriesKMeans(metric="dtw", init=K[[0,1]], n_init=1)`: labels, both centroids (1e-10),
  final inertia, n_iter=3.
- Invariants: monotone non-increasing DBA cost, identical series = fixed point, radius-0 band = weighted
  arithmetic mean (and Euclidean k-means), weight-scale invariance, labels are argmin, inertia consistent,
  k=n singletons, empty-cluster re-seeding, determinism, predict(train) == labels, error paths.

## Gates
- `cargo fmt --all` clean; `cargo clippy -p fdars-core --all-targets --features linalg,parallel -- -D warnings` clean.
- `cargo test --lib` filters dtw_kmeans / metric::dba / metric::dtw / metric::window: 32 passed; doc tests 4 passed.
- `cargo check --features serde` and `--no-default-features` OK; `cargo doc` no new warnings; pmat max cognitive 16 (dtw_kmeans) / 15 (dba).

## Deviations / notes
- `DbaResult::cost` is the inertia of the *returned* barycenter (tslearn returns the pre-update loss);
  `DtwKMeansResult::inertia` likewise uses final centroids/labels. Empty clusters are re-seeded in place
  (kshape convention) instead of tslearn's restart. Medoid = argmin Σ w·DTW² (DBA objective);
  dtaidistance's sampled medoid uses root distances.
- tslearn quirk found: `dtw_barycenter_averaging` forwards `metric_params` raw to numba, so
  `global_constraint="sakoe_chiba"` is silently ignored there (integer code 2 is needed), while
  `TimeSeriesKMeans` passes the same params to both `cdist_dtw` (string needed) and DBA — so no consistent
  windowed tslearn k-means golden exists; windowed k-means is covered by invariants instead.
- The runbook mentioned ~18 clippy findings; there were 9 sites (18 diagnostic lines).

## Remaining
- None for #45. Multivariate (d > 1) DBA is not covered (FdMatrix is univariate per row).
