# Pitfalls Research

**Domain:** k-Shape clustering + Shape-Based Distance (SBD) via FFT cross-correlation — Rust numerical FDA library (fdars-core v0.34.0)
**Researched:** 2026-09-02
**Confidence:** HIGH

---

## Critical Pitfalls

### Pitfall 1: FFT Length Too Short — Circular Cross-Correlation Wraps and Corrupts Shifts

**What goes wrong:**
The SBD between two series of length `n` is computed via the FFT of their cross-correlation. If the FFT is planned for length `n` (or any length < `2n - 1`), the linear cross-correlation wraps circularly — lags from the negative end alias onto lags at the positive end. The resulting NCC vector has wrong values at every lag, the argmax picks the wrong shift, and `sbd = 1 - max(NCC)` is silently incorrect. This is the most common FFT cross-correlation bug and is guaranteed to produce wrong results on any pair where the optimal shift is nonzero.

The correct minimum FFT length for two series of length `n` is `2n - 1`. In practice, round up to the next power of two for efficiency:
```
fft_len = (2 * n - 1).next_power_of_two()  // minimum 2n-1, rounded up
```
The existing ACF code in `seasonal/mod.rs` uses `(2 * n).next_power_of_two()` — this is safe (slightly larger than the minimum, not smaller). Use the same pattern.

**Why it happens:**
Developers plan the FFT for the series length `n` (the natural "size of the data"). The cross-correlation of two length-`n` sequences has `2n - 1` meaningful lags; a length-`n` FFT discards half of them and wraps the rest. The rustfft API does not warn about this — it plans whatever length is requested.

**How to avoid:**
- Plan FFT and IFFT for `fft_len = (2 * n - 1).next_power_of_two()`.
- Zero-pad both `x_znorm` and `y_znorm` to `fft_len` before FFT (append zeros, not prepend).
- Validate with the shifted-copy test: `sbd(x, shift(x, k))` must achieve NCC = 1.0 (distance 0) at shift `k`, not at shift `k ± n`.
- Mirror the existing `seasonal/mod.rs` ACF pattern which is correct: `let fft_len = (2 * n).next_power_of_two();`.

**Warning signs:**
- `sbd(x, shift(x, k))` returns nonzero distance for any k ≠ 0.
- The optimal shift returned by `sbd` is always 0 or always some wrap-around value near `fft_len - n`.
- NCC vector has a noticeably asymmetric or multi-peaked shape for a simple shifted pair.

**Phase to address:** Phase 61 (SBD distance core). The FFT length is determined at the point the FFT plan is created; this is the foundational numerical primitive.

**Verification hook:**
```rust
// Shifted-copy test: SBD of a series with a shifted copy must be ~0.
let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0];
let shift_k = 3usize;
let mut y = x.clone();
y.rotate_right(shift_k);  // or build a time-shifted copy
let (dist, s) = sbd(&x, &y).unwrap();
assert!(dist < 1e-10, "sbd of shifted copy must be ~0, got {dist}");
assert_eq!(s, shift_k as isize, "returned shift must equal the applied shift");
```

---

### Pitfall 2: NCC Normalization by Count Instead of by Norms — SBD Is Wrong

**What goes wrong:**
The normalized cross-correlation (NCC) in SBD is defined as the **coefficient-normalized** cross-correlation — divided by `‖x‖ · ‖y‖` (where `x` and `y` are the already-z-normalized series). This makes NCC bounded in `[-1, 1]` and the SBD formula `dist = 1 - max(NCC)` bounded in `[0, 2]`.

Three incorrect alternatives that silently produce wrong distances:

1. **Biased count normalization:** divide each lag's cross-correlation sum by `n` (the biased estimator). This produces values with the same range as the unbiased version on average but is not bounded to `[-1, 1]` and differs from the Paparrizos & Gravano (2015) definition.
2. **Unbiased count normalization:** divide by `n - |lag|`. Values near the extremes of the lag range (where only a few overlapping elements exist) blow up, producing NCC > 1 at large lags.
3. **No normalization at all:** raw cross-correlation sums; distance is scale-dependent even after z-normalization because the L2 norms of different series are not exactly 1 post z-norm in finite-length arithmetic.

The correct formula for the FFT-based NCC:
```
cc_raw[s] = IFFT(FFT(x_znorm) · conj(FFT(y_znorm)))[s]   (with fft_len ≥ 2n−1)
NCC[s]    = cc_raw[s] / (‖x_znorm‖₂ · ‖y_znorm‖₂)
SBD(x,y)  = 1 − max_s NCC[s]
```

The denominator `‖x_znorm‖₂ · ‖y_znorm‖₂` is computed from the original (pre-FFT) z-normalized vectors, not from the FFT output.

**Why it happens:**
The biased and unbiased estimators are the standard statistical cross-correlation normalization; many signal-processing references use them. The coefficient normalization `‖x‖·‖y‖` is from the Pearson correlation definition applied to time-series shifts and is the specific form used by tslearn's `_normalized_cc`. Developers copy signal-processing references and miss this distinction.

**How to avoid:**
- Compute `norm_x = x_znorm.iter().map(|v| v*v).sum::<f64>().sqrt()` and `norm_y` analogously before FFT.
- Divide `cc_raw[s]` by `norm_x * norm_y` at every lag. Guard against the case where `norm_x * norm_y < 1e-10` (both series are constant — see Pitfall 3) by returning distance 1.0 in that case.
- Do not use `n`, `fft_len`, or any lag-dependent denominator.
- The IFFT output from rustfft is scaled by `fft_len` (rustfft does not divide by N on IFFT); divide the raw IFFT output by `fft_len` before applying the `‖x‖·‖y‖` normalization, or fold the `fft_len` scaling into the denominator.

**Warning signs:**
- NCC values outside `[-1, 1]` for any lag.
- `sbd(x, x)` (self-distance) is not exactly 0.0 (it must be — NCC at lag 0 for a series with itself equals 1.0).
- `sbd(x, y) ≠ sbd(y, x)` (asymmetry in the count-normalized variant due to the lag ordering).

**Phase to address:** Phase 61 (SBD distance core).

**Verification hook:**
```rust
// Self-distance must be exactly zero.
let x = vec![1.0, 3.0, 2.0, 5.0, 4.0, 1.0];
let (d_self, _) = sbd(&x, &x).unwrap();
assert!(d_self.abs() < 1e-10, "sbd(x,x) must be 0, got {d_self}");

// SBD symmetry: sbd(x,y) == sbd(y,x) up to floating point.
let y = vec![2.0, 1.0, 4.0, 3.0, 0.0, 5.0];
let (dxy, _) = sbd(&x, &y).unwrap();
let (dyx, _) = sbd(&y, &x).unwrap();
assert!((dxy - dyx).abs() < 1e-10, "sbd must be symmetric: {dxy} vs {dyx}");
```

---

### Pitfall 3: Z-Normalization Missing Before NCC — SBD Loses Shape Invariance

**What goes wrong:**
SBD requires both series to be **z-normalized before** computing the cross-correlation. Omitting the z-normalization step makes SBD sensitive to mean offset and amplitude — two series with the same shape but different offsets or scales return nonzero distance. The `1 - max(NCC)` formula is designed to measure shape similarity, which only works when the inputs have zero mean and unit-ish variance.

The z-normalization must be applied to each series independently before NCC, even if the FdMatrix rows were normalized at a prior preprocessing stage (which may have used a different normalization or been applied to different data).

Specifically: use `z_normalize_window` (from `src/shapelet/distance.rs`, shipped in v0.33.0) on each series, which produces population-std normalization with the constant-window guard (`std ≤ 1e-12` → zero vector). Do not reuse any series-level normalization that was applied to the whole FdMatrix.

**Why it happens:**
The `FdMatrix` used for clustering may already have been preprocessed. Developers assume "normalized input" means SBD can skip its own z-norm. In fact the SBD z-norm is an intrinsic part of the NCC computation, not a caller responsibility — both series must be independently z-normalized for each SBD call.

**How to avoid:**
- Inside `sbd(x: &[f64], y: &[f64])`, always call `z_normalize_window(x)` and `z_normalize_window(y)` at the top of the function. Never trust the caller to have done this.
- Reuse the v0.33.0 `shapelet::z_normalize_window` (or `z_normalize_into` for the allocation-free variant) — same convention, same `STD_EPS = 1e-12` constant-window guard.
- The constant-window guard returns the zero vector for a constant series; in that case the cross-correlation is all zeros, `‖x‖·‖y‖ = 0`, and `sbd` should return distance 1.0 and shift 0 (documented policy).

**Warning signs:**
- `sbd(x, x + constant)` returns nonzero for any nonzero constant.
- `sbd(x, scale * x)` returns nonzero for any scale ≠ 1.
- Clustering puts clearly similar-shaped but differently-offset curves in different clusters.

**Phase to address:** Phase 61 (SBD distance core).

**Verification hook:**
```rust
let x = vec![1.0, 2.0, 3.0, 2.0, 1.0];
// Offset invariance.
let x_offset: Vec<f64> = x.iter().map(|v| v + 100.0).collect();
let (d_off, _) = sbd(&x, &x_offset).unwrap();
assert!(d_off < 1e-10, "SBD must be offset-invariant: {d_off}");
// Scale invariance.
let x_scaled: Vec<f64> = x.iter().map(|v| v * 50.0).collect();
let (d_sc, _) = sbd(&x, &x_scaled).unwrap();
assert!(d_sc < 1e-10, "SBD must be scale-invariant: {d_sc}");
```

---

### Pitfall 4: Wrong Lag Extraction from FFT Output — fftshift Omission

**What goes wrong:**
The IFFT of `FFT(x) · conj(FFT(y))` produces the circular cross-correlation in standard FFT output ordering. Lag 0 is at index 0; positive lags `+s` are at indices `1, 2, ..., s`; negative lags `-s` are at indices `fft_len - s, ..., fft_len - 1`. Without applying `fftshift` (reordering to put lag 0 in the center), the `argmax` picks the wrong shift — specifically, a lag that corresponds to `fft_len - true_lag` rather than `true_lag` when the optimal shift is negative.

For SBD the convention is: the returned shift `s` is the integer by which series `y` should be cyclically shifted (or equivalently, `x` advanced by `-s`) to maximally align with `x`. This shift is in the range `-(n-1) .. +(n-1)`. Extracting it requires converting the raw argmax index back to a signed lag:
```
raw_argmax = argmax(NCC over 0..fft_len)
shift = if raw_argmax <= n-1 { raw_argmax as isize }
        else { raw_argmax as isize - fft_len as isize }
```
(Only the first `n` and last `n-1` positions of the IFFT output correspond to meaningful lags; the middle positions are zero-padded artifacts.)

**Why it happens:**
Developers compute `argmax(cc_raw)` without converting the raw FFT index to a signed lag. The bug is invisible when the optimal shift is 0 or when a small positive shift is the global optimum. It surfaces on data with large negative shifts or when the optimal alignment requires the second series to be left-shifted.

**How to avoid:**
- After IFFT, extract only the `2n - 1` meaningful lag positions: the first `n` elements (lags `0` to `n-1`) and the last `n-1` elements (lags `-(n-1)` to `-1`) of the `fft_len`-element IFFT output.
- Apply the argmax over only those `2n - 1` values (not over the full `fft_len` buffer) to avoid picking a zero-pad artifact.
- Convert the raw index to a signed shift using the formula above.
- Test with negative shifts: `sbd(shift(x, +3), x)` must return shift `-3` (not `fft_len - 3`).

**Warning signs:**
- `sbd(y_shifted_right, x)` returns a positive shift equal to `fft_len - k` rather than `-k`.
- Applying the returned shift to re-align two series makes them more misaligned, not less.
- Shift sign tests fail: expected `-3`, got `fft_len - 3`.

**Phase to address:** Phase 61 (SBD distance core).

**Verification hook:**
```rust
// Negative-shift test.
let x = vec![0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0];
let y = vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0, 0.0];  // x shifted right by 2
let (d, s) = sbd(&x, &y).unwrap();
assert!(d < 1e-6, "sbd of shifted copy must be near 0: {d}");
// Shift must be ±2, not fft_len - 2.
assert!(s.abs() <= 3, "shift must be small, not a wrap-around: {s}");
```

---

### Pitfall 5: Shape Extraction Centroid Is the Wrong Eigenvector (or No Eigenvector)

**What goes wrong:**
The k-Shape centroid is **not** the arithmetic mean of the aligned members. It is the **top eigenvector** of the matrix:
```
M = S^T · (I − (1/n) · 1·1^T) · S    (the "shape extraction" matrix)
```
where `S` is the `n × m` matrix of shift-aligned, z-normalized member curves (one row per cluster member). `M` is `m × m` symmetric positive semidefinite. The centroid is the eigenvector corresponding to the **largest eigenvalue** of `M`.

Common wrong implementations:

1. Using the arithmetic mean of `S` rows as the centroid (this is what k-means does; k-Shape does not).
2. Using the bottom eigenvector (nalgebra's `symmetric_eigen()` returns eigenvalues in **ascending** order — taking index 0 gives the smallest, not the largest).
3. Computing the SVD of `S` and taking the first right singular vector (this is the top eigenvector of `S^T S`, not of `S^T (I - 11^T/n) S` — the centering projection is missing).
4. Computing eigenvalues of `S · S^T` (the `n × n` Gram, not the `m × m` shape matrix) and using the wrong eigenvector.

**Why it happens:**
The Rayleigh-quotient / top-eigenvector formulation is stated in Paparrizos & Gravano (2015) Eq. 1–4 but the centering projection `(I - 11^T/n)` is easy to skip — it looks like it just centers the rows, which a developer may do as a preprocessing step before building `M = S^T S`. But `S^T (I - 11^T/n) S ≠ S^T S` unless the rows of `S` already sum to zero, which they do not in general even after z-normalization (z-normalization makes population std ≈ 1 and mean ≈ 0, but the mean of the z-normalized column-wise matrix `S` is not exactly zero unless the rows are already mean-subtracted within each time index).

**How to avoid:**
- Build `M` as `S^T · P_orth · S` where `P_orth = I - (1/n) · ones · ones^T` is the centering projection.
- Use `nalgebra::DMatrix::symmetric_eigen()` on `M`. Sort eigenvalues in **descending** order (negate the ascending order from nalgebra: index the last eigenvector, not the first). See `pace_fpca.rs:186-191` for the established ascending-to-descending sort pattern using `pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Equal))`.
- Take the eigenvector at the largest eigenvalue — column index corresponding to the largest eigenvalue after sorting.
- After extracting the eigenvector, re-z-normalize it (see Pitfall 6).

**Warning signs:**
- Centroids converge to the mean curve (arithmetic average), not a shape representative.
- Clusters do not improve after centroid update — the centroid step has no effect.
- `centroid.dot(mean_of_members)` is near zero (centroid is orthogonal to members), which happens when the bottom eigenvector is used.
- k-Shape applied to two obviously shifted-motif groups returns centroids that look like noise.

**Phase to address:** Phase 62 (k-Shape fit — shape extraction centroid).

**Verification hook:**
```rust
// Two-group shifted-motif recovery.
// Group 0: 5 copies of [0,1,2,1,0,...] at offset 0.
// Group 1: 5 copies of [0,0,1,2,1,...] (shifted right by 1).
// After fit with n_clusters=2, centroids must correlate > 0.95 with the group prototype.
let res = kshape_fd(&data, &KShapeConfig { n_clusters: 2, ..Default::default() }).unwrap();
// Centroid 0 shape must resemble group prototype 0 (up to sign — see Pitfall 6).
```

---

### Pitfall 6: Sign Ambiguity of the Centroid Eigenvector — Inverted Centroid

**What goes wrong:**
The top eigenvector of a symmetric matrix is unique up to sign: both `v` and `-v` are valid eigenvectors. When used as a k-Shape centroid, the wrong sign causes the centroid to be the **mirror image** of the cluster's representative shape. In the next iteration, member curves are SBD-aligned to an inverted centroid, the optimal shift is in the wrong direction, and the algorithm oscillates or converges to a degenerate solution.

The sign convention from the Paparrizos & Gravano paper and from tslearn's implementation: choose the sign of the eigenvector such that it correlates **positively** with the cluster members. Specifically, compute `dot(v, mean_of_S_rows)` — if it is negative, flip `v`.

**Why it happens:**
Nalgebra's `symmetric_eigen` returns eigenvectors with arbitrary sign. The existing `fix_svd_signs` / `dominant_sign_negative` infrastructure in `regression.rs` solves a similar problem for FPCA (make the dominant entry positive). But that convention is not the right one for k-Shape centroids — the dominant-entry rule may still produce a centroid that anti-correlates with most members if the dominant entry happens to be negative on the prototype curve.

**How to avoid:**
- After extracting the top eigenvector `v`, compute the sum `dot_sum = Σ_i dot(S_row_i, v)`.
- If `dot_sum < 0`, negate `v`: `v = -v`.
- This is the tslearn convention and is equivalent to "the centroid correlates positively with the cluster mean."
- Do **not** reuse `dominant_sign_negative` from `regression.rs` for this — it applies a different convention.

**Warning signs:**
- Centroid looks like the "upside-down" version of the expected shape.
- Inertia (sum of SBD to assigned centroid) is large even after convergence on a clean motif dataset.
- Adding a sign-flip test that checks `dot(centroid, prototype) > 0` fails on synthetic data.

**Phase to address:** Phase 62 (k-Shape fit — shape extraction centroid).

**Verification hook:**
```rust
// After shape extraction for a cluster whose members all look like [0,1,2,1,0]:
let centroid = shape_extract(&s_matrix).unwrap();
let member_mean: Vec<f64> = /* arithmetic mean of S rows */;
let dot: f64 = centroid.iter().zip(member_mean.iter()).map(|(a,b)| a*b).sum();
assert!(dot > 0.0, "centroid must correlate positively with member mean, dot={dot}");
```

---

### Pitfall 7: Members Not Shift-Aligned to the Current Centroid Before Shape Extraction

**What goes wrong:**
The shape extraction matrix `M = S^T (I - 11^T/n) S` requires `S` to contain the shift-aligned versions of each member curve. "Shift-aligned" means each member `x_i` has been cyclically shifted by the optimal shift `s_i = argmax_s NCC(centroid, x_i)` from the previous SBD call.

If the raw (unaligned) member curves are used to build `S`, the extracted centroid averages across time-misaligned curves and is meaningless — it captures phase noise, not shape. The algorithm will not converge.

This is a two-step per-member operation per iteration: (1) compute SBD to get the optimal shift `s_i`, (2) circularly shift the z-normalized member by `s_i` before inserting it as a row of `S`. The SBD step in the assignment phase gives the cluster label; the shift from that same SBD call must be stored and reused in the centroid update phase.

**Why it happens:**
The assignment step outputs `cluster[i]` (which cluster curve `i` belongs to). Developers use only the cluster label and discard the shift. The centroid update then uses the original FdMatrix rows, not the shifted versions.

**How to avoid:**
- `sbd` must return both the distance and the optimal shift: `fn sbd(x: &[f64], y: &[f64]) -> Result<(f64, isize), FdarError>`.
- During the assignment sweep, store `shifts[i] = s_i` alongside `cluster[i]`.
- In the centroid update, for cluster `c`, collect all `i` with `cluster[i] == c`, z-normalize `data.row(i)`, circularly shift by `shifts[i]`, and insert as row `i` of `S`.
- The circular shift is a cyclic rotation of the z-normalized row vector; it is not a padding operation.

**Warning signs:**
- After one iteration, centroids look like blurred or flat versions of the input curves (phase-averaging artifact).
- Inertia decreases by less than 1% per iteration from the start, even on a clean two-group dataset.
- Centroids are visually different from any individual member.

**Phase to address:** Phase 62 (k-Shape fit). The `sbd` signature in Phase 61 must expose the shift return value so Phase 62 can store it.

**Verification hook:**
- Run k-Shape on two clean groups of five identical shifted-motif curves each.
- After convergence, check: `centroid_0.corr(prototype_0) > 0.99` and `centroid_1.corr(prototype_1) > 0.99`.
- If alignment is broken, centroids will have much lower correlation with their prototypes.

---

### Pitfall 8: Centroid Not Re-Z-Normalized After Shape Extraction

**What goes wrong:**
After extracting the top eigenvector `v` as the new centroid, `v` has arbitrary L2 norm (it is an eigenvector, normalized to unit L2 by the eigendecomposition). When `v` is used as the reference in the next iteration's SBD calls, the NCC formula divides by `‖centroid‖ · ‖member‖`. If `‖centroid‖ ≠ 1` (which it will not be in general as a unit L2 eigenvector), the NCC denominator is wrong, and the distances are miscalibrated.

The paper and tslearn both z-normalize the extracted eigenvector before using it as the centroid in the next iteration. This is a separate z-normalization from the population-std normalization applied to the member curves — it is applied to the `m`-dimensional eigenvector itself.

**Why it happens:**
The eigenvector is already L2-normalized to unit length by nalgebra, so the implementer assumes it is "normalized enough." But SBD z-normalization is population-std normalization (mean-zero, std-one), not L2-unit-norm normalization. These are different: a unit-L2 vector has `‖v‖ = 1` but may have nonzero mean and `std = 1/√m`.

**How to avoid:**
- After extracting `v` as the top eigenvector, apply `z_normalize_window(&v)` to get the centroid for the next iteration.
- Store the centroid already-normalized in the `KShapeResult::centroids` field (same convention as `Shapelet.values` in the shapelet module — stored pre-normalized).

**Warning signs:**
- SBD distances to the centroid are systematically different from distances between two member curves of similar shape.
- Inertia is not monotonically non-increasing across iterations (renormalization error perturbs the objective).

**Phase to address:** Phase 62 (k-Shape fit — centroid update step).

**Verification hook:**
- After every centroid update, assert `mean(centroid).abs() < 1e-10` and `std(centroid) ≈ 1.0 ± 1e-6` before the next assignment step.

---

### Pitfall 9: Empty Cluster During k-Shape Iterations — Algorithm Panics or Produces Invalid State

**What goes wrong:**
At any iteration, a cluster may lose all its members if SBD-distances pull every curve to other centroids. This happens most often when `k > number_of_natural_clusters` (the overpartitioned case) or when a centroid is initialized too close to another. Without a recovery mechanism, the centroid update for the empty cluster will:
- Index into an empty slice → panic or undefined behavior.
- Produce a NaN centroid (eigendecomposition of a zero matrix).
- Leave the centroid unchanged (from the previous iteration), causing the cluster to stay empty forever.

**Why it happens:**
k-Shape shares this failure mode with k-means but has no established "absorb the nearest point" recovery documented in the original paper. tslearn handles it by re-initializing the empty cluster's centroid to a random member from the dataset. The existing `kernel_kmeans.rs` uses the "farthest point from its current cluster" strategy.

**How to avoid:**
- Mirror `kernel_kmeans.rs`'s `recover_empty_clusters` pattern: after each assignment step, identify empty clusters and reassign the point currently maximizing `sbd(point, assigned_centroid)` to the empty cluster.
- Alternatively (and simpler to implement correctly): re-initialize the empty cluster centroid to a randomly selected data point (using the per-restart seed with the `seed + restart_idx` convention).
- Guard the shape extraction: if `cluster_size[c] == 0`, skip the eigenvector computation for cluster `c` and log a warning or run the recovery.
- Never panic on an empty cluster — return `FdarError::ComputationFailed` only if recovery fails after `max_iter` attempts.

**Warning signs:**
- `thread 'test' panicked at 'called Option::unwrap() on a None value'` during centroid update.
- Final cluster assignments have some cluster label appearing zero times.
- Inertia is `NaN` or `Inf` after one iteration.

**Phase to address:** Phase 62 (k-Shape fit).

**Verification hook:**
```rust
// k > natural clusters test (same as kernel_kmeans smoke test).
let k = 4;  // but data has only 2 natural groups
let res = kshape_fd(&data, &KShapeConfig { n_clusters: k, ..Default::default() }).unwrap();
// Every cluster label in 0..k appears at least once.
let mut sizes = vec![0usize; k];
for &c in &res.cluster { sizes[c] += 1; }
assert!(sizes.iter().all(|&s| s >= 1), "empty cluster survived: {sizes:?}");
```

---

### Pitfall 10: Objective Non-Monotonicity — k-Shape Inertia Increases Mid-Run

**What goes wrong:**
k-Shape's objective (sum of SBD distances from each curve to its assigned centroid's SBD) is guaranteed to be non-increasing at the assignment step. The centroid update step, however, does not guarantee a decrease — specifically, if the centroid sign is flipped incorrectly (Pitfall 6), the alignment step in the next assignment uses an inverted centroid, producing larger SBD distances than before the update. The result is oscillation: the algorithm does not converge and `converged = false` is the permanent status.

Also: if the `n_init` restarts are compared by total inertia but inertia is computed on a different (e.g., unaligned) basis across restarts, the best-restart selection is meaningless.

**How to avoid:**
- Track inertia using SBD-to-assigned-centroid, computed **after** the assignment step with the current centroids (not before updating centroids). This is the standard k-means convergence check.
- Convergence criterion: `|prev_inertia - inertia| / prev_inertia < tol` OR labels unchanged from the previous iteration (same `tol = 1e-4` and `max_iter = 300` defaults as `KernelKmeansConfig`).
- Use a signed-label-based convergence check: if `new_cluster == old_cluster`, converge regardless of inertia change.
- If inertia increases more than `tol` for 3+ consecutive iterations, this indicates a sign bug (Pitfall 6) — add a debug assertion.

**Warning signs:**
- Inertia oscillates between two values and never decreases monotonically.
- `converged = false` for all restarts even on clean synthetic data.
- Best restart selected by `n_init` comparison has higher inertia than some other restart.

**Phase to address:** Phase 62 (k-Shape fit — convergence tracking).

**Verification hook:**
- On a clean two-group dataset with `n_init=1`, assert that inertia is strictly non-increasing across iterations (allow ≤ 1e-10 floating-point tolerance).

---

### Pitfall 11: k-Shape n_init Restarts with Wrong Seeding — Non-Determinism or Identical Restarts

**What goes wrong:**
Two failure modes in the `n_init` restart loop:

1. **Identical restarts:** if all restarts use the same seed, they produce identical initializations and identical results. The multi-start provides no benefit.
2. **Non-deterministic restarts:** if the seed is not threaded and the RNG is initialized from `rand::thread_rng()`, two calls with the same config return different results, breaking reproducibility.

The correct pattern (from `kernel_kmeans.rs`): restart `r` is seeded `StdRng::seed_from_u64(config.seed.wrapping_add(r as u64))`. This produces `n_init` distinct random initializations while remaining fully reproducible.

**Why it happens:**
Developers copy the `seed + r` pattern but use `seed + r` as an addition that can overflow for large seeds (fixed by `wrapping_add`), or they forget to thread `restart_idx` and use a constant seed.

**How to avoid:**
- `KShapeConfig` must include `seed: u64` and `n_init: usize` (both required for the restart loop).
- Use `StdRng::seed_from_u64(config.seed.wrapping_add(restart as u64))` for restart `r`. Mirror the `kernel_kmeans.rs` convention verbatim.
- Initialization: assign each curve to a random cluster `rng.gen_range(0..k)`, then run `ensure_no_empty_random` (same helper from `kernel_kmeans.rs`) to guarantee no cluster starts empty.

**Warning signs:**
- Two calls with the same config return different `cluster` assignments.
- `n_init=10` produces the same inertia as `n_init=1` (all restarts are identical).
- `assert_eq!(a.cluster, b.cluster)` in the determinism test fails.

**Phase to address:** Phase 62 (k-Shape fit).

**Verification hook:**
```rust
let cfg = KShapeConfig { seed: 42, n_init: 5, ..Default::default() };
let a = kshape_fd(&data, &cfg).unwrap();
let b = kshape_fd(&data, &cfg).unwrap();
assert_eq!(a.cluster, b.cluster, "same seed must give identical labels");
assert_eq!(a.inertia.to_bits(), b.inertia.to_bits());
```

---

### Pitfall 12: IFFT Scale Factor Not Removed — NCC Values Off by fft_len

**What goes wrong:**
rustfft performs unnormalized IFFT: the output of `plan_fft_inverse(fft_len).process(buf)` is scaled by `fft_len`. The raw cross-correlation values in `buf` after IFFT are each `fft_len` times the true cross-correlation. If this scale factor is not divided out before computing the coefficient-normalized NCC (or if it is folded into the `‖x‖·‖y‖` denominator incorrectly), the NCC values are off by `fft_len` and the argmax picks the right index but the distance value `1 - max(NCC)` is meaningless.

Concretely: if `fft_len = 32` and the true NCC max is `0.9`, the raw IFFT output gives `28.8` at that lag, and `1 - 28.8 / (‖x‖·‖y‖)` is a large negative number — which clips the returned SBD distance to a nonsensical value.

**Why it happens:**
The existing seasonal ACF code (`seasonal/mod.rs:374`) divides by `fft_len * n * var` — folding the IFFT scale into the normalization. For NCC the analogous step is dividing by `fft_len` (from the IFFT) and then by `‖x‖·‖y‖` (the coefficient normalization). Developers new to rustfft are aware of the FFT `1/n` convention in MATLAB/Python's `numpy.fft.ifft` and forget that rustfft does not include it.

**How to avoid:**
- After `ifft.process(&mut buf)`, divide the `buf[s].re` values by `fft_len as f64` before applying the `‖x‖·‖y‖` normalization. Alternatively, fold the `fft_len` into the denominator: `ncc[s] = buf[s].re / (fft_len as f64 * norm_x * norm_y)`.
- Add a `sbd(x, x)` self-distance test as the first smoke test — it will fail obviously if the IFFT scale is wrong (NCC will be `n` instead of `1.0`, and `1 - n = large negative`).

**Warning signs:**
- `sbd(x, x)` returns a large negative number or a number much greater than 1.
- NCC values at any lag exceed 1.0 or are much less than -1.0.

**Phase to address:** Phase 61 (SBD distance core).

**Verification hook:**
- Self-distance test (same as Pitfall 2 hook): `sbd(x, x)` must be `< 1e-10`.
- NCC bounds test: for any pair `(x, y)`, `max(NCC) <= 1.0 + 1e-10` and `min(NCC) >= -1.0 - 1e-10`.

---

### Pitfall 13: SBD-Based k-Medoids Feeds Wrong Distance — SBD Is Not the Euclidean L2 Distance

**What goes wrong:**
The existing `kmedoids_from_distances` in `alignment/clustering.rs` takes a precomputed `FdMatrix` distance matrix and runs PAM-style k-medoids. Reusing this for SBD-based k-medoids is the correct architectural decision — but only if the SBD pairwise distance matrix is computed correctly (using FFT-NCC, not Euclidean or DTW distances).

The failure mode: a developer computes the "distance matrix" using `distance.rs`'s `pairwise_distance_matrix` with an L2 metric and feeds it to `kmedoids_from_distances` instead of computing the `n × n` SBD pairwise matrix. The medoids are then L2-medoids, not SBD-medoids.

**Why it happens:**
`pairwise_distance_matrix` is already in `distance.rs` and is easy to call. The developer may not realize that a new `sbd_distance_matrix` function is needed that uses the FFT-NCC path.

**How to avoid:**
- Implement `sbd_distance_matrix(data: &FdMatrix) -> Result<FdMatrix, FdarError>` as a wrapper that calls `sbd(row_i, row_j)` for all pairs and assembles the symmetric matrix.
- Pass this matrix — not any L2/DTW matrix — to `kmedoids_from_distances`.
- The doctest for SBD k-medoids must show the `sbd_distance_matrix` call explicitly so there is no ambiguity.

**Warning signs:**
- SBD k-medoids and L2 k-medoids produce identical results on data where SBD should differ (e.g., curves with same shape but different offsets).

**Phase to address:** Phase 63 (SBD-based k-medoids integration).

**Verification hook:**
- Offset-invariance test: run SBD k-medoids on two groups of curves where group membership is shape-based (not amplitude-based). Purity must be > 0.9. L2 k-medoids on the same data should have purity < 0.7 (offset differences confuse L2 but not SBD).

---

### Pitfall 14: Predict Uses Re-Estimated Normalization — Not Stored Centroids

**What goes wrong:**
Out-of-sample predict assigns each new curve to the fitted centroid minimizing SBD. The failure mode: predict re-z-normalizes or re-estimates the centroids from the test data (analogous to Shapelet Pitfall 9 for shapelets). The centroids must be used exactly as stored in `KShapeResult` — already z-normalized, no re-estimation.

Additionally, the SBD in predict must use the stored centroids as `y` in `sbd(new_curve_znorm, centroid_stored)`. If the centroid is passed as `x` instead of `y`, the argmax shift has the wrong sign (the NCC is asymmetric in lag direction), though the distance value is correct. For predict, only the distance (not the shift) is needed, so the asymmetry does not matter for correctness of labels — but using `y = centroid` as the reference is the tslearn convention and should be followed for consistency.

**Why it happens:**
Same structural error as Pitfall 9 (shapelet transform consistency): the predict path independently "normalizes the centroid" rather than trusting the stored pre-normalized version.

**How to avoid:**
- `KShapeResult::centroids` stores already-z-normalized centroid vectors (type `Vec<Vec<f64>>`).
- `predict` iterates over centroids: `sbd(z_normalize_window(&new_row), &centroids[c])` for each `c`, assigns to argmin.
- Never call `z_normalize_window` on a stored centroid inside predict.
- Mirror `KernelKmeansResult::predict` design: stored state reused, no re-estimation.

**Warning signs:**
- `predict(train_data)` returns different labels than the training-time `cluster` field.
- `predict` on an exact copy of a training curve returns a different cluster than the fit result.

**Phase to address:** Phase 62 (k-Shape fit — predict method on `KShapeResult`).

**Verification hook:**
```rust
// Exact-copy predict consistency.
let res = kshape_fd(&data, &cfg).unwrap();
let preds = res.predict(&data).unwrap();
assert_eq!(preds, res.cluster, "predict on train data must reproduce training labels");
```

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Plan FFT for length `n` (not `2n−1`) | One less calculation | Circular wrap corrupts all shifts (Pitfall 1) — silently wrong | Never |
| Use count normalization (divide by `n` or `n−|lag|`) instead of `‖x‖·‖y‖` | Simpler code | NCC not in [-1,1]; SBD formula wrong (Pitfall 2) | Never |
| Use arithmetic mean as k-Shape centroid | Avoids eigendecomposition | Wrong algorithm — k-Shape degenerates to k-means on z-normalized data (Pitfall 5) | Never |
| Skip the centering projection `(I - 11^T/n)` | Simpler matrix product | Wrong eigenvector; centroid does not capture shape (Pitfall 5) | Never |
| Ignore eigenvector sign | No extra computation | Centroid may be inverted; algorithm oscillates (Pitfall 6) | Never |
| Discard shift from SBD at assignment time | Half the state to track | Centroid update uses unaligned members (Pitfall 7) | Never |
| Skip IFFT scale division | Simpler formula | NCC values off by `fft_len`; SBD is wrong (Pitfall 12) | Never |
| Hard-code `n_init = 1` | Faster, simpler | k-Shape is highly sensitive to init; one restart often converges to a local minimum | Only in dev smoke tests, never shipped |
| Re-use dominant_sign_negative from regression.rs for centroid sign | Less code | Wrong sign convention for k-Shape; may invert centroid (Pitfall 6) | Never — implement the correlation-based sign rule |

---

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Allocating a new FFT plan per SBD call | Planner construction is ~100× more expensive than one FFT | Build `FftPlanner` + plans once; share via `Arc<dyn Fft>` across the pairwise loop | n > 20 curves |
| Building `S` matrix by cloning FdMatrix rows without reuse | O(n·m) alloc per iteration × max_iter | Reuse a pre-allocated `n_cluster × m` buffer; only recompute the rows of the affected cluster | Large n, m |
| Computing the full `n × n` SBD pairwise matrix for every k-Shape iteration | O(n²) SBD calls per iteration (each O(m log m)) | k-Shape does **not** need the full pairwise matrix — compute `sbd(curve_i, centroid_c)` per assignment sweep, which is O(n·k) SBD calls | n > 50, k > 3 |
| `to_dmatrix()` inside the SBD hot loop | nalgebra matrix copy for every SBD pair | SBD distance is pure `&[f64]` arithmetic (FFT of slices); never call `to_dmatrix` in the SBD path | Every SBD call |
| Sequential pairwise SBD distance matrix | O(n²) sequential FFTs | Use `iter_maybe_parallel!` at the pair level for the pairwise matrix function; each SBD call is independently parallelizable | n > 100 for distance matrix |
| Allocating a new `Vec<Complex<f64>>` scratch buffer per SBD call | O(fft_len) alloc per pair, millions of times in clustering | Reuse a pre-allocated scratch buffer across SBD calls (pass as `&mut Vec<Complex<f64>>`) | n > 20 curves, tight loops |

---

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| `shapelet::z_normalize_window` (v0.33.0) | Assume shapelet z-norm convention matches SBD z-norm | Both use population std (`ddof=0`) and `STD_EPS=1e-12` — reuse `z_normalize_window` directly; document the shared convention |
| `alignment/clustering.rs` `kmedoids_from_distances` | Feed an L2 pairwise matrix instead of an SBD matrix | Implement `sbd_distance_matrix` using the SBD core and pass its output; the k-medoids PAM engine is distance-agnostic |
| `FdMatrix` column-major row access | Iterate rows with `data[(i,j)]` inside the FFT loop — scattered memory access | Use `data.row(i)` to copy the row to a contiguous `Vec<f64>` before FFT; same pattern as `fourier.rs` |
| `parallel.rs` `iter_maybe_parallel!` | Apply to the inner FFT buffer computation (inside one SBD call) | Parallelize at the outer level: across pairs in `sbd_distance_matrix`, or across curves in the assignment sweep; the FFT inner loop is sequential |
| `rustfft` IFFT scale | Expect IFFT to divide by `fft_len` (numpy convention) | rustfft IFFT does not divide — divide manually by `fft_len` or fold into the NCC denominator (see Pitfall 12) |
| nalgebra `symmetric_eigen` eigenvalue order | Assume descending order (largest first) | nalgebra returns ascending order — take the **last** sorted column, not the first; mirror the `pace_fpca.rs:191` descending-sort pattern |
| `dominant_sign_negative` in `regression.rs` | Reuse for centroid sign fix | Wrong convention for k-Shape — implement a separate correlation-based sign rule: `if dot(v, mean_S_rows) < 0 { v = -v }` |

---

## "Looks Done But Isn't" Checklist

- [ ] **FFT zero-padding:** FFT planned for `(2*n-1).next_power_of_two()`, not `n`. Shifted-copy test passes (Pitfall 1).
- [ ] **NCC normalization:** coefficient-normalized by `‖x_znorm‖·‖y_znorm‖`, not by count. NCC bounded in `[-1,1]`. Self-distance test: `sbd(x,x) ≈ 0` (Pitfall 2).
- [ ] **SBD z-normalization:** both series z-normalized inside `sbd()` unconditionally. Offset and scale invariance tests pass (Pitfall 3).
- [ ] **Lag sign extraction:** argmax converted to signed lag using `if idx <= n-1 { idx } else { idx - fft_len }`. Negative-shift test passes (Pitfall 4).
- [ ] **IFFT scale division:** raw IFFT output divided by `fft_len` before NCC computation. NCC values in `[-1,1]` (Pitfall 12).
- [ ] **Shape extraction eigenvector:** top eigenvector of `S^T (I - 11^T/n) S`, not arithmetic mean, not `S^T S`. Centroid differs from member mean. Two-group recovery test passes (Pitfall 5).
- [ ] **Eigenvalue order:** nalgebra ascending eigenvalue order reversed; centroid taken from last (largest) eigenvalue column. (Pitfall 5).
- [ ] **Centroid sign fix:** `dot(centroid, mean_S_rows) > 0` enforced. Sign-flip test passes (Pitfall 6).
- [ ] **Shift-alignment before shape extraction:** `shifts[i]` stored from SBD assignment; rows of `S` are circularly shifted z-normalized curves. Centroid correlation with prototype > 0.99 (Pitfall 7).
- [ ] **Centroid re-z-normalized:** after eigenvector extraction, `z_normalize_window(centroid)` applied. Mean ≈ 0, std ≈ 1 assertion passes (Pitfall 8).
- [ ] **Empty cluster recovery:** clusters that lose all members are reseeded. k > natural_clusters test does not panic and all cluster sizes ≥ 1 (Pitfall 9).
- [ ] **Inertia non-increasing:** inertia check per iteration; convergence on label stability or tol. Two-group dataset converges in < 20 iterations (Pitfall 10).
- [ ] **Determinism with n_init:** `seed.wrapping_add(restart)` seeding. Determinism test: two same-seed calls return identical labels and inertia (Pitfall 11).
- [ ] **SBD k-medoids uses correct distance:** `sbd_distance_matrix` (FFT-NCC) fed to `kmedoids_from_distances`, not L2. Offset-invariance purity test passes (Pitfall 13).
- [ ] **Predict consistency:** `predict(train_data) == res.cluster`. Stored centroids not re-normalized. Exact-copy test passes (Pitfall 14).
- [ ] **No `to_dmatrix` in hot loop:** SBD is pure `&[f64]` slice arithmetic. No nalgebra conversion in `sbd()`.
- [ ] **FFT planner reuse:** `FftPlanner` constructed once per `kshape_fd` call, not per SBD pair.

---

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Wrong FFT length (circular wrap) | HIGH — all distances wrong | Change `fft_len` to `(2*n-1).next_power_of_two()`; rerun all SBD-dependent tests; one-line fix but all cached results are invalid |
| Wrong NCC normalization | HIGH — all distances silently miscalibrated | Replace denominator with `fft_len * norm_x * norm_y`; one-line fix; rerun all SBD tests |
| Missing z-normalization in `sbd()` | MEDIUM — wrong distances for non-centered data | Add `z_normalize_window` calls at top of `sbd()`; no API change |
| Wrong lag sign extraction | MEDIUM — wrong shift on negative lags, correct distance | Fix the signed-lag conversion after argmax; one-line change; shift tests catch it |
| Arithmetic mean instead of eigenvector | HIGH — wrong algorithm entirely | Rewrite shape extraction; no API change; results change completely |
| Wrong eigenvector (e.g., bottom instead of top) | HIGH — wrong centroids | Fix the eigenvalue sort / column index; rerun recovery tests |
| Missing sign fix | MEDIUM — oscillation / inverted centroids | Add `dot_sum < 0` flip; convergence tests catch it |
| Missing shift-alignment | HIGH — centroid update is meaningless | Modify `sbd` to return shift; store `shifts[i]` in assignment loop; rewrite centroid update |
| Missing centroid re-z-norm | LOW — calibration drift over iterations | Add `z_normalize_window` at end of centroid update; one-line fix |
| No empty-cluster recovery | MEDIUM — panic or NaN in degenerate cases | Add recovery logic mirroring `kernel_kmeans.rs`; no API change |
| IFFT scale not divided out | HIGH — NCC off by fft_len | Add `/fft_len as f64` after IFFT; one-line fix; self-distance test catches it immediately |

---

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| P1: FFT zero-padding (circular wrap) | Phase 61 — SBD distance core | Shifted-copy test: `sbd(x, shift(x,k)) ≈ 0` at correct shift |
| P2: NCC coefficient normalization | Phase 61 — SBD distance core | Self-distance test: `sbd(x,x) ≈ 0`; symmetry test: `sbd(x,y) == sbd(y,x)` |
| P3: Z-normalization required in SBD | Phase 61 — SBD distance core | Offset and scale invariance tests |
| P4: Lag sign extraction (fftshift) | Phase 61 — SBD distance core | Negative-shift test: expected shift sign matches applied shift |
| P5: Wrong centroid (mean vs eigenvector) | Phase 62 — k-Shape fit, centroid update | Two-group shifted-motif recovery: centroid correlation > 0.99 with prototype |
| P6: Eigenvector sign ambiguity | Phase 62 — k-Shape fit, centroid update | Sign-flip test: `dot(centroid, mean_S_rows) > 0` after every extraction |
| P7: Shift-alignment before shape extraction | Phase 62 — k-Shape fit, centroid update | Centroid-member correlation test; inertia monotone-decrease check |
| P8: Centroid re-z-normalization | Phase 62 — k-Shape fit, centroid update | Mean/std assertion after each centroid update step |
| P9: Empty cluster recovery | Phase 62 — k-Shape fit | k > natural_clusters test; no panic; all sizes ≥ 1 |
| P10: Objective monotonicity | Phase 62 — k-Shape fit, convergence | Inertia-per-iteration monotone-decrease check on two-group synthetic data |
| P11: Determinism with n_init | Phase 62 — k-Shape fit | Two same-seed calls produce identical labels and inertia bits |
| P12: IFFT scale factor | Phase 61 — SBD distance core | Self-distance and NCC-bounds tests |
| P13: SBD k-medoids distance source | Phase 63 — SBD k-medoids | Offset-invariance purity test vs L2 k-medoids |
| P14: Predict uses stored centroids | Phase 62 — k-Shape fit, predict | `predict(train) == res.cluster`; exact-copy predict test |

---

## Sources

- Paparrizos, J., Gravano, L. (2015). k-Shape: Efficient and Accurate Clustering of Time Series. *SIGMOD '15*. doi:10.1145/2723372.2737793. — Defines SBD (Eq. 1–4), shape extraction (Eq. 5–8), k-Shape algorithm. Primary correctness reference.
- tslearn source: `tslearn/clustering/kshape.py` (`_normalized_cc`, `_sbd`, `_extract_shape`, `KShape.fit`) — Reference implementation for NCC formula, IFFT scale, sign fix, shift-alignment before shape extraction, centroid re-z-normalization.
- rustfft documentation: `rustfft.rs-lang.github.io` — FFT/IFFT unnormalized convention; `FftPlanner` reuse pattern; `plan_fft_forward` / `plan_fft_inverse`.
- fdars-core `src/seasonal/mod.rs:350` — Established `(2 * n).next_power_of_two()` zero-padding pattern for FFT-based ACF in this codebase; safe template for SBD.
- fdars-core `src/kernel_kmeans.rs` — Empty-cluster recovery, n_init seeding (`seed.wrapping_add(restart)`), convergence check pattern, predict design, KernelKmeansConfig defaults.
- fdars-core `src/regression.rs:197-244` — `dominant_sign_negative` / `fix_svd_signs` eigenvector sign convention (different from k-Shape sign rule — do not reuse).
- fdars-core `src/pace_fpca.rs:186-191` — nalgebra ascending eigenvalue order pattern; descending sort for top eigenvector extraction.
- fdars-core `src/shapelet/distance.rs` — `z_normalize_window` / `z_normalize_into` with `STD_EPS = 1e-12`; reusable directly for SBD z-normalization.
- fdars-core `src/alignment/clustering.rs` — `kmedoids_from_distances` interface; `KMedoidsConfig` seeding convention; k-medoids integration scaffolding.
- fdars-core `src/metric/fourier.rs` — `FftPlanner` reuse pattern across rows; established convention for sharing plans via `plan_fft_forward(m)` + `fft.as_ref()`.

---
*Pitfalls research for: k-Shape clustering + SBD via FFT cross-correlation (fdars-core v0.34.0, GAP-03)*
*Researched: 2026-09-02*
