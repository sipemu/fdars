# Elastic Alignment & Shape Analysis -- References

This document maps each public function in the `alignment` and `elastic_*` modules to the paper(s) that describe the underlying method.

## SRSF Transform & Core DP Alignment

| Function | Method | Reference |
|----------|--------|-----------|
| `srsf_transform` | Square-Root Slope Function: q(t) = sign(f') sqrt(\|f'\|) | Srivastava et al. (2011), S2 |
| `srsf_inverse` | Reconstruct curve from SRSF | Srivastava et al. (2011) |
| `reparameterize_curve` | Apply warping function f(gamma(t)) | Srivastava et al. (2011) |
| `dp_alignment_core` | Dynamic programming alignment with coprime neighborhood | Srivastava et al. (2011), S3; Robinson (2012) |
| `dp_lambda_penalty` | Roughness penalty lambda (slope-1)^2 dt | Tucker et al. (2013), S2.2 |

## Pairwise Alignment & Distance Matrices

| Function | Method | Reference |
|----------|--------|-----------|
| `elastic_align_pair` | Fisher-Rao pairwise elastic alignment | Srivastava et al. (2011), Theorem 3.1 |
| `elastic_distance` | Elastic distance after optimal alignment | Srivastava et al. (2011) |
| `amplitude_distance` | Amplitude component of elastic distance | Srivastava & Klassen (2016), S4.3 |
| `phase_distance_pair` | Phase distance (geodesic distance of warp from identity) | Srivastava & Klassen (2016), S4.3 |
| `elastic_self_distance_matrix` | O(n^2) pairwise elastic distance matrix | -- |
| `elastic_cross_distance_matrix` | n1 x n2 cross-distance matrix | -- |
| `amplitude_self_distance_matrix` | Pairwise amplitude distances | -- |
| `phase_self_distance_matrix` | Pairwise phase distances | -- |

## Karcher Mean

| Function | Method | Reference |
|----------|--------|-----------|
| `karcher_mean` | Karcher / Frechet mean on elastic manifold | Srivastava et al. (2011), S5; Tucker et al. (2013), S2.3 |
| `sqrt_mean_inverse` | Karcher mean of warps on Hilbert sphere | Srivastava & Klassen (2016), S8.3 |

## N-Dimensional Alignment

| Function | Method | Reference |
|----------|--------|-----------|
| `elastic_align_pair_nd` | Elastic alignment for R^d-valued curves | Srivastava & Klassen (2016), Ch. 12 |
| `elastic_distance_nd` | Elastic distance for multidimensional curves | Srivastava & Klassen (2016) |
| `srsf_transform_nd` | SRVF for R^d curves: q(t) = f'(t) / sqrt(\|f'(t)\|) | Srivastava & Klassen (2016), S12.2 |
| `karcher_mean_nd` | Karcher mean for R^d curves | Srivastava & Klassen (2016), S12.4 |
| `karcher_covariance_nd` | Covariance estimation for R^d curves | Srivastava & Klassen (2016) |
| `pca_nd` | PCA for multidimensional aligned curves | Srivastava & Klassen (2016) |

## Bayesian Alignment

| Function | Method | Reference |
|----------|--------|-----------|
| `bayesian_align_pair` | Preconditioned Crank-Nicolson MCMC on Hilbert sphere | Cheng et al. (2016); Lu et al. (2017) |

## Closed Curve Alignment

| Function | Method | Reference |
|----------|--------|-----------|
| `elastic_align_pair_closed` | Closed curve alignment with rotation search | Srivastava & Klassen (2016), Ch. 11 |
| `elastic_distance_closed` | Elastic distance for periodic curves | Srivastava & Klassen (2016) |
| `karcher_mean_closed` | Karcher mean for closed curves | Srivastava & Klassen (2016), S11.4 |

## Robust Karcher Estimation

| Function | Method | Reference |
|----------|--------|-----------|
| `karcher_median` | Geometric median via Weiszfeld algorithm on elastic manifold | Weiszfeld (1937); Fletcher et al. (2009) |
| `robust_karcher_mean` | Trimmed Karcher mean (remove most-distant curves) | Fraiman & Muniz (2001) |

## Outlier Detection

| Function | Method | Reference |
|----------|--------|-----------|
| `elastic_outlier_detection` | SRVF outlier detection with Tukey fence (Q3 + 1.5 IQR) | Tukey (1977); Srivastava & Klassen (2016) |

## Geodesic Paths

| Function | Method | Reference |
|----------|--------|-----------|
| `curve_geodesic` | Geodesic interpolation in SRSF space + Hilbert sphere | Srivastava & Klassen (2016), S4.5 |
| `curve_geodesic_nd` | Geodesic paths for R^d curves | Srivastava & Klassen (2016), Ch. 12 |

## Partial Matching

| Function | Method | Reference |
|----------|--------|-----------|
| `elastic_partial_match` | Elastic partial matching via sliding variable-length window | Srivastava & Klassen (2016), S4.7 |

## Multi-Resolution Alignment

| Function | Method | Reference |
|----------|--------|-----------|
| `elastic_align_pair_multires` | Coarse DP on subsampled grid + gradient refinement | Robinson (2012) |

## Shape Confidence Intervals

| Function | Method | Reference |
|----------|--------|-----------|
| `shape_confidence_interval` | Bootstrap confidence bands for Karcher mean | Kurtek et al. (2012) |

## Transfer Alignment

| Function | Method | Reference |
|----------|--------|-----------|
| `transfer_alignment` | Cross-population alignment via bridging warps | Kurtek & Srivastava (2015) |

## Diagnostics

| Function | Method | Reference |
|----------|--------|-----------|
| `diagnose_alignment` | Warp complexity, smoothness, non-monotonicity assessment | -- |

## Elastic Clustering

| Function | Method | Reference |
|----------|--------|-----------|
| `elastic_kmeans` | k-means++ with Fisher-Rao distances and Karcher centroids | Srivastava & Klassen (2016), S8.5; Arthur & Vassilvitskii (2007) |
| `elastic_hierarchical` | Hierarchical clustering (single/complete/average linkage) | -- |
| `cut_dendrogram` | Cut dendrogram at k clusters | -- |

## Shape Analysis (Quotient Spaces)

| Function | Method | Reference |
|----------|--------|-----------|
| `orbit_representative` | Quotient space representative (reparameterization / translation / scale) | Srivastava & Klassen (2016), Ch. 9 |
| `shape_distance` | Distance in quotient shape space | Srivastava & Klassen (2016), S9.3 |
| `shape_mean` | Karcher mean in shape space | Srivastava & Klassen (2016), S9.4 |
| `shape_self_distance_matrix` | Pairwise shape distance matrix | -- |

## Peak Persistence

| Function | Method | Reference |
|----------|--------|-----------|
| `peak_persistence` | Topology-based lambda selection via peak birth/death | Edelsbrunner et al. (2002) (concept); adapted for elastic alignment |

## Lambda Cross-Validation

| Function | Method | Reference |
|----------|--------|-----------|
| `lambda_cv` | K-fold / LOO CV for alignment penalty lambda | Tucker et al. (2013), S2.4 |

## Elastic Depth

| Function | Method | Reference |
|----------|--------|-----------|
| `elastic_depth` | Amplitude + phase decomposed functional depth | Srivastava & Klassen (2016); Lopez-Pintado & Romo (2009) |

## Horizontal FPNS

| Function | Method | Reference |
|----------|--------|-----------|
| `horiz_fpns` | Functional Principal Nested Spheres for warps | Jung et al. (2012) |

## Warping Statistics

| Function | Method | Reference |
|----------|--------|-----------|
| `warp_statistics` | Pointwise mean, variance, confidence bands for warps | Srivastava & Klassen (2016), S8.3 |

## Phase Box Plots

| Function | Method | Reference |
|----------|--------|-----------|
| `phase_boxplot` | Modified band depth box plot for warps | Lopez-Pintado & Romo (2009) |

## TSRVF (Transported SRVF)

| Function | Method | Reference |
|----------|--------|-----------|
| `tsrvf_transform` | Transport to tangent space at Karcher mean | Su et al. (2014) |
| `tsrvf_transform_with_method` | LogMap, Schild's ladder, or pole ladder transport | Su et al. (2014); Lorenzi & Pennec (2014) |
| `parallel_transport_schilds` | Schild's ladder parallel transport | Ehlers et al. (2011) |
| `parallel_transport_pole` | Pole ladder parallel transport | Lorenzi & Pennec (2014) |

## Generative Models

| Function | Method | Reference |
|----------|--------|-----------|
| `gauss_model` | Independent Gaussian sampling from amplitude/phase FPCAs | Tucker et al. (2013), S3 |
| `joint_gauss_model` | Joint Gaussian sampling preserving amplitude-phase correlation | Tucker et al. (2013), S3.2 |

## Landmark-Constrained Alignment

| Function | Method | Reference |
|----------|--------|-----------|
| `elastic_align_pair_constrained` | DP alignment with landmark waypoints | Srivastava & Klassen (2016), S4.6 |

## Set-Level Operations

| Function | Method | Reference |
|----------|--------|-----------|
| `align_to_target` | Align all curves to a given target | Srivastava et al. (2011) |
| `elastic_decomposition` | Amplitude / phase distance decomposition | Srivastava & Klassen (2016), S4.3 |

---

## Elastic Models

### Elastic FPCA

| Function | Method | Reference |
|----------|--------|-----------|
| `vert_fpca` | Vertical (amplitude) FPCA in augmented SRSF space | Tucker et al. (2013), S2.5 |
| `horiz_fpca` | Horizontal (phase) FPCA via shooting vectors on Hilbert sphere | Tucker et al. (2013), S2.6 |
| `joint_fpca` | Joint amplitude-phase FPCA on concatenated scores | Tucker et al. (2013), S2.7 |

### Elastic Regression

| Function | Method | Reference |
|----------|--------|-----------|
| `elastic_regression` | Scalar-on-function regression with alignment: y = integral(q beta) dt | Tucker et al. (2014) |
| `predict_elastic_regression` | Prediction from elastic regression | Tucker et al. (2014) |

### Elastic PCR

| Function | Method | Reference |
|----------|--------|-----------|
| `elastic_pcr` | Principal component regression on elastic FPCA scores | Tucker et al. (2013); Tucker et al. (2014) |

### Elastic Logistic Regression

| Function | Method | Reference |
|----------|--------|-----------|
| `elastic_logistic` | Logistic regression with alignment (Armijo line search) | Tucker et al. (2014) |

### Elastic Changepoint Detection

| Function | Method | Reference |
|----------|--------|-----------|
| `elastic_amp_changepoint` | Amplitude CUSUM changepoint detection | Tucker & Yarger (2023) |
| `elastic_ph_changepoint` | Phase CUSUM changepoint detection | Tucker & Yarger (2023) |
| `elastic_fpca_changepoint` | FPCA-based Hotelling T-squared CUSUM changepoint | Tucker & Yarger (2023) |

### Scalar-on-Shape Regression

| Function | Method | Reference |
|----------|--------|-----------|
| `scalar_on_shape` | y = h(shape_inner_product) + g(amplitude) + epsilon | Srivastava & Klassen (2016), S10.3 |

---

## Bibliography

- Arthur, D. & Vassilvitskii, S. (2007). k-means++: the advantages of careful seeding. *Proceedings of the 18th Annual ACM-SIAM Symposium on Discrete Algorithms*, 1027--1035. [doi:10.1145/1283383.1283494](https://doi.org/10.1145/1283383.1283494)
- Cheng, W., Dryden, I. L., & Huang, X. (2016). Bayesian registration of functions and curves. *Bayesian Analysis*, 11(2), 447--475. [doi:10.1214/15-BA957](https://doi.org/10.1214/15-BA957)
- Edelsbrunner, H., Letscher, D., & Zomorodian, A. (2002). Topological persistence and simplification. *Discrete & Computational Geometry*, 28(4), 511--533. [doi:10.1007/s00454-002-2885-2](https://doi.org/10.1007/s00454-002-2885-2)
- Ehlers, K., Guha, J., & Goldberg, D. (2011). Schild's ladder parallel transport for an arbitrary connection. *General Relativity and Gravitation*, 43, 2139--2154. [doi:10.1007/s10714-011-1178-x](https://doi.org/10.1007/s10714-011-1178-x)
- Fletcher, P. T., Venkatasubramanian, S., & Joshi, S. (2009). The geometric median on Riemannian manifolds with application to robust atlas estimation. *NeuroImage*, 45(1), S143--S152. [doi:10.1016/j.neuroimage.2008.10.052](https://doi.org/10.1016/j.neuroimage.2008.10.052)
- Fraiman, R. & Muniz, G. (2001). Trimmed means for functional data. *Test*, 10(2), 419--440. [doi:10.1007/BF02595706](https://doi.org/10.1007/BF02595706)
- Jung, S., Dryden, I. L., & Marron, J. S. (2012). Analysis of principal nested spheres. *Biometrika*, 99(3), 551--568. [doi:10.1093/biomet/ass022](https://doi.org/10.1093/biomet/ass022)
- Kurtek, S. & Srivastava, A. (2015). Elastic symmetry analysis of anatomical structures. *Proceedings of the IEEE Workshop on Mathematical Methods in Biomedical Image Analysis*, 1--8. [doi:10.1109/MMBIA.2012.6164733](https://doi.org/10.1109/MMBIA.2012.6164733)
- Kurtek, S., Srivastava, A., Klassen, E., & Ding, Z. (2012). Statistical modeling of curves using shapes and related features. *Journal of the American Statistical Association*, 107(499), 1152--1165. [doi:10.1080/01621459.2012.699770](https://doi.org/10.1080/01621459.2012.699770)
- Lopez-Pintado, S. & Romo, J. (2009). On the concept of depth for functional data. *Journal of the American Statistical Association*, 104(486), 718--734. [doi:10.1198/jasa.2009.0108](https://doi.org/10.1198/jasa.2009.0108)
- Lorenzi, M. & Pennec, X. (2014). Efficient parallel transport of deformations in time series of images: from Schild's to pole ladder. *Journal of Mathematical Imaging and Vision*, 50, 5--17. [doi:10.1007/s10851-013-0470-3](https://doi.org/10.1007/s10851-013-0470-3)
- Lu, Y., Herbei, R., & Kurtek, S. (2017). Bayesian registration of functions with a Gaussian process prior. *Journal of Computational and Graphical Statistics*, 26(4), 894--904. [doi:10.1080/10618600.2017.1336445](https://doi.org/10.1080/10618600.2017.1336445)
- Robinson, D. T. (2012). Functional data analysis and partial shape matching in the square root velocity framework. PhD thesis, Florida State University.
- Srivastava, A. & Klassen, E. P. (2016). *Functional and Shape Data Analysis*. Springer. [doi:10.1007/978-1-4939-4020-2](https://doi.org/10.1007/978-1-4939-4020-2)
- Srivastava, A., Wu, W., Kurtek, S., Klassen, E., & Marron, J. S. (2011). Registration of functional data using Fisher-Rao metric. [arXiv:1103.3817](https://arxiv.org/abs/1103.3817)
- Su, J., Kurtek, S., Klassen, E., & Srivastava, A. (2014). Statistical analysis of trajectories on Riemannian manifolds: bird migration, hurricane tracking and video surveillance. *Annals of Applied Statistics*, 8(1), 530--552. [doi:10.1214/13-AOAS701](https://doi.org/10.1214/13-AOAS701)
- Tucker, J. D. & Yarger, D. (2023). Elastic functional changepoint detection of climate impacts from localized sources. *Environmetrics*, 35(1), e2826. [doi:10.1002/env.2826](https://doi.org/10.1002/env.2826)
- Tucker, J. D., Wu, W., & Srivastava, A. (2013). Generative models for functional data using phase and amplitude separation. *Computational Statistics & Data Analysis*, 61, 50--66. [doi:10.1016/j.csda.2012.11.015](https://doi.org/10.1016/j.csda.2012.11.015)
- Tucker, J. D., Wu, W., & Srivastava, A. (2014). Analysis of proteomics data: Bayesian alignment of functions. *Electronic Journal of Statistics*, 8(2), 1769--1797. [doi:10.1214/14-EJS937](https://doi.org/10.1214/14-EJS937)
- Tukey, J. W. (1977). *Exploratory Data Analysis*. Addison-Wesley.
- Weiszfeld, E. (1937). Sur le point pour lequel la somme des distances de n points donnes est minimum. *Tohoku Mathematical Journal*, 43, 355--386.
