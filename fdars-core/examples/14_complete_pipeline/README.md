# Example 14: Complete FDA Pipeline

## What this demonstrates

An end-to-end functional data analysis workflow integrating concepts from all previous examples: simulate multi-group data with noise and outliers, smooth with P-splines, detect and remove outliers, perform FPCA for dimensionality reduction, cluster curves, and characterize clusters using depth measures and PC scores.

This shows how FDA methods compose into a coherent analysis pipeline.

## API functions used

- `simulation::sim_fundata()` — generate synthetic functional data
- `simulation::add_error_pointwise()` — add measurement noise
- `basis::pspline_fit_1d()` — P-spline smoothing
- `outliers::outliers_threshold_lrt()` — bootstrap outlier threshold
- `outliers::detect_outliers_lrt()` — flag outliers
- `regression::fdata_to_pc_1d()` — FPCA for dimensionality reduction
- `clustering::kmeans_fd()` — k-means clustering
- `clustering::silhouette_score()` — clustering validation
- `depth::modified_band_1d()` — depth-based characterization

## How to run

```bash
cargo run --example complete_pipeline
```

## Expected output

Step-by-step pipeline output: data generation summary, smoothing statistics, detected outliers, FPCA variance explained, clustering results with accuracy and silhouette scores, depth statistics per cluster, and mean PC scores by cluster.

## Key concepts

- **Pipeline**: simulate → smooth → detect outliers → FPCA → cluster → characterize
- **Outlier removal**: improves downstream FPCA and clustering by removing contamination
- **FPCA + clustering**: cluster on PC scores for efficient, noise-reduced grouping
- **Depth characterization**: identifies the most typical curve in each cluster
