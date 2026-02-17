# Example 07: Functional Data Clustering

## What this demonstrates

Grouping functional observations into clusters using k-means and fuzzy c-means. Three groups of curves are generated from different processes (Fourier/Exponential, Legendre/Linear, Wiener/Wiener) with offsets, then combined. Clustering quality is evaluated using silhouette scores and the Calinski-Harabasz index.

Fuzzy c-means provides soft memberships (each curve belongs to all clusters with different degrees), which is useful when groups overlap.

## API functions used

- `clustering::kmeans_fd()` — k-means with k-means++ initialization
- `clustering::fuzzy_cmeans_fd()` — fuzzy c-means clustering
- `clustering::silhouette_score()` — per-curve silhouette scores
- `clustering::calinski_harabasz()` — Calinski-Harabasz index

## How to run

```bash
cargo run --example clustering
```

## Expected output

K-means results (convergence, within-SS, assignments, accuracy), silhouette scores per cluster, CH index, evaluation across k=2..5, and fuzzy c-means membership degrees.

## Key concepts

- **K-means++**: smart initialization that spreads initial centers apart
- **Silhouette score**: [-1, 1] measure of how well each point fits its cluster vs nearest other
- **Calinski-Harabasz**: ratio of between-cluster to within-cluster variance (higher = better)
- **Fuzzy c-means**: soft assignments with fuzziness parameter controlling overlap degree
