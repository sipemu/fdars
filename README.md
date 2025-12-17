# Functional Data Analysis (FDA)

High-performance Functional Data Analysis tools implemented in Rust with R bindings.

## Repository Structure

```
├── fdars-core/     # Pure Rust library (publishable to crates.io)
├── fdars/          # R package with Rust backend
└── Cargo.toml      # Workspace configuration
```

### fdars-core (Rust)

Pure Rust implementation of FDA algorithms. Can be used independently in Rust projects.

**Features:**
- Functional data operations (mean, centering, derivatives, norms)
- Depth measures (Fraiman-Muniz, modal, band, random projection, etc.)
- Distance metrics (Lp, Hausdorff, DTW, Fourier-based)
- Basis representations (B-splines, Fourier, P-splines)
- Clustering (k-means, fuzzy c-means)
- Smoothing (Nadaraya-Watson, local polynomial, k-NN)
- Regression (functional PCA, PLS, ridge)
- Outlier detection

```toml
[dependencies]
fdars-core = "0.1"
```

See [fdars-core/README.md](fdars-core/README.md) for details.

### fdars (R Package)

R package providing FDA functions powered by the Rust backend.

**Installation:**

```r
# From GitHub (requires Rust toolchain)
devtools::install_github("sipemu/fdars", subdir = "fdars")

# From binary release (no Rust required)
# Download from GitHub Releases, then:
install.packages("path/to/fdars_x.y.z.tgz", repos = NULL, type = "mac.binary")  # macOS
install.packages("path/to/fdars_x.y.z.zip", repos = NULL, type = "win.binary")  # Windows
```

See [fdars/README.md](fdars/README.md) for details.

## License

MIT
