//! Unified FDA data container for pipeline interchange.
//!
//! [`FdaData`] is a single layered container that flows between pipeline nodes.
//! Nodes read from existing layers and add new ones — data is additive, never
//! destructive. This replaces per-type wire enums with a composable structure.
//!
//! # Design
//!
//! - **Core**: curves (FdMatrix) + argvals + metadata (grouping, scalars)
//! - **Layers**: optional analysis results keyed by [`LayerKey`]
//! - Nodes declare what they *require* via `require_*` helpers
//! - Nodes add results via `set_layer`
//! - Layers compose: FPCA + Depth + Outliers can all coexist on one `FdaData`
//!
//! # Example
//!
//! ```
//! use fdars_core::wire::*;
//! use fdars_core::matrix::FdMatrix;
//!
//! let mut fd = FdaData::from_curves(
//!     FdMatrix::zeros(10, 50),
//!     (0..50).map(|i| i as f64 / 49.0).collect(),
//! );
//!
//! // A depth node reads curves, adds a Depth layer
//! let scores = vec![0.5; 10];
//! fd.set_layer(LayerKey::Depth, Layer::Depth(DepthLayer {
//!     scores,
//!     method: "fraiman_muniz".into(),
//! }));
//!
//! // Downstream node checks what's available
//! assert!(fd.has_layer(&LayerKey::Depth));
//! assert!(!fd.has_layer(&LayerKey::Fpca));
//! ```

use crate::matrix::FdMatrix;
use std::collections::HashMap;

// ─── Core Container ─────────────────────────────────────────────────────────

/// Unified FDA data object for pipeline interchange.
///
/// Carries functional data (curves + domain) plus optional analysis layers.
/// Nodes read what they need and add their results as new layers.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FdaData {
    // ── Core functional data ──
    /// Functional observations (n × m). `None` for tabular-only data.
    pub curves: Option<FdMatrix>,
    /// Evaluation grid (length m).
    pub argvals: Option<Vec<f64>>,

    // ── Metadata ──
    /// Named grouping variables (multiple allowed).
    pub grouping: Vec<GroupVar>,
    /// Named scalar variables (each length n).
    pub scalar_vars: Vec<NamedVec>,
    /// Tabular data for non-functional variables (n × p).
    pub tabular: Option<FdMatrix>,
    /// Column names for tabular data.
    pub column_names: Option<Vec<String>>,

    // ── Analysis layers ──
    /// Analysis results keyed by layer type.
    pub layers: HashMap<LayerKey, Layer>,
}

/// A named vector of f64 values (e.g., a scalar covariate or response).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NamedVec {
    pub name: String,
    pub values: Vec<f64>,
}

/// Named grouping variable with string labels.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GroupVar {
    /// Variable name (e.g., "treatment", "sex").
    pub name: String,
    /// Per-observation labels (length n).
    pub labels: Vec<String>,
    /// Unique labels in order of first appearance.
    pub unique: Vec<String>,
}

// ─── Layer Keys & Types ─────────────────────────────────────────────────────

/// Key identifying a layer type.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum LayerKey {
    /// Functional PCA decomposition.
    Fpca,
    /// PLS decomposition.
    Pls,
    /// Elastic alignment (Karcher mean + warps).
    Alignment,
    /// Precomputed n×n distance matrix.
    Distances,
    /// Functional depth scores.
    Depth,
    /// Outlier detection flags.
    Outliers,
    /// Cluster assignments.
    Clusters,
    /// Scalar-on-function regression fit.
    Regression,
    /// Function-on-scalar regression fit.
    FunctionOnScalar,
    /// Tolerance / confidence bands.
    Tolerance,
    /// Mean curve.
    Mean,
    /// SPM Phase I chart.
    SpmChart,
    /// SPM Phase II monitoring result.
    SpmMonitor,
    /// Explainability result (SHAP, PDP, etc.).
    Explain,
    /// User-defined extension.
    Custom(String),
}

/// Analysis result attached to an [`FdaData`].
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum Layer {
    Fpca(FpcaLayer),
    Pls(PlsLayer),
    Alignment(AlignmentLayer),
    Distances(DistancesLayer),
    Depth(DepthLayer),
    Outliers(OutlierLayer),
    Clusters(ClusterLayer),
    Regression(RegressionLayer),
    FunctionOnScalar(FosrLayer),
    Tolerance(ToleranceLayer),
    Mean(MeanLayer),
    SpmChart(SpmChartLayer),
    SpmMonitor(SpmMonitorLayer),
    Explain(ExplainLayer),
    Custom(CustomLayer),
}

// ─── Layer Structs ──────────────────────────────────────────────────────────

/// FPCA decomposition.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FpcaLayer {
    pub eigenvalues: Vec<f64>,
    pub variance_explained: Vec<f64>,
    /// Eigenfunctions (m × ncomp), each column is one eigenfunction.
    pub eigenfunctions: FdMatrix,
    /// Scores (n × ncomp).
    pub scores: FdMatrix,
    /// Mean function (length m).
    pub mean: Vec<f64>,
    /// Integration weights (length m).
    pub weights: Vec<f64>,
    pub ncomp: usize,
}

/// PLS decomposition.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PlsLayer {
    /// Weight vectors (m × ncomp).
    pub weights: FdMatrix,
    /// Scores (n × ncomp).
    pub scores: FdMatrix,
    /// Loadings (m × ncomp).
    pub loadings: FdMatrix,
    pub ncomp: usize,
}

/// Elastic alignment result.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AlignmentLayer {
    /// Aligned curves (n × m).
    pub aligned: FdMatrix,
    /// Warping functions (n × m).
    pub warps: FdMatrix,
    /// Karcher mean (length m).
    pub mean: Vec<f64>,
    /// Mean SRSF (length m).
    pub mean_srsf: Vec<f64>,
    /// Optional: number of alignment iterations performed.
    pub n_iter: Option<usize>,
    /// Optional: whether the alignment converged.
    pub converged: Option<bool>,
}

impl AlignmentLayer {
    /// Reconstruct a [`crate::alignment::KarcherMeanResult`] from this layer's fields.
    ///
    /// This enables downstream functions that require `&KarcherMeanResult`
    /// (e.g., `elastic_fpca`, `elastic_changepoint`) to work from a
    /// serialized/restored `AlignmentLayer`.
    pub fn to_karcher_mean_result(&self) -> crate::alignment::KarcherMeanResult {
        crate::alignment::KarcherMeanResult {
            mean: self.mean.clone(),
            mean_srsf: self.mean_srsf.clone(),
            gammas: self.warps.clone(),
            aligned_data: self.aligned.clone(),
            n_iter: self.n_iter.unwrap_or(0),
            converged: self.converged.unwrap_or(true),
            aligned_srsfs: None,
        }
    }
}

/// Precomputed n×n distance matrix with method metadata.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DistancesLayer {
    /// Symmetric n×n distance matrix.
    pub dist_mat: FdMatrix,
    /// Distance method used (e.g., "elastic", "l2", "dtw", "amplitude", "phase").
    pub method: String,
}

/// Functional depth scores.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DepthLayer {
    /// Depth score per observation (length n).
    pub scores: Vec<f64>,
    /// Method name.
    pub method: String,
}

/// Outlier detection result.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct OutlierLayer {
    /// Outlier flag per observation (length n).
    pub flags: Vec<bool>,
    /// Detection threshold.
    pub threshold: f64,
    /// Method name.
    pub method: String,
    /// Optional: MEI scores (for outliergram).
    pub mei: Option<Vec<f64>>,
    /// Optional: MBD scores (for outliergram).
    pub mbd: Option<Vec<f64>>,
    /// Optional: magnitude outlyingness.
    pub magnitude: Option<Vec<f64>>,
    /// Optional: shape outlyingness.
    pub shape: Option<Vec<f64>>,
    /// Optional: outliergram parabola intercept coefficient.
    pub outliergram_a0: Option<f64>,
    /// Optional: outliergram parabola linear coefficient.
    pub outliergram_a1: Option<f64>,
    /// Optional: outliergram parabola quadratic coefficient.
    pub outliergram_a2: Option<f64>,
}

/// Cluster assignments.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ClusterLayer {
    /// Cluster label per observation (0-indexed, length n).
    pub labels: Vec<usize>,
    /// Number of clusters.
    pub k: usize,
    /// Method name.
    pub method: String,
    /// Optional: cluster centers (k rows × m cols).
    pub centers: Option<FdMatrix>,
    /// Optional: medoid indices (length k).
    pub medoid_indices: Option<Vec<usize>>,
    /// Optional: silhouette scores (length n).
    pub silhouette: Option<Vec<f64>>,
}

/// Scalar-on-function regression fit.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RegressionLayer {
    /// Method name (e.g., "fregre_lm", "fregre_pls", "fregre_np", "elastic").
    pub method: String,
    /// Functional coefficient β(t) (length m). `None` for nonparametric.
    pub beta_t: Option<Vec<f64>>,
    /// Fitted values (length n).
    pub fitted_values: Vec<f64>,
    /// Residuals (length n).
    pub residuals: Vec<f64>,
    /// Observed response (length n).
    pub observed_y: Vec<f64>,
    /// R².
    pub r_squared: f64,
    /// Adjusted R².
    pub adj_r_squared: Option<f64>,
    /// Intercept.
    pub intercept: f64,
    /// Number of components used (0 for nonparametric).
    pub ncomp: usize,
    /// Evaluation grid for β(t).
    pub argvals: Option<Vec<f64>>,
    /// Pointwise standard errors of β(t).
    pub beta_se: Option<Vec<f64>>,
    /// Optional: human-readable model name.
    pub model_name: Option<String>,
    /// Optional: number of training observations.
    pub n_obs: Option<usize>,
    /// Optional: FPCA decomposition used by the model (needed for explain functions).
    pub fpca: Option<Box<FpcaLayer>>,
    /// Optional: model selection details (ncomp candidates, GCV/AIC/BIC scores).
    #[cfg(feature = "serde")]
    pub selection_extra: Option<serde_json::Value>,
    /// Optional: model selection details.
    #[cfg(not(feature = "serde"))]
    pub selection_extra: Option<HashMap<String, Vec<f64>>>,
}

/// Function-on-scalar regression fit.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FosrLayer {
    /// Coefficient functions (p × m), one per predictor.
    pub coefficients: FdMatrix,
    /// Fitted curves (n × m).
    pub fitted: FdMatrix,
    /// R² per grid point (length m).
    pub r_squared_t: Vec<f64>,
}

/// Tolerance / confidence band.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ToleranceLayer {
    /// Lower bound (length m).
    pub lower: Vec<f64>,
    /// Upper bound (length m).
    pub upper: Vec<f64>,
    /// Center (length m).
    pub center: Vec<f64>,
    /// Method name.
    pub method: String,
}

/// Mean curve.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MeanLayer {
    /// Mean function (length m).
    pub mean: Vec<f64>,
}

/// SPM Phase I chart.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SpmChartLayer {
    /// T² control limit.
    pub t2_limit: f64,
    /// SPE control limit.
    pub spe_limit: f64,
    /// Phase I T² statistics.
    pub t2_stats: Vec<f64>,
    /// Phase I SPE statistics.
    pub spe_stats: Vec<f64>,
    /// Number of FPC components.
    pub ncomp: usize,
    /// Significance level.
    pub alpha: f64,
    /// Optional: eigenvalues from FPCA (length ncomp).
    pub eigenvalues: Option<Vec<f64>>,
    /// Optional: FPCA mean function (length m).
    pub fpca_mean: Option<Vec<f64>>,
    /// Optional: FPCA rotation/eigenfunctions (m × ncomp).
    pub fpca_rotation: Option<FdMatrix>,
    /// Optional: FPCA integration weights (length m).
    pub fpca_weights: Option<Vec<f64>>,
}

impl SpmChartLayer {
    /// Create from an [`crate::spm::SpmChart`] (lossless — stores all fields needed for monitoring).
    pub fn from_chart(chart: &crate::spm::SpmChart) -> Self {
        Self {
            t2_limit: chart.t2_limit.ucl,
            spe_limit: chart.spe_limit.ucl,
            t2_stats: chart.t2_phase1.clone(),
            spe_stats: chart.spe_phase1.clone(),
            ncomp: chart.eigenvalues.len(),
            alpha: chart.config.alpha,
            eigenvalues: Some(chart.eigenvalues.clone()),
            fpca_mean: Some(chart.fpca.mean.clone()),
            fpca_rotation: Some(chart.fpca.rotation.clone()),
            fpca_weights: Some(chart.fpca.weights.clone()),
        }
    }

    /// Check if this layer has enough state for field-based monitoring.
    pub fn can_monitor(&self) -> bool {
        self.eigenvalues.is_some()
            && self.fpca_mean.is_some()
            && self.fpca_rotation.is_some()
            && self.fpca_weights.is_some()
    }
}

/// SPM Phase II monitoring result.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SpmMonitorLayer {
    /// T² statistics for new observations.
    pub t2_stats: Vec<f64>,
    /// SPE statistics for new observations.
    pub spe_stats: Vec<f64>,
    /// T² control limit.
    pub t2_limit: f64,
    /// SPE control limit.
    pub spe_limit: f64,
    /// T² alarm flags.
    pub t2_alarms: Vec<bool>,
    /// SPE alarm flags.
    pub spe_alarms: Vec<bool>,
}

/// Extra data for explain layers.
///
/// When the `serde` feature is enabled this is [`serde_json::Value`] (arbitrary
/// JSON). Otherwise it is a flat `HashMap<String, Vec<f64>>`.
#[cfg(feature = "serde")]
pub type ExplainExtra = serde_json::Value;
/// Extra data for explain layers (flat map when `serde` is disabled).
#[cfg(not(feature = "serde"))]
pub type ExplainExtra = HashMap<String, Vec<f64>>;

/// Explainability result.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ExplainLayer {
    /// Method name (e.g., "shap", "pdp", "ale", "permutation_importance").
    pub method: String,
    /// Values (interpretation depends on method).
    pub values: Vec<f64>,
    /// Labels for the values.
    pub labels: Vec<String>,
    /// Additional method-specific data (arbitrary JSON when `serde` feature is
    /// enabled, flat `HashMap<String, Vec<f64>>` otherwise).
    pub extra: Option<ExplainExtra>,
}

/// Custom layer data type.
///
/// When the `serde` feature is enabled this is [`serde_json::Value`] (arbitrary
/// JSON). Otherwise it is a flat `HashMap<String, Vec<f64>>`.
#[cfg(feature = "serde")]
pub type CustomData = serde_json::Value;
/// Custom layer data type (flat map when `serde` is disabled).
#[cfg(not(feature = "serde"))]
pub type CustomData = HashMap<String, Vec<f64>>;

/// User-defined layer for extensions.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CustomLayer {
    pub name: String,
    pub data: CustomData,
}

// ─── FdaData Constructors ───────────────────────────────────────────────────

impl FdaData {
    /// Create from functional curves + grid.
    pub fn from_curves(curves: FdMatrix, argvals: Vec<f64>) -> Self {
        Self {
            curves: Some(curves),
            argvals: Some(argvals),
            grouping: Vec::new(),
            scalar_vars: Vec::new(),
            tabular: None,
            column_names: None,
            layers: HashMap::new(),
        }
    }

    /// Create from tabular (non-functional) data.
    pub fn from_tabular(tabular: FdMatrix, column_names: Vec<String>) -> Self {
        Self {
            curves: None,
            argvals: None,
            grouping: Vec::new(),
            scalar_vars: Vec::new(),
            tabular: Some(tabular),
            column_names: Some(column_names),
            layers: HashMap::new(),
        }
    }

    /// Create empty container.
    pub fn empty() -> Self {
        Self {
            curves: None,
            argvals: None,
            grouping: Vec::new(),
            scalar_vars: Vec::new(),
            tabular: None,
            column_names: None,
            layers: HashMap::new(),
        }
    }

    // ── Requirement checks ──

    /// Require functional curves to be present.
    pub fn require_curves(&self) -> Result<(&FdMatrix, &[f64]), String> {
        match (&self.curves, &self.argvals) {
            (Some(c), Some(a)) => Ok((c, a)),
            _ => Err("FdaData requires functional curves + argvals".into()),
        }
    }

    /// Require a specific layer to be present.
    pub fn require_layer(&self, key: &LayerKey) -> Result<&Layer, String> {
        self.layers
            .get(key)
            .ok_or_else(|| format!("FdaData missing required layer: {key:?}"))
    }

    // ── Layer access ──

    /// Check if a layer is present.
    pub fn has_layer(&self, key: &LayerKey) -> bool {
        self.layers.contains_key(key)
    }

    /// Get a layer by key.
    pub fn get_layer(&self, key: &LayerKey) -> Option<&Layer> {
        self.layers.get(key)
    }

    /// Set (add or replace) a layer.
    pub fn set_layer(&mut self, key: LayerKey, layer: Layer) {
        self.layers.insert(key, layer);
    }

    /// Remove a layer.
    pub fn remove_layer(&mut self, key: &LayerKey) -> Option<Layer> {
        self.layers.remove(key)
    }

    /// List all layer keys present.
    pub fn layer_keys(&self) -> Vec<&LayerKey> {
        self.layers.keys().collect()
    }

    // ── Typed layer accessors ──

    /// Get FPCA layer if present.
    pub fn fpca(&self) -> Option<&FpcaLayer> {
        match self.layers.get(&LayerKey::Fpca)? {
            Layer::Fpca(l) => Some(l),
            _ => None,
        }
    }

    /// Get distances layer if present.
    pub fn distances(&self) -> Option<&DistancesLayer> {
        match self.layers.get(&LayerKey::Distances)? {
            Layer::Distances(l) => Some(l),
            _ => None,
        }
    }

    /// Get alignment layer if present.
    pub fn alignment(&self) -> Option<&AlignmentLayer> {
        match self.layers.get(&LayerKey::Alignment)? {
            Layer::Alignment(l) => Some(l),
            _ => None,
        }
    }

    /// Get regression layer if present.
    pub fn regression(&self) -> Option<&RegressionLayer> {
        match self.layers.get(&LayerKey::Regression)? {
            Layer::Regression(l) => Some(l),
            _ => None,
        }
    }

    /// Get cluster layer if present.
    pub fn clusters(&self) -> Option<&ClusterLayer> {
        match self.layers.get(&LayerKey::Clusters)? {
            Layer::Clusters(l) => Some(l),
            _ => None,
        }
    }

    /// Get depth layer if present.
    pub fn depth(&self) -> Option<&DepthLayer> {
        match self.layers.get(&LayerKey::Depth)? {
            Layer::Depth(l) => Some(l),
            _ => None,
        }
    }

    /// Get outlier layer if present.
    pub fn outliers(&self) -> Option<&OutlierLayer> {
        match self.layers.get(&LayerKey::Outliers)? {
            Layer::Outliers(l) => Some(l),
            _ => None,
        }
    }

    // ── Metadata helpers ──

    /// Number of observations (from curves, tabular, or first scalar var).
    pub fn n_obs(&self) -> usize {
        if let Some(c) = &self.curves {
            return c.nrows();
        }
        if let Some(t) = &self.tabular {
            return t.nrows();
        }
        self.scalar_vars.first().map_or(0, |v| v.values.len())
    }

    /// Number of grid points (0 if no functional data).
    pub fn n_points(&self) -> usize {
        self.argvals.as_ref().map_or(0, |a| a.len())
    }

    /// Add a scalar variable.
    pub fn add_scalar(&mut self, name: impl Into<String>, values: Vec<f64>) {
        self.scalar_vars.push(NamedVec {
            name: name.into(),
            values,
        });
    }

    /// Get a scalar variable by name.
    pub fn get_scalar(&self, name: &str) -> Option<&[f64]> {
        self.scalar_vars
            .iter()
            .find(|v| v.name == name)
            .map(|v| v.values.as_slice())
    }

    /// Add a grouping variable with per-observation string labels.
    ///
    /// Unique labels are computed automatically in order of first appearance.
    pub fn add_grouping(&mut self, name: impl Into<String>, labels: Vec<String>) {
        let mut unique = Vec::new();
        for lab in &labels {
            if !unique.contains(lab) {
                unique.push(lab.clone());
            }
        }
        self.grouping.push(GroupVar {
            name: name.into(),
            labels,
            unique,
        });
    }

    /// Look up a grouping variable by name.
    pub fn get_grouping(&self, name: &str) -> Option<&GroupVar> {
        self.grouping.iter().find(|g| g.name == name)
    }
}

// ─── From conversions ──────────────────────────────────────────────────────

impl From<&crate::scalar_on_function::FregreLmResult> for RegressionLayer {
    fn from(fit: &crate::scalar_on_function::FregreLmResult) -> Self {
        let n_tune = fit.fpca.scores.nrows();
        let eigenvalues: Vec<f64> = fit
            .fpca
            .singular_values
            .iter()
            .map(|s| s * s / (n_tune as f64 - 1.0).max(1.0))
            .collect();
        let total_var: f64 = eigenvalues.iter().sum();
        let variance_explained = if total_var > 0.0 {
            eigenvalues.iter().map(|&ev| ev / total_var).collect()
        } else {
            vec![0.0; eigenvalues.len()]
        };

        let fpca_layer = FpcaLayer {
            eigenvalues,
            variance_explained,
            eigenfunctions: fit.fpca.rotation.clone(),
            scores: fit.fpca.scores.clone(),
            mean: fit.fpca.mean.clone(),
            weights: fit.fpca.weights.clone(),
            ncomp: fit.ncomp,
        };

        RegressionLayer {
            method: "fregre_lm".into(),
            beta_t: Some(fit.beta_t.clone()),
            fitted_values: fit.fitted_values.clone(),
            residuals: fit.residuals.clone(),
            observed_y: Vec::new(),
            r_squared: fit.r_squared,
            adj_r_squared: Some(fit.r_squared_adj),
            intercept: fit.intercept,
            ncomp: fit.ncomp,
            argvals: None,
            beta_se: Some(fit.beta_se.clone()),
            model_name: None,
            n_obs: Some(fit.fitted_values.len()),
            fpca: Some(Box::new(fpca_layer)),
            selection_extra: None,
        }
    }
}

impl From<&crate::scalar_on_function::PlsRegressionResult> for RegressionLayer {
    fn from(fit: &crate::scalar_on_function::PlsRegressionResult) -> Self {
        RegressionLayer {
            method: "fregre_pls".into(),
            beta_t: Some(fit.beta_t.clone()),
            fitted_values: fit.fitted_values.clone(),
            residuals: fit.residuals.clone(),
            observed_y: Vec::new(),
            r_squared: fit.r_squared,
            adj_r_squared: Some(fit.r_squared_adj),
            intercept: fit.intercept,
            ncomp: fit.ncomp,
            argvals: None,
            beta_se: None,
            model_name: None,
            n_obs: Some(fit.fitted_values.len()),
            fpca: None, // PLS uses a different decomposition; no FPCA layer
            selection_extra: None,
        }
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_curves_basic() {
        let fd = FdaData::from_curves(
            FdMatrix::zeros(10, 50),
            (0..50).map(|i| i as f64 / 49.0).collect(),
        );
        assert_eq!(fd.n_obs(), 10);
        assert_eq!(fd.n_points(), 50);
        assert!(fd.require_curves().is_ok());
        assert!(!fd.has_layer(&LayerKey::Fpca));
    }

    #[test]
    fn add_and_retrieve_layers() {
        let mut fd = FdaData::from_curves(
            FdMatrix::zeros(5, 20),
            (0..20).map(|i| i as f64 / 19.0).collect(),
        );

        fd.set_layer(
            LayerKey::Depth,
            Layer::Depth(DepthLayer {
                scores: vec![0.5; 5],
                method: "fraiman_muniz".into(),
            }),
        );

        assert!(fd.has_layer(&LayerKey::Depth));
        assert!(!fd.has_layer(&LayerKey::Fpca));
        assert!(fd.depth().is_some());
        assert_eq!(fd.depth().unwrap().scores.len(), 5);
        assert_eq!(fd.layer_keys().len(), 1);
    }

    #[test]
    fn require_missing_layer_errors() {
        let fd = FdaData::from_curves(FdMatrix::zeros(3, 10), vec![0.0; 10]);
        assert!(fd.require_layer(&LayerKey::Fpca).is_err());
    }

    #[test]
    fn scalar_vars() {
        let mut fd = FdaData::empty();
        fd.add_scalar("height", vec![170.0, 180.0, 165.0]);
        assert_eq!(fd.get_scalar("height").unwrap(), &[170.0, 180.0, 165.0]);
        assert!(fd.get_scalar("weight").is_none());
        assert_eq!(fd.n_obs(), 3);
    }

    #[test]
    fn multiple_layers_compose() {
        let mut fd = FdaData::from_curves(FdMatrix::zeros(10, 30), vec![0.0; 30]);

        fd.set_layer(
            LayerKey::Depth,
            Layer::Depth(DepthLayer {
                scores: vec![0.5; 10],
                method: "fm".into(),
            }),
        );
        fd.set_layer(
            LayerKey::Outliers,
            Layer::Outliers(OutlierLayer {
                flags: vec![false; 10],
                threshold: 0.1,
                method: "lrt".into(),
                mei: None,
                mbd: None,
                magnitude: None,
                shape: None,
                outliergram_a0: None,
                outliergram_a1: None,
                outliergram_a2: None,
            }),
        );
        fd.set_layer(
            LayerKey::Distances,
            Layer::Distances(DistancesLayer {
                dist_mat: FdMatrix::zeros(10, 10),
                method: "elastic".into(),
            }),
        );

        assert_eq!(fd.layer_keys().len(), 3);
        assert!(fd.depth().is_some());
        assert!(fd.outliers().is_some());
        assert!(fd.distances().is_some());
    }

    #[test]
    fn regression_layer_from_fregre_lm() {
        let (n, m) = (20, 30);
        let data = FdMatrix::from_column_major(
            (0..n * m)
                .map(|k| {
                    let i = (k % n) as f64;
                    let j = (k / n) as f64;
                    ((i + 1.0) * j * 0.2).sin()
                })
                .collect(),
            n,
            m,
        )
        .unwrap();
        let y: Vec<f64> = (0..n).map(|i| (i as f64 * 0.5).sin()).collect();
        let fit = crate::scalar_on_function::fregre_lm(&data, &y, None, 3).unwrap();

        let layer = RegressionLayer::from(&fit);
        assert_eq!(layer.method, "fregre_lm");
        assert_eq!(layer.ncomp, 3);
        assert_eq!(layer.fitted_values.len(), n);
        assert_eq!(layer.residuals.len(), n);
        assert!(layer.fpca.is_some());
        let fpca = layer.fpca.as_ref().unwrap();
        assert_eq!(fpca.ncomp, 3);
        assert_eq!(fpca.mean.len(), m);
        assert_eq!(fpca.eigenfunctions.shape(), (m, 3));
        assert_eq!(fpca.scores.shape(), (n, 3));
        assert_eq!(fpca.weights.len(), m);
        assert_eq!(fpca.eigenvalues.len(), 3);
        // Variance explained should sum to ~1.0
        let ve_sum: f64 = fpca.variance_explained.iter().sum();
        assert!(
            (ve_sum - 1.0).abs() < 1e-10,
            "variance_explained sum = {ve_sum}"
        );
        assert!(layer.beta_t.is_some());
        assert!(layer.beta_se.is_some());
        assert_eq!(layer.n_obs, Some(n));
        assert!((layer.r_squared - fit.r_squared).abs() < 1e-14);
    }

    #[test]
    fn regression_layer_from_pls() {
        let n = 30;
        let m = 50;
        let t: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
        let vals: Vec<f64> = (0..n)
            .flat_map(|i| {
                t.iter()
                    .map(move |&tj| (2.0 * std::f64::consts::PI * tj).sin() + 0.1 * i as f64)
            })
            .collect();
        let data = FdMatrix::from_column_major(vals, n, m).unwrap();
        let y: Vec<f64> = (0..n).map(|i| 2.0 + 0.5 * i as f64).collect();

        let fit = crate::scalar_on_function::fregre_pls(&data, &y, &t, 3, None).unwrap();

        let layer = RegressionLayer::from(&fit);
        assert_eq!(layer.method, "fregre_pls");
        assert_eq!(layer.ncomp, 3);
        assert_eq!(layer.fitted_values.len(), n);
        assert!(layer.fpca.is_none()); // PLS has no FPCA decomposition
        assert!(layer.beta_t.is_some());
        assert!(layer.beta_se.is_none()); // PLS has no beta SE
        assert_eq!(layer.n_obs, Some(n));
        assert!((layer.r_squared - fit.r_squared).abs() < 1e-14);
    }
}
