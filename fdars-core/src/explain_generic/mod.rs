//! Generic explainability for any FPC-based model.
//!
//! Provides the [`FpcPredictor`] trait and generic functions that work with
//! any model that implements it — including linear regression, logistic regression,
//! and classification models (LDA, QDA, kNN).
//!
//! The generic functions delegate to internal helpers from [`crate::explain`].

use crate::explain::project_scores;
use crate::matrix::FdMatrix;
use crate::scalar_on_function::{sigmoid, FregreLmResult, FunctionalLogisticResult};

pub mod ale;
pub mod anchor;
pub mod counterfactual;
pub mod friedman;
pub mod importance;
pub mod lime;
pub mod pdp;
pub mod prototype;
pub mod saliency;
pub mod shap;
pub mod sobol;
pub mod stability;

#[cfg(test)]
mod tests;

// Re-export all public items from submodules
pub use ale::generic_ale;
pub use anchor::generic_anchor;
pub use counterfactual::generic_counterfactual;
pub use friedman::generic_friedman_h;
pub use importance::{generic_conditional_permutation_importance, generic_permutation_importance};
pub use lime::generic_lime;
pub use pdp::generic_pdp;
pub use prototype::generic_prototype_criticism;
pub use saliency::{generic_domain_selection, generic_saliency};
pub use shap::generic_shap_values;
pub use sobol::generic_sobol_indices;
pub use stability::{generic_stability, generic_vif};

// ---------------------------------------------------------------------------
// TaskType + FpcPredictor trait
// ---------------------------------------------------------------------------

/// The type of prediction task a model solves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TaskType {
    Regression,
    BinaryClassification,
    MulticlassClassification(usize),
}

/// Trait abstracting over any FPC-based model for generic explainability.
///
/// Implement this for a model that projects functional data onto FPC scores
/// and produces a scalar prediction (value, probability, or class label).
pub trait FpcPredictor: Send + Sync {
    /// Mean function from FPCA (length m).
    fn fpca_mean(&self) -> &[f64];

    /// Rotation matrix from FPCA (m × ncomp).
    fn fpca_rotation(&self) -> &FdMatrix;

    /// Number of FPC components used.
    fn ncomp(&self) -> usize;

    /// Training FPC scores matrix (n × ncomp).
    fn training_scores(&self) -> &FdMatrix;

    /// What kind of prediction task this model solves.
    fn task_type(&self) -> TaskType;

    /// Integration weights from FPCA (length m).
    fn fpca_weights(&self) -> &[f64];

    /// Predict from FPC scores + optional scalar covariates → single f64.
    ///
    /// - **Regression**: predicted value
    /// - **Binary classification**: P(Y=1)
    /// - **Multiclass**: predicted class label as f64
    fn predict_from_scores(&self, scores: &[f64], scalar_covariates: Option<&[f64]>) -> f64;

    /// Project functional data to FPC scores.
    fn project(&self, data: &FdMatrix) -> FdMatrix {
        project_scores(
            data,
            self.fpca_mean(),
            self.fpca_rotation(),
            self.ncomp(),
            self.fpca_weights(),
        )
    }
}

// ---------------------------------------------------------------------------
// Implement FpcPredictor for FregreLmResult
// ---------------------------------------------------------------------------

impl FpcPredictor for FregreLmResult {
    fn fpca_mean(&self) -> &[f64] {
        &self.fpca.mean
    }

    fn fpca_rotation(&self) -> &FdMatrix {
        &self.fpca.rotation
    }

    fn ncomp(&self) -> usize {
        self.ncomp
    }

    fn training_scores(&self) -> &FdMatrix {
        &self.fpca.scores
    }

    fn task_type(&self) -> TaskType {
        TaskType::Regression
    }

    fn fpca_weights(&self) -> &[f64] {
        &self.fpca.weights
    }

    fn predict_from_scores(&self, scores: &[f64], scalar_covariates: Option<&[f64]>) -> f64 {
        let ncomp = self.ncomp;
        let mut yhat = self.coefficients[0]; // intercept
        for k in 0..ncomp {
            yhat += self.coefficients[1 + k] * scores[k];
        }
        if let Some(sc) = scalar_covariates {
            for j in 0..self.gamma.len() {
                yhat += self.gamma[j] * sc[j];
            }
        }
        yhat
    }
}

// ---------------------------------------------------------------------------
// Implement FpcPredictor for FunctionalLogisticResult
// ---------------------------------------------------------------------------

impl FpcPredictor for FunctionalLogisticResult {
    fn fpca_mean(&self) -> &[f64] {
        &self.fpca.mean
    }

    fn fpca_rotation(&self) -> &FdMatrix {
        &self.fpca.rotation
    }

    fn ncomp(&self) -> usize {
        self.ncomp
    }

    fn training_scores(&self) -> &FdMatrix {
        &self.fpca.scores
    }

    fn task_type(&self) -> TaskType {
        TaskType::BinaryClassification
    }

    fn fpca_weights(&self) -> &[f64] {
        &self.fpca.weights
    }

    fn predict_from_scores(&self, scores: &[f64], scalar_covariates: Option<&[f64]>) -> f64 {
        let ncomp = self.ncomp;
        let mut eta = self.intercept;
        for k in 0..ncomp {
            eta += self.coefficients[1 + k] * scores[k];
        }
        if let Some(sc) = scalar_covariates {
            for j in 0..self.gamma.len() {
                eta += self.gamma[j] * sc[j];
            }
        }
        sigmoid(eta)
    }
}

// ---------------------------------------------------------------------------
// Shared helpers used across submodules
// ---------------------------------------------------------------------------

/// Compute the baseline metric for a model on training data.
pub(super) fn compute_baseline_metric(
    model: &dyn FpcPredictor,
    scores: &FdMatrix,
    y: &[f64],
    n: usize,
) -> f64 {
    match model.task_type() {
        TaskType::Regression => {
            // R²
            let y_mean: f64 = y.iter().sum::<f64>() / n as f64;
            let ss_tot: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
            if ss_tot == 0.0 {
                return 0.0;
            }
            let ss_res: f64 = (0..n)
                .map(|i| {
                    let s: Vec<f64> = (0..model.ncomp()).map(|k| scores[(i, k)]).collect();
                    let pred = model.predict_from_scores(&s, None);
                    (y[i] - pred).powi(2)
                })
                .sum();
            1.0 - ss_res / ss_tot
        }
        TaskType::BinaryClassification => {
            let correct: usize = (0..n)
                .filter(|&i| {
                    let s: Vec<f64> = (0..model.ncomp()).map(|k| scores[(i, k)]).collect();
                    let pred = model.predict_from_scores(&s, None);
                    let pred_class = if pred >= 0.5 { 1.0 } else { 0.0 };
                    (pred_class - y[i]).abs() < 1e-10
                })
                .count();
            correct as f64 / n as f64
        }
        TaskType::MulticlassClassification(_) => {
            let correct: usize = (0..n)
                .filter(|&i| {
                    let s: Vec<f64> = (0..model.ncomp()).map(|k| scores[(i, k)]).collect();
                    let pred = model.predict_from_scores(&s, None);
                    (pred.round() - y[i]).abs() < 1e-10
                })
                .count();
            correct as f64 / n as f64
        }
    }
}

/// Compute the metric for permuted scores.
pub(super) fn compute_metric_from_score_matrix(
    model: &dyn FpcPredictor,
    score_mat: &FdMatrix,
    y: &[f64],
    n: usize,
) -> f64 {
    let ncomp = model.ncomp();
    match model.task_type() {
        TaskType::Regression => {
            let y_mean: f64 = y.iter().sum::<f64>() / n as f64;
            let ss_tot: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
            if ss_tot == 0.0 {
                return 0.0;
            }
            let ss_res: f64 = (0..n)
                .map(|i| {
                    let s: Vec<f64> = (0..ncomp).map(|k| score_mat[(i, k)]).collect();
                    let pred = model.predict_from_scores(&s, None);
                    (y[i] - pred).powi(2)
                })
                .sum();
            1.0 - ss_res / ss_tot
        }
        TaskType::BinaryClassification => {
            let correct: usize = (0..n)
                .filter(|&i| {
                    let s: Vec<f64> = (0..ncomp).map(|k| score_mat[(i, k)]).collect();
                    let pred = model.predict_from_scores(&s, None);
                    let pred_class = if pred >= 0.5 { 1.0 } else { 0.0 };
                    (pred_class - y[i]).abs() < 1e-10
                })
                .count();
            correct as f64 / n as f64
        }
        TaskType::MulticlassClassification(_) => {
            let correct: usize = (0..n)
                .filter(|&i| {
                    let s: Vec<f64> = (0..ncomp).map(|k| score_mat[(i, k)]).collect();
                    let pred = model.predict_from_scores(&s, None);
                    (pred.round() - y[i]).abs() < 1e-10
                })
                .count();
            correct as f64 / n as f64
        }
    }
}
