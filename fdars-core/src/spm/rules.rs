//! Runs and zone rules for control chart analysis.
//!
//! Implements Western Electric (WE) and Nelson rules for detecting
//! non-random patterns in control chart data. These rules supplement
//! T²/SPE alarm logic by detecting non-random patterns (trends, runs,
//! stratification) that may indicate process shifts even when individual
//! points remain within control limits.
//!
//! # Theoretical ARL₀ (independent normal data)
//!
//! | Rule | ARL₀ |
//! |------|------|
//! | WE1 (3σ) | 370.4 |
//! | WE2 (2/3 beyond 2σ) | 122.0 |
//! | WE3 (4/5 beyond 1σ) | 90.7 |
//! | WE4 (8 same side) | 119.7 |
//! | Nelson5 (6 monotone) | 360.0 |
//! | Nelson6 (14 alternating) | 182.0 |
//! | Nelson7 (15 within 1σ) | 44.1 |
//! | All WE combined | ~46 |
//!
//! # Assumptions
//!
//! These rules assume observations are independent and identically distributed
//! under in-control conditions. When applied to autocorrelated data (e.g.,
//! EWMA-smoothed statistics), false alarm rates may be inflated.
//!
//! # Computational complexity
//!
//! All rules evaluate in O(n·W) time where W is the maximum window size
//! (1 for WE1, 15 for Nelson7). Memory usage is O(W) per rule.
//!
//! # NaN handling
//!
//! NaN values in the input are not handled specially; comparisons involving
//! NaN return false, so NaN points will not trigger WE1–WE3 or Nelson5–Nelson6
//! violations but may affect WE4/Nelson7 (which check all-same-side or
//! all-within conditions).
//!
//! # References
//!
//! - Western Electric Company (1956). *Statistical Quality Control Handbook*.
//!   Western Electric Co., Indianapolis. Chapter 4, pp. 25–28.
//! - Nelson, L.S. (1984). The Shewhart control chart — tests for special
//!   causes. *Journal of Quality Technology*, 16(4), 237-239, p. 238.
//! - Nelson, L.S. (1985). Interpreting Shewhart X̄ control charts. *Journal
//!   of Quality Technology*, 17(2), 114-116, p. 115.
//!
//! # False Alarm Rates
//!
//! For false alarm rate estimation under these rules with independent data,
//! the theoretical ARL₀ for WE1 alone is ~370 (at 3σ). Combining multiple
//! rules increases sensitivity but reduces ARL₀; empirical calibration via
//! `arl0_t2` or simulation is recommended when using multiple rules
//! simultaneously.

use crate::error::FdarError;

/// A control chart pattern rule.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ChartRule {
    /// WE1: Any single point beyond 3σ.
    WE1,
    /// WE2: 2 of 3 consecutive points beyond 2σ on the same side.
    ///
    /// Multiple overlapping windows may detect the same event; deduplicate
    /// by index if needed.
    WE2,
    /// WE3: 4 of 5 consecutive points beyond 1σ on the same side.
    ///
    /// Multiple overlapping windows may detect the same event; deduplicate
    /// by index if needed.
    WE3,
    /// WE4: 8 consecutive points on the same side of center.
    WE4,
    /// Nelson5: 6 points in a row steadily increasing or decreasing.
    Nelson5,
    /// Nelson6: 14 points in a row alternating up and down.
    Nelson6,
    /// Nelson7: 15 consecutive points within 1σ of center (stratification).
    Nelson7,
    /// Custom run rule: `n_points` consecutive points beyond `k_sigma`.
    CustomRun {
        /// Number of consecutive points required.
        n_points: usize,
        /// Sigma threshold.
        k_sigma: f64,
    },
}

/// A detected rule violation.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct RuleViolation {
    /// The rule that was violated.
    pub rule: ChartRule,
    /// Indices of the points involved in the violation.
    pub indices: Vec<usize>,
}

/// Evaluate a set of chart rules against a sequence of values.
///
/// Each [`RuleViolation`] in the returned vector contains the triggering
/// rule and the indices of the observations involved. Multiple violations
/// from different rules may overlap. For multi-rule evaluation, consider
/// using [`western_electric_rules`] or [`nelson_rules`] convenience
/// functions.
///
/// # Arguments
/// * `values` - Monitoring statistic values (in time order)
/// * `center` - Center line (e.g. process mean)
/// * `sigma` - Standard deviation estimate
/// * `rules` - Rules to evaluate
///
/// # Example
///
/// ```
/// use fdars_core::spm::rules::{evaluate_rules, ChartRule};
/// let values = vec![0.0, 0.1, 0.0, -0.1, 0.0, 0.0, 0.1, 10.0]; // outlier at index 7
/// let violations = evaluate_rules(&values, 0.0, 1.0, &[ChartRule::WE1]).unwrap();
/// assert!(!violations.is_empty());
/// assert_eq!(violations[0].indices, vec![7]);
/// ```
///
/// # Errors
///
/// Returns [`FdarError::InvalidParameter`] if `sigma` is not positive.
#[must_use = "violations should not be discarded"]
pub fn evaluate_rules(
    values: &[f64],
    center: f64,
    sigma: f64,
    rules: &[ChartRule],
) -> Result<Vec<RuleViolation>, FdarError> {
    if sigma <= 0.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "sigma",
            message: format!("sigma must be positive, got {sigma}"),
        });
    }

    let mut violations = Vec::new();
    for rule in rules {
        let mut rule_violations = evaluate_single_rule(values, center, sigma, rule);
        violations.append(&mut rule_violations);
    }
    Ok(violations)
}

/// Apply all four Western Electric rules.
///
/// # Example
///
/// ```
/// use fdars_core::spm::rules::western_electric_rules;
/// let values = vec![0.1, -0.2, 0.3, -0.1, 0.2, -0.3, 0.1, -0.2]; // in-control
/// let violations = western_electric_rules(&values, 0.0, 1.0).unwrap();
/// assert!(violations.is_empty()); // no violations for mild fluctuations
/// ```
///
/// # Errors
///
/// Returns [`FdarError::InvalidParameter`] if `sigma` is not positive.
#[must_use = "violations should not be discarded"]
pub fn western_electric_rules(
    values: &[f64],
    center: f64,
    sigma: f64,
) -> Result<Vec<RuleViolation>, FdarError> {
    evaluate_rules(
        values,
        center,
        sigma,
        &[
            ChartRule::WE1,
            ChartRule::WE2,
            ChartRule::WE3,
            ChartRule::WE4,
        ],
    )
}

/// Apply Nelson rules 5-7 (in addition to WE rules which cover Nelson 1-4).
///
/// # Errors
///
/// Returns [`FdarError::InvalidParameter`] if `sigma` is not positive.
#[must_use = "violations should not be discarded"]
pub fn nelson_rules(
    values: &[f64],
    center: f64,
    sigma: f64,
) -> Result<Vec<RuleViolation>, FdarError> {
    evaluate_rules(
        values,
        center,
        sigma,
        &[
            ChartRule::WE1,
            ChartRule::WE2,
            ChartRule::WE3,
            ChartRule::WE4,
            ChartRule::Nelson5,
            ChartRule::Nelson6,
            ChartRule::Nelson7,
        ],
    )
}

fn evaluate_single_rule(
    values: &[f64],
    center: f64,
    sigma: f64,
    rule: &ChartRule,
) -> Vec<RuleViolation> {
    match rule {
        ChartRule::WE1 => eval_we1(values, center, sigma),
        ChartRule::WE2 => eval_we2(values, center, sigma),
        ChartRule::WE3 => eval_we3(values, center, sigma),
        ChartRule::WE4 => eval_we4(values, center),
        ChartRule::Nelson5 => eval_nelson5(values),
        ChartRule::Nelson6 => eval_nelson6(values),
        ChartRule::Nelson7 => eval_nelson7(values, center, sigma),
        ChartRule::CustomRun { n_points, k_sigma } => {
            eval_custom_run(values, center, sigma, *n_points, *k_sigma)
        }
    }
}

// WE1: any single point beyond 3σ
fn eval_we1(values: &[f64], center: f64, sigma: f64) -> Vec<RuleViolation> {
    let mut violations = Vec::new();
    for (i, &v) in values.iter().enumerate() {
        if (v - center).abs() > 3.0 * sigma {
            violations.push(RuleViolation {
                rule: ChartRule::WE1,
                indices: vec![i],
            });
        }
    }
    violations
}

// WE2: 2 of 3 consecutive points beyond 2σ on the same side
fn eval_we2(values: &[f64], center: f64, sigma: f64) -> Vec<RuleViolation> {
    let mut violations = Vec::new();
    if values.len() < 3 {
        return violations;
    }
    for start in 0..values.len() - 2 {
        let window = &values[start..start + 3];
        // Check upper side
        let above_2s: Vec<usize> = window
            .iter()
            .enumerate()
            .filter(|(_, &v)| v - center > 2.0 * sigma)
            .map(|(j, _)| start + j)
            .collect();
        if above_2s.len() >= 2 {
            violations.push(RuleViolation {
                rule: ChartRule::WE2,
                indices: above_2s,
            });
        }
        // Check lower side
        let below_2s: Vec<usize> = window
            .iter()
            .enumerate()
            .filter(|(_, &v)| center - v > 2.0 * sigma)
            .map(|(j, _)| start + j)
            .collect();
        if below_2s.len() >= 2 {
            violations.push(RuleViolation {
                rule: ChartRule::WE2,
                indices: below_2s,
            });
        }
    }
    violations
}

// WE3: 4 of 5 consecutive points beyond 1σ on the same side
fn eval_we3(values: &[f64], center: f64, sigma: f64) -> Vec<RuleViolation> {
    let mut violations = Vec::new();
    if values.len() < 5 {
        return violations;
    }
    for start in 0..values.len() - 4 {
        let window = &values[start..start + 5];
        // Check upper side
        let above_1s: Vec<usize> = window
            .iter()
            .enumerate()
            .filter(|(_, &v)| v - center > sigma)
            .map(|(j, _)| start + j)
            .collect();
        if above_1s.len() >= 4 {
            violations.push(RuleViolation {
                rule: ChartRule::WE3,
                indices: above_1s,
            });
        }
        // Check lower side
        let below_1s: Vec<usize> = window
            .iter()
            .enumerate()
            .filter(|(_, &v)| center - v > sigma)
            .map(|(j, _)| start + j)
            .collect();
        if below_1s.len() >= 4 {
            violations.push(RuleViolation {
                rule: ChartRule::WE3,
                indices: below_1s,
            });
        }
    }
    violations
}

// WE4: 8 consecutive points on the same side of center
fn eval_we4(values: &[f64], center: f64) -> Vec<RuleViolation> {
    let mut violations = Vec::new();
    if values.len() < 8 {
        return violations;
    }
    for start in 0..values.len() - 7 {
        let window = &values[start..start + 8];
        let all_above = window.iter().all(|&v| v > center);
        let all_below = window.iter().all(|&v| v < center);
        if all_above || all_below {
            violations.push(RuleViolation {
                rule: ChartRule::WE4,
                indices: (start..start + 8).collect(),
            });
        }
    }
    violations
}

// Nelson5: 6 points in a row steadily increasing or decreasing
fn eval_nelson5(values: &[f64]) -> Vec<RuleViolation> {
    let mut violations = Vec::new();
    if values.len() < 6 {
        return violations;
    }
    for start in 0..values.len() - 5 {
        let window = &values[start..start + 6];
        let increasing = window.windows(2).all(|w| w[1] > w[0]);
        let decreasing = window.windows(2).all(|w| w[1] < w[0]);
        if increasing || decreasing {
            violations.push(RuleViolation {
                rule: ChartRule::Nelson5,
                indices: (start..start + 6).collect(),
            });
        }
    }
    violations
}

// Nelson6: 14 points in a row alternating up and down
fn eval_nelson6(values: &[f64]) -> Vec<RuleViolation> {
    let mut violations = Vec::new();
    if values.len() < 14 {
        return violations;
    }
    for start in 0..values.len() - 13 {
        let window = &values[start..start + 14];
        let alternating = window
            .windows(3)
            .all(|w| (w[1] > w[0] && w[1] > w[2]) || (w[1] < w[0] && w[1] < w[2]));
        if alternating {
            violations.push(RuleViolation {
                rule: ChartRule::Nelson6,
                indices: (start..start + 14).collect(),
            });
        }
    }
    violations
}

// Nelson7: 15 consecutive points within 1σ of center (stratification)
fn eval_nelson7(values: &[f64], center: f64, sigma: f64) -> Vec<RuleViolation> {
    let mut violations = Vec::new();
    if values.len() < 15 {
        return violations;
    }
    for start in 0..values.len() - 14 {
        let window = &values[start..start + 15];
        let all_within = window.iter().all(|&v| (v - center).abs() <= sigma);
        if all_within {
            violations.push(RuleViolation {
                rule: ChartRule::Nelson7,
                indices: (start..start + 15).collect(),
            });
        }
    }
    violations
}

// CustomRun: n_points consecutive beyond k_sigma (on either side)
fn eval_custom_run(
    values: &[f64],
    center: f64,
    sigma: f64,
    n_points: usize,
    k_sigma: f64,
) -> Vec<RuleViolation> {
    let mut violations = Vec::new();
    // A negative sigma threshold is meaningless — no point can be beyond
    // a negative multiple of sigma, so return early with no violations.
    if k_sigma < 0.0 {
        return violations;
    }
    if n_points == 0 || values.len() < n_points {
        return violations;
    }
    for start in 0..=values.len() - n_points {
        let window = &values[start..start + n_points];
        let all_beyond = window.iter().all(|&v| (v - center).abs() > k_sigma * sigma);
        if all_beyond {
            violations.push(RuleViolation {
                rule: ChartRule::CustomRun { n_points, k_sigma },
                indices: (start..start + n_points).collect(),
            });
        }
    }
    violations
}
