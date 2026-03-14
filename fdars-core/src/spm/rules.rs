//! Runs and zone rules for control chart analysis.
//!
//! Implements Western Electric (WE) and Nelson rules for detecting
//! non-random patterns in control chart data. These supplement the
//! standard T²/SPE alarm logic with pattern-based detection.

use crate::error::FdarError;

/// A control chart pattern rule.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ChartRule {
    /// WE1: Any single point beyond 3σ.
    WE1,
    /// WE2: 2 of 3 consecutive points beyond 2σ on the same side.
    WE2,
    /// WE3: 4 of 5 consecutive points beyond 1σ on the same side.
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
/// # Arguments
/// * `values` - Monitoring statistic values (in time order)
/// * `center` - Center line (e.g. process mean)
/// * `sigma` - Standard deviation estimate
/// * `rules` - Rules to evaluate
///
/// # Errors
///
/// Returns [`FdarError::InvalidParameter`] if `sigma` is not positive.
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
/// # Errors
///
/// Returns [`FdarError::InvalidParameter`] if `sigma` is not positive.
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
        let all_within = window.iter().all(|&v| (v - center).abs() < sigma);
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
