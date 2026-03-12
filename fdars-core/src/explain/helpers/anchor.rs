//! Anchor beam search helpers.

use crate::matrix::FdMatrix;

use super::stability::quantile_sorted;

/// Beam search for anchor rules in FPC score space.
pub(crate) fn anchor_beam_search(
    scores: &FdMatrix,
    ncomp: usize,
    n: usize,
    observation: usize,
    precision_threshold: f64,
    n_bins: usize,
    same_pred: &dyn Fn(usize) -> bool,
) -> (super::super::advanced::AnchorRule, Vec<bool>) {
    let bin_edges: Vec<Vec<f64>> = (0..ncomp)
        .map(|k| compute_bin_edges(scores, k, n, n_bins))
        .collect();

    let obs_bins: Vec<usize> = (0..ncomp)
        .map(|k| find_bin(scores[(observation, k)], &bin_edges[k], n_bins))
        .collect();

    let beam_width = 3;
    let mut best_conditions: Vec<usize> = Vec::new();
    let mut best_precision = 0.0;
    let mut best_matching = vec![true; n];
    let mut used = vec![false; ncomp];

    for _iter in 0..ncomp {
        let mut candidates = beam_search_candidates(
            scores,
            ncomp,
            &used,
            &obs_bins,
            &bin_edges,
            n_bins,
            &best_conditions,
            &best_matching,
            same_pred,
            beam_width,
        );

        if candidates.is_empty() {
            break;
        }

        let (new_conds, prec, matching) = candidates.remove(0);
        used[*new_conds.last().expect("non-empty collection")] = true;
        best_conditions = new_conds;
        best_precision = prec;
        best_matching = matching;

        if best_precision >= precision_threshold {
            break;
        }
    }

    let rule = build_anchor_rule(
        &best_conditions,
        &bin_edges,
        &obs_bins,
        best_precision,
        &best_matching,
        n,
    );
    (rule, best_matching)
}

/// Compute quantile bin edges for a column of scores.
fn compute_bin_edges(scores: &FdMatrix, component: usize, n: usize, n_bins: usize) -> Vec<f64> {
    let mut vals: Vec<f64> = (0..n).map(|i| scores[(i, component)]).collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut edges = Vec::with_capacity(n_bins + 1);
    edges.push(f64::NEG_INFINITY);
    for b in 1..n_bins {
        edges.push(quantile_sorted(&vals, b as f64 / n_bins as f64));
    }
    edges.push(f64::INFINITY);
    edges
}

/// Find which bin a value falls into given bin edges.
fn find_bin(value: f64, edges: &[f64], n_bins: usize) -> usize {
    for bi in 0..n_bins {
        if value >= edges[bi] && value < edges[bi + 1] {
            return bi;
        }
    }
    n_bins - 1
}

/// Compute which observations match a bin constraint on a component.
fn apply_bin_filter(
    current_matching: &[bool],
    scores: &FdMatrix,
    component: usize,
    bin: usize,
    edges: &[f64],
    n_bins: usize,
) -> Vec<bool> {
    let lo = edges[bin];
    let hi = edges[bin + 1];
    let is_last = bin == n_bins - 1;
    (0..current_matching.len())
        .map(|i| {
            current_matching[i]
                && scores[(i, component)] >= lo
                && (is_last || scores[(i, component)] < hi)
        })
        .collect()
}

/// Evaluate a candidate condition: add component to current matching and compute precision.
fn evaluate_anchor_candidate(
    current_matching: &[bool],
    scores: &FdMatrix,
    component: usize,
    bin: usize,
    edges: &[f64],
    n_bins: usize,
    same_pred: &dyn Fn(usize) -> bool,
) -> Option<(f64, Vec<bool>)> {
    let new_matching = apply_bin_filter(current_matching, scores, component, bin, edges, n_bins);
    let n_match = new_matching.iter().filter(|&&v| v).count();
    if n_match == 0 {
        return None;
    }
    let n_same = (0..new_matching.len())
        .filter(|&i| new_matching[i] && same_pred(i))
        .count();
    Some((n_same as f64 / n_match as f64, new_matching))
}

/// Evaluate all unused components in beam search and return sorted candidates.
fn beam_search_candidates(
    scores: &FdMatrix,
    ncomp: usize,
    used: &[bool],
    obs_bins: &[usize],
    bin_edges: &[Vec<f64>],
    n_bins: usize,
    best_conditions: &[usize],
    best_matching: &[bool],
    same_pred: &dyn Fn(usize) -> bool,
    beam_width: usize,
) -> Vec<(Vec<usize>, f64, Vec<bool>)> {
    let mut candidates: Vec<(Vec<usize>, f64, Vec<bool>)> = Vec::new();

    for k in 0..ncomp {
        if used[k] {
            continue;
        }
        if let Some((precision, matching)) = evaluate_anchor_candidate(
            best_matching,
            scores,
            k,
            obs_bins[k],
            &bin_edges[k],
            n_bins,
            same_pred,
        ) {
            let mut conds = best_conditions.to_vec();
            conds.push(k);
            candidates.push((conds, precision, matching));
        }
    }

    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    candidates.truncate(beam_width);
    candidates
}

/// Build an AnchorRule from selected components, bin edges, and observation bins.
fn build_anchor_rule(
    components: &[usize],
    bin_edges: &[Vec<f64>],
    obs_bins: &[usize],
    precision: f64,
    matching: &[bool],
    n: usize,
) -> super::super::advanced::AnchorRule {
    use super::super::advanced::{AnchorCondition, AnchorRule};

    let conditions: Vec<AnchorCondition> = components
        .iter()
        .map(|&k| AnchorCondition {
            component: k,
            lower_bound: bin_edges[k][obs_bins[k]],
            upper_bound: bin_edges[k][obs_bins[k] + 1],
        })
        .collect();
    let n_match = matching.iter().filter(|&&v| v).count();
    AnchorRule {
        conditions,
        precision,
        coverage: n_match as f64 / n as f64,
        n_matching: n_match,
    }
}
