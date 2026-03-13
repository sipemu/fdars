/// Result of a tolerance band computation.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ToleranceBand {
    /// Lower bound at each evaluation point
    pub lower: Vec<f64>,
    /// Upper bound at each evaluation point
    pub upper: Vec<f64>,
    /// Center function (typically the mean)
    pub center: Vec<f64>,
    /// Half-width at each evaluation point
    pub half_width: Vec<f64>,
}

/// Type of tolerance band.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum BandType {
    /// Independent interval at each evaluation point
    Pointwise,
    /// Single scaling factor across all points (wider, controls family-wise error)
    Simultaneous,
}

/// Non-conformity score for conformal prediction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum NonConformityScore {
    /// Supremum norm: max_t |y(t) - center(t)|
    SupNorm,
    /// L2 norm: sqrt(sum (y(t) - center(t))^2)
    L2,
}

/// Multiplier distribution for Degras SCB.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum MultiplierDistribution {
    /// Standard normal multipliers
    Gaussian,
    /// Rademacher multipliers (+1/-1 with equal probability)
    Rademacher,
}

/// Bootstrap method for the equivalence test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum EquivalenceBootstrap {
    /// Multiplier bootstrap (Gaussian or Rademacher weights)
    Multiplier(MultiplierDistribution),
    /// Percentile bootstrap (resample with replacement)
    Percentile,
}

/// Result of a functional equivalence test (TOST).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct EquivalenceTestResult {
    /// Test statistic: sup_t |d_hat(t)|
    pub test_statistic: f64,
    /// Bootstrap p-value
    pub p_value: f64,
    /// Critical value c_alpha from bootstrap distribution
    pub critical_value: f64,
    /// Simultaneous confidence band for the mean difference
    pub scb: ToleranceBand,
    /// Whether the entire SCB lies within [-delta, delta]
    pub equivalent: bool,
    /// Equivalence margin
    pub delta: f64,
    /// Significance level
    pub alpha: f64,
}

/// Exponential family for generalized FPCA tolerance bands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ExponentialFamily {
    /// Gaussian (identity link)
    Gaussian,
    /// Binomial (logit link)
    Binomial,
    /// Poisson (log link)
    Poisson,
}
