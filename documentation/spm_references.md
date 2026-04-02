# Statistical Process Monitoring -- References

This document maps each public function in the `spm` module to the paper(s) that describe the underlying method.

## Phase I / Phase II Framework

| Function | Method | Reference |
|----------|--------|-----------|
| `spm_phase1` | FPCA-based Phase I chart (T-squared + SPE) | Horvath & Kokoszka (2012), Ch. 13, pp. 323--352; Flores et al. (2022) |
| `spm_monitor` | Phase II monitoring against Phase I limits | Horvath & Kokoszka (2012) |
| `mf_spm_phase1` | Multivariate functional Phase I (MFPCA) | Happ & Greven (2018); Horvath & Kokoszka (2012) |
| `mf_spm_monitor` | Multivariate functional Phase II | Happ & Greven (2018) |

## Monitoring Statistics

| Function | Method | Reference |
|----------|--------|-----------|
| `hotelling_t2` | Hotelling T-squared statistic | Hotelling (1947), pp. 111--113 |
| `hotelling_t2_regularized` | T-squared with eigenvalue floor | -- |
| `spe_univariate` | Squared Prediction Error (Q-statistic) | Bersimis et al. (2007), S2.1, pp. 519--522 |
| `spe_multivariate` | Multivariate SPE | Bersimis et al. (2007) |

## Control Limits

| Function | Method | Reference |
|----------|--------|-----------|
| `t2_control_limit` | Chi-squared T-squared UCL | Woodall & Ncube (1985), S2, pp. 286--288 |
| `spe_control_limit` | Moment-matched chi-squared SPE UCL | Box (1954), Theorem 1, pp. 292--295 |
| `t2_limit_robust` | Bootstrap / KDE / empirical T-squared limits | Efron & Tibshirani (1993), S13.3, pp. 178--182; Silverman (1986), S3.4.1 |
| `spe_limit_robust` | Bootstrap / KDE / empirical SPE limits | Efron & Tibshirani (1993); Silverman (1986) |

## EWMA

| Function | Method | Reference |
|----------|--------|-----------|
| `ewma_scores` | Exponentially Weighted Moving Average | Roberts (1959), S2, pp. 241--243 |
| `spm_ewma_monitor` | EWMA-based T-squared / SPE monitoring | Roberts (1959); Lucas & Saccucci (1990), Table 3 |

## MEWMA

| Function | Method | Reference |
|----------|--------|-----------|
| `spm_mewma_monitor` | Multivariate EWMA | Lowry et al. (1992), S4, pp. 46--53 |

## CUSUM

| Function | Method | Reference |
|----------|--------|-----------|
| `spm_cusum_monitor` | Univariate + multivariate CUSUM | Page (1954), pp. 100--115; Crosier (1988), S2, Eq. 2.1, pp. 291--303 |
| `spm_cusum_monitor_with_restart` | CUSUM with automatic restart | Crosier (1988) |

## Adaptive EWMA

| Function | Method | Reference |
|----------|--------|-----------|
| `spm_amewma_monitor` | Adaptive EWMA (time-varying lambda) | Sparks (2000), Eqs. 3--5, pp. 162--164; Capizzi & Masarotto (2003), S2, pp. 200--202 |

## FRCC (Covariate-Adjusted)

| Function | Method | Reference |
|----------|--------|-----------|
| `frcc_phase1` | Functional Regression Control Chart | Capezza et al. (2020), S3.1, S4, pp. 477--500 |
| `frcc_monitor` | FRCC Phase II monitoring | Capezza et al. (2020) |

## Profile Monitoring

| Function | Method | Reference |
|----------|--------|-----------|
| `profile_phase1` | Profile monitoring (rolling FOSR windows) | Bartlett (1946), S3, pp. 27--41; Ledolter & Swersey (2007), Ch. 6 |
| `profile_monitor` | Phase II profile monitoring | Bartlett (1946) |

## Partial-Domain Monitoring

| Function | Method | Reference |
|----------|--------|-----------|
| `spm_monitor_partial` | BLUP-based score imputation for incomplete curves | Yao et al. (2005), S3, Eq. 6, pp. 577--590 (PACE) |
| `spm_monitor_partial_batch` | Batch partial-domain monitoring | Yao et al. (2005) |

## Elastic / Phase-Aware SPM

| Function | Method | Reference |
|----------|--------|-----------|
| `elastic_spm_phase1` | Elastic alignment + amplitude/phase SPM | Srivastava et al. (2011), Theorem 3.1; Tucker et al. (2013), S2.1, Eqs. 3--5, pp. 52--54 |
| `elastic_spm_monitor` | Phase II elastic SPM | Tucker et al. (2013) |

## Iterative Phase I

| Function | Method | Reference |
|----------|--------|-----------|
| `spm_phase1_iterative` | Iterative Phase I with outlier removal | Sullivan & Woodall (1996), S3, pp. 398--408; Chenouri et al. (2009), S2, pp. 259--271; Rousseeuw & Leroy (1987), S1.3, S4.1 |

## Contribution Diagnostics

| Function | Method | Reference |
|----------|--------|-----------|
| `t2_contributions` | Per-variable T-squared contributions | Kourti & MacGregor (1996), S3, pp. 413--416 |
| `spe_contributions` | Per-variable SPE contributions | Westerhuis et al. (2000), Eq. 8, p. 100 |
| `t2_pc_contributions` | Per-component T-squared contributions | Kourti & MacGregor (1996), S3, p. 414 |
| `t2_pc_significance` | Per-component chi-squared significance | Kourti & MacGregor (1996) |

## Component Selection

| Function | Method | Reference |
|----------|--------|-----------|
| `select_ncomp` | Automatic ncomp (scree elbow, variance threshold, optimal rate, broken stick) | Cattell (1966), p. 252; Hall & Horowitz (2007), Theorem 1, p. 73; Jackson (1991), Ch. 4; Kaiser (1960), pp. 145--146 |

## Control Chart Rules

| Function | Method | Reference |
|----------|--------|-----------|
| `western_electric_rules` | Western Electric rules (WE1--WE4) | Western Electric (1956), Ch. 4, pp. 25--28 |
| `nelson_rules` | Nelson rules (N5--N7 + WE1--WE4) | Nelson (1984), p. 238; Nelson (1985), p. 115 |
| `evaluate_rules` | Generic rule evaluation engine | Western Electric (1956); Nelson (1984, 1985) |

## ARL Computation

| Function | Method | Reference |
|----------|--------|-----------|
| `arl0_t2` | In-control ARL (T-squared, Monte Carlo) | -- |
| `arl1_t2` | Out-of-control ARL (T-squared) | -- |
| `arl0_ewma_t2` | In-control ARL (EWMA T-squared) | Lucas & Saccucci (1990) |
| `arl0_spe` | In-control ARL (SPE chart) | -- |

## Multivariate FPCA

| Function | Method | Reference |
|----------|--------|-----------|
| `mfpca` | Multivariate Functional PCA | Happ & Greven (2018), S2.2--2.3, Eqs. 3--5, pp. 649--659 |

---

## Bibliography

- Bartlett, M. S. (1946). The statistical analysis of series of events. *Journal of the Royal Statistical Society, Series B*, 8(1), 27--41. [doi:10.1111/j.2517-6161.1946.tb00105.x](https://doi.org/10.1111/j.2517-6161.1946.tb00105.x)
- Bersimis, S., Psarakis, S., & Panaretos, J. (2007). Multivariate statistical process control charts: an overview. *Quality and Reliability Engineering International*, 23(5), 517--543. [doi:10.1002/qre.829](https://doi.org/10.1002/qre.829)
- Box, G. E. P. (1954). Some theorems on quadratic forms applied in the study of analysis of variance problems, I. *Annals of Mathematical Statistics*, 25(2), 290--302. [doi:10.1214/aoms/1177728786](https://doi.org/10.1214/aoms/1177728786)
- Capezza, C., Lepore, A., Menafoglio, A., Palumbo, B., & Vantini, S. (2020). Control charts for monitoring ship operating conditions and CO2 emissions based on scalar-on-function regression. *Applied Stochastic Models in Business and Industry*, 36(3), 477--500. [doi:10.1002/asmb.2513](https://doi.org/10.1002/asmb.2513)
- Capizzi, G. & Masarotto, G. (2003). An adaptive exponentially weighted moving average control chart. *Technometrics*, 45(3), 199--207. [doi:10.1198/004017003000000023](https://doi.org/10.1198/004017003000000023)
- Cattell, R. B. (1966). The scree test for the number of factors. *Multivariate Behavioral Research*, 1(2), 245--276. [doi:10.1207/s15327906mbr0102_10](https://doi.org/10.1207/s15327906mbr0102_10)
- Chenouri, S., Steiner, S. H., & Variyath, A. M. (2009). A multivariate robust control chart for individual observations. *Journal of Quality Technology*, 41(3), 259--271. [doi:10.1080/00224065.2009.11917781](https://doi.org/10.1080/00224065.2009.11917781)
- Crosier, R. B. (1988). Multivariate generalizations of cumulative sum quality-control schemes. *Technometrics*, 30(3), 291--303. [doi:10.1080/00401706.1988.10488402](https://doi.org/10.1080/00401706.1988.10488402)
- Efron, B. & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall/CRC. [doi:10.1007/978-1-4899-4541-9](https://doi.org/10.1007/978-1-4899-4541-9)
- Flores, M., Naya, S., Fernandez-Casal, R., Zaragoza, S., Rana, P., & Tarrio-Saavedra, J. (2020). Constructing a control chart using functional data. *Mathematics*, 8(1), 58. [doi:10.3390/math8010058](https://doi.org/10.3390/math8010058)
- Hall, P. & Horowitz, J. L. (2007). Methodology and convergence rates for functional linear regression. *Annals of Statistics*, 35(1), 70--91. [doi:10.1214/009053606000001109](https://doi.org/10.1214/009053606000001109)
- Happ, C. & Greven, S. (2018). Multivariate functional principal component analysis for data observed on different (dimensional) domains. *Journal of the American Statistical Association*, 113(522), 649--659. [doi:10.1080/01621459.2017.1305535](https://doi.org/10.1080/01621459.2017.1305535)
- Horvath, L. & Kokoszka, P. (2012). *Inference for Functional Data with Applications*. Springer. [doi:10.1007/978-1-4614-3655-3](https://doi.org/10.1007/978-1-4614-3655-3)
- Hotelling, H. (1947). Multivariate quality control, illustrated by the air testing of sample bombsights. In *Techniques of Statistical Analysis* (pp. 111--184). McGraw-Hill.
- Jackson, J. E. (1991). *A User's Guide to Principal Components*. Wiley. [doi:10.1002/0471725331](https://doi.org/10.1002/0471725331)
- Kaiser, H. F. (1960). The application of electronic computers to factor analysis. *Educational and Psychological Measurement*, 20, 141--151. [doi:10.1177/001316446002000116](https://doi.org/10.1177/001316446002000116)
- Kourti, T. & MacGregor, J. F. (1996). Multivariate SPC methods for process and product monitoring. *Journal of Quality Technology*, 28(4), 409--428. [doi:10.1080/00224065.1996.11979699](https://doi.org/10.1080/00224065.1996.11979699)
- Ledolter, J. & Swersey, A. (2007). *Testing 1-2-3: Experimental Design with Applications in Marketing and Service Operations*. Stanford University Press.
- Lowry, C. A., Woodall, W. H., Champ, C. W., & Rigdon, S. E. (1992). A multivariate exponentially weighted moving average control chart. *Technometrics*, 34(1), 46--53. [doi:10.2307/1269551](https://doi.org/10.2307/1269551)
- Lucas, J. M. & Saccucci, M. S. (1990). Exponentially weighted moving average control schemes: properties and enhancements. *Technometrics*, 32(1), 1--12. [doi:10.2307/1269835](https://doi.org/10.2307/1269835)
- Nelson, L. S. (1984). The Shewhart control chart -- tests for special causes. *Journal of Quality Technology*, 16(4), 237--239. [doi:10.1080/00224065.1984.11978921](https://doi.org/10.1080/00224065.1984.11978921)
- Nelson, L. S. (1985). Interpreting Shewhart X-bar control charts. *Journal of Quality Technology*, 17(2), 114--116. [doi:10.1080/00224065.1985.11978943](https://doi.org/10.1080/00224065.1985.11978943)
- O'Neill, M. E. (2014). *PCG: A Family of Simple Fast Space-Efficient Statistically Good Algorithms for Random Number Generation*. Harvey Mudd College Technical Report HMC-CS-2014-0905. [www.pcg-random.org](https://www.pcg-random.org/paper.html)
- Page, E. S. (1954). Continuous inspection schemes. *Biometrika*, 41(1--2), 100--115. [doi:10.1093/biomet/41.1-2.100](https://doi.org/10.1093/biomet/41.1-2.100)
- Roberts, S. W. (1959). Control chart tests based on geometric moving averages. *Technometrics*, 1(3), 239--250. [doi:10.1080/00401706.1959.10489860](https://doi.org/10.1080/00401706.1959.10489860)
- Rousseeuw, P. J. & Leroy, A. M. (1987). *Robust Regression and Outlier Detection*. Wiley. [doi:10.1002/0471725382](https://doi.org/10.1002/0471725382)
- Silverman, B. W. (1986). *Density Estimation for Statistics and Data Analysis*. Chapman & Hall. [doi:10.1007/978-1-4899-3324-9](https://doi.org/10.1007/978-1-4899-3324-9)
- Sparks, R. S. (2000). CUSUM charts for signalling varying location shifts. *Journal of Quality Technology*, 32(2), 157--171. [doi:10.1080/00224065.2000.11979988](https://doi.org/10.1080/00224065.2000.11979988)
- Srivastava, A., Wu, W., Kurtek, S., Klassen, E., & Marron, J. S. (2011). Registration of functional data using Fisher-Rao metric. [arXiv:1103.3817](https://arxiv.org/abs/1103.3817)
- Sullivan, J. H. & Woodall, W. H. (1996). A comparison of multivariate control charts for individual observations. *Journal of Quality Technology*, 28(4), 398--408. [doi:10.1080/00224065.1996.11979698](https://doi.org/10.1080/00224065.1996.11979698)
- Tucker, J. D., Wu, W., & Srivastava, A. (2013). Generative models for functional data using phase and amplitude separation. *Computational Statistics & Data Analysis*, 61, 50--66. [doi:10.1016/j.csda.2012.11.015](https://doi.org/10.1016/j.csda.2012.11.015)
- Western Electric Company (1956). *Statistical Quality Control Handbook*. Western Electric Co.
- Westerhuis, J. A., Gurden, S. P., & Smilde, A. K. (2000). Generalized contribution plots in multivariate statistical process monitoring. *Chemometrics and Intelligent Laboratory Systems*, 51(1), 95--114. [doi:10.1016/S0169-7439(00)00062-9](https://doi.org/10.1016/S0169-7439(00)00062-9)
- Woodall, W. H. & Ncube, M. M. (1985). Multivariate CUSUM quality-control procedures. *Technometrics*, 27(3), 285--292. [doi:10.1080/00401706.1985.10488049](https://doi.org/10.1080/00401706.1985.10488049)
- Yao, F., Muller, H.-G., & Wang, J.-L. (2005). Functional data analysis for sparse longitudinal data. *Journal of the American Statistical Association*, 100(470), 577--590. [doi:10.1198/016214504000001745](https://doi.org/10.1198/016214504000001745)
