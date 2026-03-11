# fdars TODO

## Explainability Coverage

### Current State
- `FregreLmResult`, `FunctionalLogisticResult`: Full explainability (27 dedicated + 15 generic functions)
- `ClassifFit` (LDA/QDA/kNN): 15 generic functions via `FpcPredictor` trait
- All other models: No explainability

### Easy Win
- **`ElasticPcrResult`**: Already stores elastic FPCA (vert/horiz/joint). Implement `FpcPredictor` to unlock all 15 generic functions.

### Needs Plumbing
- **`RidgeResult`**: Store `FpcaResult` in the result struct, then implement `FpcPredictor`.
- **`PlsResult`**: PLS weights/scores serve a similar role to FPCA rotation/scores. Could implement `FpcPredictor` using PLS components as the perturbation space.

### Perturbation Space Design Problem

FPCA is the current perturbation space for all generic explainability (SHAP, LIME, ALE, PDP, etc.). This is correct for models that are defined in FPC space (linear, logistic, LDA/QDA/kNN) but not universally valid:

- **Phase variation (elastic models)**: FPCA mixes amplitude and timing. Perturbing FPC scores of unaligned curves produces unrealistic shapes. The natural perturbation space is (aligned amplitude scores, warping scores) — i.e., the SRVF decomposition the elastic model already uses.
- **Localized features**: FPCA eigenfunctions are global. If a localized feature (e.g., a spike at t=0.3) drives prediction, many FPCs are needed to represent it. Basis coefficients or wavelets would be more faithful.
- **Nonlinear manifold structure**: Linear FPC perturbations can produce off-manifold curves (e.g., negative values where positivity is required).
- **Nonparametric models (`FregreNpResult`)**: The model exploits functional distances directly, not FPC structure. FPCA-based perturbations may miss the variation that matters for prediction.

### Possible Approaches

1. **Pragmatic (FPCA everywhere)**: Implement `FpcPredictor` for all scalar-on-function models by projecting training data through FPCA externally. Simple, gives useful first-order answers, but may be misleading for elastic/NP models. Requires documenting the limitation.

2. **Model-specific perturbation spaces**: Each model defines its own natural perturbation space:
   - FPC-based models → FPC scores (current)
   - Elastic models → (amplitude FPC scores, warping FPC scores)
   - Basis models → basis coefficients
   - NP models → functional distance-based perturbations

   This is the principled approach but requires a more general trait than `FpcPredictor`.

3. **Generic `FunctionalPredictor` trait**: A minimal trait that only requires `predict(data) -> Vec<f64>`. Explainability functions handle perturbations externally (e.g., FPCA the data, perturb, reconstruct, predict). The model doesn't need to know about the perturbation space. Flexible but pushes complexity to the caller.

### Not Applicable (don't predict scalars from functions)
- `FosrResult` / `FosrFpcResult` — function-on-scalar (reversed direction)
- `FmmResult` — functional mixed model (predicts functions)
- `SmoothBasisResult` — curve smoothing (no scalar response)
- `FregreNpResult` — could work with approach 2 or 3, but not with `FpcPredictor`

## Regression Diagnostics

### Current State
All regression diagnostics are hardcoded to `&FregreLmResult` (or `&FunctionalLogisticResult` for logistic variants). Meanwhile, conformal prediction already supports all models via closure-based generic functions.

### Inherently LM-Specific (need hat matrix H = X(X'X)⁻¹X')
These rely on the linear model's hat matrix for leverage and cannot be meaningfully generalized:
- `influence_diagnostics` — Cook's distance, leverage values
- `dfbetas_dffits` — leave-one-out influence on coefficients and fitted values
- `prediction_intervals` — needs residual variance estimate + hat matrix diagonal

### Could Generalize but Currently Don't
- **`loo_cv_press`**: Just needs refit on n-1 observations. Could accept a `fit_predict` closure (same pattern as conformal prediction).
- **`regression_depth`** / **`regression_depth_logistic`**: Works on FPC scores + predictions. Already has two separate implementations. Could be unified via `FpcPredictor`.
- **`fpc_vif`** / **`fpc_vif_logistic`**: Measures multicollinearity in FPC scores — purely a property of the score matrix, model-independent. Already has a generic version (`generic_vif`) but the dedicated versions are duplicated.

### Recommended Actions
1. Deprecate `fpc_vif` and `fpc_vif_logistic` in favor of `generic_vif` (already exists).
2. Unify `regression_depth` and `regression_depth_logistic` into a `generic_regression_depth` via `FpcPredictor`.
3. Add `generic_loo_cv_press` using a closure-based refit pattern (like `cv_conformal_regression`).
