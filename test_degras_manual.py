import json
import math

# Load data and expected values
with open('validation/data/standard_50x101.json') as f:
    d = json.load(f)

with open('validation/expected/tolerance_expected.json') as f:
    exp = json.load(f)

n = d['n']
m = d['m']
print(f"n = {n}, m = {m}, sqrt(n) = {math.sqrt(n)}")
print(f"R critical value: {exp['degras_critical_value']}")

# Reconstitute data matrix
data = d['data']
mat = [data[i*m:(i+1)*m] for i in range(n)]

# Compute raw mean
raw_mean = [sum(mat[i][j] for i in range(n)) / n for j in range(m)]
print(f"\nRaw mean (first 5): {raw_mean[:5]}")

# Compute smoothed mean (from expected)
smooth_mean = exp['degras_center']
print(f"R smooth mean (first 5): {smooth_mean[:5]}")

# Compute residuals and sigma at first few points
for j in [0, 1, 2, 3, 4]:
    residuals = [mat[i][j] - smooth_mean[j] for i in range(n)]
    # Two ways to compute sigma: dividing by n (biased) vs n-1 (unbiased)
    var_biased = sum(r**2 for r in residuals) / n
    var_unbiased = sum(r**2 for r in residuals) / (n - 1)
    sigma_biased = math.sqrt(var_biased)
    sigma_unbiased = math.sqrt(var_unbiased)
    
    print(f"\nPoint {j}:")
    print(f"  R smooth mean: {smooth_mean[j]:.6f}")
    print(f"  Sigma (div by n): {sigma_biased:.6f}")
    print(f"  Sigma (div by n-1): {sigma_unbiased:.6f}")
    # Expected half-width from formula: c * sigma / sqrt(n)
    hw_biased = exp['degras_critical_value'] * sigma_biased / math.sqrt(n)
    hw_unbiased = exp['degras_critical_value'] * sigma_unbiased / math.sqrt(n)
    print(f"  Expected hw (div by n): {hw_biased:.6f}")
    print(f"  Expected hw (div by n-1): {hw_unbiased:.6f}")

# Check what the R code does: colMeans((mat - smoothed_mean)^2)
# This is: mean of residuals^2 = sum(r^2) / n
print("\n\n=== R Code Analysis ===")
print("R code: sigma_hat <- sqrt(colMeans((mat - matrix(smoothed_mean, ...))^2))")
print("This divides by n (biased estimator), NOT n-1")

# So expected hw = c * sigma(div by n) / sqrt(n)
hw_expected = exp['degras_critical_value'] * math.sqrt(sum((mat[i][0] - smooth_mean[0])**2 for i in range(n)) / n) / math.sqrt(n)
print(f"\nExpected half-width at j=0: {hw_expected:.6f}")
print(f"This is: {exp['degras_critical_value']:.6f} * {math.sqrt(sum((mat[i][0] - smooth_mean[0])**2 for i in range(n)) / n):.6f} / {math.sqrt(n):.6f}")
