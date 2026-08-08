---
status: testing
phase: 06-conditional-svd-library-comparison
source: [06-VERIFICATION.md]
started: 2026-08-08T22:09:50Z
updated: 2026-08-08T22:09:50Z
---

## Current Test

number: 1
name: nalgebra-vs-faer table numbers match the on-disk p6_svd_* artifacts (±5%, run1 + run2)
expected: |
  Every cell median in the AUDIT-REPORT Phase 6 table matches the criterion median
  (middle bracket value) from the corresponding run artifact to within ±5%, for both
  run1 and run2 columns. Verifier already confirmed run1 medians match exactly (0%
  deviation); human sign-off is retained for run2 per the PLAN's explicit backstop.
awaiting: user response

## Tests

### 1. Table number accuracy (±5%, run1 + run2)
expected: Every cell median in the AUDIT-REPORT table matches the criterion median from the corresponding run artifact within ±5%. Spot-check N=500,M=200: nalgebra=41.026 ms, faer(seq)=23.084 ms (both exact matches to the artifact middle bracket).
result: [pending]

### 2. M=500 crossover observation is correct
expected: At N=500,M=500 faer(seq) beats nalgebra in both runs (run1 189.70 ms vs 358.31 ms ≈ 1.9×; run2 114.98 ms vs 324.62 ms ≈ 2.8×). The report states faer wins at every cell including the M=500 probe — no crossover back to nalgebra was observed. Human judgment on whether the tested cells sufficiently characterize the crossover boundary.
result: [pending]

### 3. Backlog P6-1 severity set from MEASURED speedup
expected: The P6-1 Severity+Effort field (P2/S) is justified by the measured 1.8× at the primary cell N=500,M=200, NOT the assumed 3–10× range from RESEARCH, and states an explicit downgrade condition. Human judgment on whether P2 (vs P3) is appropriate at 1.8×.
result: [pending]

## Summary

total: 3
passed: 0
issues: 0
pending: 3
skipped: 0
blocked: 0

## Gaps
