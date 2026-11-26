# Quick 261126-vzq — alignment bug batch (#77, #78, #79)

## Goal
Fix three reported alignment bugs, each test-first with an independent oracle, one commit per issue.

## Tasks
1. **#77 transfer_alignment warp order** — total warp must be `γ_i∘γ_b` (`compose_warps(within, bridge)`),
   not `γ_b∘γ_i`. Test: non-commuting two-peak example (target = μ₂∘w_±); aligned target curves must
   match the source mean shape, and `reparameterize(target_i, gammas_i) == aligned_i`.
2. **#78 landmark_register zero warps** — inconsistent input returned `FdMatrix::zeros` warps.
   Return identity warps instead (issue's accepted alternative; changing to `Result` would break the
   public signature — out of scope for a bug-fix batch). Same fix in `detect_and_register`.
   Test: every returned warp row is a valid identity warp (monotone, endpoints = domain).
3. **#79 TSRVF ladders** — Schild's/pole ladders were mis-implemented (Schild returned
   `log_to(exp_from v)`, pole returned `-log_to(exp_from(-v))`). Implement the textbook constructions
   (Lorenzi & Pennec 2014). Oracles: closed-form parallel transport on the unit Hilbert sphere
   (pole ladder exact on symmetric spaces; Schild third-order error) and agreement of all three
   TSRVF methods (the transported vector lies on the geodesic, so PT(−log_q μ) = log_μ q exactly).

## Gates
fmt, clippy (--all-targets linalg,parallel,serde), targeted lib tests (default + no-default linalg),
rustdoc -Dwarnings, pmat quality-gate.
