# Claude Adversarial Raw-Result Review: EXP-013

- **Reviewer**: Claude Opus via Claude Code 2.1.220
- **Tools**: read-only; no fallback model configured
- **Initial verdict**: BLOCK
- **Final verdict**: PASS

## Blocking Finding And Resolution

- The execution record initially pointed the pre-measurement Boolean-assertion traceback at the successful preflight log. The source now correctly names `/tmp/exp013_preflight_attempt1.log`; Claude verified that it contains only the claimed traceback and no completed gate measurement.

## Confirmed Result

- Formal threshold is `95.71%` from parent `95.61%`; observed best `95.11%` is a valid no-improvement with delta `-0.50` points.
- Exit `0`, complete summary, `300.0s` charged training, `457.2s` total, 132/132 once-per-epoch evaluations, and all restore/coverage/nonfinite/RNG audits pass.
- The run achieved full preregistered dose with `25598` steps and `158` EMA updates, but the final-16 EMA mean was only `95.073750%` with range `0.10`, failing the `95.64%` mechanism target.
- Claude approved wording that this single fixed-seed historical-parent comparison is consistent with the fixed-scale-40 cosine-normalized package being harmful here, while rejecting statistical-significance, scale-specific, general-transfer, or early-trace causal claims.

## Final Report Review

- **Initial verdict**: BLOCK. Claude rejected an unsupported scale-sensitivity statement, an unjustified confidence upgrade for EMA horizon tuning, and a later ambiguity between whole-model EMA-relative distance and normalized classifier-direction distance.
- **Resolution**: the report now states that only scale 40 was tested once, keeps shorter-lag EMA at low confidence without a numeric distance comparison, and explicitly scopes absolute classifier-direction geometry, the `0.8%` exposure deficit, and single-run causal limits.
- **Final verdict**: PASS. Claude recomputed the delta, tail gap, dose floors, and EMA balance and found no remaining evidence, arithmetic, classification, or wording blocker.
