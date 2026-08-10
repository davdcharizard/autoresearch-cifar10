# Adversarial Implementation Review Addendum — EXP-014

**Reviewer**: Claude plan critic
**Verdict**: **REJECT** the proposed production hook/boolean refinement; approve the clean reviewed production implementation with a disposable preflight-only proof variant.

## Finding

The proposed detach-until-step-two mechanism is mathematically faithful, but it does not belong in production `train.py`. The deterministic CUDA failure is preflight-only. A persistent tensor hook and mutable forward branch would add avoidable callback/synchronization risk to a launch-bound experiment and would leave verification scaffolding in the successful artifact.

Default CUDA convolution backward is not bitwise repeatable even for two identical control models. Strict deterministic mode additionally rejects `adaptive_max_pool2d` backward. Therefore the first-step causal proof must separate analytic and empirical claims rather than change production behavior.

## Required Resolution

1. Keep production `train.py` as originally reviewed: `fc(avg_features) + max_fc(max_features)`, with no hook, boolean, conditional detach, or deterministic-mode toggle.
2. In the external preflight harness, use a disposable candidate subclass whose first-step max pool consumes `out.detach()`. Because `max_fc.weight` is exactly zero, detached and clean production variants have analytically identical first-step feature gradients.
3. Use a fresh deterministic control/candidate pair for each hard-target and probability-target first-backward check. Require bitwise-equal logits, loss, and all accepted gradients, plus finite nonzero `max_fc.weight.grad`.
4. Verify the clean production candidate's post-update max-path representation gradient separately in normal mode. This is a finite/nonzero engagement observation, not a bitwise deterministic claim.
5. Explicitly confirm the accepted average-pool/backbone backward is deterministic and control-matching once the max-pool input-gradient edge is absent.

## Response

All required changes are accepted. No production hook was added, so the tracked implementation remains the clean reviewed design. The corrected preflight uses the disposable detached subclass only for the strict first-step proof and exercises the real production forward separately for max-path engagement.
