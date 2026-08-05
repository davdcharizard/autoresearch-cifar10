**Prioritized Feedback**

1. **Idea-03 has a major schedule-mechanics error.** With `train.py:264-268`, old decay is `0.4/0.85 = 0.4706 * (1-p)`, while proposed decay is `0.45/0.90 = 0.5 * (1-p)`. That means the new LR is higher through the decay, not lower, and it reaches low-LR thresholds later. The proposal’s “more low-LR tail” claim (`idea-03.md:12`, `:54`) is wrong. Fix by testing either shorter warmup at `PEAK_LR=0.4`, or a peak-LR raise framed as exploration, not tail reallocation.

2. **Idea-01 over-imports airbench’s 0.78 LR factor.** airbench96 changes architecture, widths, activation style, cutout, warmup, TTA, and optimizer details together; the raw source confirms `block2=384`, GELU ConvGroups, `tta_level=2`, cutout 12, and 10% warmup. Applying `0.40 -> 0.31` to only one new `Residual(256)` (`idea-01.md:35-39`) may under-step the current DavidNet. Consider a milder LR like `0.35-0.36`, or be explicit that this is a confounded capacity+LR point.

3. **Idea-01’s real failure mode is under-annealing.** Prior learnings say most accuracy arrives in the low-LR tail (`03-experiment-learnings.md:76-78`), and EXP-003 peaked late at ep162/174. Adding depth cuts epochs and lowering LR slows optimization, so a throughput smoke only checks wall time, not convergence. If selected, monitor trajectory against EXP-003 and consider identity/zero-init for the new residual branch.

4. **Idea-02 is valid but has no flip-only safety floor.** It replaces the current flip-TTA logits (`train.py:159-164`) with a 6-pass blend; because the loop can only call `evaluator.evaluate` once per epoch (`01-definition.md:32`, `train.py:327`), it cannot score both flip-only and multi-crop. If translate crops hurt, best accuracy can fall despite unchanged training. A more conservative translate weight would reduce this risk.

5. **Idea-02’s wall-clock estimate needs a real forced-TTA timing check.** Eval is outside the 300s training budget but inside the 600s wall cap (`01-definition.md:43-45`); `prepare.py:32-43` runs fp32 inference, and padded/sliced crops may be slower than the proposal assumes. Not fatal, but time one `model.tta=True` eval before committing.

6. **Idea-01 has a spec inconsistency.** The title says “extra conv to the layer2/3 stems” (`idea-01.md:1`), but the concrete edit only adds `Residual(256)` (`idea-01.md:24-31`). Fix the proposal text so implementation is unambiguous.

**Scored Verdict**

- **Idea-01: Evidence/reasoning 7/10.** Best match to an untried diagnosed limiter, with airbench support for capacity scaling, but the exact LR transfer is weak and confounded. **Potential impact 8/10.** Highest ceiling; it is the only idea plausibly moving beyond tiny eval/schedule noise.

- **Idea-02: Evidence/reasoning 8/10.** Strong local precedent from flip-TTA and exact airbench implementation. **Potential impact 5/10.** Likely positive but marginal after flip-TTA and random-crop training; may miss the +0.1pp bar.

- **Idea-03: Evidence/reasoning 4/10.** EXP-003 supports LR retuning generally, but the proposed warmup+peak mechanism is mathematically wrong. **Potential impact 4/10.** Cheap, but likely small and noisy.

**Pick: Idea-01.** It has the best upside and attacks the least-spent limiter, representational capacity. Idea-02 is cleaner but probably too small; Idea-03 should not run as written.

External checks used: airbench paper https://arxiv.org/abs/2404.00498 and airbench96 source https://raw.githubusercontent.com/KellerJordan/cifar10-airbench/master/legacy/airbench96.py.
