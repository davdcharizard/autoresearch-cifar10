**Prioritized Feedback**
1. **Idea-01’s ReZero safety claim is materially overstated.** The widened `GatedResidual(384)` is identity at init, but `conv_bn(128,384)` and `conv_bn(384,512)` are new random transforms, so the model is not bit-equivalent to the EXP-004 net. See idea-01 lines 12-13, 32-34 vs `train.py` lines 150-151. Address by dropping the “no early disruption / no LR retune” claim, or use a safer width-320 probe / function-preserving widening scheme.

2. **Idea-01’s main risk is probably decisive: ~106 epochs is below the learned annealing comfort zone.** The proposal itself estimates 142→106 epochs, while learnings say most gain arrives in the low-LR tail and EXP-004 already spent 32 epochs for +0.13pp. See idea-01 lines 65-69, 80-92; learnings lines 79-81; EXP-004 lines 19, 23-26. If width is chosen, 320 channels is the more defensible risk-adjusted variant.

3. **Idea-02 likely cannot clear the hard bar alone.** It targets a real eval-side variance source, but the remaining marginal gain is only translate-over-flip, after RandomCrop(pad=4) training and existing flip-TTA. The proposal’s central estimate is only 96.04% and airbench full-TTA is cited around 96.05%, below the 96.10 bar. See idea-02 lines 73-85 and brainstorm lines 52-58. Best used as a composable later add-on, not the primary EXP-005 bet.

4. **Idea-02’s “cannot regress” claim should be softened.** Training cannot destabilize, but logits averaged over reflect-padded shifted crops can still reduce measured accuracy on the test set. See idea-02 lines 46, 71-79. Keep the wall-clock smoke, but also frame it as low training risk, not zero metric risk.

5. **Idea-03’s weakest assumption is that layer3 capacity is the right capacity.** EXP-004 proved layer2 8x8 capacity helped; it did not prove an extra 4x4 semantic block helps. The proposal’s falsification language should not conclude “capacity saturated” if layer3 fails; it may only show late/coarse capacity is ineffective. See EXP-004 lines 23-26, 34-37; idea-03 lines 84-93, 101-103.

6. **Idea-03’s throughput argument is plausible but unverified.** FLOPs are equal across placements, but 4x4/512 conv kernels, optimizer updates, and EMA over +4.7M params may not be cheaper in wall time than the activation-footprint argument predicts. See idea-03 lines 21-31, 55, 86-88. The proposed throughput smoke is mandatory.

7. **No candidate has a hard-constraint violation.** All stay in `train.py`, preserve one `Eval.evaluate` call per epoch, keep seed fixed, and respect the 300s training budget / 600s wall cap as framed in the goal definition lines 21-35 and 41-50.

**Scored Verdict**
- **Idea-01, widen block2 256→384**  
  Evidence/reasoning: **6/10**. Capacity is validated and airbench width is relevant, but the airbench transfer is confounded and the identity-init argument does not cover the widened path.  
  Potential impact: **8/10**. Highest upside if the wider mid-level basis trains enough, but the ~106-epoch estimate makes the upside hard to realize.

- **Idea-02, multi-crop TTA**  
  Evidence/reasoning: **7/10**. Mechanism is clean, allowed by prior flip-TTA success, and matches airbench TTA, but the marginal translate-only evidence is weak.  
  Potential impact: **4/10**. Most likely gain is real but below +0.10pp; central estimate misses the bar.

- **Idea-03, second ReZero block in layer3**  
  Evidence/reasoning: **8/10**. Directly extends the just-validated ReZero capacity lever, keeps the cleanest single-variable implementation, and has a credible throughput advantage, pending smoke.  
  Potential impact: **7/10**. Diminishing returns are real, but it has enough upside to clear 96.10 without the severe under-annealing risk of 384 width.

**Pick: Idea-03.** It is the strongest single EXP-005 bet because it attacks the validated limiter with the least disruptive capacity add: one ReZero-gated block, no architecture-wide width shock, and likely more annealing budget than idea-01. Idea-02 is safer but too low-ceiling for the hard bar; idea-01 has more raw upside but its core safety premise is weaker and its epoch loss is too large near the airbench96 ceiling.
