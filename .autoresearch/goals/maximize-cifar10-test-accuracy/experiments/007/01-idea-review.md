**Prioritized Feedback**

**Candidate 1: Widen layer2 256→384**

1. Main risk is under-modeled throughput loss versus the fixed 300s budget. In `train.py:150-151`, this makes the two `GatedResidual` 8x8 convs 2.25x heavier and the `layer3` stem 1.5x heavier. Since learnings say most gain arrives in the low-LR tail and the loop is time-budgeted, fewer updates can erase capacity gains. Address by treating `num_epochs/img_per_sec` as first-class evidence; if epochs collapse below the expected ~115-130 range, the 320-width fallback is the right refinement.

2. The airbench96 support is confounded. Brainstorm cites airbench’s 96.03% net as wider, GELU, cutout12, different whitening, and 3-conv groups; the absolute delta over current 96.00 is only 0.03pp, below the documented ~0.1pp noise floor. Use airbench mainly to justify the 384 target width, not as proof that this single change has >0.1pp headroom.

3. “No LR retune” is plausible but not proven. Unlike adding a ReZero block, widening changes the main path at `layer2[0]` and `layer3[0]`; Kaiming+BN makes this likely stable, but not identity-preserving. Keep LR fixed for attribution, but judge early trajectory for instability or slow fit.

4. Execution is clean enough: only `train.py` changes, no eval hack, no dependency issue. Shape ripple is simple and well specified.

**Candidate 2: GELU + Cutout12**

1. This does not attack the diagnosed limiter. Diagnosis says capacity is binding and the +0.1pp bar sits at the noise floor; GELU/cutout is a recipe-polish bet. Airbench’s total advantage over current best is only 0.03pp and is confounded by architecture. Standalone EXP-007 upside is probably sub-noise.

2. “Throughput-free” is overstated. Cutout size is effectively free, but GELU is not literally free versus `nn.ReLU(inplace=True)` in `conv_bn` (`train.py:101-106`). The cost may be small, but the proposal should verify epoch count rather than assume identical throughput.

3. The recipe-alignment claim is not exact. Brainstorm describes airbench blocks as Conv→MaxPool→BN→GELU, while current `conv_bn` is Conv→BN→activation and pooling is outside the helper. This is activation/regularization alignment, not a close airbench port.

4. Combining GELU and Cutout12 improves chance of clearing noise but sacrifices attribution and introduces cancellation risk: stronger cutout may underfit the current smaller net. Better used later as a rider on a capacity win.

**Candidate 3: Second ReZero block at layer2**

1. The key assumption is diminishing returns will not dominate. EXP-004’s first layer2 ReZero block gained +0.13pp, barely above the ~0.1pp noise floor; a second similar block plausibly has lower marginal value while still costing epochs. This is the central weakness.

2. Capacity-vs-epochs looks less favorable than Idea 1. It adds two more serial 8x8/256 convs, likely reducing epochs to ~125-135, but its ceiling is only “more of the same” rather than a wider representation. The execution is cleaner, but the expected headroom is thinner.

3. The “identity-equivalent” cleanliness claim is overclaimed. The new block is identity at forward init, but adding modules before `layer3` changes RNG consumption during `self.apply` (`train.py:155`), so downstream initial weights differ from EXP-004. Not fatal, but attribution is not bit-clean.

4. The required two-step ReZero smoke is correctly identified and should be mandatory if this is run.

**Scored Verdict**

| Idea | Evidence / Reasoning | Potential Impact |
|---|---:|---:|
| Idea 1: Widen layer2 | 8/10 — Best alignment with the diagnosed capacity limiter, backed by EXP-004’s positive layer2 capacity result and airbench’s 384-width precedent, though airbench evidence is confounded. | 8/10 — Largest plausible chance to clear >0.1pp; real risk is losing too many epochs under 300s. |
| Idea 2: GELU + Cutout12 | 4/10 — Uses real fast-CIFAR precedent, but the mechanism is weakly tied to the stated limiter and the airbench delta is below noise. | 3/10 — Cheap, but likely sub-0.1pp standalone; better as a later additive rider. |
| Idea 3: Second layer2 ReZero | 7/10 — Strongest direct prior lineage from EXP-004 and clean implementation, but extrapolates from first block to second despite obvious diminishing returns. | 5/10 — Could work, but likely smaller upside than widening and still pays a meaningful epoch cost. |

**Pick: Idea 1, widen layer2 256→384.**

It is the only candidate with a credible path to clearly beating the ~0.1pp noise floor while directly attacking the diagnosed capacity bottleneck. Idea 3 is cleaner but has weaker marginal headroom; Idea 2 is too likely to disappear into measurement noise. Run Idea 1 as EXP-007, but make epoch count and throughput part of the verdict, because under-annealing is the main way it fails.
