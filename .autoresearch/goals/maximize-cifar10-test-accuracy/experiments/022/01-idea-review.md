**Prioritized Feedback**

1. **Do not trust matched-epoch literature numbers without a budget-size proof.** WRN’s 97.1% with cutout is strong evidence, but it is for WRN-28-10, not necessarily WRN-16-4/22-4/16-8 under 300s. Fix: make the first experiment a size-gated WRN sweep and pick by measured `num_epochs >= 130`, not by parameter count or paper accuracy.

2. **Under-anneal is the decisive risk for all three.** The history says costlier changes fail when they cut the low-LR tail: EXP-007 hit only 94 epochs and was still climbing; EXP-013 lost 18-26 epochs to SAM; EXP-016 lost fused-kernel throughput. Any candidate below ~110 epochs is a bad test of backbone quality. Healthy target is 130-173 epochs, per `03-experiment-learnings.md`.

3. **A: WRN is the cleanest ceiling attack, but only if “budget-sized” is enforced brutally.** It directly addresses the documented DavidNet ceiling from EXP-014/021 and has the best external ceiling evidence. Hidden assumption: the smaller WRN that fits 300s keeps enough of WRN-28-10’s ceiling to clear 96.48. Fix: pre-smoke WRN-16-4, WRN-22-4, WRN-16-8; do not run the largest if it projects below 130 epochs.

4. **A: Do not assume the DavidNet recipe gains stack additively.** EMA/TTA and whitening are called backbone-agnostic in the brainstorm, but their measured gains came on DavidNet. Whitening especially changes the WRN stem input shape if retained, and GAP removes `scale_out`. Fix: keep EMA/TTA/aug, but treat whitening/LR/scale removal as confounds; run a same-session DavidNet compiled control and consider a minimal WRN LR sanity check if early trajectory is broken.

5. **B: Resolution-preserving pre-act ResNet is too close to the exhausted search space.** Its mechanism is plausible, but EXP-014 closed width and EXP-021 closed depth at the useful 8x8 stage when fully annealed. “More 16x16 capacity” may just be another capacity-placement variant, not a wholesale backbone. Fix: only pursue B if implemented as a genuinely new stage topology, not a DavidNet patch.

6. **B has severe under-anneal risk despite sounding lighter.** 16x16 conv capacity is expensive; adding blocks before downsampling costs more than the already-failed 8x8 additions. If it fits fewer than WRN while offering lower ceiling evidence, it is dominated. Fix: cap channels/depth aggressively and pre-smoke; reject if it cannot match WRN’s epoch count.

7. **C is novel but weakly evidenced for the target threshold.** The brainstorm itself says ConvMixer reaches around 96% and often needs longer training/heavy aug. That is below the required 96.48 bar and below WRN evidence. Fix: keep as a diversification experiment after WRN fails, or only run if a smoke test shows unexpectedly high throughput and healthy early accuracy.

8. **C’s under-anneal risk is not just model size; it is kernel efficiency.** Large-kernel depthwise convs can be slow on this H20 setup, similar in spirit to EXP-016’s fused-kernel loss and EXP-005’s “FLOP-equal != wall-clock-equal” warning. If depthwise kernels force tiny hidden dim or low epochs, the model loses both capacity and anneal. Fix: benchmark actual compiled steps for k=5/7/9 and hidden dim before committing.

**Scores**

- **A: Budget-sized pre-activation WRN**
  - Evidence/reasoning: **9/10**. Best literature support, matches the mandated “different backbone,” and directly attacks the EXP-014/021 ceiling diagnosis.
  - Potential impact: **8/10**. Published ceiling is comfortably above 96.48, but the budget-sized variant may not inherit all of it.

- **B: Resolution-preserving pre-activation ResNet**
  - Evidence/reasoning: **6/10**. Mechanism is coherent, but it leans heavily on capacity-placement logic that prior fully annealed DavidNet tests have mostly exhausted.
  - Potential impact: **5/10**. Could clear the bar if DavidNet’s ceiling is specifically early pooling, but likely ties if the limiter is broader ResNet-9/DavidNet topology.

- **C: ConvMixer isotropic backbone**
  - Evidence/reasoning: **4/10**. Mechanistically distinct, but cited accuracy is only around 96% and budget behavior is unknown.
  - Potential impact: **6/10**. Highest novelty, but high chance of sub-baseline or under-annealed result within 300s.

**Pick**

Pick **A: budget-sized pre-activation WRN** as the highest-EV first bet. It is the only candidate with both a documented ceiling above the required 96.48 and a direct match to the diagnosed need: leave the exhausted DavidNet family. The experiment should be won or lost on size selection: run the largest WRN variant that still lands in the healthy anneal band, with a smaller fallback, and reject any result below ~110 epochs as an under-anneal artifact rather than a backbone verdict.
