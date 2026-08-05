**Prioritized Feedback**

1. **Idea-01 is the only finalist that attacks the stated open question, but it is not truly a different backbone.**  
   It adds one more `GatedResidual(256)` inside the current DavidNet `layer2`, so it tests “more depth at the proven 8x8 stage,” not a wholesale structural break from DavidNet. That is still defensible because EXP-004 is the only architecture win and airbench’s cited 95→96 step is depth, but the proposal should not oversell it as a different backbone. Refinement: frame it as the minimal depth probe before spending a loop on a full pre-activation/deeper backbone.

2. **Idea-01’s feasibility hinges on epoch count more than the proposal admits.**  
   EXP-014 compile bought +12% and EXP-014’s 320-width cell still only reached 143 epochs. A second 256-channel two-conv block may cost more than the reclaimed compile headroom, and EXP-005/007/013/016 show under-anneal is the dominant failure mechanism. Refinement: make `num_epochs >=135` a hard pre-run gate; if it drops below that, abort or shrink to a single extra conv.

3. **Idea-01 has the strongest evidence, but the evidence is indirect.**  
   EXP-004 showed one layer2 ReZero block helped +0.13pp, and airbench supports depth, but EXP-014’s “capacity saturation” result cuts against “more of same stage capacity” unless depth is materially different from width. This is plausible, not proven. Refinement: require paired same-session control plus confirmation pair because low-control-draw artifacts have recurred in EXP-016/017/019/020.

4. **Idea-02 is clean and low-risk, but its likely upside is below the benchmark’s noise floor.**  
   Alternating flip has good external support from airbench and no constraint issue, but the goal file and learnings put the noise floor around 0.1pp. Since this only derandomizes one already-present augmentation while RandomCrop/Cutout/RandomErasing remain stochastic, it may be real but unmeasurable. Refinement: run it only after higher-upside structural probes, or combine later with a winning deeper backbone.

5. **Idea-02 has an implementation trap with shuffled persistent workers.**  
   The proposal correctly notes `persistent_workers=True` and shuffle. Per-epoch global flip parity is weaker than true per-sample alternating flip and could create epoch-level distribution oscillation. Refinement: wrap the dataset to return `(image, target, idx)` and compute flip from `(epoch + idx) % 2`; avoid relying on worker-local counters.

6. **Idea-03 is feasible but weakly connected to the diagnosed limiter.**  
   AdaptiveConcatPool changes the readout, not the backbone’s nonlinear depth or function class. The claim that avg+max “strictly dominates” is only representational; optimization and regularization can still worsen. EXP-019’s SE null is not identical, but it is a warning that head-side/channel-stat additions are likely redundant. Refinement: keep as a cheap cleanup experiment, not the next lead.

7. **No finalist is a reward hack or hard-constraint violation if implemented as described.**  
   All stay within `train.py`, keep one eval per epoch, avoid seed search, and do not touch `prepare.py`. The main integrity risk is Idea-01’s compile warmup: BN buffers, RNG isolation, and eval using the uncompiled EMA path must follow the EXP-014 validated recipe.

**Scored Verdict**

| Idea | Potential Impact | Success Likelihood | Evidence / Reasoning |
|---|---:|---:|---|
| Idea-01: compile-funded extra layer2 ReZero depth | **7/10** | **5/10** | Highest upside because it tests the only remaining structural crack: depth at 8x8, backed by EXP-004 and airbench. Likelihood is only moderate because it is still DavidNet-internal and may under-anneal or hit the same ceiling. |
| Idea-02: alternating horizontal flip | **3/10** | **6/10** | Strong external rationale and low feasibility risk, but expected effect is probably at or below the 0.1pp noise/improvement threshold on this saturated recipe. |
| Idea-03: AdaptiveConcatPool head | **4/10** | **4/10** | Cheap and untried, but weakly targets the diagnosed limiter; likely redundant with existing 512-channel feature quality and prior channel/readout-side nulls. |

**Pick: Idea-01, compile-funded DEPTH at the proven 8x8 stage.**

It wins because it is the only finalist with a credible path to a >0.1pp gain by changing the model’s nonlinear capacity, not just sampling variance or readout statistics. The pick should be narrowed: run it as a disciplined depth test with EXP-014 compile machinery, hard epoch gating, same-session control, and confirmation pair. If this ties, the evidence strongly says the next move must be a genuinely different backbone, not more DavidNet micro-levers.
