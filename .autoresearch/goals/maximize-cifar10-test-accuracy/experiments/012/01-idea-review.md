**Prioritized Feedback**

1. **Idea 1 overstates the LS-retune case.** EXP-011 already tested `LABEL_SMOOTHING=0.1` under CutMix and it improved ep25 but lowered the ceiling, 96.32 vs 96.40; learnings call the annealed optimum “LS-insensitive” in that range. This is confounded by CutMix, so not fatal, but make WD-shaping the primary test and add an LS0.1-only cell before claiming target-space staleness.

2. **Idea 3’s epoch target is too forgiving.** The learnings say capacity that cuts epochs under-anneals, and EXP-005 already lost at 131 epochs while EXP-007’s 256→384 fell to 94 and finished still climbing. A 256→320 widen is materially different, but “120-135 epochs” may still be below the safe low-LR-tail band. Pre-register `num_epochs` as the first decision metric and treat <135 as likely under-anneal, not a fair capacity verdict.

3. **Idea 2’s cosine mechanism is partly misstated.** With the same `PCT_START`, cosine decay is above linear for the first half of the tail and below it only late; it does not simply create “more low-LR time.” If the intended limiter is low-LR settling, the stronger test is cosine plus `PCT_START=0.10` or an explicit longer low-LR tail, while keeping `EMA_WARMUP_FRAC` aligned.

4. **Idea 1’s ReZero-alpha decay argument needs instrumentation.** `GatedResidual.alpha` starts at zero, so WD only matters after alpha grows; the proposal cites alpha gradients but not actual alpha magnitude or decay/data-gradient ratio during training. Log final alpha and preferably compare WD-shaping-only vs baseline before attributing any gain to alpha decoupling.

5. **All three sit near the 0.1pp noise floor; same-session controls matter.** The protocol notes fixed seed still varies with host throughput/epoch count. Scalar and schedule tweaks especially need a same-session baseline or a pre-registered multi-cell ranking; stored 96.38 alone is too weak for a +0.05-0.10pp read.

6. **No fatal constraint issue found.** All finalists can stay inside `train.py`, use existing dependencies, keep one eval per epoch, and avoid seed manipulation. The main risks are weak attribution and under-anneal, not rule violations.

**Scored Verdict**

- **Idea 1, Recipe-scalar refresh:** Evidence/reasoning **7/10** — WD-shaping is code-specific and well aligned with the regularization-allocation diagnosis, but the LS half is weakened by EXP-011. Potential impact **5.5/10** — plausible +0.1-0.2pp bundle, but individual knobs are likely sub-noise.

- **Idea 2, One-cycle cosine reshape:** Evidence/reasoning **6/10** — schedule is stale and tail importance is real, but the cosine/low-LR mechanism is only partly aligned. Potential impact **4.5/10** — clean and cheap, but likely a few hundredths unless pct/peak riders hit.

- **Idea 3, Mild layer2 widen 256→320:** Evidence/reasoning **5.5/10** — EXP-004 supports capacity at layer2 and EXP-007 does not fully refute width, but repeated capacity under-anneal is the dominant local evidence. Potential impact **7/10** — highest ceiling if it preserves enough epochs, but the epoch-cost assumption is unproven.

**Pick: Idea 1, Recipe-scalar refresh, with WD-shaping as the headline.** It best matches the current diagnosis after CutMix saturation and Muon parity: a throughput-free, non-input-augmentation regularization allocation change with a net-specific ReZero-alpha reason. Run WD-shaping-only first or include an LS-only control; do not let the weaker LS-retune claim carry the experiment.
