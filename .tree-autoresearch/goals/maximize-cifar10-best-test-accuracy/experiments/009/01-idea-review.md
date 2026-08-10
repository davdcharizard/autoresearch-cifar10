# Claude Adversarial Idea Review: EXP-009

## Prioritized Feedback

1. **All identity-centered BF16 gates have a forward dead zone near one.** Require a separate 200-step GPU-0 live-gate smoke with `max|gate-1| > 0.02`, and use a fixed higher-LR, zero-weight-decay gate group so the experiment tests an active mechanism.
2. **Define descriptor normalization precisely.** Per-sample standardization across the channel descriptor removes global-activation-scale drift and makes gate logits comparable through training.
3. **Descriptor and gated channel identities must align.** For ECA especially, deriving a descriptor before `conv2` and gating after channel mixing destroys its self-gating interpretation. Gate the same tensor used for the descriptor.
4. **SE must not reroll parent initialization.** Standard `nn.Linear` construction consumes global RNG before the classifier. Use raw parameters initialized from a dedicated CPU generator and keep the parent's global initialization stream bitwise identical.
5. **Use an EXP-002-relative latency gate.** The inherited EXP-004 SAM-weighted gate is wrong for this no-SAM base. Compare with the parent under one harness, allow at most 1.10x latency, and project at least 26,000 steps.
6. **Late-stage ECA's semantic-stage rationale is unmeasured.** It reduces coverage mainly to save launches before cost is known, while the goal needs a large effect.
7. **ECA capacity may be too small for this fork.** A shared 3-5-tap kernel assumes useful channel adjacency and offers only 20-26 parameters. Cost is not the binding resource here.
8. **SE reduction 16 under-sizes 128-channel gates.** Use a fixed hidden-width floor of 16 units.
9. **Keep optimizer separation exact.** Only new gate parameters receive zero decay and the LR multiplier; all parent parameters, including BatchNorm affine tensors, retain their exact optimizer settings.
10. **CutMix may teach diffuse gates.** Mixed composites occupy half the early batches, and EXP-002 has no SAM tail. This is the leading null mechanism and must be stated without tuning around it.

## Scored Verdict

### Late-Stage Identity-Centered SE
- **Evidence and reasoning: 7/10.** Canonical channel recalibration without a locality assumption; fixable RNG and bottleneck issues.
- **Potential impact: 8/10.** Only finalist with enough cross-channel capacity to plausibly clear the 0.30-point practical bar.

### All-Block Identity-Centered ECA
- **Evidence and reasoning: 6/10.** Clean measurement design, but descriptor placement needs correction and the parent-specific latency gate was wrong.
- **Potential impact: 5/10.** Full coverage helps, but 26 shared-band parameters impose a low ceiling on arbitrary learned channel order.

### Late-Stage Identity-Centered ECA
- **Evidence and reasoning: 4/10.** Strict subset of all-block ECA justified mainly by unmeasured launch concerns.
- **Potential impact: 3/10.** Twenty parameters and reduced coverage are weakly matched to a 0.30-point target.

## Pick

**Late-Stage Identity-Centered SE**, conditional on dedicated RNG-isolated raw-parameter initialization, per-sample descriptor standardization, hidden width 16, a fixed 5x no-decay gate group, live-gate verification, and a parent-relative latency gate.
