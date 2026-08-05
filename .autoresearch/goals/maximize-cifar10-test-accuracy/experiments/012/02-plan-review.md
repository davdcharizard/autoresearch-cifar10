# EXP-012 Adversarial Plan Review

Offline local `plan-critic` fallback review, 2026-07-24.

## Prioritized Concerns

1. **Critical - RNG preservation**: registering refinement after final BN/classifier is insufficient because module constructors consume RNG before `self.apply`. Use explicit two-phase initialization or RNG isolation, or accepted initialization values will shift.
2. **High - scored training RNG**: even if accepted weights match, unisolated refinement construction/initialization advances the global CPU RNG and changes DataLoader shuffle, worker seeds, and augmentations. Require exact post-construction RNG equality against no-refinement construction.
3. **High - preflight invocation**: the mandatory complex preflight has no named runnable implementation. Define a reproducible command/script while retaining `train.py` as the only project source modification.

## Disposition

All concerns are valid. The plan now initializes the complete accepted model exactly as before, then constructs and initializes the refinement inside `torch.random.fork_rng(devices=[])`, which restores the accepted post-init CPU RNG state. Static tests require byte-identical accepted tensors and exact post-construction RNG equality. The preflight is a named experiment artifact, `experiments/012/preflight.py`, run by exact command and not imported by production; only `train.py` remains in the git/source diff.
