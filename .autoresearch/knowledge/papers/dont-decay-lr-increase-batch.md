# Don't Decay the Learning Rate, Increase the Batch Size (Smith, Kindermans, Ying, Le)

- **arXiv**: 1711.00489 (ICLR 2018)
- **Status**: TESTED AND REFUTED for this project's regime (EXP-059, no-improvement)

## Paper claim

SGD gradient-noise scale g ≈ ε·N/B (LR × dataset size / batch). Decaying the LR and
increasing the batch size are equivalent noise-reduction moves — so one can replace
late-phase LR decay with batch-size increases at fixed LR, reaching the same test
accuracy in fewer parameter updates (ImageNet ResNet-50, Inception; also CIFAR Wide-ResNets).

## Project test (EXP-059)

Late batch step 512 → 1024 at p ≥ 0.75 of the charged budget, LR schedule unchanged
(stacks an extra implicit anneal on the cosine tail rather than substituting). Switch
implemented unchargeably in the fetch path (paired loader batches + concat + pin before t0).
Both mechanisms delivered: tail noise halved, plus ~6% per-image throughput dividend at 1024
(+2 equivalent epochs, 142 ep / 11,933 steps). Read: 96.51 = family mean−0.4σ — exact null.

## Why it does not transfer here

- The one-cycle cosine at depth-20/heavy-aug already saturates late-phase noise reduction;
  an extra tail noise halving is REDUNDANT with the explicit anneal (no residual sharpening
  for it to do). Consistent with the level-closure: 1024-from-start lost at both canonical
  LR scalings (EXP-012 linear, EXP-022 √).
- The paper's benefit currency is "fewer parameter updates at equal accuracy" — worthless
  under a fixed wall-clock budget where the schedule is time-keyed and updates are not the
  scarce resource.
- Pre-registered inheritance (brainstorm-059): the multi-step ramp (512→768→1024) and the
  noise-UP mirror (512→256 tail) inherit this null — same noise trajectory class, no
  separate mechanism.

## Reusable engineering (independent of the null)

- Multi-shape compiled training REQUIRES `torch.compile(model, dynamic=False)` + 3-iter
  uncharged warmup per shape; default automatic-dynamic merges shapes into one graph
  ~18% slower at every shape (see infra-errors.md EXP-059).
- Two-shape GPU probe with a P512-in-family-band anchor check catches such silent
  recompile pathologies pre-launch; probes gate at host load < 40.
