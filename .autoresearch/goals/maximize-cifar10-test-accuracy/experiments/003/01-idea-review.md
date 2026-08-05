Prioritized feedback:

1. **Idea-04 Muon has the biggest correctness/attribution gap.** The brainstorm says “wd unchanged,” but `idea-04.md §3a/3b` recommends no weight decay on Muon conv params while keeping WD only on BN/head SGD params. That is optimizer plus regularization change, not a clean Muon test, and could easily dominate the result. Fix: either implement decoupled WD for Muon params or explicitly reframe the idea as “Muon + conv WD removal” and own the confound.

2. **Idea-03 width fights the strongest prior learning.** `03-experiment-learnings.md` says most accuracy arrives in the low-LR tail, and EXP-002 still fit 183 epochs. `idea-03.md` estimates 1.25x width drops to ~115 epochs. That spends the validated lever, update count, to buy an unverified capacity lever. Fix: choose a smaller multiplier, late-stage-only widening, or require evidence that train/test loss is capacity-bound before making width the EXP-003 bet.

3. **Idea-04’s evidence does not support its concrete port.** It cites airbench Muon, but then drops airbench’s weight renorm, uses a hand-chosen `MUON_LR=0.05`, eager Newton-Schulz, different WD, no whitening, and a 183-epoch recipe. The cited mechanism is real, but the chosen scaling is mostly unvalidated. Fix: if run later, start from a closer reference port or first run a tiny LR/NaN smoke that does not become adaptive metric tuning.

4. **Idea-01 whitening is mechanically the cleanest, but its upside is likely compressed.** `idea-01.md` correctly notes airbench’s whitening gain is mostly an epochs-to-94% accelerator, while this benchmark already fully anneals for ~183 epochs. It may improve conditioning without moving final `best_test_acc` by the required +0.1pp. Fix: keep the first run simple, but record early-epoch and tail deltas so a null is interpretable.

5. **Idea-01 has an RNG confound in the proposed builder.** The sample code uses `torch.randperm` after `torch.manual_seed(42)`, which can advance global RNG state before dataloader shuffling/augmentation and make attribution noisier. Fix: use a local `torch.Generator().manual_seed(...)`, a deterministic stride, or all patches.

6. **Idea-01’s “weight decay corrupts frozen params” warning is overstated.** In PyTorch SGD, params with `grad is None` are skipped, so `requires_grad=False` usually prevents WD updates. Filtering optimizer params is still the right defensive implementation, but it is not the main scientific risk.

7. **Idea-03’s smoke gate is borderline for a single EXP-003 choice.** Throughput-only probing is not a metric violation, but using a killed run to decide width makes the proposal less clean under the fixed-budget spirit. Fix: commit 1.25x up front or reserve promotion/demotion for EXP-004.

Scored verdict:

| Idea | Evidence / reasoning | Potential impact |
|---|---:|---:|
| **01 whitening** | **8/10**: canonical fast-CIFAR front-end, cleanly targets conditioning, fits `train.py`/deps/eval constraints; main caveat is weak direct evidence at 183 epochs. | **5.5/10**: plausible +0.1 to +0.3pp, but ceiling is modest because this is no longer an extreme early-convergence regime. |
| **03 wider net** | **6/10**: WRN/airbench capacity argument is legitimate, but no observed capacity bottleneck here and it directly reduces the learned-important low-LR tail. | **8/10**: highest path toward 96% if capacity is binding and 115 epochs suffice. |
| **04 Muon** | **5/10**: strong optimizer concept, but concrete port has unpinned LR/scaling and a WD inconsistency. | **7/10**: could clear the bar if tuned, but one-shot EXP-003 has substantial divergence/under-step risk. |

**Pick: Idea-01, frozen patch whitening.**

It wins for EXP-003 because it preserves the proven 300s training dynamics, keeps EMA/TTA intact, has no hard-constraint issue, and attacks optimization conditioning without spending the low-LR tail. Width has the highest ceiling but knowingly trades away the strongest prior lever. Muon is interesting but too many coupled unknowns for the next single experiment.
