# Idea-02: Alternating (derandomized) horizontal flip

## Summary
Replace `transforms.RandomHorizontalFlip()` (each image independently flipped with p=0.5 every epoch) with airbench's **derandomized "alternating flip"**: each training image is shown un-flipped on even-numbered visits and flipped on odd-numbered visits (or, per-epoch, the whole dataset's flip parity toggles). Over any two consecutive epochs every image is seen exactly once in each orientation, eliminating the sampling variance of independent coin-flips while preserving the exact same marginal flip distribution. airbench reports this "improves the performance of every training [they] considered" where horizontal flipping helps at all — a free, deterministic variance reduction on an augmentation we already use.

## Reasoning
- **Distinct mechanism from the saturated aug axes.** EXP-008/011/015 closed the input-aug *content* lane across all three mechanisms (occlusion / mixing / transform). Alternating flip is NOT a new aug content — it changes the *sampling* of an existing aug from i.i.d. Bernoulli to a low-discrepancy (antithetic) schedule. It reduces gradient-estimate variance per epoch, which is an orthogonal lever to "add more augmentation diversity."
- **Strong, specific external evidence.** airbench (arXiv:2404.00498) makes derandomized flip a headline contribution and states it improves *every* applicable training — an unusually strong, broad claim from a high-signal source, and CIFAR-10 horizontal flip is unambiguously beneficial here (it's already in the recipe).
- **Throughput-free and zero under-anneal risk.** It is a CPU-worker transform change (same cost as RandomHorizontalFlip); num_epochs is unchanged → it sidesteps the project's #1 failure mode entirely.
- **Cheap to attribute.** Single-variable, no compile, no architecture change — the cleanest possible experiment.

## Sources
- airbench / Keller Jordan, arXiv:2404.00498 — derandomized/alternating flip, "improves every training considered" (knowledge/references/fast-cifar10-recipes.md, WebSearch 2026-06-30).
- Antithetic-variates / low-discrepancy sampling (variance-reduction rationale).

## Estimated Effort
Low. Implement a custom flip that keys on a per-sample visit counter or per-epoch parity. The DataLoader uses `persistent_workers=True` + shuffle, so per-sample parity needs a counter that survives worker forks (e.g., index-parity: flip iff `(epoch + sample_index) % 2`), or simplest: a per-epoch global parity that flips the entire batch deterministically on alternate epochs (coarser but still antithetic over 2-epoch windows). Smoke: confirm over 2 epochs each image seen once each orientation; num_epochs unchanged; best==per-epoch-max.

## Risk Assessment
- **Magnitude risk (primary)**: the gain is pure variance reduction on an aug already present — airbench's improvement is real but may be ≤0.1pp on our heavily-regularized 150-epoch net, i.e. may not clear the +0.1pp bar even if directionally positive. This is the main reason it is a strong #2 but not the lead.
- **Implementation subtlety**: with shuffled persistent workers, achieving true per-image antithetic pairing is fiddly; the per-epoch-parity fallback is simpler but only antithetic at the dataset level, weakening the variance-reduction. Mitigation: prefer index-parity `(epoch+idx)%2` computed in the transform via a wrapped dataset that passes the index.
- **Interaction with RandomCrop/Cutout/RandomErasing**: those remain i.i.d.; only flip is derandomized — no interaction expected, but verify ep25 not depressed.
