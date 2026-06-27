# Report EXP-030: Concat avg+max global pooling head
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-030.md
- **Plan**: plans/plan-030.md
- **Log**: logs/exp-log-030.md

## Goal

Maximize CIFAR-10 best_test_acc (%) within the fixed 300s training budget. Baseline 96.71 @ 1990397; bar 96.81; baseline mean ≈96.57, σ ≈0.16 (EXP-027). Question tested: does the cifar10-fast-pedigree head modernization — concat(avg, max) global pooling feeding a 512-d classifier — raise the converged plateau? It was the last untried architecture change believed to pass all five structural laws (free dt, init-informative, numerics/noise/normalization neutral).

## Idea & Hypothesis

Average pooling over 64 spatial positions dilutes localized high-evidence activations; a parallel max pathway preserves peak evidence and the classifier learns to weight both. Pedigree: cifar10-fast ResNet-9 head, fastai default. Hypothesis: dt 22.4±0.5ms, ~139 epochs, family-tracking early trajectory (the new fc is only 2,560 extra params), plateau shifted by the head's true effect; success iff best ≥ 96.81; a noise-band plateau closes the head axis.

## Approach

train.py only (+4/−2): `self.fc = nn.Linear(2 * w3, num_classes)` and the forward's pooling line replaced by `torch.cat([adaptive_avg_pool2d, adaptive_max_pool2d], dim=1)`. Nothing else — optimizer groups, init, schedule, loaders, compile, eval untouched. Params 4,288,586 (+2,560 exact, confirmed by startup print).

## Execution

Two runs. Run 1 (15:23Z): CONTENTION_KILLED at ~62% — windows jumped 22.7→40–44ms (4 consecutive >30ms); foreign PID 754819 confirmed live on GPU 0 at kill time (1014MiB, 34% util) — a textbook validation of the inline watchdog. Run 2 launched via a wait-for-GPU composite (polled until the foreign process cleared, ~3.5 min; user independently confirmed the GPU freed): rc=0, profile pristine (0/263 windows >27ms; epochs 137 vs 137.2 expected), stands for verification. No code changes between runs.

## Results

- **Primary metric**: 95.80 (baseline: 96.71, delta: −0.91, −0.94%) — −0.77 vs the baseline mean, far outside the ±0.15 noise band
- **Observations**: The head is throughput-free as predicted (+0.3ms → 137 epochs, −2) — the deficit is pure optimization quality. The hypothesis's "family-tracking early" prediction failed decisively and CONSISTENTLY across both runs: behind the baseline family at every waypoint (ep1 18.4 vs ~38; ep10 66.5 vs ~78; ep30 80.8 vs ~93; ep60 85.6 vs ~94.9; ep90 88.1 vs ~96; ep110 93.4 vs ~96.4), with the gap slowly closing through the anneal and the run STILL CLIMBING at cutoff (best 95.80 at ep136 — EXP-016's starvation signature). Final test_loss 0.2047 vs ~0.185.
- **Analysis**: The toll is trajectory-long, not init-localized, which points past the tiny re-initialized fc to the max pathway itself: max-pooling routes gradients only through each channel's argmax position — a discontinuous, high-variance credit-assignment path that keeps the head (and the trunk features serving it) poorly organized throughout the long high-LR phase. This is a SIXTH variant of the deferral law: a change can be free in dt, params, numerics, noise, and normalization constants and still lose by degrading per-step optimization quality during peak heat. The cifar10-fast pedigree transfers from a ~10-epoch final-iterate regime where the head's extra evidence pays immediately and the drag has no 100-epoch window to compound; under our max-over-checkpoints budget the run never converts. The transfer record of fixed-epoch/speedrun architecture evidence to this campaign is now 0-for-10.
- **Key Learning**: dt-free architecture additions can still lose by optimization quality — max-pool's argmax-routed gradients drag the entire trajectory under a long high-LR schedule; the head axis (avg vs avg⊕max) is closed, average pooling stands.

## Verification

- **Conditions**: condition 1 failed (95.80 < 96.81); conditions 2–3 skipped per first-failure-stop (both would pass: 499.5s ≤ 600; eval_lines 137 = num_epochs)
- **Review Notes**: results trustworthy — pristine contention profile on the standing run, params +2,560 exact, epochs at throughput expectation, two-run agreement on the trajectory shape; no false-failure risk (−0.91 ≈ 5.7σ).
- **Verdict**: no-improvement
- **Verdict Basis**: condition failure — valid clean result well below the bar; no hard-constraint violations (train.py only, GPU 0 honored including the wait-for-free protocol, eval once/epoch, ≤600s)

## Unexplored Avenues

- **Gated/learned blend instead of concat (e.g., αavg + (1−α)max, scalar α)**: avoids doubling the fc but keeps the discontinuous max gradient — the measured failure mode — so expected to inherit the drag. Low value.
- **LogSumExp (soft-max) pooling**: smooths the argmax discontinuity (dense gradients) while approximating max evidence — the principled fix for the diagnosed failure mode, but it adds a pointwise special function over 64×512×256 elements (EXP-026's cost ladder says price it first) and has no short-budget pedigree. Possible future candidate if the pool of ideas empties further; gate-price the dt.
- **Concat pooling introduced LATE (after the high-LR phase)**: a mid-run architecture switch — EXP-025/029 measured this recipe's violent response to mid-run distribution/calibration changes; rejected.

## Next Steps

1. **Progressive resizing (brainstorm-030's documented radical fallback)** (medium-low): the only remaining lever that ADDS epochs (~+15) without per-shape numerics changes; multi-mechanism risk (low-res toll, BN shock at switch, dual-shape compile) — needs a careful plan with pre-warmed shapes and a switch-point gate.
2. **LogSumExp pooling with an early-dt gate** (low): fixes EXP-030's diagnosed mechanism but pays a pointwise-op tax; only worth it if the dt prices under ~+0.5ms.
3. **Accept-and-replicate posture** (standing): with every axis closed (capacity, recipe, noise, augmentation, init, activations, throughput, optimizer geometry, normalization, head), remaining candidates are increasingly speculative; future loops should weight clean closure of remaining ideas and revisit combination plays only where mechanisms are provably independent.

## Exit Action Results
