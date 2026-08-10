# Proposal: CutMix Alpha 0.5 at the Accepted Probability and Phase Schedule

## Intervention

Change exactly one production literal in `train.py`:

```python
CUTMIX_ALPHA = 0.5  # accepted EXP-010 value: 1.0
```

Keep `CUTMIX_PROBABILITY = 0.5` and the complete accepted phase schedule unchanged. In particular, train the width-2 postactivation ResNet-20 on random crop, horizontal flip, and RandAugment N1/M7 for the first 80% of the 300 counted seconds; apply CutMix to 50% of those strong-phase batches; then explicitly stop the eight persistent workers and rebuild the weak crop/flip loader with hard integer targets and no CutMix. Retain the 80% LR boundary (`0.1` hold, followed by the existing `0.01`-to-`1e-4` cosine refinement), batch 128, standard SGD momentum 0.9, all-parameter weight decay `1e-4`, seed 42, evaluator, evaluation schedule, timer, model, and 1,073,962-parameter count.

This is an isolated geometry change, not a probability, duration, optimizer, or augmentation-stack change. There is no fallback alpha, probability interpolation, phase retiming, or valid-run reroll.

## Mechanism and Beta-Distribution Geometry

For symmetric `lambda ~ Beta(alpha, alpha)`, both alpha 1 and alpha 0.5 have mean 0.5, but they distribute mass very differently. Alpha 1 is uniform, with variance `1/12`. Alpha 0.5 is the U-shaped arcsine distribution, with density concentrated near 0 and 1 and variance `1/8`, 1.5 times larger. Only one third of continuous alpha-0.5 draws fall in `[0.25, 0.75]`, versus one half for alpha 1. Thus the candidate replaces moderate-area mixtures with more near-original and near-donor rectangles without changing how often the CutMix branch is selected.

A useful measure of two-class target ambiguity is `2*lambda*(1-lambda)`. Its expectation under a symmetric beta is `alpha/(2*alpha+1)`: `1/3` for alpha 1 and `1/4` for alpha 0.5, a 25% reduction. Including the fixed p=0.5 gate, expected ambiguity contribution per strong batch falls from `1/6` to `1/8`. The candidate therefore tests whether EXP-010 used slightly more per-event ambiguity than this short strong phase can comfortably fit, while preserving the successful frequency of class-bearing regional replacement.

These continuous formulas are rationale, not assumed production behavior. Torchvision converts the sampled lambda into an integer 32x32 rectangle, clips it at image boundaries, and adjusts the target coefficient to the realized area. Rounding and clipping can create near-no-op or near-full events and change the effective moments. The proposal consequently requires an empirical adjusted-lambda gate before production.

## Relation to the Accepted Point and the p=0.75 Failure

EXP-010 is strong evidence that CutMix itself should remain: alpha 1 at p=0.5 improved the then-frontier by 0.60 points to 94.15%, retained 99.10% of exposure, lowered final NLL to 0.1934, and converted an 89.73% switch checkpoint into a 93.16% first weak checkpoint and a terminal best. Its strong checkpoint was only 0.35 points below the otherwise accepted non-CutMix width-2 EXP-007 result (90.08%). Alpha 0.5 targets that small fit deficit, not a broad failure of the recipe.

EXP-011 changed a different axis. Raising alpha-1 CutMix probability from 0.5 to 0.75 increased the expected ambiguity dose per strong batch from `1/6` to `1/4`, a 50% increase. Exposure stayed equal, but switch accuracy fell 2.91 points to 86.82%, crossed the registered 87.08% underfit marker, and best accuracy fell to 94.00%. The candidate holds the accepted event count fixed and moves dose in the opposite direction, to `1/8`, so it is not an interpolation above the failed probability. It directly asks whether less ambiguous geometry can recover clean strong fit while retaining enough regional invariance for the hard weak tail.

The evidence does not establish that this direction will help. EXP-010 won despite its small switch deficit, and moderate-lambda rectangles may be exactly where CutMix supplies its useful localization and occlusion pressure. Alpha 0.5 could therefore hollow out the winning signal rather than refine it. That adversarial interpretation from EXP-021 is the proposal's main scientific risk and must remain explicit.

## Feasibility and Isolation

The implementation is already supported by the installed `torchvision.transforms.v2.CutMix` and by `F.cross_entropy` with probability targets. Alpha does not change tensor shapes, target format, model work, loss path, parameter count, or worker lifecycle, so meaningful throughput or VRAM movement is not expected.

The existing `cutmix_collate` wraps both the p=0.5 gate and transform in `torch.random.fork_rng(devices=[])`. Given identical worker-entry state, changing alpha leaves the hard/mixed decision identical and restores CPU RNG after either branch, preventing CutMix-internal draws from perturbing later crop, flip, RandAugment, shuffle, or gate streams. Hard events should therefore be bitwise identical between arms; only selected CutMix rectangles, pixels, and probability targets may differ. Because prior forkserver experiments showed that seed alone does not replay post-transform batches, paired safety evidence must use one persisted set of exact post-N1/M7 tensors, labels, gate decisions, and pre-CutMix RNG states rather than independently regenerated loaders.

Before a production run, verify at least 10,000 controlled CutMix calls per alpha and require finite normalized `[128, 10]` targets whose coefficients match recovered pasted area; alpha-0.5 adjusted-lambda variance at least 1.25 times alpha 1; mean adjusted `2*lambda*(1-lambda)` no more than 0.85 times alpha 1; adjusted means within 0.05; and at least 70% nonzero, non-full alpha-0.5 rectangles. On a persisted production-distribution corpus of at least 200 batches, require identical arm initialization and hard/mixed decisions, bitwise-identical hard events, valid geometry/targets, finite losses, gradients, parameters, BN buffers, and momentum state, no candidate-only greater-than-95% one-class concentration, and candidate terminal loss EMA no more than 1.5 times control. These are implementation and catastrophic-safety gates, not evidence that alpha 0.5 will generalize better.

Confirm compute neutrality with alternating fresh-process timing: candidate/control median synchronized GPU-step ratio no greater than 1.01, projected updates at least 26,629 (99% of EXP-010's 26,898), loader delivery comfortably above GPU consumption, peak allocation below 650 MiB, and projected total runtime below 540 seconds. Production must use one idle 98-GB H20, redirect output only to `run.log`, stop at 300 counted seconds, finish below 600 total seconds, evaluate at most once per epoch with no candidate-specific extra looks, realize approximately 50% mixed strong batches, and switch exactly once to hard weak targets.

## Hypothesis, Risks, and Falsification

**Primary hypothesis:** alpha-0.5 CutMix at p=0.5 reduces per-event ambiguity enough to recover plateau fit while preserving the regional invariance and hard-tail conversion of EXP-010, reaching `best_test_acc >= 94.25%` with at least 99% of accepted optimizer exposure.

The main risks are loss of moderate-size occlusion/localization, excess near-no-op or near-full-donor events after 32x32 quantization, same-class pairings that further reduce effective regularization, and overcorrecting a switch deficit of only 0.35 points. The upside is correspondingly modest: this is scalar tuning around a successful frontier and runs against prior strategic guidance favoring orthogonal representation changes. A bare 0.10-point pass is only ten CIFAR-10 test examples at one fixed seed, so it meets the protocol but should not be interpreted as a precise general effect size.

Pre-register switch accuracy at least 90.00% as evidence for the proposed fit-recovery mechanism, with EXP-010's 89.73% as the direct control and 87.08% as severe underfit. Retaining first-weak accuracy at least 93.16% is supporting evidence that endpoint-heavy geometry did not discard the useful regional signal. These are diagnostic comparisons, never mid-run vetoes or tuning signals.

A valid seed-42 run below 94.25% falsifies this exact operating point and it must be rejected without changing alpha, probability, phase timing, or seed. If switch fit improves to at least 90.00% but the primary metric misses, the ambiguity mechanism worked locally but failed to preserve/translate CutMix's generalization benefit—evidence that moderate rectangles were useful. If the metric clears 94.25% without improved switch fit, the accuracy result succeeds for the goal but the registered fit-recovery explanation is falsified. Failure of the empirical geometry, paired safety, throughput, scope, hardware, or runtime gates makes the experiment invalid/no-go rather than evidence about accuracy.
