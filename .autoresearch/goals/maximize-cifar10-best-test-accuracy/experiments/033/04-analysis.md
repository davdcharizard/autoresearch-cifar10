# Report EXP-033: Conservative Small-Area Random Erasing
- **Created**: 2026-08-06

## Goal

Maximize seed-42 CIFAR-10 `best_test_acc` above the moving 94.15% baseline, with at least +0.10 percentage points required, by changing only `train.py` under the fixed 300-second training and 600-second wall protocols.

## Idea & Hypothesis

Compose a mild unlabeled-occlusion prior with the accepted strong curriculum: apply p=0.25 Random Erasing at requested area 2-10%, aspect ratio 0.3-3.3, and CIFAR-mean fill after N1/M7/ToTensor but before Normalize and CutMix. The hypothesis was that only 1.1-1.9% unconditional erased pixels would add absence robustness without crossing the recipe's strong-underfit boundary, retain at least 99% exposure, and reach at least 94.25% accuracy.

## Approach

`train.py` received one top-level forkserver-picklable wrapper around torchvision RandomErasing. Its call runs inside `torch.random.fork_rng(devices=[])`, preserving subsequent accepted crop/flip/RandAugment and CutMix RNG streams. The strong transform alone instantiated the fixed p/scale/ratio/mean-fill/non-inplace policy between ToTensor and Normalize; weak/eval transforms, model, optimizer, schedule, collator contract, evaluator, and logging remained literal. Production telemetry was intentionally omitted, with exact masks, per-image RNG states, and CutMix boxes/targets persisted in ignored preflight artifacts instead.

## Execution

Static and semantic checks passed. The first exact-corpus controller attempt exposed a PyTorch CPUGenerator fault when replaying pinned/shared RNG-state views; cloning the batched state and each row into contiguous parent-owned memory resolved it without changing seed or data. A separate arithmetic correction widened only the achieved-mask fidelity bound: torchvision rounds rectangle dimensions, whose true legal range under the fixed requested policy is 1.5625-10.9375%.

The final immutable corpus contained 200 strong batches selected from the natural first 205 to balance 100 hard/100 CutMix decisions, plus 64 weak batches. Both copied model/SGD arms completed all 264 steps and the live 5,000-batch candidate/control loader checks completed. The trajectory then failed multiple preregistered safety gates, so fresh paired timing and the scored production run were correctly skipped. No `run.log`, evaluation, or primary metric was produced.

## Results

- **Primary metric**: NaN (baseline: 94.15%, delta: N/A; production not authorized)
- **Observations**: Policy geometry was exact and in range: 6,384/25,600 erased examples (24.9375%), 100% placement, 5.9499% conditional and 1.4838% unconditional mean area, and 19.7266% maximum effective post-CutMix mask. Despite only a 1.0525x maximum loss ratio and lower terminal strong/weak loss EMAs (0.9590x/0.9269x), the candidate produced >95% candidate-only class concentration at steps 4, 20, 21, and 22; maxima reached 8.9304x logit RMS, 4.4059x gradient norm, and 3.0674x update norm. Live delivery retained 89.08% throughput (159.07 vs 178.57 batches/s), but rollover-inclusive p95 waits were 45.014ms candidate and 38.830ms control, so the absolute 1.5ms gate was non-specific even though it formally failed.
- **Analysis**: The intervention achieved exactly its intended sparse geometry and did not worsen average short-horizon loss, yet it changed early class and optimizer geometry far beyond the registered safety envelope. Lower loss cannot clear the candidate-only concentration veto, and running production after observing it would violate the precommitted protocol. This does not prove that all Random Erasing schedules are accuracy-negative: no scored result exists, and paired trajectories naturally decorrelate after distinct inputs. It does discredit this exact always-strong p=0.25/2-10% mean-fill composition under the established safety standard. Together with EXP006, unlabeled mean-filled deletion now has negative evidence both as aggressive augmentation replacement and as mild composition, while class-bearing CutMix remains the validated regional prior.
- **Key Learning**: Only 1.48% unconditional mean-fill deletion still triggered early candidate-only concentration and 8.93x logit geometry; retire this exact occlusion composition.

## Verification

- **Conditions**: Baseline/source, static semantics, and exact corpus passed; trajectory safety failed; live p95 also failed; timing and production were skipped.
- **Review Notes**: The evidence is internally consistent and provenance-complete. Corpus hash `eff90f5701a303152c9ee44082713fbf63ce31e2561159a108c941a421cecb3d` registers the natural source stream and exact 100/100 CutMix split; report hash `278efad81e7fb2f562a4ca5757e75fb9d51540a4c6f20ea703e8b55b2bc6f76d` was serialized before assertions. The controller-only pinned-state failure was resolved before the final corpus and did not alter experimental settings.
- **Verdict**: invalid
- **Verdict Basis**: Production and evaluation were deliberately blocked by preregistered trajectory-integrity gates, leaving only partial preflight evidence and no trustworthy primary metric; the index therefore records NaN rather than treating the veto as a scored no-improvement.

## Unexplored Avenues

- Tail-only erasing after the LR/augmentation boundary would test a different, lower-step geometry, but it would corrupt the validated weak hard-label refinement objective and currently lacks positive directional evidence.
- Label-aware regional deletion or donor replacement is mechanistically different, but accepted CutMix already supplies that prior; another occlusion experiment needs evidence beyond parameter interpolation.
- A substantially rarer/smaller erasing point might avoid the trajectory veto, but tuning p/area after this failure is explicitly outside EXP033 and would be weakly motivated given the small realized 1.48% dose already failed.

## Next Steps

- **Preserve the accepted data curriculum and test a zero-recurring-cost representation lever** (medium confidence): Conv-only fan-out initialization is isolated to the stem/transitions and avoids adding further strong-view distortion, but needs strict relative-update gates.
- **Prefer intrinsically bounded architectural changes over new max/attention branches** (medium confidence): recent concentration failures argue for fixed, narrow perturbations with no globally recruited path.
- **Retire adjacent unlabeled-occlusion tuning** (high confidence): EXP006 and EXP033 now cover aggressive replacement and conservative composition without positive evidence.

## Exit Action Results

- No exit actions were configured.
