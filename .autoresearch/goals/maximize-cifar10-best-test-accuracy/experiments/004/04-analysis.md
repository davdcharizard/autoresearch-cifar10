# Report EXP-004: Plateau-Only Conservative RandAugment
- **Created**: 2026-08-05

## Goal

Increase CIFAR-10 `best_test_acc (%)`, higher is better, from the moving baseline of `91.83%`. A valid improvement must reach at least `91.93%`, modify only `train.py`, use one idle H20 under the fixed 300-second counted training budget, and finish within 600 seconds total.

## Idea & Hypothesis

Use `RandAugment(num_ops=1, magnitude=7)` during the 80% high-LR exploration phase, then return to the accepted crop/flip pipeline for hard-label low-LR refinement. External Claude adversarial review selected input invariance over batch scaling and preactivation, but required the weak tail to address short-horizon underfit and BatchNorm distribution mismatch. The hypothesis predicted at least `91.93%` while preserving near-baseline optimizer exposure.

## Approach

Changed only `train.py`. Added weak and strong transform pipelines, a shared forkserver DataLoader factory, explicit persistent-worker lifecycle management, and a one-time phase switch. The crossing batch still used `lr=0.1`; training broke immediately afterward, evaluated, stopped all eight strong workers, rebuilt a weak loader, and continued with the existing `0.01`-to-`1e-4` cosine tail. Model, hard-label loss, seed, optimizer, schedule fractions, and fixed evaluator remained unchanged.

## Execution

Before the single GPU run, a reviewed `/tmp` diagnostic measured control loading at 329.3-382.5 batches/s, strong loading at 165.5-175.8, weak loading at 342.1, and the real forkserver transition at 2.612s; every worker exited and projected runtime was 338.6s. Static, formatting, scope, hardware-idleness, and lifecycle checks passed. One fixed-seed H20 run then exited `0` without retries in 340.7 seconds total.

## Results

- **Primary metric**: `92.30%` (baseline: `91.83%`, delta: `+0.47` percentage points, `+0.51%` relative)
- **Observations**: The final strong checkpoint was `84.60%` at epoch 79. After the switch at exactly 80.0%, the first weak-tail epoch reached `91.43%`, epoch 82 reached `91.96%`, and epoch 98 peaked at `92.30%`; final was `92.23%`. The run retained 38,358 of EXP-002's 38,629 steps (99.3%), 99 epochs, 25 unique evaluations, 300.0 counted seconds, 340.7 total seconds, 330.1 MB peak VRAM, and 269,722 parameters.
- **Analysis**: The hypothesis is supported. Worker-side augmentation added substantial training difficulty without reducing synchronized optimizer exposure, while the weak hard-label tail converted the learned invariances into clean-test accuracy and let BatchNorm statistics resettle. The immediate 6.83-point jump after switching shows that evaluating the strongly augmented model alone would misdiagnose its learned representation; the phase composition, not always-on RandAugment, is the validated recipe. RandAugment necessarily changes the fixed-seed augmentation stream, so exact causal size cannot be separated from draw changes in one run, but the 0.47-point gain exceeds the minimum by 0.37 and satisfies the declared protocol without seed selection.
- **Key Learning**: Plateau-only RandAugment raised accuracy 0.47 points while preserving 99.3% of baseline steps; hard-label tail refinement worked.

## Verification

- **Conditions**: All passed: `92.30% >= 91.93%`, exit 0, ten unique finite summary keys, 300.0 seconds counted training, 340.7 seconds total, 25 unique evaluations ending at epoch 99, one switch at 80.0%, and unchanged parameter count.
- **Review Notes**: Results confirmed trustworthy. Only the reviewed `train.py` intervention changed; evaluator, preparation, seed, model, and optimizer were untouched; the only H20 was idle; old workers terminated; the current-run log and summary were unique. The augmentation RNG stream changes as an inherent effect of the method, so the report avoids claiming a precise causal effect size.
- **Verdict**: improvement
- **Verdict Basis**: Every integrity and runtime condition passed, and the primary metric exceeded the moving baseline by 0.47 points and the acceptance threshold by 0.37.

## Unexplored Avenues

- Tune magnitude around 5-9 while preserving one operation, the 80% boundary, and the weak tail. This tests augmentation strength without discarding the validated phase composition.
- Move the strong-to-weak boundary slightly earlier or later while keeping magnitude 7. The large immediate tail recovery suggests refinement duration may be tunable.
- Replace private iterator shutdown with a purpose-built phase-aware dataset/worker control only if more frequent augmentation phases become valuable; for one switch, the measured current path is correct and inexpensive.

## Next Steps

- **High confidence**: keep the complete accepted EXP-004 recipe and brainstorm one isolated augmentation-strength or boundary refinement.
- **Medium confidence**: test same-width preactivation on top of the accepted data recipe only if external review judges its shallow-depth upside sufficient.
- **Medium confidence**: explore Cutout or RandomErasing as an alternative plateau-only input regularizer, de-bundled from RandAugment.

## Exit Action Results

- None defined.
