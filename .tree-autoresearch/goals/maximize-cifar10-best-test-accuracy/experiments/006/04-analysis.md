# Report EXP-006: Shared-Budget CutMix and Manifold Mixup
- **Created**: 2026-08-05

## Goal

Increase CIFAR-10 `best_test_acc` under the frozen 300-second charged training budget using a genuine `train.py` training change. The parent and global best were EXP-004 at 95.40%; the necessary parent-relative margin required at least 95.50%.

## Idea & Hypothesis

Preserve EXP-004's early 0.5 mixed-batch gate and clean-tail periodic SAM, but route 25% of selected early batches from CutMix to one-pass `Beta(2,2)` manifold mixup after block 2 or 4. Claude's randomized idea review selected this over ASAM and compact ResNeXt because it preserved independent images and nearly all throughput while importing an orthogonal representation-flattening mechanism. The preregistered hypothesis was 95.55-95.80% with at least 24,000 steps.

## Approach

Only `train.py` changed. Every selected batch consumes the exact parent seed-42 CutMix lambda, center, and CUDA permutation before private routing, preserving the parent's RNG stream over shared prefixes. A seed-43 CPU generator routes 75% of selected batches to unchanged CutMix and splits the remaining 25% between boundaries 2 and 4. Seed-44 CPU/CUDA generators supply exact four-uniform `Beta(2,2)` coefficients and manifold permutations. The model performs one out-of-place hidden interpolation, restores channels-last layout, pairs labels with the same permutation, and retains default clean, evaluator, and SAM calls. Claude's plan review added actual-parent parity, exact fixed-seed counts, Beta-shape moments, layout assertions, and Decimal threshold derivation.

## Execution

One fixed-seed run launched on physical GPU 0 with a 600-second timeout and complete log capture. Static, deterministic-policy, actual-parent CutMix, forward/loss, BF16 GPU, drop-path, BatchNorm, SAM, and latency smokes all passed before launch. The run exited 0 without retry, metric-driven adjustment, traceback, OOM, nonfinite value, or overlap assertion. It completed 25,644 steps and 132 evaluations in 300.0 charged and 455.8 total seconds.

## Results

- **Primary metric**: 95.41% (parent: 95.40%, delta vs parent: +0.01 points, +0.01%; global best: 95.40%)
- **Observations**: Best and final accuracy both reached 95.41%, but the final four evaluations ranged from 95.26% to 95.41%, so the nominal delta is below within-run tail variation. Final loss worsened from the parent's 0.1654 to 0.1749, while the child's final-four loss range was only 0.0011. Throughput was preserved: 25,644 steps versus 25,560, equal peak allocation at 1,190.5 MiB, and 132 epochs in both runs.
- **Analysis**: The intended policy executed at the registered dose: 2,561 manifold batches replaced approximately 2,557 parent CutMix batches while total selected exposure and early work remained nearly fixed. Accuracy therefore indicates an indistinguishable marginal exchange rate, not evidence that manifold mixup is intrinsically ineffective. The materially worse loss suggests that hidden interpolation did not replace CutMix's input-space, area-adjusted soft-label pressure; downstream BatchNorm updates on interpolated features may also contribute. Wall-clock scheduling gave the child 39 extra validated SAM pulses, so even the +0.01 cannot be causally assigned to manifold mixing. This specific 75/25 reallocation is discredited; the broader mechanism remains unproven at an approximately 10%-of-all-steps dose.
- **Key Learning**: Replacing one quarter of validated CutMix exposure with low-dose manifold mixup leaves accuracy within noise and worsens final test loss.

Raw terminal evidence preserved before log cleanup:

```text
mix_policy: selected=10257 clean=10412 eligible=20669 ratio=0.4963 late=0
cutmix: applied=7696 eligible=20669 ratio=0.3723
manifold: applied=2561 eligible=20669 ratio=0.1239 selected_share=0.2497 boundary2=1324 boundary4=1237 discarded_cutmix_specs=2561 lambda_mean=0.494529 min_lambda_mean=0.308301
sam: applied=2488 eligible=4975 ratio=0.5001 first_step=20670 first_progress=0.7500
best_test_acc:    95.41%
final_test_acc:   95.41%
final_test_loss:  0.1749
training_seconds: 300.0
total_seconds:    455.8
startup_seconds:  1.2
peak_vram_mb:     1190.5
num_epochs:       132
num_steps:        25644
num_params:       2,748,890
```

## Verification

- **Conditions**: Runtime, scope, hardware, configuration, policy, and summary integrity passed. The primary condition failed: 95.41% was below the 95.50% threshold.
- **Review Notes**: Claude's post-run adversarial review reconciled every recorded value with the raw log and approved RNG separation, discarded-spec parity, Beta shape, label pairing, and channels-last handling. It flagged that `late=0` is an unreachable counter rather than an independent cutoff measurement, that BF16 hidden interpolation is a minor paper-fidelity difference, and that wall-clock progress prevents exact phase-dose matching despite fixed seeds. These concerns narrow attribution but do not invalidate the result.
- **Verdict**: no-improvement
- **Verdict Basis**: The run is trustworthy and all hard constraints passed, but the +0.01-point delta missed the required +0.10-point parent-relative margin.

## Unexplored Avenues

- Apply manifold mixup additively to part of the parent's clean half while retaining full CutMix exposure. This would isolate its effect, although the observed loss regression warns that the combined regularization may be excessive.
- Move interpolation after the final normalized representation or explicitly control BatchNorm statistics to avoid updating downstream running statistics on mixed features.
- Perform hidden interpolation in FP32 before restoring BF16/channels-last execution. This would improve fidelity but is unlikely by itself to produce a detectable effect.

## Next Steps

- **High confidence**: Add preregistered model-weight EMA while retaining full CutMix and SAM doses; it is orthogonal and directly targets the observed late-iterate variation without another model pass.
- **Medium confidence**: Explore a lightweight additive representation mechanism such as channel recalibration that does not consume validated augmentation or optimizer exposure.
- **Low confidence**: Revisit manifold mixup only through an additive, attribution-preserving design with explicit BatchNorm handling, not another CutMix substitution ratio.
