# Brainstorm EXP-003
**Created**: 2026-08-05

## Web Search & Literature Review

- **When Does Label Smoothing Help?** (`goals/maximize-cifar10-best-test-accuracy/knowledge/papers/label-smoothing.md`)
  Modest soft-target regularization can improve generalization and calibration at near-zero runtime cost, but should not be blindly stacked with other soft-label methods.
- **mixup: Beyond Empirical Risk Minimization** (`goals/maximize-cifar10-best-test-accuracy/knowledge/papers/mixup.md`)
  Convex input and target interpolation improves CIFAR-10 generalization across residual-family models, with conservative alpha values limiting convergence slowdown.
- **RandAugment** (`goals/maximize-cifar10-best-test-accuracy/knowledge/papers/randaugment.md`)
  A small, dependency-free augmentation search space can improve CIFAR accuracy, though distortion strength and host-transform cost are operating-point risks.
- **When, Where and Why to Average Weights?** (`goals/maximize-cifar10-best-test-accuracy/knowledge/papers/weight-averaging.md`)
  Late trajectory averaging can complement annealing cheaply, but this BatchNorm model needs an explicit buffer strategy.
- **SGDR** (`goals/maximize-cifar10-best-test-accuracy/knowledge/papers/sgdr.md`)
  Time-aligned cosine refinement is evidence-backed for CIFAR and has now been validated locally by EXP-002.
- **Wide Residual Networks** (<https://bmva-archive.org.uk/bmvc/2016/papers/paper087/index.html>)
  Widening residual blocks is a strong CIFAR capacity lever, though the published training regime does not establish the best width under this repository's fixed 300-second horizon.

## Experimental History Review

- The current moving baseline is `91.83%` at commit `5016cc4` (EXP-002); success now requires at least `91.93%` under the fixed protocol.
- EXP-001's 15% high-LR hold plus cosine/Nesterov reached `91.57%`. Its very low terminal train loss and lower test accuracy suggest that prematurely removing high-LR exploration harmed implicit regularization.
- EXP-002 restored standard momentum, held `lr=0.1` for 80%, stepped to `0.01`, and cosine-decayed to `1e-4`; it improved `+0.16` points while preserving 38,629 steps and ending at `91.82%`.
- The validated schedule and persistent workers should be held fixed so EXP-003 isolates a new lever. The remaining quality gap is generalization under hard-label cross-entropy, while the 269,722-parameter model and 330 MB peak allocation leave capacity as an orthogonal possibility.
- Dense late evaluation found only a `0.01`-point best/final gap, so measurement sparsity is not the current limiter. Untried gaps include target regularization, stronger augmentation, weight averaging, and model representation/capacity.

## Collected Ideas

- **Modest label smoothing** — Set cross-entropy label smoothing to a conservative fixed value such as `0.05`, leaving the successful optimizer and all transforms unchanged. It directly targets overconfident hard-label fitting at negligible compute cost and is supported by the saved NeurIPS study.
- **Conservative mixup** — Mix each minibatch with a permutation using a low Beta alpha such as `0.2`, then use the corresponding weighted pair of cross-entropies. It targets memorization and local nonlinearity more strongly than label smoothing, with evidence on CIFAR residual networks but some fixed-budget convergence risk.
- **Single-operation RandAugment** — Insert `transforms.RandAugment(num_ops=1, magnitude=5-7)` before tensor conversion while holding optimization fixed. It expands image-level invariances without dependencies; the main risks are excessive distortion and higher end-to-end host time.
- **Late parameter EMA** — Maintain an exponential moving average only during the low-LR tail and evaluate the averaged parameters, with an explicit BatchNorm buffer policy. This targets endpoint variance and sharpness around the converged trajectory, but an incorrect BN treatment could erase the benefit.
- **Modestly wider residual network** — Increase stage widths from `16/32/64` to `24/48/96` while retaining depth and schedule semantics. The H20 has ample memory and the tiny baseline may be capacity-limited, but slower steps reduce the number of examples processed in the fixed 300 seconds.
- **Preactivation residual representation** — Convert blocks to BN-ReLU-Conv preactivation and use projection shortcuts where needed, retaining roughly comparable depth. This attacks optimization and representation quality rather than regularization, but is a larger code change whose benefit may require schedule retuning.
- **Canonical channel standardization** — Replace unit standard deviations with canonical CIFAR-10 channel standard deviations. This corrects input scaling before the first convolution and may improve conditioning, though BatchNorm likely absorbs much of the effect and expected impact is small.
- **Sharpness-aware update moonshot** — Use a low-frequency or tail-only SAM-style perturbation so only part of training pays the second forward/backward pass. This directly targets flat-minimum generalization, but reduced update throughput and implementation complexity make it high-risk under 300 seconds.

## Combinations

- **Validated schedule + label smoothing**: Keep every successful EXP-002 mechanism fixed and add only mild target smoothing. This is stronger than schedule work alone because it attacks the remaining hard-label generalization gap, and cleaner than stacking multiple regularizers because its causal effect stays identifiable.
- **Validated schedule + conservative mixup**: The long high-LR plateau preserves exploration while mixup regularizes interpolation between samples; the deep tail can then refine soft-target training. Together they plausibly avoid the undertraining risk of mixup paired with early annealing.
- **Modest width + label smoothing**: Added capacity can lower representation bias while smoothing restrains its tendency to memorize. The combination has more upside than either alone, but it is deferred as a first test because a bundled result would not identify whether capacity or regularization mattered.
- **Late EMA + dense tail checkpoints**: Averaging can stabilize the same low-LR trajectory already observed every epoch, while dense checkpoints provide a direct raw-model comparator. This is stronger diagnostically than EMA alone, but BatchNorm state remains a material complication.

## Candidate Ideas

### 1.5x-Width ResNet-20 Capacity Test
**Summary**: Increase only the stage widths from `16/32/64` to `24/48/96`, producing exactly 605,026 parameters while retaining the current post-activation block, parameter-free shortcuts, and every EXP-002 training choice. This isolates capacity without bundling preactivation, projections, precision changes, or extra regularization. Full proposal: `proposals/idea-03.md`.

**What it targets**: Representation capacity as an orthogonal limiter. EXP-002 uses only 269,722 parameters and 330.1 MB on a 98 GB H20, so memory headroom is enormous; the opposing limiter is how many updates a wider model can complete in 300 seconds.

**Reasoning**: Wide Residual Network results support width as a strong CIFAR lever, and the current tiny network likely underutilizes the H20. A 1.5x multiplier is a conservative trade: 2.24x parameters/nominal convolution work, widths aligned to multiples of eight, and less throughput risk than doubling width.

**Sources**: Wide Residual Networks primary paper; EXP-002 runtime/VRAM measurements; `proposals/idea-03.md`.

**Estimated Effort**: medium.

**Risk Assessment**: The model may complete too few epochs for the unchanged schedule to fit it, or extra capacity may simply lower train loss and worsen overfitting. Published wider-network gains come from longer recipes, so transfer to this time budget is uncertain.

### Conservative Mixup on the Validated Schedule
**Summary**: Apply canonical minibatch mixup with `alpha=0.2` on every training batch, sampling a GPU permutation and scalar Beta coefficient and computing the weighted pair of cross-entropies. Preserve all accepted EXP-002 optimizer, schedule, architecture, data-loader, evaluation, and seed choices. Full proposal: `proposals/idea-02.md`.

**What it targets**: Memorization and overly sharp interpolation between hard-labeled CIFAR examples. It attacks the same broad generalization gap as smoothing but also regularizes the input manifold, giving it a higher potential ceiling.

**Reasoning**: The ICLR mixup evidence includes CIFAR residual-family models, and conservative `alpha=0.2` draws retain strong endpoint supervision. The long high-LR plateau may give mixed examples enough exploration before the validated low-LR refinement tail.

**Sources**: `goals/maximize-cifar10-best-test-accuracy/knowledge/papers/mixup.md`; EXP-002 analysis and learnings; `proposals/idea-02.md`.

**Estimated Effort**: medium.

**Risk Assessment**: Mixup can slow finite-horizon fitting, reduce the number of steps through counted GPU work, and shift BatchNorm statistics toward interpolated inputs. If it over-regularizes, a valid run could regress even though longer training would eventually benefit.

### Modest Label Smoothing on the Validated Schedule
**Summary**: Add only `label_smoothing=0.05` to the training cross-entropy while keeping EXP-002's model, augmentation, standard momentum, 80% `lr=0.1` hold, low-LR cosine tail, evaluation cadence, and fixed seed unchanged. The small value is deliberately below the common `0.1` setting to limit underfitting in a roughly 100-epoch ResNet-20 run. Full proposal: `proposals/idea-01.md`.

**What it targets**: The remaining hard-label generalization gap. EXP-002 drove late training loss to about `0.06` while final fixed-evaluator loss was `0.2843`; these losses are not directly comparable, but they are consistent with very confident fitting and no target-space regularizer.

**Reasoning**: Label smoothing directly penalizes extreme class margins at near-zero compute and memory cost. It is the cleanest de-bundled follow-up to EXP-002 and should leave the validated schedule and approximately 38,600-step horizon intact.

**Sources**: `goals/maximize-cifar10-best-test-accuracy/knowledge/papers/label-smoothing.md`; EXP-002 analysis and learnings; `proposals/idea-01.md`.

**Estimated Effort**: low.

**Risk Assessment**: The accuracy gain may be below the required `0.10` point or smoothing may compound high-LR implicit regularization and underfit. Training-loss scale changes by construction, so interpreting the higher loss as optimization failure would be invalid.

## Review

The mandatory external Claude review completed successfully with no fallback (`01-idea-review.md`). It found no hard-constraint violation and selected modest label smoothing. The significant caution retained is that the train/test loss comparison is only suggestive because the distributions and transforms differ, and label smoothing can improve calibration without moving top-1 accuracy. No implementation change can remove that metric risk without turning the experiment into a sweep; the clean, fixed `epsilon=0.05` test remains worthwhile because it leaves throughput and schedule behavior unchanged. Claude's proposed mixup-off tail refinement is preserved as a future direction rather than folded into this experiment, because the winning label-smoothing intervention should remain isolated.

## Idea Evaluation

Adopt the external verdict. Claude scored label smoothing `8/10` for evidence/reasoning and `5/10` for impact, versus mixup at `6.5/10` on both and width at `4.5/10` evidence / `6/10` impact. Label smoothing wins on evidence-to-risk alignment: it targets the diagnosed limiter and preserves the complete accepted optimization trajectory. Mixup has a higher ceiling but risks short-horizon undertraining and mixed-image BatchNorm shift; width attacks a less-supported limiter while potentially discarding 25-40% of updates.

## Chosen Idea
**Selected**: Modest Label Smoothing on the Validated Schedule

**Why this idea**:
It is the cleanest causal extension of EXP-002: add one target-space regularizer at effectively zero runtime cost while preserving architecture, augmentation, optimizer, 80% high-LR exploration, low-LR cosine refinement, evaluation cadence, and seed. External Claude adversarial review independently judged it the strongest match to the observed generalization gap and the least likely to turn the 300-second horizon into a confound.

**Hypothesis**:
With `label_smoothing=0.05`, the otherwise unchanged EXP-002 recipe will reduce excessive hard-label confidence without materially changing its approximately 38,600-step horizon, raising `best_test_acc` from `91.83%` to at least `91.93%` in one fixed-seed H20 run.
