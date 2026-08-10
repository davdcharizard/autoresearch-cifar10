# Brainstorm EXP-020
**Created**: 2026-08-06

## Web Search & Literature Review

- **Positive-Negative Momentum: Manipulating Stochastic Gradient Noise to Improve Generalization** (`experiments/020/papers/positive-negative-momentum.md`; https://proceedings.mlr.press/v139/xie21h.html)
  ICML 2021 reports ResNet-18 CIFAR-10 error 4.48% for PNM versus 5.01% for momentum SGD. PNM alternates two momentum streams and amplifies gradient noise at matched expected direction without a second gradient evaluation; default `beta0=1` is a concrete, literature-backed point.
- **Lookahead Optimizer: k Steps Forward, 1 Step Back** (`experiments/020/papers/lookahead-optimizer.md`; https://papers.nips.cc/paper_files/paper/2019/hash/90fd4f88f588ae64038134f1eeaa023f-Abstract.html)
  NeurIPS 2019 reports that periodic interpolation between fast SGD weights and a slow trajectory can reduce variance and improve CIFAR/ImageNet performance at low claimed cost. It acts throughout training, unlike the failed post-hoc EXP-018 mean, but remains exposed to averaging lag.
- **Regularization in ResNet with Stochastic Depth** (`experiments/020/papers/stochastic-depth-regularization.md`; https://papers.nips.cc/paper_files/paper/2021/file/82ba9d6eee3f026be339bb287651c3d8-Paper.pdf)
  NeurIPS 2021 analyzes stochastic depth as explicit/gradient regularization and notes that full-block skipping can reduce training compute. Shallow ResNet-20 and local strong underfit make the compute/generalization trade unusually delicate here.
- **Towards Understanding How Momentum Improves Generalization** (`experiments/019/papers/momentum-generalization.md`)
  Momentum may preserve shared-feature signal under heterogeneous margins, but this does not establish isolated Nesterov's sign or PNM's exact behavior under the accepted CutMix trajectory.
- **Existing goal knowledge** (`knowledge/README.md`)
  Width, RandAugment, CutMix, Option-A transitions, ECA, averaging, and batch scaling evidence constrain representation/data proposals; this loop prioritizes mechanisms not already rejected by the 19-experiment history.

## Experimental History Review

- EXP-010 remains the 94.15% frontier: width-2 postactivation ResNet-20, all-parameter decay 1e-4, N1/M7 plus 50% alpha-1 CutMix through 80%, then a hard weak cosine tail. Preserve this full recipe unless the candidate explicitly replaces one mechanism.
- The dominant measured systems cost is model backward (75.46%); optimizer work is only 1.67%, so low-arithmetic optimizer changes should retain exposure. Accuracy is limited by generalization under a short strong phase, not basic terminal convergence.
- Stronger CutMix, alternate decay, early weak switching, Cutout, label smoothing, identity-oriented initialization/graph changes, learned transition shortcuts, and uniform weak-tail SWA did not clear the gate. Fit improvements have repeatedly failed to predict late generalization.
- EXP-019 proved the Mixup collator/lifecycle feasible but was invalid before production because fresh forkserver safety attempts did not replay post-transform batches. Any future data-policy gate must persist exact augmented tensors; alpha-0.4 Mixup did not earn an accuracy verdict.
- Nesterov remains unresolved because EXP-001 confounded it with a harmful short high-LR hold. The exact one-keyword intervention is attributable but has weak direct evidence and a required 1.9x first update.
- PNM is untested locally and has the strongest close external result, but it requires a custom two-stream optimizer with exact coupled-decay and normalization semantics. Lookahead is also untested but overlaps the averaging failure family. Stochastic depth is untested and could trade lower step time for lower per-block updates.

## Collected Ideas

- **Default-beta0 Positive-Negative Momentum** - Replace ordinary SGD momentum with the ICML PNM recurrence using `beta0=1`, two alternating buffers, and the paper's noise normalization. This directly targets generalization through controlled gradient-noise geometry with low model cost and close CIFAR/ResNet evidence, but custom optimizer semantics and the abrupt 80% LR step require exact recurrence gates.
- **Isolated PyTorch Nesterov** - Add only `nesterov=True` to accepted SGD. This resolves EXP-001's confound with maximum attribution and negligible expected cost; its literature support is directional and the stronger current-gradient weight may amplify CutMix noise rather than improve generalization.
- **Whole-trajectory Lookahead SGD** - Wrap accepted momentum SGD with default-like slow interpolation every five inner steps. Unlike EXP-018, synchronization changes the search trajectory from the beginning and evaluation always sees installed weights; nevertheless, repeated pullback can waste a fixed 300-second improving path and adds a hyperparameter pair.
- **Low-rate batchwise stochastic depth** - Randomly skip a small fixed fraction of same-width residual branches during the strong phase and use the full expectation-scaled graph at evaluation. Conditional block skipping can reduce the dominant backward cost while adding ensemble regularization, but ResNet-20 is shallow and skipped branches receive fewer updates.
- **Deterministic pool-first Option-A transitions** - Replace transition shortcut slicing with 2x2 average pooling while preserving zero channel padding and no projection/BN. This isolates anti-aliasing from EXP-017's learned normalized shortcut, but adds pooling backward and may suppress high-frequency CIFAR detail.
- **Final-stage identity-scale ECA** - Add zero-start `2*sigmoid` ECA gates only to layer3 residual outputs. It targets semantic channel allocation at tiny parameter cost, but global descriptors can conflict with CutMix regions and sequential kernels may lose exposure.
- **Orthogonality penalty on convolution kernels** - Add a small literature-backed soft orthogonality regularizer to convolution weights to stabilize feature diversity. It targets generalization without changing inference, but penalty construction/backward is likely expensive relative to the 300-second budget and introduces a poorly transferred coefficient.
- **Hard-tail momentum reset** - Zero optimizer momentum exactly at the 80% augmentation/LR transition so noisy strong-view history cannot contaminate weak refinement. This simplifies state at a real distribution boundary, but accepted momentum continuity may be beneficial and no external evidence predicts a 0.10-point gain.
- **Moonshot Swapout residual sampling** - Randomly select identity, residual, or combined paths during strong training, then use the complete graph for evaluation. Literature reports CIFAR gains from implicit architectural ensembles, but the method strongly disturbs a shallow model already near underfit and complicates BatchNorm/evaluation scaling.

## Combinations

- **PNM + hard-tail ordinary momentum**: use PNM only during the noisy N1/M7+CutMix plateau, then deterministically convert its state to ordinary momentum at the 80% boundary. This could combine exploration noise with stable refinement, but state conversion and two simultaneous interventions make attribution weaker than testing full-run PNM first.
- **Stochastic depth + width recovery**: use saved block compute to widen channels slightly while keeping measured step time near accepted. The cross could turn skipped work into more representation capacity, but per-branch exposure and width change confound the result and BF16 width-3 evidence offers no safe shortcut.
- **Lookahead + Nesterov inner optimizer**: Nesterov could explore faster while slow interpolation damps its current-gradient overshoot. The mechanisms plausibly complement, but neither is locally validated and their combination would be impossible to attribute.

## Candidate Ideas

### Low-Rate Batchwise Stochastic Depth
**Summary**: During only the 80% strong phase, independently skip each of the six non-entry same-width residual branches with fixed batchwise probability 0.05. Keep all stage-entry/transition blocks active, scale surviving residuals by `1/0.95`, and run the full unscaled graph in weak training and evaluation. See `proposals/idea-03.md`.

**What it targets**: Both the 75.46% model-backward bottleneck and generalization. True conditional branch skipping should convert a small amount of convolution/BN work into more global updates while injecting mild architectural ensemble noise.

**Reasoning**: NeurIPS analysis supports stochastic depth as residual regularization with possible compute savings. The proposal scopes it conservatively for shallow ResNet-20, preserves transition feature creation, and requires at least 1% measured schedule speedup plus 25,900 effective updates per selected branch. Exact post-transform batch persistence incorporates EXP-019's protocol lesson.

**Sources**: `proposals/idea-03.md`; `papers/stochastic-depth-regularization.md`; `02-system-understanding.md`; EXP-010, EXP-012, EXP-015, and EXP-019 analyses.

**Estimated Effort**: high

**Risk Assessment**: Only 73.5% of strong batches execute all six selected branches; shallow-model underfit, sparse BN moments, frozen momentum/decay on drops, residual scaling variance, Python branching, and extra evaluation opportunities can erase the small compute benefit.

### Isolated PyTorch Nesterov Momentum
**Summary**: Add only `nesterov=True` to the accepted SGD constructor with momentum 0.9, dampening zero, and coupled all-parameter decay 1e-4. Every model, data, schedule, timer, lifecycle, and evaluator semantic remains byte-identical. See `experiments/019/proposals/idea-01.md`.

**What it targets**: Online basin selection and current-gradient response at essentially unchanged measured cost, while finally resolving EXP-001's Nesterov/schedule confound.

**Reasoning**: It is the most attributable surviving experiment and preserves accepted deterministic scale after the initial transient better than paper-default PNM. Momentum/generalization and nonconvex-acceleration papers provide only directional plausibility, not a direct ordinary-versus-Nesterov CIFAR gain. Its primary value is a clean answer with modest upside.

**Sources**: `experiments/019/proposals/idea-01.md`; `experiments/019/papers/momentum-generalization.md`; `experiments/019/papers/nesterov-nonconvex.md`; EXP-001, EXP-002, and EXP-010 analyses.

**Estimated Effort**: low

**Risk Assessment**: The first direction is exactly 1.9x ordinary momentum, current-gradient weighting may amplify N1/M7/CutMix noise, coupled decay dynamics change, and faster fit need not improve late generalization. The predicted ceiling is close to the ten-image gate.

### Default-beta0 Positive-Negative Momentum
**Summary**: Replace accepted PyTorch momentum SGD with the ICML PNM recurrence for the full run: `beta1=0.9`, `beta0=1`, two zero-initialized alternating FP32 momentum streams, paper normalization by `sqrt(5)`, and accepted coupled decay applied before the stream update. Preserve the complete model/data/schedule/evaluator contract. See `proposals/idea-01.md`.

**What it targets**: The diagnosed generalization bottleneck through optimizer-shaped anisotropic gradient noise at low model cost. It changes no examples, capacity, or evaluation opportunities.

**Reasoning**: Xie et al. report a 0.53-point CIFAR-10 gain over momentum SGD on ResNet-18, the closest positive external result among current candidates. However, exact recurrence analysis reveals the main transfer gap: at the same numeric LR, zero-initialized paper PNM starts at 0.1699x the accepted first direction and has roughly 22.36x smaller steady constant-gradient drift than PyTorch momentum. Coupled local decay also differs from the official code default. This is a high-evidence but high-mismatch optimizer replacement, not a pure noise intervention.

**Sources**: `proposals/idea-01.md`; `papers/positive-negative-momentum.md`; official PNM paper/code; EXP-008/009 decay findings; EXP-019 replayability learning.

**Estimated Effort**: high

**Risk Assessment**: Severe strong-phase underfit from deterministic-scale mismatch is plausible; two-stream cancellation can remove useful shared-feature history; CutMix already adds noise; custom foreach state can cost exposure or implement the paper incorrectly.

## Review

Mandatory external Claude review completed successfully and is preserved in `01-idea-review.md`. It selected isolated Nesterov (evidence 5/10, impact 4/10) over pinned PNM (5/10, 3/10) and stochastic depth (3/10, 3/10).

The review's decisive mechanical point is that ordinary and Nesterov momentum share the same steady constant-gradient scale: with `mu=0.9`, ordinary uses `b -> 10d` and Nesterov uses `d + 0.9b -> 10d`. The 1.9x first direction is a transient rather than a sustained effective-LR change. In contrast, paper-default PNM at unchanged numeric LR settles near `d/sqrt(5)`, about 22.36x below accepted PyTorch momentum, making severe fixed-budget underfit the expected result. No scale/LR rescue is imported into this loop because that would be a new PNM proposal requiring independent review.

Stochastic depth was rejected because returning identity on dropped branches overlaps the count-2 shallow-residual underfit family, while its projected schedule-weighted compute saving is too small to offset BN, per-branch update, and evaluation-count risks. Nesterov is retained exactly as proposed: one keyword, no warmup, clipping, LR compensation, momentum reset, or fallback.

## Idea Evaluation

| Idea | Evidence / reasoning | Potential impact | Decision |
| --- | ---: | ---: | --- |
| Isolated PyTorch Nesterov | 5/10 | 4/10 | Advance; clean steady-scale intervention and unresolved EXP-001 confound. |
| Default-beta0 PNM | 5/10 | 3/10 | Reject as pinned; external evidence is nullified by local deterministic-scale/horizon/decay mismatch. |
| Low-rate batchwise stochastic depth | 3/10 | 3/10 | Reject; shallow identity-oriented underfit risk dominates marginal compute benefit. |

## Chosen Idea
**Selected**: Isolated PyTorch Nesterov Momentum

**Why this idea**:
It is the only finalist whose exact mechanics preserve the accepted steady optimization scale, do not alter the model/data/evaluation surface, and cleanly resolve a standing local confound. Its direct evidence and ceiling are modest, but the proposed benefit is reachable within the fixed budget and any outcome is attributable to one declared optimizer keyword.

**Hypothesis**:
Adding only `nesterov=True` to accepted SGD will preserve at least 99% of EXP-010's optimizer exposure and switch fit near the accepted 89.73%, while a better online basin raises `best_test_acc` from 94.15% to at least 94.25%, with a point prediction of 94.30%.
