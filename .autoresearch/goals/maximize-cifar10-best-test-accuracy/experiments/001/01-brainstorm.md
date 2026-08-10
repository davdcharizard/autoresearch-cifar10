# Brainstorm EXP-001
**Created**: 2026-08-05

## Web Search & Literature Review

- **mixup: Beyond Empirical Risk Minimization** (`experiments/001/papers/mixup.md`, https://arxiv.org/abs/1710.09412)
  Convexly mixed inputs and targets improve CIFAR-10 generalization across residual and wider architectures, but add a soft-target interaction and some timed-step work.
- **RandAugment: Practical Automated Data Augmentation with a Reduced Search Space** (`experiments/001/papers/randaugment.md`, https://papers.nips.cc/paper/2020/hash/d85b63ef0ccb114d0a3bb7b7d808028f-Abstract.html)
  A two-parameter augmentation policy can achieve strong CIFAR-10 results without a separate search, though host-side transform cost is risky under this run's 600-second wall limit.
- **SGDR: Stochastic Gradient Descent with Warm Restarts** (`experiments/001/papers/sgdr.md`, https://arxiv.org/abs/1608.03983)
  Cosine annealing gives strong anytime CIFAR performance and can be expressed in elapsed training time so the entire schedule completes despite variable step throughput.
- **When, Where and Why to Average Weights?** (`experiments/001/papers/weight-averaging.md`, https://proceedings.mlr.press/v267/ajroldi25a.html)
  Late trajectory averaging is cheap and works best with annealing, but BatchNorm state handling or recalibration adds risk in this nearly timeout-bound run.
- **When Does Label Smoothing Help?** (`experiments/001/papers/label-smoothing.md`, https://papers.nips.cc/paper_files/paper/2019/hash/f1748d6b0fd9d439f71450117eba2725-Abstract.html)
  Modest softening of one-hot targets often improves generalization at negligible compute cost, making it a clean low-risk objective intervention.

## Experimental History Review

- First experiment under this goal; no previous experimental approaches or failures exist.
- The unmodified baseline is `91.67%` after 38,525 steps and 99 epochs in 300.0 counted training seconds (`04-results.tsv`).
- The baseline ends at learning rate `0.01`: it reaches the 32,000-step decay only around 83% of the budget and never reaches the 48,000-step `0.001` phase. Missing terminal refinement is the clearest optimization gap.
- End-to-end time is `595.4s`, only 4.6 seconds below the hard limit. Ninety-nine evaluations and untimed input loading consume almost as much wall time as the fixed training budget, so added host work or validation is unsafe.
- The 269,722-parameter model peaks at only 330.1 MB on a 97,871 MiB H20. Capacity and accelerator utilization have substantial untested headroom, but exploiting them requires a higher-risk coordinated change.
- Untried gaps include horizon-aligned scheduling, Nesterov, label smoothing, Mixup/CutMix, RandAugment, weight averaging, mixed precision, compilation, larger batches, preactivation/wider residual models, and GPU-resident augmentation.

## Collected Ideas

- **Budget-aligned cosine SGD** — Replace unreachable fixed step milestones with monotone cosine decay driven by counted training time, optionally adding Nesterov. This directly targets the missing low-LR refinement phase and costs essentially no throughput.
- **Late trajectory EMA/SWA** — Average parameters over the final portion of training to reduce trajectory variance. Literature supports combining averaging with annealing, but BatchNorm buffers need explicit handling and a recalibration pass may breach the wall limit.
- **Label smoothing** — Set cross-entropy smoothing to 0.1 while preserving every other baseline choice. This tests a near-zero-cost generalization regularizer with unusually clean attribution.
- **Mixup or CutMix** — Use conservative soft-target batch mixing, initially Mixup with alpha 0.2 and no label smoothing. CIFAR evidence is stronger than for label smoothing alone, but timed-step overhead and slower short-horizon fitting are risks.
- **RandAugment** — Add a low-operation, moderate-magnitude torchvision policy before tensor conversion. This imports strong CIFAR augmentation evidence but can increase the already critical untimed host cost.
- **BF16, channels-last, and compilation** — Adapt execution to the H20 so more useful examples or a stronger model fit inside 300 counted seconds. The opportunity is large, but compile latency and changed numerical behavior require guards.
- **Wider preactivation ResNet** — Increase stage widths to 32/64/128 and use full preactivation blocks. The H20 has ample capacity, while a roughly 4x larger residual model may lift the small baseline's representational ceiling if throughput remains adequate.
- **GPU-resident vectorized crop/flip** — Store CIFAR-10 as GPU uint8 and reproduce random crop/flip in batched tensor operations. This simplifies the slow host path and creates total-time margin without changing augmentation semantics.
- **Sparse validation with final-focused annealing** — Evaluate at a few budget checkpoints rather than every epoch. It does not directly raise accuracy, but it removes the dominant total-time risk and enables more ambitious training changes while staying within the rule of no more than one validation per epoch.
- **Small modern ConvNet moonshot** — Replace post-activation ResNet20 with a compact ConvNeXt-like or depthwise residual network tailored to 32x32 inputs. It may improve capacity efficiency, but architecture and optimizer tuning are too coupled for the first loop.

## Combinations

- **Cosine schedule + label smoothing**: horizon-aligned convergence and low-cost regularization address underconvergence and overconfidence together. The combination may beat either alone, but a clean first loop should isolate the schedule because it repairs a measured defect.
- **H20 execution + wider preactivation model**: BF16, larger batches, channels-last, compilation, and a GPU input path can pay the compute cost of a stronger residual representation. The resource exchange plausibly beats accelerating the capacity-limited baseline or widening it without systems support.
- **Sparse validation + throughput modernization**: reducing validation calls creates the wall-clock margin needed for compilation and a richer model while preserving the 300-second active-training budget. This is stronger than either alone because throughput changes could otherwise increase epoch count and cause more evaluations.
- **Cosine schedule + late EMA**: smooth annealing supplies a stable late trajectory for averaging, and averaging may reduce endpoint variance. It is deferred until a BatchNorm policy can be validated without an extra untimed data pass.

## Candidate Ideas

### Budget-Aligned Cosine SGD with Nesterov
**Summary**: Replace `MultiStepLR([32000, 48000])` with one monotone cosine decay from 0.1 to 1e-4 driven by `total_training_time / TIME_BUDGET_S`, and enable Nesterov momentum. Keep the architecture, batch size, transforms, loss, seed, evaluation cadence, and all other hyperparameters fixed. The full design is in `proposals/idea-01.md`.

**What it targets**: The baseline never reaches its second milestone and receives no updates at its intended final learning rate. This candidate directly repairs the measured horizon/schedule mismatch without consuming meaningful compute or wall time.

**Reasoning**: At baseline throughput, step 48,000 would occur around 374 counted seconds, beyond the fixed 300-second budget. A time-indexed schedule guarantees terminal refinement even if throughput changes. SGDR supplies CIFAR evidence for smooth cosine annealing, while Nesterov is a low-cost complement. The proposal intentionally defers EMA/SWA to preserve attribution and avoid BatchNorm/runtime risk.

**Sources**: `proposals/idea-01.md`; `experiments/001/papers/sgdr.md`; `experiments/001/papers/weight-averaging.md`; baseline summary.

**Estimated Effort**: low

**Risk Assessment**: Cosine may anneal too early and reduce useful high-LR exploration; Nesterov is a second simultaneous variable. If it fails, cosine-only and a short flat high-LR hold are straightforward discriminating follow-ups.

### Low-Cost Label Smoothing
**Summary**: Change only the training cross-entropy call to `label_smoothing=0.1`, leaving the baseline input pipeline, optimizer, schedule, model, seed, and validation unchanged. Defer Mixup, CutMix, and RandAugment so the result remains attributable. The full design is in `proposals/idea-02.md`.

**What it targets**: The one-hot objective can promote overconfident fitting and weak generalization in the small network. This candidate tests that quality gap without adding host transforms, model state, or material GPU work to a run with only 4.6 seconds of wall margin.

**Reasoning**: NeurIPS evidence shows soft targets can improve classification generalization, and PyTorch supports the change directly in the existing loss. It is the lowest operational-risk candidate and cleanly establishes whether simple objective regularization is useful before richer mixing policies.

**Sources**: `proposals/idea-02.md`; `experiments/001/papers/label-smoothing.md`; `experiments/001/papers/mixup.md`; `experiments/001/papers/randaugment.md`.

**Estimated Effort**: low

**Risk Assessment**: The effect may be below the required 0.1-point gain, or smoothing may underfit under the short/misaligned schedule. Ordinary runtime jitter could still breach 600 seconds even though the change itself adds almost no wall cost.

### H20 Throughput Dividend for a Wider PreAct ResNet
**Summary**: Move CIFAR-10 to GPU-resident uint8 storage, reproduce crop/flip in vectorized Torch, use BF16 autocast, channels-last, guarded `torch.compile`, batch 512, and sparse validation, then spend the throughput margin on a 2x-width full-preactivation ResNet20. Use an elapsed-time warmup/cosine schedule and hard targets. The full design is in `proposals/idea-03.md`.

**What it targets**: The baseline uses only 330.1 MB on a 98 GB H20 and a 0.27M-parameter post-activation model, while CPU input/evaluation overhead leaves almost no total-time margin. This candidate targets useful model capacity per active-training second and the separate wall-clock bottleneck identified above.

**Reasoning**: Hopper-friendly BF16 and larger static batches should improve convolution efficiency; GPU-resident augmentation removes per-sample host work; sparse validation recovers wall time; preactivation and 4x parameter capacity provide direct accuracy headroom. The combined resource-exchange hypothesis has the highest upside and is explicitly designed to keep all moved training work inside the 300-second accounting.

**Sources**: `proposals/idea-03.md`; `experiments/001/papers/sgdr.md`; baseline summary and `train.py` execution structure.

**Estimated Effort**: high

**Risk Assessment**: Many coordinated changes weaken attribution. Compilation may fail or consume startup margin, vectorized crop semantics may drift, large-batch optimization may underfit, BF16 may alter stability, and the wider model may consume the entire throughput gain. Guarded eager fallback and later ablations reduce but do not remove these risks.

## Review

The cross-model review in `01-idea-review.md` selected Budget-Aligned Cosine SGD with Nesterov. Two concerns materially refine the experiment:

- Immediate cosine decay could remove too much of the high-LR exploration phase. Use a 15% counted-time hold at `lr=0.1`, then cosine-decay to `1e-4` over the remaining 85%.
- Retaining 99 evaluations leaves only 4.6 seconds below the hard timeout. Evaluate at 20%, 40%, then every 10% from 60% through 90%, plus the terminal model. Enable persistent training-loader workers after diagnostics measured later epochs at about 1s with persistence versus 19s without it. This bounds evaluation cost while covering the prior 60%-80% blind spot.

Nesterov remains enabled because its overhead and risk are small; if the combined result fails, cosine-with-standard-momentum is the pre-registered disambiguation. The reviewer also recommended de-bundling the H20 proposal before a later capacity experiment and rejected label smoothing as a weak first move because it does not address the measured schedule defect.

## Idea Evaluation

Adopt the scored verdict from `01-idea-review.md`. Budget-aligned cosine received the strongest evidence/reasoning score (`9/10`) because it repairs a quantified defect: the baseline reaches only 38,525 steps and never enters its planned `lr=0.001` phase. Its moderate upside is more reliable than label smoothing, while the higher-ceiling H20/wider-model candidate is too confounded for the first experiment. Validating the time-aligned schedule first also establishes a component needed by a later de-bundled capacity experiment.

## Chosen Idea
**Selected**: Budget-Aligned Cosine SGD with Nesterov

**Why this idea**:
It directly fixes the strongest measured limiter at negligible compute and memory cost. The experiment keeps model capacity, data augmentation distribution, batch size, loss, seed, and optimizer scale fixed, while the reviewer-driven 15% hold preserves early exploration. Seven budget-positioned evaluations and persistent training-loader workers remove measured timeout risks without changing the fixed evaluator. This is the best first causal test and a useful foundation for later regularization or wider-model experiments.

**Hypothesis**:
Holding `lr=0.1` for the first 15% of counted training time and then cosine-decaying to `1e-4`, with Nesterov momentum, will raise `best_test_acc` from `91.67%` to at least `91.77%` because the model gains thousands of low-LR updates while retaining early high-LR exploration. The run should complete below 600 seconds with persistent training-loader workers and evaluation at 20/40/60/70/80/90% plus termination.
