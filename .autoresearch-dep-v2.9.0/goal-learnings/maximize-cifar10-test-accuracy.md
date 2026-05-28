# Goal Learnings: Maximize CIFAR-10 Test Accuracy

**Goal**: goals/maximize-cifar10-test-accuracy.md
**Experiment Index**: experiment-indices/maximize-cifar10-test-accuracy.tsv

<!-- Per-goal qualitative distillation, agent-appended during the analyze phase.
     Three categories below. Each bullet follows the
     3-line, budget-strict format:

       - **{statement — ≤150 chars}** ({source refs})
         Evidence: {1-2 lines, MUST cite a source path — report / log / JSON / URL}
         Implication: {1-2 lines}

     HARD CEILING: if a bullet's total length exceeds ~3-4 sentences after composition, REWRITE it
     shorter. Push any detail that does not fit into the experiment report and cite the report
     path from Evidence. Do NOT commit an over-budget bullet. The Failed Approaches section
     uses Mechanism/Insight in place of Evidence/Implication — same budget shape. -->

## Protocol Findings

<!-- Measurement, verification, or benchmarking discoveries specific to this goal's metric.
     Examples:
       - "Stage-1 measurements are contaminated by CUDA autotune warmup — measure warm."
       - "Stored baselines expire same-day; require a same-day pristine reference run."

     Good format (fits budget length, cites report):
       - **Baseline pLDDT drifts ~0.3/day; require same-day pristine reference** (EXP-006)
         Evidence: reports/exp-report-006.md § Verification Results — 8-case drift table
         Implication: capture pristine baseline same-day as experiment measurement. -->

### High Importance
### Medium Importance
### Low Importance

## Failed Approaches

### High Importance
### Medium Importance

- **NUM_BLOCKS=4 (ResNet-26) costs 22% epochs (96→75); -0.15pp from undertrained deeper model** (count: 1, EXP-028)
  Mechanism: 5.8M params at 20-21ms/step yields 75 epochs. Deeper model converged slower early and was still improving at epoch 75 (best=final). 96.31% vs 96.46% baseline. The 33% capacity increase cannot compensate for 22% fewer epochs at this accuracy level.
  Insight: At 96.46%, ANY capacity increase costing >10% throughput will fail. The model needs both capacity AND epochs. WIDTH_MULT=4 + NUM_BLOCKS=3 appears optimal for 300s budget. Depth scaling only viable if throughput is simultaneously recovered (e.g., via larger batch).

- **Pre-activation blocks (BN-ReLU-Conv) cause ~6% throughput regression from lost cuDNN kernel fusion** (count: 1, EXP-021)
  Mechanism: cuDNN fuses Conv→BN→ReLU but not BN→ReLU→Conv. Per-step time rose from 15.6ms to 16.6ms, cutting epochs from 99→93 in 300s budget. best_test_acc 96.23% vs 96.46% baseline (-0.23pp).
  Insight: In throughput-constrained regimes, operator ordering affects kernel fusion and thus epoch count. Idea not exhausted if throughput can be recovered (custom kernels or layered with a throughput-positive change).

- **SE blocks (r=16) add ~9ms/step overhead regardless of implementation, reducing epochs from 98→83** (count: 2, EXP-011/EXP-012)
  Mechanism: Both nn.Linear (EXP-011: 95.45%) and Conv2d(1x1) (EXP-012: 95.23%) produce identical 18-19ms/step. Overhead is intrinsic to SE computation (global avg pool + FC layers + sigmoid + multiply), not channels_last format conversion.
  Insight: SE approach is exhausted for this model/budget. Any SE variant costs ~9ms/step on H20, cutting ~15 epochs. No implementation fix exists — the computation itself is the bottleneck.

- **EMA weight averaging exhausted: 3 variants all fail for compounding reasons** (count: 3, EXP-013/EXP-014/EXP-023)
  Mechanism: Parameter-only EMA (EXP-013): BN buffer mismatch, 94.98%. Full state_dict β=0.999 (EXP-014): too conservative, +0.05pp but 6-epoch throughput cost. Cubic-scheduled EMA (EXP-023): catastrophic shadow lag from near-zero early alpha, 96.02% vs 96.46% (-0.44pp).
  Insight: Each variant addressed the prior failure mode but revealed a new one. The approach class is exhausted — no EMA schedule/β/scope combination can simultaneously avoid BN mismatch, shadow lag, and throughput cost within 300s budget.

- **CutMix α=1.0 over-regularizes when stacked on TrivialAugmentWide+RandomErasing+WD=5e-4** (count: 1, EXP-010)
  Mechanism: Combined regularization slowed convergence; 96 epochs insufficient — best_test_acc 95.03% vs 95.39% baseline. Model still improving at epoch 96.
  Insight: Cross-sample augmentation stacks multiplicatively with per-sample augmentation; reduce α or replace RandomErasing rather than adding on top.

### Low Importance

- **torch.compile (default mode) provides zero speedup on H20 for small ResNet+AMP+channels_last** (count: 1, EXP-008)
  Mechanism: Inductor found no fusion opportunities beyond what AMP+channels_last already exploit; per-step time remained 9-10ms, identical to baseline. Model too small for graph-level fusion wins.
  Insight: Throughput gains on H20 for this model class should target batch size or architecture changes, not compiler optimizations. torch.compile overhead cost 1 epoch (82 vs 83).

- **Shifting LR drops earlier (0.35/0.55 vs 0.5/0.75) reduces final accuracy ceiling** (count: 1, EXP-006)
  Mechanism: Shorter high-LR exploration (35% vs 50% of budget) limits the basin of attraction quality; the extended LR=0.001 phase can't compensate.
  Insight: The (0.5, 0.75) schedule is near-optimal; high-LR exploration time is the primary driver of accuracy ceiling, not low-LR polish duration.

- **Nesterov + label_smoothing=0.1 reduces epoch count and peaks lower** (count: 1, EXP-004)
  Mechanism: Per-step overhead cost 4 epochs (65 vs 69); label smoothing made training harder without adding data diversity, causing the model to peak earlier and lower (93.28% vs 93.33%).
  Insight: In the 300s time-budget regime, prefer throughput-neutral or throughput-positive changes. The epoch count is the binding constraint.

- **CosineAnnealingLR with T_max >> actual epoch count** (count: 1, EXP-000; resolved by EXP-020)
  Mechanism: T_max=200 with 91 actual epochs decays LR only to 0.058 — never reaching the low-LR regime needed for convergence.
  Insight: Failure was parameter misconfiguration, not cosine decay itself. EXP-020 with correct T_max=100 achieved 96.46% (+0.55pp), confirming cosine is superior to step decay.

- **BN momentum 0.5 yields only +0.02pp at ~98 epochs — stat convergence already adequate** (count: 1, EXP-016)
  Mechanism: With ~19K updates at default momentum 0.1, running stats are well-converged by end of training. High momentum (0.5) adds noise during warmup (81-86% oscillation) without improving final accuracy (95.59% vs 95.57%).
  Insight: BN momentum tuning is a short-training optimization (sub-30 epochs, per speedrun recipes). Skip unless epoch count drops drastically.

- **Mixup α=0.2 replacing RandomErasing yields -0.04pp; destabilizes late-training polish phase** (count: 1, EXP-017)
  Mechanism: α=0.2 mild interpolation (λ median ~0.85) less effective than per-sample occlusion; test acc peaked 95.53% mid-training then oscillated 95.26-95.49% in polish phase, unlike stable baseline convergence.
  Insight: RandomErasing > mild Mixup for this model/budget. If retrying mixup, use mWh scheduling (disable in polish phase) or stack at reduced strengths rather than replace.

- **Stochastic depth (DropPath p_L=0.9) over-regularizes the shallow 9-block ResNet-20** (count: 1, EXP-018)
  Mechanism: Linearly spaced drop rates 0.011→0.1 across 9 blocks, stacked on TrivialAugmentWide+RandomErasing+LS=0.2+WD=5e-4. Final-phase training losses ~1.08-1.10 (higher than baseline), accuracy plateaued at 95.24% vs 95.57% baseline (-0.33pp).
  Insight: Regularization stack is near saturation — structural regularizers (DropPath, CutMix) compound with input-space augmentation to cause under-fitting. Gains must come from capacity, optimization, or data efficiency, not more regularization.

- **Augmentation-quality swap (reflect padding + Cutout replacing RandomErasing) yields +0.07pp — noise floor** (count: 1, EXP-022)
  Mechanism: Replaced zero-padding with reflect-padding in RandomCrop and swapped RandomErasing(p=0.25) for Cutout(12px, p=0.5). 96.53% vs 96.46% baseline (+0.07pp), below 0.1pp threshold. Zero throughput cost (99 epochs).
  Insight: At 96.46%, swapping between comparable occlusion methods is in the noise floor. Augmentation pipeline is near-optimal; gains must come from capacity, optimization dynamics, or fundamentally different training signals.

- **BN bias 64x LR multiplier causes severe instability at lr=0.2; -1.99pp regression** (count: 1, EXP-024)
  Mechanism: Effective BN bias LR peaked at 12.8 during 5-epoch warmup, causing wild test accuracy oscillations (26-65%) through epoch 20. Model only stabilized after epoch 80. The 64x multiplier was tuned for airbench96 (lr=9.0, 37 epochs) — not transferable to standard SGD setups.
  Insight: Idea not exhausted — lower multiplier (4-8x) calibrated to our lr=0.2 setup or warmup-only scaling may work. The concept (faster BN bias convergence) is sound but the magnitude was wrong by ~10x.

- **Gradient centralization yields only +0.03pp and costs 3 epochs from Python-level gradient iteration** (count: 1, EXP-025)
  Mechanism: Per-parameter GC loop adds ~0.5ms/step, costing 96→99 epoch difference. Wider test accuracy oscillations observed throughout training (dips to 77-78% in high-LR phase vs tighter baseline). GC's weight-space regularization is marginal when model already well-regularized.
  Insight: Python-level per-parameter gradient modification is not truly zero-cost at 16ms/step. Weight-space regularization via GC is redundant with augmentation+WD+LS stack. Optimizer tricks requiring per-parameter iteration should be implemented as custom C++/CUDA extensions to be viable.

- **Nesterov momentum yields only +0.06pp (96.52%); optimizer tricks converge to ~96.5% ceiling** (count: 1, EXP-026)
  Mechanism: Nesterov's look-ahead gradients produced wider mid-training oscillations but marginally better final convergence (+0.06pp vs +0.03pp for GC). 96 epochs at 16ms/step — truly zero per-step overhead. Three optimizer experiments (EXP-024/025/026) all fail to break 96.56%.
  Insight: Optimizer-level tricks alone cannot break through ~96.5% with this model capacity and training budget. The ceiling is likely capacity-bound, not optimization-bound. Next experiments should target capacity (depth/width) or fundamentally different training signals.

- **Shortened warmup (5→3 epochs) hurts -0.07pp vs Nesterov alone; 5-epoch warmup is load-bearing** (count: 1, EXP-027)
  Mechanism: 3-epoch warmup reaches full LR too quickly, causing early instability that 2 extra epochs (98 vs 96) cannot compensate. Nesterov amplifies the instability via look-ahead gradients during aggressive LR ramp-up. 96.45% vs 96.52% (Nesterov alone).
  Insight: WARMUP_EPOCHS=5 is optimal — shorter hurts convergence quality more than it helps from extra epochs. Do not reduce warmup below 5 epochs.

- **Learned 1x1 conv shortcut projections hurt -0.03pp; zero-padding acts as implicit regularization** (count: 1, EXP-029)
  Mechanism: Replacing zero-padded shortcuts with learned 1x1 conv+BN at stage transitions (96.43% vs 96.46%) removes an implicit information bottleneck. Zero throughput cost confirmed (16ms/step, 98 epochs). Zero-padding forces half the shortcut channels to zero, acting as structural regularization.
  Insight: In the current well-regularized regime, removing any form of regularization (even implicit architectural regularization) hurts. Zero-padding shortcuts are a feature, not a bug.

- **Per-channel std normalization causes -1.79pp regression; std=(1,1,1) is load-bearing** (count: 1, EXP-030)
  Mechanism: Changing std from (1,1,1) to (0.247,0.244,0.262) widens input range 4x, effectively 4x-ing first-layer learning rate. All hyperparameters were tuned for std=1 input scale. 94.67% vs 96.46% baseline.
  Insight: Input normalization and optimizer hyperparameters are tightly coupled. NEVER change std without retuning LR/WD. The std=(1,1,1) setup is NOT non-standard — it's the correct choice for this training recipe.

## Patterns

### High Importance

- **Orthogonal stacking: combining near-miss changes on different mechanism axes compounds past single-variable noise floor** (EXP-031)
  Evidence: reports/exp-report-031.md — Nesterov (+0.06pp alone) + reflect padding = +0.10pp combined. Same-axis combo (Nesterov+warmup, EXP-027) interfered (-0.07pp vs Nesterov alone).
  Implication: When individual zero-cost changes hit the noise floor, stack 2-3 changes on DIFFERENT axes (optimizer, data, evaluation). Avoid same-axis combinations. This is the validated strategy for squeezing out marginal gains.

- **Cosine decay to ~0 LR supersedes MultiStepLR step-decay; final LR floor matters most** (EXP-020)
  Evidence: reports/exp-report-020.md § Results — 96.46% vs 95.91% (+0.55pp); best=final epoch confirms model still improving at epoch 99
  Implication: Use cosine warmup+decay (5-epoch warmup, ESTIMATED_EPOCHS=100) as the default LR schedule. Lower final LR (~0 vs 0.002) and smooth transitions through intermediate regimes both contribute.

- **[Superseded by EXP-020 cosine schedule] Baseline MultiStepLR first drop at step 32K is critical for convergence** (EXP-000)
  Evidence: reports/exp-report-000.md § Results — removing step decay caused 2.93pp regression despite adding regularization
  Implication: Historical — cosine schedule provides a smoother path to the same low-LR regime without abrupt drops

- **[Superseded by EXP-020 cosine schedule] Wall-clock-fractional schedule correctly generalizes the He-2015 step-decay** (EXP-001)
  Evidence: reports/exp-report-001.md § Results — LR drops at pct_done=50.1% and 75.0% delivered +3pp jump at first drop
  Implication: Historical — cosine schedule is now the default; wall-clock-fractional no longer needed

### Medium Importance

- **~83-98 epochs complete in 300s budget on H20 with WIDTH_MULT=4 ResNet-20** (EXP-007, EXP-009)
  Evidence: reports/exp-report-007.md — 83 epochs at batch 128; reports/exp-report-009.md — 98 epochs at batch 256
  Implication: LR schedules and epoch-based hyperparameters should be calibrated for ~100 epochs; batch 256 is the current throughput ceiling

- **Width-2x ResNet-20 completes ~69 epochs in 300s at ~11ms/step** (EXP-001)
  Evidence: reports/exp-report-001.md § Results — 69 epochs, 26737 steps, 11ms/step, 598.7 MB peak VRAM
  Implication: Width-4x would yield ~35-40 epochs; throughput improvements (AMP, larger batch) needed to maintain epoch count at wider widths

- **First LR drop delivers the majority of accuracy gain; second drop contributes marginally without augmentation** (EXP-001, updated EXP-002)
  Evidence: reports/exp-report-001.md § Results — first drop: +3pp, second: +0.02pp; reports/exp-report-002.md — with augmentation, second drop: +0.52pp
  Implication: Augmentation creates a larger optimization gap for the polish phase to close; the second LR drop's contribution scales with training difficulty

- **TrivialAugmentWide + RandomErasing adds +0.63pp on width-2x model with negligible throughput cost** (EXP-002)
  Evidence: reports/exp-report-002.md § Results — 92.92% vs 92.29% baseline, 68 vs 69 epochs (1 fewer), same 11ms/step
  Implication: Augmentation is a free lunch on H20 for this model size; capacity-then-regularization compounding confirmed (+1.20pp total from BASE)

- **WD=5e-4 adds +0.41pp and amplifies the second LR drop's contribution from +0.52pp to +1.8pp** (EXP-003)
  Evidence: reports/exp-report-003.md § Results — 93.33% vs 92.92%; model still converging at epoch 69
  Implication: Stronger regularization (WD+aug) synergistically increases the optimization gap the polish phase closes; throughput gains (more epochs) would compound

- **AMP (FP16) gives 1.54x throughput (106 vs 69 epochs) but is unstable at abrupt LR steps** (EXP-005, resolved by EXP-020)
  Evidence: reports/exp-report-005.md § Results — 94.44% (+1.11pp); epochs 34-52 oscillated 68-82% at LR=0.01 step; EXP-020 cosine: no instability
  Implication: Cosine decay eliminates AMP instability by avoiding abrupt LR transitions; this concern is resolved with the current schedule

- **Batch doubling (128→256) with linear LR scaling + 5-epoch warmup yields +0.57pp from 18% more epochs** (EXP-009)
  Evidence: reports/exp-report-009.md § Results — 98 vs 83 epochs; per-step time 16ms vs 9ms; throughput gain sublinear (~1.18x) due to compute overhead beyond VRAM
  Implication: Further batch scaling (512+) will hit diminishing throughput returns; throughput-to-accuracy conversion still strong — each additional epoch contributes meaningfully

- **Label smoothing 0.2 is a zero-cost output regularizer orthogonal to input-space augmentation** (EXP-015)
  Evidence: reports/exp-report-015.md § Results — 95.57% vs 95.39% baseline (+0.18pp), 98 epochs (identical to baseline), ~16,300 img/s unchanged
  Implication: Output-distribution regularization composes cleanly with input-space augmentation (TrivialAugmentWide+RandomErasing) without over-regularization; 0.2 validated by hlb-CIFAR10

- **Horizontal-flip TTA adds +0.34pp at zero training cost by averaging logits from original and flipped test images** (EXP-019)
  Evidence: reports/exp-report-019.md § Results — 95.91% vs 95.57% baseline; eval overhead ~117s outside training budget
  Implication: TTA is orthogonal to all training changes and should be permanent; model trains with RandomHorizontalFlip so flip-equivariant features are already learned

### Low Importance
