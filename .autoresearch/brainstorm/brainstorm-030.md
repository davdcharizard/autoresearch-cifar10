# Brainstorm EXP-030
**Created**: 2026-06-09
**Goal**: goals/improve-cifar10-test-accuracy.md

## Web Search & Literature Review

- **Yong, Huang, Hua & Zhang — "Gradient Centralization: A New Optimization Technique for Deep Neural Networks" (ECCV 2020), arXiv:2004.01461**
  Gradient Centralization (GC): before the optimizer step, project each weight tensor's gradient onto a hyperplane with zero mean — i.e. subtract the mean of the gradient computed over the INPUT/fan-in dimensions, per output unit. For a conv weight `(C_out, C_in, kH, kW)` subtract the mean over dims (C_in,kH,kW) per output channel; for a linear `(out, in)` subtract the per-row mean. Costs one mean-subtraction per param tensor → essentially FREE / compute-neutral. Reported small consistent top-1 gains across CIFAR/ImageNet ResNets (~+0.2-0.6%) and faster, more stable training. Mechanism: GC constrains the weight-space (a regularization on the loss landscape) AND smooths/standardizes the gradient, which the authors show both accelerates convergence and improves generalization. A drop-in 3-line addition between `loss.backward()` and `optimizer.step()`. Core torch ops only (no new dep).

## Experimental History Review

Current best **96.22%** (EXP-012, commit 6c417a4); bar = 96.32 (+0.1pp). 30 experiments; the plateau is exhaustively mapped (~22 axes closed): scalar knobs bracketed (LR-peak/Cutout/LS/WD/batch); aug family closed (TA ceiling; Mixup/CutMix/dropout regress); regularizer-adding fails (convergence-bound); compute-adding hits the epoch wall (k≥5, pre-act, BlurPool); batch-scaling compute-bound; weight-averaging/convergence-polish move loss not top-1 (EMA/SWA/LS-down/Bag-of-Tricks); downsampling/anti-aliasing closed both sides (BlurPool+ResNet-D); activation closed both recipes (SiLU EXP-010/028); **LR-schedule axis fully closed (peak EXP-016/017 + floor EXP-019/020 + shape/SGDR EXP-029)**.

**Key untried gap — the OPTIMIZER / gradient-dynamics class.** The optimizer (SGD + Nesterov + momentum 0.9) has only ever had its WEIGHT DECAY swept (EXP-005); the gradient-flow / update-rule itself has never been modified. Gradient Centralization is the cleanest, best-documented, compute-NEUTRAL entry into this class. It is distinct from every closed axis: NOT capacity (no epoch wall), NOT data aug, NOT a weight-average/polish (it changes the training TRAJECTORY via the gradient, not a post-hoc average), NOT a scalar-knob. CAUTION flagged by project-insight "adding regularizers fails (convergence-bound)": GC has a regularization interpretation, BUT it is ALSO claimed to ACCELERATE convergence — and a convergence-bound recipe is exactly the regime where a convergence-accelerator (as opposed to a convergence-COSTING regularizer like dropout) could pay off. Genuinely uncertain EV → a real experiment. (Confirmed via TSV scan: no GC/optimizer-modification experiment exists.)

**Also-untried but lower-value**: per-channel input std-norm (now correctly assessed as a LIKELY REGRESSION — frozen eval at std=(1,1,1) → train/test BN-scale mismatch, NOT a clean null); SE channel-attention was tested only on the OLD pre-TA recipe (EXP-008, null) and adds dt — low-value re-test.

## Candidate Ideas

### 1. Gradient Centralization (GC) on the SGD step
**Summary**: Between `loss.backward()` and `optimizer.step()` (train.py L237-238), centralize the gradient of every weight tensor with `ndim > 1`: `g.sub_(g.mean(dim=tuple(range(1, g.ndim)), keepdim=True))` — i.e. subtract the per-output-unit mean over the fan-in dims (conv: over (C_in,kH,kW) per C_out; linear: per row). Leave 1-D params (BN γ/β, biases) untouched. No config change; compute/param-neutral.

**Reasoning**: The optimizer/gradient-dynamics class is the single genuinely-unexplored lever left (only WD swept). GC is the best-documented compute-neutral member: ~+0.2-0.6% on CIFAR/ImageNet ResNets, a 3-line drop-in. Its mechanism (constrain weight space + standardize the gradient → accelerate convergence AND regularize the landscape) targets BOTH binding constraints of this recipe simultaneously: it could help the convergence-bound recipe converge better in the fixed budget AND improve the generalization gap — without adding compute (no epoch wall) and without being a convergence-COSTING regularizer (unlike dropout/Mixup, which failed). It changes the optimization trajectory, the class that can move top-1, unlike the closed post-hoc weight-averaging/polish levers.

**Sources**: Yong et al. ECCV 2020 (arXiv:2004.01461); train.py L237-238 (backward/step hook point), L192-197 (optimizer); project-insights (convergence-bound; polish-vs-top1); TSV (optimizer class untried).

**Estimated Effort**: low (3-line loop over `model.parameters()` before `optimizer.step()`; one 300s run).

**Risk Assessment**: (a) Magnitude — GC's reported gains are largest for deep/large nets; on a shallow well-tuned 9-block net the effect may be within the ~0.2pp noise floor → no-improvement. (b) Convergence-bound caveat — if GC's regularization half dominates its acceleration half here, it could mildly hurt (like the other regularizers). (c) Throughput — the per-step mean-subtraction is a handful of tiny reductions, negligible vs the convs; VERIFY epochs ~91 (expect throughput-neutral). Fails gracefully to no-improvement; compute/param-neutral so no epoch-wall/integrity risk.

### 2. Per-channel input std-normalization
**Summary**: Change train `transforms.Normalize` std `(1,1,1)`→true CIFAR std `(0.2470,0.2435,0.2616)`.

**Reasoning**: Last untouched input-pipeline scalar; closes the normalization axis. Compute-neutral.

**Sources**: train.py L152-155; prepare.py L13 (eval frozen at std=(1,1,1)).

**Estimated Effort**: low.

**Risk Assessment**: Now assessed as a LIKELY REGRESSION, not a clean null: eval is frozen at std=(1,1,1), so scaling only the TRAIN inputs by ~1/0.25 makes BN running stats (computed at train scale) mismatch the eval-scale inputs → predictable accuracy drop. Low informational value (an artifact, not a finding). Demoted to axis-closer only.

### 3. Squeeze-Excitation channel attention re-test on the TA recipe
**Summary**: Add SE blocks (global-avg-pool → 2 FC with reduction → sigmoid channel gate) to each BasicBlock, on the current 96.22 TA recipe.

**Reasoning**: SE was tested only on the OLD pre-TA recipe (EXP-008, 95.86 ≈ then-baseline) — a re-test on the stronger recipe.

**Sources**: EXP-008 (pre-TA SE null); train.py BasicBlock.

**Estimated Effort**: medium (new SE submodule per block).

**Risk Assessment**: SE adds compute (global pool + 2 FC per block) → small dt rise → mild epoch-wall risk (cf. EXP-028 SiLU's dt cost). Its prior null was attributed to "not channel-gating-limited" (recipe-independent), so EV is low AND it carries a throughput cost. Weaker than GC (which is truly compute-neutral and in an unexplored class). Demoted.

## Idea Evaluation

All three respect the hard constraints (train.py-only, no new deps, single GPU, ≤1 eval/epoch, no seed hacking) and are untried on the current recipe.

- **Evidence strength**: GC (1) has documented CIFAR ResNet top-1 gains AND opens the genuinely-unexplored optimizer/gradient-dynamics class. Std-norm (2) is a predicted regression. SE (3) already nulled (pre-TA) and adds compute.
- **Mechanism clarity**: GC is the crispest — it modifies the gradient/update to both accelerate convergence and regularize the landscape, hitting both binding constraints; and it's compute-neutral so neither effect is masked by an epoch wall. Std-norm predicts a mismatch regression; SE's mechanism already shown non-binding here.
- **Expected impact**: GC is the only candidate with a documented POSITIVE top-1 effect that is also compute-neutral and in an untried class. The others are closers/re-tests.
- **Risk profile**: GC fails gracefully (compute-neutral); std-norm risks a real regression; SE risks an epoch-wall confound.
- **Feasibility**: GC and std-norm are trivial; SE is medium.

GC (1) clearly leads: it is the best-evidenced, compute-neutral, genuinely-untried lever, opening the last unexplored class (optimizer/gradient dynamics) with a mechanism that uniquely targets BOTH the convergence-bound AND generalization-gap constraints of this recipe at once, without the epoch wall. Std-norm and SE are demoted (predicted-regression closer / compute-costing re-test).

## Chosen Idea
**Selected**: Gradient Centralization (GC) on the SGD step

**Why this idea**:
After 30 experiments closing ~22 axes, the optimizer/gradient-dynamics class is the one genuinely-unexplored lever (only WD ever swept). Gradient Centralization is its best-documented, compute-NEUTRAL member: a 3-line drop-in with reported ~+0.2-0.6% CIFAR-ResNet gains whose mechanism — constrain the weight space + standardize the gradient — simultaneously targets this recipe's two binding constraints (convergence-bound in the fixed budget AND a generalization gap at fixed capacity) without adding compute (so no epoch wall) and without being a convergence-COSTING regularizer (the class that failed). It changes the optimization trajectory, which can move top-1, unlike the closed post-hoc weight-averaging/polish levers.

**Hypothesis**:
Adding Gradient Centralization to the SGD update (centralize every `ndim>1` weight gradient before the step) lifts `best_test_acc` above the 96.32 bar at an unchanged ~91 epochs / 4,299,866 params / dt ~8ms / <600s, by accelerating convergence and improving the loss landscape at zero compute cost. Falsifiable: if epochs hold (~91, confirming compute-neutrality) but accuracy lands within ±0.2pp of 96.22, GC's benefit is below the noise floor on this shallow well-tuned net → the optimizer/gradient-dynamics axis is (start of) closed and the 96.22 plateau stands.
