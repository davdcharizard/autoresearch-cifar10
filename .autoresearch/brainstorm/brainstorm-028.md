# Brainstorm EXP-028
**Created**: 2026-06-09
**Goal**: goals/improve-cifar10-test-accuracy.md

## Web Search & Literature Review

- **Ramachandran, Zoph & Le — "Searching for Activation Functions" (2017), arXiv:1710.05941** (no local knowledge entry yet; will distill if chosen)
  Introduced **Swish / SiLU** = `x·σ(x)`, a smooth, non-monotonic activation found by RL search. Reported consistent small top-1 gains over ReLU across architectures (incl. ResNets on CIFAR/ImageNet) by replacing every ReLU, at negligible extra cost. The smoothness gives non-zero gradient for small negative pre-activations (no hard "dead-ReLU" zero region), which can improve optimization of deep/wide nets. SiLU is a single sigmoid-multiply — cheap and, critically here, a pointwise op that fuses into the conv epilogue under torch.compile.
- **Misra — "Mish: A Self Regularized Non-Monotonic Activation Function" (2019), arXiv:1908.08681**
  **Mish** = `x·tanh(softplus(x))`. Similar smooth non-monotonic shape to SiLU, reported marginally stronger CIFAR gains in the original benchmarks, but ~2× the pointwise cost (a `tanh` AND a `softplus`/exp), i.e. a larger throughput risk under our compute-`dt`-gated budget.
- Both `F.silu`/`nn.SiLU` and `F.mish`/`nn.Mish` are **core PyTorch ops** (no new dependency — satisfies the hard constraint).

## Experimental History Review

Current best **96.22%** (EXP-012, commit 6c417a4); bar = 96.32 (+0.1pp). 28 experiments; the plateau is exhaustively mapped (~20 axes closed):
- **Scalar knobs all bracketed**: LR-peak 0.2 interior optimum (EXP-016/017), Cutout-16 interior (EXP-013/021), LS 0.1, WD 1e-4 (EXP-005), batch 128 throughput-optimal (EXP-025).
- **Augmentation family closed**: TA+Cutout is the ceiling; Mixup/CutMix regress (EXP-011/018), policy saturated (EXP-014).
- **Regularizer-adding fails** — recipe is convergence-bound not overfit-bound (EXP-005/011/018/022).
- **Compute-adding hits the epoch wall** (High insight): k≥5 (EXP-004/009), pre-act (EXP-015), BlurPool (EXP-024) all under-train.
- **Batch-scaling** compute-bound (EXP-025); **weight-averaging/convergence-polish** (EMA/SWA/LS-down/Bag-of-Tricks) move LOSS not TOP-1 (EXP-006/019/020/023/026).
- **Downsampling/anti-aliasing axis now CLOSED from both sides**: BlurPool (EXP-024, compute-confounded) AND ResNet-D (EXP-027, fair 89-ep test, −0.47pp, loss worse too).

**Key untried gap**: the **activation function** has NEVER been changed — every experiment uses `F.relu` (train.py L89, L92, L127). This is the orthogonal axis explicitly flagged in the EXP-009 goal-learning ("try orthogonal axes — activation function, LR-schedule/optimizer tuning"). It is NOT a regularizer (so not subject to the convergence-bound failure), NOT capacity (so not the epoch wall, IF kept cheap), and NOT convergence-polish (it changes the representation/optimization landscape, which can move top-1, unlike EMA/SWA). The other remaining cheap probe — per-channel input std-norm — is assessed as expected null/mild-regression (train/test std mismatch, eval frozen at std=(1,1,1)).

## Candidate Ideas

### 1. SiLU/Swish activation (ReLU → SiLU everywhere)
**Summary**: Replace all three `F.relu(...)` calls (BasicBlock L89 pre-residual, L92 post-residual, stem L127) with `F.silu(...)`. SiLU = `x·σ(x)`, a smooth non-monotonic activation. Single one-line-each change; no other config touched (LR 0.2, batch 128, Cutout 16, TA, LS 0.1, WD 1e-4, cosine-to-0, params 4,299,866 all unchanged). Kaiming init is left as-is — every conv is BN-followed so the slight init-gain mismatch is absorbed.

**Reasoning**: The activation function is the single largest untried orthogonal lever (flagged in EXP-009 learning). SiLU is documented to give small consistent top-1 gains over ReLU on ResNets by smoothing the optimization landscape and removing the hard dead-zero region — a *representation/optimization* improvement, the class that CAN move top-1 (unlike the convergence-polish levers that only moved loss). Crucially it is **near-compute-neutral** at our launch-bound 8ms/step: SiLU is one pointwise sigmoid-mul that torch.compile fuses into the conv epilogue, so under launch-bound (GPU waiting on kernel launches, not pointwise FLOPs) the added cost is ~zero. This sidesteps the epoch-wall failure mode that killed every compute-adding structural change. Targets generalization through the activation, not capacity or regularization.

**Sources**: Ramachandran et al. 2017 (arXiv:1710.05941); train.py L89/L92/L127; goal-learnings EXP-009 orthogonal-axis flag; project-insights epoch-wall High entry (why cheap matters) + polish-vs-top1 Medium entry (why activation ≠ polish).

**Estimated Effort**: low (3 one-token edits `relu`→`silu`; one 300s run).

**Risk Assessment**: (a) Throughput — must VERIFY epoch-neutrality (epochs ≥ ~85). If torch.compile does NOT fuse SiLU and dt rises, epochs could drop and confound the test (mitigated: SiLU is a standard fused op; expect dt ~8ms). (b) Magnitude — on an already well-tuned shallow net the gain may be below the ~0.2pp noise floor → no-improvement. (c) Init-gain mismatch (kaiming assumes ReLU) is a non-issue (BN after every conv). Fails gracefully to no-improvement; compute/param-neutral so no epoch-wall/integrity risk.

### 2. Mish activation (ReLU → Mish everywhere)
**Summary**: Same swap as Idea 1 but to `F.mish` (= `x·tanh(softplus(x))`). Smooth non-monotonic, reported marginally stronger CIFAR gains than SiLU in the original Mish benchmarks.

**Reasoning**: If a smooth activation helps here, Mish may extract slightly more than SiLU per its CIFAR literature. Same orthogonal-axis rationale.

**Sources**: Misra 2019 (arXiv:1908.08681); train.py L89/L92/L127.

**Estimated Effort**: low (3 edits; one run).

**Risk Assessment**: HIGHER throughput risk than SiLU — Mish is ~2× the pointwise cost (tanh + softplus/exp). If those don't fully fuse, dt rises and we hit the epoch wall (the dominant failure mode here), confounding the test exactly like EXP-024. Given the epoch wall is the #1 killer, the cheaper SiLU is the safer first probe; Mish is the natural follow-up only if SiLU shows positive signal AND stays throughput-neutral.

### 3. Per-channel input std-normalization
**Summary**: Change train `transforms.Normalize` std from `(1,1,1)` to true CIFAR per-channel std `(0.2470,0.2435,0.2616)`.

**Reasoning**: Last untouched input-pipeline scalar; closes the normalization axis. Compute-neutral.

**Sources**: train.py L152-155; prepare.py L13 (eval frozen at std=(1,1,1)).

**Estimated Effort**: low.

**Risk Assessment**: Expected NULL or mild REGRESSION — eval normalizes with std=(1,1,1), so changing only TRAIN std creates a train/test input-scale mismatch (BN running stats computed at train scale, eval inputs at a different scale). Weaker closer than the activation probe; deferred.

## Idea Evaluation

All three respect the hard constraints (train.py-only, no new deps, single GPU, ≤1 eval/epoch, no seed hacking) and are untried.

- **Evidence strength**: Ideas 1 & 2 both have documented ResNet/CIFAR top-1 gains AND a project-specific rationale (the explicitly-flagged untried orthogonal axis). Idea 3 is assessed as expected-null with a mismatch-regression risk — weakest.
- **Mechanism clarity**: Ideas 1 & 2 are crisp — a smoother non-monotonic activation improves the optimization landscape / removes the dead-ReLU zero region, a representation-level change that can lift top-1 (distinct from the polish levers that only moved loss). Idea 3's mechanism predicts ~null.
- **Expected impact**: Ideas 1 & 2 are the only remaining levers targeting GENERALIZATION through a non-capacity, non-regularizer, non-polish change. Idea 3 is a closer.
- **Risk profile**: Idea 1 (SiLU) has the safest profile — cheapest activation, best odds of staying in the throughput-neutral band that every prior structural change failed. Idea 2 (Mish) carries real epoch-wall risk (2× pointwise cost) which is precisely this project's dominant failure mode. Idea 3 risks a train/test-mismatch regression.
- **Feasibility**: all low-effort.

Between the two activation ideas, **SiLU leads**: same evidence class and mechanism as Mish but materially lower throughput risk, which dominates here (the epoch wall has killed every compute-adding change — EXP-004/009/015/024). Testing the cheaper activation first cleanly isolates "does a smooth activation help?" without confounding it with a dt increase. Mish becomes the logical EXP-029 if SiLU shows a positive, throughput-neutral signal. Idea 3 is demoted to a later closer.

## Chosen Idea
**Selected**: SiLU/Swish activation (ReLU → SiLU everywhere)

**Why this idea**:
The activation function is the single largest never-tried orthogonal lever on this goal (explicitly flagged in the EXP-009 learning), and it is the only remaining change that targets generalization through the optimization/representation landscape rather than capacity (epoch wall), regularization (convergence-bound), or convergence-polish (loss-not-top1). SiLU specifically is documented to lift ResNet top-1 over ReLU and — uniquely important for this compute-`dt`-gated budget — is a single fused pointwise op that should stay throughput-neutral at the launch-bound 8ms/step, sidestepping the epoch wall that confounded every prior structural change. It is the cheapest, safest first test of the smooth-activation hypothesis; Mish is the follow-up if it pays off.

**Hypothesis**:
Replacing every ReLU with SiLU lifts `best_test_acc` above the 96.32 bar at an unchanged ~91 epochs / 4,299,866 params / dt ~8ms / <600s, by smoothing the optimization landscape and removing the dead-ReLU zero region. Falsifiable: if epochs drop materially (dt rose because SiLU didn't fuse) the test is compute-confounded (cf. EXP-024); if epochs hold but accuracy lands within ±0.2pp of 96.22, a smooth activation does not help this already-well-tuned shallow net and the plateau stands (→ try Mish, then accept ceiling).
