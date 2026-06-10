# Brainstorm EXP-042
**Created**: 2026-06-09
**Goal**: goals/improve-cifar10-test-accuracy.md

<!-- NOTE: revised during planning. Two earlier leads were discarded after the prior-knowledge
     checks: (1) a "fuller augmentation cooldown" — the cooldown axis is CLOSED (count:3, High
     Importance: base-noise ±0.25pp > any benefit); (2) an AdamW optimizer swap — the High
     Importance polish-vs-top1 insight is axis-independent and explicitly covers the OPTIMIZER
     axis ("top-1 needs capacity or fundamentally different generalization, NOT optimization").
     The chosen lead aligns with what that insight says DOES move top-1, and with its explicit
     endorsement of an auxiliary decayed loss. TTA remains off-limits (integrity-rejected). -->

## Web Search & Literature Review

- **Deeply-Supervised Nets (Lee et al., AISTATS 2015) / GoogLeNet auxiliary classifiers
  (Szegedy et al., CVPR 2015)**: attach lightweight classifier heads to intermediate layers and add
  their cross-entropy as an auxiliary loss (GoogLeNet weight 0.3, discarded at inference). Provides
  direct gradient signal to mid-level features → better-conditioned optimization and more
  discriminative intermediate representations → a generalization/feature-quality mechanism distinct
  from capacity, augmentation, optimizer, and objective tweaks. Eval uses the main head only.
- The project's own **project-insights.md line 82** explicitly prescribes this lever: "if multi-scale
  signal is wanted, use an AUXILIARY decayed loss on early layers, never input-concatenation into the
  main head" — written after the multi-scale-head failure (EXP-032). This is a direct internal endorsement.

## Experimental History Review

Current best = baseline 96.22 (EXP-012, TA+Cutout, commit 6c417a4). Bar = 96.32 (+0.1pp). 31
consecutive no-improvements since EXP-012. Decisive context from goal-learnings + project-insights:

- **Polish-vs-top1 wall (High Importance, axis-independent):** on this k=4/300s net, compute-neutral
  changes that lower eval loss do NOT raise top-1 — confirmed across weight-averaging (EMA/SWA),
  init/WD (zero-γ, no-bias-decay), optimizer/gradient-dynamics (GC), and objective/loss-shape
  (PolyLoss, cosine head, LS). "Top-1 gains require capacity or fundamentally different
  generalization, not optimization/objective polish." (project-insights L61)
- **Multi-scale-head (EXP-032) regressed −1.5pp** by routing layer2 features INTO the main classifier
  (input-concatenation), disrupting the coarse-to-fine hierarchy. The post-mortem's prescription:
  inject mid-level signal via an AUXILIARY decayed loss, NOT into the main head (project-insights L82).
  → Deep supervision is the untried, explicitly-endorsed form of "use mid-level signal."
- **CLOSED axes (do not retry):** augmentation family entirely incl. cooldown (EXP-033/034/035, count:3);
  LR schedule; capacity (k both ways + reallocation EXP-038); regularizer-ADDS (dropout EXP-022, SAM);
  architecture (preact/blurpool/ResNet-D/multi-scale-head/SE); activations; weight-averaging; classifier
  head; large-batch; bag-of-tricks; objective/loss-shape; cheap throughput (cudnn.benchmark).
- **TTA is OFF-LIMITS** (integrity-rejected EXP-027/032). The optimizer FAMILY (AdamW) is untested but
  the polish insight predicts it is optimization-not-generalization (kept as a fallback candidate).
- **Fairness:** dt must stay ~8ms (epoch-neutral, ~88–94 ep). Per project-insights L77/L108, any per-step
  Python loop over params and any sub-ms extra op can cost dt — the aux head must be cheap and stay inside
  the compiled forward graph (not an eager per-param loop).

## Candidate Ideas

### 1. Deep supervision — auxiliary classifier on layer2 with a decayed aux loss (eval uses main head only)
**Summary**: Add a lightweight auxiliary head — `nn.Linear(w2=128, 10)` fed by a global-avg-pool of
layer2's output (16×16×128) — used ONLY during training. The training loss becomes
`L = CE_main(+LS) + λ(t)·CE_aux(+LS)`, where `λ(t) = LAMBDA_AUX·(1 − frac)` decays linearly from 0.3 to 0
over the time budget. `forward()` returns `(main_logits, aux_logits)` when `self.training`, and just
`main_logits` when eval — so the frozen `Eval.evaluate()` (which does `model.eval(); model(inputs)`)
scores the unchanged main path. The aux head is discarded at inference (standard deep-supervision practice).

**Reasoning**: This is the ONE untried lever the project's own High Importance insights point to as
top-1-relevant. The polish wall is about OPTIMIZATION/objective changes; deep supervision instead changes
WHAT the intermediate features must encode (a generalization/feature-quality mechanism via auxiliary
gradient signal), which is the class the insight says CAN move top-1. It is the explicitly-endorsed,
hierarchy-preserving way to use mid-level signal (project-insights L82) — the opposite of EXP-032's
hierarchy-disrupting input-concatenation (aux gradients flow back through layer2/layer1 to sharpen their
features, but the main coarse-to-fine forward path is untouched). The decayed λ→0 means the final, evaluated
iterates optimize the pure main objective, sidestepping the regularizer-underfit wall that sank dropout/SAM.
Compute-light: one global pool + a 128×10 matmul + one extra loss term, all inside the compiled forward;
the extra backward shares the conv backward → dt expected ~8ms (must verify, epoch-neutral).

**Sources**: Deeply-Supervised Nets (Lee et al. 2015); GoogLeNet aux classifiers (Szegedy et al. 2015);
project-insights.md L61 (polish wall), L82 (auxiliary-decayed-loss prescription); EXP-032 (multi-scale-head
regression — the failure this fixes); goal-learnings (regularizer-underfit, fairness/dt notes).

**Estimated Effort**: low–medium — add `aux_fc`, branch `forward` on `self.training`, compute the weighted
two-term loss with a decayed λ in the training loop; verify dt and that eval still gets a single tensor.

**Risk Assessment**: (a) "needs depth" null — deep supervision's benefit grows with depth; on a shallow
9-block net the gain may be small/within-noise (like zero-init-γ EXP-026). (b) dt risk if the aux head's
fwd/bwd is non-trivial → mild epoch cost (mitigated by a single tiny linear + keeping it compiled).
(c) λ too high could distract the main objective early — mitigated by moderate 0.3 and decay→0. Worst case:
graceful within-noise no-improvement. No eval/integrity risk (main head only at inference), no scope risk.

### 2. AdamW optimizer-family swap (fallback)
**Summary**: Replace SGD+Nesterov with `AdamW(lr≈2e-3, betas=(0.9,0.999), weight_decay=0.05)`, same
cosine+warmup schedule/aug/LS/compile/seed. The single largest untested axis (all 43 runs used SGD).
**Reasoning**: "Optimizer is fair game"; adaptive steps may converge faster in 300s; closes the axis.
**Sources**: AdamW (Loshchilov & Hutter, ICLR 2019); EXP-030/031/036/041 (SGD-modification optimizer runs).
**Estimated Effort**: low.
**Risk Assessment**: High prior of regression/null — the High Importance polish insight is axis-independent
and explicitly covers the optimizer axis ("optimization, not generalization"), plus the adaptive
generalization gap and an LR-retuning confound. Demoted below idea 1 for exactly this reason.

### 3. Lookahead optimizer wrapping SGD (k=5, α=0.5)
**Summary**: Slow-weight interpolation every 5 fast steps; keeps tuned SGD, `torch._foreach_` update.
**Reasoning**: Untried; trajectory smoothing → flatter minima (a generalization mechanism).
**Sources**: Lookahead (Zhang et al., NeurIPS 2019); EXP-019/020 (SWA), EXP-030/031 (GC).
**Estimated Effort**: medium.
**Risk Assessment**: Same trajectory-smoothing family as SWA (null) and GC (loss-only) → polish-wall prior
predicts loss-only/null. Lowest axis-closure value of the three.

## Idea Evaluation

**Evidence strength**: Idea 1 is the only candidate whose mechanism matches what the project's own High
Importance evidence says moves top-1 here (generalization/feature-quality, not optimization), and it is
*explicitly prescribed* by project-insights L82. Ideas 2 and 3 are optimization-class changes the
axis-independent polish wall predicts will not raise top-1.

**Mechanism clarity**: Idea 1 has a clear, hierarchy-preserving mechanism (auxiliary gradient sharpens
mid-level features without disturbing the main forward path) and a clean fix for EXP-032's failure mode.
Idea 2/3 mechanisms (faster/flatter optimization) are well-understood but in the closed polish class.

**Risk / value**: Idea 1 is compute-light, eval-clean (no integrity gray area, unlike TTA), graceful-failing,
and directly tests the endorsed generalization lever. Idea 2 is clean but predicted-polish; idea 3 is
predicted-polish with low closure value. Idea 1 is the best risk-adjusted, most insight-aligned probe.

## Chosen Idea
**Selected**: Idea 1 — Deep supervision (auxiliary layer2 classifier, decayed aux loss, eval uses main head only)

**Why this idea**:
After the cooldown axis turned out CLOSED and TTA is integrity-rejected, the remaining open levers split into
(a) optimization-class changes (AdamW/Lookahead) that the axis-independent High Importance polish insight
predicts won't move top-1, and (b) generalization-class changes. Deep supervision is the one
generalization-class lever that is genuinely untried, legitimate, eval-clean, compute-light, AND explicitly
prescribed by the project's own post-mortem of EXP-032 ("use an auxiliary decayed loss on early layers").
It changes what the intermediate features must encode rather than how the optimizer converges — the class the
insight says CAN move top-1 — while preserving the load-bearing coarse-to-fine hierarchy and (via λ→0)
optimizing the pure main objective at the end. Single, well-scoped, low-risk change with clean attribution.

**Hypothesis**:
Adding an auxiliary classifier on layer2 with a decayed cross-entropy loss (λ: 0.3→0) will sharpen mid-level
feature discriminativeness via direct gradient supervision and lift `best_test_acc` above the 96.32 bar at a
throughput-neutral ~88–94 epochs, with the main eval path unchanged. Honest prior: deep supervision's benefit
is depth-scaling, so on this shallow 9-block net the gain may be modest/within-noise. Falsified if
best_test_acc lands within noise of baseline (depth-insufficient) or dt rises enough to underfit (epoch
confound), or if it regresses (aux gradient distorts the main path despite decay).
