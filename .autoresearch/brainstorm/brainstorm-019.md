# Brainstorm EXP-019
**Created**: 2026-06-08
**Goal**: goals/improve-cifar10-test-accuracy.md

## Web Search & Literature Review

- **SWA — Izmailov et al. 2018, "Averaging Weights Leads to Wider Optima and Better Generalization" (UAI 2018, arXiv:1803.05407)** (knowledge/papers/swa.md to be created)
  Core claim: averaging multiple points along the SGD trajectory that are sampled under a HIGH/CONSTANT (or cyclic) LR finds a solution in a WIDER, flatter region of the loss surface that generalizes better than any single SGD iterate, at essentially no extra training cost. Reported consistent CIFAR-10/100 gains (~0.5–1.3pp) over SGD-with-the-same-budget across VGG/PreResNet/WideResNet. The CRITICAL precondition: the LR in the averaging phase must stay at a non-trivial floor (constant or cyclic) so the iterate keeps MOVING through the flat region — averaging a settled, annealed-to-0 tail reproduces the endpoint and adds nothing. BN running stats must be recomputed for the averaged weights before evaluation (a forward pass over training data) because the averaged params were never the live BN-tracking model. Implemented in core PyTorch via `torch.optim.swa_utils.AveragedModel` + `update_bn` (no new dependency).
- **SAM — Foret et al. 2021, "Sharpness-Aware Minimization for Efficiently Improving Generalization" (ICLR 2021, arXiv:2010.01412)**
  Minimizes loss in an ε-ball (flat minima) via a two-step gradient (ascent to worst-case perturbation, then descent). Strong CIFAR generalization gains BUT ~2× forward+backward per step. Relevant as an alternative generalization-targeting lever; the 2× cost is the concern under our fixed 300s budget.

## Experimental History Review

Current best = **96.22%** (EXP-012, commit 6c417a4): k=4 WideResNet (4.3M params) + Cutout(16) + TrivialAugmentWide + torch.compile(reduce-overhead), bf16, channels_last, Nesterov SGD, time-fraction cosine LR (peak 0.2, anneal→0), label smoothing 0.1, WD 1e-4. ~91 epochs, loss 0.195.

18 experiments; ~11 axes closed. Binding constraint diagnosis: **generalization at fixed k=4 capacity in 300s** (capacity scaling hits a monotone epoch wall, EXP-004/009; optimization recipe — LR-peak, activation, block-order — all settled).

Closed axes (do NOT revisit): capacity k>4 (epoch wall, EXP-004/009); LR-peak (interior optimum at 0.2, EXP-016/017); block micro-architecture/pre-activation (EXP-015); activation SiLU (EXP-010); channel attention SE (EXP-008); weight decay (EXP-005); more-epochs/throughput alone (EXP-007); auto-aug POLICY (EXP-014); aug STRENGTH down (EXP-013); label-mixing aug Mixup/CutMix (EXP-011/018, Medium-importance closed).

**Untried gap that this experiment targets**: EXP-006 (Low Importance failed approach) evaluated an EMA copy (decay 0.999) bolted onto the cosine-to-0 schedule → 95.97, no gain. The recorded Insight is explicit: *"weight averaging needs a moving iterate (constant/cyclic terminal LR) to help; it's a no-op with cosine-to-0. **Don't retry EMA/SWA variants unless the schedule is changed to keep a terminal-LR floor.**"* The one condition under which the learnings sanction a retry has NEVER been tested — proper SWA with a constant-LR averaging tail is a genuine gap, not a repeat. This is the single best-evidenced remaining lead that directly attacks the binding generalization constraint without adding capacity (which the epoch wall forbids).

## Candidate Ideas

### 1. SWA with a constant-LR averaging tail (proper Stochastic Weight Averaging)
**Summary**: Change the LR schedule from "cosine peak 0.2 → 0 over the full budget" to "cosine peak 0.2 → a moderate floor (the SWA LR, e.g. ~0.05) reached at ~75% of the time budget, then HOLD that floor constant for the final ~25%." During that constant-LR tail, maintain a running average of the model weights (one snapshot per epoch via `torch.optim.swa_utils.AveragedModel`). Per epoch in the tail, recompute BatchNorm running statistics for the averaged model (a short forward-only pass over a subset of training batches — partial-pass BN estimate, cheap) and evaluate the AVERAGED model (so per-epoch eval count is unchanged: raw model during the main phase, SWA model during the tail). `best_test_acc` is the max over all epochs as before. No new dependency (core `torch.optim.swa_utils`), train.py-only, params unchanged → throughput-near-neutral.

**Reasoning**: SWA directly targets the binding constraint — generalization at fixed capacity — by finding a flatter, wider optimum via trajectory averaging, the exact mechanism the literature shows yields ~0.5–1.3pp on comparable CIFAR WRN/ResNet setups (Izmailov 2018). It is the precise variant the goal-learnings sanction: EXP-006 failed *only* because cosine-to-0 settled the iterate; supplying a constant-LR floor keeps the iterate moving through the flat region so the average is meaningfully different from (and flatter than) any endpoint. Near-compute-neutral (averaging is free; BN-recompute kept cheap via a partial pass), so unlike capacity/SAM it does NOT pay the epoch-wall tax.

**Sources**: knowledge/papers/swa.md (to create from arXiv:1803.05407); goal-learnings EXP-006 Insight ("retry SWA IF terminal-LR floor"); project-insights High "generalization-bound at fixed capacity"; `torch.optim.swa_utils` (core torch).

**Estimated Effort**: medium — LR-schedule edit + AveragedModel maintenance + a cheap BN-recompute helper + branching which model is evaluated in the tail. All localized to train.py's loop and `lr_at_fraction`.

**Risk Assessment**: Main risk — the constant-LR tail forgoes the cosine-to-0 final sharpening, so if averaging fails to recover that benefit the run could mildly regress (~−0.3pp); fails gracefully to no-improvement. Secondary risks: BN-recompute overhead eating epochs (mitigate with a partial-batch pass, ~50–100 batches); choosing the SWA floor LR and tail-fraction (literature default ≈ a moderate constant; pick from cosine value at ~75%). Assumption: the seeded eval harness accepts an arbitrary nn.Module (it does — `evaluator.evaluate(model, device)` is generic).

### 2. Sharpness-Aware Minimization (SAM)
**Summary**: Replace the plain SGD step with SAM's two-step update — compute gradient, ascend to the worst-case ε-perturbation of weights, recompute gradient there, then apply the descent step with the base optimizer. Targets flat minima for better generalization.

**Reasoning**: Like SWA, SAM attacks generalization directly and has strong CIFAR evidence (Foret 2021). It is a genuinely untried mechanism orthogonal to everything closed so far.

**Sources**: arXiv:2010.01412; project-insights High "generalization-bound at fixed capacity".

**Estimated Effort**: medium — implement the two-step perturbation loop (no library needed) within the training loop.

**Risk Assessment**: SAM roughly DOUBLES forward+backward per step → on this net ~8ms→~15ms/step → ~halved epochs (~40–45). EXP-009 (k=5, 41 epochs) showed severe undertraining (94.21) at that epoch count — SAM is highly likely to hit the SAME epoch wall and regress, the dominant failure mode here. Lower expected value than SWA, which is compute-neutral. Efficient-SAM variants exist but add complexity/risk.

### 3. Per-channel input std-normalization
**Summary**: The code normalizes with `std=(1,1,1)` (only the per-channel mean is subtracted). Replace with the true CIFAR-10 per-channel std (≈(0.247,0.243,0.261)) so inputs are unit-variance per channel.

**Reasoning**: Standard preprocessing; in principle conditions the input distribution better. Cheap, one-line, low-risk probe — the one remaining untried input-side knob.

**Sources**: train.py L152-155 (the `std=(1,1,1)` comment explicitly flags this); standard CIFAR practice.

**Estimated Effort**: low — change one tuple.

**Risk Assessment**: The first layer is Conv→BatchNorm; BN almost certainly absorbs a per-channel affine rescale of the input, making this an expected NULL (within noise). Low ceiling — useful mainly to definitively close the input-normalization axis. Won't destabilize anything.

## Idea Evaluation

**Evidence strength**: SWA and SAM both have strong, comparable-setting CIFAR literature; SWA additionally is the variant the project's own goal-learnings explicitly name as the sanctioned retry condition (EXP-006) — that is the strongest possible project-specific evidence that this is the right next probe. Idea 3 has weak ceiling (BN absorbs it).

**Mechanism clarity**: All three have clear mechanisms. SWA's is the most defensible *for our exact situation*: EXP-006 isolated WHY weight averaging failed (no moving iterate), and SWA supplies precisely the missing condition. SAM's mechanism is sound but its 2× cost collides with the well-established epoch wall.

**Expected impact**: SWA highest — compute-neutral, so a literature-scale gain (even discounted for our short budget + strong recipe) plausibly clears the +0.1pp bar and the ~0.2pp noise floor. SAM's expected impact is dragged negative by halved epochs (EXP-009 precedent). Idea 3 expected ≈ 0.

**Risk profile**: SWA fails gracefully (mild regression at worst). SAM's most likely outcome is a regression from undertraining. Idea 3 is safe but near-null.

**Feasibility**: SWA and SAM comparable (medium); Idea 3 trivial. SWA's medium effort is justified by far the best evidence+impact.

Conclusion: **SWA (Idea 1)** dominates on evidence, mechanism fit to our diagnosed constraint, expected impact, and risk profile. SAM is the natural fallback only if SWA succeeds and we want to push generalization further with a bigger budget; Idea 3 is a cheap axis-closer, not a lead.

## Chosen Idea
**Selected**: SWA with a constant-LR averaging tail (proper Stochastic Weight Averaging)

**Why this idea**:
It is the single remaining lever that is simultaneously (a) backed by strong comparable-setting literature (Izmailov 2018, ~0.5–1.3pp on CIFAR WRN/ResNet), (b) explicitly sanctioned by this goal's own learnings as the untested condition under which weight averaging should help (EXP-006: "retry SWA IF the schedule keeps a terminal-LR floor"), (c) a direct attack on the diagnosed binding constraint (generalization at fixed k=4 capacity), and (d) near-compute-neutral, so it sidesteps the epoch wall that sinks capacity scaling and SAM. EXP-006 did not test SWA — it tested EMA on a cosine-to-0 schedule, which the analysis showed is a no-op; proper SWA with a constant-LR tail is a genuine, well-motivated gap.

**Hypothesis**:
Replacing the cosine-to-0 schedule with cosine→moderate-floor (≈0.05) for ~75% of the budget then a constant-floor averaging tail for the final ~25%, while evaluating the BN-recomputed weight-average in the tail, will lift `best_test_acc` above the 96.32 bar (+0.1pp over 96.22) by converging to a flatter, better-generalizing optimum at no epoch cost. If it fails, the most likely outcome is a mild regression (~−0.3pp) from forgoing cosine-to-0 sharpening without sufficient flat-region averaging benefit — which would definitively close the weight-averaging axis for this budget.
