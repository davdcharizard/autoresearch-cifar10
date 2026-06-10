# Report EXP-022: WRN-style dropout in the residual blocks (p=0.1)
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-022.md
- **Plan**: plans/plan-022.md
- **Log**: logs/exp-log-022.md

## Goal
Maximize CIFAR-10 `best_test_acc` (%), higher-is-better, by editing only `train.py` within a fixed 300s training budget on one H20. Baseline = **96.22%** (EXP-012, commit 6c417a4); success bar = **96.32%** (+0.1pp).

## Idea & Hypothesis
Chosen idea: add WRN-style dropout (`nn.Dropout`, p=0.1) between the two 3×3 convs of every residual block — the placement Zagoruyko & Komodakis 2016 prescribe for Wide ResNets. Selected because the model is generalization-bound and every closed regularization axis lives at a *different* locus (input-aug TA/Cutout, weight-space WD, label-space LS/mixing, trajectory SWA); intermediate-FEATURE dropout is the one distinct, untested locus, and it is the regularizer this exact architecture's own paper recommends. The project's High-Importance meta-insight (don't close a regularization axis from weak-variant nulls — test the strongest mechanistically-distinct variant, as TrivialAugment proved) directly endorsed probing it. Hypothesis: reduced feature co-adaptation lifts best_test_acc above 96.32; alternatively, if acc falls / loss rises, dropout under-fits at the short ~92-epoch budget and the mechanism is closed.

## Approach
Three single-purpose edits to `train.py` (scope = train.py only): (1) `DROPOUT_P = 0.1` constant; (2) `self.dropout = nn.Dropout(p=DROPOUT_P)` in `BasicBlock.__init__`; (3) `out = self.dropout(out)` in `BasicBlock.forward`, between `relu(bn1(conv1(x)))` and `conv2`. p=0.1 chosen (not the paper's 0.3) as a budget-appropriate mild first probe to avoid the under-fitting that sank strong-aug CutMix (EXP-018). Everything else identical to the EXP-012 baseline. `nn.Dropout` is core torch — no new dependency. No deviations from plan.

## Execution
Single run, no retries. Clean startup (params 4,299,866 unchanged, clean compile — no graph break on `nn.Dropout`, no NaN). Completed 84 epochs in 400.7s wall-clock (300.0s training), peak VRAM 537.9MB. Eval correctness automatic: the frozen `Eval.evaluate()` calls `model.eval()`, so dropout is identity at test time.

## Results
- **Primary metric**: best_test_acc = **94.85%** (baseline: 96.22, delta: **−1.37pp**, −1.42%)
- **Observations**: final_test_loss = **0.2236**, ROSE sharply vs baseline 0.195 — a clear UNDER-fit / over-regularization signature (the network cannot fit even the training objective well under the added stochasticity at the budget). num_epochs 84 vs baseline 91 — dropout's per-step RNG mask cost a few epochs, but the large loss rise is far beyond what 7 fewer epochs explains; this is genuine under-fit, not an epoch-count confound.
- **Analysis**: Hypothesis REFUTED. This is the second-largest aug/reg regression on the project (after CutMix −1.08pp) and confirms a now-strong pattern: the EXP-012 recipe (TA + Cutout(16) + LS 0.1 + WD 1e-4) is **regularization-saturated**, and ADDING any further regularizer — stronger weight decay (EXP-005, null), label-mixing aug (Mixup/CutMix, EXP-011/018, regress), or now feature dropout (EXP-022, large regress) — fails because the binding constraint at this short budget is the **opposite** of overfitting: more regularization slows/blocks convergence in the ~84–92 epoch window. The one gain (TrivialAugment, EXP-012) was a *substitution/diversification* of input augmentation that improved generalization without adding a convergence-slowing penalty — not "more total regularization". The meta-insight ("test the strong distinct variant") was correctly applied here, and the answer is now definitive: the distinct feature-dropout locus does not help at this budget either.
- **Key Learning**: WRN in-block dropout (even mild p=0.1) under-fits the already-saturated recipe at the 84-epoch budget (94.85, −1.37pp, loss 0.195→0.224); ADDING regularizers is exhausted — convergence speed, not overfitting, is binding.

## Verification
- **Conditions**: Cond 1 (best_test_acc ≥ 96.32) FAILED (94.85); Conds 2–3 skipped per protocol (would have passed — clean 400.7s run, scope = train.py only, params 4,299,866, eval-count 84 == epochs, nn.Dropout is core torch).
- **Review Notes**: Results confirmed trustworthy — clean exit, params/scope intact, single distinct change, loss-rise diagnostic internally consistent with the under-fit mechanism. No integrity/reward-hacking concerns.
- **Verdict**: no-improvement
- **Verdict Basis**: necessary condition failure (primary metric well below bar and baseline).

## Unexplored Avenues
- **Even smaller p (0.05) or dropout only in the last stage**: could reduce the convergence penalty, but the loss rise at p=0.1 is so large that a smaller p most likely lands ≈ baseline at best (the saturation/convergence story holds). Low value.
- **Dropout WITH a weaker companion regularizer** (e.g. drop LS to 0.05 to "make room"): a co-tune, but multi-knob co-tuning of a saturated recipe is low-evidence and hard to attribute. Low value.
- The broader truth: every "add a regularizer" avenue is now closed. Future gains must come from a mechanism that improves generalization WITHOUT adding a convergence-slowing penalty (like TA did), or from a non-regularization axis entirely.

## Next Steps
- **Per-channel input std-normalization** (std=(1,1,1)→true CIFAR std) — the last untried cheap single-knob probe; it does NOT add a convergence penalty (it's an input rescale), so it sidesteps the saturation wall. Confidence LOW it gains (first layer Conv→BN almost certainly absorbs it → expected null), but it cleanly closes the input-normalization axis in one run. (medium confidence clean null; low confidence gain.)
- **Label-smoothing sweep 0.1→0.05** (brainstorm-022 Idea 2) — REDUCING a regularizer, which (unlike adding one) is consistent with the "convergence-binding, not overfit-binding" finding; the one regularizer-direction not yet tested. Could let the model sharpen within the budget. (low-medium confidence — LS top-1 effects usually small, but direction now has a supporting rationale.)
- After these, ~15 axes are exhausted and the honest call is the **96.22 plateau is generalization/convergence-bound at fixed k=4 capacity in 300s**. Further gains would need more radical, convergence-neutral architecture changes (e.g. BlurPool anti-aliased downsampling, Zhang 2019). (low confidence any single knob clears 96.32.)

## Exit Action Results
<!-- No exit actions defined for this goal. -->
