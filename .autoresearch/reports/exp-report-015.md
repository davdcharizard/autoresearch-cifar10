# Report EXP-015: Pre-activation (true-WRN) BasicBlocks — BN→ReLU→conv, bare-conv shortcuts, final BN→ReLU
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-015.md
- **Plan**: plans/plan-015.md
- **Log**: logs/exp-log-015.md

## Goal
Maximize CIFAR-10 `best_test_acc` (%), higher-is-better, editing only `train.py` within the
fixed 300s training budget on a single H20. Baseline = **96.22%** (EXP-012, commit 6c417a4);
success bar = **96.32%** (+0.1pp). This experiment tested whether converting the model's
post-activation ResNet-v1 blocks to the canonical pre-activation WideResNet formulation lifts accuracy.

## Idea & Hypothesis
Chosen idea: convert the post-activation `BasicBlock` (conv→BN→ReLU ×2, ReLU after the residual
add) to the canonical **pre-activation** form (He et al. 2016, arXiv:1603.05027): `out = relu(bn1(x))`
feeding both `conv1` and a bare 1×1 projection shortcut, then `conv2(relu(bn2(out)))`, `out += shortcut`,
**no** post-add ReLU; stem becomes a bare conv (stem BN dropped) and a final BN→ReLU is added before
global pooling. Selected because the augmentation axis is now mapped (TA wins, policy saturated) and
pre-activation is the most evidence-backed *structural* lever left — our model is described as
"WideResNet-style" but actually runs ResNet-v1 blocks, and pre-act is the formulation true WRN uses,
at ~zero compute cost. Hypothesis: cleaner identity/gradient path lifts `best_test_acc` above 96.32
(expected ~96.3–96.6) with a corroborating `final_test_loss ≤ 0.195` at unchanged throughput/epochs.

## Approach
Single `train.py` edit (no hyperparameter changes — inherited the full EXP-012 recipe: k=4, batch 128,
peak LR 0.2 cosine, Nesterov, WD 1e-4, LS 0.1, Cutout(16), TrivialAugment, torch.compile, seed 42):
1. **`BasicBlock` → pre-activation**: `bn1` resized to `in_channels`; forward = `relu(bn1(x))` →
   shared pre-activated input feeds `conv1` and the projection shortcut → `conv2(relu(bn2(out)))` →
   `out += shortcut`, no post-add ReLU. Downsample shortcut is now a bare 1×1 conv (no BN); identity
   blocks store `shortcut = None` and add the raw `x` (true identity path).
2. **`ResNet`**: dropped stem `bn1` (bare `conv1` stem), added `self.bn_final = BatchNorm2d(w3)`,
   forward = `conv1 → layer1..3 → relu(bn_final) → avgpool → fc`.
No deviations from plan-015. Ruff clean; `git diff` = train.py only.

## Execution
One run, no retries. Clean startup: `num_params = 4,298,970` (−896 from baseline, exactly the
expected BN-restructuring delta — not a capacity change), clean compile, no traceback, no NaN.
Training healthy (ep 1 test_acc 44.42%, loss descended normally). Run exited 0 in 403.4s total
(300.0s training) < 600s budget. **Ran only 78 epochs / 30,246 steps** — notably fewer than EXP-012's
91 epochs; dt jittered 8–16ms (avg ~12,900 img/s).

## Results
- **Primary metric**: best_test_acc = **95.85%** (baseline: 96.22, delta: **−0.37pp**, −0.38%)
- **Observations**: final_test_loss 0.2012 (> EXP-012's 0.195); only 78 epochs vs 91. Pre-activation
  was **NOT throughput-neutral in practice** — the restructured block + extra final BN→ReLU evidently
  produced a less-efficient torch.compile graph (or more launch overhead), costing ~14% of the step
  budget. 95.85 sits right in the compiled-k4 null band (≈95.92 ± noise from EXP-007/008/010/011).
- **Analysis**: Hypothesis NOT supported. Two effects are confounded: (a) pre-act's expected
  accuracy benefit is small-to-null on a *shallow* ResNet-20 (pre-act's documented gain is largest for
  very deep nets — He 2016), and (b) the realized throughput drop cost ~13 epochs of training. Even
  generously attributing the whole −0.37pp to lost epochs, there is no positive signal: pre-act did not
  clear the bar and the loss rose. The block-ordering axis is settled for this shallow k=4 net.
- **Key Learning**: Pre-activation (true-WRN block ordering) gives no accuracy gain on this shallow
  k=4 ResNet-20 and is not throughput-neutral under torch.compile (78 vs 91 epochs) — block-ordering axis closed.

## Verification
- **Conditions**: Cond 1 (clean completion < 600s, no traceback) PASS; **Cond 2 (best_test_acc ≥ 96.32) FAIL** (95.85); Cond 3 (scope) skipped — not reached after Cond 2 failed (scope was clean for the record: train.py only, eval-count 78 == num_epochs, seed 42 intact, params sane).
- **Review Notes**: Results confirmed trustworthy — clean run, sane param count, scope intact, no parsing anomalies. The fewer-epochs confound is noted but does not rescue the result (no positive signal in either acc or loss).
- **Verdict**: no-improvement
- **Verdict Basis**: condition failure (primary metric did not clear the bar; −0.37pp below baseline).

## Unexplored Avenues
- **Pre-activation WITHOUT torch.compile** (or with `mode="default"`): would isolate the pure
  accuracy effect of block ordering from the compile-graph throughput penalty. Low value, though —
  eager pre-act at k=4 would run far fewer epochs and the per-epoch accuracy was already not better.
- **Pre-activation only helps deeper/wider nets**: the literature gain is depth-driven; would require
  more blocks/depth, which is blocked by the 300s epoch wall (width/depth axis already closed). Not promising.

## Next Steps
1. **LR-schedule micro-tuning on the TA recipe** (peak LR 0.2 → 0.15 or 0.25; warmup fraction) —
   the natural fallback from brainstorm-015 Idea 2; only WD was ever swept (EXP-005, old recipe). Trivial,
   compute-free, orthogonal to the (now-mapped) augmentation and (now-closed) architecture axes. Confidence: low-medium.
2. **A different aug *mechanism*** stacked on the TA+Cutout(16) recipe (e.g. CutMix, or larger Cutout ≥16) —
   the augmentation-strength axis pointed toward "more aug" (TA gained, less-aug lost). Confidence: low.
3. Failing both, the honest call is that the **96.0/96.22 regime is generalization-bound** at fixed k=4
   capacity within 300s — ~9 axes now exhausted. Confidence: medium that little remains above noise.

## Exit Action Results
- No exit actions defined for this goal — skipped.
