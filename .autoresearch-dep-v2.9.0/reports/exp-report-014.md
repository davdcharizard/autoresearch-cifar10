# Report EXP-014: Full State Dict EMA (β=0.999)
- **Created**: 2026-05-27
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-014.md
- **Plan**: plans/plan-014.md
- **Log**: logs/exp-log-014.md

## Goal

Maximize CIFAR-10 test accuracy (best_test_acc, %, higher is better). Baseline: 95.39% (EXP-009, commit cfe19c2). Verification threshold: best_test_acc > 95.49% (baseline + 0.1pp).

## Idea & Hypothesis

Full state_dict EMA (β=0.999) — maintain an exponential moving average of the entire model state_dict (including BatchNorm running_mean/running_var buffers) and swap it in for evaluation. This is the direct fix for EXP-013's parameter-only EMA failure, where BN buffers were excluded from the shadow causing eval-time mismatch. Hypothesis: fixing the BN inclusion would unlock EMA's smoothing benefit, achieving > 95.49%.

## Approach

Three localized changes to train.py following plan-014.md exactly:
1. Added `import copy` and `EMA_BETA = 0.999` constant
2. Initialized `ema_shadow = copy.deepcopy(model.state_dict())` after model creation — captures all tensors including BN buffers
3. After `scaler.update()` + `scheduler.step()`, added `torch.no_grad()` block iterating over `model.state_dict()` to update ema_shadow via `lerp_()` for float tensors and `copy_()` for integer tensors (num_batches_tracked)
4. Before evaluation, saved current state_dict via `copy.deepcopy()`, loaded ema_shadow into model, ran eval, restored original state_dict

Used `model.state_dict()` per step (slightly less efficient than caching tensor references) to ensure correctness — state_dict() is the authoritative view of all model state.

## Execution

Single run on H20 GPU with 300s budget. Training completed successfully (exit code 0) with no errors. Achieved 92 epochs (vs ~98 baseline) — the EMA overhead from per-step `model.state_dict()` calls plus `copy.deepcopy()` for eval swap cost ~6 epochs (~17ms/step vs 16ms baseline).

## Results

- **Primary metric**: 95.44% (baseline: 95.39%, delta: +0.05pp, +0.05%)
- **Observations**: EMA with full state_dict fixed the BN mismatch from EXP-013 — accuracy climbed normally from epoch 1 with no early suppression. However, the improvement was marginal (+0.05pp), below the 0.1pp verification threshold. The EMA overhead cost ~6 epochs of throughput. The late-training recovery pattern from EXP-013 was less pronounced because the BN fix eliminated early suppression, making EMA and non-EMA trajectories closer throughout training.
- **Analysis**: The hypothesis was partially validated — full state_dict EMA does fix the BN mismatch and provides a small accuracy benefit. However, β=0.999 is too conservative for a 92-epoch budget: the EMA shadow lags too far behind the rapidly changing weights in early training, and the throughput cost (6 fewer epochs) partially offsets the smoothing benefit. The net effect is near-zero. EMA's effectiveness scales with epoch count — with only 92 epochs, the window for EMA to provide meaningful smoothing after the final LR drop is too short.
- **Key Learning**: Full state_dict EMA correctly fixes BN mismatch but β=0.999 is too conservative for ~92-epoch training — the throughput cost (~6 epochs) nearly offsets the smoothing benefit, yielding only +0.05pp.

## Verification

- **Conditions**: Condition 1 (best_test_acc > 95.49%) FAILED; Conditions 2-3 PASSED
- **Review Notes**: Results confirmed trustworthy — metric plausible and consistent with training dynamics observed
- **Verdict**: no-improvement
- **Verdict Basis**: Primary metric 95.44% did not exceed verification threshold of 95.49% (baseline 95.39% + 0.1pp)

## Unexplored Avenues

- **Lower β (0.99 or 0.995)**: More aggressive averaging would track faster-moving weights better during the short training window, reducing the lag penalty. Literature suggests β should be tuned to epoch count — hlb-CIFAR10 uses a per-epoch-count β schedule.
- **Cached tensor references instead of per-step state_dict()**: Maintaining direct references to model parameters and buffers (via `named_parameters()` + `named_buffers()`) would eliminate the per-step `model.state_dict()` overhead, recovering ~6 epochs. Combined with lower β, this could tip the balance.
- **EMA warmup (no EMA for first N epochs)**: Skip EMA updates during early high-LR training where weights change rapidly, then enable EMA only in the last 50-75% of training where smoothing is most beneficial.

## Next Steps

1. **Mixup α=0.2 replacing RandomErasing** (medium confidence) — cross-sample regularization orthogonal to TrivialAugmentWide, replacing rather than stacking per EXP-010 lesson. Lower α avoids the over-regularization that hurt CutMix.
2. **Lower EMA β (0.995) with cached references** (medium confidence) — addresses both failure modes of EXP-014: too-conservative β and throughput overhead. But requires careful implementation of buffer tracking.
3. **Label smoothing 0.2 standalone** (low confidence) — hlb-CIFAR10 uses 0.2 (not 0.1 as in failed EXP-004). Different α and no Nesterov overhead might change the outcome.

## Exit Action Results
