# Report EXP-000: Budget-matched modern training recipe (same ResNet-20)
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-000.md
- **Plan**: plans/plan-000.md
- **Log**: logs/exp-log-000.md

## Goal
Maximize best_test_acc (%) of the CIFAR-10 ResNet-20 baseline within the fixed 300s wall-clock training budget, modifying only train.py. Direction: higher is better. Baseline at experiment start: 91.97% (BASE row, commit 14247e0). Question tested: does a training recipe matched to the actual time budget beat the 2016 step-decay recipe whose schedule never completes?

## Idea & Hypothesis
Chosen from three candidates (recipe modernization, GPU-resident data pipeline, shallow-wide architecture swap): modernize the recipe on the unchanged architecture. Rationale: the baseline demonstrably truncates its LR schedule (only ~37.5k of 64k scheduled steps fit in 300s, so the second LR drop never fires), and super-convergence/cifar10-fast evidence says budget-matched one-cycle schedules dominate truncated step decay. Hypothesis: time-keyed one-cycle (peak 0.4, 15% warmup) + bf16 + TF32 + channels_last + batch 512 (nesterov, selective WD, label smoothing 0.1) reaches ≥92.5% via (a) the anneal completing and (b) ≥1.5× more epochs in the same budget.

## Approach
All changes in train.py (only permitted file): replaced MultiStepLR with a module-level `lr_at(progress)` one-cycle function keyed to `total_training_time / TIME_BUDGET_S` — the anneal completes regardless of realized throughput; batch 128→512 with peak LR 0.4 (linear scaling); SGD nesterov with two param groups (WD 5e-4 on ndim>1 weights, 0 on BN/bias); label_smoothing 0.1; bf16 autocast around forward+loss; TF32 (`set_float32_matmul_precision("high")`); `cudnn.benchmark=True`; channels_last for model and inputs; `persistent_workers=True`; MAX_STEPS raised to non-binding 1M. Eval cadence (once/epoch), seed 42, summary block, and per-step synchronize timing untouched. Deviation from plan: branch named `autoresearch/exp-000` (skill convention) instead of the plan's `exp/000-budget-matched-recipe`.

## Execution
Single run, no retries, no adjustments. Launched as background process on GPU 0 (confirmed free), output to run.log. Early signal at epoch 1 healthy (test_acc 43.98%, no divergence under warmup). Throughput ~60k img/s (vs ~16k baseline). Run completed cleanly, exit 0. No entries in Errors & Dead Ends.

## Results

- **Primary metric**: best_test_acc 93.16% (baseline: 91.97%, delta: +1.19 pp, +1.29%)
- **Observations**:
  - Throughput 3.75× (60k vs 16k img/s) from bf16+channels_last+TF32+batch 512 → 345 epochs vs 97 in the same 300s; steps/epoch dropped 387→97 (batch 512), so total steps were slightly fewer (33.5k vs 37.5k) — the win is *images* seen and schedule shape, not step count.
  - final_test_acc 93.01% is itself +1.04 pp over baseline — the gain is genuine end-of-training improvement, not max-over-345-evals harvesting.
  - total_seconds 596.7 — only 3.3s under the 600s hard cap. Per-epoch eval (~0.85s × 345 ≈ 295s) now rivals the training budget itself. This is the binding risk for any further throughput gains.
  - peak_vram_mb 479.7 (vs 330.1) — nothing on a 98GB H20.
- **Analysis**: Hypothesis confirmed and exceeded (predicted ≥92.5%, got 93.16%). Both mechanisms delivered: the completed cosine anneal (final LR ~0) plus 3.6× more epochs. The recipe foundation (schedule, precision, layout, batch) is now in place for the higher-ceiling follow-ups (GPU-resident pipeline, wider architecture).
- **Key Learning**: The binding defect was schedule truncation, and wall-clock is now dominated by eval overhead, not training — future throughput gains must manage the 600s total cap, e.g. via fewer epochs (more steps/epoch) or accepting it as the new constraint frontier.

## Verification
- **Conditions**: all passed — (1) clean completion, total 596.7s ≤ 600s; (2) 93.16 ≥ 92.07 (baseline+0.1); (3) evals = 345 = num_epochs (once per epoch)
- **Review Notes**: results confirmed trustworthy — diff confined to train.py, seed unchanged (42), eval harness untouched, final_test_acc independently confirms the gain; improvement came through the intended intervention class (training recipe).
- **Verdict**: improvement
- **Verdict Basis**: all conditions passed + meaningful improvement (+1.19 pp ≫ 0.1 pp bar)

## Unexplored Avenues
- **GPU-resident data pipeline** (brainstorm idea 2): host DataLoader still bounds throughput; airbench-style on-GPU augmentation could push img/s several-fold — but the 600s wall-clock cap means raw epoch count cannot grow much; pair with larger batch (fewer, bigger steps) or it converts into eval-overhead, not accuracy.
- **Shallow-wide architecture** (brainstorm idea 3): ResNet-9-style or 2-4× wider ResNet-20 — the speedrun-proven path to 94%+; now attributable cleanly on top of the tuned recipe.
- **Recipe micro-tuning**: peak LR sweep (0.3–0.6), warmup fraction, TrivialAugment/Cutout addition, test-time-free tricks (e.g., derandomized flip from airbench).
- **Eval-overhead management**: with 345 epochs the run barely fit 600s; future experiments that raise throughput should consider larger batch (fewer epochs at same images/s) to keep eval count down.

## Next Steps
1. **Widen the architecture on top of this recipe** (e.g., 2-4× wider ResNet-20 or ResNet-9 style) — high confidence: capacity is now the limiting factor; speedrun literature shows 94%+ at this time scale; wider also means fewer img/s → fewer epochs → relieves the 600s cap.
2. **GPU-resident augmentation pipeline** — medium confidence: clear throughput win, but must be converted into accuracy via batch size rather than epoch count due to the wall-clock cap.
3. **Augmentation upgrade (Cutout/RandomErasing)** — medium confidence: standard +0.3-0.5pp on CIFAR-10 at ResNet-20 scale, cheap to add via torchvision transforms.

## Exit Action Results
