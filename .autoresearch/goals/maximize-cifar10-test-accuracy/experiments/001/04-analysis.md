# Report EXP-001: ResNet-9 (DavidNet) + time-based one-cycle on CIFAR-10
- **Created**: 2026-06-28

## Goal
Maximize `best_test_acc` (%) on CIFAR-10 (higher is better) within a fixed 300s wall-clock **training-time** budget, editing only `train.py`. Baseline (from `04-results.tsv`): **91.57%** (unmodified CIFAR ResNet-20). Improvement bar: ≥ +0.1pp (≥ 91.67%).

## Idea & Hypothesis
Chosen from brainstorm (Idea 02, Codex review pick): replace the deep-thin ResNet-20 + `MultiStepLR` with the wide-shallow 9-layer residual net ("DavidNet"/ResNet-9) from David Page's cifar10-fast, trained under a **time-based triangular one-cycle** LR plus the standard fast-CIFAR stack (Cutout, label smoothing, bf16). Reasoning: the baseline's milestone schedule never reaches its 2nd LR drop within 300s, so the model is read under-annealed; a one-cycle that *completes* its anneal within budget, on a higher-capacity net that converges in far fewer epochs, is the path multiple independent implementations replicate at 94.0–94.3%. Hypothesis: this raises `best_test_acc` to ~93.5–94.3%, clearing the bar with large margin.

## Approach
Single-file rewrite of `train.py` (only editable file; `prepare.py` frozen). Key changes:
- **Architecture**: ResNet-20 (269,722 params) → DavidNet/ResNet-9 (6,573,120 params): prep 3→64, stages 64→128(+Residual)→256→512(+Residual) with MaxPool(2) each, global MaxPool(4), bias-free Linear 512→10, logits ×0.125 (`scale_out`).
- **Schedule**: removed `MultiStepLR`; LR set each step from `progress = total_training_time / TIME_BUDGET_S` — linear ramp 0→0.4 over the first 15% of the budget, then linear decay 0.4→0. This is the review-driven refinement (brainstorm concern #1): keying on elapsed *training time* needs no off-budget step calibration, can't overrun, and guarantees the anneal completes by 300s regardless of throughput.
- **Optimizer/loss**: SGD+Nesterov (mom 0.9, wd 5e-4); `CrossEntropyLoss(label_smoothing=0.2)`, mean reduction. Convention pinned together: peak LR 0.4 (mean-loss), wd 5e-4, scale_out 0.125.
- **Regularization/throughput**: Cutout 8×8 (pure-torch, after Normalize) on top of pad-4-crop + flip; bf16 autocast + `channels_last` + `cudnn.benchmark` (no GradScaler); batch 512; train DataLoader `persistent_workers=True, prefetch_factor=4`.
- **Preserved verbatim**: the `while total_training_time < TIME_BUDGET_S` loop, per-step `synchronize()`/`dt` budget meter, the single `evaluator.evaluate` per epoch, `best_acc` tracking, `torch.manual_seed(42)`, and the summary prints. No TTA (keeps eval single-forward). Normalization kept identical to the frozen eval (mean=(0.4914,0.4822,0.4465), std=(1,1,1)). Added a per-epoch `wall:` print for live 10-min-cap monitoring.

## Execution
One run, no retries, no errors. Launched on **GPU 1** under `timeout 600 bash -c 'CUDA_VISIBLE_DEVICES=1 uv run train.py > run.log 2>&1'`. Milestone 1 (implement) passed `py_compile` + a one-batch bf16/channels_last fwd+bwd+step smoke test (confirmed output (8,10), 6.57M params) before the official run. Training was healthy from step 1 — no divergence — at ~32,000 img/s (≈2× baseline) and ~16ms/step. **192 epochs / 18,529 steps** fit in the 300s training budget; total wall 447.4s (comfortably under the 600s cap); startup 1.2s (env pre-built); peak VRAM 1.6 GB.

## Results
- **Primary metric**: **95.22%** (baseline: 91.57%, delta: **+3.65pp**, +3.99%)
- **Observations**: Accuracy climbed smoothly with the one-cycle anneal and jumped sharply in the final ~20% of training as LR→0: ep10 84.19% → ep102 89.91% (@55% progress, LR≈0.21) → ep187 95.05% → final ep192 best 95.22%. The bulk of the gain came in the low-LR tail — direct confirmation of the under-annealing diagnosis. Result **exceeded** the ~93.5–94.3% hypothesis by ~1pp, landing in airbench-95 territory despite using a plain ResNet-9 (no whitening/TTA). Likely because the 300s budget allowed ~192 epochs — far more than DavidNet's canonical 24 — under a single completing cycle, and the H20 throughput (bf16+channels_last+batch 512) made those epochs cheap.
- **Analysis**: Hypothesis validated and then some. Both levers contributed: (a) a schedule that actually finishes annealing within budget, and (b) a higher-capacity, faster-converging architecture. The 6.5M-param net used only 1.6 GB of 98 GB — VRAM is nowhere near a constraint, leaving large headroom for wider/deeper nets.
- **Key Learning**: Under a fixed *training-time* budget, a time-keyed one-cycle that completes its anneal on a wide-shallow ResNet-9 (bf16+channels_last) reaches ~95.2% — the schedule-completion + architecture swap is worth +3.65pp over the ResNet-20 baseline.

## Verification
- **Conditions**: all passed. (1) runs clean, exit 0, total 447.4s < 600s; (2) training_seconds=300.0 (≥295) and `prepare.py` byte-unchanged with `TIME_BUDGET_S=300`; (3) best 95.22 ≥ 91.67 (Python float compare); (4) only `train.py` changed, seed `torch.manual_seed(42)` intact (no seed search), one `evaluator.evaluate` call, no test-set/eval-internals access.
- **Review Notes**: Results confirmed trustworthy — 95.2% is consistent with the fast-CIFAR literature for this net class; metric came from the frozen eval harness (single forward, matched normalization, no TTA); no reward-hacking surface touched.
- **Verdict**: improvement
- **Verdict Basis**: all necessary conditions passed + meaningful improvement (+3.65pp, well above the +0.1pp bar).

## Unexplored Avenues
- **Whitening initial conv (airbench/Idea 03)**: a frozen eigendecomposition-initialized first conv accelerates convergence further; documented path to 95–96%+. Now the natural next step on a stable fast-CIFAR base.
- **Flip TTA inside `forward`** (eval-mode gated): a legitimate ~+0.2–0.4pp at eval, untried here to keep eval single-forward; cheap to add.
- **Wider/deeper net**: only 1.6 GB VRAM used — widths could grow substantially, or add the airbench residual ConvGroup ("96-net") depth, given the large epoch budget.
- **Recipe tuning**: LR peak sweep {0.2,0.4,0.6}, label-smoothing 0.1 vs 0.2, Cutout size, BN/no-decay-on-bias, EMA of weights — each plausibly worth tenths of a pp.
- **More throughput** (torch.compile warmed in startup, larger batch) → more epochs in budget, though returns may diminish past ~192 epochs of one cycle.

## Next Steps
1. **Whitening front-end on the ResNet-9 base** (high confidence) — add a frozen whitening conv (Idea 03's core mechanism) to the now-working DavidNet; documented to push toward 95–96%+.
2. **Add flip TTA + light recipe tuning** (medium-high) — eval-mode flip averaging plus an LR/label-smoothing/EMA sweep on the current net for cheap incremental gains.
3. **Scale model width/depth** (medium) — exploit the 96 GB / 1.6 GB VRAM headroom with a wider or deeper net under the same time-based one-cycle.

## Exit Action Results
<!-- No exit actions defined for this goal. -->
