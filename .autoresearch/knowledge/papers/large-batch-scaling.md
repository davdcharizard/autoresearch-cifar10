# Large-Batch Training: Batch-Size / LR Scaling

**Sources**:
- Smith, Kindermans, Ying, Le — "Don't Decay the Learning Rate, Increase the Batch Size" (ICLR 2018), https://arxiv.org/abs/1711.00489
- Goyal et al. — "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour" (2017), linear scaling rule + warmup

## Key Insights

- **Batch↑ ≡ LR-decay (Smith 2018)**: Increasing batch size during training produces the SAME train/test learning curves as decaying the LR. Reaches **equivalent test accuracy after the same number of epochs, but with FEWER parameter updates**. Holds for SGD, momentum, Nesterov, Adam.
- **Linear scaling rule (Goyal 2017)**: multiply LR by k when multiplying batch by k; use a gradual warmup to avoid early-training instability at the higher peak LR. Validated to large batches on ImageNet ResNet.
- **Corollary for a compute-time-gated budget**: if per-step compute time `dt` stays ~flat as batch grows (i.e. the net is launch-bound, not compute-bound), a larger batch processes proportionally MORE images per second of compute. Under a budget gated on Σ`dt` (not wall-clock), that means MORE effective epochs in the same budget — a free-epochs lever, provided the recipe is not already epoch-saturated.

## Relevance to this project (CIFAR-10 k=4 WRN, 300s/H20)

- The 300s budget in `train.py` gates on `total_training_time` = Σ(per-step compute `dt`), with the timer started AFTER the dataloader yields (L218-242) — so #steps ≈ 300/mean(dt) and epochs = steps·batch/50000. Batch scaling adds effective epochs IFF `dt` stays flat.
- k=4 is launch-bound (~8ms/step, VRAM ~0.5GB of 98GB) ONLY at batch 128. EXP-025 tested batch 256 (PEAK_LR 0.2→0.4, warmup 0.05→0.08).
- **EXP-025 RESULT — premise FALSIFIED:** at batch 256 the net is COMPUTE-bound, not launch-bound: dt rose ~8→15ms steady (24-28ms warmup, mean ~21.5ms ≈ 2.7×). So batch scaling did NOT keep dt flat — it collapsed optimizer updates 61% (35.5k→14k steps), cut epochs 91→72, and regressed acc to 93.84 (−2.38pp, loss 0.195→0.258). The compute-`dt`-gated budget rewards MORE updates at smaller effective batch; larger batches strictly reduce updates/images. The launch-bound "free-epochs" lever does NOT exist past batch 128 on this net. Do NOT raise batch size here; always verify the dt-vs-batch curve before assuming launch-bound at a new batch size.
