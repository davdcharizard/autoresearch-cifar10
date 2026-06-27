# Progressive Resizing (fastai DAWNBench / MosaicML Composer) — measured on this project

**Source**: MosaicML Composer progressive-resizing method card; fastai DAWNBench ImageNet recipe (background knowledge). Measured in-project: EXP-031.

## Technique
Train the early portion of a wall-clock budget at reduced input resolution, finish at full resolution. Native home regime is wall-clock-budgeted training (MosaicML claims quality-neutral at −30% wall time on ResNet-50/ImageNet, ramping ~0.7–0.75 linear scale to full res by mid-training).

## Measured result on CIFAR-10 / WRN-20-4x / H20, 300s charged budget (EXP-031)
- Config: 24px for first 50% of budget (in-step charged `F.interpolate` bilinear), 32px second half; `torch.compile(dynamic=False)` + dual-shape startup warmup (both graphs land in startup, switch is stall-free).
- Throughput: 24px dt 13.5ms vs 22.4ms at 32px (0.60×, near the 0.5625 FLOPs ideal — H20 compute-bound at this shape). Epochs 185 vs 139 (+46).
- Quality: best 96.69 vs baseline 96.71 (mean 96.57, σ 0.16) — **zero conversion**. Plateau level unchanged; the low-res phase's advantage is transit-speed only. Switch adaptation is NOT the problem: eval dipped 80.5 at the switch epoch, recovered within one epoch.
- Interpretation: ImageNet 224→160 discards redundancy; CIFAR 32→24 discards signal. The technique's quality-neutrality does not transfer down-scale. Step-conversion laws (EXP-006: +25 full-res epochs = +0.48) are conditional on epochs carrying the SAME training distribution.

## Protocol byproducts (reusable)
- High-epoch runs bust the 600s wall cap via uncharged costs: ~1.3s eval/epoch + loader stalls (8 workers cannot feed 13.5ms steps). Validated levers: eval thinning in non-plateau phases (once/epoch is a ceiling) + 2× loader workers → 457s at 185 epochs.
- Phase-aware watchdog thresholds (per-segment dt expectations keyed on pct_done) work; per-segment post-hoc awk profile in plan-031.
