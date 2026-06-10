# TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation

- **Source**: Müller & Hutter, ICCV 2021 — https://arxiv.org/abs/2103.10158
- **torchvision**: `torchvision.transforms.TrivialAugmentWide` (available ≥0.12; confirmed in 0.24.1, no new dep)

## Key Insight
TrivialAugment applies exactly **one** augmentation op per image, chosen uniformly at random, at a strength sampled
uniformly at random from the op's range. It is **parameter-free** (no policy search, no magnitude/num_ops tuning) yet
matches or beats the heavily-tuned AutoAugment and RandAugment on CIFAR-10/100 and ImageNet with WideResNets.

## Why it works here (validated, EXP-012)
- Adds **photometric + geometric** diversity (rotate/shear/translate/color/contrast/brightness/sharpness/solarize/
  posterize) — a mechanism **orthogonal** to Cutout (occlusion) and Mixup (interpolation). The canonical strong-aug
  CIFAR-WRN recipe is **TA + Cutout**, which is exactly what worked.
- On this project's k=4 WRN-ResNet-20 + Cutout + 300s budget, adding `TrivialAugmentWide()` lifted best_test_acc
  **96.00 → 96.22** (+0.22pp) with **test loss 0.195 < 0.204** (loss↓ AND acc↑ — a genuine generalization gain),
  at a fair 91-epoch converged run. See reports/exp-report-012.md.

## Practical notes
- Place it on the **PIL image**, before `ToTensor()` (after the geometric crops/flips). It works on PIL or uint8 tensor.
- It is a **single cheap CPU op** with no GPU sync — with 8 dataloader workers it did NOT throttle the launch-bound
  GPU (dt stayed 8ms/step ≈ compiled-k4). This is unlike per-sample CPU ops that call `.item()` (EXP-002 bottleneck).
- Has a `num_magnitude_bins` arg (default 31) — an untried micro-knob.
- Pairs with `torch.compile(reduce-overhead)` for throughput headroom; compile has a null standalone accuracy effect
  here (EXP-007), so gains are attributable to TA.
