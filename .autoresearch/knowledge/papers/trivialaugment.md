# TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation

- **Source**: arXiv 2103.10158 (Müller & Hutter, ICCV 2021)
- **Relevance**: validated in-project by EXP-004 (+0.17pp on top of RandomErasing at 114 epochs)

## Key points

- TA samples ONE augmentation op + ONE magnitude uniformly per image — no search, no tuning. The "Wide" variant (wider magnitude range) is the recommended/SOTA one and is what `torchvision.transforms.TrivialAugmentWide()` implements (defaults: 31 magnitude bins, NEAREST interpolation).
- CIFAR protocol in the paper: flip + pad-and-crop + TA + **16px cutout applied AFTER TA** — i.e., policy augmentation composes with occlusion erasing by design. Ordering matters: TA on PIL images (before ToTensor), erasing on tensors (after Normalize).
- Paper schedules are 200 epochs (WRN-40-2 / WRN-28-10, batch 128, cosine). Published gains +0.4–0.6pp over baseline aug; at our 114 one-cycle epochs we measured +0.17pp on a recipe already containing RandomErasing.
- In-project finding (EXP-004): mid-schedule accuracy runs several pp BELOW the less-augmented recipe and overtakes only in the final anneal — do not abort on depressed mid-run eval under TA.

## Gotchas

- TA is a PIL-stage transform: must be inserted before ToTensor or it throws on tensors (v1 API).
- CPU cost is one PIL op/image — absorbed by 8 DataLoader workers at ~19k img/s (GPU-bound at 4x width); could become the bound at lower widths or fewer workers.
