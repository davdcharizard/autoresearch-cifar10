# CIFAR-10 Label Errors & Confident Learning — closure source for EXP-069
- **Sources**: Confident Learning (arXiv 1911.00068); Pervasive Label Errors in Test Sets
  (arXiv 2103.14749); https://l7.curtisnorthcutt.com/label-errors
- **Status**: measured-closed for this project (EXP-069, NO-LAUNCH closure)

## Key Facts
- CIFAR-10 test set: 54 CONFIRMED mislabels (0.54%) out of 221 suspected (MTurk-validated,
  arXiv 2103.14749). Train set comes from the same labeling pipeline — natural noise rate
  ~0.5–1% suspected, confirmed-rate lower.
- The often-quoted "+0.9pp from cleaning CIFAR-10" (Confident Learning) is measured under
  **20–40% ADDED synthetic noise** — two orders of magnitude above the natural rate. No
  published ≥0.3pp cleaning gain exists at the natural rate.

## Relevance
EXP-069 closed the label-noise-curation sub-class on this arithmetic: ≤1% correctable labels
under LS 0.1 + TA+RE (both measured noise-robustness suppliers, EXP-036/050/051) gives a
≤0.1pp effect ceiling ≪ the +0.28 effect-size screen. Identification also prices on the
charged budget (cost-landing fail); imported precomputed error lists ride on externally
trained models (pretrained-knowledge class).

## Reuse Rule
When any future candidate cites dataset-cleaning gains, regime-check the NOISE RATE of the
published experiment against the natural rate before treating the gain as evidence —
noise-rate mismatch voids transfer exactly like augmentation-regime mismatch (EXP-037 law).
