# When Does Label Smoothing Help?
- **Authors**: Rafael Muller, Simon Kornblith, Geoffrey E. Hinton
- **Venue**: NeurIPS 2019
- **URL**: https://papers.nips.cc/paper_files/paper/2019/hash/f1748d6b0fd9d439f71450117eba2725-Abstract.html

## Key Contributions
- Shows that softening hard targets can improve generalization and calibration.
- Finds tighter within-class representations but reduced inter-class similarity information.

## Relevance
Small label smoothing is computationally cheap, but its marginal value must be tested carefully when Mixup or CutMix already supplies soft targets.

## Key Techniques
- Cross-entropy against a one-hot/uniform target mixture.
- Confidence regularization through target smoothing.
