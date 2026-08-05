# When Does Label Smoothing Help?
- **Authors**: Rafael Mueller, Simon Kornblith, Geoffrey E. Hinton
- **Venue**: NeurIPS 2019
- **URL**: https://papers.nips.cc/paper_files/paper/2019/hash/f1748d6b0fd9d439f71450117eba2725-Abstract.html

## Key Contributions
- Shows that softening hard class targets can improve generalization and calibration.
- Explains how label smoothing discourages overconfident representations and clusters examples by class.

## Relevance
The successful WRN reached near-zero training loss while test accuracy plateaued at 93.38%, a classic setting where mild target regularization may improve generalization. A late hard-label phase can limit under-convergence risk.

## Key Techniques
- Mix hard one-hot targets with a small uniform label distribution.
- Avoid excessive smoothing or stacking several soft-target methods without calibration.
