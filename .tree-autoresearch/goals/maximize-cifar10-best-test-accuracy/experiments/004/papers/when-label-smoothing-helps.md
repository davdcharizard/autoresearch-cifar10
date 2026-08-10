# When Does Label Smoothing Help?
- **Authors**: Rafael Muller, Simon Kornblith, Geoffrey E. Hinton
- **Venue**: NeurIPS 2019
- **URL**: https://papers.nips.cc/paper_files/paper/2019/hash/f1748d6b0fd9d439f71450117eba2725-Abstract.html

## Key Contributions
- Shows that softening hard targets can improve generalization and calibration.
- Finds that label smoothing changes penultimate-layer representations into tighter class clusters.
- Identifies a tradeoff: the more compressed inter-class structure can harm later knowledge distillation even when classification improves.

## Relevance
The current training already uses soft CutMix labels on roughly half of early batches, but clean batches retain hard targets. Small label smoothing on clean examples is nearly free and could regularize confidence through a distinct output-space mechanism; applying it indiscriminately to mixed batches risks redundant softening.

## Key Techniques
- Cross-entropy against a mixture of one-hot and uniform targets.
- Use small smoothing strength to control overconfidence.
- Evaluate interaction with other soft-label mechanisms rather than assuming additivity.
