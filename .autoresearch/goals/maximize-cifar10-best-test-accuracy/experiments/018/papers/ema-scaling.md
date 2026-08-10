# How to Scale Your EMA
- **Authors**: Dan Busbridge, Jason Ramapuram, Pierre Ablin, Tatiana Likhomanenko, Eeshan Gunesh Dhekane, Xavier Suau Cuadros, Russell Webb
- **Venue**: NeurIPS 2023
- **URL**: https://papers.nips.cc/paper_files/paper/2023/hash/e7681dd6fe16052433ab68cd1555bdc9-Abstract-Conference.html

## Key Contributions
- Studies model exponential moving averages as an optimization object rather than an implementation afterthought.
- Notes that model EMA can improve robustness and generalization in supervised learning.
- Derives a rule for preserving EMA dynamics when batch size changes and validates it across architectures, optimizers, and modalities.

## Relevance

An EMA shadow can be updated outside the synchronized counted interval and evaluated on the existing schedule, potentially smoothing the noisy weak-tail trajectory without reducing optimizer exposure. This experiment keeps batch 128 fixed, so the paper does not select a unique decay; an EMA candidate therefore carries a tunable-timescale and BatchNorm-buffer confound that must be pre-registered or avoided.

## Key Techniques
- Maintain a functional copy whose parameters move toward online weights at a fixed decay.
- Tie the averaging timescale to update frequency when comparing batch regimes.
- Separate online optimization from the model used for evaluation.

