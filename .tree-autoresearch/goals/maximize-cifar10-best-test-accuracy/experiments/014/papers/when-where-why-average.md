# When, Where and Why to Average Weights?
- **Authors**: Niccolo Ajroldi, Antonio Orvieto, Jonas Geiping
- **Venue**: ICML 2025
- **URL**: https://proceedings.mlr.press/v267/ajroldi25a.html

## Key Contributions
- Benchmarks checkpoint averaging across seven modern architecture/dataset workloads.
- Finds broad optimization-efficiency gains and milder generalization gains at minimal implementation and memory cost.
- Reports that averaging and learning-rate annealing work best together rather than treating averaging as a replacement for decay.

## Relevance
EXP-011 already validates low-cost late full-state EMA under a cosine-annealed schedule. The paper supports retaining annealing, but it does not identify whether EMA, uniform averaging, or interpolation is best for this 75-second clean/SAM tail. Any descendant must preserve full-state BatchNorm handling and avoid coefficient search on test accuracy.

## Key Techniques
- Average checkpoints along the optimization trajectory.
- Pair averaging with the existing annealed learning-rate regime.
- Compare trajectory diversity and average-to-online distance, not only endpoint accuracy.
