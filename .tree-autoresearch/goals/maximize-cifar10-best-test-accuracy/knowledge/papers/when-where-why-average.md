# When, Where and Why to Average Weights?
- **Authors**: Niccolo Ajroldi, Antonio Orvieto, Jonas Geiping
- **Venue**: ICML 2025
- **URL**: https://proceedings.mlr.press/v267/ajroldi25a.html

## Key Contributions
- Benchmarks checkpoint averaging across seven architecture/dataset workloads.
- Reports broad optimization-efficiency gains and milder generalization gains with little implementation or memory cost.
- Finds that averaging and learning-rate annealing are complementary rather than substitutes.

## Relevance
EXP-011 already validates sparse full-state EMA together with cosine annealing. Future averaging changes must compute the implemented kernel, including initialization mass, rather than reasoning from an idealized normalized exponential. Uniform averaging, EMA bias correction, and hybrid parameter/buffer kernels are distinct interventions; BatchNorm state and once-per-epoch evaluation constrain clean comparisons.

## Key Techniques
- Average checkpoints along the optimization trajectory.
- Pair averaging with an annealed learning-rate schedule.
- Audit effective checkpoint weights, sample size, state age, and BatchNorm semantics.
