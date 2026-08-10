# When, Where and Why to Average Weights?
- **Authors**: Niccolo Ajroldi, Antonio Orvieto, Jonas Geiping
- **Venue**: ICML 2025
- **URL**: https://proceedings.mlr.press/v267/ajroldi25a.html

## Key Contributions
- Benchmarks checkpoint averaging across seven modern architectures and datasets.
- Finds significant training-efficiency gains and mild generalization gains at minimal implementation and memory cost.
- Finds that averaging complements learning-rate annealing; combining them performs best rather than replacing decay.

## Relevance
The accepted run already has an effective 20% cosine refinement tail and finishes at its best. A shadow average restricted to this weak annealed tail could smooth trajectory noise without changing data pressure or the accepted strong phase. BatchNorm buffers and counted update overhead remain local protocol risks.

## Key Techniques
- Average checkpoints or online weights along one late training trajectory.
- Retain learning-rate annealing alongside averaging.
- Select the averaging window and state handling explicitly.
