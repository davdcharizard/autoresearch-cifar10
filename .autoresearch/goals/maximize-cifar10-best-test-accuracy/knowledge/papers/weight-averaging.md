# When, Where and Why to Average Weights?
- **Authors**: Niccolo Ajroldi, Antonio Orvieto, Jonas Geiping
- **Venue**: ICML 2025
- **URL**: https://proceedings.mlr.press/v267/ajroldi25a.html

## Key Contributions
- Benchmarks checkpoint averaging across modern optimization workloads.
- Finds efficiency gains and mild generalization improvements at minimal memory cost.
- Finds that combining averaging with learning-rate annealing works best.

## Relevance
An exponential moving average or late trajectory average can improve the evaluated model without extra optimizer steps. For this BatchNorm model, averaged weights require careful buffer handling or a final statistics refresh; an extra full data pass could threaten the 10-minute total limit.

## Key Techniques
- Average model parameters over a selected late-training window.
- Combine averaging with learning-rate annealing rather than treating it as a replacement.
- Handle non-parameter state, especially BatchNorm running statistics, explicitly.

