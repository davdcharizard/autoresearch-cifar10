# How to Scale Your EMA
- **Authors**: Dan Busbridge et al.
- **Venue**: NeurIPS 2023
- **URL**: https://papers.nips.cc/paper_files/paper/2023/hash/e7681dd6fe16052433ab68cd1555bdc9-Abstract-Conference.html

## Key Contributions
- Model EMA dynamics depend jointly on momentum, batch size, and update frequency.
- A scaling rule can preserve the effective EMA horizon when training scale changes.

## Relevance
This goal's wall-clock budget changes realized update counts and phase dose when throughput changes. Future EMA designs should preregister an effective sample/time horizon and derive per-update decay from it rather than importing a familiar scalar.

## Key Techniques
- Express EMA through an effective time constant or horizon.
- Adjust decay when cadence changes so equivalent training exposure receives equivalent weight.
