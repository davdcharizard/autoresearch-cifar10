# How to Scale Your EMA
- **Authors**: Dan Busbridge, Jason Ramapuram, Pierre Ablin, Tatiana Likhomanenko, Eeshan Gunesh Dhekane, Xavier Suau Cuadros, Russell Webb
- **Venue**: NeurIPS 2023
- **URL**: https://papers.nips.cc/paper_files/paper/2023/hash/e7681dd6fe16052433ab68cd1555bdc9-Abstract-Conference.html

## Key Contributions
- Shows that model EMA dynamics change when batch size or update frequency changes.
- Derives a scaling rule intended to preserve EMA behavior across training scales.
- Demonstrates the rule across architectures, optimizers, and data modalities.

## Relevance
EXP-007 has a fixed batch size but a wall-clock-dependent number of updates, and any sparse EMA changes its effective horizon. The paper supports treating decay and update cadence as one preregistered time constant rather than choosing a familiar decay independently of the realized update schedule.

## Key Techniques
- Express EMA momentum through its effective averaging horizon.
- Adjust decay when update frequency changes so the same amount of training data receives comparable weight.
- Treat the EMA copy as a functional model whose dynamics must be specified, not incidental bookkeeping.
