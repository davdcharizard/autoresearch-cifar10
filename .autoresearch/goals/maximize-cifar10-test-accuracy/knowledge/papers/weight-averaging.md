# When, Where and Why to Average Weights?
- **Authors**: Niccolo Ajroldi, Antonio Orvieto, Jonas Geiping
- **Venue**: ICML 2025
- **URL**: https://proceedings.mlr.press/v267/ajroldi25a.html

## Key Contributions
- Evaluates checkpoint averaging across multiple modern training workloads.
- Finds that averaging can accelerate training and mildly improve generalization at minimal implementation and memory cost.

## Relevance
EXP-001's last evaluations were tightly clustered around 93.3%, making iterate averaging a plausible low-overhead way to reduce SGD variance without changing the successful WRN or data pipeline.

## Key Techniques
- Average parameters from a selected portion of the training trajectory.
- Choose the averaging window carefully to avoid bias from early, under-trained iterates.
