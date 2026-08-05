# Knowledge Base Index

This directory contains persistent external knowledge that informs brainstorming and planning across all research loops. Unlike experiment logs (which are loop-specific), entries here represent standing background context pulled in throughout the research process: relevant paper distillations, reference implementation notes, and domain knowledge.

## How to use
- `research-brainstorm` loads this README as a lightweight index, then loads specific entries on demand
- Entries are added by researchers (manually), by `/lit-search` (automated paper distillations), or proposed by the agent after consulting external sources
- Add entries for any paper, reference implementation, or domain insight that is likely to inform future experiments

## Venues
- `venues.md` in this directory defines which academic proceedings `/lit-search` searches for this goal — seeded with the default top CS/ML conferences (NeurIPS, ICML, ICLR)
- Edit it to match this goal's research domain; venues are per-goal, so different goals can search different domains

## Papers

| File | Topic | Key Takeaway |
|------|-------|-------------|
| `papers/wide-residual-networks.md` | CIFAR residual architecture | Shallower, wider residual networks can improve accuracy and compute efficiency. |
| `papers/mixup.md` | Mixed-sample regularization | Convex image/label interpolation improves CIFAR generalization at low GPU overhead. |
| `papers/randaugment.md` | Automated augmentation | A small shared-magnitude search space yields strong CIFAR augmentation policies. |
| `papers/time-matters-regularization.md` | Regularization scheduling | Early regularization can retain its benefit after being removed for late convergence. |
| `papers/weight-averaging.md` | Parameter averaging | Carefully windowed averaging can mildly improve generalization at low overhead. |
| `papers/label-smoothing.md` | Target regularization | Mild soft targets reduce overconfidence and can improve classification generalization. |

## References

| File | Topic | Key Takeaway |
|------|-------|-------------|
| _(empty — add reference notes here)_ | | |

## Domain Notes

| File | Topic | Key Takeaway |
|------|-------|-------------|
| _(empty — add domain-specific notes here)_ | | |
