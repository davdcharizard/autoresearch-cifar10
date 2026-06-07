# Report EXP-037
- **Created**: 2026-06-04
## Results
- 95.15%, 49 epochs. Epoch-level sync no help — loss.item() still forces per-step GPU sync.
- Also used T_max=49 instead of 43, causing misalignment.
## Verdict: no-improvement
