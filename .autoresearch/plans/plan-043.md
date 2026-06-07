# Plan EXP-043: Multi-step LR decay
## Code Changes
- **train.py**: Replace the LR schedule. Remove cosine, add multi-step:
  - Replace `cosine_scheduler` and `SequentialLR` with:
    ```python
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=WARMUP_EPOCHS)
    step_scheduler = MultiStepLR(optimizer, milestones=[25, 35], gamma=0.1)
    scheduler = SequentialLR(optimizer, [warmup_scheduler, step_scheduler], milestones=[WARMUP_EPOCHS])
    ```
  - milestones=[25, 35] in MultiStepLR → fires at scheduler steps 25 and 35, which are epochs 30 and 40 (after 5 warmup epochs)
  - Remove COSINE_T_MAX usage (no longer needed)
