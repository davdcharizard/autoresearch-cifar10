# Report EXP-045: BFloat16 + T_max=43
## Results
- best=96.07%, final=95.66%, 55 epochs. BF16 is FASTER (55 vs 49 with FP16+GradScaler).
- 0.41% best/final gap from LR restart (T_max=43 finishes at epoch 48, 7 extra epochs with rising LR).
- Highest result on this slower system. BF16 is the breakthrough — it eliminates GradScaler overhead AND provides faster BF16 compute.
## Key Learning
BFloat16 without GradScaler gives 12% more epochs (55 vs 49). Must align T_max or clamp LR.
## Verdict: no-improvement (96.07% < 96.49%)
## Next: BF16 + T_max=50 (align to 55 epochs) for full utilization
