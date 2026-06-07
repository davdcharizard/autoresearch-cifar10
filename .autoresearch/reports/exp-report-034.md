# Report EXP-034: persistent_workers=True
- **Created**: 2026-06-04

## Results
- **Primary metric**: 94.08% (baseline: 96.39%, delta: -2.31%)
- **Key observation**: 48 epochs — same as EXP-030/032/033. persistent_workers had no effect.
- **CRITICAL FINDING**: System consistently getting 48 epochs now vs 54-58 in EXP-020 through EXP-029. This is a HW/system change that reduces training speed by ~15%. All experiments since EXP-030 are confounded by this. T_max=49 doesn't finish cosine decay in 48 epochs (LR is 0.004 at termination, not 0).
- **Next**: Run exact baseline code (no changes) to establish current-system baseline before tuning further.

## Verification
- **Verdict**: no-improvement
## Exit Action Results
