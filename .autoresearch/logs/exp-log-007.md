# EXP-007: k=4 + EMA + Weight Decay 5e-4

## Execution
- **Created**: 2026-05-28
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-007
- **Commit**: 6e24ddb
- **PR**: (pending)
- **Outcome**: completed

## Implementation Notes
### Summary
Added EMA (deepcopy of model before compile, decay=0.999), increased weight decay to 5e-4.
### Surprises & Discoveries
Run 1 produced 12.04% — EMA model BN layers had uninitialized running stats because ema_model was always in eval mode. Fixed by copying buffers.
### Decisions
Copy BN buffers directly (not EMA-smoothed) since running stats should reflect recent training data.

## Run Log
### Run 1 (failed)
- 12.04% — BN buffer issue
### Run 2 (success)
- best_test_acc: 95.73%, final: 95.73%, 55 epochs, 4.3M params, 551MB VRAM

## Verification Results
All 4 conditions passed. 95.73% >= 95.35%.

## Errors & Dead Ends
### 2026-05-28 — EMA model BN running stats uninitialized
- Error: 12.04% accuracy (random chance)
- Root cause: ema_model.eval() meant BN used initial running stats (zeros), not training stats
- Fix: copy model.buffers() to ema_model.buffers() alongside parameter EMA update
- Do NOT retry: always copy BN buffers when using EMA with BatchNorm models

## Human Notes
