# EXP-008: k=4 + Stochastic Depth + EMA

## Execution
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-008
- **Outcome**: failed

## Run Log
### Run 1 (failed — Python control flow broke torch.compile)
- 91.31%, 40.8s startup (recompilation)
### Run 2 (failed — tensor masking, no compute skip)
- 93.22%, drop_rate=0.2 too aggressive for 9-block model

## Errors & Dead Ends
### 2026-05-28 — Stochastic depth incompatible with shallow model + torch.compile
- Root cause 1: Python branching breaks torch.compile graph tracing
- Root cause 2: Even with tensor masking, 0.2 drop rate removes too much capacity from 9 blocks
- Do NOT retry: stochastic depth not suitable for models with < 30 blocks

## Human Notes
