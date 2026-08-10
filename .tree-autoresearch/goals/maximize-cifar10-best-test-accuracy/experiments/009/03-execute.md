# EXP-009: Late-Stage Identity-Centered SE with Isolated Initialization

## Execution

Overall Status & Info:
- **Created**: 2026-08-05
- **Autonomy**: autopilot
- **Experiment Branch**: tree-autoresearch/maximize-cifar10-best-test-accuracy-exp-009
- **Base Node**: 002
- **Commit**: `762b609`
- **Outcome**: failed - fixed SE package exceeded the parent-relative latency gate before a metric run

## Implementation Notes

### Summary

Added four raw-parameter identity-centered SE modules to the 128/256-channel residual outputs. A dedicated CPU generator initializes only first-layer weights; final layers are zero so the parent function is exact. Descriptor standardization and excitation run in an autocast-disabled FP32 region, effective gates multiply the same residual tensor in activation dtype, and a separate 5x-LR/no-decay optimizer group owns exactly the 25,408 new elements. Training-only sampled device buffers audit gate activity without evaluation pollution or per-step host synchronization.

### Surprises & Discoveries

- Exact parent preservation passed: all 83 shared state tensors, global RNG state, and initial CPU/BF16 logits were bitwise identical.
- The gates became strongly active within 200 steps without broad saturation, so the BF16 dead zone was not the limiting failure.
- The FP32 descriptor and four small affine paths were launch/precision expensive: candidate median production latency was 20.7% above parent despite less than 1% extra parameters.
- One dtype harness initially included the parent classifier's intended BF16 `F.linear` in the FP32 SE assertion; restricting the assertion to the eight SE affine calls resolved this harness-only error.

### Decisions

- The fixed late-four placement, FP32 math, hidden width, and LR group were not changed after the failed latency gate. No full training/evaluation was launched.

## Experimental Adjustments

- None.

## Run Log

### Run 1

Metadata:
- **Job ID**: N/A - full run not launched
- **Log file(s)**: N/A - preflight output distilled below; no `run.log` created
- **WandB**: N/A
- **Status**: failed at preflight
- **Started**: 2026-08-05 17:50 UTC
- **Ended**: 2026-08-05 17:54 UTC

Description:
- Implement and preflight the fixed late-stage identity-centered SE package on physical GPU 0. The run could proceed only if gates became materially active without saturation and matched production latency projected at least 26,000 EXP-002-equivalent steps.

Observations:
- Static compile/Ruff/diff checks passed; only `train.py` differs.
- Inventory passed: 4 modules, 16 gate tensors, 25,408 gate elements, 2,774,298 total parameters.
- GPU-0 integration passed: eight SE affine calls remained FP32 inside outer BF16 autocast, initial BF16 output matched parent bitwise, staged gradients reached `weight2` then `weight1`, optimizer groups were disjoint/complete, and inference-mode evaluation did not mutate statistics.
- Live-gate smoke passed after 200 steps. Per-module mean deviations were 0.202099/0.137746/0.121413/0.034505; fractions beyond two ULPs were 0.915283/0.880506/0.885749/0.647850; saturation fractions were 0.001388/0.000081/0.000076/0; all nonfinite counts were zero.
- Latency gate failed. Parent median/p90/mean were 10.087693/10.285463/10.192185 ms; candidate 12.177348/12.979396/12.286809 ms over 300 measured iterations after 50 warmups.

Key Metrics:
- Candidate/parent median latency ratio: 1.20715 (required <=1.075).
- Projected steps: approximately 23,154 from `27,950 / 1.20715` (required >=26,000).
- Primary metric: unavailable; no evaluation launched.

## Verification Results

### Conditions Checked

- Scope/static/initialization/inventory/GPU integration/live gate: pass.
- Parent-relative latency and exposure: fail, ratio 1.20715 and projected steps about 23,154.
- Full-run protocol and accuracy: skipped after fixed-package gate failure.

### Informational Metrics

- No training metrics available.

## Errors & Dead Ends

### 2026-08-05 - FP32 late-stage SE exceeds fixed latency budget
- Error: `candidate median ratio 1.20715 > 1.075; projected steps 23,154 < 26,000`
- Root cause: four FP32 pooled-standardization and bottleneck-affine gate paths add charged kernel/precision overhead that parameter count alone does not predict.
- Source: Run 1 latency values above.
- Do NOT retry: do not shrink placement, change precision, hidden width, LR scale, or activity statistics on this node.

## Human Notes

> User requires physical GPU 0 and Claude adversarial review only. Claude completed both EXP-009 idea and plan reviews; no fallback reviewer was used.
