# EXP-021: Stage-2 training-only companion classifier

## Execution

Overall Status & Info:
- **Created**: 2026-08-06
- **Autonomy**: autopilot
- **Experiment Branch**: tree-autoresearch/maximize-cifar10-best-test-accuracy-exp-021
- **Base Node**: 004
- **Commit**: `763b98e`
- **Outcome**: failed — complete valid run, primary metric below the parent and formal gate

## Implementation Notes

### Summary

Implemented a fixed full-run companion objective in `train.py`. A dedicated-seed `Linear(128,10)` head taps block 3 only on explicit training forwards; default/evaluator calls keep the original main-logit contract and skip all companion work. Every primary and scheduled SAM replay computes the same hard/CutMix-aware `main CE + 0.15 * companion CE`, with the head owned by inherited SGD and SAM. Minimal diagnostics record exact training/evaluator isolation and cadence-512 stage-2 feature norms.

### Surprises & Discoveries

The existing EXP004 file already has an import-safe main guard and helper-level SAM seams, so no structural refactor was needed. The companion can be registered after the inherited model-wide initializer while preserving global CPU RNG: save/restore around `nn.Linear` construction, then overwrite it using a local seed-42021 generator.

### Decisions

Following plan review, aggregate per-step loss diagnostics and gradient-share reductions were omitted. Only core call reconciliation is run-failing; feature-norm cadence and terminal head displacement remain report-only. All training companion operations and sparse norm reductions sit within the inherited charged step; terminal scalar reads and reporting remain outside charged training like the parent summary.

## Experimental Adjustments

- **No candidate adjustment after deterministic smoke**: smoke passed inherited state/default-logit/RNG parity, exact target formulas, auxiliary-only gradient reach, two-pass SAM radius/restoration, sparse audit routing, and zero evaluator calls.
- **Preflight environment check stopped before timing**: the first attempt emitted no gate vector because physical GPU 0's compute-app query reported PID 80142 with 3,384 MiB allocated. Subsequent process/pmon evidence showed this was stale accounting rather than active compute; no candidate measurement occurred. (ref: `/tmp/exp021_preflight.log` traceback)
- **Used the one allowed pre-vector harness repair**: PID 80142 had no `/proc` entry, was named `[Not Found]`, consumed 0% GPU, and was absent from `nvidia-smi pmon`; the compute-app query was stale accounting rather than an active process. The harness now records both lists but gates contamination on active `pmon` PIDs. Candidate code, fixtures, timing formulas, and thresholds are unchanged; no further preflight repair/rerun is permitted. (ref: post-failure `ps`, compute-app, utilization, and pmon checks)
- **Decisive repaired preflight passed**: median ratio `1.022327`, p90 `1.025486`, MAD/median `0.003090`, parent drift `0.010438`, projected 25,001 steps / 129 epochs, zero live-allocation growth, exact 512 primary / 49 replay / 561 head calls in all seven rounds, zero evaluator calls, and clean active-process monitoring. (ref: `/tmp/exp021_preflight_rerun.log` complete JSON)

## Run Log

### Run 1

Metadata:
- **Job ID**: local unified-exec session 32815
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/tree-v0-gpt-5-6-sol/run.log`
- **WandB**: N/A
- **Status**: completed (research no-improvement)
- **Started**: 2026-08-06 16:54:51 UTC
- **Ended**: 2026-08-06 17:02:17 UTC

Description:
- The sole metric run will test whether fixed-weight stage-2 deep supervision raises EXP004 beyond 95.50% while preserving the 300-second budget and at least approximately 24,000 steps. The training-only head is never evaluated; any accuracy movement comes through its effect on the shared backbone/main classifier. Stronger stable-tail criteria determine later EMA composability but do not alter the formal tree verdict.

Observations:
- The sole metric process exited 0 after 300.0 charged and 433.8 total seconds, completing 25,336 steps and 130 epochs/evaluations. No traceback, OOM, runtime error, nonfinite, or integrity failure appeared. (source: `run.log` L267-L288; evaluation-line count = 130)
- Companion dose reconciled exactly: 25,336 primary losses plus 2,402 SAM replays produced 27,738 head forwards, while all 130 evaluations left the head counter unchanged. CutMix applied 10,180/20,533 and SAM 2,402/4,803 from progress 0.7500. (source: `run.log` L267-L269)
- The 50 cadence samples reconciled. Pooled stage-2 mean L2 was 5.593, 4.347, 4.133, and 5.988 across charged-time quartiles, showing substantial U-shaped conditioning drift in the BN-free tap; this is descriptive, not causal proof of the accuracy miss. Head displacement was 11.435 with zero nonfinite state. (source: `run.log` L270-L276)
- The final-16 live accuracies averaged 94.944375%, ranged 94.66-95.11%, and ended at 95.10%. The best was only 0.01 above final but 0.165625 above the 16-value mean, so the late endpoint recovered from a materially lower early-tail region without reaching the parent. (source: `run.log` L277-L281)

Key Metrics:
- `best_test_acc`: 95.11% (EXP004 parent 95.40%, -0.29 point; formal threshold 95.50%, failed) (source: `run.log` L279; `tree.sh show ... 004`)
- `final_test_acc`: 95.10%; `final_test_loss`: 0.1883, versus parent 95.40% / 0.1654 (source: `run.log` L280-L281; EXP004 `04-analysis.md`)
- Exposure: 25,336 steps / 130 epochs; charged 300.0 s; total 433.8 s; peak VRAM 1,190.6 MiB; 2,750,180 parameters (source: `run.log` L282-L288)
- Composition conjunction: failed (`best 95.11 <95.60`, tail mean 94.944375 <95.50); dose passed 24,000 and best-final gap passed 0.15. (source: `run.log` L277-L287)

## Verification Results

### Conditions Checked

- **Execution integrity and constraints — PASS**: physical GPU 0, one visible H20, exit 0, 300.0 charged seconds, 433.8 total seconds, 130/130 evaluation cadence, unchanged evaluator/default path, exact companion/CutMix/SAM counts, zero nonfinite, and only tracked `train.py` changed. (source: pre-launch commands; `run.log` L267-L288; git scope checks)
- **Primary metric — FAIL**: 95.11% is 0.29 points below EXP004's 95.40% and 0.39 below the required 95.50%. This is a valid no-improvement, not a crash or invalid result. (source: `run.log` L279; parent tree card)
- **Remaining verification**: stopped after the primary necessary condition failed; raw evidence is preserved for independent analysis audit.

### Informational Metrics

- Not collected as a verification-success appendix because the necessary metric condition failed; all durable run and mechanism values are transcribed under Run 1.

## Errors & Dead Ends

### 2026-08-06 — External GPU process blocked preflight before timing
- Error: `RuntimeError: external GPU process before timing: [80142]`
- Root cause: physical GPU 0 reported an external-container compute context before any paired measurement; utilization was 0% but 3,384 MiB remained allocated.
- Source: `/tmp/exp021_preflight.log` L1-L4; `nvidia-smi --query-compute-apps` after exit
- Do NOT retry: do not run a paired vector while PID 80142 is present, kill an unowned process, change gates, or treat a contaminated vector as decisive.

## Independent Result Audit

- Claude returned `AUDIT_VERDICT: PASS`, independently recomputing the 95.11 maximum and no-improvement verdict, log freshness/scope, 130/130 evaluation cadence, budget, exact companion/CutMix/SAM/audit arithmetic, tail/composition criteria, isolated initialization, charged work, and absence of reward hacking (`04-result-review.md`).

## Human Notes

> Autopilot session; no execution-phase human intervention.
