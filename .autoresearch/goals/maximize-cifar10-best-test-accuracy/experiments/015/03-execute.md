# EXP-015: Same-Width Residual Identity Initialization

## Execution

Overall Status & Info:
- **Created**: 2026-08-06
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-best-test-accuracy-015
- **Commit**: (pending — committed on loop success)
- **Outcome**: completed — valid no-improvement run

## Implementation Notes

### Summary

Added a literal keyword-only `zero_init_residual` flag to `BasicBlock` and passed it only to block indices 1 and 2 in each stage when stride/channels prove a same-width identity. Exactly six existing `bn2.weight` tensors (448 scalars) initialize to zero. All three stage-entry blocks, including padded transitions, remain accepted; no production logging or other training/evaluation mechanic changed.

### Surprises & Discoveries

An out-of-distribution Gaussian first-step probe tripped the 95% class-concentration sentinel by one example: candidate was 128/128 one class and the equally untrained control 127/128. The production-distribution N1/M7 probe passed unchanged thresholds at 92.97% candidate concentration, max gamma 0.00330, improving replay loss, and complete second-backward recruitment. The first paired timing attempt then found one cold control-inference outlier despite 100 process-local warmups; all candidate, training, and remaining control measurements were stable.

### Decisions

Kept the reviewed code and every threshold unchanged. The Gaussian probe is not used as production safety evidence; the preregistered gate is evaluated on real strong views. The isolated timing instability is categorized as infrastructure and permits one exact protocol retry under the execution skill; no warmup/order/script/gate change is made.

## Experimental Adjustments

- **Removed production gamma logging before implementation**: mandatory Claude plan review found it contradicted the selected proposal; gamma recruitment is external preflight only. (ref: `02-plan-review.md`)
- **Retry exact timing protocol once**: attempt 1 passed all candidate/system ratios but failed only control inference CV due one cold outlier. (ref: `00-paired-timing-attempt1.md`)
- **Accepted exact timing retry**: unchanged attempt 2 passed all gates with 0.999219x training and 0.999527x inference ratios. (ref: `00-paired-timing.md`)

## Run Log

### Run 1

Metadata:
- **Job ID**: local PID 2327085 (watchdog PID 2327081)
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.5-gpt-5-6-sol-adversarial/run.log` (pending)
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-08-06 07:18 UTC
- **Ended**: 2026-08-06 07:25 UTC

Description:
- One fixed-seed H20 run will test whether selectively identity-initializing six same-width residual blocks improves the accepted 94.15% recipe to at least 94.25%. The production run is forbidden until structural, first-update, 64-step fit, and stable paired timing gates all pass. No fallback method or valid accuracy rerun is permitted.

Observations:
- Static/structural checks passed: only `train.py`, 1,073,962 parameters, six exact zero gammas, three gamma-one entries, aligned nonselected state/RNG, exact identity outputs, and live padded transition halves. (source: structural preflight output)
- Real first-update checks passed for hard/soft targets: max gamma <=0.00330, improving replay loss, concentration 92.97%, and every gated convolution recruited on backward two. (source: `00-first-update.py` output)
- 64-step real strong fit passed: candidate/control terminal-loss EMA ratio 0.769176 and candidate terminal concentration 26.56%. (source: `00-short-fit.py` output)
- Timing attempt 1 training passed at 0.991940x and 27,116 projected steps; inference control CV failed due one isolated cold outlier. (source: `00-paired-timing-attempt1.md`)
- Exact timing retry passed: 0.999219x training, 26,919 projected steps, 0.999527x inference, CV <=0.910%, and identical 598.671 MiB allocation. (source: `00-paired-timing.md`)
- Production startup passed on pinned CUDA index 0 with 1,073,962 parameters, 390 batches/epoch, approximately 11 ms steps, and loss declining from 2.07 to 1.59 by step 400. (source: `run.log` startup/progress lines)
- The strong switch reached 86.48%, below the 87.08 underfit marker and 3.25 points below EXP-010. The first weak checkpoint recovered to 93.17%, but the tail peaked/finalized at only 93.80%. (source: `run.log` evaluation trajectory)
- Budget/lifecycle completed normally: 26,983 steps, 300.0 counted seconds, 332.9 total, 19 unique evaluations, one 80.0% switch, eight workers stopped, and 10,716/21,528 CutMix batches. (source: `run.log` switch and summary)

Key Metrics:
- best/final test accuracy: 93.80% / 93.80% @ epoch 70 (source: `run.log` final evaluation/summary)
- final test loss: 0.2064 (source: `run.log` final summary)
- exposure: 70 epochs / 26,983 steps / 300.0 counted seconds (source: `run.log` final summary)
- total/startup time: 332.9 / 1.0 seconds (source: `run.log` final summary)
- model/memory: 1,073,962 parameters / 598.7 MiB peak allocation (source: `run.log` final summary)
- switch/first weak: 86.48% / 93.17% (source: `run.log` evaluation trajectory)

## Verification Results

### Conditions Checked

- Scope/structure/first-update/short-fit: passed.
- **Primary metric**: failed — 93.80% is 0.35 points below the 94.15 frontier and 0.45 below the 94.25 success threshold.
- **Completion and summary**: passed — exit 0 with all ten standard finite fields.
- **Budget and scope**: passed — 300.0 counted seconds, 332.9 total, only `train.py`, and 1,073,962 parameters.
- **Exposure and timing**: passed — 26,983 steps exceeded both the 26,629 floor and EXP-010's 26,898 by 85.
- **Lifecycle/evaluation**: passed — one 80.0% switch, eight workers stopped, 49.78% CutMix, hard weak targets, and 19 evaluations on 19 unique epochs.
- **Mechanism**: failed to improve generalization — switch fit crossed the 87.08 underfit marker downward; the weak tail recovered but final NLL 0.2064 remained worse than 0.1934.

### Informational Metrics

- First-update max gamma: 0.003296; 64-step loss EMA ratio: 0.769176. (source: preflight outputs)
- Timing ratio/projected steps: 0.999219 / 26,919; inference ratio: 0.999527. (source: `00-paired-timing.md`)
- Strong switch / first weak / final: 86.48% / 93.17% / 93.80%, versus EXP-010 89.73% / 93.16% / 94.15%. (source: `run.log` trajectory)
- Final/best gap: 0.00 points; final NLL: 0.2064. (source: `run.log` final summary)

## Errors & Dead Ends

### 2026-08-06 — Cold first control inference timing outlier
- Error: `accepted inference CV=6.282% >2%; first mean=2.2608ms versus remaining 1.9689-1.9855ms`
- Root cause: isolated fresh-process/cold-device timing instability; paired candidate and every training trial remained stable.
- Source: `00-paired-timing-attempt1.md`
- Do NOT retry: do not change candidate, warmup, ordering, thresholds, or accept the unstable attempt; permit only one identical full timing retry.

### 2026-08-06 — Selective zero-gamma suppresses strong fit
- Error: `switch_acc=86.48% <87.08% underfit marker; best_test_acc=93.80%`
- Root cause: delaying six of nine residual branches materially weakened representation learning under the short strong phase despite bounded first updates, better 64-step fit, and equal full-run exposure.
- Source: `run.log` evaluation trajectory and final summary
- Do NOT retry: do not rerun seed 42, zero additional blocks, or weaken preflight; any residual-initialization revisit needs a nonzero/smaller gate and a separately reviewed mechanism.

## Human Notes

> Autopilot execution; mandatory external Claude idea and plan reviews completed successfully with no fallback reviewer.
