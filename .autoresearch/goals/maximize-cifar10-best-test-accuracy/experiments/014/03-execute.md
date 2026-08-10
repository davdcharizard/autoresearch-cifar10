# EXP-014: Zero-Initialized Concatenated Average-Max Readout

## Execution

Overall Status & Info:
- **Created**: 2026-08-06
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-best-test-accuracy-014
- **Commit**: (pending — committed on loop success)
- **Outcome**: completed — valid no-improvement run

## Implementation Notes

### Summary

Preserved the accepted average classifier and added a bias-free 128-to-10 `max_fc` after global initialization inside a CPU RNG fork, zeroing all 1,280 new weights. The forward now sums average and adaptive-maximum logits. Every existing evaluation and the final summary report the max/average classifier weight-norm ratio; no evaluator, optimizer, schedule, data, augmentation, batch, timer, worker, or seed mechanics changed.

### Surprises & Discoveries

Default CUDA backward was not bitwise repeatable even between two identical control constructions, while strict deterministic mode rejects `adaptive_max_pool2d` backward. Mandatory Claude implementation review rejected putting a detach/hook workaround into production. Its required disposable detached preflight variant proved accepted first gradients under deterministic kernels, while the clean production model separately proved max-path engagement after the first update. The timing harness initially needed one path-only fix because nested script execution did not include the project root on `sys.path`; no GPU measurement had begun before that correction.

### Decisions

Kept production `train.py` exactly at the originally reviewed clean design, with no verification hook, stateful boolean, conditional detach, or deterministic-mode toggle. Adopted all requirements from `02-plan-review-implementation-addendum.md`: fresh hard/soft deterministic pairs for first-step identity and normal-mode finite/nonzero second-step max-path verification. No gate or experiment setting was relaxed.

## Experimental Adjustments

- **Externalized deterministic first-step proof**: Claude rejected production verification scaffolding; a disposable detached subclass supplies the analytic-equivalent first-step proof without changing the experiment artifact. (ref: `02-plan-review-implementation-addendum.md`)
- **Passed tightened paired feasibility**: candidate/control training ratio was 1.001381 with 26,860 projected steps; inference ratio was 1.026495 and projected total 330.779 seconds. (ref: `00-paired-timing.md`)

## Run Log

### Run 1

Metadata:
- **Job ID**: local PID 2304724 (watchdog PID 2304720)
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.5-gpt-5-6-sol-adversarial/run.log`
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-08-06 06:17 UTC
- **Ended**: 2026-08-06 06:23 UTC

Description:
- One fixed-seed local H20 run will test whether the zero-initialized class-specific maximum readout improves the accepted 94.15% recipe to at least 94.25%. The run preserves all EXP-010 mechanics and changes only endpoint aggregation plus provenance logging. It must complete 300 counted training seconds under the 600-second wall limit with 18-19 unique evaluations and at least 26,242 updates.

Observations:
- Structural preflight passed: 1,075,242 parameters; bitwise accepted state/post-construction RNG; deterministic hard/soft initial logits, loss, and accepted gradients; finite nonzero first max gradient/update; finite nonzero second-step selected-location max gradient. (source: preflight command output)
- Paired timing passed all preregistered training, inference, memory, stability, exposure, and total-runtime gates. (source: `00-paired-timing.md`)
- Production startup passed on CUDA with 1,075,242 parameters, 390 batches per epoch, finite early loss, and approximately 11 ms steps. (source: `run.log` startup/progress lines)
- The max branch immediately dominated: the first evaluation at epoch 14 was 10.00% with `max_readout_ratio=3.963452`; all 19 evaluations remained at 10.00%, and the final ratio was 3.963398. (source: `run.log` evaluation and summary lines)
- Lifecycle and budget completed normally: one 80.0% switch stopped eight workers after 10,659/21,417 CutMix batches, and the process exited 0 with no traceback, OOM, or nonfinite signal. (source: `run.log` switch, error scan, and process exit)

Key Metrics:
- best/final test accuracy: 10.00% / 10.00% @ epoch 69 (source: `run.log` final summary)
- final test loss: 2.3026 (source: `run.log` final summary)
- exposure: 69 epochs / 26,803 steps / 300.0 counted seconds (source: `run.log` final summary)
- total/startup time: 329.8 / 1.0 seconds (source: `run.log` final summary)
- model/memory: 1,075,242 parameters / 598.7 MiB peak allocation (source: `run.log` final summary)
- max-readout ratio: 3.963452 at first eval; 3.963398 final (source: `run.log` evaluation and summary lines)

## Verification Results

### Conditions Checked

- **Primary metric**: failed — 10.00% is 84.15 points below the 94.15 frontier and 84.25 points below the 94.25 success threshold.
- **Completion and summary**: passed — exit 0 with all ten standard finite fields plus `max_readout_ratio`.
- **Budget and exposure**: passed — 300.0 counted seconds, 329.8 total seconds, and 26,803 steps >=26,242.
- **Scope and structure**: passed — only tracked `train.py` changed; parameter count 1,075,242; one evaluator call site remained unchanged.
- **Lifecycle and evaluation**: passed — one 80.0% switch, eight workers stopped, 49.77% CutMix, hard weak targets, and 19 evaluations on 19 unique epochs.
- **Max-path provenance**: passed mechanically but falsified the hypothesis — first ratio 3.963452 proved strong engagement rather than a no-op, while immediate chance accuracy showed uncontrolled readout domination.

### Informational Metrics

- Paired training ratio: 1.001381; projected steps: 26,860; candidate peak: 598.686 MiB. (source: `00-paired-timing.md`)
- Paired inference ratio: 1.026495; projected total: 330.779 seconds. (source: `00-paired-timing.md`)
- Strong switch / first weak / final accuracy: 10.00% / 10.00% / 10.00%; all far below EXP-010's 89.73% / 93.16% / 94.15%. (source: `run.log` evaluation trajectory)
- Final/best gap: 0.00 points; final NLL 2.3026 versus EXP-010's 0.1934. (source: `run.log` final summary)

## Errors & Dead Ends

### 2026-08-06 — Timing harness project-root import
- Error: `ModuleNotFoundError: No module named 'train'`
- Root cause: direct execution of the nested ignored timing script placed only its experiment directory on `sys.path`.
- Source: first `00-paired-timing.py` invocation before any GPU worker launched
- Do NOT retry: nested diagnostic scripts that import tracked project modules must explicitly add the current project root.

### 2026-08-06 — Zero-initialized max classifier dominates immediately
- Error: `best_test_acc=10.00%; first max_readout_ratio=3.963452`
- Root cause: the unnormalized global-max feature gradient drove the zero-initialized classifier to nearly four times the average classifier norm before the first evaluation, collapsing predictions to chance; the weak tail and LR decay could not recover.
- Source: `run.log` evaluation trajectory and final summary
- Do NOT retry: do not use an independently learned raw global-max classifier at the accepted LR without a pre-registered scale/gate/normalization mechanism that controls its first updates.

## Human Notes

> Autopilot execution; mandatory Claude adversarial reviews used with no fallback reviewer.
