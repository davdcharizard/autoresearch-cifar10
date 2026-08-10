# EXP-022: Fixed 288-channel final-stage taper

## Execution

Overall Status & Info:
- **Created**: 2026-08-06
- **Autonomy**: autopilot
- **Experiment Branch**: tree-autoresearch/maximize-cifar10-best-test-accuracy-exp-022
- **Base Node**: 011
- **Commit**: `beea2ef`
- **Outcome**: failed

## Implementation Notes

### Summary

Changed only the last two residual block channel specifications from 256 to 288, the tail BatchNorm and classifier input dimensions, and two truthful architecture metadata strings. The six-block topology, all forward/training mechanisms, seed, timing, evaluator, EMA, SAM, and summary logic remain byte-identical to EXP-011.

### Surprises & Discoveries

The inherited model has no hidden width constant beyond the two block specs and tail layers, so the taper required no new code path, module, or state handler. The generic SAM snapshot and full-state EMA construction naturally enumerate the widened tensors.

### Decisions

Kept all parameter/MAC and mechanism diagnostics in transient verification harnesses instead of production. This preserves an exact allowlist of six semantic source-line replacements and prevents validation code from changing charged work or the fixed summary.

## Experimental Adjustments

- **Corrected the transient parent SHA assertion before any numeric gate**: The first CPU harness used the parent Git blob prefix as though it were a file SHA-256 prefix. Replaced it with the independently printed full SHA-256; production code and verification semantics are unchanged. (ref: `/tmp/exp022_cpu_verify.log` first traceback)
- **Added the repository root to the transient verifier import path**: Running a `/tmp` script makes `/tmp` its import root, so the parent snapshot could not resolve frozen `prepare.py`. Added the current repository root before import; no source or measurement semantics changed. (ref: `/tmp/exp022_cpu_verify.log` second traceback)

## Run Log

### Run 1

Metadata:
- **Job ID**: N/A — deterministic CPU/GPU verification
- **Log file(s)**: `/tmp/exp022_cpu_verify.log`, `/tmp/exp022_gpu_verify.log`
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-08-06 17:23 UTC
- **Ended**: 2026-08-06 17:29 UTC

Description:
- Verified the fixed architecture, deterministic construction, parameter/MAC inventory, gradients, augmentation and stochastic-depth controls, production-faithful SAM, full-state EMA, optimizer identities, memory, and exact restoration before any paired timing or accuracy access.

Observations:
- CPU verification passed with parent SHA-256 `ee0605ba...`, 3,260,442 parameters, 425,315,136 MACs/image, 16 convolutions, 22 expected shape-different state keys, 83/83 candidate self-equal state tensors, and six/zero active/terminal drop draws. Only 19/61 common-shaped tensors matched the parent and post-construction RNG differed, confirming the preregistered package confound. (source: `/tmp/exp022_cpu_verify.log`)
- Candidate-only H20 verification passed at 656.104 MiB peak. SAM perturbation was 0.05000001 with relative norm 0.0006063687; 30 EMA samples split 15/15 ordinary/SAM, one exact swap/restore passed, and restore/coverage/nonfinite/RNG failures were zero. (source: `/tmp/exp022_gpu_verify.log`)
- Steady-state live allocation changed by -159,744 bytes, satisfying the <=1 MiB growth bound; EMA inventory expanded generically to 44 parameter tensors / 3,260,442 elements and 26 float buffers / 3,872 elements. (source: `/tmp/exp022_gpu_verify.log`)

Key Metrics:
- CPU architecture: params `3260442`, MACs/image `425315136`, shape diffs `22`, candidate deterministic state `83/83`. (source: `/tmp/exp022_cpu_verify.log`)
- GPU integration: peak `656.104 MiB`, allocation growth `-159744 bytes`, SAM epsilon `0.05000001`, EMA updates/parity/restores `30/15:15/1`. (source: `/tmp/exp022_gpu_verify.log`)

### Run 2

Metadata:
- **Job ID**: N/A — local preflight
- **Log file(s)**: `/tmp/exp022_preflight.log`
- **WandB**: N/A
- **Status**: failed at decisive feasibility gate
- **Started**: 2026-08-06 17:30:35 UTC
- **Ended**: 2026-08-06 17:31:15 UTC

Description:
- The sole frozen accuracy-blind paired preflight will compare exact EXP-011 and width 288 on physical GPU 0. It includes a shared real-CIFAR finite conditioning trace, five alternating timing rounds over the exact ordinary/CutMix/SAM mix, separate evaluation timing, and preregistered early/late dose projections. The first timing datum is decisive; no timing rerun or width fallback is permitted.

Observations:
- Frozen preflight harness SHA-256: `7ba77de9d0c4924bd2ca9efa3aee7afaa9312e2f5ae481bd31e00cbc003e0626`; syntax check passed before launch. (source: `sha256sum /tmp/exp022_preflight.py`)
- Physical GPU 0 was the sole visible NVIDIA H20 at 97,871 MiB. The known stale PID 80142 / 3,384 MiB context remained in the raw compute-app query, while `pmon` showed no active process and utilization was 0%, so contamination checks passed. (source: pre-launch `nvidia-smi` output; EXP-021 infra note)
- The accuracy-blind real-CIFAR trace was finite in both arms: parent/candidate losses 2.533124/2.503272, logit L2 32.2162/37.8397, and gradient L2 1.54581/1.52757. These were non-selective diagnostics. (source: `/tmp/exp022_preflight.log` `PREFLIGHT_JSON`)
- All parent drift and ratio-dispersion gates passed, but charged cost did not: ordinary/CutMix/SAM median ratios were 1.163564/1.158708/1.160245 and the weighted median was 1.160794, above 1.10. Every charged path also exceeded the 1.13 maximum-ratio ceiling; fifth-round CutMix/SAM ratios rose to 1.272699/1.248961, but the decisive median rejection does not depend on those outliers. (source: `/tmp/exp022_preflight.log` `PREFLIGHT_JSON`)
- Early/late ratios 1.161156/1.161904 projected 17,962.28 early and 4,252.50 late steps, totaling 22,214.78 steps, 113 complete epochs, and 137.705 EMA samples. The step, epoch, and EMA gates all failed; projected total 446.75 seconds passed. No evaluator call, test-loader iteration, accuracy result, metric process, or `run.log` occurred. (source: `/tmp/exp022_preflight.log` `PREFLIGHT_JSON`)
- Width 288's weighted ratio 1.160794 is effectively the same as EXP-014 width 320's 1.160975 despite 8.33% versus 17.56% MAC growth. This falsifies the linear MAC interpolation prior for this tensor-shape change and localizes the constraint to H20 shape/kernel efficiency rather than memory (joint peak 728.967 MiB). (source: `/tmp/exp022_preflight.log`; EXP-014 `04-analysis.md`)
- The raw-result audit independently recomputed the path gates and projections, confirmed no accuracy exposure or integrity issue, and returned PASS for `crash/NaN`. The configured Claude command returned empty, so the fallback audit records its full arithmetic in the durable artifact. (source: `03-result-review.md`)

Key Metrics:
- path ratio medians ordinary/CutMix/SAM/eval: `1.163564 / 1.158708 / 1.160245 / 1.167767`; weighted median `1.160794`; weighted MAD/median `0.001326`. (source: `/tmp/exp022_preflight.log`)
- parent drift ordinary/CutMix/SAM: `0.009309 / 0.012005 / 0.009301`; path ratio dispersion `0.002632 / 0.001163 / 0.000515`. (source: `/tmp/exp022_preflight.log`)
- projections: early `17962.28`, late `4252.50`, total steps `22214.78`, epochs `113`, EMA samples `137.705`, total seconds `446.75`; joint peak `728.967 MiB`. (source: `/tmp/exp022_preflight.log`)

## Verification Results

### Conditions Checked

- **Mandatory accuracy-blind feasibility gate**: failed. Weighted charged ratio `1.160794 >1.10`, charged-path maxima exceeded 1.13, projected steps `22214.78 <23200`, epochs `113 <118`, and EMA samples `137.705 <140`; parent stability, dispersion, and projected total time passed. (source: `/tmp/exp022_preflight.log`)
- **Metric-run completion and primary accuracy**: skipped after mandatory preflight failure; `best_test_acc=NaN` because accuracy was never queried and no `run.log` was created.
- **Static scope/integrity**: passed before the preflight. The zero-context diff contains only the two block specs, final BN/classifier widths, and two metadata replacements; syntax and `git diff --check` passed. (source: implementation diff and Run 1)

### Informational Metrics

- Candidate correctness: 3,260,442 parameters, 425,315,136 MACs/image, 656.104 MiB candidate-only peak, exact SAM epsilon 0.05000001, 30 balanced EMA samples, zero state/RNG failures. (source: Run 1 logs)
- Preflight systems result: weighted ratio 1.160794, projected 22,214.78 steps / 113 epochs / 137.705 EMA samples, joint peak 728.967 MiB. (source: `/tmp/exp022_preflight.log`)

## Errors & Dead Ends

### 2026-08-06 — Parent snapshot used the wrong hash namespace
- Error: `AssertionError` comparing SHA-256 `ee0605...` against Git blob prefix `de2cafc7`.
- Root cause: The transient harness confused `git diff`'s blob-id prefix with a raw file SHA-256.
- Source: `/tmp/exp022_cpu_verify.log` first traceback before architecture or timing output.
- Do NOT retry: record the raw `sha256sum` of materialized snapshots and compare the full digest, not a Git blob identifier.

### 2026-08-06 — Transient parent import could not resolve prepare.py
- Error: `ModuleNotFoundError: No module named 'prepare'`.
- Root cause: Python placed the `/tmp` harness directory, not the repository root, at the head of `sys.path`.
- Source: `/tmp/exp022_cpu_verify.log` second traceback before model construction or timing output.
- Do NOT retry: transient out-of-tree import harnesses must insert the verified repository root before loading a source snapshot.

### 2026-08-06 — Width 288 failed the H20 feasibility envelope
- Error: `weighted_ratio=1.160794 >1.10; projected_steps=22214.78 <23200; projected_epochs=113 <118; projected_ema=137.705 <140`.
- Root cause: Widening the last two blocks to 288 incurred about 16.1% across ordinary, CutMix, and SAM paths, essentially matching width 320 despite half its incremental MAC fraction; dense 288-channel kernels did not realize the interpolated efficiency prior.
- Source: `/tmp/exp022_preflight.log` `PREFLIGHT_JSON`.
- Do NOT retry: do not rerun width 288, relax the gate, substitute 272/304, or expose accuracy after the preregistered rejection; future capacity work needs a qualitatively different tensor/kernel geometry.

## Human Notes

> Autopilot execution; no researcher intervention.
