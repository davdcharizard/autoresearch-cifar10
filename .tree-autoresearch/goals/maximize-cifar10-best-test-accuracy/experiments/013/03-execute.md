# EXP-013: Fixed-scale-40 cosine-normalized classifier

## Execution

Overall Status & Info:
- **Created**: 2026-08-06
- **Autonomy**: autopilot
- **Experiment Branch**: tree-autoresearch/maximize-cifar10-best-test-accuracy-exp-013
- **Base Node**: 011
- **Commit**: `d15e138`
- **Outcome**: failed (valid no-improvement; metric run completed with full integrity)

## Implementation Notes

### Summary

Preserved the parent's `nn.Linear(256,10,bias=True)` construction and initialization, froze its initialized-zero bias immediately after model initialization, and replaced only the final affine call with FP32 scale-40 normalized feature/weight logits. Added setup, every-evaluation online, EMA-source, and terminal online/EMA geometry diagnostics plus stored/trainable count and optimizer/SAM/EMA ownership checks. Evaluation lines now include charged progress for final-tail analysis; all parent data, loss, optimizer settings, CutMix, SAM, EMA, and evaluation routing remain intact.

### Surprises & Discoveries

Claude's first implementation review identified that a bare per-epoch geometry exception would bypass the complete summary and that online geometry alone did not describe EMA-sourced predictions. Geometry failures are now deferred, request loop exit after the current evaluation, flow through both terminal audits and the complete summary, then raise nonzero. EMA epochs print distinct online and EMA geometry.

### Decisions

The initial geometry dictionary retains a tiny detached normalized-row tensor for terminal reference formatting/diagnostics; it is outside optimizer, SAM, and EMA ownership. Finite norm change is observation only. Claude's final review returned `PASS`; the review is preserved in `03-implementation-review.md`.

## Experimental Adjustments

- **Deferred per-epoch audit failures and added EMA-source geometry**: Required by Claude to preserve summary evidence and accurately audit the weights producing EMA-sourced accuracy. No classifier recipe changed. (ref: `03-implementation-review.md`)

## Run Log

### Run 1

Metadata:
- **Job ID**: local PID 1793531 (timeout PID 1746498; execution session 60165)
- **Log file(s)**: `run.log` (metric), `/tmp/exp013_preflight.log` (first complete preflight), `/tmp/exp013_preflight_attempt1.log` (pre-measurement harness error)
- **WandB**: N/A
- **Status**: completed; research no-improvement
- **Started**: 2026-08-06T08:01:54Z (agent-recorded launch window; the metric process reports authoritative `total_seconds=457.2`)
- **Ended**: 2026-08-06T08:10:25Z (agent-recorded verification timestamp)

Description:
- Deterministic correctness and integration gates followed by the one-shot paired preflight and, only if those gates pass, one fixed-seed metric launch on physical GPU 0. No test accuracy is queried before the metric launch, and no alternate scale or metric retry is permitted.

Observations:

- CPU deterministic smoke passed: stored/trainable counts `2748890/2748880`, initialization/RNG parity, exact scale-40 formula and scaling invariance, parent-identical CutMix, and 43-tensor SAM ownership.
- GPU-0 integration smoke passed with finite FP32 logits/loss under BF16 autocast, valid gradients/update/SAM replay, two EMA updates, exact restoration, five classifier audit lines, and peak allocation `1092.289 MiB`.
- The first preflight attempt stopped before a complete measurement because of the vector-valued finite predicate documented below. The one allowed harness correction was applied; no gate value was observed or discarded.
- The first complete preflight printed `GPU_PREFLIGHT_OK`. Claude Opus independently reviewed the implementation and raw evidence and returned `VERDICT: PASS` with no blockers, authorizing exactly one metric launch.
- The one-shot metric process exited `0` after a complete summary. It achieved `95.11%`, below the parent `95.61%` and formal `95.71%` threshold, so EXP-013 is a valid no-improvement. No metric retry or scale change was made.

Key Metrics:

- Preflight parent drift: `0.00993782` (gate `<=0.03`).
- Preflight candidate/parent median ratio: `1.00957979` (gate `<=1.03`); MAD/median dispersion `0.00256197` (gate `<=0.005`).
- Projected training exposure: `25553.21` steps (formal launch gate `>=25000`; projected mechanism floor `>=25300`).
- Projected total runtime: `452.19s` (gate `<600s`); peak allocation `1237.258 MiB`.
- Early trace parent/candidate CE: step 1 `2.69307/6.39663`; step 25 `2.01222/1.94741`; step 50 `2.16727/2.28832`; step 100 `1.65738/1.61824`; step 200 `1.49950/1.40714`. Informational only.
- EMA preflight: 30 updates per arm, 15 ordinary/15 SAM samples, one EMA evaluation/swap, one exact restore, and zero coverage/nonfinite/RNG/restore failures.
- Metric summary: best/final `95.11%/95.07%`, loss `0.1765`, charged/total/startup `300.0s/457.2s/1.6s`, peak allocation `1221.4 MiB`, `132` epochs, `25598` steps, stored/trainable parameters `2748890/2748880`.
- Metric dose: CutMix `10275/20704` (`0.4963`); SAM `2447/4894` (`0.5000`) beginning at step `20706`, progress `0.7500`; EMA `158` updates split `79/79` ordinary/SAM with 26 EMA evaluations/swaps/restores and zero failures.
- Final 16 EMA evaluations (epochs 117-132): `95.07, 95.01, 95.05, 95.05, 95.11, 95.09, 95.06, 95.08, 95.10, 95.07, 95.07, 95.08, 95.11, 95.08, 95.08, 95.07`; mean `95.073750`, range `0.10`, progress span `0.142190`.
- Terminal classifier audit: initial/online/EMA norm means `1.43738306/3.26836252/3.30340576`; online/EMA cosine means `0.61419338/0.61962318`; online-EMA raw/normalized distances `0.14033714/0.02613891`; bias stayed exactly zero and ownership was optimizer yes, SAM no, EMA indices `42/43`.

## Verification Results

### Conditions Checked

- **GPU and launch isolation**: PASS. Physical GPU 0 was `NVIDIA H20, 97871 MiB`; CUDA exposed one H20 through `CUDA_VISIBLE_DEVICES=0`; exactly one bounded metric command was launched.
- **Run integrity and budget**: PASS. Exit `0`; charged training `300.0s`; total `457.2s <600s`; complete summary present; no traceback/CUDA/OOM/audit/RuntimeError/NaN/Inf signature.
- **Evaluation cadence and source accounting**: PASS. `132` evaluation lines equal `132` epochs; `106` live plus `26` EMA evaluations equals `132`.
- **Scope and state ownership**: PASS. Only tracked `train.py` differs from `d68f73a`; stored/trainable counts are `2748890/2748880`; classifier bias/optimizer/SAM/EMA and all restore/coverage/RNG checks pass.
- **Formal improvement**: FAIL. `95.11% <95.71%` required against parent `95.61%`; delta versus parent is `-0.50` percentage points.
- **Mechanism support**: FAIL at full preregistered dose. `25598 >=25300` steps and `158 >=155` EMA updates pass, but exposure was about `0.8%` below the parent's `25798` steps, matching the `1.0096` preflight latency ratio; final-16 EMA mean `95.073750 <95.64`. Classifier and CutMix/SAM/EMA integrity remained intact.

### Informational Metrics

- Best/final accuracy `95.11%/95.07%`; final loss `0.1765`; training/total/startup `300.0s/457.2s/1.6s`; peak VRAM `1221.4 MiB`; epochs/steps `132/25598`.
- Final-16 EMA mean/min/max/range `95.073750/95.01/95.11/0.10`; parent reference tail mean `95.493125`.
- Terminal online/EMA classifier raw norm means `3.26836252/3.30340576`, pairwise cosine means `0.61419338/0.61962318`, normalized-direction distance `0.02613891`.

## Errors & Dead Ends

### 2026-08-06 — Preflight harness used a non-scalar Boolean assertion
- Error: `RuntimeError: Boolean value of Tensor with more than one value is ambiguous`
- Root cause: the accuracy-blind early trace called `assert torch.isfinite(torch.tensor([lp, lc]))` without reducing the two-element Boolean tensor. The run stopped at the first trace checkpoint, before any complete timing measurement or gate result.
- Source: `/tmp/exp013_preflight_attempt1.log` (pre-measurement harness error log)
- Do NOT retry: do not treat vector-valued PyTorch predicates as Python scalars; reduce the finite check with `.all().item()`. One straightforward harness-error retry is permitted by the execution protocol.

## Human Notes

> The user requires physical GPU 0 and Claude as the sole adversarial reviewer. Never use a fallback reviewer; pause for the user if Claude is unavailable.
