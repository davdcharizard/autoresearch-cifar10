# Proposal: Move the Coupled Strong/LR Boundary to 75%

## Decision and falsifiable hypothesis

Change only `LR_HOLD_FRACTION` from `0.8` to `0.75`. In the current program this one constant governs four coupled behaviors: the end of LR `0.1`, the break from the strong loader, the N1/M7+CutMix-to-weak/hard loader rebuild, and the start of dense-tail evaluation. The candidate therefore preserves the locally important synchronization of view strength, target type, and LR while reallocating 15 counted seconds from high-LR strong exploration to low-LR weak refinement.

**Hypothesis:** the accepted CutMix model has learned sufficient broad/regional invariance by 75%, and lengthening the weak hard-label cosine phase from 20% to 25% will convert more of that representation into clean top-1 accuracy. With unchanged per-step work, the candidate should retain approximately 26.9k optimizer steps, improve final NLL relative to EXP010's 0.1934, and raise seed-42 `best_test_acc` from 94.15% to at least the formal **94.25%** threshold. Point prediction: **94.28%**. A completed lower score falsifies the accuracy claim; it is not grounds to try 77.5% or another boundary within EXP030.

## Mechanism and historical distinction

The proposed mechanism is a time reallocation, not a new optimizer or augmentation. The first 75% remains the exact accepted width-2 ResNet-20 objective: LR 0.1, N1/M7, alpha-1 CutMix on probability 0.5 of strong batches, ordinary momentum, and all-parameter decay `1e-4`. At the one boundary, all three sources of difficulty change together: LR steps to approximately 0.01, strong views become crop/flip, and all targets become hard labels. The remaining 25% follows the same cosine endpoint `1e-4`, but over 75 rather than 60 counted seconds.

This is materially different from the two local early-removal failures:

- EXP005 moved only the data boundary to 75% while LR remained 0.1 until 80%. Its harmful interval was weak hard-label training at high LR, and it lost 0.18 points. EXP030 creates no intended weak-at-LR-0.1 interval.
- EXP027 removed only CutMix at 70%, retaining N1/M7 and LR 0.1. Hard strong-view training immediately collapsed fit and lost 0.46 points. EXP030 never trains hard N1/M7 in a dedicated high-LR bridge.
- EXP002 showed that an 80% plateau beat a bundled 15% schedule on the pre-RandAugment, narrow model, but did not compare 75% and 80% under the current CutMix/width-2 recipe.
- EXP010 explicitly left an earlier full boundary open with care: its strong checkpoint was 89.73%, its first weak checkpoint jumped to 93.16%, final NLL improved to 0.1934, and accuracy was still rising to its final 94.15%. Those observations support testing more refinement time, but they do not prove the strong representation is mature by 75%.

The tension is the research question. The promoted local pattern says long high-LR exploration matters, and EXP005/027 protect *synchronization* at the boundary. EXP030 preserves synchronization but challenges whether the synchronized boundary must be exactly 80%. A loss would mean the last 15 seconds of strong CutMix exploration are more valuable than the extra low-LR clean refinement.

## Exact implementation and schedule semantics

The entire tracked diff must be exactly:

```diff
-LR_HOLD_FRACTION = 0.8
+LR_HOLD_FRACTION = 0.75
```

Do not add a second phase constant. Keeping one constant is essential because the existing predicates remain coupled:

```python
if progress <= LR_HOLD_FRACTION:
    lr = LR
else:
    cosine_progress = (progress - LR_HOLD_FRACTION) / (
        1.0 - LR_HOLD_FRACTION
    )
    lr = MIN_LR + 0.5 * (ANNEAL_START_LR - MIN_LR) * (
        1.0 + math.cos(math.pi * cosine_progress)
    )
```

The same constant already controls the post-step strong-loop break, `dense_tail_due`, and loader switch. Thus the last strong batch begins below 75% at LR 0.1 and may cross the boundary by one timed step; after its evaluation and loader rebuild, the first weak batch must begin above 75% with LR approximately 0.01. The cosine then reaches about 0.00905 at 80%, 0.00352 at 90%, and `1e-4` at 100%. This is a longer traversal of the same `[0.01, 1e-4]` curve, not a change to its endpoints.

Leave `EVAL_CHECKPOINTS=(0.2,0.4,0.6,0.7)` unchanged. Dense evaluation begins automatically at the new 75% boundary and remains at most once per epoch. Also leave the model, initialization, optimizer, batch size, transforms, CutMix alpha/probability/RNG isolation, target handling, workers, timer, evaluator, seed, precision, and summary unchanged.

## Preflight and integrity checks

This scalar changes no model kernel or per-step operation, so a paired GPU timing campaign would not test the hypothesis and is unnecessary. Before the one scored run, require:

- the moving baseline is still 94.15 at `7c1e7d8`, current `train.py` matches that source before editing, and afterward only the one registered line differs among tracked files;
- AST/static checks find one definition of `LR_HOLD_FRACTION`, value exactly 0.75, and all four intended consumers still reference it; no new literal phase threshold or evaluation is added;
- a deterministic boundary simulation of the unchanged formulas verifies LR 0.1 below/through the hold, a jump to about 0.01 immediately above it, monotonic cosine decay thereafter, and exact endpoint `1e-4` at progress 1.0;
- the existing switch still shuts down the strong loader once, rebuilds one weak loader, removes CutMix with RandAugment, and asserts one-dimensional weak targets;
- syntax, formatting/lint, scope, seed, parameter-count expectation, summary schema, and timeout command all pass;
- exactly one idle NVIDIA H20 with approximately 97,871 MiB is available, and no stale `run.log` or completed renamed log remains.

The first post-switch progress/LR record in production must show low LR, not 0.1. If floating-point boundary coincidence creates a weak step at LR 0.1, the intended coupled intervention did not execute and the run is invalid; do not change the comparator as an in-experiment rescue.

## Expected execution and diagnostics

Run exactly once at seed 42:

```bash
timeout --kill-after=5s 595s uv run train.py > run.log 2>&1
```

Expected exposure is about 26,900 updates, with roughly 20.2k strong batches and 6.7k weak batches rather than EXP010's 21,446 strong batches and roughly 5.45k weak batches. The weaker loader/loss path should not reduce counted exposure; **26,629 steps** (99% of EXP010) is a diagnostic expectation, not a post-hoc reason to discard an otherwise protocol-valid fixed-budget result. The longer dense tail will add approximately three to four epoch evaluations and some wall time, but should remain far below 600 seconds because evaluation is excluded from the 300-second training counter.

Record and interpret:

- the final pre-switch evaluation, exact switch progress/epoch/step, worker-stop count, CutMix/strong counts, and realized CutMix fraction (expected 45-55% while eligible);
- the first post-switch LR record and confirmation that every weak target is hard;
- 70% accuracy, 75% switch accuracy, first weak accuracy, best epoch, final accuracy/NLL, and best-final gap;
- strong and weak train-loss EMA slopes, especially whether the weak phase recovers faster than the loss of five percentage points of strong exposure;
- total strong/weak batches, epochs, optimizer steps, evaluation count and uniqueness, counted/total/startup time, VRAM, and parameter count.

Do not use EXP010's 89.73% 80%-switch accuracy as a pass/fail threshold for the earlier 75% checkpoint; the horizons differ. Use it only as context. The decisive favorable signature is healthy pre-switch fit followed by an earlier weak-phase jump, final NLL at or below 0.1934, and `best_test_acc >=94.25%`. A lower switch score that later wins still supports the net time-reallocation hypothesis. Conversely, better NLL without the top-1 gate is calibration-only no-improvement.

## Risks and failure signatures

- **Strong-underfit risk — high:** EXP010's gain depended on 80% of N1/M7+CutMix exploration. Removing about 1.3k strong updates may leave invariances or regional features immature, and low LR may be unable to create them later.
- **History risk — medium-high:** EXP005 and EXP027 do not duplicate this synchronized candidate, but together they are strong evidence that the neighborhood around the protected boundary is sensitive.
- **Over-refinement risk — medium:** the longer hard weak tail may over-specialize to crop/flip views or sharpen confidence without moving top-1.
- **Evaluation-wall risk — low:** beginning dense-tail evaluation five percentage points earlier adds passes but not counted training; total wall time must still stay below 600 seconds.
- **Runtime/exposure risk — low:** the training graph is identical and weak data production has ample headroom; any large step loss indicates contention or protocol failure rather than the scalar mechanism.
- **Causal-resolution risk — medium:** one fixed seed estimates the net schedule change, not a precise effect size. A 0.10-0.20-point gain is formally positive but weak evidence relative to trajectory noise.

Predeclared harmful signatures are: materially lower 75% fit followed by incomplete recovery; first weak accuracy failing to jump; final NLL above 0.1934; an early best followed by tail regression; or lower best accuracy despite equal exposure. None is an early-stop or rescue trigger—the valid run must finish.

## Verification and verdict

Require exit zero; exactly one finite ten-field summary; approximately 300.0-301.0 counted seconds; total below 600 seconds; 1,073,962 parameters; one switch near 75% with exactly eight workers stopped; no strong/soft target after the switch; 45-55% CutMix among eligible strong batches; no duplicate evaluation epoch and no more than one evaluation per epoch; one H20; fixed seed 42; and no retry.

The formal verdict follows only the goal and protocol:

- **Improvement:** all integrity conditions pass and `best_test_acc >=94.25%` (94.15 baseline plus the required 0.10 points).
- **No improvement:** the run is protocol-valid but `best_test_acc <94.25%`, regardless of NLL, exposure, or a favorable intermediate checkpoint.
- **Invalid/crash:** wrong scope/hardware/seed, malformed transition or target provenance, first weak step at high LR, evaluator/timer violation, nonfinite or incomplete summary, nonzero exit, or runtime at least 600 seconds.

Do not rerun a valid completion and do not rescue with 77.5%, a different tail start LR, a separate augmentation boundary, changed CutMix, extra checkpoints, another seed, or threshold relaxation. Any such change is a new experiment with a new hypothesis and ID.
