# Brainstorm EXP-008
**Created**: 2026-06-08
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

<!-- This file is focused on IDEATION only.
     Goal statement, primary metric, direction, hard constraints, and verification criteria
     live in the goal file (see pointer above). Baseline lives in experiment-indices/maximize-cifar10-best-test-accuracy.tsv.
     Do not duplicate those fields here — always point to the source of truth. -->

## Web Search & Literature Review

- **SGDR / cosine scheduling note** (`knowledge/papers/sgdr-cosine-schedule.md`)
  Smooth learning-rate decay is a plausible CIFAR schedule alternative, but EXP-000 shows the horizon must be calibrated to actual fixed-budget steps rather than the unreachable `MAX_STEPS`.

- **EXP-002 report** (`reports/exp-report-002.md`)
  The best current recipe is FP32 compile/channels-last ResNet-20, which reached 43,398 steps and 91.95% after the first LR drop while never reaching the second 48k milestone.

- **EXP-003 report** (`reports/exp-report-003.md`)
  Moving the second LR drop to 40k reached LR 0.001 but reduced peak accuracy to 91.85%, suggesting the LR 0.001 phase is not helpful for this fixed-budget ResNet-20.

- **EXP-007 report** (`reports/exp-report-007.md`)
  TF32 failed as a speed lever and reduced step budget to 37,922, reinforcing that EXP-008 should preserve the proven FP32 throughput path and focus on reachable schedule behavior.

No new external search was needed; the relevant evidence is already in the local knowledge base and prior experiment reports.

## Experimental History Review

- Current baseline remains EXP-002 at `best_test_acc=91.95%`; the tightened goal requires `>=92.05%`.
- EXP-002 is the only accepted improvement and establishes the recipe to preserve: ResNet-20, crop/flip augmentation, SGD, `[32000, 48000]` milestones, FP32, channels-last, cuDNN benchmark, and `torch.compile`.
- EXP-003 rules out the exact second-drop-at-40k approach because the LR 0.001 phase reduced accuracy.
- EXP-004/005 show extra overhead or strong regularization can erase the fixed-budget benefit.
- EXP-006/007 show missed or shortened schedule phases are a dominant failure mode; larger capacity and TF32 both reduced usable schedule exposure.
- EXP-000 bundled cosine with cutout, label smoothing, and Nesterov, so it does not isolate schedule-only behavior on the successful FP32 baseline.

## Candidate Ideas

### 1. Earlier First Drop with No Reachable Second Drop
**Summary**: Keep the EXP-002 FP32 throughput ResNet-20 recipe, but change milestones from `[32000, 48000]` to `[30000, 64000]`. This moves the first drop modestly earlier while making the second drop unreachable under the current fixed-budget step count.

**Reasoning**: EXP-002 peaked after the first LR drop and never reached the second. EXP-003 suggests LR 0.001 is harmful or unnecessary, but it did not test whether more LR 0.01 time helps. A 30k first drop adds roughly 2k-3k extra low-LR refinement steps while preserving enough high-LR training and avoiding the failed 0.001 phase.

**Sources**: `reports/exp-report-002.md`; `reports/exp-report-003.md`; `reports/exp-report-007.md`.

**Estimated Effort**: low

**Risk Assessment**: If the first drop is too early, the model may underfit slightly and fall below baseline. The failure mode should be clean no-improvement.

### 2. Budget-Calibrated Cosine Decay to LR 0.01
**Summary**: Replace `MultiStepLR` with a cosine schedule over the expected 43k-step horizon, using an `eta_min` near 0.01 so the run receives smooth decay without entering the EXP-003 LR 0.001 regime.

**Reasoning**: The SGDR note supports cosine schedules for CIFAR-style SGD, and EXP-000's cosine failure was confounded by strong regularization and an overlong 64k horizon. A calibrated horizon and higher floor would directly test smooth schedule shape on the successful FP32 baseline.

**Sources**: `knowledge/papers/sgdr-cosine-schedule.md`; `reports/exp-report-000.md`; `reports/exp-report-002.md`.

**Estimated Effort**: low

**Risk Assessment**: Cosine may decay too early across the whole training run and reduce high-LR exploration. It also changes more of the schedule shape than a single milestone edit.

### 3. ResNet-32 with Much Earlier First Drop
**Summary**: Retry ResNet-32 with a first LR drop around 14k-16k steps and no unreachable late-stage assumptions, using EXP-006's measured 23,642-step budget.

**Reasoning**: EXP-006 did not actually test low-LR ResNet-32 because the first drop at 26k was missed. A measured early drop would test whether the larger model can benefit from low-LR refinement within its slower step budget.

**Sources**: `reports/exp-report-006.md`; `knowledge/papers/wide-residual-networks.md`.

**Estimated Effort**: low

**Risk Assessment**: This repeats a recent capacity direction and risks another large undertraining miss. It is less conservative than schedule-only exploitation of the current best recipe.

## Idea Evaluation

The strongest next move is the earlier-first-drop schedule-only experiment. It preserves the only validated recipe, targets the observed mechanism behind EXP-002's success, and avoids the exact failed mechanism from EXP-003: an LR 0.001 phase. The expected effect size is modest, but the tightened threshold only requires +0.10 points, and a small extension of LR 0.01 refinement is a plausible way to clear it without adding overhead or regularization.

The cosine candidate is also reasonable, but it changes the whole LR trajectory and EXP-000 already showed that badly calibrated cosine can undertrain within the fixed budget. It is better as a follow-up if milestone retuning cannot extract more from the baseline. ResNet-32 with much earlier drops remains scientifically interesting, but EXP-006's large miss makes it a lower-confidence choice while simple ResNet-20 schedule variants remain untested.

## Chosen Idea
**Selected**: Earlier First Drop with No Reachable Second Drop

**Why this idea**:
It is the narrowest schedule intervention left on the successful FP32 throughput baseline. It tests more LR 0.01 refinement without reintroducing the LR 0.001 phase that EXP-003 showed to be harmful.

**Hypothesis**:
Changing the ResNet-20 scheduler milestones to `[30000, 64000]` will preserve high-LR learning, increase the amount of LR 0.01 refinement before the 300s budget ends, and raise `best_test_acc` to at least `92.05%`.
