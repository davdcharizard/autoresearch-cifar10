# Brainstorm EXP-004
**Created**: 2026-06-10
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

- **TrivialAugment (arXiv 2103.10158, ICCV 2021)** — web search this loop: tuning-free augmentation; on CIFAR-10 WRN-class models it is SOTA-competitive with zero search cost. Crucially, the paper's CIFAR protocol applies flip + pad-and-crop + TA + **16px cutout after TA** — i.e., TA is designed to compose with occlusion erasing, exactly the EXP-003 recipe. Available as `transforms.TrivialAugmentWide` in torchvision 0.24.1 (verified installed; operates on PIL images, so it must be inserted BEFORE ToTensor). Paper schedules are 200 epochs at batch 128; our 114 one-cycle epochs are shorter, so expect a fraction of the published gain (+0.4–0.6pp over baseline aug on WRN-40-2/28-10).
- **Random Erasing (arXiv 1708.04896)** — validated in-project by EXP-003 (+0.83pp); kept in all candidates.
- **WRN (arXiv 1605.07146)** (knowledge/README.md): width gains continue past 4x given enough effective training signal.

## Experimental History Review

- **Trajectory**: 91.97 → 93.16 (EXP-000 recipe) → 95.23 (EXP-001 4x width) → 94.41 (EXP-002 8x, no-improvement) → **96.06 (EXP-003 RandomErasing)**. Baseline: 96.06 @ 3a62d44.
- **Patterns** (goal-learnings): time-keyed one-cycle composable (3x validated); occlusion regularization free accuracy once capacity exists (+0.83pp, zero throughput cost); width scales steeply but trades against epochs under the fixed time budget.
- **Failed Approaches**: capacity without throughput (8x → 40 epochs, undertrained). Note: that failure was WITHOUT augmentation pressure; exp-report-003 argues the width-epoch optimum may have moved right, but RE also slows per-epoch fitting, so wider+fewer-epochs under stronger augmentation cuts both ways.
- **Protocol findings**: eval overhead manageable at 4x width (399.5s total, comfortable under 600s cap).
- **Untried gaps**: policy-based augmentation (TA/AutoAugment), width 5–6x on the regularized recipe, torch.compile throughput, ResNet-9 topology, larger batch.

## Candidate Ideas

### 1. TrivialAugmentWide on top of the regularized recipe
**Summary**: Insert `transforms.TrivialAugmentWide()` into `train_tf` after RandomHorizontalFlip and before ToTensor (PIL-stage op). Keep RandomErasing. Everything else unchanged.

**Reasoning**: EXP-003 just validated the regularization axis with the gain landing at the TOP of the published range — evidence the 4x net is still capacity-rich relative to augmented data complexity, i.e., the axis is not exhausted. TA is the strongest tuning-free policy augmentation, and the paper's own CIFAR protocol stacks occlusion cutout after TA, so composition with EXP-003 is literature-standard, not a gamble. Cost is a single PIL op on 8 CPU workers — throughput at the GPU-bound 4x width should be unchanged.

**Sources**: arXiv 2103.10158 (TA, ICCV 2021 — protocol composes TA + cutout); exp-report-003 § Next Steps (high confidence); torchvision 0.24.1 `TrivialAugmentWide` (verified installed).

**Estimated Effort**: trivial (one transform line)

**Risk Assessment**: Main risk is over-regularization at 114 one-cycle epochs (paper uses 200): TA slows fitting, and if epochs become binding the run could land within noise or slightly below baseline. Failure mode is a clean no-improvement; zero stability/wall-clock risk. Watch for epoch-1 accuracy drop and a final<best signature.

### 2. Width 6x on the regularized recipe (stage widths 96/192/384, ~9.6M params)
**Summary**: WIDTH_MULT 4 → 6, keep RandomErasing and recipe. Targets the width-epoch optimum, which EXP-003's augmentation pressure may have shifted right of 4x.

**Reasoning**: exp-report-003 argues more signal per epoch under augmentation re-motivates width. But the counter-mechanism is real: RE also slows convergence, and 6x gets only ~65–75 epochs. EXP-002 lost 0.82pp at 40 epochs without RE; 6x+RE at ~70 epochs is genuinely uncertain in both directions.

**Sources**: experiment-indices rows 001/002/003; arXiv 1605.07146; exp-report-003 § Next Steps (medium confidence).

**Estimated Effort**: trivial

**Risk Assessment**: Higher variance than Idea 1 — could be +0.5pp or −0.5pp. Clean failure mode (no-improvement), no stability risk. Better attempted after the cheap augmentation gain is banked, so its baseline contribution compounds.

### 3. torch.compile for throughput (more epochs at 4x)
**Summary**: `model = torch.compile(model)`; expected 1.2–1.8x img/s on small CNNs, converting to ~140–200 epochs.

**Reasoning**: More epochs would specifically help the now-more-augmented recipe (stronger augmentation raises epochs-to-converge). But EXP-003 finished with final=best — the anneal converged at 114 epochs — so extra epochs convert weakly today; the lever matters mainly as an enabler for the width push (Idea 2).

**Sources**: PyTorch torch.compile docs; goal-learnings § Patterns (throughput lever status); exp-report-003 § Next Steps (low-medium confidence).

**Estimated Effort**: low, but fiddly failure modes (graph breaks, recompiles, autocast interaction, compile time added to wall clock)

**Risk Assessment**: Crash/instability tail risk and indirect accuracy mechanism; weakest immediate-expected-value of the three.

## Idea Evaluation

**Evidence strength**: Idea 1 strongest — published SOTA-class gains in this exact model family, with the paper's own protocol demonstrating composition with occlusion erasing, plus EXP-003's in-project evidence that the regularization axis is paying at the top of published ranges. Idea 2 rests on a directional argument with a real counter-mechanism. Idea 3's evidence concerns speed, not accuracy.

**Mechanism clarity**: Idea 1 — clear and just-validated: raise effective data complexity to convert remaining capacity into generalization. Idea 2 — two opposing mechanisms (more signal per epoch vs fewer epochs), net sign unknown. Idea 3 — indirect chain with a weak final link (epochs not currently binding).

**Expected impact**: Idea 1: +0.2–0.5pp (short-schedule fraction of published +0.4–0.6). Idea 2: −0.5 to +0.5pp. Idea 3: ~0 now.

**Risk profile**: Idea 1 and 2 fail gracefully; Idea 1 has the tighter variance. Idea 3 carries crash tail.

**Feasibility**: Ideas 1–2 are one-line; Idea 3 fiddly.

**Sequencing**: banking Idea 1 first compounds — if TA helps, Idea 2's later width push starts from a stronger regularized recipe, which is precisely the condition under which width is re-motivated.

## Chosen Idea
**Selected**: Idea 1 — TrivialAugmentWide on top of the regularized recipe

**Why this idea**:
Highest evidence-to-risk ratio and exploits the axis EXP-003 just proved is hot: published gains in this exact model class with a protocol that explicitly composes with the existing RandomErasing, one line of code, zero throughput/stability risk, and banking it first compounds with the width experiment queued next.

**Hypothesis**:
Inserting TrivialAugmentWide before ToTensor (keeping RandomErasing) in the 4x-wide net's train transform will raise best_test_acc from 96.06% to ≥96.25%, because EXP-003's gain at the top of the published range shows capacity remains under-regularized, and TA raises effective data complexity further at zero GPU cost; throughput and epoch count will be essentially unchanged (~114 epochs), with the main risk being over-regularization at the short schedule.
