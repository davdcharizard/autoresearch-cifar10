# Brainstorm EXP-006
**Created**: 2026-07-24

## Web Search & Literature Review

- **Wide Residual Networks** (`knowledge/papers/wide-residual-networks.md`): shallower, wider residual networks can improve CIFAR representation quality and compute efficiency, but local H20 timing is still decisive under this fixed-time objective.
- **When, Where and Why to Average Weights?** (`knowledge/papers/weight-averaging.md`): carefully windowed parameter averaging can mildly improve generalization at low overhead when early, under-trained iterates are excluded.
- **Time Matters in Regularizing Deep Networks** (`knowledge/papers/time-matters-regularization.md`): regularization is most valuable early, supporting experiments that preserve the accepted clean late tail.
- **When Does Label Smoothing Help?** (`knowledge/papers/label-smoothing.md`): mild soft targets can reduce overconfidence, but stacking them with validated mixup risks excessive target regularization.
- **RandAugment** (`knowledge/papers/randaugment.md`): low-dimensional augmentation policies can improve CIFAR accuracy, although CPU transform overhead and the three local augmentation regressions lower its priority.

No network search was performed because this autoresearch session is explicitly offline; all external grounding came from the local, previously distilled knowledge base.

## Experimental History Review

- EXP-001 replaced the original ResNet-20 recipe with WRN-16-2, selective decay, persistent workers, and a counted-time warmup/cosine schedule, raising the baseline from 91.54% to 93.38% with about 146 passes.
- EXP-002 added alpha-0.2 batchwise mixup through 65% of counted time and a 35% hard-label tail, reaching the current 94.07% best with 141.9 passes and final accuracy equal to best.
- EXP-003 replaced mixup with shared-rectangle CutMix and regressed to 93.72% at normal exposure; spatial mixed labels are not the next priority.
- EXP-004 shortened mixup to 50% and regressed to 93.91%; the 50-65% regularization interval is useful.
- EXP-005 strengthened mixup to alpha 0.4 and regressed to 93.57% with test loss rising from 0.2432 to 0.2737; stronger input interpolation over-regularizes this WRN.
- The accepted path is compute-efficient, stably converged, and now limited by small remaining generalization/endpoint gains rather than an obvious failure to train. Untried gaps include late iterate averaging, schedule-floor isolation, internal WRN regularization, and additional capacity.
- A local matched synthetic H20 preflight measured WRN-16-3 at only 56.8% of WRN-16-2 image throughput. Calibrated to EXP-002, width 3 projects about 80.6 passes, and batch 512-768 recovered less than 6%; this fails the preregistered 60%-throughput and 85-pass gates in `proposals/idea-01.md`.

## Collected Ideas

- **Late hard-tail EMA** — initialize an averaged shadow at the 65% mixup transition, update all floating model state with decay 0.999, and use it for the existing evaluations. It targets residual late-iterate variance without changing early learning and is supported by the local weight-averaging distillation.
- **Zero-floor cosine** — change only `MIN_LR` from 0.002 to 0.0 so the final hard-label refinement settles instead of retaining residual Nesterov motion. It isolates an untested scalar from the successful EXP-001 bundle with no throughput or RNG cost.
- **Early WRN block dropout** — apply p=0.10 dropout between the two residual-branch convolutions only until the 65% transition. It targets feature co-adaptation through a WRN-native mechanism while preserving the validated clean tail.
- **WRN-16-3 capacity** — widen stage channels from 32/64/128 to 48/96/192 while retaining the accepted optimizer and mixup recipe. It targets representation capacity and fits easily in VRAM, but the measured 43% exposure loss makes it infeasible for EXP-006.
- **In-model channel standardization** — divide centered inputs by canonical CIFAR-10 channel standard deviations inside the model so both frozen evaluator and training inputs receive the same conditioning. This targets first-layer optimization, but first-layer scaling plus BatchNorm may offer little headroom and has weaker local evidence than the finalists.
- **Late weight-decay removal** — set convolutional weight decay to zero at the 65% transition, preserving early norm regularization while allowing the hard-label tail to fit margins more freely. It is cheap and follows the early-regularization prior, but it entangles optimizer-state behavior and has no direct local evidence that late decay is limiting.
- **Mild GPU Cutout** — erase a small random square only during early training and retain mixup. It adds spatial robustness without CPU transforms, but stacking another input regularizer after the CutMix and alpha-0.4 regressions has a high over-regularization risk.
- **Hard-tail label smoothing** — use epsilon 0.05 cross entropy after mixup ends. It targets overconfidence, but it removes the genuinely hard-label tail that EXP-002's trajectory supports and is therefore low priority.
- **Mixed-precision capacity moonshot** — use channels-last autocast to recover enough throughput for a wider or deeper model. Its upside is high, but numerical behavior, optimizer stability, and the local width-3 compute gap require several coupled changes, making a one-run attribution poor.

## Combinations

- **Zero-floor cosine + late EMA**: the lower endpoint could make the averaging window more stationary while EMA smooths the remaining noise. The cross may beat either alone, but it should only follow isolated evidence because a win would otherwise be unattributable.
- **WRN-16-3 + mixed precision**: Tensor Core throughput could offset some capacity cost while the wider model raises its ceiling. This is stronger than naive width alone, but the 43% measured gap and multiple optimization changes make it a later engineering experiment.
- **Early block dropout + late EMA**: dropout could diversify early features, then EMA could stabilize the clean tail after dropout is removed. The temporal separation is coherent, but current evidence cannot distinguish whether either component alone is sufficient.
- **Channel standardization + zero-floor cosine**: better early conditioning and a quieter endpoint act at opposite ends of training. The combination is plausible but unnecessary until each low-cost scalar/conditioning hypothesis is tested independently.

## Candidate Ideas

### EMA Over the Hard-Label Tail
**Summary**: At the existing 65% mixup transition, initialize a shadow copy of the accepted model and update all floating parameters and BatchNorm buffers after each optimizer step with EMA decay 0.999. Use the live model before initialization and the EMA model afterward at the unchanged evaluation points; never evaluate both at one checkpoint.

**What it targets**: Three augmentation changes have failed while final accuracy remains close to or equal to best, suggesting the accepted late basin is good but may retain enough SGD/checkpoint variance for averaging to improve generalization.

**Reasoning**: A 0.999 decay has an effective 1,000-step horizon and negligible initialization mass by the end of the roughly 9,900-step hard tail. It leaves early representation learning untouched, consumes no RNG, adds little memory, and is directly supported by the local parameter-averaging literature note.

**Sources**: `proposals/idea-02.md`; `knowledge/papers/weight-averaging.md`; EXP-001 and EXP-002 analyses.

**Estimated Effort**: medium.

**Risk Assessment**: Gains may be smaller than 0.10 points because the accepted endpoint is already stable. Averaged BatchNorm buffers approximate rather than exactly recalibrate statistics, and per-step shadow updates consume counted time and could erase a small statistical benefit if not implemented with grouped tensor operations.

### Early p=0.10 WRN Block Dropout
**Summary**: Add p=0.10 dropout after the second BN/ReLU and before `conv2` in every residual branch, active only during the first 65% of counted training. Disable it exactly when mixup ends so the accepted hard-label, low-LR tail is otherwise unchanged.

**What it targets**: The model may still have feature co-adaptation that input mixup does not address. Weak residual-branch masking attacks this internal generalization gap while keeping identity shortcuts deterministic.

**Reasoning**: The placement is native to WRN residual branches, the probability is conservative for the small WRN-16-2, and the time-limited policy follows the local early-regularization evidence. It is orthogonal to changing mixup strength or spatial label construction and adds no parameters or forward pass.

**Sources**: `proposals/idea-03.md`; `knowledge/papers/wide-residual-networks.md`; `knowledge/papers/time-matters-regularization.md`; EXP-003 through EXP-005 analyses.

**Estimated Effort**: medium.

**Risk Assessment**: Mixup plus dropout may over-regularize the small model just as alpha 0.4 did, and dropout is often more useful in wider models with redundant features. CUDA masks change the subsequent random stream, and mask-generation overhead can modestly reduce exposure.

### Zero-Floor Cosine for the Hard-Label Tail
**Summary**: Change only `MIN_LR = 0.002` to `0.0`, retaining every accepted model, optimizer, mixup, loader, seed, and evaluation setting. This converts the schedule to a canonical cosine-to-zero endpoint. The schedule difference is negligible early, reduces LR by 26.4% at 90% progress and 59.3% at 95%, and cuts hard-tail LR area by 7.9% while leaving exposure unchanged.

**What it targets**: The accepted run continued improving through its hard-label tail but still applied LR 0.002 at the endpoint. This candidate tests whether residual late Nesterov motion prevents the already learned representation from settling into a slightly better classifier.

**Reasoning**: EXP-001 never isolated its LR floor, while EXP-002 finished at its best with test loss 0.2432. A one-line zero floor is an interpretable endpoint with no operation-graph, RNG, memory, or throughput change. It provides the cleanest discrimination between useful continued motion and unnecessary endpoint motion.

**Sources**: `proposals/idea-04.md`; EXP-001 and EXP-002 analyses; `03-experiment-learnings.md`.

**Estimated Effort**: low.

**Risk Assessment**: The accepted tail may still need its nonzero floor, so zero can freeze useful margin refinement. Expected upside is probably close to the 0.10-point acceptance threshold, and `MIN_LR` also changes the warmup start from 0.002 to zero, although that early effect is small and disclosed.

## Review

The blind critic selected early p=0.10 WRN block dropout because it is the only finalist with a plausible effect comfortably larger than the 0.10-point acceptance margin. The review also found that the local evidence does not demonstrate feature co-adaptation or directly validate dropout in this small WRN; additive over-regularization is the leading failure mode. Those concerns are accepted. EXP-006 will keep the exact `bn2/ReLU -> dropout -> conv2` placement, remove dropout once at 65% with mixup, require a warm matched-path exposure preflight, and interpret a normal-exposure loss/accuracy regression as rejection of this specific stacked regularizer.

EMA was downgraded because the accepted endpoint is already stable and still improving, its BatchNorm buffers are only an approximation for averaged weights, and EMA-only late evaluations can lag or hide the live trajectory. Zero-floor cosine was the cleanest probe, but it targets no observed endpoint instability and likely has too little ceiling; its warmup coupling would also need separating for a strictly late-only test. Full concerns and scores are in `01-idea-review.md`.

## Idea Evaluation

Adopt the critic's verdict in `01-idea-review.md`. Block dropout scored lower on direct evidence than zero-floor cosine but higher on potential impact, 7/10 versus 4/10, and is the only finalist whose credible upside is not concentrated at the task's minimum acceptance margin. The capacity proposal remains excluded by measured throughput rather than by reviewer preference.

## Chosen Idea
**Selected**: Early p=0.10 WRN Block Dropout

**Why this idea**:
It tests an orthogonal internal generalization mechanism while preserving every accepted architectural, optimization, input-mixup, and late-tail component. The conservative probability, residual-branch-only placement, and removal at the validated 65% boundary contain the main over-regularization risk. Unlike EMA or a zero LR floor, it has a credible multi-tenth upside if feature co-adaptation is a remaining limiter.

**Hypothesis**:
Applying p=0.10 dropout between `bn2`/ReLU and `conv2` in each residual branch during the first 65% of counted training, then disabling it alongside mixup, will improve feature diversity without materially reducing exposure. A valid run should retain at least 95% of EXP-002's 141.9 passes and achieve `best_test_acc >= 94.17%`; a lower score with normal exposure and higher final loss will indicate that mixup plus block dropout over-regularizes WRN-16-2.
