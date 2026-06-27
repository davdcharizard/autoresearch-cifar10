# Brainstorm EXP-018
**Created**: 2026-06-10
**Goal**: goals/maximize-cifar10-test-accuracy.md

<!-- This file is focused on IDEATION only.
     Goal statement, primary metric, direction, hard constraints, and verification criteria
     live in the goal file (see pointer above). Baseline lives in experiment-indices/{slug}.tsv.
     Do not duplicate those fields here — always point to the source of truth. -->

## Web Search & Literature Review

- **Bag of Tricks for Image Classification with CNNs** (arXiv 1812.01187, CVPR 2019 — https://arxiv.org/abs/1812.01187)
  Zero-initializing γ in the LAST BN of each residual block makes every block an identity map at init, so forward/backward signal initially propagates through the shortcuts — "eases optimization at the start of training". Stacked with linear-scaled LR + warmup (our exact recipe components) it contributes to a ~+1% ResNet-50 gain; it is an init-time-only change with zero throughput/param/VRAM cost.
- **Accurate, Large Minibatch SGD** (arXiv 1706.02677 — https://arxiv.org/pdf/1706.02677)
  Goyal et al. use the same zero-γ init as a standard ingredient of large-batch (8k) + warmup training — the regime our batch-512 + 0.15-warmup recipe is a scaled version of. Independent corroboration that zero-γ composes with, rather than replaces, warmup.
- **ReZero is All You Need** (arXiv 2003.04887 — https://arxiv.org/pdf/2003.04887)
  Identity-at-init via a learned scalar accelerates EARLY convergence dramatically at depth; corroborates the mechanism (signal propagation at init) even though ReZero itself targets much deeper nets.
- **Existing knowledge base** (knowledge/README.md): regnet-design-spaces.md now records that depth reallocation does NOT transfer to depth 20 (EXP-017); airbench reference notes init-time tricks (whitening first conv) are load-bearing in fast-CIFAR regimes — init is a known-productive axis in this domain.

## Experimental History Review

- **Current best**: 96.71 @ 1990397 (EXP-006 recipe). Twelve consecutive no-improvements (EXP-007…017).
- **Closed axes**: every training constant single-knob (certified local optimum, goal-learnings § Patterns High); schedule in every dimension — heat ±, warmup length, anneal family (EXP-010/014/015/016); uniform capacity both directions (EXP-002/005/007/008); per-stage DEPTH allocation (EXP-017: [2,3,4] at equal FLOPs ran faster yet −0.28pp — early-stage blocks are irreplaceable at depth 20); regularization both sides; batch; smoothing (EMA collapses the max-statistic).
- **What the max-statistic rewards** (project-insights Medium): a long CONVERGED plateau — cosine's cold tail manufactures ~10 near-peak evals. EXP-016 showed schedules that are still climbing at cutoff lose. Corollary never yet exploited from the FRONT of the run: anything that pulls convergence EARLIER at unchanged throughput extends the plateau without touching the closed schedule axis.
- **Untried gaps**: weight/BN INITIALIZATION (the only recipe component never probed in 18 experiments — current init is Kaiming-normal on convs with default BN γ=1); optimizer family (momentum value is heat-adjacent, likely priced by the closed heat axis); width asymmetry (exp-report-017 partially rehabilitated it: adds stage-3 capacity WITHOUT removing early depth, but pays ~14 epochs of the binding resource).

## Candidate Ideas

### 1. Zero-init residual: γ=0 in each block's final BN (bn2)
**Summary**: After `self.apply(self._weights_init)`, zero the scale of every `BasicBlock.bn2`: each residual branch then outputs 0 at init, so every block is an identity map and the net starts as (stem → identity cascade → pooled linear head). One ~3-line init-time change (`for m in model.modules(): if isinstance(m, BasicBlock): nn.init.zeros_(m.bn2.weight)`); zero params/FLOPs/dt/VRAM delta; all training constants untouched.

**Reasoning**: Two independent literature lines (Bag of Tricks, Goyal et al.) use exactly this in exactly our regime — large batch + warmup + step/cosine — and report it eases early high-LR optimization. The in-project mechanism is sharper than the literature's: under the max-over-evals metric with fixed wall clock, the gain does not need to come from a better FINAL solution — pulling the convergence onset earlier lengthens the converged plateau (more near-peak evals to max over), the exact currency EXP-011/016 identified. Our trajectory data shows the early phase is where the noise is: baseline spends ~8 epochs getting to 75%. It is also the only never-probed component of the recipe (init), so the single-knob-bracketing closure does not price it.

**Sources**: arXiv 1812.01187 § 4 (Zero γ); arXiv 1706.02677; arXiv 2003.04887 (mechanism); project-insights § Medium (converged-plateau currency); goal-learnings § Patterns High (recipe components); reports/exp-report-016.md (plateau arithmetic).

**Estimated Effort**: low (~3-line diff; single run).

**Risk Assessment**: Identity-at-init weakens early feature learning slightly (blocks must "turn on" via gradients to γ) — some reports find zero-γ neutral-to-slightly-negative on small/shallow nets at long schedules; the bet is that our time-keyed warmup + 144-epoch budget is long enough to turn all 9 blocks on and short enough that the early-easing matters. Failure mode is graceful (clean no-improvement at identical signatures). One subtlety: our shortcuts are PARAMETER-FREE (pad/stride), so identity-at-init is exact except at stage transitions, where the strided slice still downsamples — fine, the literature's setting includes projection shortcuts and works regardless.

### 2. Width asymmetry 64/128/320 ([3,3,3] preserved)
**Summary**: Widen only stage 3 from 256 → 320 (= 5×64, fully aligned); ~+1.8M params, stage-3 FLOPs ×1.56 ⇒ total FLOPs ~+17% ⇒ ~124–130 projected epochs at naive scaling (must measure compiled dt; EXP-007 says inductor gains shrink with width).

**Reasoning**: EXP-017 partially rehabilitated this: it isolates "more stage-3 capacity" from "less stage-1 depth" (the part that failed). But it spends the binding resource (epochs) — the exact mechanism that closed uniform capacity 3x — and EXP-017's deeper lesson (params are not the limiting factor; allocation position is priced steeply) cuts against any params-up move.

**Sources**: reports/exp-report-017.md § Next Steps 1 + § Unexplored Avenues; goal-learnings § Failed Approaches High; project-insights § High (alignment), § Medium (measured-dt rule).

**Estimated Effort**: low (2-line diff + dt gate).

**Risk Assessment**: Likely reruns the capacity-vs-epochs failure at reduced dose; epochs drop below ~130 makes the EXP-012 lesson (extra epochs ≈ trajectory quality, 1:1 at best) the ceiling on any gain. Graceful failure, low information content (a fourth point on a thrice-measured curve).

### 3. Whitening first-conv init (airbench-style)
**Summary**: Initialize `conv1` filters as ZCA-whitening patches computed from training data (optionally frozen), per Keller Jordan's airbench. Decorrelates inputs at the first layer from step 0.

**Reasoning**: In-domain evidence (airbench's 94%-in-seconds CIFAR recipes treat it as load-bearing) and same "init is the untried axis" logic as Idea 1. But our preprocessing keeps std=1 per-band (unnormalized variance) and TA/RE distort the patch statistics the whitening would be computed from; airbench pairs it with a GPU-resident, differently-normalized pipeline. Transfer is uncertain and implementation (patch extraction, eigendecomposition, dtype/layout interaction with channels_last+compile) is the heaviest of the three.

**Sources**: knowledge/README.md → arXiv 2404.00498 + github.com/KellerJordan/cifar10-airbench; infra-errors (CPU margin — whitening is init-time, no per-image cost, OK).

**Estimated Effort**: medium-high (~40 lines, numerical edge cases, compile interaction).

**Risk Assessment**: Failure could be non-graceful (init-scale mismatch with the untouched Kaiming layers → unstable early high-LR phase under peak 0.4); attribution muddied by interaction with the frozen normalization choice. Higher variance in both directions.

## Idea Evaluation

**Evidence strength**: Idea 1 wins — two independent peer-reviewed recipes apply it in our exact regime (large batch + warmup), and it attacks the only never-probed recipe component. Idea 3's evidence is in-domain but from a materially different pipeline. Idea 2's evidence base is mostly negative in-project.

**Mechanism clarity**: Idea 1's mechanism is doubly grounded: literature (identity-at-init eases early high-LR optimization) AND project-specific (earlier convergence onset ⇒ longer converged plateau ⇒ more near-peak evals for the max-statistic — the EXP-011/016 currency, exploited from the front of the run for the first time). Idea 2's mechanism must overcome a measured 3x-failed trade. Idea 3's mechanism is clear but its preconditions (input statistics) don't match our pipeline.

**Expected impact**: literature suggests +0.2–1pp at fixed epochs for Idea 1 components; even the conservative in-project translation (a few extra converged evals) targets exactly the +0.1pp bar. Idea 2's best case is bounded by the 1:1 epochs-for-quality exchange measured in EXP-012. Idea 3 high variance.

**Risk profile**: Idea 1 is the safest possible intervention — zero cost on every measured signature, graceful failure, perfect attribution (init-only diff). Idea 3 is the riskiest. Idea 2 graceful but low-information.

**Feasibility**: Idea 1 is a 3-line diff; Idea 2 trivial; Idea 3 medium-high.

Idea 1 dominates on every criterion.

## Chosen Idea
**Selected**: Zero-init residual: γ=0 in each block's final BN (bn2)

**Why this idea**:
It is the only never-probed component of the certified-optimal recipe (initialization), carries two independent peer-reviewed precedents from exactly our training regime (large batch + warmup: Bag of Tricks, Goyal et al.), costs nothing on any measured signature (params/FLOPs/dt/VRAM identical — perfect attribution), and its in-project mechanism targets the metric's actual currency: easing the early high-LR phase pulls the convergence onset earlier, lengthening the converged plateau the max-statistic harvests (project-insights Medium, EXP-011/016). The structural axes (schedule, capacity, allocation) are closed; init is the live one.

**Hypothesis**:
With γ=0 in all nine bn2 layers and every other byte identical to baseline @ 1990397, the run shows a visibly faster early trajectory (test_acc at epochs 5–20 above baseline's 63.8–76.1 trail), unchanged throughput signatures (~22.4ms dt, ~139 epochs, ~1613MB VRAM, 4,286,026 params), an equal-or-longer converged plateau, and best_test_acc ≥ 96.81% (baseline 96.71 + 0.1). If instead the early gain decays to a converged plateau at-or-below baseline, the result cleanly establishes that init effects wash out within the 139-epoch budget, closing the cheap end of the init axis.
