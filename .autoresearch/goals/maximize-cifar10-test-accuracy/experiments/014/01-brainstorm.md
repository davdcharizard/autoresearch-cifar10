# Brainstorm EXP-014
**Created**: 2026-06-29

<!-- Goal/metric/constraints live in goals/maximize-cifar10-test-accuracy/01-definition.md; baseline (96.38) in 04-results.tsv. -->

## Web Search & Literature Review

- **"94% on CIFAR-10 in 3.29 Seconds on a Single GPU" (airbench, arXiv:2404.00498)**: applied `torch.compile` to a near-identical fast-CIFAR ResNet for a **~14% throughput reduction**, "mathematically equivalent (up to small float differences) to the non-compiled variant"; explicitly notes the one-time multi-minute compile is only worthwhile amortized over many steps → maps directly onto our **off-budget compile-warmup** pattern (compile before `t_start_training`, like the off-budget whitening eigendecomp).
- **General torch.compile reports on small CIFAR convnets** (NVIDIA Nsight/PyTorch 2.0 blog; TorchInductor tips): 7–14% from compile alone; up to ~40% stacked with cudnn.benchmark + AMP; `mode="reduce-overhead"`/default balances compile-time vs win for training, `max-autotune` for max throughput. Inductor's win on a conv net is the elementwise/normalization glue (BN affine + ReLU + residual add + ReZero scalar-mul + autocast casts), not the cuDNN convs.
- **Ghost BatchNorm** (Hoffer et al. NeurIPS 2017 "Train longer, generalize better"; "Four Things Everyone Should Know to Improve BatchNorm" arXiv:1906.03548; "A New Look at Ghost Normalization" arXiv:2007.08554; "Ghost Noise for Regularizing DNNs" arXiv:2305.17205): normalizing over small sub-batches injects statistic-noise as a throughput-free regularizer; ~0.5–1.0pp gains in related settings; helps even under **super-convergence on CIFAR-10** (our one-cycle regime). The classic cifar10-fast/DavidNet recipe splits batch 512 into ghost batches — the one major recipe component never adopted here.

## Experimental History Review

Baseline **96.38** (EXP-008). Six axes exhausted within the ~0.1pp noise floor across 7 straight no-improvements (EXP-006→013):
- **Optimizer** (EXP-009/010 Muon): tuned Muon ties SGD (96.33). Optimizer-bound → falsified.
- **Eval-TTA** (EXP-006 multi-crop): translate-over-mirror increment < noise.
- **Input-aug** (EXP-008 strong-aug WON +0.38pp → 96.38; EXP-011 CutMix 2nd aug ties +0.02pp): input-space aug saturated after the first strong one.
- **Regularization scalars** (EXP-012 WD-shaping ties, LS<0.2 degrades): allocation axis exhausted; the ReZero gate is "not accuracy-limiting".
- **Loss-geometry** (EXP-013 SAM): tail-only SAM loses — 2× compute cost removes ~26 anneal epochs.
- **Large capacity** (EXP-005 4×4 deepen −0.10; EXP-007 256→384 widen cut 150→94, −0.15, still-climbing): under-anneal, NOT capacity-saturation. Pre-registered milder step **256→320** never run.

**What worked & is in the base recipe**: DavidNet/ResNet-9 + time-based one-cycle (EXP-001), EMA+flip-TTA (EXP-002), frozen ZCA whitening conv (EXP-003), ReZero Residual(256)@layer2 (EXP-004 +0.13pp — capacity at the 8×8 stage IS useful when it anneals), strong CPU aug (EXP-008).

**Untried gaps**: (1) **throughput itself** — every prior experiment reshuffled within the ~150-epoch budget or SPENT epochs; none tried to BUY more via faster code. (2) The pre-registered **mild capacity 256→320**. (3) **Ghost BN** — the one major DavidNet recipe component never adopted.

## Diagnosis — what limits the objective

The net is **regularization-bound near a generalization ceiling at ~96.4** *within a fixed 300s budget that fits ~150 epochs*. The unifying constraint behind all 7 no-improvements is the **epoch budget itself**: at fixed wall-time, throughput sets the epoch count, and the epoch count gates BOTH anneal completion (most accuracy lands in the low-LR tail, EXP-001) AND realizable capacity (EXP-005/007/013 all under-annealed). Every lever WITHIN the ~150-epoch budget has saturated; every lever that EXPANDS capacity at the cost of epochs under-anneals. The one lever never touched is **throughput** — increasing it is the *meta-lever* that either directly adds anneal epochs or funds capacity that previously under-annealed. This is the highest-leverage remaining direction precisely because it changes the binding constraint rather than reshuffling within it.

## Collected Ideas

- **[Throughput/meta-lever]** `torch.compile` (TorchInductor) with off-budget compile warmup → fuse BN/ReLU/ReZero-mul/autocast glue → +7–15% img/s → more anneal epochs OR fund capacity. *(genuinely untried; attacks the epoch-budget limiter directly)*
- **[Experimental history]** Mild capacity widen layer2 256→320 (pre-registered milder retry of EXP-007's failed 384).
- **[Literature/regularization]** Ghost BatchNorm — sub-batch BN statistics for distinct, throughput-free regularization noise; proven DavidNet trick.
- **[Throughput]** Reduce per-step `torch.cuda.synchronize()` granularity (sync every N steps, time the block) to remove CPU/GPU serialization bubbles — honest budget accounting preserved.
- **[Schedule]** Cosine / earlier-peak anneal-tail reshape — throughput-free, cannot under-anneal; spends more time at low LR. *(low confidence; literature gains <0.1pp)*
- **[Algorithm/rep]** Dirac/identity-init learnable stem + whitening-enabled higher PEAK_LR — faster early convergence to free tail epochs. *(untried EXP-003 rider)*
- **[Moonshot/combo]** `torch.compile` throughput headroom SPENT on the mild capacity add — compile buys ~12%, capacity costs ~12% → epochs hold ~150 WITH more capacity that now ANNEALS, resolving the EXP-007 under-anneal root cause.

## Combinations

- **compile + mild capacity (256→320)**: compile's throughput funds the capacity epoch-cost so it anneals — beats either alone (compile-only may be anneal-saturated at 150; capacity-only under-anneals). This is the high-upside framing, structured as a same-session cell within idea-01.
- **compile + less per-step sync**: stack two throughput levers for a larger epoch gain (both off-budget/free).
- **compile + cosine tail**: more epochs AND a better-shaped anneal where accuracy concentrates.

## Candidate Ideas

### 1. torch.compile throughput (off-budget compile warmup), optionally funding mild capacity
**Summary**: Wrap the training model in `torch.compile` and pay the one-time compilation OFF-BUDGET via a warmup forward+backward at the exact static train shape (512,3,32,32) channels_last/bf16, placed before `t_start_training` (mirroring the off-budget whitening eigendecomp). The timed loop then runs an already-compiled model at higher img/s. Same-session 3-cell env-toggle design: cell-0 no-compile baseline, cell-A compile-only (spend throughput on more anneal epochs ~150→~165), cell-B compile + layer2 256→320 (spend the throughput to offset the capacity epoch-cost so the added capacity anneals). EMA/eval stay correct for free: the `AveragedModel` EMA copy is uncompiled, so eval runs uncompiled (zero eval recompile). Full proposal: `proposals/idea-01.md`.

**What it targets**: The epoch-budget meta-limiter (diagnosis above; learnings §Failed/Medium under-anneal). It is the ONLY candidate that increases the number of available optimizer-update/anneal epochs rather than reshuffling within or spending them — directly attacking the constraint behind EXP-005/007/013.

**Reasoning**: airbench (arXiv:2404.00498) got ~14% throughput from `torch.compile` on a near-identical recipe, math-equivalent. The off-budget warmup makes the compile cost free w.r.t. the 300s budget — the codebase already has precedent (off-budget whitening). Inductor fuses exactly this net's elementwise/BN/ReZero glue. cell-B specifically resolves the EXP-007 under-anneal root cause: capacity at 8×8 is proven useful (EXP-004 +0.13pp) but bigger steps under-annealed; throughput headroom lets capacity anneal.

**Sources**: `proposals/idea-01.md`; arXiv:2404.00498 (airbench); `train.py` budget accounting (268–314), off-budget whitening (232–238), EMA (255–257); learnings §Failed/Medium (under-anneal), §Patterns (EXP-004 capacity-at-8×8); project-insights §High (throughput buys epochs; per-step-cost trades against updates).

**Estimated Effort**: Medium — compile + warmup is ~15 lines; cell-B width parametrization + 3-cell harness adds modest plumbing.

**Risk Assessment**: (a) Compile cost leaking on-budget if warmup is misplaced/shape-mismatched → catastrophic first-step inflation; mitigated by exact-shape warmup + `drop_last=True`. (b) Gains <5% on an already-fast net → cell-A ties cell-0 (clean negative on the throughput lever). (c) cell-A marginal-anneal: extra epochs past ~150 may give little if anneal-saturated. (d) cell-B under-anneal if compile gain < capacity cost (num_epochs is the pre-registered gate). (e) Inductor bf16 drift (small; same-session control absorbs it). (f) OptimizedModule/optimizer param-aliasing — verify in smoke.

### 2. Mild capacity widen layer2 256→320 (standalone, pre-registered)
**Summary**: Widen the proven 8×8 layer2 stage from 256→320 channels — the explicit pre-registered next step after EXP-007's 256→384 under-annealed. Two integer-literal edits in `ResNet9.__init__` (conv_bn(128,320), GatedResidual(320), conv_bn(320,512) layer3 stem). +1.03M params (~47% of 384's +2.21M, i.e. the "~1.25× cost" step the learnings call for). The GatedResidual ReZero α=0 identity-init means no LR retune (clean single-variable capacity probe, like EXP-004). Full proposal: `proposals/idea-02.md`.

**What it targets**: The capacity-vs-epoch tradeoff at the fixed budget (learnings §Failed/Medium). 384 added real capacity but cost too many epochs (150→94, still-climbing); 320 bets on landing on the profitable side of the curve.

**Reasoning**: Capacity at the 8×8 stage is the proven full-throughput place to add it (EXP-004 +0.13pp; EXP-005 showed 4×4 is kernel-slow AND unused). 320 adds ~half of 384's marginal cost → predicted ~120–135 epochs (vs 384's 94), above the ~110 under-anneal cliff. EXP-004 won (+0.13pp) while also dropping epochs (174→142), so a mild-capacity/mild-epoch-loss step can net positive.

**Sources**: `proposals/idea-02.md`; EXP-007 analysis (pre-registers 256→320), EXP-004 (capacity-at-8×8), EXP-005 (FLOP≠wall-clock), EXP-012 (gate not accuracy-limiting); project-insights §High.

**Estimated Effort**: Low — two integer literals; one run + mandatory same-session 256 control cell.

**Risk Assessment**: (a) Under-anneal (dominant) — if 320 lands <120 epochs, mirrors EXP-007. (b) Capacity genuinely near-saturated (EXP-012 probe) → even well-annealed 320 ties. (c) cuDNN width-320 efficiency (320=64×5, generally fine; 288 fallback). (d) Thin margin near the ~0.1pp noise floor.

### 3. Ghost BatchNorm (sub-batch statistic-noise regularization)
**Summary**: Replace the 10 `nn.BatchNorm2d` in `conv_bn` with a hand-written pure-torch `GhostBatchNorm2d` that, in training, normalizes each forward over independent ghost sub-batches of the 512 mini-batch (try 4×128 — DavidNet default — and 8×64) via the classic `[N,C,H,W]→[N/g, g*C, H, W]` single-`F.batch_norm` reshape (no Python loop → throughput-free), folding the g per-ghost running-stat updates back into the `[C]` buffers so eval BN stays calibrated. Distinct regularization space (BN activation statistics) — orthogonal to every regularizer already tried. Full proposal: `proposals/idea-03.md`.

**What it targets**: The regularization ceiling — but via the one mechanism (BN-statistic noise) NOT yet perturbed (vs input pixels / weights / targets / loss-geometry already tried), and the one major DavidNet recipe component genuinely missing here.

**Reasoning**: Ghost BN is the canonical large-batch generalization regularizer (Hoffer 2017), reported ~0.5–1.0pp in related settings and shown to help under super-convergence on CIFAR-10 (our regime). Throughput-free. It closes a real recipe gap rather than adding a redundant 2nd-of-a-class regularizer.

**Sources**: `proposals/idea-03.md`; arXiv:1906.03548, 2007.08554, 2305.17205; Hoffer NeurIPS 2017; `knowledge/references/fast-cifar10-recipes.md`; learnings (regularization-bound, throughput-free sub-lever saturation EXP-011/012/013).

**Estimated Effort**: Medium — ~20-line module but correctness traps (buffer-update fold, channels_last reshape, bf16 affine) demand smoke-testing.

**Risk Assessment**: (a) Over-regularization / tie (MOST LIKELY — net is at its regularization ceiling; could depress early convergence like CutMix with no annealed gain; honestly more-likely-than-not to tie). (b) Running-stat miscalibration breaking eval BN (#1 correctness trap; mitigated by the `.mean(0)` fold + eval-acc smoke). (c) Throughput regression if channels_last reshape forces a per-BN copy (num_epochs is the gate). (d) bf16 view/affine dtype mismatch.

## Review

Cross-model adversarial review by Codex (`01-idea-review.md`). Verdict — **Idea-01: impact 7/10, soundness 6/10 (PICK); Idea-02: 4/10 impact, 7/10 soundness; Idea-03: 5/10 impact, 4/10 soundness.** Codex picks **Idea-01, specifically the compile-funded layer2-320 configuration** (NOT compile-only), as the only finalist with a credible path to >0.1pp by changing the epoch/capacity tradeoff that sank EXP-007.

Top concerns and resolutions (all folded into the chosen idea / deferred to planning):
1. **Compile cost staying off-budget** — confirmed sound IF warmup is before `t_start_training` (line 268) at exact train shape/dtype/autocast/train-mode/channels-last + backward; any missed graph compiling inside the loop is charged via the in-loop `synchronize()`. → Warmup discipline is mandatory; smoke must confirm the first timed steps are NOT inflated.
2. **BN running-stat pollution during warmup (real bug the proposal missed)** — warmup forward passes on dummy data mutate BN `running_mean/var`. → **Snapshot all BN buffers before warmup and restore after** (warmup does no `optimizer.step`, so params/momentum are already clean; just zero grads + restore BN buffers). Resolution adopted.
3. **Eval recompile vs 10-min wall** — pre-EMA early epochs would eval the compiled model at batch 256/16, compiling eval graphs (off training-budget but on the wall cap). → **Do NOT rebind `model`**; keep the original `model` uncompiled for ALL eval/EMA and use a separate `train_fwd = torch.compile(model)` ONLY for the training forward. The `AveragedModel` EMA copy is already uncompiled. Zero eval recompile. Resolution adopted (cleaner than the proposal's in-place rebind).
4. **3-cell wall-cap** — Codex flagged a single back-to-back process could breach 600s. → Each cell is a SEPARATE `train.py` process under its own `timeout 600` (each <600s wall), launched sequentially by one driver — exactly how EXP-012/013 passed NC1. No single process exceeds the wall. Clarified for planning.
5. **param/EMA aliasing under torch.compile** — confirmed sound in torch 2.9.1 (`OptimizedModule` shares the same tensors via `_orig_mod`); still smoke-test that `optimizer.step()` changes what the compiled forward sees.
6. **EV** — compile-only is probably sub-noise (extra epochs past ~150 worth <0.1pp near the ceiling); the high-upside cell is **compile + 320**. Standalone 320 (idea-02) is lower EV (spends epochs in a regime where capacity adds lost). → Headline = compile+320; compile-only and no-compile are controls. If compile throughput is negligible in smoke, do NOT promote standalone 320 as the fallback.

## Idea Evaluation

Adopt the reviewer's pick: **Idea-01 (torch.compile throughput), headlined by the compile-funded layer2 256→320 cell.** This is the only candidate that attacks the diagnosed limiter (the epoch budget) directly rather than reshuffling within it — and the compile+320 framing specifically resolves the EXP-007 under-anneal root cause (capacity at the proven 8×8 stage that previously couldn't anneal now gets the throughput to do so). Idea-02 is absorbed as cell-B's width change but is strictly stronger when funded by compile than standalone. Idea-03 (Ghost BN) is deprioritized: lowest soundness (BN running-stat/layout hazards, "throughput-free" unproven) and modal-tie EV on an already regularization-bound net — kept on the shelf as the strongest remaining throughput-free regularization probe if the throughput thesis fails. Full scored critique in `01-idea-review.md`.

## Chosen Idea
**Selected**: torch.compile throughput (off-budget compile warmup), headlined by the compile-funded layer2 256→320 cell.

**Why this idea**:
Seven straight no-improvements have exhausted every lever WITHIN the fixed ~150-epoch budget (optimizer, eval-TTA, input-aug, regularization scalars, loss-geometry) and every lever that SPENDS epochs has under-annealed (EXP-005/007/013). The diagnosis isolates the **epoch budget itself** as the binding constraint. `torch.compile` is the only candidate that increases the epoch budget rather than reshuffling within it: the airbench paper (arXiv:2404.00498) got ~14% throughput from it on a near-identical recipe, math-equivalent, and the off-budget warmup pattern (precedented by the off-budget whitening eigendecomp) keeps the one-time compile cost out of the 300s budget. Codex's pick and the EV analysis both land on **compile-funded capacity**: compile-only is likely sub-noise near the ceiling, but spending the bought throughput on layer2 256→320 lets capacity that previously under-annealed (EXP-007) actually complete its anneal — the highest-upside remaining move, attacking the exact failure mode that sank capacity adds.

**Hypothesis**:
`torch.compile` (with an off-budget warmup at the exact static train shape, BN buffers restored) yields **+7–15% training img/s** with the compile cost OFF the 300s budget (off-budget warmup → `training_seconds`≈300 unchanged, first timed steps not inflated). Concretely, across three same-session separate-process cells (each <600s wall):
- **cell-0** (no compile, 256): same-session baseline ≈ 96.3–96.5 @ ~150 epochs.
- **cell-A** (compile, 256): ~+10% img/s → ~160–170 epochs; accuracy ties-to-marginally-beats cell-0 (≤+0.1pp; tests whether extra anneal epochs help near the ceiling — likely sub-noise).
- **cell-B** (compile, 320): the compile throughput offsets the ~1.25× capacity cost so epochs hold ≥~140 (vs EXP-007's 94) WITH +1.03M annealing capacity at the proven 8×8 stage → **clears 96.48 (+0.1pp over the 96.38 baseline) AND beats cell-0 by >0.1pp**.
Falsifiable: if smoke shows compile throughput <5%, or cell-B under-anneals (epochs <120, best==final still-climbing), the throughput-funded-capacity thesis is rejected and the epoch budget is confirmed irreducible by code-fusion at this net/GPU.
