# Brainstorm EXP-015
**Created**: 2026-06-30

<!-- Ideation only. Goal/metric/constraints live in goals/{slug}/01-definition.md; baseline in 04-results.tsv. -->

## Web Search & Literature Review

- **TrivialAugment: Tuning-Free Yet SOTA Data Augmentation** (Müller & Hutter, ICCV 2021 — https://openaccess.thecvf.com/content/ICCV2021/papers/Muller_TrivialAugment_Tuning-Free_Yet_State-of-the-Art_Data_Augmentation_ICCV_2021_paper.pdf)
  Parameter-FREE: applies ONE uniformly-sampled op at a uniformly-sampled magnitude per image — no N/M to tune. Matches RL-searched AutoAugment/RandAugment across CIFAR-10/100 on WRN-28-10 & ShakeShake. The "wide" magnitude range is aggressive; stacks on the standard flips+pad-crop+Cutout baseline.
- **RandAugment** (Cubuk et al., CVPRW 2020 — https://arxiv.org/abs/1909.13719)
  Reduced search space, 2 hyperparams: `num_ops` N (default 2), `magnitude` M (default 9, of 31 bins). On CIFAR-10 the **baseline already includes flips+pad-crop+Cutout**, and RandAugment ADDED on top reaches "competitive (within 0.1%) or SOTA across four architectures" — i.e. the increment over a Cutout baseline is architecture-dependent and sometimes only ~0.1%.
- **Practitioner benchmark, ResNet-18/CIFAR-10** (Raschka, https://sebastianraschka.com/blog/2023/data-augmentation-pytorch.html)
  AutoAugment ≈ +12% over NO aug; RandAugment +~2pp over that. CRUCIAL CAVEAT for our budget: these were run to **1000–2000 epochs**, and the validation-accuracy slope was still positive at 1000ep → strong policy aug NEEDS long schedules to converge. The increment over a flips+crop+Cutout baseline (what we have) is far smaller than the headline over-no-aug numbers.
- **Existing knowledge**: `knowledge/references/mixing-augmentation.md` (EXP-011 CutMix tied — mixing aug saturated), `knowledge/references/fast-cifar10-recipes.md` (airbench deliberately uses LIGHT aug because it trains only ~10–40 epochs — too few to absorb strong aug; we run ~150ep, the opposite regime).

## Experimental History Review

- **Current best 96.38 (EXP-008)**; 8 straight no-improvements since (EXP-006→014). Strong **generalization-ceiling** diagnosis (~96.3–96.5), NOT epoch/throughput-bound — proven by EXP-014: torch.compile bought a clean +12% epochs (154→173) yet accuracy was flat (+0.03pp), and compile-funded capacity (layer2 256→320, annealed at a healthy 143ep) LOST −0.08pp.
- **What worked**: DavidNet+one-cycle (EXP-001), EMA+flip-TTA (EXP-002), ZCA whitening conv (EXP-003), ONE ReZero block @layer2/8×8 (EXP-004), **stronger occlusion aug Cutout12+RandomErasing (EXP-008, +0.38pp — the largest lever since EXP-001)**.
- **Exhausted axes (do not retry same approach)**: optimizer swap (Muon, EXP-009/010 — regularization-bound not optimizer-bound); width@8×8 (EXP-007/014 capacity-saturated); depth@4×4 (EXP-005 — unused + slow kernel); **mixing aug (CutMix, EXP-011 — tied, redundant with occlusion)**; regularization scalars WD-shaping + LS retune (EXP-012); loss-geometry tail-SAM (EXP-013 — 2× cost under-anneals); eval-side multi-crop TTA (EXP-006); buying epochs (EXP-014).
- **Untried gaps**: (a) **policy-based geometric+photometric augmentation (AutoAugment/RandAugment/TrivialAugment)** — a DIFFERENT aug mechanism than the occlusion (Cutout/RE) and mixing (CutMix) already tried; (b) **anneal SHAPE** (triangular → cosine / longer tail) — flagged throughput-free + untested in EXP-012; (c) **depth at the proven 8×8 stage** (2nd ReZero block @layer2) — only width@8×8 and depth@4×4 were tested.
- **Protocol constraints**: ~0.1pp run-to-run noise floor → require SAME-SESSION baseline + clearly >0.1pp effect; fixed seed (no re-roll); watch `num_epochs` (~142–155 clean band) as the first-class throughput diagnostic; budget is COMPUTE-time (dataloader wait is off the per-step timer → a CPU-aug bottleneck inflates WALL `total_seconds` toward the 600s cap rather than cutting epochs — watch wall too).

## Collected Ideas

- **(lit/orthogonal)** Policy augmentation: insert `RandAugment`/`TrivialAugmentWide`/`AutoAugment(CIFAR10)` (torchvision, already a dep) before ToTensor — a geometric+photometric aug class never tried here.
- **(orthogonal)** Anneal-shape reshape: triangular → cosine decay, or extend the low-LR tail (lower PCT_START, or a flatter tail) — throughput-free.
- **(history recombine)** 2nd ReZero block (depth) at the proven layer2/8×8 stage, funded by EXP-014's banked +12% compile throughput so it anneals.
- **(simplification)** DROP RandomErasing and REPLACE with policy aug (avoid stacking 3 occlusion-like augs that over-regularize a 150ep budget).
- **(algorithm)** Higher whitening-enabled PEAK_LR with a longer warmup (whitening conditions the input; EXP-003 noted "whitening-enabled higher LR" untried).
- **(orthogonal)** Dropout before the FC layer (classic regularizer, untested on this net).
- **(moonshot)** Curriculum augmentation: ramp policy-aug strength UP early then OFF in the low-LR tail so EMA averages clean-image iterates (most accuracy lands in the tail here).

## Combinations

- **Policy-aug + curriculum tail-off**: strong aug raises the generalization ceiling but needs epochs to converge; disabling it in the final ~15% lets the net anneal/EMA on clean data — directly mitigates the 150-epoch under-fit risk that is policy-aug's main threat here.
- **Policy-aug (REPLACE RandomErasing) + mild magnitude**: swap one occlusion lever for the richer transform lever instead of stacking, keeping total regularization load tuned to the budget.
- **Policy-aug + cosine tail**: a smoother low-LR tail gives the harder (policy-augmented) task more effective anneal time where accuracy concentrates.

## Candidate Ideas

### 1. Policy-based augmentation (RandAugment / TrivialAugment) with strength control + curriculum tail-off
**Summary**: Insert a torchvision policy-augmentation transform (`RandAugment(num_ops, magnitude)`, `TrivialAugmentWide()`, or `AutoAugment(AutoAugmentPolicy.CIFAR10)`) into `train_tf` immediately after `RandomCrop`+`RandomHorizontalFlip` and before `ToTensor` (these transforms consume PIL/uint8). This adds a genuinely new augmentation MECHANISM — random geometric (rotate, shear, translate) and photometric (contrast, brightness, color, sharpness, posterize, solarize, equalize) transforms — distinct from the occlusion (Cutout/RandomErasing) and mixing (CutMix) classes already tested. Because strong policy aug makes the task harder and the canonical recipes use 200–2000 epochs (we have ~150), the design must control strength: test MILD settings (e.g. RandAugment N=1–2, M≈6–9; or TrivialAugment), optionally apply probabilistically (p<1), optionally REPLACE RandomErasing rather than stack, and optionally CURRICULUM tail-off (disable in the final ~15% so the net anneals on clean data and EMA averages clean iterates). Run as a SAME-SESSION multi-cell sweep: c0 = baseline control, then 2–3 strength/curriculum variants. It is throughput-free (CPU workers), so `num_epochs` should stay ~150; watch wall `total_seconds` for a CPU bottleneck.
**What it targets**: The diagnosed **generalization ceiling** (`project-insights.md` High-Importance EXP-014 entry; `03-experiment-learnings.md` Low-Failed EXP-014). A generalization ceiling on fixed data is canonically raised by increasing effective data diversity. The two aug mechanisms tried (occlusion EXP-008 won once; mixing EXP-011 tied) do NOT include the transform-based policy class that is THE documented lever taking CIFAR-10 ResNets from ~96→97%+. It attacks the limiter directly and throughput-free, the lane EXP-008 proved most productive.
**Reasoning**: (a) The single largest post-EXP-001 win was stronger augmentation (EXP-008 +0.38pp); augmentation is the proven productive lane on this net. (b) Policy aug is a mechanistically DIFFERENT class from everything tried — the diminishing-returns insight ("a 2nd SAME-class lever ties") was established for a 2nd occlusion/mixing aug, not for an unexplored class. (c) torchvision ships all three transforms (no new dependency). (d) Our ~150-epoch budget is far longer than airbench's ~37 — we are on the right side of the "enough epochs to absorb aug" line, unlike fast recipes that avoid strong aug.
**Sources**: TrivialAugment (ICCV 2021), RandAugment (CVPRW 2020), Raschka practitioner benchmark; `03-experiment-learnings.md` Patterns/EXP-008; `knowledge/references/mixing-augmentation.md`; `knowledge/references/fast-cifar10-recipes.md`. Proposal: `proposals/idea-01.md`.
**Estimated Effort**: low (a few lines in `train_tf` + an env-toggle for strength/curriculum; multi-cell same-session run).
**Risk Assessment**: PRIMARY RISK — **under-fit at 150 epochs**: strong policy aug's canonical gains are at 200–2000ep; too-aggressive magnitude could depress the annealed optimum (LOSE), visible as a low ep25 and best==final-still-climbing. Mitigations: mild magnitude, probabilistic apply, REPLACE-not-stack RandomErasing, curriculum tail-off; the same-session sweep brackets strength. SECONDARY — increment over a Cutout baseline can be small ("within 0.1%" on some architectures, RandAugment paper) → may land within the noise floor; bias toward a clearly-mild-but-real setting and read the same-session delta. TERTIARY — CPU-bound dataloader inflates WALL time toward the 600s cap (not epochs); watch `total_seconds`.

### 2. Anneal-shape reshaping (triangular → cosine / extended low-LR tail), throughput-free
**Summary**: Replace the current piecewise-linear triangular one-cycle (`lr = PEAK·progress/PCT` then `PEAK·(1−progress)/(1−PCT)`) with a cosine half-period decay after warmup, and/or extend the low-LR tail (e.g. decay to a small floor over a longer fraction, or lower PCT_START so more of the budget is annealing). Most accuracy lands in the low-LR tail (EXP-001 Pattern), so the SHAPE of that tail — how long the model spends at small LR — may shift the annealed optimum. Pure schedule math, zero throughput cost, `num_epochs` unchanged. Same-session multi-cell over 2–3 shapes vs the triangular control.
**What it targets**: The generalization ceiling, via better use of the anneal phase where accuracy concentrates. Distinct from capacity/aug — it reshapes HOW the existing capacity converges rather than adding any.
**Reasoning**: Schedule shape was explicitly flagged untried + throughput-free in EXP-012's Insight ("the only untried levers with ceiling clearly above noise are schedule-shape and a mild capacity step"). Cosine is the de-facto SOTA anneal for CIFAR ResNets; the current linear tail may leave a small amount on the table.
**Sources**: `03-experiment-learnings.md` EXP-012 Insight (schedule-shape untried); EXP-001 tail-concentration Pattern. (Developed inline; not advanced to a proposal file.)
**Estimated Effort**: low (rewrite the LR block; env-toggle the shape).
**Risk Assessment**: Likely SUB-NOISE — schedule shape is a tuning move on an already-well-annealed recipe; the triangular one-cycle is already near-optimal for this net, so the expected effect may sit at/under the ~0.1pp floor. Worst case ties baseline. Lower expected ceiling than policy aug, but near-zero downside and zero throughput cost.

### 3. Capacity via DEPTH at the proven layer2/8×8 stage (2nd ReZero block), compile-funded
**Summary**: Add a SECOND `GatedResidual(256)` (ReZero, α=0 identity-init) to layer2, the 8×8 stage where the FIRST ReZero block won (EXP-004, +0.13pp). Only width@8×8 (EXP-007/014, capacity-saturated) and depth@4×4 (EXP-005, unused+slow kernel) have been directly tested; depth at the proven full-throughput 8×8 stage has NOT. Fund the per-step cost with EXP-014's banked off-budget torch.compile (+12% throughput) so the added block anneals (~150ep), avoiding the under-anneal trap.
**What it targets**: Residual representational capacity at 8×8 — testing whether the ceiling is a WIDTH saturation (EXP-014) that depth can still move, or a stage-level capacity saturation that depth cannot.
**Reasoning**: EXP-014 registered "different base architecture" as the top next step; this is the most surgical capacity-relocation consistent with the one capacity lever (depth@8×8) not yet falsified. ReZero needs no LR retune (clean single-variable test).
**Sources**: `03-experiment-learnings.md` EXP-004 Pattern / EXP-014 Failed-Low / Medium under-anneal; `knowledge/references/rezero-identity-init.md`, `knowledge/references/torch-compile-throughput.md`. (Developed inline; not advanced to a proposal file.)
**Estimated Effort**: medium (add block + integrate the compile-warmup recipe from EXP-014).
**Risk Assessment**: LIKELY REDUNDANT with the capacity-saturation finding — EXP-014's healthily-annealed 320-width still lost, strongly implying 8×8 capacity is saturated regardless of width-vs-depth. Adds compile complexity (BN-restore, param-aliasing, eval-boundary traps) for a low-confidence capacity bet. Medium-low confidence; included as the registered "capacity" probe but ranked below the untested aug class.

## Review

Cross-model adversarial review (Codex) in `01-idea-review.md`. Scored verdict: **Idea 1 wins** (evidence 6/10, impact 7/10) over Idea 2 (5/3 — overlaps the anneal/epoch axis EXP-014 falsified) and Idea 3 (4/4 — too aligned with the capacity path EXP-014 just closed). Top concerns and resolutions:

1. **EXP-014 closed the capacity/epochs axis → Ideas 2 & 3 are low-EV diagnostics, not headline bets.** Resolved: adopt Idea 1; demote 2/3 to non-pursued alternates.
2. **Idea 1 overstates the aug lane — a 2nd strong input aug (CutMix) already tied; policy aug can UNDER-FIT at ~150ep. Make cells mild & COMPARATIVE: baseline / policy REPLACING RandomErasing/Cutout / one conservative variant — do NOT stack strong aug on top.** Resolved: redesigned to a 3-cell replacement design (c0 baseline; cA RandAugment(1,6) replacing RandomErasing; cB TrivialAugment replacing RandomErasing), mild magnitude, no stacking (`proposals/idea-01.md`).
3. **Implementation trap: `persistent_workers=True` means mid-training mutation of `train_set.transform` won't reach worker copies → curriculum tail-off silently no-ops.** Resolved: DROPPED the curriculum tail-off; use FIXED mild strength (cleaner single-variable test, no worker-visibility hazard).
4. **"Throughput-free" needs wall-time proof — CPU-side policy aug can inflate WALL `total_seconds` toward the 600s cap without cutting `num_epochs`.** Resolved: pre-smoke worker throughput; make BOTH `num_epochs` (~142–155) and `total_seconds` (<600s) verdict metrics.
5. **Idea 2 likely sub-noise; Idea 3 poor post-EXP-014 prior.** Resolved: not pursued this loop.
7. **Missing proposal files lowered auditability.** Resolved: wrote `proposals/idea-01.md` (the lead, refined); ideas 2/3 documented inline.

## Idea Evaluation

Adopt the reviewer's pick — **Idea 1, policy-based augmentation** — with the strict mild/replacement design. It is the only candidate attacking a mechanism not already directly falsified: the diagnosed limiter is a generalization ceiling, and the only augmentation MECHANISM untested here (transform-based policy aug) is the canonical lever for raising CIFAR-10 ResNet accuracy past ~96. Ideas 2 (schedule shape) and 3 (depth@8×8) both sit on axes EXP-014 effectively closed (anneal/epochs and within-architecture capacity) and scored below Idea 1 on both criteria. No override. Full scored critique: `01-idea-review.md`.

## Chosen Idea
**Selected**: Idea 1 — Mild policy-based augmentation (RandAugment / TrivialAugment), replacement design (`proposals/idea-01.md`).

**Why this idea**:
After 8 no-improvements the net is at a generalization ceiling, and the highest-EV remaining move attacks data diversity — the canonical way to raise such a ceiling on fixed data. Augmentation is the proven productive lane here (EXP-008's Cutout12+RandomErasing was the largest post-EXP-001 win, +0.38pp), and the ONE augmentation mechanism never tried is transform-based policy aug (geometric+photometric), mechanistically distinct from the occlusion (EXP-008 won) and mixing (EXP-011 tied) classes. torchvision ships RandAugment/TrivialAugment/AutoAugment (no new dependency), and our ~150-epoch budget is far longer than fast recipes that avoid strong aug — the right regime to absorb it. The reviewer ranked it the clear winner; its one real risk (under-fit at 150ep) is bracketed by the mild + replacement + same-session-control design.

**Hypothesis**:
Replacing RandomErasing with a MILD policy augmentation (RandAugment num_ops=1, magnitude≈6, or TrivialAugmentWide) while keeping Cutout12 will add geometric+photometric diversity orthogonal to occlusion and raise `best_test_acc` clearly above the SAME-SESSION baseline (>0.1pp) without under-fitting — i.e. ep25 within ~0.5pp of the baseline, `num_epochs` in the ~142–155 band, and wall `total_seconds` < 600s. If instead every policy cell ties-or-loses the same-session control at healthy epoch counts and normal ep25, the input-augmentation lane is confirmed saturated across all three mechanisms and the ceiling is not augmentation-movable.
