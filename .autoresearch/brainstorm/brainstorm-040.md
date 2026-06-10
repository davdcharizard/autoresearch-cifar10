# Brainstorm EXP-040
**Created**: 2026-06-09
**Goal**: goals/improve-cifar10-test-accuracy.md

<!-- Ideation only. Metric/direction/constraints/verification live in the goal file;
     baseline (96.22, commit 6c417a4) lives in experiment-indices/improve-cifar10-test-accuracy.tsv. -->

## Web Search & Literature Review

No new external search. Grounding from the project knowledge base (`knowledge/README.md`) + the in-scope
file (`train.py`) + standard PyTorch performance practice:
- **`torch.backends.cudnn.benchmark = True`** (PyTorch perf docs): for FIXED input shapes (here always
  128×3×32×32, `drop_last=True`), cuDNN benchmarks its convolution algorithms once and caches the fastest
  per shape. Standard, safe, ~zero-risk throughput lever; warmup cost is a few steps. Commonly paired with
  `torch.compile`.
- **`torch.compile(mode="max-autotune")`** (PyTorch docs): Triton+cuDNN kernel autotuning on top of
  CUDA graphs — can lower steady-state dt below `reduce-overhead`, but the autotuning runs lazily on the
  first compiled call, which here is INSIDE the timed loop (train.py L233) → its (tens-of-seconds) cost
  is charged to the 300s budget.
- Project precedent: **torch.compile reduce-overhead (EXP-007)** was accepted as a pure throughput
  enabler — dt 10–11ms→8ms, 77→89 ep, null standalone accuracy effect but it BOUGHT epochs. A further
  throughput cut is the same legitimate mechanism.

## Experimental History Review

**Current best / baseline**: 96.22% (EXP-012, 6c417a4); bar 96.32 (+0.1). 40 experiments; the plateau is
extremely well-confirmed.

**The plateau is bounded by THREE established walls (project-insights):**
1. **Compute/epoch wall** — ANY non-trivial FLOP add OR wall-clock-heavier change (down to the sub-ms-op
   scale, EXP-039's two `F.normalize`) → fewer epochs → under-train → regress
   (EXP-004/009/015/024/036/038/039).
2. **Polish-vs-top1** — compute-neutral OPTIMIZATION polish (EMA/SWA/GC/LS/bag-of-tricks) lowers loss/
   flatness not top-1.
3. **Regularizer-adds underfit at 300s** — dropout, CutMix, Mixup all underfit the convergence-bound
   recipe.

**CLOSED axes (~32)**: capacity BOTH ways (widening EXP-004/009; FLOP-neutral reallocation EXP-038);
ALL augmentation (strength/policy/mixing/cooldown/border, through EXP-037); ENTIRE LR schedule
(peak/floor/shape, EXP-016/017/029); regularizer-adds; architecture (SiLU/preact/ResNet-D/BlurPool/
multi-scale-head/SE); optimizer dynamics (GC EXP-030/031) + objective (SAM EXP-036); weight-averaging
(EXP-006/019/020); classifier head BOTH sub-levers (feature-aggregation EXP-032, scoring-geometry/cosine
EXP-039); input std-norm INFEASIBLE; large-batch (EXP-025); bag-of-tricks (EXP-026).

**The single most important UNRESOLVED question** (flagged in exp-report-039 Next Steps, confidence
medium): is the net **convergence-bound** (still improving at the 91-epoch / 300s operating point → more
epochs would lift top-1) or **epoch-saturated** (flat past ~85 ep → more epochs won't help)? Strong
indirect evidence for convergence-bound: every recent regression (SAM 76 ep, fat-head 73 ep, cosine 83 ep)
under-trained relative to the ~91-ep baseline, and heavy TrivialAugment recipes on CIFAR WRNs typically
need 200+ epochs to fully converge in the literature. EXP-007's "epoch-saturated past ~77" was on the
PRE-TrivialAugment recipe — the saturation point almost certainly shifted higher once strong aug was
added (EXP-012). **No experiment has tried to REDUCE dt to ADD epochs** — all prior throughput work
(EXP-007) was about ABSORBING added cost, never about buying NET-new epochs. This is the untried gap.

**Untried gap**: per-step **throughput headroom** under the fixed dt-gated budget. The budget gates on
`Σ per-step GPU dt` (train.py L242); the model is LAUNCH-BOUND at batch 128 (~8ms/step ≈ 13% of H20 bf16
peak, EXP-007). Any dt reduction converts directly into more epochs (300/dt steps) at the same 300s.

## Candidate Ideas

First-principles on the bottleneck: the metric is bounded at ~96.2 at 91 ep / 300s. The budget gates on
GPU step dt only (dataloader/aug CPU time is pre-`t0`, unbudgeted). The net is launch-bound, so dt is
dominated by kernel-launch + execution overhead that `reduce-overhead` cudagraphs already partly removes.
The untested hypothesis is convergence-bound; the way to test/exploit it is to cut dt → add epochs.

### 1. cuDNN convolution-algorithm autotuning (`torch.backends.cudnn.benchmark = True`)
**Summary**: Add one line near setup (before model/compile): `torch.backends.cudnn.benchmark = True`.
Input shapes are fixed (128×3×32×32, `drop_last=True`), so cuDNN benchmarks each conv shape's algorithms
once and caches the fastest. Everything else in the recipe is unchanged. If steady-state dt drops, the
time-fraction LR schedule (driven by `total_training_time/TIME_BUDGET_S`) still anneals fully — the run
simply fits more epochs in 300s.
**Reasoning**: The net is launch/execution-bound (13% of peak); the ~20 conv layers have a handful of
fixed shapes ideal for cuDNN benchmark-mode algorithm selection. This is the canonical, lowest-risk way
to shave conv dt and is routinely paired with `torch.compile`. If it cuts dt even ~5–10%, that is ~4–9
more epochs; if the net is convergence-bound, that lifts best_test_acc toward/over the bar. Crucially the
experiment is INFORMATIVE in every outcome: (a) dt↓ and acc↑ ⇒ convergence-bound CONFIRMED + improvement
(reframes the whole remaining search toward throughput); (b) dt↓ but acc flat ⇒ epoch-saturated (closes
the throughput→epochs direction); (c) dt unchanged ⇒ compile already conv-optimal (closes the
cuDNN-autotune sub-lever). Legitimate throughput lever, exact same mechanism as the accepted EXP-007
torch.compile.
**Sources**: PyTorch performance docs (cudnn.benchmark for fixed shapes); train.py L181-190 (compile
setup), L213-265 (dt-gated loop); project-insights EXP-007 (launch-bound, throughput buys epochs);
exp-report-039 Next Steps (throughput→epochs is the under-explored axis).
**Estimated Effort**: low — one line + one run.
**Risk Assessment**: LOW. Cannot crash (standard flag); warmup cost is a few steps (negligible vs 300s).
Worst case is a dt-neutral null (compile already chose optimal conv algos) — clean, no confound. Does not
touch model, params, seed, eval, or data.

### 2. `torch.compile(mode="max-autotune")` — aggressive kernel autotuning
**Summary**: Change L190 `mode="reduce-overhead"` → `mode="max-autotune"` (Triton + cuDNN autotuning on
top of CUDA graphs). Higher ceiling on dt reduction than #1.
**Reasoning**: max-autotune benchmarks multiple kernel implementations and can beat reduce-overhead's
steady-state dt, plausibly a larger epoch gain than #1.
**Sources**: PyTorch torch.compile docs; train.py L190.
**Risk Assessment**: MEDIUM-HIGH. The first compiled call is INSIDE the timed loop (L233), so
max-autotune's autotuning (tens of seconds for ~10 unique conv/matmul shapes) is charged to the 300s
budget — it could eat 30–120s, NET-REDUCING epochs unless steady-state dt drops enough to repay the
compile tax. Mitigating it cleanly (warm up the compile OUTSIDE the timer) requires either junk
optimizer steps (would be unbudgeted real training → budget-gaming) or fragile fwd/bwd-without-step
warmup. High risk of a net-negative or `invalid`-looking run. Lower EV than #1 for a first attempt;
revisit only if #1 confirms convergence-bound and the compile tax can be measured/contained.

### 3. Offload Cutout to the CPU dataloader (remove it from the timed GPU path)
**Summary**: Move `cutout_batch` (train.py L223, currently a ~6-eager-kernel GPU op inside the timed
loop) into the `train_tf` transform as a vectorized tensor-slice Cutout (no per-sample `.item()` sync,
to avoid the EXP-002 throttle), running in parallel dataloader workers. The GPU step loses the cutout
launches → dt drops → more epochs.
**Reasoning**: Reliably removes real GPU work from the budgeted path (unlike #1/#2 whose dt gain is
uncertain), guaranteeing more epochs — a clean convergence-bound test.
**Sources**: train.py L44-57 (cutout), L156-167 (train_tf), L223; project-insights EXP-002 (per-sample
CPU `.item()` throttles the dataloader — must vectorize to avoid).
**Risk Assessment**: MEDIUM. (a) Optics: it REVERSES the deliberate EXP-002/003 decision to put Cutout
on the GPU, and it exploits that dataloader time is unbudgeted — the analyze phase could read it as
gaming the budget measurement → `invalid`, even though it is genuine identical augmentation + real extra
epochs. (b) Wall-clock: if the dataloader (already running TrivialAugmentWide on CPU) can't keep up, the
GPU starves (pre-`t0`, unbudgeted) → epoch wall-clock rises → risks the <600s total-wall constraint.
Lower EV than #1 given the `invalid` risk under the "avoid reward-hacking / survive benchmark-composition
change" guidance.

## Idea Evaluation

All three target the SAME well-justified, untested hypothesis — the net is convergence-bound, so cutting
per-step dt to buy net-new epochs should lift top-1 — via three different mechanisms with real trade-offs
on risk vs. dt-cut ceiling.

- **#1 (cuDNN benchmark)** has the best risk/reward: unambiguously legitimate (identical to the accepted
  EXP-007 throughput mechanism), cannot crash or be ruled invalid, one-line change, and INFORMATIVE in
  all three outcomes — it either delivers an improvement, confirms epoch-saturation, or closes the
  cuDNN sub-lever. The downside is a possible dt-neutral null if compile already picked optimal conv
  algos, but even that is a clean, useful finding.
- **#2 (max-autotune)** has the highest dt-cut ceiling but the compile-tax-in-budget risk makes it likely
  net-negative or invalid-looking on a first attempt; the clean warmup workarounds border on budget-
  gaming. Deprioritized until #1 establishes whether epochs even matter.
- **#3 (CPU cutout)** is the most RELIABLE dt reducer but carries a real `invalid`/reward-hacking optics
  risk (reverses the deliberate GPU-cutout choice, exploits the unbudgeted dataloader) plus a wall-clock
  risk. Deprioritized for the same "first, cleanly test the hypothesis" reason.

**#1 wins**: lowest risk, legitimate-by-precedent, and the most informative clean test of the single
biggest open question (convergence-bound vs epoch-saturated). It is the right first probe of the only
untried axis after 40 experiments.

## Chosen Idea
**Selected**: cuDNN convolution-algorithm autotuning — set `torch.backends.cudnn.benchmark = True` at
setup so cuDNN caches the fastest conv algorithm for the fixed 128×3×32×32 shapes, cutting per-step dt
and fitting more epochs into the fixed 300s budget; everything else unchanged.

**Why this idea**:
After 40 experiments, capacity, augmentation, LR schedule, regularizers, architecture, optimizer
dynamics/objective, weight-averaging, and both classifier-head sub-levers are closed. The one untried
axis is per-step THROUGHPUT under the dt-gated budget — no experiment has tried to REDUCE dt to ADD
epochs (EXP-007 only absorbed added cost). cuDNN benchmark-mode is the canonical, lowest-risk dt lever
for a launch-bound net with fixed input shapes, is legitimate by the EXP-007 precedent, and cleanly tests
the pivotal open question of whether the net is convergence-bound.

**Hypothesis**:
Enabling `cudnn.benchmark` lets cuDNN select faster convolution algorithms for the fixed shapes, lowering
steady-state dt below ~8ms and fitting more than ~91 epochs into the 300s budget; if the net is
convergence-bound at this operating point, the extra epochs lift best_test_acc above the bar 96.32.
Honest most-likely outcome: a small dt change (compile may already be near-conv-optimal) → either a
modest gain if convergence-bound, or a within-noise null. A dt-neutral result closes the cuDNN-autotune
sub-lever; a dt-drop-but-acc-flat result is the high-value finding that the net is epoch-saturated and
the throughput→epochs direction should be abandoned.
