# Brainstorm EXP-008
**Created**: 2026-06-08
**Goal**: goals/improve-cifar10-test-accuracy.md

## Web Search & Literature Review

- **Hu et al. 2018, "Squeeze-and-Excitation Networks" (CVPR)**: per-block channel recalibration — global-average-
  pool → FC(C→C/r) → ReLU → FC(C/r→C) → sigmoid → channel-wise scale of the block output. ~+0.3–0.7pp on CIFAR
  ResNets/WRNs at tiny param/FLOP cost; a different axis than width (accuracy *per channel*, not more channels).
- **PyTorch `torch.compile`/TorchInductor + CUDA graphs (official docs)**: fuses pointwise ops and amortizes
  kernel launches — established in EXP-007 to give ~30% throughput on this launch-bound k=4 net. Relevant here
  because SE adds several small per-block kernels (GAP, 2 FCs, sigmoid, broadcast-mul) whose launch cost is
  exactly what compile/CUDA-graphs absorb.

## Experimental History Review

Current best **96.00%** (EXP-003, commit f59de56). 9 experiments; baseline unchanged since EXP-003.

**FOUR axes now saturated/exhausted** (goal-learnings + reports):
- Width — k=4 sweet spot; k=6 compute-bound → underfit 95.26 (EXP-004).
- Regularization — Cutout + label smoothing + WD near-saturated (EXP-005).
- Weight-averaging — EMA redundant with cosine-to-0 (EXP-006).
- Training-length/throughput — torch.compile bought 77→89 epochs but acc 95.92; epochs saturated past ~77 (EXP-007).

**Key enabling result (EXP-007)**: `torch.compile(mode="reduce-overhead")` gives ~30% throughput on this net
(default mode net-negative). goal-learnings § Patterns + project-insights: "spend the freed throughput on a
DIFFERENT axis (capacity or architecture), not more epochs." Crucially, **compiled-k4 alone = 95.92 ≈ baseline**
(EXP-007) — so compile has no standalone accuracy effect, making it a clean *enabler*: any gain from
compiled-k4+<change> over ~96.0 is attributable to <change>, not to compile.

**Untried axis**: accuracy-*per-epoch* architecture (channel attention / better blocks) at fixed width — never
tested. SE was floated in brainstorm-006 but deprioritized ONLY for launch-bound throughput risk; EXP-007
removed that blocker.

**Key regime facts**: VRAM free (450–510 MB / 98 GB); 300s wall-clock binds; noise floor ~0.2pp (target ≥0.3pp).

## Candidate Ideas

### 1. Compiled k=4 + Squeeze-Excitation blocks
**Summary**: Add an SE module to each `BasicBlock` (GAP → FC(C→C/r) → ReLU → FC(C/r→C) → sigmoid → scale the
block's conv2 output before the residual add; reduction r=16), and wrap the training model in
`torch.compile(mode="reduce-overhead")` so SE's extra small kernels don't cost epochs. Width stays k=4 (the
proven sweet spot); all recipe knobs fixed. Compile is the *enabler* (keeps the well-trained ~80-epoch regime);
SE is the accuracy intervention.

**Reasoning**: This is the first test of an **accuracy-per-epoch architectural** lever — orthogonal to the four
saturated axes. SE is well-evidenced on CIFAR ResNets (~+0.3–0.7pp) at ~5% params / negligible FLOPs, and VRAM
is free. The one prior reason to avoid it (its many small kernels cutting epochs in this launch-bound regime —
the EXP-002/EXP-004 trap) is now mitigated by compile (EXP-007: ~30% throughput, CUDA graphs absorb launch
overhead). Attribution is clean: compiled-k4 alone is 95.92 (EXP-007), so beating ~96.1 isolates SE's effect.

**Sources**: Hu et al. 2018 (SENet); EXP-007 report + goal-learnings (compile recipe + "spend throughput on a
different axis"); brainstorm-006 § Candidate 2 (SE deprioritized for throughput — now unblocked).

**Estimated Effort**: medium (new SE module + integrate into all 18 blocks; verify shapes; smoke-test compiled-SE
throughput in planning to confirm epochs stay high).

**Risk Assessment**: (a) Even with compile, SE may not fully recover throughput → fewer epochs → benefit masked
(mitigated: smoke-test in planning; abort/record if epochs collapse). (b) SE may add little at this budget if the
model is at a data/architecture ceiling → graceful no-improvement. (c) Two nominal changes (compile+SE), but
compile's null accuracy effect (EXP-007) keeps attribution clean. Worst case: graceful no-improvement.

### 2. Compiled k=5 (re-open the capacity axis)
**Summary**: Widen to k=5 ({80,160,320}, ~6.7M params) and compile, spending compile's throughput on the
biggest historical lever (capacity).

**Reasoning**: Capacity was the dominant lever (+2.84pp at k=4) and we have NEVER tested >k4 with adequate
epochs — k=6 failed only because it was epoch-starved (35 ep). Compile partially buys those epochs back.

**Sources**: EXP-001 (width = dominant lever), EXP-004 (k=6 epoch-starved), EXP-007 (compile throughput).

**Estimated Effort**: low (one constant + compile).

**Risk Assessment**: Width scales compute ~k², so compiled k=5 likely fits only ~55–65 epochs (vs k=4's ~89) —
still in the *rising* part of the epoch curve, so the capacity benefit may be cancelled by epoch starvation (a
milder k=6 redux). A no-improvement here is **ambiguous** (no-capacity-benefit vs under-training), which is why
it's the fallback, not the lead.

### 3. Compiled k=6 (aggressive capacity)
**Summary**: Retry k=6 ({96,192,384}, 9.7M) with compile to buy back some of the epochs EXP-004 lacked.

**Reasoning**: Directly revisits the EXP-004 failure with the new throughput tool.

**Sources**: EXP-004 report; EXP-007 (compile).

**Risk Assessment**: Compile helps *least* on compute-bound models (launch overhead is already a small fraction),
so compiled k=6 likely still only ~40 epochs → high chance of repeating the EXP-004 underfit. Lowest EV of the three.

## Idea Evaluation

All respect hard constraints (train.py-only, no new deps — SE uses `nn` layers, compile is core torch; single
H20/300s; eval once/epoch; seed 42; no reward hacking — changes train-time architecture, not the frozen eval).

- **Fit to the open problem**: four axes are saturated; the genuinely untried one is *accuracy-per-epoch
  architecture*. Idea 1 (SE) is the only candidate on that axis. Ideas 2–3 retry the *capacity* axis, which is
  epoch-bound under this budget even with compile (width eats the compile savings via k² compute).
- **Epoch regime**: idea 1 keeps the well-trained ~80-epoch regime (compile offsets SE's small cost) → clean
  test. Ideas 2–3 drop to ~55/~40 epochs → epoch-starvation confound makes a null result uninterpretable.
- **Evidence**: SE has direct CIFAR evidence and a clear mechanism (learned channel gating). Capacity has strong
  in-project evidence (+2.84pp) BUT only when adequately trained — exactly what k=5/k6 can't guarantee here.
- **Attribution**: EXP-007's compiled-k4=95.92 gives idea 1 a clean reference to isolate SE. Idea 2/3 conflate
  capacity with under-training.
- **Risk profile**: idea 1 fails gracefully (no-improvement) and is smoke-testable for throughput beforehand;
  idea 2/3 risk an ambiguous underfit.

Idea 1 (compiled k=4 + SE) is the highest risk-adjusted EV: it opens the one untried axis, keeps a well-trained
regime via the now-validated compile enabler, has clean attribution, and is throughput-verifiable in planning.
Capacity (idea 2) is the natural next fallback if SE lands in the noise.

## Chosen Idea
**Selected**: Compiled k=4 + Squeeze-Excitation blocks

**Why this idea**: It targets the only unexhausted lever — accuracy-per-epoch architecture — using SE, which is
well-evidenced on CIFAR and was previously blocked *only* by the launch-bound throughput cost that EXP-007's
compile result now removes. Width/regularization/weight-averaging/training-length are all saturated; pure
capacity (k=5/k6) is epoch-bound under the 300s budget. Compile keeps the proven ~80-epoch regime so SE gets a
fair test, and compiled-k4's null accuracy effect (95.92, EXP-007) makes the attribution clean. Failure is
graceful.

**Hypothesis**: Adding SE channel recalibration (r=16) to the k=4 WideResNet, with `torch.compile(reduce-
overhead)` keeping the epoch count ≳75, will lift `best_test_acc` past the +0.1pp bar (≥96.10%, expected
~96.2–96.6%) at roughly unchanged epochs and a small (~5%) param increase, because adaptive channel gating adds
representational efficiency the four saturated axes could not.
