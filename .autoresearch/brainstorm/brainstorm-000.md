# Brainstorm EXP-000
**Created**: 2026-06-08
**Goal**: goals/improve-cifar10-test-accuracy.md

## Web Search & Literature Review
Grounded in well-established fast-CIFAR-10-training literature (no live search needed — this is
heavily-trodden ground; citations are canonical):

- **Smith & Topin, "Super-Convergence: Very Fast Training of NNs Using Large Learning Rates" (2018)** (arXiv:1708.07120)
  One-cycle LR (warmup to a high peak, then anneal to ~0) trains CIFAR nets to high accuracy in
  far fewer iterations than step schedules. Strong fit for a fixed-budget regime where the schedule
  must fully anneal within the available steps.
- **Loshchilov & Hutter, "SGDR: Cosine Annealing" (2017)** (arXiv:1608.03983)
  Smooth cosine decay to near-zero reliably beats coarse step drops at equal budget; the final
  low-LR phase is where most test-accuracy gain is consolidated.
- **David Page, "How to Train Your ResNet" series (myrtle.ai, 2018) / DAWNBench CIFAR-10 entries**
  Mixed precision, a one-cycle schedule, and light regularization (label smoothing, Cutout) are
  the standard levers that take a CIFAR ResNet to ~94% quickly. Throughput (steps/sec) is the
  binding resource under a wall-clock budget.
- **Szegedy et al., label smoothing (2016) / He et al. "Bag of Tricks" (2019)** (arXiv:1812.01187)
  Label smoothing, Nesterov momentum, and cosine LR are low-risk, consistently-positive additions
  for ResNet image classification.

## Experimental History Review
First experiment under this goal — no prior history.

**Baseline diagnostics (from the baseline run that established 91.73%):**
- Budget allowed only **34,861 steps / 90 epochs** in 300s (MAX_STEPS=64000 never reached).
- The LR schedule `MultiStepLR(milestones=[32000, 48000], gamma=0.1)` is **mistuned for the budget**:
  LR stays at 0.1 for ~92% of training, drops to 0.01 only at step 32000 (~the last 8%), and the
  second drop at 48000 **never fires**. The schedule essentially never anneals — a large, near-certain
  source of lost accuracy.
- Training is **fp32** with peak VRAM only 330 MB of 98 GB — enormous headroom; throughput is left
  on the table (no AMP, no channels_last).
- Two clear, independent levers: (A) a schedule that actually anneals within the real step budget,
  and (B) higher throughput → more epochs in the same 300s.

## Candidate Ideas

### 1. Budget-matched annealing schedule + light regularization (schedule-only)
**Summary**: Keep the model, optimizer family, batch size, and fp32 precision unchanged. Replace
`MultiStepLR` with a schedule driven by the *elapsed-time fraction* `total_training_time / TIME_BUDGET_S`
(robust to throughput changes): a short warmup then cosine/one-cycle anneal to ~0 by the end of the
budget. Add Nesterov momentum and label smoothing (≈0.1). No throughput changes.

**Reasoning**: Directly fixes the single clearest defect — the schedule never anneals. Cosine/one-cycle
annealing to near-zero is the highest-confidence win in the fast-CIFAR literature. Time-fraction driving
means the schedule completes regardless of how many steps fit.

**Sources**: SGDR (1608.03983), Super-Convergence (1708.07120), Bag of Tricks (1812.01187); baseline diagnostics above.

**Estimated Effort**: low (edit the scheduler block + loss + optimizer flags in `train.py`).

**Risk Assessment**: Low. Worst case is a mild miss if the peak LR is too aggressive; falls back to
no-improvement, never a crash. Time-driven LR is a small, well-contained change.

### 2. Throughput-first modernization: bf16 AMP + channels_last + budget-matched schedule
**Summary**: Idea 1's budget-matched annealing schedule, PLUS convert training to **bf16 autocast**
(`torch.autocast`, no GradScaler needed for bf16 on H20) and **channels_last** memory format to raise
steps/sec. More steps in 300s → more epochs → more annealing cycles of useful learning. Add Nesterov +
label smoothing. Batch size kept at 128 to isolate the precision/layout effect (VRAM headroom is huge,
so a later bump is trivial).

**Reasoning**: Under a fixed wall-clock budget, throughput is the binding resource. bf16 on Hopper-class
H20 is numerically safe (wide exponent, no loss scaling) and channels_last suits conv nets. This couples
the near-certain schedule win (Idea 1) with a throughput multiplier, compounding both levers. fp32 + 330 MB
VRAM means the baseline is leaving the most obvious efficiency gains untouched.

**Sources**: David Page / DAWNBench (mixed precision is standard); PyTorch AMP + channels_last docs;
SGDR / Super-Convergence for the schedule; baseline diagnostics (fp32, 330 MB, 90 epochs).

**Estimated Effort**: medium (scheduler + autocast wrapping of forward/loss + `.to(memory_format=channels_last)`).

**Risk Assessment**: Low–medium. bf16 is robust (no scaler), but on a tiny model the kernel/dataloader
overhead may cap the realized speedup — downside is simply "schedule win only," still ≥ Idea 1. channels_last
on this small net is neutral-to-positive. No crash risk. (torch.compile deliberately excluded for now:
its first-step compile cost is counted inside the training loop and would eat the budget — defer.)

### 3. Architecture + augmentation modernization
**Summary**: Modernize the network (e.g., projection/conv shortcuts instead of channel-padding identity,
a stronger stem, possibly a slightly wider ResNet) and add stronger augmentation (Cutout, and/or mixup),
on top of AMP + a budget-matched schedule.

**Reasoning**: Higher ceiling — architecture + augmentation are how DAWNBench-class recipes reach ~94%.

**Sources**: Bag of Tricks (1812.01187), Cutout (DeVries & Taylor 1708.04552), David Page series.

**Estimated Effort**: high (multiple coupled changes; harder attribution).

**Risk Assessment**: Medium–high. Many simultaneous changes muddy attribution; stronger augmentation can
*hurt* at a short 300s/90-epoch budget (under-fitting), and architecture changes risk slower steps or
instability. Best deferred until the training-recipe foundation (schedule + precision) is locked in.

## Idea Evaluation
All three respect the hard constraints (only `train.py`, no new deps, single GPU, fixed budget, eval
once/epoch, no seed hacking).

- **Evidence strength**: Ideas 1 and 2 rest on the strongest, most directly-applicable evidence — the
  baseline's own schedule is provably never annealing, and cosine/one-cycle annealing is the most
  reliable CIFAR lever. Idea 3's gains are real but more setup- and budget-sensitive.
- **Mechanism clarity**: Crisp for 1 and 2 — (a) annealing LR to ~0 consolidates test accuracy in the
  final phase; (b) bf16/channels_last raise steps/sec → more epochs in 300s. Idea 3's mechanism (capacity/
  regularization) is real but its sign at a 90-epoch budget is uncertain (augmentation can under-fit).
- **Expected impact**: Idea 2 ≥ Idea 1, because it stacks the throughput multiplier on top of the
  near-certain schedule fix, targeting *both* binding levers at once. Idea 3 has the highest ceiling but
  the widest variance.
- **Risk profile**: 1 and 2 fail gracefully (no-improvement); 3 has crash/under-fit risk.
- **Feasibility**: 1 and 2 are small, contained edits; 3 is multi-change.

Idea 2 dominates: it contains Idea 1's near-certain win and adds a low-risk throughput multiplier with
clear mechanism and huge VRAM headroom. Idea 3 is the natural *next* loop once the recipe foundation is set.

## Chosen Idea
**Selected**: Idea 2 — Throughput-first modernization (bf16 AMP + channels_last + budget-matched annealing schedule + Nesterov + label smoothing)

**Why this idea**:
It attacks both binding constraints of the fixed-budget regime simultaneously. (1) The baseline's
step-space `MultiStepLR` never anneals within the ~35k steps that fit in 300s — replacing it with a
time-fraction-driven cosine/one-cycle anneal to ~0 is the single highest-confidence accuracy lever.
(2) The baseline trains in fp32 using only 330 MB of 98 GB — bf16 autocast (safe on H20, no GradScaler)
plus channels_last raises steps/sec, buying more epochs in the same wall-clock. Nesterov + label smoothing
are low-risk, consistently-positive additions. Every change is well-evidenced, contained to `train.py`,
and fails gracefully to no-improvement rather than crashing.

**Hypothesis**:
Replacing the never-annealing step schedule with a budget-matched cosine/one-cycle anneal, training in
bf16 with channels_last, and adding Nesterov + label smoothing will raise `best_test_acc` by at least
0.1 pp over the 91.73% baseline (expected to clear the bar comfortably — likely into the ~93% range),
while completing cleanly within the 300s budget on a single GPU. The schedule fix is expected to provide
most of the gain; the precision/layout change provides additional headroom via more epochs.
