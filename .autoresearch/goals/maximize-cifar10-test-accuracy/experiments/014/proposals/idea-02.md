# Idea-02: Mild layer2 widen 256→320 (the pre-registered milder retry of EXP-007)

## Summary

Widen the proven 8×8 layer2 stage from 256 to 320 channels — the explicit
pre-registered next step after EXP-007's 256→384 widen under-annealed. The bet
is that 320 adds enough capacity to matter (capacity at 8×8 is proven useful by
EXP-004) while costing far fewer epochs than 384, landing on the profitable side
of the capacity-vs-epochs tradeoff. This is a clean single-variable capacity
probe: the GatedResidual ReZero α=0 identity init means no LR retune is needed
(same property that made EXP-004 a clean test). Standalone; not paired with a
throughput lever (kept separate as idea-01).

## Mechanism / Reasoning

The named limiter (from `03-experiment-learnings.md`, Medium-Importance "Adding
net capacity that costs too many epochs UNDER-ANNEALS") is the
capacity-vs-epoch-count tradeoff at the fixed 300s TRAINING budget. Accuracy
concentrates in the low-LR anneal tail (EXP-001), so removing epochs truncates
the tail and can leave added capacity unrealized. EXP-007 (256→384) removed too
many epochs (150→94) and finished still-climbing (best==final), losing −0.15pp —
a verdict on the *magnitude*, explicitly NOT on width-at-8×8 as a mechanism.

Why 320 could win where 384 lost: the dominant cost terms are the two 8×8
GatedResidual convs, whose FLOPs/params scale as channel². At 256→320 that block
grows to 320²/256² = **1.5625×** of baseline; at 256→384 it grew to
384²/256² = **2.25×**. So 320 is roughly the "1.25× extra cost" step the
learnings call for (≈half of 384's marginal cost), keeping most of the epochs
while still adding real capacity. Capacity at the 8×8 stage is the *proven*
full-throughput place to add it: EXP-004's ReZero Residual(256) at layer2 gave
+0.13pp, and EXP-005 showed the 4×4 stage is both kernel-slow (~10%) and unused —
so 8×8/width is the right axis.

## Concrete implementation sketch (train.py-specific)

In `ResNet9.__init__` (train.py lines 149–151), change three channel counts so
the pool chain and 16→8→4 spatial geometry are untouched:

- Line 150: `self.layer2 = nn.Sequential(conv_bn(128, 320), nn.MaxPool2d(2), GatedResidual(320))`
- Line 151: `self.layer3 = nn.Sequential(conv_bn(320, 512), nn.MaxPool2d(2), Residual(512))`

That is the *entire* diff (two literals: 256→320 in layer2, and the layer3 stem
input 256→320). `GatedResidual(320)` keeps `self.alpha = nn.Parameter(zeros(1))`
(line 134) → exact identity at init, live gradient via ReLU-safe ReZero (lines
119–137), no LR retune. `fc` stays `nn.Linear(512, …)` (layer3 output unchanged).
Whitening front-end, EMA (line 255), TTA, optimizer (line 244), and the
time-keyed one-cycle (lines 286–290) are all untouched. VRAM is a non-issue
(1.6 GB of 98 GB, project-insights High).

## Expected effect (quantified, with epoch-cost prediction)

Parameter delta (3×3 conv = c_in·c_out·9, bias=False; BN = 2·c_out):

| stage | 256 (base) | 320 | 384 (EXP-007) |
|---|---|---|---|
| layer2[0] conv_bn | 295,424 | 369,280 | 443,136 |
| GatedResidual | 1,180,673 | 1,844,481 | 2,658,433 |
| layer3 stem conv_bn | 1,180,672 | 1,475,584 | 1,770,496 |
| **Δ vs base** | — | **+1.03M** | **+2.21M** |

So 320 adds ~1.03M params — about **47%** of 384's +2.21M (the learnings'
"+2.2M" figure reproduces exactly). FLOPs track params here (same spatial
sizes), so 320's added compute is ~half of 384's.

Epoch-cost prediction: EXP-007's 384 cut 150→94 (−56 epochs, −37%). Naively
halving the marginal cost predicts ~150−28 ≈ **122 epochs**, consistent with the
learnings' pre-registered **~120–135** target. BE CONSERVATIVE: project-insights
(EXP-005) warns FLOP-equal ≠ wall-clock-equal — cuDNN kernel efficiency varies
with shape, and EXP-007's 94 also absorbed shared-host contention, so the true
landing could be lower. The under-anneal cliff is **~110 epochs** (EXP-004 band
142–150; ≤110 = cliff). **If 320 lands below ~120, it likely under-anneals like
384 and ties-or-loses.**

Honest expected value: the gain must exceed the accuracy lost to ~15–30 removed
epochs. EXP-004 added capacity (+0.13pp) while *also* dropping epochs (174→142)
and still won, so a similar mild-capacity/mild-epoch-loss step can net positive —
but the margin is thin and sits near the ~0.1pp noise floor. This is plausibly
profitable, not clearly so; treat as medium-confidence.

## Risks

1. **Under-anneal (dominant).** If 320 costs >30 epochs (lands <120), the tail is
   truncated and the verdict mirrors EXP-007. Pre-registered num_epochs is the
   first-class read.
2. **Capacity genuinely near-saturated.** EXP-012's mechanism probe found the
   ReZero gate "not accuracy-limiting"; if true, even a well-annealed 320 only
   ties. Then the win must come from the extra layer2[0]/layer3-stem width, not
   the gate.
3. **cuDNN width-320 efficiency.** 320 = 64×5 — a multiple of 64, generally
   efficient for cuDNN tensor-core kernels (not a pathological non-power-of-2).
   Low risk, but verify throughput empirically; if 320 is unexpectedly slow,
   288 (=32×9, still 1.27× block cost) is a cheaper fallback.
4. **Noise floor.** ~0.1pp run-to-run jitter from epoch-count variance — a thin
   single-run win is unproven without the same-session control.

## Verification approach

- **Mandatory same-session no-widen (256) baseline cell** in the same run/host
  conditions — the stored 96.38 is too weak at the ~0.1pp noise floor
  (project-insights, EXP-012/013 both required this).
- **Pre-register num_epochs** as a first-class diagnostic. Read the
  best-vs-final and the trajectory shape: if best==final (monotone rise to the
  end), that is the EXP-007 under-anneal signature → capacity axis then strongly
  exhausted at this budget. If peaked-then-dipped (fully annealed) and still
  ≤ baseline → 8×8 width genuinely saturated.
- **Win = clear ≥+0.1pp over the stored 96.38 baseline AND beat the same-session
  256 cell.** Confirm training_seconds≈300 and wall <600s (≤1 eval/epoch).

## Optional throughput pairing (out of scope here)

Pairing the widen with a throughput lever (e.g. torch.compile, idea-01) could buy
back removed epochs and let MORE capacity anneal, potentially unlocking the full
384 step. Noted as a follow-up; THIS proposal stays a clean single-variable
mild-widen probe so the verdict on width-at-320 is unconfounded.

## Effort

Low — two integer literal edits in `ResNet9.__init__`; no recipe/LR/optimizer
changes. One training run plus the mandatory same-session 256 control cell.

## Sources

- EXP-007 (256→384 under-anneal, +2.2M, 150→94, −0.15pp, pre-registers 256→320):
  `.../experiments/007/04-analysis.md`; learnings §Failed/Medium and
  §Failed/Low (EXP-007 bullet).
- EXP-004 (ReZero Residual(256) at layer2, +0.13pp, capacity-at-8×8 proven):
  `.../experiments/004/04-analysis.md`; learnings §Patterns/High.
- EXP-005 (4×4 deepen failed; FLOP-equal ≠ wall-clock-equal, ~10% kernel
  penalty): `.../experiments/005/04-analysis.md`; project-insights Medium.
- EXP-012 (ReZero gate "not accuracy-limiting" probe):
  `.../experiments/012/04-analysis.md`.
- Capacity-vs-epoch tradeoff + score-by-gain-minus-removed-epochs:
  `.../project-notes/project-insights.md` §High.
- Noise floor / same-session baseline requirement: learnings §Protocol/High.
