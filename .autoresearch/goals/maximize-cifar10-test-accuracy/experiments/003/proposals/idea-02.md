# Idea-02: Alternating-flip augmentation (airbench) replacing random horizontal flip

## Summary

Replace the per-sample i.i.d. `RandomHorizontalFlip()` in the train transform with
airbench's **alternating flip**: a deterministic per-image scheme keyed on
`(epoch XOR per-image-random-bit)` parity, so every training image is shown
flipped on exactly every other epoch and unflipped on the rest, rather than a
random coin per visit. This guarantees balanced orientation coverage per image
and removes the run-to-run orientation imbalance that a Bernoulli flip leaves.

This is the cheapest documented gain in the fast-CIFAR lineage. **But our run does
~183 epochs, while airbench's measured gains are at 10–40 epochs, so I am honestly
skeptical the marginal gain clears the +0.1pp bar here.** The proposal is written
to be correct and cheap if pursued, with an explicit expected-magnitude argument
that leans pessimistic.

## Limiter targeted

From the diagnosis chain, the binding limiter at 95.72% is *not* throughput
(30k img/s, VRAM 1.6/98 GB) and not the LR-tail denoising (solved by EXP-002
EMA+TTA). The remaining levers are **representation/regularization quality of the
training signal**. Alternating flip targets the *augmentation noise / orientation
coverage* sub-lever: with i.i.d. flipping, over a finite number of epochs each
image's flipped-vs-unflipped exposure is a binomial draw, so some images are
over-shown in one orientation. Alternating flip makes that exposure exactly
balanced (50/50) for every image, reducing the variance/bias of the augmented
empirical distribution the SGD trajectory integrates over. The causal chain to the
metric: balanced per-image orientation exposure → lower-variance augmented gradient
signal / less orientation bias in the learned features → marginally better
generalization at eval. This is a *training-signal-quality* edit, orthogonal to the
EXP-002 eval-side EMA+TTA win.

## Evidence

- **airbench paper (arXiv:2404.00498), alternating flip.** Definition (quoted from
  §on alternating flip): "For the first epoch, we randomly flip 50% of inputs as
  usual. Then on epochs {2,4,6,...}, we flip only those inputs which were not
  flipped in the first epoch, and on epochs {3,5,7,...}, we flip only those inputs
  which were flipped in the first epoch." Implementation form:
  `flip_mask = ((hashed_indices + epoch) % 2 == 0)`. Reported effect: "consistently
  boosts performance by the equivalent of a 0–25% training speedup" across
  airbench94/airbench96; Table 1 (20 ep, no cutout, with TTA): random flip
  94.557% → alternating 94.653% (**+0.097pp**); Table 2 speedup estimates 27.1%
  @20ep, 38.3% @40ep (without TTA). The author calls it the source of the final
  ~10% of airbench's speedup over prior work.
- **airbench legacy code** (`legacy/airbench96.py`) uses the coarser whole-dataset
  variant: comment "Flip all images together every other epoch. This increases
  diversity relative to random flipping" + `if self.epoch % 2 == 1: images =
  images.flip(-1)`. The paper's per-image hashed scheme is the principled version;
  I propose the per-image scheme (Option (b)), not the coarse whole-batch one,
  because the whole-batch variant interacts badly with BatchNorm (every batch in an
  even epoch is 100% one orientation → BN running stats see orientation-correlated
  batches; see Risk 4).
- **Our history.** EXP-002 analysis (`experiments/002/04-analysis.md`) shows the
  recipe is at 95.72% with EMA+TTA, 183 epochs/300s. The single most relevant data
  point against this idea is *the epoch count*: airbench's +0.097pp is measured at
  **20 epochs**; the magnitude shrinks as epochs grow because the binomial
  orientation imbalance that alternating flip removes is O(1/sqrt(n_epochs)) and is
  already small by ~180 epochs.

## Concrete change to `train.py`

Everything is local to `main()`'s data pipeline plus a tiny module-level helper.
`prepare.py` is untouched (frozen). No new dependencies (pure torch / torchvision).

### 1. Remove `RandomHorizontalFlip` from the CPU transform

In `main()`, the train transform (currently lines 139–147) becomes:

```python
train_tf = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        # RandomHorizontalFlip() removed — flip now handled deterministically
        # per-image by the alternating-flip dataset wrapper below.
        transforms.ToTensor(),
        transforms.Normalize(EVAL_MEAN, EVAL_STD),
        Cutout(8),
    ]
)
```

`RandomCrop` and `Cutout` are kept exactly as-is (per the idea brief). Note the
flip must remain *upstream of Cutout's spatial position*; since flip is a global
horizontal mirror and Cutout's patch center is drawn independently each call, doing
the flip inside the dataset wrapper (which runs the whole `train_tf` then flips) is
equivalent in distribution — but cleanest is to flip the PIL image *before*
`ToTensor`. See implementation choice below.

### 2. Add an alternating-flip dataset wrapper (module level)

The flip needs two pieces of state the default `shuffle=True` DataLoader does not
expose to a transform: the **sample index** and the **current epoch**. A thin
`Dataset` wrapper around `datasets.CIFAR10` provides both. The epoch counter is a
shared mutable cell the training loop bumps each epoch; workers read it through the
forked copy at iteration start.

```python
import torchvision.transforms.functional as TF  # already-available torchvision

class AlternatingFlipDataset(torch.utils.data.Dataset):
    """Wrap CIFAR10 with deterministic per-image alternating horizontal flip.

    Each image i has a fixed random bit b_i (drawn once from a fixed-seed
    generator). On epoch e it is flipped iff (b_i + e) is odd. So every image is
    shown flipped on exactly every other epoch — balanced 50/50 coverage — rather
    than an i.i.d. coin per visit (airbench alternating flip, arXiv:2404.00498).
    The base transform (RandomCrop/ToTensor/Normalize/Cutout) runs as usual; the
    horizontal mirror is applied to the PIL image up front.
    """

    def __init__(self, base, base_tf, epoch_cell, seed=42):
        self.base = base            # datasets.CIFAR10 with transform=None
        self.base_tf = base_tf      # the Compose WITHOUT RandomHorizontalFlip
        self.epoch_cell = epoch_cell  # list of length 1: epoch_cell[0] = epoch idx
        g = torch.Generator().manual_seed(seed)
        self.flip_bit = (torch.rand(len(base), generator=g) < 0.5)  # b_i

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, target = self.base[idx]            # PIL image, no transform yet
        e = self.epoch_cell[0]
        if (int(self.flip_bit[idx]) + e) % 2 == 1:
            img = TF.hflip(img)                 # mirror BEFORE the base transform
        return self.base_tf(img), target
```

Construction in `main()` (replacing lines 149–151):

```python
epoch_cell = [0]  # shared mutable epoch counter; updated each epoch below
base_set = datasets.CIFAR10(DATASET_DIR, train=True, download=True, transform=None)
train_set = AlternatingFlipDataset(base_set, train_tf, epoch_cell, seed=42)
```

The DataLoader construction (lines 152–161) is unchanged — same
`shuffle=True, num_workers=8, persistent_workers=True, prefetch_factor=4`.

### 3. Plumb the epoch counter to workers

The loop already maintains `epoch` (line 194, incremented at line 199). With
`persistent_workers=True`, workers are forked ONCE and keep their own copy of the
dataset object, so we cannot rely on mutating `epoch_cell[0]` in the parent being
seen by workers mid-life. Two correct options:

- **Option A (preferred, minimal): set the epoch BEFORE creating the iterator each
  loop and disable persistent workers.** Set `persistent_workers=False`. Then each
  `for ... in train_loader` re-forks workers, which read the current `epoch_cell[0]`
  at fork time. In `main()` set `epoch_cell[0] = epoch` immediately after
  `epoch += 1` (after line 199, before the inner `for`). Cost: worker re-fork per
  epoch. At 183 epochs and ~1.6s/epoch this adds DataLoader startup latency each
  epoch — **measure**; if it costs more than a couple epochs of budget, prefer B.

- **Option B (keeps persistent_workers, uses worker_init + atomic file/shared
  tensor): use a shared-memory tensor for the epoch.** Make `epoch_cell` a
  `torch.zeros(1, dtype=torch.long).share_memory_()` tensor instead of a Python
  list. Persistent workers hold a reference to the *same* shared tensor, so the
  parent writing `epoch_cell[0] = epoch` is visible to all workers immediately.
  Reading it in `__getitem__` via `int(self.epoch_cell[0].item())` is cheap. This
  preserves `persistent_workers=True` and `prefetch_factor=4`. **This is the right
  design** given EXP-001's learning that throughput buys epochs; it avoids per-epoch
  re-fork entirely.

  Caveat for B: prefetching means a worker may build the *next* epoch's batches
  using the *current* `epoch_cell` value before the parent bumps it (the bump
  happens after the inner-loop break). The parity is therefore "approximately"
  aligned — at most a fraction of one epoch's batches use the previous parity. This
  does NOT break correctness of the *alternating* property (each image still
  alternates every epoch); it only blurs the exact epoch boundary by up to
  `prefetch_factor` batches. Acceptable. Bump `epoch_cell[0] = epoch` right after
  `epoch += 1`.

Recommend **Option B** (shared tensor, persistent workers retained) as the primary
plan and **Option A** as fallback if shared-memory plumbing misbehaves.

### 4. Eval path: unchanged

The frozen `Eval.evaluate` calls `model(inputs)` on the test set with no flip
(prepare.py is frozen). The EXP-002 flip-TTA in `ResNet9.forward` (lines 114–119)
is unchanged and stays gated to the tail. Alternating flip is a *train-time only*
edit; the test-time orientation handling is entirely the existing TTA. No conflict:
TTA averages both orientations at eval; alternating flip just balances which
orientation the *training* gradient saw. They are orthogonal.

## Interaction with existing flip-TTA (called out explicitly)

EXP-002's flip-TTA already averages logits over `x` and `x.flip(-1)` at eval in the
final 20%. That makes the *model's eval prediction* orientation-symmetric by
construction. Alternating flip does not touch eval; it only affects what the trained
weights saw. The two are independent: TTA fixes test-time orientation variance,
alternating flip marginally improves the train-time augmentation distribution. There
is a mild theoretical reason the marginal value of *train-time* orientation balance
is *lower* when eval is already flip-symmetric (TTA partly launders residual
orientation bias in the weights), which further dampens the expected gain here.

## Expected magnitude (honest)

Pessimistic. airbench's headline +0.097pp (Table 1) is at **20 epochs**; the speedup
framing (27–38%) is also an epoch-starved-regime statement. The orientation
imbalance alternating flip removes scales as ~1/sqrt(epochs): going from ~20 to ~183
epochs shrinks the imbalance by ~3x, so the expected accuracy delta plausibly shrinks
from ~+0.1pp toward **+0.0 to +0.05pp** — i.e. *at or below* the +0.1pp bar. The
EXP-002 flip-TTA interaction dampens it further. My honest central estimate is
**+0.02pp (range −0.05 to +0.10pp)**, with a real chance of measuring net-zero or a
tiny regression from run noise. This is a *lower-expected-value* idea than the
whitening front-end or a capacity increase flagged in EXP-002's Next Steps. I would
rank it below those; its merit is that it is cheap and additive if it does land.

One upside scenario: if the shared-tensor plumbing is clean, this is nearly free to
*stack* on top of a stronger base later (whitening/wider net), where airbench reports
it remains additive — so even a null result here is informative for composition.

## Strongest risk

**The gain is too small to clear +0.1pp at 183 epochs** (the magnitude argument
above). This is the dominant risk and the honest reason to deprioritize: the very
property that makes alternating flip valuable (fixing finite-epoch orientation
imbalance) is largely already averaged out at our epoch count. A single-seed run
(no seed hacking allowed, fixed seed 42) has run-to-run noise plausibly comparable
to the expected effect, so even a true +0.05pp could read as noise either sign.

Secondary risks:
- **Throughput regression eats epochs (Risk).** Moving the flip from torchvision's
  C `RandomHorizontalFlip` into a Python `__getitem__` branch + `TF.hflip` is
  per-sample CPU work, but `RandomHorizontalFlip` was *also* per-sample CPU work in
  the same Compose — net CPU cost is ~equal. The real risk is Option A's per-epoch
  worker re-fork; Option B avoids it. With 8 workers and 1.6 GB VRAM headroom,
  DataLoader is not the bottleneck (30k img/s vs GPU-bound compute), so I expect
  ≤1–2 epochs lost, within EXP-002's 183 vs EXP-001's 192 normal variation. Must be
  confirmed by the run's `num_epochs`.
- **Correctness of the alternating scheme (Risk).** The parity must be per-image and
  must actually alternate each epoch. Verify offline before the official run: for a
  fixed idx, `(flip_bit[idx] + e) % 2` must toggle as `e` increments, and across idx
  at fixed `e` it must be ~50% flipped (balanced batches → BN sees mixed
  orientations every batch, unlike the coarse whole-epoch variant). A unit check on
  10 indices over 4 epochs suffices.
- **Epoch-counter plumbing staleness (Risk).** With persistent workers + prefetch,
  the worker may read a one-behind epoch for the first few batches of an epoch
  (Option B caveat). This does not break the alternating property; it only blurs the
  boundary by ≤`prefetch_factor` batches. Acceptable. The failure mode to avoid is
  the counter *never* updating in workers (silent: every epoch uses epoch 0 parity →
  degenerates to a *fixed* per-image flip, strictly worse than random flip). Guard
  by asserting in a smoke test that a worker observes `epoch_cell[0] > 0` after the
  parent bumps it (e.g. log the parity histogram for epoch 1 vs epoch 2 and confirm
  they differ).

## Effort

**Low–medium** for one experiment loop. The code is ~25 lines (one Dataset wrapper +
two construction lines + one epoch-cell bump). The medium component is the
epoch-counter-to-worker plumbing (shared-memory tensor) and the smoke test proving
workers actually see the updated epoch — that is the only non-trivial correctness
surface. No architecture or schedule changes; EMA+TTA untouched. The honest
recommendation: given the low expected magnitude vs the bar, this is a *lower
priority* than whitening or capacity, and is best considered as a cheap additive
rider on a stronger base rather than a standalone bar-clearing bet.
