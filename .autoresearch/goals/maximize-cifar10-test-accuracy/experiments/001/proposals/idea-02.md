# Proposal idea-02: Replace ResNet-20 with a fast-CIFAR ResNet-9 (DavidNet) + one-cycle

## Summary

Swap the deep-thin CIFAR ResNet-20 (270k params, never finishes its 64k-step schedule in
300s) for the wide-shallow 9-layer residual network from David Page's `cifar10-fast` /
DAWNBench recipe ("DavidNet" / ResNet-9, ~6.5M params). Pair it with the recipe that makes
that net famous: batch 512, SGD+Nesterov, a triangular one-cycle LR, weight decay 5e-4,
label smoothing, Cutout(8x8) on top of pad-4/crop/flip, and bf16 autocast + channels_last
for throughput. The published recipe reaches ~94% in 24 epochs / <94s on a single V100/A10.
On the H20 with a 300s training budget we can comfortably run the full schedule (and likely
several full schedules' worth of epochs), so the schedule actually *finishes* inside the
budget — directly fixing the diagnosed limiter. Target: **93.5-94.3%** best_test_acc
vs. the 91.57% baseline.

This is the single highest-upside idea in the slate. The main implementation risk is the
output-scale / LR / loss-reduction convention (the recipe sums the loss; we use mean
reduction and must translate the LR accordingly). I give the exact translated numbers below.

## What it targets

**Named limiter (from diagnosis):** "only ~37k steps / ~96 epochs fit in 300s so the
[ResNet-20] MultiStepLR [32k,48k]/64k schedule never finishes." A schedule that never
reaches its final decay leaves the model far from its converged accuracy.

**Causal chain to the metric:**
1. ResNet-9 is ~4x faster *per epoch to a given accuracy* than ResNet-20 on this task: it is
   shallow (9 conv layers vs. 20), so each forward/backward is cheap relative to the
   representational capacity it buys, and batch 512 + bf16 + channels_last maximize H20
   throughput. The published recipe converges in 24 epochs.
2. The one-cycle LR schedule is defined over a *fixed, small* number of epochs (24) and is
   tuned to *complete* (anneal to ~0) well inside the time budget. Unlike the baseline, the
   LR annealing phase — which is where most of the final accuracy is gained — actually
   happens.
3. A converged schedule on a 6.5M-param net with cutout + label smoothing lands at ~94% in
   the literature, ~2.5pp above the baseline, far exceeding the +0.1pp bar.

The lever is "genuinely better training code": a better architecture/optimizer/schedule
matched to the compute budget, not seed luck.

## Exact `train.py` changes

All changes are inside `train.py` (the only editable file). `prepare.py` (frozen eval +
`TIME_BUDGET_S=300`) is untouched. The time-budgeted `while` loop, the per-step
`torch.cuda.synchronize()` timing, the `evaluator.evaluate(model, device)` call (one per
epoch), and the summary block are kept structurally identical so timing/accounting and the
"at most one validation per epoch" constraint are preserved.

### 1. Normalization consistency (HARD constraint — must match frozen eval)

`prepare.py` `Eval` uses `mean=(0.4914,0.4822,0.4465)`, `std=(1,1,1)`, i.e. only mean
subtraction, no scaling. **Keep exactly this** in `train_tf`. The DavidNet reference uses a
different std, but we are bound to the eval's transform. Cutout must be applied *after*
`ToTensor`+`Normalize` and must zero pixels to **0.0** (which, post mean-subtraction, equals
the dataset mean in raw space — the standard cutout-with-mean-fill behavior). Concretely:

```python
mean, std = (0.4914, 0.4822, 0.4465), (1.0, 1.0, 1.0)  # MUST match prepare.py Eval
train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4),          # pad-4 reflect/zero + crop (keep default)
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
    Cutout(size=8),                                # zero an 8x8 square; see class below
])
```

A small `Cutout` transform (no new deps — pure torch) operating on the normalized CHW
tensor:

```python
class Cutout:
    def __init__(self, size=8):
        self.size = size
    def __call__(self, img):  # img: tensor [C,H,W]
        h, w = img.shape[1], img.shape[2]
        cy, cx = torch.randint(h, (1,)).item(), torch.randint(w, (1,)).item()
        y1, y2 = max(0, cy - self.size // 2), min(h, cy + self.size // 2)
        x1, x2 = max(0, cx - self.size // 2), min(w, cx + self.size // 2)
        img[:, y1:y2, x1:x2] = 0.0
        return img
```

### 2. Architecture: ResNet-9 / DavidNet

Replace the `BasicBlock` + `ResNet` classes with the DavidNet structure. Channel
progression `3 -> 64 -> 128 -> 256 -> 512`, residual blocks on the 128 and 512 stages,
max-pool after each stage, global max-pool, linear `512 -> 10`, logits scaled by `0.125`
(divide-by-8; the reference notes "output scale is important"). Use plain `nn.ReLU` and
standard `nn.BatchNorm2d` (torch-native, robust) rather than the reference's CELU(alpha) +
GhostBatchNorm — those are second-order tweaks worth ~0.1-0.3pp and add complexity/risk; we
can revisit them as a follow-up if the base recipe lands.

```python
def conv_bn(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )

class Residual(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = conv_bn(c, c)
        self.conv2 = conv_bn(c, c)
    def forward(self, x):
        return x + self.conv2(self.conv1(x))

class ResNet9(nn.Module):
    def __init__(self, num_classes=10, scale_out=0.125):
        super().__init__()
        self.scale_out = scale_out
        self.prep   = conv_bn(3, 64)
        self.layer1 = nn.Sequential(conv_bn(64, 128),  nn.MaxPool2d(2), Residual(128))
        self.layer2 = nn.Sequential(conv_bn(128, 256), nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(conv_bn(256, 512), nn.MaxPool2d(2), Residual(512))
        self.pool   = nn.MaxPool2d(4)          # 4x4 -> 1x1 after three /2 pools (32->16->8->4)
        self.fc     = nn.Linear(512, num_classes, bias=False)
        self.apply(self._init)
    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
    def forward(self, x):
        x = self.prep(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x) * self.scale_out
```

Spatial check: input 32x32 -> prep 32 -> layer1 (pool) 16 -> layer2 (pool) 8 ->
layer3 (pool) 4 -> `MaxPool2d(4)` -> 1x1. Correct. Param count ~6.5M (still tiny on an
H20; VRAM at batch 512 in bf16 is a few GB — well within the 98GB / soft constraint).

### 3. Optimizer, schedule, loss

Replace the hyperparameter block and optimizer/scheduler. Use **mean** cross-entropy
(PyTorch default) with `label_smoothing=0.2`, and translate the reference LR/WD accordingly
(see Reasoning for the derivation):

```python
BATCH_SIZE = 512
EPOCHS = 24                 # target schedule length; budget permitting it completes & repeats
PEAK_LR = 0.4               # mean-loss convention (myrtle "lambda"); per-sample
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4         # NOT scaled by batch (mean-loss convention)
WARMUP_FRAC = 5/24          # LR ramps 0->PEAK over first ~5 epochs, then anneals to ~0

optimizer = optim.SGD(model.parameters(), lr=PEAK_LR, momentum=MOMENTUM,
                      weight_decay=WEIGHT_DECAY, nesterov=True)

steps_per_epoch = len(train_loader)               # 50000//512 = 97 with drop_last
total_sched_steps = EPOCHS * steps_per_epoch
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=PEAK_LR, total_steps=total_sched_steps,
    pct_start=WARMUP_FRAC, anneal_strategy="linear",
    div_factor=1e9,        # start LR ~0 (triangular, matches the reference's 0->peak ramp)
    final_div_factor=1e9,  # end LR ~0
)
criterion = nn.CrossEntropyLoss(label_smoothing=0.2)
```

Loss line in the loop becomes `loss = criterion(outputs, targets)`.

**Schedule-vs-budget guard (important):** `OneCycleLR` raises if `scheduler.step()` is called
more than `total_steps` times. The budgeted `while` loop can run more steps than
`EPOCHS*steps_per_epoch` if there's time left. Two safe options — pick one in the plan:
- **(A) Single one-cycle, then stop early:** keep `MAX_STEPS = total_sched_steps` so the run
  ends when the cycle completes. Simplest; uses however much of the 300s the 24-epoch cycle
  needs (likely well under budget on H20). Risk: leaves budget unused.
- **(B) Size the cycle to the budget:** first do a short timing calibration (e.g. measure
  median step time over the first ~50 steps), estimate how many epochs fit in ~280s, set
  `EPOCHS` to that (clamped to >=24 if it fits, or fewer if not), build the scheduler once,
  and guard `scheduler.step()` with `if step < total_sched_steps`. This fills the budget
  with one well-formed cycle. **Recommended.** A longer single cycle (e.g. 30-50 epochs) on
  this net typically adds a few tenths of a pp over the 24-epoch cycle.

Do **not** restart multiple independent one-cycles back-to-back — repeated cold LR ramps
waste the annealing benefit. One cycle sized to the budget is the right design.

### 4. Throughput: channels_last + bf16 autocast

H20 supports bf16; bf16 autocast needs no `GradScaler` (unlike fp16), keeping the loop
simple and numerically safe. Apply once after `.to(device)`:

```python
model = ResNet9().to(device, memory_format=torch.channels_last)
torch.backends.cudnn.benchmark = True
```

In the step (inputs also to channels_last):
```python
inputs = inputs.to(device, non_blocking=True).to(memory_format=torch.channels_last)
with torch.autocast("cuda", dtype=torch.bfloat16):
    outputs = model(inputs)
    loss = criterion(outputs, targets)
loss.backward(); optimizer.step()
if step < total_sched_steps: scheduler.step()
```

Eval is called as-is via `evaluator.evaluate(model, device)`; the frozen eval runs the model
in fp32/default — bf16 autocast is training-only, so eval correctness is unaffected.
(Optional: the model still produces correct logits in eval mode without autocast.)

### 5. Keep intact

The `while total_training_time < TIME_BUDGET_S` loop, per-step `synchronize()` + `dt`
accounting, the `if step % 50` logging, the one `evaluator.evaluate` per epoch, `best_acc`
tracking, and the final `print` summary (`best_test_acc:` etc.) all stay. Update the
`num_params` / model-name prints. Keep `torch.manual_seed(42)` (no seed hacking).

## Reasoning with cited pointers

**Architecture + recipe source (concrete numbers):**
- johanwind, "94% on CIFAR-10 in 94 lines and 94 seconds"
  (https://johanwind.github.io/2022/12/28/cifar_94.html): channel progression
  `3->64->128->128->128->256->512->512->512`, max-pool after each channel doubling, final
  4x4 global max-pool + linear `512->10`, **output divided by 8** ("output scale is
  important... amplifying implicit bias by small initialization"); batch 512, 24 epochs,
  SGD+Nesterov mom 0.9, **label smoothing 0.2**, cutout 8x8 + pad-4-crop + hflip, fp16. The
  LR is given per-batch as triangular `[0.1/BS, 0.6/BS, 0]` over epochs `[0, 24/5, 24]`,
  and crucially the **loss is summed** (`reduction='none'` then `.sum().backward()`), with
  `weight_decay = 5e-4*BATCH_SIZE`. Reported **94.02%** mean of 40 runs, <94s on an A10.
- 99991/cifar10-fast-simple (model.py / train.py): confirms the residual blocks sit on the
  128 and 512 stages (two `Residual` adds), global max-pool, linear with `scale_out`.
- Myrtle.ai "How to Train Your ResNet" Post 5 (Hyperparameters,
  https://myrtle.ai/2018/09/24/how-to-train-your-resnet-5) and Post 3 (Regularisation,
  https://myrtle.ai/how-to-train-your-resnet-3-regularisation/): the **mean-loss**
  convention hyperparameters are `lambda (max LR)=0.4, rho (mom)=0.9, alpha (wd)=5e-4`, LR
  ramps linearly 0->peak over the first 5 epochs then decays linearly to 0; cutout 8x8 takes
  the 35-epoch baseline to a median 94.3%. Final tuned result 94.2%.

**LR/WD/loss-reduction translation (the one subtle correctness point):** johanwind *sums*
the per-sample loss over a batch of 512, so its gradient is 512x larger than PyTorch's
default *mean* loss; that is exactly why their LR is `0.6/512` and their WD is `5e-4*512`.
Converting to the standard **mean** reduction we use, the equivalent peak LR is
`0.6/512 * 512 = 0.6` (per-sample mean-loss), and weight decay is plain `5e-4`. The myrtle
"mean-loss" lambda is quoted as **0.4**. I propose `PEAK_LR=0.4` (myrtle's tuned value) with
mean loss + `WEIGHT_DECAY=5e-4`, which is the internally-consistent, well-documented
operating point. The "flat direction" insight (Post 5) — what matters is
`lambda*alpha/(1-rho) = 0.4*5e-4/0.1 = 0.002` — means the recipe is forgiving to modest LR
mis-set, de-risking this translation. (If 0.4 underperforms, 0.5-0.6 is the safe sweep
direction; this is a fast follow within one loop.)

**Why it fits the budget (fixes the limiter):** the published 24-epoch cycle is <94s on a
V100/A10. The H20 is materially faster, and bf16 + channels_last + batch 512 push img/s
higher still. 24 epochs x 97 steps = ~2330 steps; even at a conservative ~6-7k img/s that is
well under 300s, so the full annealing phase completes — the exact thing the ResNet-20
baseline cannot do. Option (B) sizes a single longer cycle to consume the remaining budget.

**Why it beats 91.57% by a wide margin:** the literature mean is ~94.0-94.3% for this exact
recipe; even discounting for our conservative simplifications (plain ReLU/BN instead of
CELU/GhostBN, the eval's no-scaling normalization which differs from the reference's
per-channel std) we expect to clear ~93%.

## Expected accuracy estimate

**93.5-94.3%** best_test_acc. Justification: published recipe is 94.0-94.3% (johanwind 40-run
mean 94.02%, myrtle tuned 94.2%). Two downward adjustments: (1) we use plain BatchNorm and
ReLU instead of GhostBatchNorm + CELU (~ -0.1 to -0.3pp per the series' ablations); (2) we
are forced onto the eval's `std=(1,1,1)` normalization rather than the reference per-channel
std, a minor input-scaling difference largely absorbed by the first BN. Offsetting upward:
on the H20 we likely fit a longer single cycle (30-50 epochs) than 24, worth a few tenths.
Net central estimate **~93.7%**, comfortably above the +0.1pp bar (need >=91.67%).

## Risk assessment (worst case)

**Strongest risk / assumption that most needs to hold:** the LR / loss-reduction / output-scale
convention. If the mean-loss LR is mis-set (e.g. someone leaves the loss summed *and* uses
LR 0.4, making the effective LR 512x too large), training diverges (NaN/loss explosion) and
best_acc could end up *below* baseline or the run crashes. Mitigation: the proposal pins the
**mean** reduction explicitly, `PEAK_LR=0.4`, `WEIGHT_DECAY=5e-4`, and keeps the `scale_out=0.125`
logit scaling (without which the effective gradient scale and label-smoothing interaction
shift). The `lambda*alpha/(1-rho)` flat-direction result means small LR errors are tolerated;
a single confirmation run will reveal divergence in the first epoch (watch the smoothed loss
print), and the safe corrective sweep is LR in {0.2, 0.4, 0.6}.

**Secondary risks:**
- *OneCycleLR step-count overrun:* if the budgeted loop calls `scheduler.step()` past
  `total_steps`, `OneCycleLR` raises and the run crashes (counts as failure). Mitigated by
  the `if step < total_sched_steps` guard (Option B) or `MAX_STEPS=total_sched_steps`
  (Option A).
- *Budget under-use:* a bare 24-epoch cycle may finish in ~60-120s, leaving ~half the budget
  idle and capping accuracy below what the budget allows. Option (B) (size cycle to budget)
  removes this; it is the recommended path.
- *Cutout-after-normalize semantics:* zeroing post-normalization fills with the dataset mean
  in raw space (correct, standard). Low risk but called out so the planner doesn't "fix" it
  to a non-zero fill.
- *channels_last/bf16 throughput not materializing:* worst case it's merely as fast as fp32
  NCHW — still ample for >=24 epochs in budget; no accuracy cost.

**Worst-case outcome:** a divergence or OneCycle overrun crash on the first run. Both are
detectable in <1 minute of the log and fixable with a one-line LR change or the step guard,
so the downside is one wasted partial run, not a dead end. The architecture/recipe itself is
extremely well-replicated (multiple independent implementations at 94%), so method risk is
low; the risk is purely in faithful porting of conventions.

## Estimated effort

**Medium** (one experiment loop). It is a focused rewrite of `train.py`'s model + optimizer +
schedule + augmentation blocks (~80-100 lines changed), all torch-native with no new deps.
The loop scaffold, timing, eval call, and summary are reused verbatim. The only genuinely
fiddly part is the LR/loss-reduction translation and the OneCycle step guard, both fully
specified above. One confirmation run plus possibly one LR re-run fits inside a single loop.

## Sources
- [94% on CIFAR-10 in 94 lines and 94 seconds — johanwind](https://johanwind.github.io/2022/12/28/cifar_94.html)
- [How to Train Your ResNet 5: Hyperparameters — Myrtle.ai](https://myrtle.ai/2018/09/24/how-to-train-your-resnet-5)
- [How to Train Your ResNet 3: Regularisation (Cutout 8x8) — Myrtle.ai](https://myrtle.ai/how-to-train-your-resnet-3-regularisation/)
- [99991/cifar10-fast-simple (model.py, train.py)](https://github.com/99991/cifar10-fast-simple)
