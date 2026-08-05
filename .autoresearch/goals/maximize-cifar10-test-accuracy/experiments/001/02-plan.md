# Plan EXP-001: ResNet-9 (DavidNet) + time-based one-cycle on CIFAR-10
- **Created**: 2026-06-28

Chosen idea: brainstorm `01-brainstorm.md` § Chosen Idea (Idea 02, refined per Codex review in `01-idea-review.md`). Baseline: **91.57%** (improvement bar **≥ 91.67%**, i.e. +0.1pp). Only `train.py` is edited; `prepare.py` is frozen.

## Milestones

### Milestone 1: Implement the new train.py
- [ ] Replace `BasicBlock`/`ResNet` with the DavidNet/ResNet-9 architecture (conv_bn helper, Residual block, ResNet9 with scale_out=0.125).
- [ ] Add a `Cutout` transform (pure torch) and wire augmentation: RandomCrop(32,pad4) + HorizontalFlip + ToTensor + Normalize(EVAL_MEAN, EVAL_STD) + Cutout(8).
- [ ] Set hyperparameters: BATCH_SIZE=512, PEAK_LR=0.4, MOMENTUM=0.9, WEIGHT_DECAY=5e-4, LABEL_SMOOTHING=0.2, PCT_START=0.15.
- [ ] Replace MultiStepLR with a **time-based one-cycle** LR set each step from `total_training_time / TIME_BUDGET_S` (triangular: linear ramp 0→PEAK over first PCT_START of budget, linear decay PEAK→0 after).
- [ ] SGD(nesterov=True, wd=5e-4); CrossEntropyLoss(label_smoothing=0.2, mean reduction).
- [ ] Throughput: `model.to(device, memory_format=torch.channels_last)`, inputs to channels_last, `torch.autocast('cuda', bfloat16)` around forward+loss (no GradScaler), `cudnn.benchmark=True`. Train DataLoader: add `persistent_workers=True, prefetch_factor=4` (reduce per-epoch worker churn — many short epochs).
- [ ] Keep verbatim: the `while total_training_time < TIME_BUDGET_S` loop, per-step `t0`/`synchronize()`/`dt` accounting, the **single** `evaluator.evaluate` per epoch, `best_acc` tracking, and the final summary prints. Keep `torch.manual_seed(42)` (no seed hacking). NO TTA (keeps eval single-forward → bounds wall clock).
- [ ] In the per-epoch eval print line, also print cumulative **wall** elapsed `time.time() - t_start` so the 10-min cap can be monitored live (current code only prints `total_seconds` at the very end — review concern #2).
- [ ] Static syntax check: `uv run python -m py_compile train.py`.
- [ ] **One-batch smoke test** (review concern #6) — run before the official run, e.g. `CUDA_VISIBLE_DEVICES=1 uv run python -c "<inline>"`: instantiate `ResNet9().to('cuda', memory_format=channels_last)`, push a `[8,3,32,32]` channels_last CUDA tensor through `torch.autocast('cuda', bfloat16)`, assert output shape `[8,10]`, compute CE loss, `loss.backward()`, `optimizer.step()` — catches spatial/dtype/scale bugs that `import` cannot.

### Milestone 2: Run the experiment
- [ ] `cd` to project root, ensure clean tree, remove any stale `run.log`.
- [ ] Launch under an **external wall-clock guard** (review concern #1): `timeout 600 bash -c 'CUDA_VISIBLE_DEVICES=1 uv run train.py > run.log 2>&1'`. Exit code 124 ⇒ wall-clock kill ⇒ failure.
- [ ] Monitor epoch-1 smoothed loss for divergence (NaN/inf or not decreasing) per Abort Criteria; watch the per-epoch wall-elapsed print vs the 600s cap.

### Milestone 3: Verify result
- [ ] On completion, `grep "^best_test_acc:\|^training_seconds:\|^total_seconds:\|^peak_vram_mb:\|^num_epochs:" run.log`.
- [ ] Apply Verification Protocol (below). Record outcome. Remove `run.log` after recording.

## Code Changes

- **train.py** (full rewrite of model + recipe; loop scaffold preserved). Conceptual diff:

  1. **Normalization constant (match frozen eval).** At top of `main` (or module scope):
     `EVAL_MEAN, EVAL_STD = (0.4914, 0.4822, 0.4465), (1.0, 1.0, 1.0)` — identical to `prepare.py` `Eval.__init__`. Used by the train transform. (Assert-by-construction: same literals.)

  2. **Cutout transform** (no new deps; operates on normalized CHW tensor, fills 0.0 = dataset mean post-subtraction):
     ```python
     class Cutout:
         def __init__(self, size=8):
             self.size = size
         def __call__(self, img):  # img: [C,H,W] tensor
             h, w = img.shape[1], img.shape[2]
             cy = int(torch.randint(h, (1,)).item()); cx = int(torch.randint(w, (1,)).item())
             s = self.size // 2
             y1, y2 = max(0, cy - s), min(h, cy + s)
             x1, x2 = max(0, cx - s), min(w, cx + s)
             img[:, y1:y2, x1:x2] = 0.0
             return img
     ```
     Train transform: `Compose([RandomCrop(32, padding=4), RandomHorizontalFlip(), ToTensor(), Normalize(EVAL_MEAN, EVAL_STD), Cutout(8)])`.

  3. **Architecture — DavidNet/ResNet-9** (replaces `BasicBlock`+`ResNet`):
     ```python
     def conv_bn(c_in, c_out):
         return nn.Sequential(
             nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
             nn.BatchNorm2d(c_out),
             nn.ReLU(inplace=True))

     class Residual(nn.Module):
         def __init__(self, c):
             super().__init__(); self.c1 = conv_bn(c, c); self.c2 = conv_bn(c, c)
         def forward(self, x): return x + self.c2(self.c1(x))

     class ResNet9(nn.Module):
         def __init__(self, num_classes=10, scale_out=0.125):
             super().__init__(); self.scale_out = scale_out
             self.prep   = conv_bn(3, 64)
             self.layer1 = nn.Sequential(conv_bn(64, 128),  nn.MaxPool2d(2), Residual(128))
             self.layer2 = nn.Sequential(conv_bn(128, 256), nn.MaxPool2d(2))
             self.layer3 = nn.Sequential(conv_bn(256, 512), nn.MaxPool2d(2), Residual(512))
             self.pool   = nn.MaxPool2d(4)
             self.fc     = nn.Linear(512, num_classes, bias=False)
             self.apply(self._init)
         @staticmethod
         def _init(m):
             if isinstance(m, (nn.Conv2d, nn.Linear)):
                 nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
         def forward(self, x):
             x = self.prep(x); x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
             x = self.pool(x).flatten(1)
             return self.fc(x) * self.scale_out
     ```
     Spatial: 32→(prep)32→(l1 pool)16→(l2 pool)8→(l3 pool)4→(MaxPool4)1. ~6.5M params (tiny on H20).

  4. **Optimizer + loss:**
     ```python
     model = ResNet9().to(device, memory_format=torch.channels_last)
     torch.backends.cudnn.benchmark = True
     optimizer = optim.SGD(model.parameters(), lr=PEAK_LR, momentum=MOMENTUM,
                           weight_decay=WEIGHT_DECAY, nesterov=True)
     criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
     ```
     Remove the `MultiStepLR` scheduler entirely.

  5. **Time-based one-cycle LR** — set at the TOP of each step body, before forward, from the running `total_training_time` (already tracked by the loop):
     ```python
     progress = min(1.0, total_training_time / TIME_BUDGET_S)
     if progress < PCT_START:
         lr_now = PEAK_LR * progress / PCT_START
     else:
         lr_now = PEAK_LR * (1.0 - progress) / (1.0 - PCT_START)
     for g in optimizer.param_groups:
         g["lr"] = lr_now
     ```
     This anneals LR→~0 by the 300s budget regardless of throughput — no step calibration, no scheduler overrun. (LR is read from the pre-step running time, so there is at most a single-step overshoot at each boundary — negligible; resolves brainstorm-review concern #1.) `lr_now` reused for logging.

  6. **Step body (forward/backward) with bf16 + channels_last:**
     ```python
     inputs = inputs.to(device, non_blocking=True).to(memory_format=torch.channels_last)
     targets = targets.to(device, non_blocking=True)
     optimizer.zero_grad(set_to_none=True)
     with torch.autocast("cuda", dtype=torch.bfloat16):
         outputs = model(inputs)
         loss = criterion(outputs, targets)
     loss.backward()
     optimizer.step()
     # remove scheduler.step()
     ```
     Keep `torch.cuda.synchronize()` + `dt` accounting UNCHANGED (the budget meter — never tamper). Keep `loss.item()` for the smoothed-loss log (optionally guard inside the `% 50` branch; harmless either way).

  7. **Summary/prints:** keep all `best_test_acc:` … `num_params:` lines. Update the model-name print to ResNet-9.

## Configuration Changes
- Architecture: ResNet-20 (270k params) → ResNet-9/DavidNet (~6.5M params) (proven fast-CIFAR net; higher convergence-per-epoch).
- BATCH_SIZE: 128 → 512 (better H20 utilization; DavidNet canonical batch).
- LR schedule: MultiStepLR[32k,48k]/64k → **time-based triangular one-cycle**, peak 0.4, warmup 15% (fixes under-annealing; completes within budget).
- Optimizer: SGD → SGD+**Nesterov**; WEIGHT_DECAY 1e-4 → 5e-4 (DavidNet recipe).
- Loss: cross_entropy → CrossEntropyLoss(**label_smoothing=0.2**), mean reduction (pin with PEAK_LR=0.4, wd=5e-4, scale_out=0.125 — review concern #4).
- Augmentation: +Cutout(8); keep pad4-crop + hflip.
- Precision/memory: fp32 NCHW → **bf16 autocast + channels_last** + cudnn.benchmark (more steps in budget; no GradScaler).
- Normalization: unchanged, mean=(0.4914,0.4822,0.4465) std=(1,1,1) (must match frozen eval — review concern #2).

## Execution Environment
- Method: local, under external wall guard — `timeout 600 bash -c 'CUDA_VISIBLE_DEVICES=1 uv run train.py > run.log 2>&1'` from project root.
- Resources: single NVIDIA H20 (**GPU 1**; GPU 0 is in use). VRAM at bs512 bf16 ≈ a few GB (well within 98GB soft constraint).
- Estimated runtime: ~300s training + eval overhead (one fp32 forward/epoch, no TTA; ~0.5–0.8s/eval incl. eval-loader worker spawn × however many epochs fit, ~120–360) + ~5s startup (env built) ≈ **6–8 min wall**. The `timeout 600` is the hard backstop; the per-epoch wall-elapsed print makes overrun visible early. If the live run trends past ~560s wall, reduce eval cadence (still ≤1/epoch — e.g. skip eval while `progress < PCT_START`, since early epochs never hold the best) before the official run.
- Log output: `run.log` in project root (redirect, no tee — avoid context flooding). Extract via `grep "^best_test_acc:\|^peak_vram_mb:\|^training_seconds:\|^total_seconds:\|^num_epochs:" run.log`.
- Tool skill: none (local run).

## Abort Criteria
- **Divergence:** epoch-1 smoothed loss is NaN/inf, or not decreasing after ~1 epoch → kill; reduce PEAK_LR (sweep {0.2, 0.4, 0.6}) or re-check the mean-loss/scale_out convention.
- **Wall-clock:** `timeout 600` returns exit code **124** (or the per-epoch wall-elapsed print approaches 600s) → treat as failure (per TASK.md 10-min rule); reduce eval cadence and re-run.
- **Crash / no summary:** `grep best_test_acc` on `run.log` is empty after the process exits → read `tail -n 50 run.log` for the stack trace and fix.
- **OOM / CUDA error:** kill, reduce batch size.

## Verification Protocol

### Verification Procedure
Baseline (from `exp-index.sh baseline`): **91.57%**. Necessary conditions (from `01-definition.md` § Verification), evaluated in order; first failure ⇒ `no-improvement` (or `invalid`/`crash` per cause):

1. **Runs clean within budget.** Command: `timeout 600 bash -c 'CUDA_VISIBLE_DEVICES=1 uv run train.py > run.log 2>&1'`. Pass iff exit code is 0 (not 124), `total_seconds` < 600, and `grep "^best_test_acc:" run.log` is non-empty. Exit 124 / empty summary ⇒ `crash`.
2. **Used the full training budget (no early-stop / budget-tamper).** `training_seconds` must be ≈ the frozen `TIME_BUDGET_S=300` (accept `>= 295`; it ends one step after crossing 300). And confirm `prepare.py` is byte-unchanged: `git diff --quiet -- prepare.py` (and `TIME_BUDGET_S=300` still in it). Any deviation ⇒ `invalid`.
3. **Improves over baseline by ≥ +0.1pp.** Compare with an explicit numeric comparator (not shell string compare):
   ```bash
   BEST=$(grep "^best_test_acc:" run.log | grep -oE "[0-9]+\.[0-9]+" | head -1)
   python -c "import sys; sys.exit(0 if float('$BEST') >= 91.67 else 1)" && echo PASS || echo "NO-IMPROVEMENT ($BEST < 91.67)"
   ```
   Pass iff `BEST >= 91.67`. Otherwise ⇒ `no-improvement`.
4. **Genuine, in-scope method change (anti-reward-hack).** Confirm ALL:
   - `git diff --name-only` (vs the experiment branch point) shows **only `train.py`**; `prepare.py` untouched.
   - Seed unchanged: `train.py` still contains `torch.manual_seed(42)` and no seed search/loop (no seed hacking).
   - Exactly one `evaluator.evaluate(...)` call site, invoked once per epoch (`grep -c "evaluator.evaluate" train.py` == 1).
   - `train.py` does **not** build or iterate the test set itself nor touch eval internals: `grep -nE "train=False|evaluator\.loader|\.loader|CIFAR10\(" train.py` must show no `train=False`, no `evaluator.loader`, and CIFAR10 only with `train=True`. Test labels are touched solely through the frozen `Eval.evaluate` path.
   Any violation ⇒ `invalid`.

Scoring metric = `best_test_acc` (best across epochs), exactly as the frozen `Eval.evaluate` reports it.

### Informational Metrics (Optional)
- peak_vram_mb: `grep "^peak_vram_mb:" run.log` — VRAM used (soft-constraint awareness).
- training_seconds / num_epochs / num_steps: `grep "^training_seconds:\|^num_epochs:\|^num_steps:" run.log` — confirms full budget used and epochs achieved.
- num_params: `grep "^num_params:" run.log` — model size (~6.5M) for the accuracy/compute trade-off.
- total_seconds: `grep "^total_seconds:" run.log` — wall clock vs the 10-min cap.
