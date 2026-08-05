# Plan EXP-003: Frozen patch-whitening first convolution
- **Created**: 2026-06-28

Chosen idea: `01-brainstorm.md` § Chosen Idea (Idea-01, Codex review pick), refined. Full proposal: `proposals/idea-01.md`. Base: current `train.py` (EXP-002, DavidNet + EMA(0.998) + tail flip-TTA, **95.72%**). Bar: **≥ 95.82%** (+0.1pp). Only `train.py` edited; `prepare.py` frozen. We prepend a **frozen 3×3/pad-1 ZCA-whitening conv** (eigendecomposition of CIFAR-10 training-patch covariance, computed once at startup OFF the 300s timer, in the exact eval normalization space) before the existing `prep`, feeding the learnable stem decorrelated 54-channel input. EMA + flip-TTA and the entire 300s one-cycle recipe stay byte-identical for clean attribution.

## Milestones

### Milestone 1: Implement whitening front-end in train.py
- [ ] Add `compute_whitening_weight(train_set, mean, kernel=3, n_img=2000, n_patches=50000, eps=1e-4)` builder. Sample raw images from `train_set.data[:n_img]` (HWC uint8), `/255`, subtract `EVAL_MEAN` (std=1, eval normalization), unfold into 3×3×3=27-dim patches, center, covariance (27×27), `torch.linalg.eigh`, scale eigvecs by `1/sqrt(eigval+eps)`, reshape to `[27,3,3,3]`, `cat([W,-W])` → `[54,3,3,3]` frozen weight. **Cap at `n_img=2000` images** (→ ~1.8M interior patches, ~190 MB fp32, sub-second eigh) so the off-budget materialization is bounded (review #2); then optionally subsample to `n_patches` via a **LOCAL `torch.Generator().manual_seed(0)`** (NOT global `torch.randperm` after `manual_seed(42)`) so global RNG state is untouched (review #5). The covariance is estimated from unpadded interior patches; the actual conv uses pad=1, so the 1-pixel border sees zero-normalized padding absent from the estimate — a negligible approximation (consistent with hlb/airbench, which whiten interior patches and apply padded), and `prep`'s BN re-normalizes downstream (review #4).
- [ ] `ResNet9.__init__`: add `self.whiten = nn.Conv2d(3, 54, 3, padding=1, bias=False)` with `requires_grad_(False)`; change `self.prep = conv_bn(54, 64)` (was `conv_bn(3,64)`); add `load_whitening(weight)` method (copies weight under `no_grad`, re-asserts `requires_grad=False`). `_forward_once` prepends `x = self.whiten(x)`.
- [ ] `main()`: after `model = ResNet9(...).to(device, channels_last)` and BEFORE the optimizer + EMA construction, compute the whitening weight and `model.load_whitening(...)`, timed separately and printed as `whitening_seconds` (OFF the 300s budget — placed before `t_start_training`).
- [ ] Build the SGD optimizer over `[p for p in model.parameters() if p.requires_grad]` (excludes the frozen whitening conv — defensive; review #6 notes SGD skips `grad is None` anyway).
- [ ] EMA construction (`AveragedModel(...)`) stays AFTER `load_whitening`, so the EMA's initial copy carries the loaded whitening; since `model.whiten.weight` never changes, its EMA equals the constant (correct at eval).
- [ ] Keep everything else byte-identical: schedule, SGD+Nesterov, wd 5e-4, LABEL_SMOOTHING=0.2, Cutout 8, EMA_DECAY 0.998 / warmup 0.15, TTA_START_FRAC 0.8, bf16+channels_last, batch 512, `torch.manual_seed(42)`/`cuda.manual_seed(42)`, budget meter, per-epoch `wall:` print, summary.
- [ ] `uv run python -m py_compile train.py`.
- [ ] **Smoke test** (scratch script, not committed, GPU 1): build model, compute+load whitening, build `ema_model`; assert `model.whiten.weight.requires_grad is False`; assert the whitening conv is **NOT** in the optimizer's param tensors (compare `id()`s) and that `sum(numel of optimizer params)` == learnable-param numel; run a `[8,3,32,32]` channels_last CUDA batch through `_forward_once` and assert the feature map entering `self.pool` is `[8,512,4,4]` (layer widths 64/128/256/512 unchanged — only `prep`'s input grew 3→54 — so the map entering `pool` is still 512×4×4 → MaxPool2d(4) → 512) i.e. the pooling chain is intact and output is `[8,10]`; exercise the **real frozen path** `from prepare import Eval; ev=Eval(); ev.evaluate(ema_model,device)` and `ev.evaluate(model,device)` return finite `(loss,acc)` in `[0,100]`. **The smoke test asserts only finiteness/shape on an untrained throwaway model; its accuracy values are discarded and used to tune/select NOTHING (no leakage into the scored run).** (Optional: confirm whitened-patch output covariance ≈ identity on a sample — sanity that whitening works.)

### Milestone 2: Run the experiment
- [ ] Clean tree, remove stale `run.log`.
- [ ] Launch wall-guarded: `timeout 600 bash -c 'CUDA_VISIBLE_DEVICES=1 uv run train.py > run.log 2>&1'`. Exit 124 ⇒ wall-kill ⇒ failure.
- [ ] Monitor epoch-1 loss (divergence) and per-epoch `wall:` vs 600s. **Log early-epoch (ep≤10) and tail eval accuracies** to compare against the EXP-002 trajectory (review concern #4 — makes a null interpretable: did conditioning speed early convergence even if final acc didn't move?).

### Milestone 3: Verify
- [ ] On completion, grep summary; apply Verification Protocol; record; remove `run.log`.

## Code Changes

- **train.py** (additive; training dynamics untouched). Three touch points (per `proposals/idea-01.md`, Option A):

  1. **Whitening builder** (new module-level function, near `Cutout`):
     ```python
     def compute_whitening_weight(train_set, mean, kernel=3, n_img=2000, n_patches=50000, eps=1e-4):
         mean_t = torch.tensor(mean).view(1, 3, 1, 1)
         n_img = min(n_img, len(train_set.data))                               # cap materialized patches
         imgs = torch.from_numpy(train_set.data[:n_img]).float().div_(255.0)   # [N,32,32,3]
         imgs = imgs.permute(0, 3, 1, 2).contiguous() - mean_t                 # [N,3,32,32], eval space
         p = imgs.unfold(2, kernel, 1).unfold(3, kernel, 1)                    # [N,3,H',W',k,k]
         p = p.permute(0, 2, 3, 1, 4, 5).reshape(-1, 3 * kernel * kernel)      # [M,27]
         if p.shape[0] > n_patches:
             g = torch.Generator().manual_seed(0)                             # LOCAL rng — no global side effect
             p = p[torch.randperm(p.shape[0], generator=g)[:n_patches]]
         # ^ n_img cap (2000) bounds materialization to ~1.8M patches before this subsample
         p = p - p.mean(0, keepdim=True)
         cov = (p.T @ p) / (p.shape[0] - 1)                                    # [27,27]
         eigvals, eigvecs = torch.linalg.eigh(cov, UPLO='U')
         W = (eigvecs / torch.sqrt(eigvals + eps).unsqueeze(0)).T              # [27,27], rows=filters
         W = W.reshape(3 * kernel * kernel, 3, kernel, kernel)                 # [27,3,k,k]
         return torch.cat([W, -W], dim=0).contiguous()                        # [54,3,k,k] fp32
     ```
     Why: decorrelating the 3×3 RGB patch covariance (currently highly anisotropic) sphereizes the first-layer loss surface → faster early convergence and a marginally lower-loss tail minimum within the same budget. Reading `train_set.data` (raw uint8) bypasses the augmenting transform so patch stats match eval inputs exactly. Cheap (≤50k patches, 27×27 eigh — milliseconds).

  2. **`ResNet9`** — frozen whitening front-end (Option A: whitening then `prep` as the learnable mixer):
     ```python
     # __init__: BEFORE self.apply(...)
     self.whiten = nn.Conv2d(3, 54, 3, padding=1, bias=False)
     self.whiten.weight.requires_grad_(False)
     self.prep = conv_bn(54, 64)          # was conv_bn(3, 64)
     # ... layer1/2/3, pool, fc, self.tta unchanged ...
     # self.apply(self._weights_init)     # kaiming-inits everything incl. whiten — overwritten later

     def load_whitening(self, weight):
         with torch.no_grad():
             self.whiten.weight.copy_(weight.to(self.whiten.weight.device, self.whiten.weight.dtype))
         self.whiten.weight.requires_grad_(False)

     def _forward_once(self, x):
         x = self.whiten(x)               # NEW first line
         x = self.prep(x)
         x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
         x = self.pool(x).flatten(1)
         return self.fc(x) * self.scale_out
     ```
     `forward` (TTA-gated) unchanged. Kernel 3 / **pad 1 preserves 32×32**, so the MaxPool chain (/2,/2,/2,/4) is untouched and the map entering `pool` stays 512×4×4 → 512. The `prep` BN self-calibrates the whitened scale (no hand-tuned output scale).

  3. **`main()`** — compute/load off-timer, filter optimizer, order EMA after load:
     ```python
     model = ResNet9(NUM_CLASSES).to(device, memory_format=torch.channels_last)
     t_w = time.time()
     w_weight = compute_whitening_weight(train_set, EVAL_MEAN, kernel=3)
     model.load_whitening(w_weight)
     whitening_seconds = time.time() - t_w
     print(f"whitening_seconds: {whitening_seconds:.2f}")
     num_params = sum(p.numel() for p in model.parameters())
     learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
     ...
     optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad],
                           lr=PEAK_LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY, nesterov=True)
     # criterion ... then ema_model = AveragedModel(model, ...)  # AFTER load_whitening
     ```
     `whitening_seconds` is printed and the block sits BEFORE `t_start_training = time.time()`, mirroring the off-budget `evaluator = Eval()` construction. **Timing semantics (precise):** `t_start` is at the top of `main()`, so the whitening block runs *after* `t_start` and *before* `t_start_training` — it is therefore part of `startup_seconds` (= `t_start_training − t_start`) and `total_seconds` (= `t_end − t_start`), but is **excluded from `training_seconds`** (the per-step accumulated `total_training_time`, which is the 300s budget). The budget the goal fixes is `training_seconds`; whitening does not consume it. The 600s `timeout` wall guard still covers everything including startup.

## Configuration Changes
- Architecture front-end: none → frozen 3×3/pad-1 ZCA-whitening conv (3→54), then `prep` widened 3→54 input. ~1458 frozen + ~29k extra learnable params (negligible vs 6.57M).
- Optimizer param set: all params → `requires_grad`-filtered (excludes frozen whitening conv).
- Everything else (schedule, optimizer hyperparams, **PEAK_LR=0.4**, wd, LS, Cutout, EMA, TTA, batch, seed): **unchanged** from EXP-002 (clean single-variable A/B; PEAK_LR held fixed — a whitening-enabled LR raise is a deferred follow-up, not this run).

## Execution Environment
- Method: local, wall-guarded — `timeout 600 bash -c 'CUDA_VISIBLE_DEVICES=1 uv run train.py > run.log 2>&1'` from project root.
- Resources: single H20, **GPU 1** (GPU 0 in use). VRAM ≈ EXP-002's 1.6 GB + tiny front-end — far under 98 GB.
- Estimated runtime: ~300s training + eval; whitening eigendecomposition adds <~2s startup (off-budget). The whitening conv adds one cheap 3×3 forward per step (54 ch) → expect a small epoch-count dip (a few epochs) but ~same wall as EXP-002's 442.7s; well under 600s.
- Log output: `run.log` (redirect). Extract: `grep "^best_test_acc:\|^whitening_seconds:\|^training_seconds:\|^total_seconds:\|^peak_vram_mb:\|^num_epochs:" run.log`.
- Tool skill: none (local).

## Abort Criteria
**Discipline:** an abort means *kill and mark failed*, not mutate-and-rerun. The reviewed experiment is exactly "frozen 3×3 ZCA-whitening front-end (kernel=3, pad=1, eps=1e-4, Option A) on the EXP-002 recipe, everything else identical." **Any change to a hyperparameter or design choice — eps, kernel size, padding, output scale, whitening placement, Option B mixer, PEAK_LR — is a DIFFERENT experiment and is forbidden as a within-run patch** (closes post-hoc tuning). The ONLY in-run fix permitted is a pure *wiring* bug that prevents the unchanged experiment from running at all (a NameError/shape-mismatch/dtype crash in the new code), and only if the fix changes none of those pinned values; a genuine infra crash may be retried per the execute skill.
- **Divergence:** epoch-1 smoothed loss NaN/inf or not decreasing → kill, mark `failed`. Training recipe is unchanged, so this points to a whitening/forward **wiring** bug (e.g. a shape/dtype mismatch). A wiring fix that leaves eps/kernel/padding/scale/placement exactly as pinned above and merely makes the code run may be applied once; **if divergence is instead caused by a pinned value (e.g. eps too small blowing up low-variance directions), that is a research result → mark `failed`, do NOT retune eps in-loop** (retuning is a future experiment). If it recurs, stop and report.
- **Wall-clock:** `timeout` exit 124, or per-epoch `wall:` trending toward 600s → kill, mark `failed`.
- **Marginal/sub-bar result (monitor only, never patch):** if the tail eval trails 95.82%, let it finish — a sub-bar result is a valid `no-improvement`. Record the early-epoch vs EXP-002 deltas regardless (interpretability).
- **Crash / no summary / OOM:** grep empty after exit → `tail -n 50 run.log`; genuine infra crash retried per execute skill's infra policy.

## Verification Protocol

### Verification Procedure
Baseline (from `exp-index.sh baseline`): **95.72%**; bar **≥ 95.82%**. Conditions in order; first failure ⇒ stop with that verdict.

1. **Runs clean within the wall guard.** `timeout 600 bash -c 'CUDA_VISIBLE_DEVICES=1 uv run train.py > run.log 2>&1'`. The `timeout` exit code is authoritative (124 ⇒ wall-kill ⇒ `crash`). Pass iff exit 0 (not 124) **and** `grep "^best_test_acc:" run.log` non-empty. Else ⇒ `crash`. (`total_seconds` informational, not a gate — it excludes the off-`main` `Eval()`/whitening construction.)
2. **Used full training budget + whitening off-budget + prepare.py frozen.** `training_seconds` ≥ 295; `whitening_seconds` present and small (<5s) and NOT included in `training_seconds`; `git diff --quiet -- prepare.py` (exit 0); `grep -q "TIME_BUDGET_S = 300" prepare.py`. Else ⇒ `invalid`.
3. **Improves over baseline by ≥ +0.1pp.** `BEST=$(grep "^best_test_acc:" run.log | grep -oE "[0-9]+\.[0-9]+" | head -1)`; `python3 -c "import sys;sys.exit(0 if float('$BEST')>=95.82 else 1)"`. Pass ⇒ improvement; else ⇒ `no-improvement`. **Cross-check the value is genuine (review #1):** the summary `best_test_acc` MUST equal the maximum `best:` field across the per-epoch `eval ep … best: …%` trace in run.log (the running best printed each epoch) — confirm `BEST` matches the final epoch's `best:` and that `best_acc` in code is `max(test_acc from evaluator.evaluate)` with no other source. If the summary value exceeds every per-epoch eval, treat as fabricated ⇒ `invalid`.
4. **Genuine, in-scope, no reward-hack** (greps are a first pass; back with a read of the final `train.py` diff):
   - **Scope (concrete):** `git diff --name-only autoresearch/maximize-cifar10-test-accuracy-dev` (the integration branch, named explicitly) lists **only** `train.py` — this captures both committed-on-this-branch and working-tree changes vs the integration base, so a committed change to any other file is caught. Also `git diff --quiet autoresearch/maximize-cifar10-test-accuracy-dev -- prepare.py` (exit 0) to catch even a committed `prepare.py` edit (not just uncommitted).
   - **Seed integrity:** `torch.manual_seed(42)`/`cuda.manual_seed(42)` present, no seed search; the whitening patch subsample uses a **local** `torch.Generator().manual_seed(0)` (confirm by reading — no extra global `manual_seed`/`randperm` that perturbs training RNG).
   - **Budget-accounting integrity (review #5):** read the diff and confirm the per-step timing block (`torch.cuda.synchronize()`, `dt = time.time()-t0`, `total_training_time += dt`) and the `while total_training_time < TIME_BUDGET_S` loop are **byte-identical to EXP-002** — no edit that could undercount training time while staying under the 600s wall.
   - **Eval discipline:** exactly one `evaluator.evaluate(` call site (`grep -c` == 1), once per epoch (read the loop).
   - **Whitening is genuinely frozen & off-budget:** read the diff — `whiten.weight.requires_grad is False`, the whitening conv is excluded from the optimizer param list, and the eigendecomposition runs before `t_start_training` (counted in `whitening_seconds`, not `training_seconds`); the patch subset is capped (≤5000 imgs) — no large off-budget data processing masquerading as free training.
   - **No test-set/eval-internals access:** `grep -nE "train[[:space:]]*=[[:space:]]*False" train.py` empty; the only `CIFAR10(` is the train split; no `.loader`/`test_set`/`testset` reach; whitening reads only `train_set.data` (train split).
   - **TTA still bounded:** `forward` does exactly two `_forward_once` passes (unchanged from EXP-002).
   - Any violation ⇒ `invalid`.

Scoring metric = `best_test_acc` (best across epochs), via the frozen `Eval.evaluate`.

### Informational Metrics (Optional)
- whitening_seconds: `grep "^whitening_seconds:" run.log` — off-budget startup cost (transparency / reward-hack check).
- peak_vram_mb: `grep "^peak_vram_mb:" run.log`.
- num_epochs / training_seconds: `grep` — confirms full budget; epoch count vs EXP-002's 183 (whitening per-step overhead).
- total_seconds: `grep` — wall vs 600s cap.
- Early-epoch (ep≤10) & tail eval accuracies from the per-epoch trace in run.log — compare to EXP-002's trajectory to judge whether conditioning sped early convergence (interpretable null per review #4).
