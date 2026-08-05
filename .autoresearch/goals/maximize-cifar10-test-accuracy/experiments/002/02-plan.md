# Plan EXP-002: Weight EMA + flip-TTA on DavidNet
- **Created**: 2026-06-28

Chosen idea: `01-brainstorm.md` § Chosen Idea (Idea-02, Codex review pick), refined. Base: current `train.py` (EXP-001 DavidNet @ **95.22%**). Bar: **≥ 95.32%** (+0.1pp). Only `train.py` edited; `prepare.py` frozen. The proven training recipe stays byte-identical — we add an EMA of weights (evaluated each epoch in place of the raw iterate) and eval-time horizontal-flip TTA gated to the final 20% of training.

## Milestones

### Milestone 1: Implement EMA + flip-TTA in train.py
- [ ] Import `from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn`.
- [ ] Add `self.tta = False` to `ResNet9.__init__`; refactor `forward` into `_forward_once` + a `forward` that does flip-TTA only when `not self.training and self.tta` (averages logits of `x` and `x.flip(-1)`).
- [ ] Build `ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(EMA_DECAY), use_buffers=True).to(device, memory_format=torch.channels_last)` after the raw model is built. Constants: `EMA_DECAY=0.998`, `EMA_WARMUP_FRAC=0.15`, `TTA_START_FRAC=0.8`.
- [ ] In the loop after `optimizer.step()`: `if progress >= EMA_WARMUP_FRAC: ema_model.update_parameters(model); ema_started=True`.
- [ ] At per-epoch eval: choose `eval_target = ema_model if ema_started else model`; set `eval_target_inner.tta = (progress >= TTA_START_FRAC)` on the module that runs forward (`ema_model.module` or `model`); call `evaluator.evaluate(eval_target, device)` exactly once. Keep `best_acc` tracking.
- [ ] Keep everything else byte-identical: architecture, time-based one-cycle, SGD+Nesterov, wd 5e-4, **LABEL_SMOOTHING=0.2 unchanged**, Cutout 8, bf16+channels_last, batch 512, seed 42, budget meter, summary prints, per-epoch `wall:` print.
- [ ] `uv run python -m py_compile train.py`.
- [ ] **Smoke test** (before official run, in a scratch `python -c`/temp script, NOT committed): import OK; build `model` + `ema_model`; `ema_model.update_parameters(model)`; set `ema_model.module.tta=True` and `model.tta=True`; then exercise the **real frozen path** — `from prepare import Eval; ev = Eval(); a = ev.evaluate(ema_model, device); b = ev.evaluate(model, device)` — assert both return finite `(loss, acc)` floats in `[0,100]` and that after the call `ema_model.module.training is False`. This runs Eval's actual DataLoader + `.eval()` + CE-loss + `argmax`, confirming TTA-gated `forward` is reachable through `model(inputs)` for both the wrapped and raw models. (Eval downloads/uses the cached CIFAR test set; ~1–2s.)

### Milestone 2: Run the experiment
- [ ] Clean tree, remove stale `run.log`.
- [ ] Launch under wall guard: `timeout 600 bash -c 'CUDA_VISIBLE_DEVICES=1 uv run train.py > run.log 2>&1'`. Exit 124 ⇒ wall-kill ⇒ failure.
- [ ] Monitor epoch-1 loss (divergence) and the per-epoch `wall:` print vs 600s (TTA doubles eval only in the last ~20% of epochs).

### Milestone 3: Verify
- [ ] On completion, grep summary; apply Verification Protocol; record; remove `run.log`.

## Code Changes

- **train.py** (additive; training dynamics untouched):

  1. **Import** (top, with other torch imports): `from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn`.

  2. **Constants** (hyperparameter block): `EMA_DECAY = 0.998`, `EMA_WARMUP_FRAC = 0.15`, `TTA_START_FRAC = 0.8`.

  3. **`ResNet9` — TTA-gated forward.** Add `self.tta = False` at end of `__init__`. Replace `forward`:
     ```python
     def _forward_once(self, x):
         x = self.prep(x); x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
         x = self.pool(x).flatten(1)
         return self.fc(x) * self.scale_out

     def forward(self, x):
         if self.training or not self.tta:
             return self._forward_once(x)
         return 0.5 * (self._forward_once(x) + self._forward_once(x.flip(-1)))
     ```
     During training `self.training is True` → always single forward (no TTA cost, no BN-stat corruption). `x.flip(-1)` mirrors width (W is last axis). Averaging logits matches airbench `infer_mirror`; `scale_out=0.125` keeps logits small so the eval CE loss stays well-behaved.

  4. **Build EMA model** after `model = ResNet9(...).to(device, memory_format=channels_last)` and optimizer:
     ```python
     ema_model = AveragedModel(
         model, multi_avg_fn=get_ema_multi_avg_fn(EMA_DECAY), use_buffers=True
     ).to(device, memory_format=torch.channels_last)
     ema_started = False
     ```
     `use_buffers=True` EMA-averages BN running stats inline with the same decay (no off-budget `update_bn` pass). First `update_parameters` copies the raw params/buffers (seeds EMA at warmup, not from kaiming init).

  5. **EMA update** in the loop, immediately after `optimizer.step()`:
     ```python
     if progress >= EMA_WARMUP_FRAC:
         ema_model.update_parameters(model)
         ema_started = True
     ```
     `progress` is the same time-based value already computed at the top of the step. `update_parameters` is a foreach `lerp_` over ~6.5M params+buffers — a few fused CUDA kernels, <~1ms, only after 15% warmup. Budget is training-time gated so this cannot overrun 300s (just ~190 vs 192 epochs).

  6. **Per-epoch eval — evaluate the EMA model, TTA-gated to the tail.** Replace the eval line:
     ```python
     progress = min(1.0, total_training_time / TIME_BUDGET_S)
     if ema_started:
         ema_model.module.tta = (progress >= TTA_START_FRAC)
         eval_target = ema_model
     else:
         model.tta = (progress >= TTA_START_FRAC)   # progress<0.15 here, so False
         eval_target = model
     test_loss, test_acc = evaluator.evaluate(eval_target, device)
     ```
     Exactly one `evaluator.evaluate` per epoch (constraint satisfied). `evaluator.evaluate` calls `eval_target.eval()` (propagates `training=False` to `ema_model.module`) then `eval_target(inputs)` → `AveragedModel.forward` → `self.module.forward` (the TTA-gated ResNet9). After eval, the next epoch's `model.train()` restores the raw model to train mode; `ema_model` is only ever updated via `update_parameters`, never backprop'd.

     **Honest scope of what is measured (no false floor claim):** after warmup the scored metric `best_acc` is the best over **EMA-evaluated** epochs — the raw iterate is *not* separately evaluated (the ≤1-eval/epoch constraint forbids a second eval), so the EXP-001 95.22% is **not** a guaranteed floor of this run. The training *dynamics* are byte-identical to EXP-001, so the raw model still traverses the same ~95.22%-reaching trajectory internally; we are deliberately betting that the short-horizon EMA (denoised low-LR-tail average) + flip-TTA evaluates **at least as high** as that raw peak. If the EMA lags and the run scores below 95.32, that is a legitimate **`no-improvement`** outcome of testing this hypothesis — never `invalid`. This is the cleanest single-eval test of the EMA+TTA idea; floor-preserving alternatives (TTA-only, or raw-until-tail-then-EMA) are explicitly deferred to a **separate future experiment** rather than smuggled in as an in-run fallback.

     **BN buffers under EMA (known, accepted method):** `use_buffers=True` makes `update_parameters` EMA-average BN `running_mean`/`running_var` with the same 0.998 decay — this is the standard EMA-of-BN-stats approach (timm `ModelEmaV2`, PyTorch SWA-EMA docs), not a post-hoc `update_bn` recompute over the averaged weights. The two can differ slightly; the EMA-of-stats variant is the validated, on-budget choice (a recompute pass would consume training time). If accuracy regresses, mismatched BN stats are a specific candidate cause to examine in analysis (distinct from "decay lagged").

  7. **Summary prints**: unchanged; `num_params` still from the raw `model`. (Optionally print `EMA_DECAY` for the log.)

## Configuration Changes
- EMA of weights: none → `AveragedModel(decay=0.998, use_buffers=True)`, started at 15% progress, evaluated each epoch.
- Flip-TTA: none → eval-time logit averaging over x and x.flip(-1), gated to `progress >= 0.8`.
- Everything else (architecture, schedule, optimizer, wd, **label smoothing 0.2**, Cutout, precision, batch, seed): **unchanged** from EXP-001 (clean attribution; keeps the raw training trajectory identical so EMA averages the same path that reached 95.22% — see § Code Changes 6 for why this is not a *guaranteed* scored floor).

## Execution Environment
- Method: local, wall-guarded — `timeout 600 bash -c 'CUDA_VISIBLE_DEVICES=1 uv run train.py > run.log 2>&1'` from project root.
- Resources: single H20, **GPU 1** (GPU 0 in use). VRAM ≈ EXP-001's 1.6 GB + an EMA copy of ~6.5M params (negligible) — well within 98 GB.
- Estimated runtime: ~300s training + eval overhead. TTA doubles forward only in the last ~20% of epochs (~38 of ~190); EXP-001 total wall was 447s, so estimate **~460–500s** (< 600s cap). Per-epoch `wall:` print tracks it live.
- Log output: `run.log` (redirect, no tee). Extract: `grep "^best_test_acc:\|^peak_vram_mb:\|^training_seconds:\|^total_seconds:\|^num_epochs:" run.log`.
- Tool skill: none (local).

**Discipline:** an abort means *kill and mark failed* — it does NOT mean mutate the recipe and rerun. The reviewed experiment is exactly "EMA(0.998) + tail-gated flip-TTA, everything else EXP-001-identical." Any change to EMA decay/warmup, TTA gating, or batch is a **different experiment** and belongs to a later loop, not a within-run patch (avoids post-hoc variant selection). The only in-run retry permitted is for a pure infrastructure crash (see last bullet).

- **Divergence:** epoch-1 smoothed loss NaN/inf or not decreasing → kill, mark `failed`. The training recipe is unchanged from the working base, so this would indicate an EMA/forward **wiring bug** — fix the bug (still the same experiment) and rerun once; if it recurs, stop and report.
- **Wall-clock:** `timeout` exit 124, or per-epoch `wall:` trending toward 600s → kill, mark `failed`. Do NOT silently drop EMA/TTA to fit the budget — that would be a different experiment. (Mitigation is already designed in: TTA is gated to the final 20% of epochs.)
- **EMA regression signal (monitor only, never auto-abort or patch):** if the per-epoch EMA eval trails the EXP-001 trajectory (e.g. < ~94% deep into training), let the run finish — best-across-epochs still scores, and a sub-bar result is a valid `no-improvement`. Record the observation; a decay-0.997 or TTA-only variant is a *next-loop* experiment.
- **Crash / no summary / OOM:** grep empty after exit → `tail -n 50 run.log`. A genuine infra crash (OOM, CUDA/driver error, killed worker) may be retried per the execute skill's infra-retry policy; an OOM retry at a smaller batch is an infra workaround, and if it changes results it must be flagged as a recipe deviation in analysis rather than reported as the planned experiment.

## Verification Protocol

### Verification Procedure
Baseline (from `exp-index.sh baseline`): **95.22%**; bar **≥ 95.32%**. Conditions in order; first failure ⇒ `no-improvement` (or `invalid`/`crash`):

1. **Runs clean within the wall guard.** `timeout 600 bash -c 'CUDA_VISIBLE_DEVICES=1 uv run train.py > run.log 2>&1'`. The **`timeout` exit code is the authoritative wall guard** (exit 124 ⇒ wall-kill ⇒ `crash`), because `total_seconds` is timed from inside `main()` and excludes the top-level `Eval()` test-set construction. Pass iff exit 0 (not 124) **and** `grep "^best_test_acc:" run.log` non-empty. Else ⇒ `crash`. (`total_seconds` is recorded as informational, not a gate.)
2. **Used full training budget + prepare.py unchanged.** `training_seconds` ≥ 295; `git diff --quiet -- prepare.py` (exit 0); `grep -q "TIME_BUDGET_S = 300" prepare.py`. Else ⇒ `invalid`.
3. **Improves over baseline by ≥ +0.1pp.** `BEST=$(grep "^best_test_acc:" run.log | grep -oE "[0-9]+\.[0-9]+" | head -1)`; `python3 -c "import sys;sys.exit(0 if float('$BEST')>=95.32 else 1)"`. Pass ⇒ improvement; else ⇒ `no-improvement`.
4. **Genuine, in-scope, no reward-hack** (greps are a *first pass*; back them with a deliberate read of the final `train.py` diff — positional/aliased leakage can evade any single pattern):
   - **Scope:** `git diff --name-only <integration_branch>` lists only `train.py`.
   - **Seed integrity:** `torch.manual_seed(42)` present; no seed loop/search (`grep -nE "manual_seed|seed" train.py` reviewed by eye — only the single fixed seed).
   - **Eval discipline:** exactly one `evaluator.evaluate(` call site (`grep -c "evaluator.evaluate(" train.py` == 1), invoked once per epoch (read the loop to confirm it is not inside the step loop).
   - **No test-set/eval-internals access:** read the diff and confirm train.py never constructs the test split nor touches Eval internals — `grep -nE "train[[:space:]]*=[[:space:]]*False" train.py` empty (catches spaced variants); every `torchvision...CIFAR10(`/`datasets.CIFAR10(` instantiation in train.py is the **train** split (verify by reading each call, since the kwarg may sit on a following line); `grep -nE "evaluator\.|\.loader|test_set|testset|test_loader" train.py` shows only the sanctioned `evaluator = Eval()` construction and the one `evaluator.evaluate(` call — nothing reaching into the loader/dataset.
   - **TTA compute bounded:** confirm by reading `forward` that eval-time TTA does **exactly two** `_forward_once` passes (original + single `x.flip(-1)`), with **no loop / no multi-crop / no extra augmentations** hidden in `forward` — a single `evaluator.evaluate` call does not by itself bound per-image inference cost.
   - Any violation ⇒ `invalid`.

Scoring metric = `best_test_acc` (best across epochs), reported by the frozen `Eval.evaluate`.

### Informational Metrics (Optional)
- peak_vram_mb: `grep "^peak_vram_mb:" run.log` — VRAM (EMA copy overhead).
- num_epochs / training_seconds: `grep` — confirms full budget; epoch count vs EXP-001's 192 (EMA overhead).
- total_seconds: `grep` — wall vs 600s cap (TTA eval overhead).
- (Qualitative) per-epoch eval trace in run.log — EMA vs EXP-001 trajectory, and the tail bump when TTA engages at progress≥0.8.
