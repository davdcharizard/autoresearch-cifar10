# Plan EXP-041
**Created**: 2026-06-09
**Goal**: goals/improve-cifar10-test-accuracy.md
**Brainstorm**: brainstorm/brainstorm-041.md

## Summary
Add the **PolyLoss Poly-1** leading term to the training objective: `L = CE_with_LS + ε·(1 − p_t)`, with
`ε = 1.0` and `p_t` = softmax probability of the true (hard) class. Keep label smoothing 0.1 and
everything else in the recipe unchanged. This is a compute-free, convergence-neutral reshape of the loss
gradient (ε>0 amplifies the gradient on hard/low-`p_t` examples → a mild convergence accelerator) that
dodges all three established plateau walls and is the first probe of the one untouched compute-free axis
after 41 experiments — the objective's polynomial shape. Tested against the bar **96.32**.

## Baseline (from experiment index)
- best_test_acc baseline = **96.22%** (commit 6c417a4, EXP-012); bar = **96.32** (baseline + 0.1).
- Reference run shape: ~91 epochs, dt ~8ms/step (uncontended), final_test_loss ~0.195, params 4,299,866.

## Hypothesis
Adding `1.0·(1 − p_t)` to the CE+LS loss amplifies per-example gradients on hard examples → more effective
convergence within the fixed 300s/~91-ep budget → best_test_acc above 96.32 at throughput-neutral ~91 ep,
params unchanged. Honest most-likely outcome: within-noise (~96.0–96.3) — sibling objective tweaks were
null here (LS-down EXP-023, cosine head EXP-039) and label smoothing may partially cancel the poly term.
A clean null closes the objective-polynomial-shape sub-lever (ε=+2 noted as a possible follow-up).

## Milestones

### Milestone 1 — Code change implemented and passing local checks
- [ ] Add `EPSILON_POLY = 1.0` to the hyperparameter block (near LABEL_SMOOTHING, ~L27-28).
- [ ] Edit the loss (train.py L234-236): keep `ce = F.cross_entropy(outputs, targets,
      label_smoothing=LABEL_SMOOTHING)`, then add the Poly-1 term and combine:
      `pt = F.softmax(outputs, dim=1).gather(1, targets.unsqueeze(1)).squeeze(1)` ;
      `loss = ce + EPSILON_POLY * (1.0 - pt).mean()`.
- [ ] AST check: `uv run python -c "import ast; ast.parse(open('train.py').read()); print('OK')"`
- [ ] Numerical smoke check: `uv run python -c "..."` — build `ResNet(3,10,width_mult=4)`, forward a random
      (8,3,32,32) batch, compute the new loss; assert it is finite and ≈ CE+LS magnitude (CE≈2.3 + ε·(1−pt)
      where (1−pt)≈0.9 untrained → loss ≈ 3.2, finite).
- [ ] Diff scope check: `git diff --name-only` lists **only** `train.py`.

### Milestone 2 — Experiment running (uncontended GPU)
- [ ] Confirm a GPU is idle (`nvidia-smi`: util ~0%, mem <700MiB). NOTE: a separate autoresearch instance
      (`v2.9.5-gpt-5-5`) and/or another user's jobs intermittently occupy a GPU — pick a clean index.
- [ ] Launch `CUDA_VISIBLE_DEVICES=<idle> uv run train.py > run.log 2>&1` (background).
- [ ] Confirm `run.log` shows `Device: cuda`, `params: 4,299,866` (UNCHANGED), first eval line; early dt ~8ms.

### Milestone 3 — Run completed and verified
- [ ] Run exits 0, prints full summary block (`best_test_acc:` … `num_params:`).
- [ ] **Clean/fair run**: steady dt ~8ms, num_epochs ~88–94 (no contention; compute-neutral so epoch count
      should match baseline ~91 within jitter).
- [ ] Extract metrics, compare best_test_acc to bar 96.32 / baseline 96.22.
- [ ] Confirm clean completion (<600s wall, eval_count == num_epochs, only train.py changed, seed 42,
      num_params 4,299,866 unchanged).

## Code Changes

**File: `train.py` (loss objective — the ONLY change)**
- **Hyperparameter block (~L28)**, add:
  ```python
  EPSILON_POLY = 1.0  # PolyLoss Poly-1 leading-term coefficient (ICLR 2022); conservative low end of the
                      # paper's ImageNet ResNet +1..+2 range for the short CIFAR budget (EXP-041)
  ```
- **Loss (L232-236)**, change:
  ```python
  # before
  with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
      outputs = compiled_model(inputs)
      loss = F.cross_entropy(outputs, targets, label_smoothing=LABEL_SMOOTHING)
  # after
  with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
      outputs = compiled_model(inputs)
      ce = F.cross_entropy(outputs, targets, label_smoothing=LABEL_SMOOTHING)
      # PolyLoss Poly-1 (Leng et al., ICLR 2022): add ε·(1 - p_t), p_t = softmax prob of the true
      # class. ε>0 amplifies the gradient on hard (low-p_t) examples → mild convergence accelerator.
      pt = F.softmax(outputs, dim=1).gather(1, targets.unsqueeze(1)).squeeze(1)
      loss = ce + EPSILON_POLY * (1.0 - pt).mean()
  ```
- **Why it tests the hypothesis**: it changes ONLY the objective's gradient shape (the leading polynomial
  coefficient of CE), with no change to model, data, optimizer, schedule, eval, or seed — so any metric
  delta is attributable to the objective reshape. ε>0 emphasizes hard examples, directly probing the
  convergence-bound hypothesis. Distinct from the closed LS knob (EXP-023, which scaled the target
  distribution) and SAM (EXP-036, a compute-cost failure, not an objective-shape test).
- **Risks/edge cases**: (a) `softmax` under bf16 autocast is computed in fp32 internally (PyTorch autocast
  policy) → `pt` numerically stable; `gather`/`mean` are cheap and exact. No div-by-zero. (b) ε mistuned →
  over-amplifies hard/augmented examples → underfit/regress; ε=1.0 is conservative. (c) `loss.item()` at
  L245 still works (scalar). (d) The extra `softmax`+`gather` is compute-trivial (one op on the (128,10)
  logits already computed) — throughput-neutral; verify dt ~8ms / epochs ~91 anyway. (e) No new deps,
  params unchanged (4,299,866), eval untouched.

## Configuration Changes
One new hyperparameter `EPSILON_POLY = 1.0` and the loss formula (CE+LS → CE+LS + ε·(1−p_t)). All else
unchanged (PEAK_LR 0.2, batch 128, WD 1e-4, label smoothing 0.1, Cutout 16, TrivialAugmentWide,
cosine-to-0 LR, Nesterov m0.9, seed 42, 300s budget, torch.compile reduce-overhead, widths {64,128,256}).
ε=1.0 chosen as the conservative low end of the paper's reported +1..+2 ImageNet-ResNet range (no
published CIFAR value); large enough to perturb the objective, small enough to avoid destabilizing the
short-budget run.

## Execution Environment
- **Method**: local — `CUDA_VISIBLE_DEVICES=<idle_gpu> uv run train.py > run.log 2>&1` (background).
  Pick an idle H20 via `nvidia-smi`; the shared node intermittently saturates (another autoresearch
  instance + other users) and the budget is wall-clock-dt-gated (infra-errors). Relaunch on a clean
  window if contended.
- **Resources**: single NVIDIA H20 (either index); fixed `TIME_BUDGET_S=300`.
- **Estimated runtime**: ~390–420s wall (≈6.5–7 min), same shape as baseline (compute-neutral).
- **Log output**: stdout+stderr → `run.log` at project root (sole source of truth). Per-step lines use
  `\r`; extract dt via `tr '\r' '\n' < run.log | grep -oE "dt: [0-9]+ms"`.
- **Monitoring**: background Monitor firing on the final summary block / traceback / process exit.

## Abort Criteria
- Any Python traceback / non-zero exit, or NaN/inf in `loss:` → kill, mark failed.
- **GPU contention** (sustained dt band well above ~8ms / epoch count trending far below ~88): kill and
  relaunch on an idle window (infra workaround, not a research failure).
- Loss clearly diverging (debiased loss rising past mid-run, test_acc collapsing) → likely ε too large;
  let it finish if near done (informative) else kill.
- Total wall-clock of a single committed run approaching 10 min (600s) → kill, treat as failure.

## Verification Protocol

### Verification Procedure
Run after the committed clean run completes. Baseline = 96.22 (from `exp-index.sh baseline`).

1. **Primary metric clears the bar** (NECESSARY):
   - Command: `grep -aE "^best_test_acc:" run.log`
   - Pass iff `best_test_acc >= 96.32`. Else no-improvement.
2. **Clean completion within budget** (NECESSARY):
   - Command: `grep -aE "^best_test_acc:|^total_seconds:|^num_epochs:|^num_params:" run.log`
   - Pass iff summary block present, `total_seconds < 600`, exit 0.
3. **No hard-constraint violations** (NECESSARY):
   - `git diff --name-only` = train.py only; eval-line count == `num_epochs:` (≤1 eval/epoch); no new
     deps; seed 42; prepare.py/eval untouched; **num_params 4,299,866 UNCHANGED**.
   - **Fairness gate**: confirm the committed run was uncontended (steady dt ~8ms, num_epochs ~88–94);
     a contention-shortened run is invalid and must be re-run.
   - Timeout per command: 30s. Overall run timeout: 600s wall.

### Informational Metrics (Optional)
- `final_test_loss:` — note: PolyLoss CHANGES the train objective but eval still reports plain CE test
  loss (eval is frozen), so test_loss remains comparable to baseline 0.195 (convergence check).
- `num_epochs:` / steady dt — throughput/fairness check (expect ~91 ep / 8ms; compute-neutral).
- `peak_vram_mb:` — expect ≈ baseline 491.

## Expected Outcome / Decision
- **If `best_test_acc >= 96.32`** on a clean run: improvement — commit, merge to `autoresearch/dev`, PR.
- **If within-noise (~96.0–96.3) or below** at ~91 ep: no-improvement — the objective-polynomial-shape
  sub-lever doesn't move top-1 on this recipe (note ε=+2 as a possible one-shot follow-up before fully
  closing the objective axis).
- **If loss diverged / clear regression**: no-improvement — ε=1.0 too aggressive for the short budget;
  closes ε≥1 (a smaller ε<1 could be a follow-up, low priority).
