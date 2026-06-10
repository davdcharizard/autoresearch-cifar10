# Plan EXP-039
**Created**: 2026-06-09
**Goal**: goals/improve-cifar10-test-accuracy.md
**Brainstorm**: brainstorm/brainstorm-039.md

## Summary
Replace the plain linear+bias softmax classifier head with a **scaled-cosine (normalized-softmax)**
head: L2-normalize the penultimate (global-avg-pooled) features and the classifier weight rows, and
score by `scale · cosθ` with a fixed `scale = 16.0`. Everything else in the recipe is unchanged. This
imposes an angular decision geometry (top-1-affecting, not loss-only polish), is compute-neutral and
convergence-neutral, and is the one remaining lever that dodges all three established walls (compute,
polish-vs-top1, regularizer-underfit). Tested against the bar **96.32** (baseline 96.22 + 0.1).

## Baseline (from experiment index)
- best_test_acc baseline = **96.22%** (commit 6c417a4, EXP-012); bar = **96.32**.
- Reference run shape: ~91 epochs, dt ~8ms/step (uncontended), final_test_loss ~0.195, params 4,299,866.

## Hypothesis
A scaled-cosine head (feature+weight L2-normalized, scale 16) better-conditions class boundaries
(angular margin) and may improve generalization → best_test_acc above 96.32, at throughput-neutral
~91 ep, params essentially unchanged (−10, dropping the fc bias). Honest most-likely outcome:
within-noise (~96.0–96.3), since cosine heads give only small gains on balanced data and BN already
conditions the network. Key risk to verify: `scale` must be large enough for softmax+label-smoothing to
saturate — check final_test_loss is not inflated vs baseline 0.195 (inflation ⇒ under-confidence ⇒
scale too small).

## Milestones

### Milestone 1 — Code change implemented and passing local checks
- [ ] Edit `train.py` head: (a) L107 `self.fc = nn.Linear(w3, num_classes)` →
      `nn.Linear(w3, num_classes, bias=False)` and add `self.logit_scale = 16.0`; (b) forward L131-133
      replace `return self.fc(out)` with feature+weight L2-normalize and scaled-cosine logits.
- [ ] AST check: `uv run python -c "import ast; ast.parse(open('train.py').read()); print('OK')"`
- [ ] Instantiate + params + forward-shape:
      `uv run python -c "import torch, train; m=train.ResNet(3,10).eval();
      y=m(torch.randn(4,3,32,32)); print('out', tuple(y.shape), 'params', sum(p.numel() for p in m.parameters()))"`
      → expect out (4,10); params ≈ 4,299,856 (baseline 4,299,866 − 10 fc bias).
- [ ] Diff scope check: `git diff --name-only` lists **only** `train.py`.

### Milestone 2 — Experiment running (uncontended GPU)
- [ ] Confirm a GPU is idle (`nvidia-smi`: util ~0%, mem <700MiB, no foreign compute proc) — the
      shared H20 node is intermittently saturated by another user's jobs (infra-errors).
- [ ] Launch via the idle-GPU fair-run launcher (waits for a free GPU, verifies clean dt, accepts only
      a run reaching ≥~85 ep) so the wall-clock-dt-gated budget is fair.
- [ ] Confirm `run.log` shows `Device: cuda`, a `params:` line, first eval line; early dt ~8ms.

### Milestone 3 — Run completed and verified
- [ ] Run exits 0, prints full summary block (`best_test_acc:` … `num_params:`).
- [ ] **Clean/fair run**: dt ~8ms, num_epochs ~88–91 (no contention; if epochs materially low, re-run).
- [ ] Extract metrics, compare to bar 96.32 / baseline 96.22.
- [ ] Confirm clean completion (<600s wall, eval_count == num_epochs, only train.py changed, seed 42).

## Code Changes

**File: `train.py` (classifier head — the ONLY change)**
- **L107** (in `ResNet.__init__`):
  ```python
  # before
  self.fc = nn.Linear(w3, num_classes)
  # after
  self.fc = nn.Linear(w3, num_classes, bias=False)
  self.logit_scale = 16.0  # fixed cosine-softmax temperature (plain float, NOT a Parameter →
                           # not subject to weight decay; standard scale for ~10-class normalized softmax)
  ```
- **forward L131-133**:
  ```python
  # before
  out = F.adaptive_avg_pool2d(out, 1)
  out = out.view(out.size(0), -1)
  return self.fc(out)
  # after
  out = F.adaptive_avg_pool2d(out, 1)
  out = out.view(out.size(0), -1)
  feat = F.normalize(out, dim=1)                 # L2-normalize penultimate features
  w = F.normalize(self.fc.weight, dim=1)         # L2-normalize each class weight row
  return self.logit_scale * F.linear(feat, w)    # scaled cosine logits
  ```
- **Why it tests the hypothesis**: isolates the classifier SCORING geometry — angular (cosine) vs
  unconstrained linear — with no other recipe change, so any metric delta is attributable to the
  decision geometry. Preserves the coarse-to-fine feature hierarchy (unlike the closed multi-scale head
  EXP-032); touches only the final projection.
- **Risks/edge cases**: (a) `scale=16` mistuned → under/over-confident softmax; the diagnostic is
  final_test_loss vs baseline 0.195 (inflated ⇒ scale too small). (b) `logit_scale` as a plain float
  attribute is captured fine by `torch.compile`; it is NOT a Parameter so weight decay can't shrink it
  and it won't appear in `model.parameters()`. (c) `F.normalize` adds a 1e-12 eps internally (no div-by-0).
  No new deps, no eval-side change, seed unchanged. params −10 (fc bias dropped) — not a constraint
  (only train.py-only + budget + ≤1 eval/epoch are constrained).

## Configuration Changes
Classifier head only (linear+bias softmax → scaled-cosine, scale 16). All other hyperparameters
unchanged (PEAK_LR 0.2, batch 128, WD 1e-4, label smoothing 0.1, Cutout 16, TrivialAugmentWide,
cosine-to-0 LR, Nesterov m0.9, seed 42, 300s budget, torch.compile reduce-overhead, widths {64,128,256}).
scale=16 chosen as the standard normalized-softmax temperature for ~10 classes (large enough that
softmax+LS can saturate; small enough to avoid gradient saturation).

## Execution Environment
- **Method**: local — `CUDA_VISIBLE_DEVICES=<idle_gpu> uv run train.py > run.log 2>&1`, launched via the
  idle-GPU fair-run launcher (see Abort Criteria / infra-errors) because both shared H20s intermittently
  saturate with another user's Protenix jobs and the budget is wall-clock-dt-gated.
- **Resources**: single NVIDIA H20 (either index — identical hardware; pick whichever is idle); fixed
  `TIME_BUDGET_S=300` training compute.
- **Estimated runtime**: ~390–420s wall per clean attempt (≈6.5–7 min); plus launcher wait time if the
  node is busy.
- **Log output**: stdout+stderr → `run.log` at project root (sole source of truth). Per-step lines use
  `\r`; extract dt via `tr '\r' '\n' < run.log | grep -oE "dt: [0-9]+ms"`.
- **Monitoring**: background task notifies on launcher completion; on the committed clean run, tail
  run.log for errors + the final summary block.

## Abort Criteria
- Any Python traceback / non-zero exit, or NaN/inf in `loss:` → kill, mark failed.
- **GPU contention** (clean dt > ~13ms / epoch count trending well below ~85): the launcher early-aborts
  and retries on another idle window — NOT a research failure, an infra workaround (infra-errors).
- Total wall-clock of a single committed run approaching 10 min (600s) → kill, treat as failure.
- `final_test_loss` wildly high (e.g. > 0.30) with flat/low test_acc by mid-run → scale likely far too
  small (under-confidence); let it finish (informative) but expect no-improvement.

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
     deps; seed 42; prepare.py/eval untouched. (num_params 4,299,856 — −10 vs baseline, expected.)
   - **Fairness gate**: confirm the committed run was uncontended (dt ~8ms, num_epochs ~88–91); a
     contention-shortened run is invalid and must be re-run (not reported).
   - Timeout per command: 30s. Overall run timeout: 600s wall.

### Informational Metrics (Optional)
- `final_test_loss:` — scale-saturation diagnostic (expect ≈0.195 if scale OK; inflated ⇒ scale too
  small ⇒ under-confidence).
- `num_epochs:` / `num_steps:` — throughput/fairness check (expect ~91 ep / dt ~8ms; cosine head is
  compute-neutral).
- `peak_vram_mb:` — expect ≈ baseline.

## Expected Outcome / Decision
- **If `best_test_acc >= 96.32`** on a clean run: improvement — commit, merge to `autoresearch/dev`, PR.
- **If within-noise (~96.0–96.3) or below** at ~91 ep with loss ≈0.195: no-improvement — the cosine head
  doesn't move top-1 on this balanced/BN-conditioned net; closes the classifier-scoring sub-lever.
- **If loss inflated (scale too small)**: no-improvement, but note a scale sweep (e.g. 24/32) as a
  possible follow-up (not auto-scheduled) before fully closing the geometry axis.
