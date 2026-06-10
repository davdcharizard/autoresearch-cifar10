# Plan EXP-042: Deep supervision — auxiliary layer2 classifier with a decayed aux loss

- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-042.md

## Milestones

### Milestone 1: Implement deep-supervision changes in train.py
- [ ] Add `LAMBDA_AUX = 0.3` hyperparameter constant (decayed aux-loss weight at frac=0).
- [ ] Add `self.aux_fc = nn.Linear(w2, num_classes)` to `ResNet.__init__` (w2 = 32*k = 128).
- [ ] Modify `ResNet.forward` to compute aux logits from layer2's global-avg-pool and return
      `(main_logits, aux_logits)` when `self.training`, else just `main_logits`.
- [ ] Modify the training-loop loss: `loss = CE(main,LS) + λ(t)·CE(aux,LS)`, `λ(t)=LAMBDA_AUX·(1−frac)`.
- [ ] `ruff` clean; `python -c "import ast; ast.parse(open('train.py').read())"` parses.

### Milestone 2: Confirmed running on an idle GPU, dt is throughput-neutral
- [ ] Pick an idle GPU via `nvidia-smi` (fairness gate — both H20s can be contended by neighbors).
- [ ] Launch `CUDA_VISIBLE_DEVICES=<idle> uv run train.py > run.log 2>&1`.
- [ ] Within ~60s confirm: device prints, params ≈ 4,301,156 (+1290 vs 4,299,866), no traceback.
- [ ] Confirm steady-state dt ≈ 8ms (read dt lines). If dt rises to ~9ms it is borderline-fair
      (note epoch count); if dt ≥ ~11ms the run is compute-confounded → flag in observations.

### Milestone 3: Run completes within budget and prints summary
- [ ] Run prints the `best_test_acc:` summary block, exits 0, total_seconds < 600.
- [ ] One `eval ep` line per epoch (≤ 1 validation/epoch).

## Code Changes
- **train.py**:
  - **Add hyperparameter** (after `LABEL_SMOOTHING = 0.1`, ~L27): `LAMBDA_AUX = 0.3` with a comment
    that it is the deep-supervision auxiliary-loss weight at frac=0, decaying linearly to 0.
  - **`ResNet.__init__`** (after `self.fc = nn.Linear(w3, num_classes)`, ~L107): add
    `self.aux_fc = nn.Linear(w2, num_classes)`. (It receives the same Kaiming-normal init as `fc`
    via the existing `self.apply(self._weights_init)`.) Why: a lightweight classifier on mid-level
    (layer2, 128-ch, 16×16) features for auxiliary supervision; discarded at inference.
  - **`ResNet.forward`** (~L126-133): after `out = self.layer2(out)`, if `self.training` compute
    `aux = self.aux_fc(F.adaptive_avg_pool2d(out, 1).flatten(1))`; continue layer3→pool→`self.fc` to
    get `main`; `return (main, aux) if self.training else main`. Why: gives the frozen eval
    (`model.eval(); model(inputs)`) the UNCHANGED single-tensor main path, while training gets both
    heads. The branch is on the Python bool `self.training`, which `torch.compile` specializes
    (training always calls with `self.training=True`); eval uses the eager `model` handle.
  - **Training loop loss** (~L232-238): inside the autocast block,
    `main_out, aux_out = compiled_model(inputs)`;
    `loss_main = F.cross_entropy(main_out, targets, label_smoothing=LABEL_SMOOTHING)`;
    `lam = LAMBDA_AUX * (1.0 - total_training_time / TIME_BUDGET_S)`;
    `loss = loss_main + lam * F.cross_entropy(aux_out, targets, label_smoothing=LABEL_SMOOTHING)`.
    Why: `lam` uses the same elapsed-time fraction as `lr_at_fraction` (pre-step `total_training_time`),
    decaying 0.3→0 so the final iterates optimize the pure main objective (sidesteps regularizer
    underfit). `loss.item()` for the smoothed-loss display now reflects the combined training loss
    (informational only).
  - **Edge cases**: eval path returns a single tensor (verified — eval uses eager `model`, not
    `compiled_model`). No change to optimizer, schedule, augmentation, seed, batch, compile mode, or
    `evaluator.evaluate(...)`. Scope: train.py only.

## Configuration Changes
- LAMBDA_AUX: (new) -> 0.3 (GoogLeNet auxiliary-classifier weight; decayed to 0 over the budget so the
  main objective is pure at the end). Rationale: standard deep-supervision aux weight; decay avoids
  distorting the evaluated main head and the regularizer-underfit wall (project-insights L82, EXP-022).
- num_params: 4,299,866 -> ~4,301,156 (+1,290 from aux_fc; informational, no param-count constraint; VRAM soft).

## Execution Environment
- Method: local — `CUDA_VISIBLE_DEVICES=<idle> uv run train.py > run.log 2>&1` from the project root.
- Resources: single NVIDIA H20 (98GB). MUST be an idle GPU — the dt-gated 300s budget makes any
  contended run an unfair (under-epoch) test (see infra notes / EXP-041 Run 1 contention discard).
- Estimated runtime: ~300s training + ~40-70s startup/compile/eval ≈ 6-8 min wall (< 600s gate).
- Log output: redirect stdout+stderr to `run.log` (do NOT tee — keeps context clean). dt lines use `\r`;
  extract via `tr '\r' '\n' < run.log | grep -oE "dt: [0-9]+ms" | sort | uniq -c`.
- Tool skill: none (local run).

## Abort Criteria
- `loss` becomes NaN/inf or diverges (debiased smoothed loss climbing past ~1.0 after warmup).
- Steady-state dt ≥ ~11ms (compute-confounded — the aux backward cost would collapse epochs and
  invalidate the fair top-1 comparison): let it finish but flag as epoch-confounded in observations.
- No new output in run.log for > 2 min (hang).
- total_seconds trending past 600s — kill and treat as failure.
- A neighbor job saturates the chosen GPU mid-run (dt band jumps, e.g. ≥15ms): discard as contended,
  re-run on an idle GPU (fairness gate).

## Verification Protocol

### Verification Procedure
Baseline (from `exp-index.sh baseline` on `experiment-indices/improve-cifar10-test-accuracy.tsv`) = **96.22**;
bar = baseline + 0.1 = **96.32**.

1. **Primary metric (necessary)**: `grep -aE "^best_test_acc:" run.log`. Pass iff
   `best_test_acc >= 96.32`. Below that → no-improvement.
2. **Clean completion within budget (necessary)**:
   `grep -aE "^best_test_acc:|^total_seconds:|^num_epochs:|^num_steps:|^peak_vram_mb:" run.log`;
   confirm the summary block printed, exit code 0, and `total_seconds < 600`. Empty `best_test_acc`
   ⇒ crash (inspect `tail -n 50 run.log`).
3. **No constraint violations (necessary)**:
   - `git diff --stat` shows only `train.py` modified (prepare.py / eval harness untouched).
   - Validation ≤ 1/epoch: `grep -ac "eval ep" run.log` equals the `num_epochs:` value.
   - No new dependencies (no pyproject/imports changes beyond stdlib/torch already present), seed
     still 42 (`torch.manual_seed(42)` unchanged), no seed hacking.
4. Render verdict: improvement only if condition 1 AND 2 AND 3 all pass; else no-improvement (or
   invalid on a hard-constraint breach / untrustworthy result).
5. Remove `run.log` after recording metrics (keep the tree clean).

### Informational Metrics (Optional)
- peak_vram_mb: `grep -aE "^peak_vram_mb:" run.log` (soft-constraint awareness; expect a small rise).
- num_epochs / num_steps: `grep -aE "^num_epochs:|^num_steps:" run.log` (throughput-neutrality check vs
  baseline ~91 ep / ~8ms — the fair-test signal that distinguishes a real effect from an epoch confound).
- steady dt distribution: `tr '\r' '\n' < run.log | grep -oE "dt: [0-9]+ms" | sort | uniq -c`.
- final_test_loss: `grep -aE "^final_test_loss:" run.log` (watch for the polish-vs-top1 pattern —
  lower loss without top-1 gain).
