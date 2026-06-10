# Plan EXP-022: WRN-style dropout in the residual blocks (p=0.1)
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-022.md

## Milestones

### Milestone 1: Code change implemented and passes local checks
- [ ] On the EXP-012 baseline train.py, add a `DROPOUT_P = 0.1` constant near the other hyperparameters.
- [ ] In `BasicBlock.__init__`, add `self.dropout = nn.Dropout(p=DROPOUT_P)`.
- [ ] In `BasicBlock.forward`, insert the dropout between the two convs: after `out = F.relu(self.bn1(self.conv1(x)))`, add `out = self.dropout(out)`, before `out = self.bn2(self.conv2(out))`.
- [ ] `uv run ruff check train.py` clean; `python -c "import ast; ast.parse(open('train.py').read())"` parses.
- [ ] `git diff --name-only` shows only `train.py`.

### Milestone 2: Experiment runs and is confirmed healthy
- [ ] Launch `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` (background). Confirm clean startup: `params: 4,299,866` (UNCHANGED — dropout has no params), clean compile, no traceback, no NaN.
- [ ] Confirm throughput-near-neutral (~8ms/step, ~15k img/s, ~88-92 epochs) — dropout is one cheap elementwise mask per block under torch.compile.

### Milestone 3: Run completes and metrics extracted
- [ ] Run exits 0, prints summary block, `total_seconds < 600`.
- [ ] Extract `best_test_acc`, `final_test_loss`, `num_epochs`, `total_seconds`, `peak_vram_mb` from run.log.

## Code Changes
- **train.py** (the ONLY file modified):
  1. Add constant (near L28, with the other hyperparameters): `DROPOUT_P = 0.1  # WRN in-block dropout (Zagoruyko & Komodakis 2016); mild value for the ~92-epoch budget`.
  2. `BasicBlock.__init__` (after `self.bn2 = ...`, before the shortcut block): add `self.dropout = nn.Dropout(p=DROPOUT_P)`.
  3. `BasicBlock.forward`: change
     ```python
     out = F.relu(self.bn1(self.conv1(x)))
     out = self.bn2(self.conv2(out))
     ```
     to
     ```python
     out = F.relu(self.bn1(self.conv1(x)))
     out = self.dropout(out)
     out = self.bn2(self.conv2(out))
     ```
  - **Why this tests the hypothesis**: places dropout exactly where the WRN paper prescribes (between the two 3×3
    convs, after the first ReLU), regularizing intermediate features — a regularization LOCUS no prior experiment on
    this goal has touched. Tests whether reducing feature co-adaptation in the wide layers lifts top-1.
  - **Risks/edge cases**: (a) under-fit at the short budget (heavily-regularized recipe + 92 ep vs WRN's 200) → loss
    rises, acc drops (graceful no-improvement; mitigated by mild p=0.1). (b) `nn.Dropout` is core torch — no new dep.
    (c) Eval correctness: the frozen `Eval.evaluate()` calls `model.eval()`, so dropout becomes identity at eval
    automatically (the training loop also calls `model.train()` each epoch) — no eval contamination, no manual toggling
    needed. (d) torch.compile handles `nn.Dropout` natively (RNG-aware); no graph break expected.

## Configuration Changes
- DROPOUT_P: (new) **0.1**. Rationale: WRN paper uses ~0.3 for 200-epoch schedules; at our ~92-epoch budget on an
  already heavily-regularized recipe (TA+Cutout+LS+WD), a mild 0.1 adds feature regularization without the
  under-fitting that sank strong-aug CutMix (EXP-018). If it gains, a follow-up sweeps p up.
- Unchanged: full EXP-012 recipe (k=4 4.3M params, PEAK_LR 0.2 cosine-to-0, batch 128, Nesterov, WD 1e-4, LS 0.1,
  TrivialAugment + Cutout(16), torch.compile reduce-overhead, bf16, channels_last, seed 42).

## Execution Environment
- Method: local — `cd <project-root> && CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`, background launch.
- Resources: 1× NVIDIA H20 (98GB); ~0.45GB VRAM expected (dropout adds negligible memory).
- Estimated runtime: ~390–420s total wall-clock (300s training + per-epoch eval), well under 600s.
- Log output: all stdout/stderr → `run.log` (source of truth).
- Tool skill: none (local run).

## Abort Criteria
- Loss NaN/inf or diverging.
- Traceback / crash at startup (e.g. compile graph break on dropout) — fix code error, counts as one retry.
- No new log output for > 3 minutes (silent hang).
- `params` ≠ 4,299,866 at startup (should be impossible — dropout adds no params).
- total wall-clock approaching 600s — kill and treat as failure.

## Verification Protocol

### Verification Procedure
Baseline (from `exp-index.sh baseline`) = **96.22%**; success bar = **96.32%** (+0.1pp).

1. **Baseline**: `bash "/SPXvePFS/users/david/Deoxys/plugins/autoresearch/skills/shared/scripts/exp-index.sh" baseline "/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-opus-4-8/.autoresearch/experiment-indices/improve-cifar10-test-accuracy.tsv"` → confirm 96.22.
2. **Cond 1 — primary metric clears bar**: `grep -aE "^best_test_acc:" run.log` → PASS iff `best_test_acc >= 96.32`. (Decisive; run finishes ≤ ~7 min.)
3. **Cond 2 — clean completion within budget**: `best_test_acc` and `total_seconds` present; `grep -ac "Traceback" run.log` == 0; `total_seconds < 600`.
4. **Cond 3 — no constraint violations**: `git diff --name-only` = train.py only; `num_params` == 4,299,866; eval-count (`grep -ac "eval ep" run.log`) == `num_epochs` (one evaluate()/epoch); no new deps (nn.Dropout is core torch); seed 42 intact.
5. Compare and render verdict. Empty `best_test_acc` ⇒ crashed (`tail -n 50 run.log`).
6. Remove `run.log` before the next experiment.

### Informational Metrics (Optional)
- peak_vram_mb: `grep -aE "^peak_vram_mb:" run.log`
- num_epochs / num_steps: `grep -aE "^num_epochs:|^num_steps:" run.log` — watch for a drop vs baseline 91 (would signal compile-graph cost, à la EXP-015).
- final_test_loss: `grep -aE "^final_test_loss:" run.log` — KEY diagnostic vs baseline 0.195. Loss DOWN+acc UP = dropout helps; loss UP = under-fit/over-regularized at this budget (mechanism closed at p≥0.1 for this budget).
