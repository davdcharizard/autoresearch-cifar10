# Plan EXP-043: AdamW optimizer-family swap

- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-043.md

## Milestones

### Milestone 1: Implement the AdamW swap in train.py
- [ ] `PEAK_LR`: 0.2 → 2e-3.
- [ ] `WEIGHT_DECAY`: 1e-4 → 0.05 (decoupled, AdamW).
- [ ] Replace the `optim.SGD(...)` constructor with `optim.AdamW(model.parameters(), lr=PEAK_LR,
      betas=(0.9, 0.999), eps=1e-8, weight_decay=WEIGHT_DECAY)`.
- [ ] `ruff check train.py` passes; `python -c "import ast; ast.parse(open('train.py').read())"` parses.

### Milestone 2: Confirmed running on an idle GPU, dt throughput-neutral
- [ ] Pick an idle GPU via `nvidia-smi` (fairness gate).
- [ ] Launch `CUDA_VISIBLE_DEVICES=<idle> uv run train.py > run.log 2>&1`.
- [ ] Within ~60s: device prints, params = 4,299,866 (UNCHANGED — optimizer swap adds no model params),
      no traceback, loss decreasing (not NaN/diverging — Adam early-step stability under 5% warmup).
- [ ] Confirm steady dt ≈ 8ms (AdamW's two `_foreach_` moment buffers are sub-ms). If dt rises, note it.

### Milestone 3: Run completes within budget and prints summary
- [ ] Run prints the `best_test_acc:` summary block, exits 0, total_seconds < 600.
- [ ] One `eval ep` line per epoch (≤ 1 validation/epoch).

## Code Changes
- **train.py**:
  - **`PEAK_LR = 0.2` → `PEAK_LR = 2e-3`** (~L23): AdamW's adaptive update needs a ~100× smaller LR than
    SGD. 2e-3 is a standard from-scratch AdamW peak for small image models with cosine+warmup. The
    `lr_at_fraction` schedule (5% warmup + cosine-to-0) scales this peak unchanged. Update the comment.
  - **`WEIGHT_DECAY = 1e-4` → `WEIGHT_DECAY = 0.05`** (~L26): AdamW uses decoupled weight decay, which is
    ~100–1000× larger than SGD's coupled WD; 0.05 is the standard decoupled value. Update the comment.
  - **Optimizer constructor** (~L192-198): replace
    `optim.SGD(model.parameters(), lr=PEAK_LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY, nesterov=True)`
    with `optim.AdamW(model.parameters(), lr=PEAK_LR, betas=(0.9, 0.999), eps=1e-8,
    weight_decay=WEIGHT_DECAY)`. Why: tests the optimizer-family hypothesis (adaptive per-parameter steps
    vs SGD+Nesterov) under an otherwise-identical recipe. `MOMENTUM` (L25) becomes unused — leave it
    defined (module-level constant, ruff does not flag it) to minimize the diff.
  - **Edge cases**: AdamW maintains two moment buffers per param (≈2× param memory in optimizer state) →
    small VRAM rise (soft constraint, ample headroom). The per-step LR set in the loop
    (`pg["lr"] = lr`) already drives every param group, so the schedule applies to AdamW unchanged. No
    change to model/data/augmentation/schedule-shape/seed/compile/eval. Scope: train.py only.

## Configuration Changes
- PEAK_LR: 0.2 → 2e-3 (AdamW-appropriate peak; cosine+warmup schedule unchanged).
- WEIGHT_DECAY: 1e-4 → 0.05 (decoupled AdamW weight decay; standard value).
- optimizer: SGD(momentum=0.9, nesterov) → AdamW(betas=(0.9,0.999), eps=1e-8).
- Rationale: literature-standard AdamW-from-scratch config (Loshchilov & Hutter 2019; timm practice) to
  give the optimizer family a fair shot, so a regression is a real verdict rather than a tuning miss.
- num_params: UNCHANGED at 4,299,866 (optimizer swap adds no model parameters).

## Execution Environment
- Method: local — `CUDA_VISIBLE_DEVICES=<idle> uv run train.py > run.log 2>&1` from the project root.
- Resources: single NVIDIA H20 (98GB). MUST be an idle GPU (dt-gated budget → contended runs are unfair).
- Estimated runtime: ~300s training + ~40-70s startup/compile/eval ≈ 6-8 min wall (< 600s gate).
- Log output: redirect stdout+stderr to `run.log` (no tee). dt lines use `\r`; extract via
  `tr '\r' '\n' < run.log | grep -oE "dt: [0-9]+ms" | sort | uniq -c`.
- Tool skill: none (local run).

## Abort Criteria
- `loss` becomes NaN/inf or diverges (debiased smoothed loss climbing past ~1.0 after warmup) — a real risk
  with adaptive optimizers if the LR is too high; if it diverges, this is a research result (AdamW@2e-3
  unstable here), NOT a retry — let it finish or kill if clearly NaN, and report.
- Steady-state dt ≥ ~11ms (compute-confounded) — let it finish but flag as epoch-confounded.
- No new output in run.log for > 2 min (hang); total_seconds trending past 600s — kill, treat as failure.
- A neighbor job saturates the chosen GPU mid-run (dt band jumps ≥15ms) — discard as contended, re-run idle.

## Verification Protocol

### Verification Procedure
Baseline (from `exp-index.sh baseline` on `experiment-indices/improve-cifar10-test-accuracy.tsv`) = **96.22**;
bar = baseline + 0.1 = **96.32**.

1. **Primary metric (necessary)**: `grep -aE "^best_test_acc:" run.log`. Pass iff `best_test_acc >= 96.32`;
   below → no-improvement.
2. **Clean completion within budget (necessary)**:
   `grep -aE "^best_test_acc:|^total_seconds:|^num_epochs:|^num_steps:|^peak_vram_mb:" run.log`; confirm the
   summary block printed, exit 0, `total_seconds < 600`. Empty `best_test_acc` ⇒ crash (`tail -n 50 run.log`).
3. **No constraint violations (necessary)**:
   - `git diff --stat` shows only `train.py` modified.
   - Validation ≤ 1/epoch: `grep -ac "eval ep" run.log` equals the `num_epochs:` value.
   - No new dependencies (AdamW is in torch.optim — no new deps); seed still 42; no seed hacking.
4. Verdict: improvement only if 1 AND 2 AND 3 all pass; else no-improvement (or invalid on a hard-constraint
   breach / untrustworthy result).
5. Remove `run.log` after recording metrics.

### Informational Metrics (Optional)
- peak_vram_mb: `grep -aE "^peak_vram_mb:" run.log` (expect a small rise from AdamW's 2 moment buffers).
- num_epochs / num_steps: `grep -aE "^num_epochs:|^num_steps:" run.log` (throughput-neutrality vs baseline ~91).
- final_test_loss: `grep -aE "^final_test_loss:" run.log` (watch the polish-vs-top1 pattern).
- dt distribution: `tr '\r' '\n' < run.log | grep -oE "dt: [0-9]+ms" | sort | uniq -c`.
