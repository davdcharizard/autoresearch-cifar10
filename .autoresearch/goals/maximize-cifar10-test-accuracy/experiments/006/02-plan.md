# Plan EXP-006: Multi-crop translation TTA (airbench96 tta_level=2)
- **Created**: 2026-06-28

## Milestones

### Milestone 1: Code change implemented + local checks pass
- [ ] Edit `ResNet9.forward` in `train.py` to replace mirror-only TTA with airbench96 mirror+translate TTA (6 views). No other line in `train.py` changes.
- [ ] `uv run python -c "import py_compile; py_compile.compile('train.py', doraise=True)"` → clean compile.
- [ ] `git diff --name-only autoresearch/maximize-cifar10-test-accuracy-dev -- ` lists ONLY `train.py`; `git diff autoresearch/maximize-cifar10-test-accuracy-dev -- train.py` content is confined to the `ResNet9.forward` body (no change to any **training-affecting** code: HPs, schedule, EMA, whitening, architecture `__init__`, optimizer, augmentation, training loop, or eval cadence).
- [ ] In-process smoke (CPU ok): build `ResNet9`, load whitening, `.eval()`, set `.tta=True`; run a `[8,3,32,32]` batch → assert output shape `[8,10]`, finite; assert the TTA forward issued exactly 6 `_forward_once` calls (monkeypatch a counter) and that `tta=False` issues 1; assert `model.training=True` path returns the single-forward result unchanged (training trajectory preserved).

### Milestone 2: Official run completes within wall guard
- [ ] Launch `timeout 600 bash -c 'CUDA_VISIBLE_DEVICES=1 uv run train.py > run.log 2>&1'` (GPU 1), capture PID, `wait`, record exit code.
- [ ] `run.log` contains exactly one `^best_test_acc:` summary line; process exit code is 0 (not 124).

### Milestone 3: Verification
- [ ] Run the Verification Procedure (C1→C2→C3, stop at first failure).

## Code Changes
- **`train.py` — `ResNet9.forward` only** (currently lines 180-185): replace the 2-view mirror TTA branch with airbench96 `tta_level=2` (mirror + diagonal-shift translation), keeping the existing `if self.training or not self.tta: return self._forward_once(x)` fast path verbatim. New TTA branch:
  ```python
  def forward(self, x):
      # Training (and eval before the TTA tail) uses a single forward. In eval
      # with TTA enabled, use airbench96 tta_level=2 (Keller Jordan,
      # arXiv:2404.00498): average logits over the mirror pair AND two
      # diagonal-shift translation crops (6 views total).
      if self.training or not self.tta:
          return self._forward_once(x)

      def mirror(v):
          return 0.5 * (self._forward_once(v) + self._forward_once(v.flip(-1)))

      h, w = x.shape[-2], x.shape[-1]
      padded = F.pad(x, (1, 1, 1, 1), mode="reflect")
      logits = mirror(x)
      logits_translate = 0.5 * (
          mirror(padded[:, :, 0:h, 0:w]) + mirror(padded[:, :, 2 : 2 + h, 2 : 2 + w])
      )
      return 0.5 * logits + 0.5 * logits_translate
  ```
  - **Why this tests the hypothesis**: the diagnosed limiter is incomplete eval-time view coverage — we currently use 2 of the 6 record views. This adds the two translation views (the documented, non-additive multi-crop component) on top of the mirror pair we already have, with zero change to the trained weights.
  - **Exactness**: pinned verbatim to upstream `airbench96.py` (fetched 2026-06-28): `F.pad(..., 'reflect')` with pad=1; crops `[0:h,0:w]` (shift −1,−1) and `[2:2+h,2:2+w]` (shift +1,+1); each mirror-averaged; final `0.5*mirror + 0.5*translate`. Using `h,w` from the tensor (= 32 for CIFAR eval) instead of hardcoding 32 — behaviorally identical on this dataset, robust to shape.
  - **Risks/edge cases**: (a) crops are non-contiguous views of the padded tensor; `_forward_once`'s first op is the whitening `Conv2d`, which accepts non-contiguous input (cuDNN handles it) — no correctness issue, at most a negligible kernel-selection cost on the off-budget eval path. (b) `F` is already imported (`torch.nn.functional as F`, line 7). (c) Reflect-pad operates on the already-normalized eval tensor (correct space — same tensor the conv stack consumes). (d) `ema_model.module.tta` is the attribute toggled by the loop (line 343); `AveragedModel.forward` delegates to `module.forward`, so the new TTA logic is exercised for the EMA eval target.

## Configuration Changes
- None. All hyperparameters, schedule, EMA, whitening, architecture, batch size, `TTA_START_FRAC=0.8`, and the training loop are unchanged. This is an **eval-only** change → the 300s training trajectory is byte-identical to the EXP-004 96.00 baseline (same seeds, same throughput, so ~142 epochs as in EXP-004).

## Execution Environment
- Method: local, single process. `timeout 600 bash -c 'CUDA_VISIBLE_DEVICES=1 uv run train.py > run.log 2>&1'` from project root.
- Resources: 1× NVIDIA H20, **GPU 1** (`CUDA_VISIBLE_DEVICES=1` — GPU 0 in use, hard constraint). VRAM ~1.6 GB (TTA adds transient activations for 32×32 crops; well within budget).
- Estimated runtime: ~500–540s wall. Training 300s fixed; eval is off-budget. Anchor: EXP-004 ran 445.2s total at 142 epochs with 2-view TTA on the final ~20%. Tripling the view count (2→6) on those ~28 epochs adds an estimated ~50–95s of off-budget eval → ~495–540s total. **< 600s with margin.**
- Log output: `run.log` (redirected). Summary block parsed for `best_test_acc`, `training_seconds`, `total_seconds`, `num_epochs`, `peak_vram_mb`.
- Tool skill: none (local run).

## Abort Criteria
- `timeout` returns exit 124 (wall ≥ 600s) → wall-clock failure; treat as failed run (the headline risk per the idea review). If diagnosed as TTA eval cost, the documented fallback is a **re-scoped eval-only re-run** that also raises `TTA_START_FRAC` 0.8→0.9 (halves the boosted-epoch count, ~halving the extra eval cost) and still captures the low-LR-tail peak where `best_acc` is set. **Scope note:** `TTA_START_FRAC` gates *which epochs enable eval-time TTA* — it has **zero effect on the training trajectory** (training is identical whether or not TTA is active at eval). So if the fallback is taken, the experiment's intended change set becomes {`ResNet9.forward` TTA logic + `TTA_START_FRAC`}, both eval-only; C2's confinement check (below) is about *training-affecting* code and is satisfied either way. The primary run holds `TTA_START_FRAC=0.8` (baseline value) so the sole change vs the 96.00 baseline is mirror→mirror+translate — cleanest attribution; the fallback is a contingency only.
- Python traceback / non-zero, non-124 exit in `run.log` (e.g. shape error in the crop logic) → code error; fix per execute-phase rules (counts as one retry).
- No `^best_test_acc:` line in `run.log` after the process exits → crash; inspect `tail -n 50 run.log`.
- Divergence (NaN/inf loss) in step logs → would indicate the edit perturbed training (it must not, being eval-only); abort and inspect the diff.

## Verification Protocol

### Verification Procedure
Baseline (from `exp-index.sh baseline`): **96.00%**, commit ae31206. Improvement bar = baseline + 0.1 = **96.10%**. Run conditions in order; STOP at the first failure (per goal: a failed necessary condition ⇒ no-improvement).

**C1 — Clean run within the wall guard** (necessary):
- **Authoritative wall guard**: `RUN_EXIT` (captured from the `timeout` wrapper) must be `0` — exit `124` means the process was killed at 600s (true process wall, including `uv`/Python import + the module-level `evaluator = Eval()` construction that precedes `t_start`). This is the real guard; the printed `total_seconds` slightly under-reports process wall (it starts after import/Eval), so it is a *secondary* check, not the primary one.
- `grep -c '^best_test_acc:' run.log` must equal `1`.
- Secondary: `awk '/^total_seconds:/{print ($2<600)?"PASS":"FAIL"}' run.log` must print `PASS` (printed-wall sanity; the authoritative guard is RUN_EXIT≠124).
- Pass = RUN_EXIT==0 AND exactly one summary line AND printed total_seconds<600; else FAIL (wall-clock or crash).

**C2 — Full training budget + scope/integrity intact** (necessary):
- Full budget: `awk '/^training_seconds:/{print ($2>=295)?"PASS":"FAIL"}' run.log` → `PASS` (training used the fixed 300s; not truncated).
- Frozen eval harness: `git diff --quiet autoresearch/maximize-cifar10-test-accuracy-dev -- prepare.py && echo PASS || echo FAIL` → `PASS` (prepare.py byte-unchanged vs the integration branch — this catches **staged or unstaged** edits, unlike a bare `git diff --quiet` which only sees the working tree). No staging is performed during the run, but the branch comparison is the robust check.
- Scope: `git diff --name-only autoresearch/maximize-cifar10-test-accuracy-dev -- | sort` lists exactly `train.py` (and nothing else).
- **Training-trajectory confinement** (authoritative integrity check): `git diff autoresearch/maximize-cifar10-test-accuracy-dev -- train.py` shows changes ONLY within the `ResNet9.forward` body — no edit to any **training-affecting** code: HP constants that drive training (PEAK_LR/EMA_DECAY/BATCH_SIZE/WEIGHT_DECAY/LABEL_SMOOTHING/PCT_START/MOMENTUM), the LR schedule, optimizer, augmentation, architecture (`__init__`), or the training loop. The eval-only TTA logic in `forward` (and, if the wall-clock fallback was taken, `TTA_START_FRAC`) is the *intended* change and is permitted — it cannot alter the trained weights. Authoritative manual read of the diff hunk(s). **Because the diff is provably confined to the eval path, the training trajectory is byte-identical to EXP-004 by construction** — this is the real guarantee, stronger than any metric cross-check.
- Training-unchanged sanity cross-check: `awk '/^num_epochs:/{print $2}' run.log` should fall in ~120–160 (EXP-004 = 142). A wide band is used deliberately: GPU 0 is running another user's job, so shared-host throughput can vary epoch count run-to-run even with identical training code. A value far outside this band warrants a look at the diff, but the diff-confinement above (not epoch count) is the authoritative proof training was untouched. Informational, not a hard gate.
- Pass = budget PASS AND prepare.py unchanged AND scope == train.py only AND diff confined to the eval path; else FAIL.

**C3 — Improvement ≥ +0.1pp + genuineness** (necessary):
- `BEST=$(awk '/^best_test_acc:/{print $2+0}' run.log)`; PASS iff `BEST >= 96.10` (i.e. `awk -v b=$BEST 'BEGIN{print (b>=96.10)?"PASS":"FAIL"}'`).
- **Metric-genuineness cross-check** (ties the summary number to the frozen harness, as prior reports did): the summary `best_test_acc` must equal the maximum per-epoch `best:` value printed right after each `Eval.evaluate` call — `awk -F'best: ' '/eval ep/{split($2,a,"%"); if(a[1]+0>m)m=a[1]+0} END{printf "%.2f\n", m}' run.log` must equal the summary `best_test_acc`. A mismatch would mean the summary is not the harness's own max → treat as suspicious (possible invalid), not a clean improvement.
- Genuineness (only meaningful if the metric clears the bar): exactly one `evaluator.evaluate(` call site in the training loop (`grep -c 'evaluator.evaluate(' train.py` == 1 → ≤1 eval/epoch; backed by the C2 diff-confinement, which proves the loop's single eval call site is untouched); seeds unchanged (`grep -n 'manual_seed' train.py` shows `torch.manual_seed(42)`, `torch.cuda.manual_seed(42)` — no seed search); the gain comes from added eval views only (the diff touches only `forward`), not from training or eval-harness changes.
- Pass = `BEST >= 96.10` AND genuineness holds → **improvement**. `BEST < 96.10` → **no-improvement** (translate views add <0.1pp on top of the mirror component already captured).

### Informational Metrics (Optional)
- peak_vram_mb: `grep '^peak_vram_mb:' run.log` — VRAM headroom (expect ~1.6 GB).
- training_seconds / num_epochs / num_steps: `grep '^training_seconds:\|^num_epochs:\|^num_steps:' run.log` — confirm full-budget use and that epoch count matches EXP-004 (training unchanged).
- num_params: `grep '^num_params:' run.log` — expect 7,784,627 (unchanged from EXP-004; no architecture change).
- total_seconds: `grep '^total_seconds:' run.log` — record the realized wall-clock vs the ~500–540s estimate (informs the TTA-cost fallback decision).
