# Plan EXP-015: Pre-activation (true-WRN) BasicBlocks — BN→ReLU→conv, bare-conv shortcuts, final BN→ReLU
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-015.md

## Milestones

### Milestone 1: Code change implemented and passes local checks
- [ ] Rewrite `BasicBlock` to pre-activation ordering (BN→ReLU→conv ×2, bare-conv projection shortcut on the shared
      pre-activated input, no post-add ReLU). `bn1` now sized to `in_channels`.
- [ ] Update `ResNet`: stem = bare `conv1` (drop stem `bn1`+ReLU); add `self.bn_final = BatchNorm2d(64*width_mult)`;
      `forward` = `conv1 → layer1..3 → relu(bn_final) → avgpool → fc`.
- [ ] `uv run ruff check train.py` clean; `git diff` = train.py only (BasicBlock + ResNet).

### Milestone 2: Experiment running and early signal confirmed
- [ ] Launch `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` (background).
- [ ] Confirm clean startup: `params ≈ 4,298,970` (slightly DOWN from 4,299,866 — expected from BN restructuring; a
      wildly different count signals a structural bug), clean compile, no traceback, no NaN.
- [ ] Read steady-state `dt`/`img/s` (step ~400–500) — expect ~8ms/step (pre-act is compute-neutral; same FLOPs).

### Milestone 3: Run completes and metrics extracted
- [ ] Run exits 0, `total_seconds < 600`.
- [ ] Extract `best_test_acc`, `num_epochs`, `final_test_loss`, `num_steps`, `num_params`.

## Code Changes
- **train.py** (the ONLY editable file):
  1. **`BasicBlock` → pre-activation** (replace the class body):
     ```
     def __init__(self, in_channels, out_channels, stride=1):
         self.bn1   = nn.BatchNorm2d(in_channels)
         self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
         self.bn2   = nn.BatchNorm2d(out_channels)
         self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
         self.shortcut = (nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
                          if (stride != 1 or in_channels != out_channels) else None)
     def forward(self, x):
         out = F.relu(self.bn1(x))
         shortcut = self.shortcut(out) if self.shortcut is not None else x
         out = self.conv1(out)
         out = self.conv2(F.relu(self.bn2(out)))
         out += shortcut
         return out
     ```
     (Canonical He-2016 pre-activation: shared first BN→ReLU feeds both conv1 and the projection shortcut; shortcut
     is a bare 1×1 conv, no BN; no post-add ReLU.)
  2. **`ResNet`**: keep `self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)` as the bare stem; REMOVE
     `self.bn1` from the stem; ADD `self.bn_final = nn.BatchNorm2d(64 * width_mult)`. `forward`:
     `out = self.conv1(x)` → layer1/2/3 → `out = F.relu(self.bn_final(out))` → adaptive_avg_pool2d → view → fc.
  - **Why this tests the hypothesis**: converts the post-activation ResNet-v1 blocks to the canonical pre-activation
    WideResNet formulation (cleaner identity/gradient path, better generalization) at ~zero compute cost. `_weights_init`
    (Kaiming on Conv2d/Linear) is unaffected. compile/eval unchanged.
  - **Risks/edge cases**: (a) num_params shifts ~−900 (BN restructuring) — EXPECTED, not a violation; (b) shallow
    ResNet-20 → gain may be small/noise; (c) implementation bug (shortcut placement / missing final BN) → guarded by
    abort criteria + early-acc check. Throughput unchanged (same FLOPs, still launch-bound, compile applies).

## Configuration Changes
- Architecture: post-activation BasicBlock → pre-activation BasicBlock + bare-conv shortcuts + stem-BN removed +
  final BN→ReLU added. No hyperparameter change (k=4, batch 128, peak LR 0.2 cosine, Nesterov, WD 1e-4, LS 0.1,
  Cutout(16), TrivialAugment, compile, seed 42 — all inherited from EXP-012 baseline, commit 6c417a4).

## Execution Environment
- Method: local — `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` (background via Bash `run_in_background`).
- Resources: single NVIDIA H20 (GPU 0); ~0.5 GB VRAM; 8 dataloader workers.
- Estimated runtime: ~300s training + ~10–20s startup + ~10–15s compile ≈ 330–360s wall-clock (< 600s budget).
- Log output: all stdout/stderr → `run.log`. Metrics via
  `grep -aE "^best_test_acc:|^total_seconds:|^num_epochs:|^final_test_loss:|^num_steps:|^num_params:" run.log`.
- Tool skill: none (local run).

## Abort Criteria
- Loss NaN/inf or divergence → kill.
- Traceback / process exit ≠ 0 at startup (e.g., shape mismatch from a wrong bn1 size or missing bn_final) → kill,
  fix, single retry.
- `num_params` wildly off ~4.30M (e.g., < 4.0M or > 4.6M) → structural bug → kill, investigate. (A ~900 decrease to
  ~4,298,970 is expected and NOT an abort.)
- Early-accuracy sanity: if `test_acc` is still < ~90% past ~40% of the budget (≈ ep 35), suspect a structural bug
  (a correct pre-act net trains as fast as the baseline) → note for analysis (let it finish unless NaN/stall).
- No log progress for > ~120s after startup → kill (silent hang).

## Verification Protocol

### Verification Procedure
Baseline = **96.22** (`exp-index.sh baseline`); success bar = **96.32** (+0.1pp per goal). After the run completes:

1. **Cond 1 — clean completion within budget**: `grep -aE "^best_test_acc:|^total_seconds:" run.log` returns a value
   AND `total_seconds < 600`, AND `grep -ac "Traceback" run.log` == 0. Pass = all hold. (Timeout: 600s.)
2. **Cond 2 — primary metric clears bar**: parse `best_test_acc`; PASS iff `best_test_acc >= 96.32` (baseline 96.22
   + 0.1). FAIL → verdict no-improvement. (Decisive condition.)
3. **Cond 3 — no constraint violations** (only if Cond 2 passes): `git diff --name-only` lists ONLY `train.py`; seed
   42 intact; eval-line count == `num_epochs` (eval once/epoch); `num_params` ≈ 4.30M (architecture change, ~−900
   expected — confirm it's a sane BN-restructuring delta, NOT a capacity change like added width/depth).

### Informational Metrics (Optional)
- `num_epochs`: `grep -a "^num_epochs:" run.log` — fairness check (expect ~88–91; pre-act is compute-neutral).
- `final_test_loss`: `grep -a "^final_test_loss:" run.log` — key corroborator vs EXP-012's 0.195 (< 0.195 ⇒ pre-act
  improved generalization; ≈ 0.195 with flat acc ⇒ negligible on this shallow net).
- `img/s` & `dt`: step ~400–500 — confirm ~8ms/step (≈ EXP-012) to rule out a throughput confound.
- `num_params`: confirm ≈ 4,298,970 (sane pre-act BN restructuring).
