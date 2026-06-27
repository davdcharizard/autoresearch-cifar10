# Plan EXP-041: Derandomized alternating horizontal flip (shared-memory epoch tensor)
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-041.md

## Milestones

### Milestone 1: Code change implemented and sanity-checked (CPU) — propagation test is MANDATORY before launch
- [x] On branch `autoresearch/exp-041` (cut from `autoresearch/dev`), edit `train.py`:
  (a) add `from PIL import Image` import;
  (b) define `class AlternatingFlipCIFAR10(datasets.CIFAR10)` taking an `epoch_box` tensor; `__getitem__(i)` mirrors the torchvision parent (data[i] → `Image.fromarray` → transform) but applies `img.transpose(Image.FLIP_LEFT_RIGHT)` iff `(int(self.epoch_box[0]) + i) % 2 == 0` BEFORE the transform stack;
  (c) remove `transforms.RandomHorizontalFlip()` from `train_tf`;
  (d) create `epoch_box = torch.zeros(1, dtype=torch.int64); epoch_box.share_memory_()` before the dataset; instantiate `AlternatingFlipCIFAR10(..., epoch_box=epoch_box)` in place of `datasets.CIFAR10`;
  (e) in the training loop, set `epoch_box[0] = epoch` immediately after `epoch += 1` (before the loader iterator is created for that epoch)
- [x] CPU sanity A — flip schedule (single process): verified for i ∈ {0,1,7,100} that epoch-0 vs epoch-1 images are mirrors and flips occur iff `(e+i)%2==0`
- [x] CPU sanity B — WORKER PROPAGATION: PASSED — `torch.equal(e0.flip(-1), e1)` across a 64-item batch with `num_workers=2, persistent_workers=True`; shared-memory epoch updates propagate to live workers
- [x] CPU sanity C — params unchanged 4,286,026; `git diff --stat` shows train.py only (28 insertions / 3 deletions)

### Milestone 2: Gated launch, clean run to completion
- [x] Copy `/tmp/exp039_composite.sh` → `/tmp/exp041_composite.sh` (baseline thresholds apply unchanged — training path is signature-identical: contention > 27ms, STARTUP_KILL tick 10, NaN/divergence/wall guards)
- [x] Run completes: rc=0, num_epochs 139, dt 22.30ms, VRAM 1613.0MB

### Milestone 3: Verification and exp-log complete
- [x] First-failure-stop verification executed (protocol below), recorded in `logs/exp-log-041.md § Verification Results` — Condition 1 FAILED on merits (96.49 < 96.81)
- [x] Diagnostics recorded regardless of verdict: ep5/10/20 evals, last-15 plateau mean/spread, final_test_loss
- [ ] run.log deleted after extraction (analyze housekeeping)

## Code Changes
- **train.py** (only editable file): ~25-line change implementing deterministic orientation coverage. The dataset subclass applies the flip pre-transform keyed on `(epoch + index) % 2`, so over any two consecutive epochs every image is seen exactly once flipped and once unflipped (iid flip leaves per-image coverage with σ≈4.2% over 139 epochs). The shared-memory tensor is the epoch channel into persistent forked workers: `share_memory_()` moves the storage to shared memory at fork time, so main-process writes are visible to workers without restarts. Flip marginal stays 50% per epoch (half of indices match parity), so augmentation STRENGTH, per-image CPU cost (one PIL transpose on ~50% of images — identical to RandomHorizontalFlip's expected cost), loader timing, and gradient-noise statistics are unchanged. Edge cases: (i) flip now precedes RandomCrop (was after) — distributionally equivalent since crop offsets are symmetric; (ii) prefetch race at epoch boundary — none in this code shape: the `for ... in train_loader` iterator is recreated each epoch AFTER `epoch_box[0] = epoch` executes; (iii) workers use fork on Linux (default) — required for shared-memory propagation; sanity B verifies end-to-end.

## Configuration Changes
- Augmentation flip stage: iid `RandomHorizontalFlip(p=0.5)` -> deterministic alternation `(epoch + index) % 2` (airbench's alternating-flip intervention, the last unmeasured mechanism class — data order/coverage; zero cost in dt/heat/noise/numerics; epoch semantics preserved so all standing signature checks apply)

## Execution Environment
- Method: local, via `/tmp/exp041_composite.sh` with `run_in_background: true`
- Resources: GPU 0 ONLY (never GPU 1; wait if busy), ~1.6GB VRAM, host load < 60 at launch
- Estimated runtime: ~480–500s total (300s charged + startup ~9–18s + 139 × ~1.3s evals); cap 600s
- Log output: `uv run train.py > run.log 2>&1` (no tee); run.log deleted after extraction
- Tool skill: none (local)

## Abort Criteria
- STARTUP_KILL: no step line by tick 10 (~150s)
- CONTENTION_KILL: 4 consecutive 15s windows > 27ms (off-rung threshold per EXP-037 protocol note; true dt 22.3–22.4 quantizes to 18/24 rungs). On kill: confirm contamination, relaunch byte-identically when gates clear
- NaN guard: any `loss: nan` → kill (not expected; numerics unchanged)
- DIVERGENCE_KILL: any eval < 15% after epoch 5
- WALL_CAP_KILL: still running at tick 44 (~660s)
- Experiment-specific monitor (not a kill): if ep1–3 evals sit far below family (~25/45/51), suspect the flip schedule misbehaving in workers despite sanity B — let the run finish (divergence guard still protects), but flag for analysis

## Verification Protocol

### Verification Procedure
First-failure-stop; baseline via `exp-index.sh baseline` at verification time (currently 96.71; bar = 96.81).

**Pre-condition (run integrity):**
- Profile: 200-step quantization-safe windows (every 4th step-line pair) — require mean 22.0–23.0ms and 0 windows > 27ms; num_epochs in 135–143. If contaminated → rerun, do not judge.
- Integrity: `num_params: 4,286,026`, `training_seconds: 300.0`, eval-line count == num_epochs.
- Timeout: greps on finished run.log; missing summary ⇒ crash (`tail -n 50 run.log`).

**Condition 1 — best_test_acc ≥ 96.81**: `grep "^best_test_acc:" run.log`. Fail → STOP, verdict `no-improvement` (rest incidental).

**Condition 2 — within budget**: composite rc == 0 AND `total_seconds` ≤ 600.0.

**Condition 3 — validation ≤ once/epoch**: `grep -c "eval ep" run.log` ≤ num_epochs.

**Diagnostics (always):** ep5/10/20 vs family (~64/~75/~79) — the mechanism predicts any benefit concentrates EARLY (coverage deficit is largest at low epoch counts); last-15 plateau mean/spread vs ~96.5/±0.15; final_test_loss vs ~0.185.

### Informational Metrics (Optional)
- peak_vram_mb: `grep "^peak_vram_mb:" run.log` — expect 1613.0
- num_epochs: `grep "^num_epochs:" run.log` — expect 139 ± 4
- num_params: `grep "^num_params:" run.log` — expect 4,286,026
