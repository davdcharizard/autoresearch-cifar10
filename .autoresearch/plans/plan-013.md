# Plan EXP-013: Reflection padding for RandomCrop
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-013.md

## Milestones

### Milestone 1: One-argument change implemented, passing static checks
- [x] Branch `autoresearch/exp-013` created from `autoresearch/dev`
- [x] `transforms.RandomCrop(32, padding=4)` → `padding_mode="reflect"` added; 1-line diff confirmed
- [x] `uv run python -m py_compile train.py` exits 0 (SYNTAX_OK)

### Milestone 2: Run launched into a clean window, early signals healthy
- [x] GPU 0 confirmed free (fully idle node) immediately before launch
- [x] Run started (task bxk0hsvm4); contention detector armed (bbr5zwtcm)
- [x] dt 22.4ms cum / ~139 epochs projected — GPU-side throughput byte-identical as predicted
- [x] Epoch-1 eval 39.30% (above the expected band — early signal cleaner, pre-saturation)

### Milestone 3: Run completed within caps, clean-run sanity passed
- [x] Run exited cleanly (exit 0) BUT total_seconds 626.9 > 600 — CAP BUSTED by ~150s of extra loader stalls (reflect pad's CPU cost; stalls are outside the timed budget so epochs were unaffected)
- [x] CONTENTION SANITY: 139/139 epochs, zero SLOW events — clean run; the overage is the change's own reproducible cost, not interference
- [x] `grep` → 96.57 / 1613.0

### Milestone 4: Verification rendered
- [x] Condition 0 CLEAN; condition 1 FAIL (626.9s > 600s); conditions 2–3 skipped per first-failure stop (metric 96.57 would also have missed); recorded in exp-log-013.md

## Code Changes
- **train.py** (only file; all other paths hook-protected): in the `train_tf` Compose, change `transforms.RandomCrop(32, padding=4)` to `transforms.RandomCrop(32, padding=4, padding_mode="reflect")`.

  Why this tests the hypothesis: zero padding puts up-to-4px black bands on border crops — statistics absent from the test distribution, forcing the network to spend capacity on artifact invariance. Reflection fills the pad with locally-natural content, so every training crop has test-like statistics: more signal per epoch at unchanged augmentation pressure (same crop geometry, same randomness). It is the single remaining 2016-era component of the recipe.

  Risks/edge cases: (a) `padding_mode="reflect"` requires pad < image size (4 < 32 — fine; torchvision supports constant/edge/reflect/symmetric); (b) runs on CPU in the 8 workers BEFORE ToTensor (PIL path) — same cost class as the existing crop, throughput untouched (EXP-004 precedent: even TA's much heavier PIL work was fully worker-absorbed); (c) no interaction with RandomErasing (applied after Normalize, on tensors) or TA (operates on the padded-cropped image as before).

## Configuration Changes
- RandomCrop padding_mode: zeros (default "constant") -> "reflect" (fast.ai DAWNBench CIFAR recipes standard practice; folklore magnitude +0.1–0.3pp; brainstorm-013 § Web Search). No numeric hyperparameter changes — schedule, batch, LR, WD, augmentation magnitudes all byte-identical to baseline 1990397.

## Execution Environment
- Method: local — `uv run train.py > run.log 2>&1` from project root, GPU 0 only (CUDA_VISIBLE_DEVICES=0; confirm free first, wait if busy — never GPU 1)
- Resources: 1x H20 (GPU 0); VRAM ~1613MB (identical to baseline)
- Estimated runtime: ~480s total (300s budget + ~139 evals + ~11s warm-cache startup; same shape as baseline so inductor cache should be warm); hard cap 600s
- Log output: all stdout/stderr to `run.log` (no tee/stream); deleted after the experiment concludes
- Tool skill: none (background Bash + Monitor watchers; contention protocol per infra-errors.md § Important)

## Abort Criteria
- Loss NaN/inf at any step (essentially impossible for a fill-content change — would indicate something else broke)
- dt-gate: projected epochs < 65 at step ~100 (would mean the transform somehow throttled the loader below GPU demand — not expected; EXP-004 precedent says worker-absorbed)
- Contention: SLOW events (windowed step time > 30ms) or collapsing epoch cadence → kill, wait for a clean window, relaunch (infrastructure retry, max 2)
- No new output in run.log for > 3 minutes
- Total wall clock reaching 590s with the run still going

## Verification Protocol

### Verification Procedure
Run from project root after the run exits. Baseline from `exp-index.sh baseline` = **96.71** (commit 1990397), direction higher; bar = 96.71 + 0.10 = **96.81**.

0. **Pre-condition (contention sanity, Protocol Findings EXP-011)**: num_epochs within ~10% of the step-100 projection (~139 expected) and zero SLOW detector events; otherwise the run is infrastructure-contaminated — do NOT evaluate conditions, rerun instead.
1. **Run completed within budget (≤ 600s total)** — `grep "^total_seconds:" run.log`; PASS if ≤ 600 and the summary block exists (clean exit). Empty grep ⇒ crash ⇒ `tail -n 50 run.log`. Timeout: n/a (post-hoc parse).
2. **best_test_acc ≥ 96.81** — `grep "^best_test_acc:" run.log`; PASS if parsed value ≥ 96.81. Timeout: n/a.
3. **Validation at most once per epoch** — `grep -c "eval ep" run.log` ≤ num_epochs from the summary. Structurally guaranteed; checked mechanically. Timeout: n/a.

First failed condition stops evaluation. On all-pass: verdict improvement; index insert; commit the one-line diff on `autoresearch/exp-013`; merge to `autoresearch/dev` (no remote — skip push/PR per TASK.md, fields stay "(pending)"/"N/A"). On condition-2 failure with a clean run: no-improvement; discard changes. Delete run.log after analysis either way.

### Informational Metrics (Optional)
- peak_vram_mb: `grep "^peak_vram_mb:" run.log` — expect ~1613MB (must be identical to baseline; any change is a red flag)
- num_epochs: `grep "^num_epochs:" run.log` — expect 139 (throughput must be untouched; fewer epochs ⇒ contention or loader throttling, investigate before trusting the metric)
- num_params: `grep "^num_params:" run.log` — must equal 4,286,026
