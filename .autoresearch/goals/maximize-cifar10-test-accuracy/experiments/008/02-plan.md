# Plan EXP-008: Stronger augmentation — Cutout 8→12 + light RandomErasing
- **Created**: 2026-06-28

## Milestones

### Milestone 1: Code change implemented + local checks pass
- [ ] Edit `train_tf` in `main()` (`train.py:205-213`): change `Cutout(8)` → `Cutout(12)`, and append `transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0.0)` as the LAST transform (after `Cutout(12)`). No other line changes anywhere.
- [ ] `uv run python -c "import py_compile; py_compile.compile('train.py', doraise=True)"` → clean compile.
- [ ] `git diff ae31206 -- train.py` is confined to exactly those two changes inside `train_tf` (the `Cutout` arg `8→12` and the one added `RandomErasing` line); `git diff --name-only ae31206 --` lists ONLY `train.py`. NO change to any HP constant (PEAK_LR, WEIGHT_DECAY, LABEL_SMOOTHING, BATCH_SIZE, schedule), architecture (`ResNet9`), optimizer, EMA, whitening, training loop, `forward`, `pool`, or `fc`. (Diff base = baseline commit `ae31206`, not the mutable branch name, so the scope check is stable.)
- [ ] In-process smoke (CPU ok): build the exact `train_tf` Compose; apply it to a random PIL uint8 32×32 image → assert output tensor `[3,32,32]` float and finite; force `RandomErasing(p=1.0,...)` once to assert it DOES zero a rectangle (mechanism live); confirm `Cutout(12)` zeroes a 12×12 (clipped) patch. Build `ResNet9` and print `num_params` — assert it is **UNCHANGED at 7,784,627** (augmentation-only change ⇒ zero architecture/param delta; this is the cross-check that nothing structural moved).
- [ ] Confirm `RandomErasing` value=0.0 == mean-fill in the frozen normalized space (mean-subtract, std=1) — i.e. consistent with the existing `Cutout` zero-fill convention; normalization stats `EVAL_MEAN/EVAL_STD` (`train.py:34`) are byte-unchanged.

### Milestone 2: Official run completes within wall guard
- [ ] **Guard (Milestone-1 applied)**: before launching, confirm the edit is in place — `git diff ae31206 -- train.py` is NON-empty and shows the `Cutout(12)` + `RandomErasing` changes (per concern #1: launching with the unedited tree would silently retest EXP-004 and invalidate attribution).
- [ ] Remove any stale sentinel/log first (`rm -f run.log run_exit.txt`), then launch the EXACT command that writes the exit code immediately after `timeout`: `( timeout 600 bash -c 'CUDA_VISIBLE_DEVICES=1 uv run train.py > run.log 2>&1'; echo "RUN_EXIT=$?" > run_exit.txt ) &` — guarantees `run_exit.txt` carries THIS run's exit code (not a stale one). Then BLOCK on the sentinel with a foreground wait loop (`for i in $(seq 1 64); do [ -f run_exit.txt ] && break; sleep 10; done`) — do NOT poll via background `sleep`.
- [ ] `run.log` has exactly one `^best_test_acc:` line; `run_exit.txt` shows `RUN_EXIT=0` (not 124).

### Milestone 3: Verification
- [ ] Run the Verification Procedure (C1→C2→C3, stop at first failure). Record `num_epochs`, `img/s`, and the ep25/ep50 trajectory as first-class diagnostics (throughput-preservation and under-fit are the two pre-registered failure modes).

## Code Changes
- **`train.py` — `main()` `train_tf` pipeline (lines 205-213)**:
  - `Cutout(8)` → `Cutout(12)`
  - append (new last transform): `transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0.0)`
  - Resulting pipeline: `RandomCrop(32,pad4) → RandomHorizontalFlip → ToTensor → Normalize → Cutout(12) → RandomErasing(...)`.
  - **Why this tests the hypothesis**: the diagnosis is that the net is **saturated / regularization-bound with a ~4× epoch surplus** (fits ~142–150 epochs in 300s vs airbench96's 37 for the same ~96%). Stronger input-space augmentation makes each epoch harder → slows *convergence* (consuming the wasted epochs) → the fully-annealed minimum generalizes better → higher `best_test_acc`. cutout12 is the documented airbench96 value (the one augmentation axis where we under-shoot; we already match LS 0.2 and crop pad 4). RandomErasing (Zhong et al., AAAI 2020) stacks a second, probabilistic, variable-size erase so the *combined* regularization step is large enough to plausibly clear the ~0.1pp noise floor (cutout12 alone is likely sub-noise).
  - **Why no other change**: single-variable attribution vs the EXP-004 baseline (96.00%). PEAK_LR=0.4, schedule, optimizer, EMA, whitening, batch size, TTA, architecture all held byte-identical.
  - **Critical throughput property (refined per review concern #2)**: both transforms run on the CPU `DataLoader` workers (`NUM_WORKERS`, `persistent_workers=True`, `prefetch_factor=4`) in parallel with the GPU step, NOT on the GPU. IMPORTANT subtlety in the timing model: `t0` is set INSIDE the loop AFTER `for inputs,targets in train_loader` yields a batch (`train.py:278-279`), and `total_training_time` accumulates only the per-step GPU `dt` — so the time spent BLOCKED waiting for the DataLoader to produce the next batch is EXCLUDED from the 300s training budget (consistent with the EXP-001 finding that the budget is GPU-training-time, not wall). Consequence: a worker bottleneck from RandomErasing would NOT reduce `num_epochs` (the budget still fits the same GPU-step-seconds) — instead it inflates the WALL clock (`total_seconds`) by adding off-budget DataLoader-wait. **So `total_seconds` (wall), not `num_epochs`, is the primary worker-saturation diagnostic**: EXP-004 ran ~440s wall; a large jump toward the 600s cap signals workers can't keep up. `num_epochs` staying ~142–150 confirms GPU-step throughput is unaffected (the expected case, since erase-augs on 32×32 are cheap). The favorable implication: this lever cannot under-ANNEAL (epochs are budget-protected); its only throughput failure mode is wall-cap inflation, caught by C1 (`total_seconds<600`, `RUN_EXIT≠124`).
  - **Risks/edge cases**: (a) **under-fit** (not under-anneal) if the combined erasing is too strong for ~142 epochs → observable via mid-trajectory (ep25 should stay near EXP-004's ~92.6%) and a still-rising `best==final` tail. This is a no-improvement with a clean cause, NOT an abort. (b) `RandomErasing` draws from the global torch RNG, seeded by the unchanged `torch.manual_seed(42)`/`torch.cuda.manual_seed(42)` — deterministic, NOT seed hacking. (c) value=0.0 = mean in raw space (std=1 normalization), matching the existing Cutout convention — no normalization mismatch. (d) `num_params`, `forward`, eval path all unchanged → `prepare.py` untouched, ≤1 eval/epoch preserved. (e) **Augmentation-strength accuracy (review concern #6)**: `scale=(0.02,0.15)` can erase up to ~15% of the 32×32 image (~154 px) — comparable to or slightly larger than the unclipped 12×12 cutout (144 px), NOT strictly "smaller"; on the 25% of images where it fires it adds a meaningful, variable-area second erase (possibly overlapping the cutout). The combined per-image erased area is therefore variable — reinforcing the under-fit watch (risk (a)), which is the correct diagnostic rather than assuming the increment is small.
  - **Known noise-floor confound (unavoidable, acknowledged)**: changing the augmentation perturbs the global RNG stream (the added `RandomErasing` draws), so the per-image data stream differs slightly run-to-run — inherent to any augmentation change and part of the ~0.1pp noise floor. Only a clearly-greater-than-noise gain registers; a sub-0.1pp single-run result is a NULL.

## Configuration Changes
- Cutout patch size: `8 → 12` (airbench96 documented value; `knowledge/references/fast-cifar10-recipes.md`).
- New augmentation: `RandomErasing(p=0.25, scale=(0.02,0.15), ratio=(0.3,3.3), value=0.0)` — light (fires on 25% of images, area ≤15%, mean-fill). Settings chosen conservative to add regularization without tipping ~142 epochs into under-fit. No HP/schedule/architecture change otherwise (PEAK_LR=0.4 held).

## Execution Environment
- Method: local, single process. `timeout 600 bash -c 'CUDA_VISIBLE_DEVICES=1 uv run train.py > run.log 2>&1'` from project root, launched per Milestone 2 with the `run_exit.txt` sentinel and a foreground sentinel-wait loop.
- Resources: 1× NVIDIA H20, **GPU 1** (`CUDA_VISIBLE_DEVICES=1` — GPU 0 in use, hard constraint). VRAM: augmentation-only change, expect ~1.6 GB (unchanged from EXP-004), a small fraction of 98 GB.
- Estimated runtime: ~430–490s wall (training 300s fixed + ~140s eval/startup). Epochs projected ~142–150 (UNCHANGED — augmentation is CPU-side); fewer evals only if throughput unexpectedly drops.
- Log output: `run.log` (redirected). Summary parsed for best_test_acc, training_seconds, total_seconds, num_epochs, num_steps, peak_vram_mb, num_params; per-epoch `eval ep` lines for the trajectory (ep25/ep50/tail); step lines for img/s.
- Tool skill: none (local run).

## Abort Criteria
- `timeout` exit 124 (wall ≥ 600s) → wall-clock failure. Unlikely (no throughput change expected), but if hit treat as failed run and investigate worker saturation.
- Python traceback / non-zero non-124 exit (e.g. a transform API error) → code error; fix per execute rules (one retry).
- Divergence (NaN/inf loss) in step logs → would indicate the augmentation destabilized training (very unlikely for erase-style augs); abort and inspect.
- **NOT an abort — the under-fit signal is the measurement**: if early loss is clearly worse than EXP-004's trajectory (ep25 well below ~92%) and the tail is still climbing (`best==final`), let the run FINISH — that is the pre-registered "too-strong augmentation" outcome that informs the analysis (back off RandomErasing / cutout→10 next), not a crash.

## Verification Protocol

### Verification Procedure
Baseline (from `exp-index.sh baseline`): **96.00%**, commit ae31206. Improvement bar = baseline + 0.1 = **96.10%**. Run conditions in order; STOP at the first failure.

**C1 — Clean run within the wall guard** (necessary):
- Authoritative: `RUN_EXIT` from `run_exit.txt` must be `0` (not `124`; the `timeout` exit is the real wall guard — it measures full process wall including import + module-level `Eval()` which precede the printed `total_seconds`).
- `grep -c '^best_test_acc:' run.log` == 1.
- Secondary: `awk '/^total_seconds:/{print ($2<600)?"PASS":"FAIL"}' run.log` == PASS.
- Pass = RUN_EXIT==0 AND one summary line AND printed total_seconds<600.

**C2 — Full training budget + scope/integrity intact** (necessary):
- Full budget: `awk '/^training_seconds:/{print ($2>=295)?"PASS":"FAIL"}' run.log` == PASS.
- Frozen eval harness: `git diff --quiet ae31206 -- prepare.py && echo PASS || echo FAIL` == PASS (diff base = baseline commit `ae31206`; catches staged+unstaged).
- Scope (tracked): `git diff --name-only ae31206 -- | sort` lists exactly `train.py`.
- **Scope (untracked, fixed per review concern #4)**: `run.log`/`run_exit.txt` are **gitignored** (confirmed via `git check-ignore`), so `git status --porcelain` will NOT show them — do not expect them there. The real check is for a stray importable `.py` in cwd that could shadow modules: `ls *.py` must list **exactly `prepare.py train.py`** (confirmed the only two root `.py` files). Additionally `git status --porcelain` should show only ` M train.py` among tracked changes. Any other `.py` in `ls *.py` is a red flag → investigate.
- Change confinement: `git diff ae31206 -- train.py` shows changes ONLY inside `train_tf` (the `Cutout` arg and the added `RandomErasing` line) — no HP/schedule/optimizer/architecture/loop/`forward` edit. PEAK_LR still 0.4: `awk '/^PEAK_LR/{print $3}' train.py` == `0.4`. Authoritative manual diff read (the diff read, not the param count, is what proves architecture/forward/schedule integrity — concern #7).
- Param cross-check (fixed per concern #7 — exact machine compare): `awk '/^num_params:/{print $2}' run.log` equals **`7,784,627`** exactly (`$2` is the comma-formatted value token; UNCHANGED from EXP-004 — a NECESSARY-not-sufficient confirmation that the change was augmentation-only and nothing structural moved; the manual diff read above is the sufficient integrity check).
- Pass = budget PASS AND prepare.py unchanged AND tracked scope==train.py only AND `ls *.py`==`prepare.py train.py` AND diff confined to the two augmentation changes AND num_params==7,784,627.

**C3 — Improvement ≥ +0.1pp + genuineness** (necessary):
- `BEST=$(awk '/^best_test_acc:/{print $2+0}' run.log)`; PASS iff `awk -v b=$BEST 'BEGIN{print (b>=96.10)?"PASS":"FAIL"}'` == PASS.
- **Noise-floor rule (per Protocol Findings)**: a result in [96.03, 96.09] is a NULL, not a win — only a clear ≥96.10 counts. (Sub-0.1pp single-run "wins" are treated as unproven; seed is fixed so noise can't be averaged out.)
- Metric-genuineness cross-check (fixed per concern #3 — parse `test_acc:`, not the circular `best:`): the summary `best_test_acc` must equal the max over the per-epoch evaluated `test_acc:` fields — `awk -F'test_acc: ' '/eval ep/{split($2,a,"%"); if(a[1]+0>m)m=a[1]+0} END{printf "%.2f\n", m}' run.log` equals the summary value. (Parsing `test_acc:` proves the summary equals the true max evaluator accuracy; the old `best:` parse was circular since `best:` is already the running best.)
- Genuineness: exactly one `evaluator.evaluate(` site (`grep -c 'evaluator.evaluate(' train.py` == 1 → ≤1 eval/epoch); seeds unchanged (`grep -n 'manual_seed' train.py` shows `torch.manual_seed(42)`/`torch.cuda.manual_seed(42)`); the gain comes from the augmentation change only.
- **Attribution check (promoted per concern #5 — not a pass/fail gate, but recorded and judged)**: the goal's necessary conditions do NOT include epoch count, so a genuine `BEST>=96.10` is an improvement by definition. BUT for clean attribution, record `num_epochs` (expect ~142–150) and `total_seconds` (expect ~440s). If `BEST>=96.10` with these in their normal bands → clean improvement attributable to the augmentation. If `BEST>=96.10` but `num_epochs` or `total_seconds` are abnormal (e.g. wall near 600s = worker saturation, or epochs far from 142–150) → still an improvement on the metric, but the analysis MUST flag the throughput confound rather than silently crediting the augmentation.
- Pass = `BEST>=96.10` AND genuineness → **improvement** (with the attribution caveat above). `BEST<96.10` → **no-improvement** (then read `num_epochs` + `total_seconds` + trajectory to diagnose sub-noise vs under-fit per below).

### Informational Metrics (recorded every run; FIRST-CLASS for attribution per concern #2/#5)
- **total_seconds (PRIMARY worker-saturation diagnostic)**: `awk '/^total_seconds:/{print $2}' run.log`. Because the 300s budget EXCLUDES DataLoader-wait (timing model above), a RandomErasing worker bottleneck manifests as INFLATED WALL, not fewer epochs. EXP-004 ref ≈ 440s. A large jump toward 600s ⇒ workers can't keep up ⇒ off-budget wall inflation (and C1 wall-cap risk). This is the check `num_epochs` alone would miss.
- **num_epochs / num_steps / img/s**: `grep '^num_epochs:\|^num_steps:' run.log`; img/s from step lines (`grep 'img/s' run.log | tail -n 3`). Epochs should stay ~142–150 (EXP-004: 142 at ~26k img/s; EXP-006: 150) — confirms GPU-step throughput unaffected (the expected case). Combined with `total_seconds`: normal epochs + normal wall = clean throughput-free attribution; normal epochs + inflated wall = worker-saturation confound.
- **Mid-trajectory (under-fit diagnostic)**: `grep 'eval ep  25\|eval ep  50' run.log` (and the tail `grep 'eval ep' run.log | tail -n 5`). EXP-004 ref: ep25 ≈ 92.6%. If ep25 is well below ~92% AND the tail is still monotonically rising with `best==final`, that is **under-fit from too-strong augmentation** (→ back off RandomErasing / cutout→10 next), distinct from sub-noise (normal trajectory, flat tail, best≈baseline).
- peak_vram_mb: `awk '/^peak_vram_mb:/{print $2}' run.log` — expect ~1.6 GB (unchanged; soft constraint).
- num_params: `awk '/^num_params:/{print $2}' run.log` — expect 7,784,627 (unchanged; the C2 cross-check).
