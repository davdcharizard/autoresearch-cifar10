# Plan EXP-044: Mild RandAugment After Crop/Flip
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-044.md

## Milestones

### Milestone 1: Implement isolated mild RandAugment probe
- [x] Create experiment branch `autoresearch/exp-044` from `autoresearch/dev`.
- [x] Modify only `train.py`.
- [x] Insert `transforms.RandAugment(num_ops=1, magnitude=5)` in `train_tf` after `transforms.RandomHorizontalFlip()` and before `transforms.ToTensor()`.
- [x] Preserve `STAGE_WIDTHS = (28, 56, 112)`, `BATCH_SIZE = 128`, `LR = 0.1`, `MOMENTUM = 0.9`, `WEIGHT_DECAY = 2e-4`, `LR_MILESTONES = [21000, 64000]`, `label_smoothing=0.05`, reflected `RandomCrop`, FP32 training, channels-last, cuDNN benchmark, `torch.compile`, seed, optimizer class, and once-per-epoch validation.
- [x] Run `uv run python -c 'from torchvision import transforms; print(hasattr(transforms, "RandAugment"))'` and confirm it prints `True`.
- [x] Run `python3 -m py_compile train.py`.
- [x] Run `uv run ruff check train.py`.
- [x] Confirm `git diff -- train.py` contains only the planned transform insertion.
- [x] Confirm transform order and anchor settings with `sed -n '112,140p' train.py` and `rg -n "STAGE_WIDTHS = \\(28, 56, 112\\)|BATCH_SIZE = 128|LR = 0\\.1|MOMENTUM = 0\\.9|WEIGHT_DECAY = 2e-4|LR_MILESTONES = \\[21000, 64000\\]|label_smoothing=0\\.05|RandomCrop\\(32, padding=4, padding_mode=\\\"reflect\\\"\\)|RandAugment\\(num_ops=1, magnitude=5\\)" train.py`.
- [x] Confirm validation cadence remains once per epoch with `rg -n "evaluator\\.evaluate|Eval\\(" train.py`.

### Milestone 2: Launch and monitor single-GPU experiment
- [x] Query the current baseline with `bash /root/.codex/plugins/cache/deoxys/autoresearch/2.9.6/skills/shared/scripts/exp-index.sh baseline .autoresearch/experiment-indices/maximize-cifar10-best-test-accuracy.tsv`; expected baseline is `93.97`, so the EXP-044 improvement threshold is `94.07`.
- [x] Check GPU availability with `nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits`.
- [x] Select one idle GPU and confirm CUDA sees only that GPU with `CUDA_VISIBLE_DEVICES=${GPU_ID} uv run python -c 'import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")'`.
- [x] Remove stale `run.log` before launch.
- [x] Launch `CUDA_VISIBLE_DEVICES=${GPU_ID} uv run train.py > run.log 2>&1` in a foreground session; detached background jobs are not reliable in this shell per EXP-042.
- [x] Monitor startup until `run.log` reports device, parameter count, 300s time budget, and `Batches per epoch: 390`.
- [x] Confirm first progress logs show `lr: 0.1000`.
- [x] Confirm the first LR drop occurs at step 21000 with `lr: 0.0100`.
- [x] Confirm no second LR drop occurs before completion, because the preserved second milestone at 64000 should be unreachable.

### Milestone 3: Capture result, verify, and clean up
- [x] Ensure the run exits before 10 minutes total wall-clock time.
- [x] Extract summary metrics with `grep "^best_test_acc:\\|^final_test_acc:\\|^final_test_loss:\\|^training_seconds:\\|^total_seconds:\\|^startup_seconds:\\|^peak_vram_mb:\\|^num_epochs:\\|^num_steps:\\|^num_params:" run.log`.
- [x] Compare `best_test_acc` to the concrete EXP-044 threshold `94.07%` (`93.97 + 0.10`).
- [x] Record verdict and metrics in `.autoresearch/logs/exp-log-044.md`; the experiment index is updated during analysis.
- [ ] On improvement, commit `train.py`, merge into `autoresearch/dev`, and skip PR creation if no remote exists.
- [x] On no-improvement, crash, or invalid result, revert the RandAugment insertion with `apply_patch` and return to `autoresearch/dev`.
- [x] Remove `run.log` and root `__pycache__` after metrics are captured and analyzed.

## Code Changes
- **train.py**: Add one training transform line, `transforms.RandAugment(num_ops=1, magnitude=5)`, after random horizontal flip and before tensor conversion. This tests mild policy augmentation while preserving the current model, optimizer, schedule, data normalization, and evaluation path.

## Configuration Changes
- Add mild RandAugment with `num_ops=1` and `magnitude=5`.
- No architecture, optimizer, LR schedule, weight decay, label smoothing, batch size, seed, dependency, or harness configuration changes.

## Execution Environment
- Method: local command from the project root using one idle selected GPU; actual launch command is `CUDA_VISIBLE_DEVICES=${GPU_ID} uv run train.py > run.log 2>&1`.
- Resources: one NVIDIA H20 class GPU; no dependency, data, harness, or package changes.
- Estimated runtime: about 7 to 9 minutes total. RandAugment may add CPU dataloader overhead, but the run must remain under the 10-minute wall-clock cap.
- Log output: stdout and stderr captured in project-root `run.log`, then summarized into `.autoresearch/logs/exp-log-044.md` before cleanup.
- Tool skill: none; this is a local single-GPU run.

## Abort Criteria
- Stop and classify as failure if total wall-clock runtime exceeds 10 minutes.
- Stop and classify as crash if `run.log` shows an exception, CUDA OOM, missing data failure, NaN/Inf loss, or no progress output after launch.
- Stop and classify as invalid if any tracked source file other than `train.py` changes, if the fixed harness is touched, or if validation runs more than once per epoch.
- Treat the result as invalid if `num_params` differs from `822,790`, because this experiment must not change architecture.
- Treat the result as no-improvement if `Batches per epoch` is not 390, if first progress lines do not show `lr: 0.1000`, if the first LR drop does not occur at step 21000, or if the post-drop LR is not `0.0100`.
- If RandAugment causes total wall-clock to approach the 10-minute cap before final metrics, allow the foreground process to finish only if it is clearly seconds away from completion; otherwise terminate and classify according to the cap.
- If a foreground session returns `-1`, immediately check `pgrep`, `/proc/<pid>/cwd`, `run.log` modification time, and GPU utilization before deciding whether the run is still alive; do not kill processes from other run directories.

## Verification Protocol

### Verification Procedure
1. Verify baseline and threshold:
   - Command: `bash /root/.codex/plugins/cache/deoxys/autoresearch/2.9.6/skills/shared/scripts/exp-index.sh baseline .autoresearch/experiment-indices/maximize-cifar10-best-test-accuracy.tsv`
   - Pass condition: output reports `baseline=93.97`; the EXP-044 improvement threshold is `best_test_acc >= 94.07`.
   - Timeout: 30 seconds.
2. Verify scope before launch:
   - Command: `git diff --name-only`
   - Pass condition: the only tracked source diff is `train.py`; `.autoresearch/` artifacts are local-only and `data/` remains untracked.
   - Timeout: 30 seconds.
3. Verify API availability, syntax, and lint:
   - Commands: `uv run python -c 'from torchvision import transforms; print(hasattr(transforms, "RandAugment"))'`, `python3 -m py_compile train.py`, and `uv run ruff check train.py`
   - Pass condition: RandAugment availability prints `True`; syntax and lint commands exit 0.
   - Timeout: 2 minutes total.
4. Verify RandAugment placement, preserved anchor, and validation cadence:
   - Commands: `sed -n '112,140p' train.py`; `rg -n "RandAugment\\(num_ops=1, magnitude=5\\)|LR = 0\\.1|WEIGHT_DECAY = 2e-4|LR_MILESTONES = \\[21000, 64000\\]|label_smoothing=0\\.05|RandomCrop\\(32, padding=4, padding_mode=\\\"reflect\\\"\\)" train.py`; `rg -n "evaluator\\.evaluate|Eval\\(" train.py`
   - Pass condition: RandAugment is between `RandomHorizontalFlip()` and `ToTensor()`, current anchor settings are preserved, one `Eval()` construction exists, and one epoch-level `evaluator.evaluate(...)` call exists.
   - Timeout: 30 seconds.
5. Verify preserved schedule behavior, batch geometry, parameter count, and active LR:
   - Commands: `grep "Batches per epoch" run.log`; `grep "lr: 0.1000" run.log`; `grep "step 21000" run.log`; `grep "step 64000" run.log || true`; `grep "^num_params:" run.log`
   - Pass condition: batches per epoch is 390, first progress logs include `lr: 0.1000`, step 21000 includes `lr: 0.0100`, step 64000 is absent, and `num_params` is `822,790`.
   - Timeout: 30 seconds.
6. Verify experiment completion:
   - Command: `CUDA_VISIBLE_DEVICES=${GPU_ID} uv run train.py > run.log 2>&1`
   - Pass condition: process exits 0 before 10 minutes total wall-clock and prints a numeric `best_test_acc`.
   - Timeout: 10 minutes.
7. Verify metric improvement:
   - Command: `grep "^best_test_acc:" run.log`
   - Pass condition: parsed `best_test_acc` is at least `94.07%`; smaller increases over 93.97 are `no-improvement` under the goal's +0.10 percentage-point rule.
   - Timeout: 30 seconds.
8. Verify hard constraints:
   - Commands: `git diff -- train.py`, `git status --short --branch`, and `grep "^training_seconds:\\|^total_seconds:" run.log`
   - Pass condition: only the planned transform insertion was present during the run, fixed training budget was used, total wall-clock stayed under 10 minutes, and no protected files changed.
   - Timeout: 1 minute.

### Informational Metrics (Optional)
- final_test_acc: `grep "^final_test_acc:" run.log` - final evaluation accuracy.
- final_test_loss: `grep "^final_test_loss:" run.log` - final evaluation loss.
- training_seconds: `grep "^training_seconds:" run.log` - fixed-budget training time.
- total_seconds: `grep "^total_seconds:" run.log` - total runtime including startup, dataloader overhead, and validation.
- startup_seconds: `grep "^startup_seconds:" run.log` - startup and compile overhead.
- peak_vram_mb: `grep "^peak_vram_mb:" run.log` - peak CUDA allocation.
- num_epochs: `grep "^num_epochs:" run.log` - epoch count completed.
- num_steps: `grep "^num_steps:" run.log` - optimization-step budget.
- num_params: `grep "^num_params:" run.log` - confirms the anchor architecture remained unchanged.
