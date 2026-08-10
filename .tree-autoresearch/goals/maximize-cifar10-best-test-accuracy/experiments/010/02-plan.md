# Plan EXP-010: Back-loaded 1-2-3 stage depth
- **Created**: 2026-08-05

## Milestones

### Milestone 1: Implement the isolated architecture change
- [x] Replace only the parent's static six-item `block_specs` with the fixed 1-2-3 sequence `[(16,64,1),(64,128,2),(128,128,1),(128,256,2),(256,256,1),(256,256,1)]` in `train.py`.
- [x] Update only the human-readable architecture/config output to report the 1-2-3 allocation; leave all training, CutMix, schedule, timer, evaluator, seed, and summary code unchanged.
- [x] Verify with `python -m py_compile train.py`, `git diff --check`, and `git diff --name-only a36dc09` (the final command must print only `train.py`).

### Milestone 2: Prove architecture and parent-source invariants
- [x] Assert the candidate resolves exactly to `[(16,64,1),(64,128,2),(128,128,1),(128,256,2),(256,256,1),(256,256,1)]`, has six blocks, three projection shortcuts, twelve block 3x3 convolutions, drop probabilities `0.08*i/6`, and exactly 3,855,578 trainable parameters.
- [x] Export `a36dc09:train.py` to a temporary file outside the repository and load it under a unique importlib module name while executing from the repository root. Pin `torch.manual_seed(42)` immediately before each parent/candidate construction; module-import side effects are outside this reseeded construction comparison.
- [x] In CPU FP32 eval mode, require the temporary parent to have the exact original block order and 2,748,890 parameters. Compare all candidate state tensors whose names and shapes are shared up to the first divergent block, documenting expected divergence rather than using a vacuous whole-model bitwise test.
- [x] Independently hook one forward to verify stage shapes and calculate exactly 392,612,352 Conv/Linear MACs per image for both candidate and parent.
- [x] Run CPU FP32 and GPU-0 BF16/channels-last forward/backward smokes; require `(256,10)` logits, finite loss, finite nonzero gradients on every trainable tensor, and no unexpected dtype/layout fallback.

### Milestone 3: Pass a paired physical-GPU-0 latency gate
- [x] Confirm physical GPU 0 is the approximately 97,871 MiB NVIDIA H20 with `nvidia-smi -i 0 --query-gpu=name,memory.total --format=csv,noheader` and confirm the visible PyTorch device under `CUDA_VISIBLE_DEVICES=0`.
- [x] In one temporary, non-repository Python harness executed with the repository root as cwd/sys.path, import the candidate from modified `train.py` and the exact parent from the uniquely named temporary `a36dc09:train.py`; use fixed synthetic batch-256 inputs/targets, channels-last, BF16 autocast, cross-entropy, backward, Nesterov update, drop scale 1.0, and CUDA synchronization.
- [x] Warm each arm for 50 steps, then collect five alternating rounds of 200 steps per arm plus 200 evaluation forwards per arm. Record per-round and aggregate median/p90/mean latency, finite loss/gradients, parameter counts, and peak VRAM in `03-execute.md`.
- [x] Proceed only if candidate median training latency is <=1.05x parent, candidate p90 is <=1.08x parent, `27950 / median_ratio >= 26500`, and measured evaluation time projects total runtime below 600 seconds. If a valid gate fails, stop without a metric run.

### Milestone 4: Execute one fixed-seed metric run
- [x] Remove any stale `run.log`, then launch exactly once with `timeout 600s env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 uv run train.py > run.log 2>&1`.
- [x] Monitor for process failure, nonfinite values, CUDA/OOM errors, stalled output, and the 600-second outer limit. Do not abort or modify the experiment based on intermediate test accuracy.
- [x] Preserve the existing step-level smooth-loss/progress records and per-epoch evaluation records. Summarize early epochs and progress bins in `03-execute.md` so an accuracy miss can be assessed for early-feature starvation; no parent early trajectory is available, so do not claim a direct historical curve comparison.

### Milestone 5: Verify and hand off for analysis
- [x] Require exit 0, a complete final summary, charged training in `[299.5, 301.0]` seconds, total runtime below 600 seconds, one evaluation per completed epoch, and candidate parameters 3,855,578. The >=26,500-step and 0.47-0.53 CutMix-ratio targets are preregistered mechanism diagnostics, not extra formal accuracy-verdict gates after a completed run.
- [x] Parse `best_test_acc`; formal success is >=95.33% versus parent EXP-002 at 95.23%. Record separately whether the stronger >=95.53% detectable-effect hypothesis and current global best 95.40% were exceeded.
- [x] Record all informational metrics and exact phase exposure, verify only `train.py` changed, then remove `run.log` before analysis completes.

## Code Changes

- **`train.py`**: Replace the static six-block specification with the fixed 1-2-3 sequence and update its human-readable label/config output. This moves one equal-MAC block from 64 channels at 32x32 to 256 channels at 8x8 while preserving the model's operator family and complete training protocol. No benchmark-only constructor parameter or dead configuration path remains in production.

No other tracked file may change. Temporary validation harnesses may exist only outside the repository and their commands/results will be copied into `03-execute.md`; they must be removed after use.

## Configuration Changes

- Static stage allocation: `(2, 2, 2)` -> `(1, 2, 3)` at unchanged widths `(64, 128, 256)`.
- Model parameters: 2,748,890 -> 3,855,578 (+40.3%).
- Conv/Linear MACs per image: 392,612,352 -> 392,612,352 (unchanged).
- Drop path remains `0.08 * global_block_index / 6`; block reallocation therefore changes stagewise regularization dose. Accuracy attribution is to the fixed architecture-plus-dose package, not pure capacity placement.
- All optimizer, LR, augmentation, timing, batch, precision, validation, seed, and evaluator settings remain identical to EXP-002.

## Execution Environment

- Method: local implementation/smokes, paired local latency preflight, then one local full run.
- Resources: physical GPU 0 only, exposed as logical CUDA device 0 through `CUDA_VISIBLE_DEVICES=0`; NVIDIA H20 with approximately 98 GB memory; existing `uv` environment; no dependency changes.
- Estimated runtime: under 2 minutes for invariant checks/preflight plus approximately 470 seconds for the full run; hard outer limit 600 seconds for the metric run.
- Log output: full metric stdout/stderr to repository-root `run.log`; preflight output captured directly into `experiments/010/03-execute.md`. `run.log` is transient and removed after results are recorded.
- Tool skill: local execution; no remote submission skill.

## Abort Criteria

- Physical GPU 0 is not the approximately 98 GB NVIDIA H20, or PyTorch does not see exactly the intended CUDA device under `CUDA_VISIBLE_DEVICES=0`.
- Any scope check shows a tracked modification outside `train.py`.
- Parent-source inventory/shared-prefix comparison, exact candidate block inventory, 3,855,578 parameter count, 392,612,352 MAC count, output shape, finite gradient, or layout/dtype smoke fails.
- A valid paired preflight exceeds 1.05x median latency, 1.08x p90 latency, projects fewer than 26,500 steps, or projects total runtime at/above 600 seconds. The first valid measurement is decisive. At most one full remeasurement is allowed only when the first run's parent alternating-round medians drift by more than 7.5%; record all first-run round values and the contamination decision before rerunning. No architecture or threshold may change.
- The full run raises an exception, OOM/CUDA error, NaN/Inf, stops making process/GPU progress for 120 seconds, or reaches the 600-second timeout. `PYTHONUNBUFFERED=1` makes log monitoring observable, but log-byte inactivity alone does not authorize killing an active process.
- Do not stop on low intermediate accuracy or loss alone unless values are nonfinite; intermediate test results cannot trigger a retry, alternative, or hyperparameter change.

## Verification Protocol

### Verification Procedure

1. From the repository root, verify the parent and threshold:
   `bash /root/david/.codex/plugins/cache/deoxys/tree-autoresearch/0.1.2/skills/shared/scripts/tree.sh show .tree-autoresearch/goals/maximize-cifar10-best-test-accuracy/04-results.tsv 002`
   Require `metric=95.23`; therefore the formal threshold is 95.33%.
2. Verify GPU identity and visibility before every GPU command:
   `nvidia-smi -i 0 --query-gpu=name,memory.total --format=csv,noheader`
   and `CUDA_VISIBLE_DEVICES=0 uv run python -c "import torch; print(torch.cuda.device_count(), torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory)"`.
   Require one visible PyTorch device named NVIDIA H20 and approximately 98 GB total memory.
3. Run static checks:
   `python -m py_compile train.py`, `git diff --check`, and `git diff --name-only a36dc09`.
   Require exit 0 and only `train.py` in the final output. Run the exact inventory/equivalence smoke described in Milestones 1-2 and require every assertion to pass before latency testing.
4. From the repository root, export the parent with `git show a36dc09:train.py > /tmp/exp010_parent_train.py`, then run the reviewed inline paired benchmark under `timeout 180s env CUDA_VISIBLE_DEVICES=0 uv run python -` using the exact harness defined in Milestones 2-3. The harness must add the repository root to `sys.path`, load the temporary parent with an explicit unique importlib module name, and reseed immediately before each model construction. Reject the candidate if any fixed latency/exposure/runtime gate fails; accuracy must not be queried during preflight. Remove the temporary parent source afterward.
5. After a passing preflight, run exactly once:
   `timeout 600s env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 uv run train.py > run.log 2>&1`.
   Treat exit 124 as timeout failure and any other nonzero exit as crash. If summary extraction is empty, inspect `tail -n 50 run.log` and classify the run as failed.
6. Extract core results with `grep "^best_test_acc:\|^peak_vram_mb:" run.log` and the full summary with `tail -n 12 run.log`. Formal improvement requires `best_test_acc >= 95.33%`, `training_seconds` in `[299.5,301.0]`, `total_seconds < 600`, `num_params = 3,855,578`, and all summary keys present. Record `num_steps >= 26500`, CutMix ratio in `[0.47,0.53]`, and `best_test_acc >= 95.53%` separately as mechanism/hypothesis checks; missing them does not override a valid formal improvement.
7. Count evaluation lines with `grep -c '^  eval ep' run.log` and compare with `num_epochs`; require equality. Scan errors with `rg -n -i 'traceback|cuda error|out of memory|(^|[^a-z])(nan|inf)([^a-z]|$)' run.log`; require no genuine error/nonfinite match.
8. Record the result, early trajectory, and mechanism/exposure audit in `03-execute.md`; verify `git diff --name-only a36dc09` is still only `train.py`; then remove `run.log` before advancing to analysis.

### Informational Metrics (Optional)

- `final_test_acc`: `grep '^final_test_acc:' run.log`.
- `final_test_loss`: `grep '^final_test_loss:' run.log`.
- `training_seconds`: `grep '^training_seconds:' run.log`.
- `total_seconds`: `grep '^total_seconds:' run.log`.
- `startup_seconds`: `grep '^startup_seconds:' run.log`.
- `peak_vram_mb`: `grep '^peak_vram_mb:' run.log`.
- `num_epochs`: `grep '^num_epochs:' run.log`.
- `num_steps`: `grep '^num_steps:' run.log`.
- `num_params`: `grep '^num_params:' run.log`.
- Architecture mechanism audit: startup config, exact block/projection/MAC inventory, paired latency ratio, projected steps, CutMix applied/eligible ratio, early smooth-loss records, and per-epoch accuracy trajectory from `run.log`.
