# Plan EXP-009: Isolated BF16 Autocast at Batch 256
- **Created**: 2026-07-24

## Milestones

### Milestone 1: Isolated BF16 implementation and static audit
- [x] Create `autoresearch/maximize-cifar10-test-accuracy-009` from clean accepted commit `eb08811`.
- [x] Modify only `train.py` to add a named BF16 autocast dtype, one setup log, and an autocast context around training forward/loss while keeping mixup interpolation, backward, SGD, master state, and evaluation FP32.
- [x] Run `uv run python -m py_compile train.py`, inspect the full diff against the treatment allowlist, and verify `prepare.py` is byte-identical to `eb08811`.

### Milestone 2: Evaluator-free production-path feasibility gate
- [x] Confirm one NVIDIA H20, local CIFAR files present, and CUDA BF16 autocast support without network access.
- [x] In one fail-closed evaluator-stubbed process, benchmark matched FP32 and final BF16 batch-256 steps using the full production `t0`-through-synchronize body, 25 warmup steps per path, and the preregistered six-window order `FP32, BF16, BF16, FP32, FP32, BF16`.
- [x] Require finite loss, `[256,10]` logits, FP32 parameters/gradients/momentum/loss, BF16 treatment logits, timing CV <=5%, BF16 throughput >=1.10x FP32, and calibrated exposure >=156.09 passes. If a semantic or stable throughput gate fails, do not run a scored fallback.

### Milestone 3: Single fixed-seed scored run
- [x] Remove stale `run.log` and execute exactly once with `timeout 600s uv run train.py > run.log 2>&1`.
- [x] Monitor startup, finite loss, throughput, the 65% transition, and completion without using interim accuracy to alter the run.
- [x] Capture exit status and require a complete summary from the single H20 run.

### Milestone 4: Protocol and result audit
- [x] Verify 300.0 rounded counted seconds, total <=600 seconds, `Device: cuda`, BF16 setup log, 691,674 parameters, batch 256, one correct mixup transition, and at most one evaluation per epoch.
- [x] Record realized steps/passes, best/final gap, final loss, VRAM, epochs, and evaluation count; compare exposure to EXP-002's 27,735 steps / 141.9 passes and the 162.8-pass proposal projection.
- [x] Accept only `best_test_acc >=94.17%`; report a one-checkpoint best spike with worse final accuracy/loss as weak mechanism evidence even if it formally passes.

## Code Changes
- **`train.py`**: add `TRAIN_AUTOCAST_DTYPE = torch.bfloat16`; print it once during setup; restructure the existing training forward/loss block so mixup interpolation and hard-label input selection occur outside `torch.autocast`, model forward and cross entropy occur inside CUDA-enabled BF16 autocast, and `loss.backward()` / `optimizer.step()` remain outside. Evaluation remains the frozen FP32 `Eval.evaluate` path. No other file changes.

## Configuration Changes
- `TRAIN_AUTOCAST_DTYPE`: new `torch.bfloat16` training compute dtype.
- `BATCH_SIZE=256`, `LR=0.2`, `MIN_LR=0.002`, `WARMUP_FRACTION=0.05`, `MOMENTUM=0.9`, `WEIGHT_DECAY=5e-4`, `MIXUP_ALPHA=0.2`, `MIXUP_END_FRACTION=0.65`, model, seed, transforms, loader, and evaluation cadence: unchanged.
- No GradScaler, model cast, optimizer-state cast, channels-last, fused SGD, compilation, batch change, or LR scaling is permitted.

## Execution Environment
- Method: fully local/offline. Run an evaluator-free synthetic preflight first, then one scored `timeout 600s uv run train.py > run.log 2>&1` only if all gates pass.
- Resources: exactly one NVIDIA H20; existing PyTorch/torchvision environment and local CIFAR-10 files; no dependency installation, network, remote service, or GitHub operation.
- Estimated runtime: preflight under one minute; scored run about 345 seconds total with 300 seconds counted training and additional legal evaluations.
- Log output: scored stdout/stderr captured only in project-root `run.log`; preflight numeric results recorded directly in `03-execute.md`. Remove `run.log` after analysis.
- Tool skill: none; local execution only.

## Abort Criteria
- Do not launch the scored run if the evaluator-free preflight fails any semantic, stability, CV, 1.10x throughput, or 156.09-pass projection gate.
- During the scored run, stop/classify on timeout exit 124, Python traceback, CUDA/OOM error, unsupported autocast kernel, non-finite loss, missing H20, or wall time reaching 600 seconds.
- Do not stop for low interim accuracy and do not retry a valid sub-threshold result. Do not rescue failure with FP16, GradScaler, batch 512, another layout, changed LR, relaxed determinism, or seed reroll.

## Verification Protocol

### Verification Procedure
1. Query the baseline with `bash /root/david/.codex/plugins/cache/deoxys/linear-autoresearch/3.0.3/skills/shared/scripts/exp-index.sh baseline .autoresearch/goals/maximize-cifar10-test-accuracy/04-results.tsv`; require `baseline=94.07`, so formal success is `best_test_acc >=94.17%`.
2. Before implementation/scoring, require `nvidia-smi --query-gpu=name --format=csv,noheader` to return exactly one `NVIDIA H20`, verify `data/cifar-10-batches-py` already exists, and confirm `torch.cuda.is_bf16_supported()` in the current `uv` environment. No download or network fallback is allowed.
3. Run `uv run python -m py_compile train.py`. Require `git diff --name-only eb08811` to list only `train.py` and `git diff --exit-code eb08811 -- prepare.py` to exit 0. Review the complete `git diff eb08811 -- train.py` and reject hunks outside this allowlist: one BF16 dtype constant, one setup log, pre-autocast `train_inputs` selection/mixup interpolation, one training-only autocast forward/loss context, and removal of the superseded duplicate forward/loss statements. Reject any evaluator, seed, schedule, model, optimizer, loader, cadence, summary, or metric-reporting change.
4. Run one inline fail-closed preflight. Import `prepare`, replace `prepare.Eval` before importing final `train.py` with a dummy whose constructor records use and whose `evaluate()` always raises, then assert `train.evaluator` is that exact dummy and no real test loader was constructed. Use fixed synthetic pinned CPU inputs/targets. Initialize one production model, clone its state into separate FP32 and BF16 models, construct identical selective SGD/Nesterov optimizers, and preserve independent but initially identical CPU/CUDA RNG states for each path so interleaving does not alter either path's stochastic sequence.
5. Factor one shared timed-step function whose only treatment argument is autocast enabled/disabled. Its timed `t0`-through-`torch.cuda.synchronize()` body must reproduce production operations in order: nonblocking input/target copies; LR calculation and both param-group writes; progress/mixup branch; `optimizer.zero_grad(set_to_none=True)`; Beta sampling and `randperm` through production `mixup_batch`; training forward/loss under enabled or disabled autocast; the Boolean `torch.isfinite(loss)` guard; backward; optimizer step; and final synchronize. Do not include post-sync EMA logging math. Warm each independently for 25 complete steps, then measure exactly three 50-step windows per continuing path in order `FP32-A, BF16-A, BF16-B, FP32-B, FP32-C, BF16-C`, restoring/updating that path's private RNG state around each window. This order is fixed before timing and may not be changed after output.
6. Record every window in ms/step. Define each path's center as the median of its three window means, BF16 throughput ratio as `FP32_median_ms / BF16_median_ms`, and variability as population CV `statistics.pstdev(window_ms) / statistics.mean(window_ms) * 100`. Record peak memory and calibrated passes `141.9 * ratio`. After a semantic treatment step, require `[256,10]` logits, finite FP32/BF16 loss, BF16 treatment logits, FP32 mixed loss, every model parameter FP32, every non-`None` parameter gradient FP32, and every created momentum buffer FP32. Require CV <=5% for both paths, ratio >=1.10, projected passes >=156.09, and no OOM. This gate selects only operational feasibility and never inspects test data or accuracy.
7. Confirm no stale log, then run the sole scored command: `timeout 600s uv run train.py > run.log 2>&1`. Require exit 0; on nonzero inspect `tail -n 50 run.log` and classify according to the abort criteria without a result-conditioned alternative.
8. Require exactly one logged `Device: cuda` and BF16 dtype line, model `WRN-16-2`, 691,674 parameters, a complete summary, rounded `training_seconds=300.0`, `num_steps<64000`, finite loss, and `total_seconds<=600`. Require one mixup-disable line near 195.0 seconds with LR 0.0612. Extract evaluated epochs from `run.log`; require uniqueness and accepted every-fifth-epoch plus terminal cadence.
9. Compute realized passes as `num_steps * 256 / 50000` and compare to 141.9 accepted, 156.09 minimum mechanism target, and 162.8 projection. Record best/final accuracy, final loss versus 0.2432, best/final gap, number of evaluations, peak VRAM, epochs, steps, and total wall time. Attribution must remain to the combined BF16-numerics-plus-throughput treatment, not exposure alone.
10. Necessary-condition verdict: require `best_test_acc >=94.17%`. A lower score is no-improvement regardless of throughput or loss. A formal pass driven by one transient checkpoint remains an improvement under the frozen metric, but the report must explicitly qualify weak final/loss support.

### Informational Metrics (Optional)
- `peak_vram_mb`, `training_seconds`, `total_seconds`, `num_epochs`, `num_steps`, `num_params`: final summary in `run.log`.
- `final_test_acc`, `final_test_loss`, best/final gap, best epoch: evaluation records and final summary in `run.log`.
- realized passes and exposure ratio: calculate from final `num_steps` and compare to EXP-002.
- BF16/FP32 preflight window times, CV, throughput ratio, dtype semantics, and projected passes: preflight output recorded inline in `03-execute.md`.
