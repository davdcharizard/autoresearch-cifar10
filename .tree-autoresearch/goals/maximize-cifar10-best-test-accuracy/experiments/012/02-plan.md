# Plan EXP-012: CutMix-complementary GPU Cutout
- **Created**: 2026-08-06

## Adversarial Review Resolution

- The frozen research goal and tree contract require `best_test_acc >=95.71%`; this remains the formal verdict even though a selected maximum is noisy. Stable-tail evidence is therefore strengthened rather than substituted: the final-16 EMA mean must reach `95.74%` to support the hypothesized approximately 0.25-point plateau lift, while a formal pass with mean below `95.59%` is explicitly selection-only evidence that falsifies the stable-effect hypothesis.
- Timing uses one decisive preflight with no rerun: parent round dispersion must be tight and the paired candidate/parent ratio is the controlling gate. Projected steps are retained only as an equivalent audit against EXP-011's realized 25,798-step exposure.
- Raw execution evidence remains available through analysis and tree insertion. Every adversarial review is Claude-only; if Claude is unavailable, execution pauses for the user rather than using a fallback reviewer.

## Milestones

### Milestone 1: Implement reference-geometry GPU Cutout
- [x] Add fixed `CUTOUT_SIZE=16` and `CUTOUT_SEED=43`; assert the current `(1,1,1)` normalization standard deviations so normalized zero remains dataset-mean fill.
- [x] Build a deterministic setup-time bank for all 32x32 uniformly sampled center coordinates. Each mask clips `[center-8, center+8)` to the image, has shape `[1,32,32]`, and is indexed row-major as `center_y*32+center_x`. Assert 1,024 unique masks, mean masked area exactly 196 pixels (19.140625%), and expected min/max areas 64/256.
- [x] Preallocate CUDA center `[256,2]`, index `[256]`, FP32 channels-last mask bank `[1024,1,32,32]`, FP32 channels-last selected-mask `[256,1,32,32]`, and masked-area audit tensors. Use a private seed-43 CUDA generator, `centers.random_(0,32,generator=...)`, row-major index arithmetic, `torch.index_select(...,out=...)`, and one in-place broadcast multiply against the FP32 channels-last inputs; avoid Boolean-mask promotion and preserve hard labels.
- [x] Inside the existing lexical `if progress < CUTMIX_END` block, apply Cutout only in the `else` branch of the unchanged parent CutMix selection. Selected CutMix batches consume no Cutout RNG; late batches cannot reach Cutout code. Keep Cutout work and its GPU-side area audit before the existing synchronization so all training work is charged.

### Milestone 2: Preserve the complete EXP-011 package and add audits
- [x] Leave WRN topology/initialization, DataLoader, crop/flip, CutMix gate/geometry/permutation/loss, optimizer, LR, drop path, SAM, charged-time EMA, evaluator, seed 42 streams, and once-per-epoch routing unchanged.
- [x] Add Cutout counters for calls/images, actual masked pixels/fraction, early complement identity, generator seed, center support, mask-bank statistics, and share of early/all steps. Require `cutmix_applied + cutout_applied == cutmix_eligible`; the lexical branch structure is the primary proof that late Cutout is impossible.
- [x] Preserve all existing CutMix/SAM/EMA terminal audits and complete final summary. Add a terminal failure path that still prints Cutout diagnostics before a nonzero exit if complement, geometry, finite-state, or parent-integrity checks fail.
- [x] Static checks must show only `train.py` differs from parent commit `d68f73a`; no dependency, evaluator, or seed-selection change is allowed.

### Milestone 3: Pass deterministic correctness gates
- [x] On CPU, enumerate all mask-bank entries and verify clipped reference geometry, row-major mapping, uniqueness, min/mean/max area, Boolean keep semantics, mean fill, outside-pixel bit identity, target identity, and channels-last preservation.
- [x] Prove private-generator determinism and parent-stream isolation: same seed-43 center sequence repeats; global CPU/CUDA, DataLoader, CutMix CPU/CUDA, and drop-path RNG states do not advance from Cutout setup/helper operations.
- [x] Simulate forced CutMix/non-CutMix/late decisions. Require selected CutMix pixels, targets, lambda, parent RNG states, and counters to be bitwise parent-identical; non-CutMix early batches always Cutout; Cutout calls/centers remain unchanged across intervening CutMix batches; late calls remain zero.
- [x] On the full WRN BF16/channels-last path, verify finite loss/gradients, one optimizer update, parent drop-path draw count, channels-last preservation, and valid Cutout area. Separately re-run production-faithful SAM perturbation/replay/BN/restore and EMA update/swap checks to prove the new setup state does not enter optimizer, SAM, EMA, or evaluator ownership.

### Milestone 4: Pass the physical-GPU-0 feasibility gate
- [x] Confirm physical GPU 0 is the approximately 97,871 MiB NVIDIA H20 and expose only it with `CUDA_VISIBLE_DEVICES=0` before every GPU command.
- [x] Benchmark the exact helper for at least 1,000 calls after warmup, including RNG, mask selection, in-place multiply, and area audit; require deterministic replay, stable allocations, and no layout/RNG failure.
- [x] Run five alternating parent/candidate BF16/channels-last rounds. Each round measures at least 100 synchronized parent-clean versus candidate-Cutout optimizer steps; across the full harness also include at least 200 parent/candidate CutMix steps, 100 production-faithful SAM steps, 30 cadence-31 EMA updates, and one exact EMA evaluation swap per arm.
- [x] For each round compute medians for clean/Cutout, CutMix, late ordinary, and SAM. Weight the full-run estimate by EXP-011's fixed path shares: Cutout/parent-clean `10512/25798`, CutMix `10345/25798`, late ordinary `2470/25798`, and SAM `2471/25798`. The single measurement is decisive with no rerun: require parent-only weighted-round drift `max/min-1 <=0.03`, paired candidate/parent weighted-ratio dispersion `MAD/median <=0.005`, candidate/parent weighted median ratio `<=1.01`, the equivalent exposure audit `25798/ratio >=25500`, projected total runtime `<600s`, and every correctness assertion. Report isolated candidate peak allocation informationally; only an allocation failure blocks execution.

### Milestone 5: Execute one fixed-seed metric run
- [x] Remove stale `run.log`, reconfirm physical GPU 0, and launch exactly once with `timeout 600s env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 uv run train.py > run.log 2>&1`.
- [x] Monitor process/GPU/log progress without pruning on intermediate accuracy. Abort only on exception, CUDA/OOM, nonfinite state, Cutout complement/late-call failure, SAM/EMA integrity failure, 120 seconds without process/GPU/log progress, or the outer timeout.
- [x] Preserve raw run and preflight evidence through analysis and tree insertion. Durably transcribe the complete summary, all CutMix/Cutout/SAM/EMA audits, evaluation-source counts, final-16 EMA accuracies and mean/range, and preflight measurements. Run Claude as the sole adversarial result reviewer against both raw and durable evidence; if Claude is unavailable, pause for the user and never fall back to another reviewer.

### Milestone 6: Verify and classify
- [x] Require exit 0, charged seconds in `[299.5,301.0]`, total runtime `<600s`, one evaluation per epoch, 2,748,890 parameters, complete summary, only `train.py` changed, and no error signature.
- [x] Mechanism dose requires at least 25,500 realized steps, at least 145 EMA samples, Cutout on every early non-CutMix batch, actual mean masked area in `[195.8,196.2]` pixels, finite nonzero EMA distances, and CutMix/SAM/EMA parent-integrity audits. A dose shortfall is `no-improvement`; complement/RNG/state/evaluation corruption is `invalid` with metric `NaN`.
- [x] Classification precedence is fixed: any integrity signature makes the run `invalid` with metric `NaN`, even if a summary printed; any nonzero exit without an integrity signature is crash/`NaN`; only exit 0 can be improvement or no-improvement.
- [x] Formal improvement is `best_test_acc >=95.71%` versus parent EXP-011 at 95.61. For causal interpretation require at least 16 EMA-source evaluations and record their charged-progress span: mean `>=95.74%` supports the hypothesized stable lift; `95.59-95.73%` is partial/unresolved; below `95.59%` falsifies the stable-effect hypothesis, including when the formal maximum passes. Below 95.71 is a single valid no-improvement, not a retry trigger.

## Code Changes

- **`train.py`**: Add reference-geometry center-sampled Cutout mask-bank construction, private CUDA RNG/buffers, complementary early-branch application, charged area accounting, startup config, integrity checks, and compact terminal audits. All EXP-011 online training, SAM, EMA, and evaluation code remains intact except for the new non-CutMix input mutation and reporting.

No other tracked file may change. Accuracy-blind smoke/benchmark harnesses and parent snapshots may live only under `/tmp` during execution and must remain available through analysis/tree insertion, then be removed before advancing to the next experiment.

## Configuration Changes

- `CUTOUT_SIZE`: new `16`, interpreted as a nominal `[center-8,center+8)` square clipped at CIFAR image edges. The direct WideResNet/CIFAR result is an all-image no-CutMix reference and an upper-bound dose, not evidence for the marginal effect in this complementary schedule.
- `CUTOUT_SEED`: new private `43`, deliberately separate from global and CutMix seed-42 streams. It fixes augmentation randomness, not model-selection randomness.
- Cutout dose: every early non-CutMix batch, no second probability. This is about 50.4% of early and 40.7% of all parent steps, materially below literature all-image Cutout while retaining all CutMix batches.
- Fill: normalized zero, guarded by `std==(1,1,1)`, corresponding exactly to per-channel dataset mean in pixel space.
- All parent hyperparameters remain read-only and frozen, including the existing `TIME_BUDGET_S=300`; this is not a configuration edit.

## Execution Environment

- Method: local CPU correctness smokes, local paired GPU preflight, then one local metric run.
- Resources: physical GPU 0 only through `CUDA_VISIBLE_DEVICES=0`; NVIDIA H20 with approximately 98 GB; existing `uv` environment; no new dependency.
- Estimated runtime: under 4 minutes for smokes/preflight plus approximately 450 seconds for the full run; each command has an explicit timeout, with the metric run capped at 600 seconds.
- Log output: metric stdout/stderr captured to repository-root `run.log`; exact preflight and terminal evidence copied into `experiments/012/03-execute.md`, with raw evidence retained through analysis/tree insertion and deleted before the next experiment.
- Tool skill: local execution; no remote submission tool.

## Abort Criteria

- Wrong physical/visible GPU, a tracked change outside `train.py`, or any change to the frozen evaluator, dataset stream, seed 42 parent streams, budget, validation cadence, model, optimizer, LR/drop path, CutMix, SAM, or EMA semantics.
- Mask-bank geometry/mapping/area failure; helper changes channels-last layout or targets; selected CutMix consumes Cutout RNG; complement identity fails; any Cutout call occurs at progress >=0.75; Cutout advances a parent RNG stream.
- Single-preflight parent drift >0.03, paired candidate/parent weighted-ratio `MAD/median >0.005`, median latency ratio >1.01, equivalent projected exposure <25,500 steps, projected total >=600 seconds, allocation failure, allocation growth after helper warmup, or any state/RNG/BN/SAM/EMA/evaluation assertion fails. There is no timing rerun or metric-aware retry.
- Full-run exception, CUDA/OOM/nonfinite error, complement/late-call/restoration/coverage/RNG failure, 120 seconds without process/GPU/log progress, or 600-second timeout. Intermediate accuracy/loss cannot trigger abort unless nonfinite.

## Verification Protocol

### Verification Procedure

1. Confirm parent and thresholds with `bash /root/david/.codex/plugins/cache/deoxys/tree-autoresearch/0.1.2/skills/shared/scripts/tree.sh show .tree-autoresearch/goals/maximize-cifar10-best-test-accuracy/04-results.tsv 011`; require parent metric 95.61, formal threshold 95.71, and reference tail mean 95.493125 from EXP-011's report.
2. Before every GPU command run `nvidia-smi -i 0 --query-gpu=name,memory.total --format=csv,noheader` and `CUDA_VISIBLE_DEVICES=0 uv run python -c "import torch; print(torch.cuda.device_count(), torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory)"`; require one visible NVIDIA H20 and approximately 97,871 MiB.
3. Run `python -m py_compile train.py`, `git diff --check`, `git diff --name-only d68f73a`, and `git status --short --untracked-files=all`; require exit 0, exactly `train.py` as the tracked code change, and no unexpected repository-local harness/log. Run the Milestone-3 geometry, RNG, complement, BF16, SAM, EMA, and audit smokes under bounded timeouts.
4. Materialize the parent with `git show d68f73a:train.py > /tmp/exp012_parent_train.py`, verify its hash against a fresh `git show`, and import parent/candidate under distinct non-`__main__` names with repository root on `sys.path`. Assert both intentionally bind the same read-only `prepare` module, parent-owned constants match, neither `main` executes, and all Cutout state is created only by explicit candidate setup inside `main`/the harness. Run the exact five-round GPU-0 latency preflight under `timeout 300s`, then measure isolated candidate peak allocation in a separate subprocess. Apply the one-shot gates in Milestone 4 without querying accuracy.
5. After the passing preflight, remove stale `run.log` and run once: `timeout 600s env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 uv run train.py > run.log 2>&1`. Apply classification precedence: any integrity signature is invalid/`NaN`; otherwise a nonzero exit is crash/`NaN`; only exit 0 reaches improvement/no-improvement classification.
6. Require `rg -c '  eval ep' run.log` to equal `num_epochs`, and live+EMA source counts to equal the same value. Scan `rg -n -i 'traceback|cuda error|out of memory|ema_audit_failed|cutout_audit_failed|runtimeerror|(^|[^a-z])(nan|inf)([^a-z]|$)' run.log`; any match is investigated and classified under the fixed rules rather than trusting metric grep alone.
7. Extract `grep '^best_test_acc:\|^peak_vram_mb:' run.log`, the complete summary, CutMix/Cutout/SAM/EMA audits, and final 16 EMA-source evaluations with charged progress. Verify all Milestone-6 integrity and dose conditions. `best_test_acc >=95.71` is improvement; otherwise it is no-improvement. Classify the final-16 mean as stable (`>=95.74`), partial (`95.59-95.73`), or falsified (`<95.59`), require all 16 values and their progress span, and do not claim Cutout alone caused a selected-maximum delta.
8. Run a Claude-only adversarial result review while raw evidence still exists; if Claude is unavailable, pause for the user. Copy the verdict and exact metrics to durable artifacts, complete analysis and tree insertion, verify only `train.py` changed, then remove `run.log`, parent snapshot, and temporary harness/log files before advancing to another experiment.

### Informational Metrics (Optional)

- `final_test_acc`, `final_test_loss`, `training_seconds`, `total_seconds`, `startup_seconds`, `peak_vram_mb`, `num_epochs`, `num_steps`, `num_params`: terminal summary.
- Cutout mechanism: calls/images, actual masked pixels/fraction and mean area, early/all-step shares, complement equality, generator seed, center support, bank min/mean/max.
- Parent mechanism preservation: CutMix applied/eligible ratio, SAM applied/eligible and first boundary, EMA updates/parity/decays/distances, evaluation-source and exact-restore counts.
- Stability context: final-16 EMA-source accuracy values with charged-progress span, mean/range, final accuracy/loss, and best epoch.
