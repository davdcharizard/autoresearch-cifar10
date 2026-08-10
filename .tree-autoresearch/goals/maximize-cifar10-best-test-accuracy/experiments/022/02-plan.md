# Plan EXP-022: Fixed 288-channel final-stage taper
- **Created**: 2026-08-06

## Adversarial Idea-Review Resolution

- Rebase the developed taper from EXP-004 onto parent EXP-011: the formal threshold is `95.71%`, timing weights use EXP-011's 25,798-step path mix, and the complete cadence-31 full-state EMA is preserved and audited at the wider shapes.
- Treat the accuracy benefit as an unverified capacity bet. EXP-014 supplies implementation and H20 cost evidence at width 320 but no accuracy result; PyramidalNet is only a mechanism prior.
- Treat shape-dependent constructor and later DataLoader RNG changes as part of one deterministic seed-42 package. Do not copy parent submatrices, burn draws, isolate the parent data stream, try another width, or rerun a seed.
- Keep formal improvement (`best_test_acc >=95.71%`) separate from scientific support based on realized dose and the late EMA plateau. Tail diagnostics never change the tree verdict.

## Milestones

### Milestone 1: Implement the fixed taper
- [x] Change only residual blocks 5-6, final BatchNorm, and classifier input from width 256 to 288; retain six blocks, strides, projections, initialization, and forward control flow.
- [x] Update architecture metadata to truthfully report stage widths `64,128,288`; retain computed parameter reporting and keep MAC accounting external to production.
- [x] Preserve every inherited data, augmentation, loss, optimizer, schedule, drop-path, SAM, EMA, timing, seed, evaluation, and final-summary behavior.

### Milestone 2: Prove architecture and state correctness
- [x] Materialize exact parent commit `d68f73a` under `/tmp`, hash-check it, import both sources without invoking `main`, and reconcile block/stride/projection/Conv inventory. Require exactly `3,260,442` trainable parameters, `425,315,136` Conv/Linear MACs/image, output `(256,10)`, and shape changes confined to blocks 5-6, tail BN, and classifier.
- [x] Reset seed and prove two candidate constructions are bitwise identical. Record, but do not gate on, parent/candidate common-tensor and post-construction RNG equality because architecture-dependent draws are an acknowledged package confound.
- [x] Run CPU FP32 and candidate-only physical-GPU-0 BF16/channels-last forward, backward, and Nesterov smokes. Require finite logits/loss/BN state, finite nonzero gradients for every trainable tensor, and candidate-only peak allocation `<4096 MiB` after all optimizer, SAM, and EMA persistent state exists.
- [x] Exercise deterministic CutMix, six active and zero terminal drop-path draws, one ordinary update, production-faithful SAM replay/BN suppression/exact restore/update, and 30 consecutive cadence-31 EMA samples split 15 ordinary/15 SAM plus one full-state EMA evaluate/swap/restore. Require complete non-aliased state, exact restoration, preserved optimizer identities, zero RNG/state failures, and bounded steady-state allocation growth `<=1 MiB` after persistent diagnostics exist.

### Milestone 3: Run the sole accuracy-blind paired preflight
- [x] Confirm physical GPU 0 is the approximately 97,871 MiB NVIDIA H20 and expose only it. Record raw compute-app contexts; classify active contamination using `nvidia-smi pmon`, PID existence, UUID, utilization, and memory before rejecting a stale context.
- [x] Monkeypatch parent and candidate evaluators so any evaluation call raises. Run one paired, deterministic real-CIFAR conditioning trace and record finite losses plus activation/gradient norms only; every finite value passes and differences cannot select, stop, or alter the candidate.
- [x] Run five alternating-order warmed timing rounds per arm using fixed synthetic tensors and the exact EXP-011 operation mix: 100 early ordinary, 40 early CutMix, 20 late ordinary, 20 late SAM, and 40 eval-mode forwards per round, including optimizer, SAM, and cadence-31 EMA work. Weight charged paths by `10512/25798`, `10345/25798`, `2470/25798`, and `2471/25798`.
- [x] Report separate candidate/parent medians for ordinary, CutMix, SAM, and evaluation paths. Derive `early_ratio=(10512*r_ordinary+10345*r_cutmix)/20857`, `late_ratio=(2470*r_ordinary+2471*r_sam)/4941`, projected early/late steps `20857/early_ratio` and `4941/late_ratio`, and total as their sum. Evaluation timing is excluded from charged dose and enters only `projected_total=1.1+300+(447.9-300-1.1)*(projected_epochs/133)*r_eval`.
- [x] On the first complete numeric vector require parent drift `(max-min)/median <=0.03` for each charged path, each paired-ratio MAD/median `<=0.01`, weighted charged median ratio `<=1.10`, every charged-path maximum paired ratio `<=1.13`, projected total steps `>=23200`, complete epochs `floor(projected_steps/195) >=118`, projected EMA samples `160*projected_late_steps/4941 >=140`, and conservative projected total `<600s`. **Decisively failed: weighted ratio 1.160794, projected 22,214.78 steps / 113 epochs / 137.705 EMA samples.**
- [x] Hash and durably record the frozen preflight harness before launch. The first observed parent/candidate timing datum makes the run decisive even if the vector is incomplete; no correction follows partial timing output. Before any timing datum exists, only a Python/shell exception, missing/wrong source, or demonstrably malformed assertion permits one bounded harness-only correction that preserves all timing semantics and discloses prior output. **No preflight rerun or correction occurred.**

### Milestone 4: Execute one fixed-seed metric run
- [x] Only after all gates pass, remove stale `run.log`, reconfirm GPU 0, and launch exactly once with `timeout 600s env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 uv run train.py > run.log 2>&1`. **Skipped as required after decisive preflight rejection; no metric launch occurred.**
- [x] Monitor process, GPU, and log liveness. Abort only on exception, CUDA/OOM, nonfinite/integrity failure, EMA/SAM restoration failure, 120 seconds without process/GPU/log progress, or the outer timeout; never prune on finite loss, intermediate accuracy, or projected/realized dose. **Metric monitoring was not reached.**
- [x] Preserve raw evidence until the full summary and exact tail values are durably transcribed into `03-execute.md`, independently reviewed, analyzed, committed, and inserted. Never retry the metric or change width/hyperparameters in response to its result. **No `run.log` exists; all preflight values are transcribed below.**

### Milestone 5: Verify and classify
- [x] Apply integrity-first precedence: scope/evaluator/seed/state corruption is invalid/NaN; no complete result or nonzero process failure is crash/NaN; only exit 0 with intact evidence can be improvement/no-improvement. **Classified pre-metric rejection as crash/NaN under the tree convention.**
- [x] Require exit 0, charged seconds `[299.5,301.0]`, total `<600s`, exactly one evaluation per epoch, complete summary, `num_params=3260442`, only tracked `train.py` changed, and no traceback/CUDA/OOM/audit/RuntimeError/NaN/Inf signature. **Metric-run conditions skipped after the failed feasibility gate.**
- [x] Formal improvement is `best_test_acc >=95.71%` versus parent 95.61%. A valid lower result is one no-improvement and never authorizes another seed, width, or recipe adjustment. **No accuracy was queried.**
- [x] Scientific support additionally requires realized steps `>=23200`, EMA updates `>=140` with ordinary/SAM imbalance at most one, final accuracy within `0.15` points of best, final-16 EMA mean `>=95.593125%` (0.10 above EXP-011), and exact CutMix/SAM/EMA audits. Report mean/min/max/range and best-minus-tail premium as context; scientific shortfall cannot override the formal verdict. **Not evaluated because no metric run occurred.**

## Code Changes

- **`train.py`**: Change block specs `(128,256,2)` and `(256,256,1)` to `(128,288,2)` and `(288,288,1)`, change final BatchNorm and classifier input to 288, and update the architecture/config label to state widths `64,128,288`. No new production diagnostic, module, branch, dependency, or tracked file is needed.

The architecture edit necessarily changes seed-42 initialization draws and later DataLoader randomness. It is evaluated as a deterministic package, without copying overlapping weights or manufacturing parent stream parity.

## Configuration Changes

- Stage widths: `64/128/256 -> 64/128/288` (late 8x8 semantic capacity only).
- Trainable parameters: `2,748,890 -> 3,260,442` (`+511,552`, `+18.61%`).
- Conv/Linear MACs/image: `392,612,352 -> 425,315,136` (`1.0832954x`).
- Every other constant remains exactly EXP-011, including batch 256, seed 42, `PEAK_LR=0.2`, `WEIGHT_DECAY=1e-4`, `MAX_DROP_PATH=0.08`, CutMix, `SAM_RHO=0.05`, and EMA start/cadence/half-life.

## Execution Environment

- Method: local deterministic CPU checks, candidate-only and paired physical-GPU-0 checks, one accuracy-blind preflight, then at most one local metric run.
- Resources: `CUDA_VISIBLE_DEVICES=0`; one NVIDIA H20 with approximately 97,871 MiB; existing `uv` environment; no dependency installation.
- Estimated runtime: 2-4 minutes for checks/preflight and about 480-500 seconds for the metric process; all workloads are bounded and the metric run has a 600-second outer timeout.
- Log output: experiment-owned `/tmp/exp022_*` harnesses/logs with `PYTHONPYCACHEPREFIX=/tmp/exp022_pycache`; repository-root `run.log` only for the metric run. Exact evidence is copied to `03-execute.md` before cleanup.
- Tool skill: local execution; no remote submission or W&B.

## Abort Criteria

- Wrong physical/visible GPU; active foreign GPU process; tracked change outside `train.py`; evaluator, seed, budget, or evaluation-cadence change; any width other than fixed 288; any training-recipe retuning.
- Architecture/count/output, finite-gradient, CutMix/drop-path, SAM replay/BN/restore, EMA cadence/coverage/swap/restore, optimizer identity, RNG, or allocation assertion failure. Every finite conditioning trace value passes; there is no qualitative collapse gate.
- Candidate-only peak `>=4096 MiB`; steady-state live allocation growth `>1 MiB`; per-path preflight drift `>0.03`, ratio dispersion `>0.01`, weighted charged ratio `>1.10`, any charged-path max ratio `>1.13`, projected steps `<23200`, projected epochs `<118`, projected EMA samples `<140`, conservative total `>=600s`, or any nonfinite/integrity failure.
- Metric exception, CUDA/OOM, nonfinite/integrity signature, 120 seconds without liveness, or 600-second timeout. Intermediate finite metrics and realized dose do not trigger pruning.
- Once any parent/candidate timing datum or metric result exists, it is never rerun. A bounded harness correction is allowed only before the first timing datum and only for an execution defect rather than a measured failure; preserve the frozen harness hash and disclose all partial output.

## Verification Protocol

### Verification Procedure

1. Run `bash /root/david/.codex/plugins/cache/deoxys/tree-autoresearch/0.1.3/skills/shared/scripts/tree.sh show .tree-autoresearch/goals/maximize-cifar10-best-test-accuracy/04-results.tsv 011`; require parent metric `95.61`, hence threshold `95.71`, and reconcile EXP-011's 25,798 steps, 133 evaluations, 160 EMA samples, and final-16 mean 95.493125 from its report.
2. Before each GPU workload run `nvidia-smi -i 0 --query-gpu=name,memory.total,uuid,utilization.gpu,memory.used --format=csv,noheader`; under `CUDA_VISIBLE_DEVICES=0` require `torch.cuda.device_count()==1` and `torch.cuda.get_device_name(0)=='NVIDIA H20'`. Cross-check compute-app anomalies with `nvidia-smi pmon -i 0 -c 1` and `ps`.
3. Run `PYTHONPYCACHEPREFIX=/tmp/exp022_pycache uv run python -m py_compile train.py`, `git diff --check`, `git diff --name-only d68f73a`, and `git status --short --untracked-files=all`; require exactly `train.py` as the tracked code change. Also compare the zero-context diff against an explicit allowlist: exactly the two block-spec lines, tail BN line, classifier line, and architecture/config metadata strings may differ from `d68f73a`; reject every other changed line. Harnesses and snapshots stay under `/tmp`.
4. Hash-check `git show d68f73a:train.py` materialized in `/tmp`, run the Milestone-2 smokes with bounded timeouts, freeze and hash the preflight harness, and run it once under `timeout 420s env CUDA_VISIBLE_DEVICES=0`. Any incomplete result after a timing datum is pre-metric crash/NaN, not permission to shrink or rerun.
5. Only after every gate passes, remove stale `run.log` and launch the exact Milestone-4 command once. Preserve exit status and raw evidence.
6. Require `rg -c '  eval ep' run.log == num_epochs`, live+EMA evaluation sources equal epochs, and zero unexplained matches from `rg -n -i 'traceback|cuda error|out of memory|ema_audit_failed|runtimeerror|(^|[^a-z])(nan|inf)([^a-z]|$)' run.log`.
7. Extract the complete summary, architecture/config metadata, CutMix/SAM/EMA audits, and final 16 evaluation values. Compute mean/min/max/range, final gap, and best-minus-tail premium; apply formal and scientific thresholds separately. Training time is structurally at least 300 seconds, while `<=301.0` detects excessive final-step overshoot.
8. Run one raw-result adversarial audit, correct evidence wording, complete analysis/commit/tree insertion, verify commit scope, then delete `run.log`, parent snapshot, and all `/tmp/exp022_*` artifacts before the next loop.

Every command named above exists in the current environment (`uv`, `timeout`, `nvidia-smi`, `rg`, and `tree.sh` were checked before writing this plan).

### Informational Metrics (Optional)

- Summary: best/final accuracy, final loss, training/total/startup seconds, peak VRAM, epochs/evaluations, steps, and parameters.
- Architecture/dose: widths, MACs, CutMix applied/eligible, SAM applied/eligible/start, EMA updates/parity/decays/distances/swaps/restores, and paired relative SAM perturbation.
- Stability: final-16 EMA values and progress, mean/min/max/range, final gap, best epoch, and `best_test_acc - final16_mean`.
