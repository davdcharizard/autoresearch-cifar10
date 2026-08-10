# Plan EXP-015: Confidence-attenuating soft-target Poly-1
- **Created**: 2026-08-06

## Adversarial Idea-Review Resolution

- Fix exactly one operating point: `POLY1_EPSILON=-0.25`. Do not sweep, proxy-select, anneal, phase-tune, or change sign/magnitude after any conditioning, timing, or accuracy observation.
- Use negative soft-target Poly-1 for every optimizer-driving gradient. Preserve plain hard CE only for the first/base SAM pass that defines the inherited rho-0.05 adversarial direction; use hard Poly-1 at perturbed weights for the sole optimizer gradient.
- For hard labels the exact multiplier is `1-0.25*p_y` in `[0.75,1]`. For CutMix there is no scalar multiplier bound: verify the full vector formula and treat ratios near a vanishing CE gradient as descriptive only.
- Keep `F.cross_entropy` unchanged and add one FP32 `softmax` for the Poly term. Do not replace parent CE with a different FP32 NLL implementation merely to reuse probabilities.
- Treat any gain as the fixed negative-Poly descent + CutMix + CE-adversary SAM + EMA package. A null does not select another coefficient or reject the whole PolyLoss family.

## Milestones

### Milestone 1: Implement the fixed loss intervention only
- [x] Add `POLY1_EPSILON=-0.25` and a sparse hard/soft-target loss helper computing parent `F.cross_entropy` plus `epsilon*(1-q dot softmax(logits.float()))`. Use the existing area-corrected CutMix lambda and no dense target tensor, persistent state, RNG, or host synchronization.
- [x] Route primary hard non-SAM and CutMix calls through Poly-1. On scheduled SAM steps keep the primary/base pass exactly hard CE, clear those gradients after perturbation, replay RNG and suppress BN as in the parent, then use hard Poly-1 on the perturbed second pass before exact restore and the sole optimizer update.
- [x] Add four lightweight integer counters: `ordinary_poly`, `cutmix_poly`, `sam_ascent_ce`, `sam_descent_poly`. Increment them inside the actual helper/CE call sites, never infer them from scheduled step categories. In the final audit require `ordinary_poly+cutmix_poly+sam_ascent_ce==num_steps`, `sam_ascent_ce==sam_descent_poly==sam_applied_batches`, `cutmix_poly==cutmix_applied_batches`, Poly calls equal `num_steps`, total loss calls equal `num_steps+sam_applied_batches`, and empty CutMix/SAM intersection.
- [x] Print truthful fixed-loss configuration and terminal counter audit. Add only outside-charged evaluation `charged_s`/`progress` and terminal existing debiased train-loss diagnostics needed for tail analysis; add no heavy probability/gradient statistics or `.item()` inside the charged step.
- [x] Preserve model/initialization, parameter count 2,748,890, data/augmentation and generators, optimizer/LR/decay, CutMix, drop path, SAM schedule/rho/replay, EMA schedule/state/evaluator, seed, timing boundaries, and once-per-epoch evaluation. Each epoch evaluates exactly one inherited source: live before EMA activation or full-state EMA afterward, never both.

### Milestone 2: Prove loss mathematics and parent mechanics
- [x] Materialize and hash-check exact parent commit `d68f73a` under `/tmp`; import parent/candidate without `main`, monkeypatch both evaluator objects immediately to raise, and prove candidate construction/model/RNG state equals parent before the new helper is called.
- [x] In FP64, compare hard Poly loss/autograd gradients with `(1-0.25*p_y)*(p-e_y)` at probabilities near uniform, 0.5, 0.9, and endpoints. Require positive multiplier `[0.75,1]` within `rtol=1e-10` in FP64 and demonstrate the loss is nonnegative.
- [x] For distinct/same-class CutMix at lambda `{0,0.01,0.25,0.5,0.7,0.75,0.99,1}`, compare sparse helper loss/gradient to dense `q`, weighted constituent losses, and `p-q-0.25*p*(q dot p-q)`. Check the lambda-0.7 fixed point near 0.678 and finite gradient magnitude, but impose no aggregate ratio bound.
- [x] With test-only epsilon zero, require parent CE loss/gradients and RNG states to match. Run CPU FP32 and physical-GPU-0 BF16/channels-last batch-256 forward/backward; require finite nonnegative loss, `(256,10)` logits, finite nonzero gradients for every trainable tensor, unchanged state inventory, and candidate-only peak `<1350 MiB` after constructing optimizer, SAM snapshots, and complete EMA state.
- [x] Exercise deterministic CutMix, six active/zero terminal drop-path draws, one ordinary Poly update, one CutMix Poly update, and one scheduled CE-ascent/Poly-descent SAM step. Require CE ascent equality to parent, zeroed ascent gradients before descent, rho-0.05 perturbation, replayed drop masks, one BN update, exact parameter restore, one momentum update, and no CutMix/SAM overlap. Keep modules in training mode on the second pass and set only `track_running_stats=False`, so batch statistics are used without a second running-stat update; never switch BN to eval mode.
- [x] Exercise 30 EMA samples selected every 31 one-based optimizer steps across both parities and one success/failure-path full-state swap/restore. Require identical parent eligibility/state semantics, complete coverage, exact restoration, optimizer identity, and zero RNG/state/mode failure. Every EMA count/floor in this plan means cadence-31 optimizer-step samples, never epochs.

### Milestone 3: Run the one complete accuracy-blind GPU-0 gate
- [x] Before every GPU command confirm physical GPU 0 is the approximately 97,871 MiB NVIDIA H20 and expose exactly one visible H20 using `CUDA_VISIBLE_DEVICES=0`.
- [x] With evaluator guards installed before any trace, run one 200-step paired real-CIFAR conditioning trace. Share each materialized transformed batch, CutMix choice/geometry/permutation, and replayed CUDA drop-path state. Require structural stream/counter equality, finite losses/gradients/state, hard multiplier range, and soft vector-formula spot checks; loss/norm differences are report-only and cannot select epsilon or abort if finite.
- [x] Run all five warmed alternating-order timing rounds per arm in fixed preregistered order, with exactly 100 early ordinary, 40 early CutMix, 20 late ordinary, 20 late SAM, and 40 fixed-synthetic eval-mode forwards per round. Include actual optimizer, CE/Poly loss paths, SAM replay, cadence-31 EMA work, and production synchronization; never iterate either test loader. No round may be dropped, reordered, or replaced before cross-round MAD is computed.
- [x] Weight only charged training paths by EXP-011 counts `10512/25798`, `10345/25798`, `2470/25798`, and `2471/25798`; synthetic evaluation timings are informational because evaluation is outside the 300-second charged budget. After all five fixed rounds form the first complete vector, require parent drift `(max-min)/median<=0.03`, paired ratio MAD/median `<=0.005`, median candidate/parent ratio `<=1.01`, every round ratio `<=1.02`, projected steps within the fixed 300 charged seconds `25798/ratio>=25300`, epochs `floor(steps/195)>=130`, cadence-31 EMA samples `160*steps/25798>=155`, conservative end-to-end total `447.9*max(1,ratio)<600s`, and binding candidate-only peak `<1350 MiB`.
- [x] The complete numeric gate is decisive. Any numeric latency/dispersion/dose/memory/stability failure creates a pre-metric `crash/NaN` leaf; never rerun timing, optimize the helper, remove audits, or change sign/magnitude/path scope. Harness correction is allowed only for an exception, missing/malformed output, or demonstrably wrong assertion before any numeric gate vector is emitted.

### Milestone 4: Execute exactly one fixed-seed metric run
- [x] Remove stale `run.log`, reconfirm GPU 0, and launch once: `timeout 600s env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 uv run train.py > run.log 2>&1`.
- [x] Monitor process, GPU, and log liveness. Do not prune on finite loss, train accuracy, intermediate test accuracy, or realized dose; abort only for exception, CUDA/OOM, nonfinite/integrity/audit failure, 120 seconds without process/GPU/log progress, or outer timeout.
- [x] Preserve raw run and all completed preflight evidence through durable transcription, Claude-only adversarial result review, commit, tree insertion, and analysis. Never retry the metric or change epsilon/SAM policy from the result.

### Milestone 5: Verify formal and scientific outcomes
- [x] Apply integrity-first precedence: scope/resource/state/RNG/evaluation corruption is `invalid/NaN`; no primary result is `crash/NaN`; only exit zero with intact evidence reaches improvement/no-improvement.
- [x] Require charged seconds `[299.5,301.0]`, total `<600s`, complete summary, exactly one evaluation per epoch, `num_params=2748890`, only tracked `train.py` changed, exact loss/counter/SAM/EMA audits, and no traceback/CUDA/OOM/audit/RuntimeError/NaN/Inf signature.
- [x] Formal improvement is `best_test_acc>=95.71%` versus parent `95.61%` (at least 9,571 correct of the fixed 10,000). A valid lower result is one no-improvement and cannot authorize coefficient/phase/SAM/seed changes.
- [x] Stable mechanism support additionally requires at least 25,300 steps, at least 155 EMA samples with ordinary/SAM imbalance at most one, final-16 EMA mean `>=95.69%`, and intact CutMix/SAM/EMA dose. Report final-16 range/final/progress, evaluation count, terminal debiased train loss, and best-minus-tail premium; scientific shortfall never overrides the formal tree verdict.

## Code Changes

- **`train.py`**: Add the fixed negative Poly-1 helper and constants; route optimizer-driving hard/CutMix/SAM-descent losses through it while retaining CE SAM ascent; add lightweight path counters/final reconciliation and outside-charged progress/train-loss diagnostics. No other tracked file changes.

The intervention changes only the optimizer-driving objective. Evaluation remains frozen CE accuracy/loss. The CE SAM first pass remains a parent component and must not be described as Poly-1 SAM ascent.

## Configuration Changes

- Training loss: parent hard/weighted CE -> fixed negative soft-target Poly-1 on optimizer-driving paths.
- `POLY1_EPSILON`: absent -> `-0.25` (maximum 25% confident-gradient attenuation; 2.5% at uniform ten-class probability and 22.5% at `p_y=0.9`).
- SAM ascent loss: unchanged hard CE; SAM descent loss: hard CE -> hard negative Poly-1.
- All model/data/optimizer/schedule/SAM-rho/EMA/evaluation constants remain exactly EXP-011.

## Execution Environment

- Method: local deterministic CPU checks, local physical-GPU-0 integration and one-shot paired preflight, then at most one local metric run.
- Resources: `CUDA_VISIBLE_DEVICES=0`; one NVIDIA H20 with approximately 97,871 MiB; existing `uv` environment; no dependency installation.
- Estimated runtime: 2-4 minutes for checks/preflight plus about 450 seconds for the metric process; metric outer timeout 600 seconds.
- Log output: `/tmp/exp015_*` for transient parent/harness/preflight evidence and repository-root `run.log` for the sole metric; set `PYTHONPYCACHEPREFIX=/tmp/exp015_pycache` for transient harnesses because shared `/tmp/__pycache__` may be non-writable.
- Tool skill: local execution; no remote platform or W&B.

## Abort Criteria

- Wrong GPU/visibility; tracked modification outside `train.py`; changed seed, budget, evaluator, evaluation cadence, model/data/optimizer/parent schedules; epsilon other than `-0.25`; Poly applied to SAM ascent or omitted from an optimizer-driving path.
- Hard/soft formula, FP32 probability, loss nonnegativity, parameter/state/RNG equality, CutMix area/target, drop-path, CE-ascent, gradient clearing, SAM perturb/replay/BN/restore/update, EMA cadence/coverage/swap/restore, optimizer identity, or counter reconciliation failure.
- First complete preflight parent drift `>0.03`, ratio dispersion `>0.005`, median ratio `>1.01`, any round `>1.02`, projection below 25,300 steps/130 epochs/155 EMA samples, total `>=600s`, candidate-only peak `>=1350 MiB`, or any numeric nonfinite/integrity failure.
- Metric exception, CUDA/OOM, audit/nonfinite/integrity signature, 120 seconds without process/GPU/log progress, or 600-second timeout. Finite conditioning/loss/accuracy/dose observations never select or prune.

## Verification Protocol

### Verification Procedure

1. Query parent with `bash /root/david/.codex/plugins/cache/deoxys/tree-autoresearch/0.1.2/skills/shared/scripts/tree.sh show .tree-autoresearch/goals/maximize-cifar10-best-test-accuracy/04-results.tsv 011`; require `metric=95.61`, commit `d68f73a`, reference 25,798 steps, 160 EMA samples, 133 evaluations, and final-16 mean 95.493125 from EXP-011.
2. Before every GPU command run `nvidia-smi -i 0 --query-gpu=index,uuid,name,memory.total --format=csv,noheader`; record physical index/UUID, then under `CUDA_VISIBLE_DEVICES=0` require `torch.cuda.device_count()==1`, visible name `NVIDIA H20`, and UUID equality with physical GPU 0 (visible `cuda:0`).
3. Run `uv run python -m py_compile train.py`, `git diff --check`, `git diff --name-only d68f73a`, and `git status --short --untracked-files=all`; require exactly `train.py` as the tracked code change. Track ignored `run.log` separately by explicit existence/stat checks.
4. Materialize/hash-check `d68f73a:train.py` under `/tmp` without modifying the repo. Run all Milestone-2 smokes under bounded timeouts, then the single Milestone-3 preflight under `timeout 420s env CUDA_VISIBLE_DEVICES=0 PYTHONPYCACHEPREFIX=/tmp/exp015_pycache`. Evaluators must raise and test loaders must show zero iterations. Any outer timeout is a pre-metric crash, not permission to reduce/rerun the workload.
5. Only after every gate passes, delete stale `run.log` and execute exactly once: `timeout 600s env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 uv run train.py > run.log 2>&1`; preserve its exit status and raw evidence.
6. Require `rg -c '  eval ep' run.log == num_epochs`, live+EMA source counts equal epochs, all four loss counters reconcile, and no unexplained match from `rg -n -i 'traceback|cuda error|out of memory|audit_failed|runtimeerror|(^|[^a-z])(nan|inf)([^a-z]|$)' run.log`.
7. Extract the complete summary/config, CutMix/SAM/EMA/loss audits, evaluation charged progress, terminal train loss, and final 16 EMA accuracies. Apply formal `>=95.71` and scientific `>=95.69` thresholds separately; charged time must be `[299.5,301.0]` and total `<600`.
8. Run Claude Opus as the sole raw-result adversarial reviewer with no fallback model. Correct evidence wording, write analysis, commit only `train.py`, insert the tree node, then remove `run.log` and all `/tmp/exp015_*` evidence before the next loop.

### Informational Metrics (Optional)

- Accuracy/loss: best/final test accuracy, final test loss, terminal debiased train loss, final-16 EMA mean/min/max/final/progress, and best-minus-tail premium.
- Runtime/dose: charged/total/startup seconds, peak MiB, epochs/evaluations, steps, CutMix eligible/applied, SAM eligible/applied/start, EMA updates/parity/source/restores.
- Loss mechanism: epsilon/sign, ordinary/CutMix/SAM-ascent/SAM-descent counts, theoretical hard multiplier range, and preflight hard/soft gradient diagnostics.
