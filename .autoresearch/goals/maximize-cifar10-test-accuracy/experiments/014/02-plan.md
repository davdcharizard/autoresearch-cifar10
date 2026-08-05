# Plan EXP-014: Safe Zero-Initialized Residual Endpoints
- **Created**: 2026-07-26

## Milestones

### Milestone 1: Exact initialization implementation
- [x] Create `autoresearch/maximize-cifar10-test-accuracy-014` from accepted `eb08811`.
- [x] Modify only source `train.py`: after unchanged model-wide initialization, zero exactly the six `PreActBlock.conv2.weight` tensors; expose a strict Boolean constructor switch only for matched preflight and log the production treatment.
- [x] Create ignored `experiments/014/preflight.py` with the exact fail-closed semantic, gradient, RNG, and timing protocol; require the path exists before invocation.
- [x] Compile/diff audit and pass topology, count, RNG, non-endpoint equality, shortcut, first/second-step gradient, and rejected-zero-BN assertions.

### Milestone 2: Matched feasibility gate
- [x] Run named fail-closed evaluator-free `experiments/014/preflight.py` on one H20.
- [x] Require all timing CV ratios <=0.05, throughput retention >=0.97, projected passes >=135, identical graph/count, and finite opened branches before scoring.

### Milestone 3: Single scored run and audit
- [x] Remove stale `run.log`; run once with `timeout 600s uv run train.py > run.log 2>&1`.
- [x] Require exit 0, full summary, one transition, accepted cadence, and 300.0-300.5 counted / <600 total seconds; record realized passes against 135 without making exposure a formal validity gate.
- [x] Accept only `best_test_acc >=94.17%`; no small-scale/selective/BN-zero/LR/architecture fallback.

## Code Changes
- **`train.py`**: add `ZERO_INIT_RESIDUAL=True`; extend `WideResNet` with strict Boolean `zero_init_residual`; call unchanged `self.apply(_weights_init)` first, then if enabled iterate exactly six `PreActBlock` modules and `init.zeros_(block.conv2.weight)`. Production instantiates with true and logs the exact endpoint count. No forward, BN, shortcut, topology, optimizer, data, schedule, mixup, evaluator, or RNG-consuming operation changes.
- **`experiments/014/preflight.py`**: ignored local research artifact for exact semantic/timing checks; never imported by production.

## Configuration Changes
- Six final residual convolutions: accepted Kaiming values -> exact zero only at initialization. All 691,674 parameters still trainable and all accepted hyperparameters unchanged.

## Execution Environment
- Local/offline, one H20, existing data/dependencies, no remote or GitHub. Preflight under one minute; one scored run about 340 seconds. Capture `run.log`, remove after analysis.

## Abort Criteria
- Do not score if counts/RNG/non-endpoint equality, residual/shortcut semantics, first/second-step gradients, CV, 97% retention, or 135-pass projection fail.
- Stop on timeout 124, traceback, OOM, non-finite loss/gradients, wrong endpoint count, missing H20, or >=600 wall seconds. Never stop for interim accuracy or rerun a valid score.

## Verification Protocol

### Verification Procedure
1. Run `exp-index.sh baseline .autoresearch/goals/maximize-cifar10-test-accuracy/04-results.tsv`; require `baseline=94.07` and `baseline_commit=eb08811`, hence threshold 94.17. Run `nvidia-smi --query-gpu=name --format=csv,noheader`; require exactly one line equal `NVIDIA H20`. Run `test -d data/cifar-10-batches-py`. Run `uv run python -m py_compile train.py`; require exit 0. Run `git diff --name-only eb08811`; require exactly `train.py`; require `git diff --quiet eb08811 -- prepare.py` and `git diff --check` both exit 0.
2. Run `uv run python .autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/preflight.py` after replacing `prepare.Eval` before importing `train.py`. Reject non-Boolean constructor values. Under identical saved RNG, build accepted false and candidate true; require identical post-construction CPU RNG, key sets/topology/count 691,674, and bitwise equality of every tensor except six `*.conv2.weight` endpoints.
3. Require exactly six `PreActBlock`s; candidate endpoints exact zero; accepted endpoints nonzero; every BN scale one/bias zero; all conv1/projection/fc tensors finite/nonzero. In train and eval mode, residual branch output must be zero initially; identity blocks return raw input and transition blocks return the accepted projected preactivation shortcut.
4. First finite CE backward: each required endpoint/classifier/projection gradient tensor must be finite with positive L2 norm; upstream residual conv1/bn2 gradients must have exactly zero norm. After one accepted SGD update, each of six endpoint weight tensors must have finite values and positive norm. Second backward must give each residual conv1 and bn2-scale gradient tensor finite values and positive L2 norm. An isolated `bn2.weight=0,bias=0` copy must demonstrate exactly zero BN-scale gradient norm and is forbidden in production.
5. Matched timing: exact accepted/candidate models and optimizers, fixed pinned inputs, per-path identical CPU/CUDA RNG snapshots restored/captured around windows, full production copies/LR/mixup-or-hard/backward/SGD/sync. Warm 25 mixup steps; measure three 50-step windows per path at 50% and 80% in order `accepted-A,candidate-A,candidate-B,accepted-B,accepted-C,candidate-C`. Use 65/35 median aggregates, CV=`pstdev/mean`, retention=`accepted/candidate`, projection=`141.9*retention`; require CV <=0.05, retention >=0.97, projection >=135, finite `[256,10]` logits/loss and no OOM.
6. Run `rm -f run.log` then sole scored `timeout 600s uv run train.py > run.log 2>&1`; require exit 0. Use `rg` on `run.log` to require exact zero-init/691,674/CUDA logs, one 65% transition near 195 seconds/LR 0.0612, no traceback/non-finite/OOM, unique fifth-plus-terminal evaluations, counted `[300.0,300.5]`, total <600, steps<64000, and complete summary.
7. Compute passes `steps*256/50000`; compare with 135 for mechanistic interpretation only. Formal verdict follows the goal: any valid run at >=94.17 is improvement even below 135 passes; any lower valid score is no-improvement. Only a stable negative at >=135 cleanly rejects the initialization mechanism.

### Informational Metrics (Optional)
- Summary metrics, passes, best epoch/gap, evaluations, preflight timing/CVs, early gradient norms, final loss versus accepted 0.2432.
