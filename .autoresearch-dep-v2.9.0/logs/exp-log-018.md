# EXP-018: Stochastic Depth (DropPath) on BasicBlock

## Execution

Overall Status & Info:
- **Created**: 2026-05-27
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-018.md
- **Plan**: plans/plan-018.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-018
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary

Implemented stochastic depth (DropPath) across all 9 BasicBlocks of the ResNet-20 model per plan-018. Four changes were made to train.py: (1) Added `DROP_PATH_RATE = 0.1` hyperparameter. (2) Added `drop_path_rate` parameter to `BasicBlock.__init__()` with storage as instance attribute. (3) Modified `BasicBlock.forward()` to apply per-sample Bernoulli masking on the residual branch during training, with 1/keep_prob scaling to preserve expected value. (4) Modified `ResNet.__init__()` to compute linearly spaced drop rates from 0 to 0.1 across all 9 blocks and distribute them via `_make_layer()` which was updated to accept and pass per-block rates to BasicBlock constructors.

### Surprises & Discoveries

No surprises — the implementation was straightforward. The existing BasicBlock forward method had a clean separation between the residual computation and shortcut path, making insertion of the drop path mask natural. The `_make_layer` method already used an enumerate loop, so adding per-block rates required minimal restructuring.

### Decisions

No deviations from the plan. All changes followed the plan exactly. AST-based syntax verification was used instead of `python -c "import train"` because the current shell environment lacks torchvision (the training environment has it).

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local background process
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-cifar10/exp-018.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-05-27
- **Ended**: 2026-05-27

Description:
- Running `uv run python train.py 2>&1 | tee exp-018.log` locally on H20 GPU. This is the stochastic depth experiment with DROP_PATH_RATE=0.1, linearly decayed from block 1 (rate ~0.011) to block 9 (rate 0.1). 92 epochs completed in 300s budget. Expecting best_test_acc ≥ 95.67% (baseline 95.57% + 0.1pp threshold).

Observations:
- Training ran without errors for 92 epochs, 17915 steps, 300.0s training time
- Throughput stable at 16-17ms/step (well under 18ms abort threshold)
- First LR drop (0.2→0.02) at ~50% (epoch 46): best jumped from ~84% to ~93%
- Second LR drop (0.02→0.002) at ~75% (epoch 73): best climbed from ~93% to 95.24%
- Final phase (75-100%) showed slow convergence plateau around 94.9-95.2%, peaking at 95.24% (epoch 92)
- Best test_acc 95.24% is 0.33pp below baseline 95.57% — stochastic depth hurt accuracy
- Loss values in final phase (~1.08-1.10) are higher than typical baseline training, suggesting DropPath's regularization was excessive for this shallow 9-block network

Key Metrics:
- best_test_acc: 95.24%
- final_test_acc: 95.24%
- final_test_loss: 0.3203
- training_seconds: 300.0
- total_seconds: 409.0
- startup_seconds: 1.2
- peak_vram_mb: 864.9
- num_epochs: 92
- num_steps: 17915
- num_params: 4,286,026

## Verification Results

### Conditions Checked

**Condition 1: best_test_acc > 95.67%**
- Command: `grep "best_test_acc:" exp-018.log | tail -1`
- Result: best_test_acc = 95.24%
- **FAIL** — 95.24% < 95.67% threshold (0.33pp below baseline 95.57%)

**Condition 2: Full summary block printed (10 fields)**
- Command: `grep -c "best_test_acc:\|final_test_acc:\|..." exp-018.log`
- Result: count = 10
- **PASS**

**Condition 3: Evaluation runs at most once per epoch**
- Command: `grep -c "eval ep" exp-018.log` vs `grep "num_epochs:" exp-018.log`
- Result: eval_count = 92, num_epochs = 92
- **PASS**

### Informational Metrics

- best_test_acc: 95.24%
- final_test_acc: 95.24%
- training_seconds: 300.0
- num_epochs: 92
- peak_vram_mb: 864.9
- num_params: 4,286,026

## Errors & Dead Ends

### 2026-05-27 — ModuleNotFoundError: torchvision not in base conda env
- Error: `ModuleNotFoundError: No module named 'torchvision'` when running `python train.py` with base conda python
- Root cause: Base conda env (`/SPXvePFS/share/miniconda3/bin/python`, Python 3.13.9) lacks torchvision. Prior experiments used `uv run python` which resolves the correct environment with torch+torchvision+CUDA.
- Source: exp-018.log L1-4 (initial failed run)
- Do NOT retry: Never use bare `python train.py` — always use `uv run python train.py`

## Human Notes

> {Researcher can add comments, corrections, or context here}
