# Plan EXP-012: TrivialAugmentWide added to the train pipeline (kept with Cutout) + compile enabler
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-012.md

## Milestones

### Milestone 1: Code changes implemented and pass local checks
- [ ] Add `transforms.TrivialAugmentWide()` to `train_tf` (after RandomHorizontalFlip, before ToTensor — operates on the PIL image).
- [ ] Add `compiled_model = torch.compile(model, mode="reduce-overhead")` after the `num_params` print; route the training-loop forward through `compiled_model(inputs)`; keep eval on the eager `model` handle.
- [ ] `uv run ruff check train.py` clean; `git diff --stat` shows ONLY train.py changed.

### Milestone 2: Experiment running and early signal confirmed
- [ ] Launch `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` (background).
- [ ] Confirm clean startup: `params: 4,299,866` (UNCHANGED — TA/compile add no params), no traceback, no NaN, compile completes.
- [ ] Read steady-state `dt`/`img/s` from step ~50–100 to gauge whether throughput is starved (epoch-budget risk).

### Milestone 3: Run completes and metrics extracted
- [ ] Run exits 0, `total_seconds < 600`.
- [ ] Extract `best_test_acc`, `num_epochs`, `final_test_loss`, `num_steps`, `num_params` from run.log.

## Code Changes
- **train.py** (the ONLY editable file):
  1. **Train transform** — insert `transforms.TrivialAugmentWide()` into the `transforms.Compose([...])` for `train_tf`,
     positioned AFTER `transforms.RandomHorizontalFlip()` and BEFORE `transforms.ToTensor()`. TA operates on the PIL
     image (CIFAR10 yields PIL), applying one random op at a uniformly random strength per image. This tests the
     hypothesis: stronger, *diverse* (photometric+geometric) augmentation improves invariance/generalization beyond
     the occlusion-only Cutout. Cutout(16) is RETAINED (it runs later, GPU-side in the loop) — TA+Cutout is the
     canonical SOTA CIFAR-WRN pairing.
  2. **Compile enabler** — after `print(f"ResNet-... | params: ...")`, add
     `compiled_model = torch.compile(model, mode="reduce-overhead")`. In the training loop change `outputs = model(inputs)`
     to `outputs = compiled_model(inputs)`. Eval stays `evaluator.evaluate(model, device)` (eager) — avoids
     recompiles and matches the EXP-007/008/010/011 validated pattern. Buys ~15% throughput headroom to absorb TA's
     extra CPU augmentation cost so the run stays converged (≳75 epochs).
  - **Risks/edge cases**: (a) TA runs per-sample on CPU in the dataloader → if 8 workers can't keep the launch-bound
    GPU fed, throughput drops and epochs fall (underfit risk — the dominant failure mode here). TA is a single cheap
    PIL op with NO GPU `.item()` sync (unlike the EXP-002 Cutout bottleneck), so the hit should be modest; compile
    offsets it. (b) TA must precede ToTensor (needs PIL/uint8 input) — placement is explicit above.

## Configuration Changes
- Train augmentation: `RandomCrop(4) + HFlip + Cutout(16)` → `RandomCrop(4) + HFlip + TrivialAugmentWide + Cutout(16)`
  (TA is parameter-free — no magnitude/num_ops to tune; TrivialAugment's design point per arXiv:2103.10158).
- Training forward: eager `model` → `torch.compile(model, mode="reduce-overhead")` (execution-only; EXP-007 null
  standalone accuracy effect → keeps attribution of any gain to TA).
- No other hyperparameters change (k=4, batch 128, peak LR 0.2, cosine, Nesterov, WD 1e-4, LS 0.1, seed 42).

## Execution Environment
- Method: local — `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` (background via Bash `run_in_background`).
- Resources: single NVIDIA H20 (GPU 0); ~0.5 GB VRAM (trivial); 8 dataloader workers.
- Estimated runtime: ~300s training + ~10–20s startup + ~10–15s compile ≈ 330–360s wall-clock (well < 600s budget).
- Log output: all stdout/stderr → `run.log`. Metrics via
  `grep -aE "^best_test_acc:|^total_seconds:|^num_epochs:|^final_test_loss:|^num_steps:|^num_params:" run.log`.
- Tool skill: none (local run).

## Abort Criteria
- Loss NaN/inf or divergence (train loss climbing without recovery) → kill.
- Traceback / process exit ≠ 0 at startup (e.g., TA placement error, compile failure) → kill, fix, single retry.
- `num_params` ≠ 4,299,866 at startup → scope/architecture error → kill, fix (TA and compile must not change params).
- No log progress for > ~120s after startup → kill (silent hang / dataloader stall).
- NOTE: low realized epoch count (throughput-starved) is NOT an abort condition — it is a planned, informative
  outcome (it would mean TA underfits at this budget). Let the run complete and record epochs for analysis.

## Verification Protocol

### Verification Procedure
Baseline = **96.00** (`exp-index.sh baseline`); success bar = **96.10** (+0.1pp per goal). After the run completes:

1. **Cond 1 — clean completion within budget**: 
   `grep -aE "^best_test_acc:|^total_seconds:" run.log` returns a value AND `total_seconds < 600`, and
   `grep -ac "Traceback" run.log` == 0. Pass = both hold. (Timeout: 600s wall-clock for the run.)
2. **Cond 2 — primary metric clears bar**: parse `best_test_acc`; PASS iff `best_test_acc >= 96.10` (i.e.
   ≥ baseline 96.00 + 0.1). FAIL → verdict no-improvement. (This is the decisive condition.)
3. **Cond 3 — no constraint violations**: `git diff --name-only` lists ONLY `train.py`; `num_params == 4,299,866`
   (unchanged → no architecture/param change, confirms TA+compile are augmentation/execution-only); seed 42 intact;
   eval count (number of `eval ep` lines) == `num_epochs` (eval once/epoch — no eval-frequency hacking). Only
   evaluated if Cond 2 passes.

### Informational Metrics (Optional)
- `num_epochs`: `grep -a "^num_epochs:" run.log` — fairness check; ≳75 ⇒ fair converged test, ≪70 ⇒ throughput-starved.
- `final_test_loss`: `grep -a "^final_test_loss:" run.log` — generalization signal vs baseline 0.204 / compiled-k4 0.208.
- `img/s` & `dt`: from step ~50–100 progress lines — quantifies TA's throughput cost vs compiled-k4 (~14.8k img/s, 8ms).
