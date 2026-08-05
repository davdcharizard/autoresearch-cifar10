**Prioritized Concerns**

1. **Plan Milestone 1 / identity-init claim, `train.py` model init**
   Appending `GatedResidual(256)` inside `layer2` before `self.apply()` changes RNG consumption before `layer3`/`fc` initialization. So `DEPTH=1` is not just “base net plus identity block”: later shared layers and the DataLoader/augmentation RNG state can differ from `DEPTH=0`. This breaks the identity-init rationale and confounds the same-seed control.
   **Fix:** initialize shared/base modules in identical order, then initialize the extra block under saved/restored RNG or `torch.random.fork_rng`; verify shared params and first training batch/augmentation stream match between c0 and cA. The smoke’s bit-equality check must compare a depth model with copied/shared base weights, not two independently initialized models.

2. **Plan Milestone 2 / Abort Criteria: single-conv fallback**
   The fallback from full `GatedResidual(256)` to a single `conv_bn(256,256)` is a different intervention: no skip path, no ReZero alpha, no identity init, different optimization behavior. Letting that fallback become the verdict/bake cell would make a win ineligible for the chosen idea.
   **Fix:** if the full ReZero block fails the sizing gate, abort EXP-021 as infeasible or write a new amended plan for the single-conv probe with its own hypothesis, smoke checks, expected params, and verdict label.

3. **Plan Milestone 3 command / GPU hard constraint**
   The per-cell command shown in line 23 omits `CUDA_VISIBLE_DEVICES=1`, despite the goal requiring GPU 1 only. If the script does not set it globally, this can run on GPU 0 or expose multiple GPUs.
   **Fix:** put `CUDA_VISIBLE_DEVICES=1` in every cell command and log the visible device plus physical GPU-1 `nvidia-smi` snapshot before each cell.

4. **Under-anneal gate: `num_epochs >= 135` is too weak for negative conclusions**
   `train.py` increments `epoch` before the batch loop, so the final partial epoch counts as a full epoch. Also, a null at 135 counted epochs is not enough to “close” depth when the compiled control is expected near 173 epochs and the current best recipe runs around 150.
   **Fix:** gate on `num_steps >= 135 * len(train_loader)` for minimal validity, and predefine that nulls below a stronger completed-step/epoch threshold are inconclusive rather than evidence that depth is closed. Include tail-shape checks such as `best==final`/still-rising behavior.

5. **Compile recompile guard is weaker than EXP-014**
   The plan references first-step/first-10-step recompile detection, but the listed code changes do not add first-step or per-epoch first-batch logging. The smoke also omits EXP-014’s eval-boundary guard. A recompile after an epoch evaluation could silently eat timed budget and reduce steps.
   **Fix:** restore the EXP-014 smoke: eval raw/EMA, switch back to train, then time a compiled step. Add logging for the first batch after every eval/epoch and abort on any compile-scale spike.

6. **Same-session control order is fixed and not counterbalanced**
   Both initial and confirmation pairs run c0 then cA. If GPU-1 load drifts monotonically, treatment is always second; the confirmation repeats the same bias direction.
   **Fix:** reverse the confirmation order or use AB/BA. Require median steady img/s for each cell to match pre-sized clean bands and discard on any foreign compute activity or unexplained throughput asymmetry, not only the broad `<22000 img/s` tripwire.

7. **Integrity check can conflict with planned artifacts**
   Verification requires `git status --porcelain` to show only ` M train.py`, while the plan also writes root `run_<cell>.log` files and records decisions in `03-execute.md`. If those are tracked or not deleted before the check, the integrity gate is ambiguous.
   **Fix:** keep transient logs in `/tmp` or delete them before NC3, and separate method-integrity checks from autoresearch bookkeeping files.

**Overall Judgment**

As written, the plan is not execution-sound enough to run as a verdict-bearing experiment. The main blocker is the RNG/init confound: without fixing it, cA is not isolated depth. The single-conv fallback also needs to be removed or split into a separate plan before any win can be considered valid.
