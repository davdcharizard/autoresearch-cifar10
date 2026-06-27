# Plan EXP-020: Projection shortcuts at stage transitions (ResNet option B, WRN-faithful)
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-020.md

## Milestones

### Milestone 1: Code change implemented and pre-validated
- [x] Replace the option-A pad shortcut in `BasicBlock` with a learned projection (1×1 conv stride-s, no bias + BN) used only when `stride != 1 or in_channels != out_channels`; `nn.Identity()` otherwise
- [x] Syntax check: `uv run python -c "import ast; ast.parse(open('train.py').read())"` exits 0
- [x] CPU pre-validation (no GPU): instantiate `ResNet(3, 10, 4)`, assert `sum(p.numel() for p in model.parameters()) == 4_327_754` (baseline 4,286,026 + 41,728), and assert a `(2,3,32,32)` forward returns shape `(2,10)`
- [x] Confirm the diff touches ONLY `BasicBlock.__init__`/`forward` — all module-level constants and `main()` byte-identical to baseline

### Milestone 2: Experiment launched and confirmed running
- [x] GPU 0 has zero compute apps at launch (the composite launcher's pre-check aborts otherwise)
- [x] Single composite background Bash command: pre-check → `rm -f run.log` → `uv run train.py > run.log 2>&1 &` → inline 15s watchdog (windowed dt from pct_done deltas, auto-kill on 4 consecutive >30ms windows) → `wait` → `TRAIN_EXIT rc=` → summary grep. NO separate-turn Monitor (first-polls after the run ends — EXP-017)
- [x] `run.log` shows the params line `ResNet-20 (4x wide) | params: 4,327,754` and step output within ~120s of launch

### Milestone 3: Run completed cleanly with on-family signatures
- [x] `TRAIN_EXIT rc=0` and summary block present in run.log
- [x] Contention sanity (pre-condition for analyzability): post-hoc windowed profile shows ≤2 of ~267 windows >30ms AND num_epochs in 133–141 (clean projection ~137–139; the two extra conv+BN kernels may cost ≤0.5ms/step). If violated → contaminated or mis-costed run: rerun once after confirming GPU 0 free; never analyze a contaminated run

### Milestone 4: Verification executed per protocol
- [x] First-failure-stop conditions checked in order (see Verification Protocol) and results recorded in exp-log-020.md

## Code Changes
- **train.py** (`BasicBlock.__init__`): replace the three shortcut bookkeeping lines
  ```python
  self.stride = stride
  self.need_pad = stride != 1 or in_channels != out_channels
  self.pad_channels = out_channels - in_channels if self.need_pad else 0
  ```
  with a learned projection shortcut (WRN/option-B; torchvision-standard conv+BN for post-activation ResNets):
  ```python
  if stride != 1 or in_channels != out_channels:
      self.shortcut = nn.Sequential(
          nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
          nn.BatchNorm2d(out_channels),
      )
  else:
      self.shortcut = nn.Identity()
  ```
- **train.py** (`BasicBlock.forward`): replace the slice-and-pad branch
  ```python
  shortcut = x
  if self.need_pad:
      shortcut = shortcut[:, :, :: self.stride, :: self.stride]
      shortcut = F.pad(shortcut, (0, 0, 0, 0, 0, self.pad_channels))
  out += shortcut
  ```
  with `out += self.shortcut(x)`.
- **Why this tests the hypothesis**: the only behavioral change is the transition shortcut path (blocks layer2[0] and layer3[0]); every other block keeps a true identity shortcut. The projection conv is picked up by the existing `_weights_init` Kaiming pass (it applies to all `nn.Conv2d`); its BN params land in the no-decay group and the 1×1 weights in the decay group via the existing `ndim` split — no optimizer-code changes needed.
- **Risks / edge cases**: (1) `nn.Identity` must be used (not `None`) so `forward` is branch-free for torch.compile; (2) projection BN initializes at γ=1 (default) — do NOT zero it (EXP-018 deferral lesson); (3) params print becomes 4,327,754 — the pre-validation pins this so attribution is exact; (4) two extra kernels at transitions could add ≤0.5ms/step — the Milestone 3 epoch window accounts for it.

## Configuration Changes
- None. All hyperparameters, schedule, augmentation, batch size, and compile setup byte-identical to baseline @ 1990397. Single-variable experiment: shortcut topology only.

## Execution Environment
- Method: local, on GPU 0 only (per goal hard constraint), via the standard composite launcher in ONE background Bash call from the project root (absolute paths; Bash cwd persists across calls — `cd` to project root inside the command):
  ```bash
  cd /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-fable-5
  # pre-check: abort if any compute app on GPU 0
  # rm -f run.log; uv run train.py > run.log 2>&1 &
  # inline watchdog loop: every 15s compute ms=(pct2-pct1)*3000/(step2-step1) from
  #   tr '\r' '\n' < run.log | grep -E "^step"; SLOW counter; CONTENTION_KILL at 4 consecutive >30ms
  # wait; echo TRAIN_EXIT rc=$?; grep "^best_test_acc:\|^final_test_acc:\|^final_test_loss:\|^training_seconds:\|^total_seconds:\|^startup_seconds:\|^peak_vram_mb:\|^num_epochs:\|^num_steps:\|^num_params:" run.log
  ```
- Resources: 1× H20 (GPU 0), ~1.7GB VRAM, ~8.5 min wall
- Estimated runtime: ~485–500s total (300s timed training + ~13s startup + ~120s eval overhead + teardown); hard cap 600s
- Log output: ALL output to `run.log` via redirection (no tee/stream, per goal procedure); run.log deleted after the experiment concludes
- Tool skill: none (local execution)

## Abort Criteria
- Watchdog: 4 consecutive 15s windows with windowed dt >30ms → auto-kill (`CONTENTION_KILL`) — contention, not a research result; rerun after confirming GPU 0 free
- No `step` lines in run.log within ~120s of launch → kill and inspect (startup hang/import error)
- Loss NaN/inf in step lines → kill; research failure, no retry
- Total wall clock exceeding 10 minutes → kill; treat as failure (goal hard constraint)
- run.log opens with CIFAR-10 download progress (data/ cache lost) → expect inflated startup; judge total_seconds accordingly (infra-errors EXP-015)

## Verification Protocol

### Verification Procedure
First-failure stop: evaluate in order; on the first FAILED condition, stop — remaining conditions are not evaluated. Baseline from `exp-index.sh baseline` = **96.71** → bar = **96.81** (baseline + 0.1, per goal necessary condition).

**Pre-condition (contention sanity, analyzability gate — not a verdict condition)**:
```bash
cd /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-fable-5
tr '\r' '\n' < run.log | grep -E "^step [0-9]+" | sed -E 's/^step 0*([0-9]+) ep [0-9]+ \(([0-9.]+)%\).*/\1 \2/' | awk 'NR>1{ms=($2-p2)*3000/($1-p1); if(ms>30) c++; n++; s+=ms} {p1=$1; p2=$2} END{printf "windows>30ms: %d of %d | mean win %.1f ms\n", c, n, s/n}'
grep "^num_epochs:" run.log
```
Pass: windows>30ms ≤ 2 AND num_epochs in 133–141. Fail → rerun (max 1 rerun), do not analyze. Timeout: 30s.

1. **best_test_acc ≥ 96.81**:
   `grep "^best_test_acc:" run.log` — numeric compare against 96.81. Empty grep = crashed run → read `tail -n 50 run.log`. Timeout: 30s.
2. **Run completed within budget without crashing**: `TRAIN_EXIT rc=0` in the launcher output AND `grep "^total_seconds:" run.log` ≤ 600. Timeout: 30s.
3. **Validation at most once per epoch**: `N_EVAL=$(tr '\r' '\n' < run.log | grep -c "eval ep")` equals `num_epochs` (one eval per epoch, none extra). Timeout: 30s.

Post-verification (either outcome): record results in exp-log-020.md; delete run.log during analyze-phase git housekeeping.

### Informational Metrics (Optional)
Collected only if all necessary conditions pass:
- peak_vram_mb: `grep "^peak_vram_mb:" run.log` (expect ~1615–1650MB; baseline 1613.0)
- num_epochs: `grep "^num_epochs:" run.log` (expect 137–139)
- num_params: `grep "^num_params:" run.log` (must read 4,327,754 — also an attribution check)
