# Plan EXP-021: Step-time engineering — torch.compile(mode="max-autotune") + SGD(fused=True)
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-021.md

## Milestones

### Milestone 1: Code change implemented and pre-validated
- [x] `torch.compile(model)` → `torch.compile(model, mode="max-autotune")` in `main()`
- [x] `optim.SGD(...)` gains `fused=True` (keeping momentum/nesterov/selective-WD param groups byte-identical)
- [x] Syntax check: `uv run python -c "import ast; ast.parse(open('train.py').read())"` exits 0
- [x] Diff check: exactly two lines touched, no constants changed (`git diff` shows only the two arguments)

### Milestone 2: Experiment launched and confirmed running
- [x] GPU 0 has zero compute apps at launch (composite launcher pre-check aborts otherwise)
- [x] Single composite background Bash: pre-check → `rm -f run.log` → `uv run train.py > run.log 2>&1 &` → inline 15s watchdog → `wait` → `TRAIN_EXIT rc=` → summary grep. NO separate-turn Monitor
- [x] Startup gate: step lines appear in run.log within **150s** of launch (max-autotune compiles longer than default's ~23s; >150s startup projects total past ~595s) — if exceeded, kill and retry ONCE with `mode="reduce-overhead"` (fallback per brainstorm)

### Milestone 3: dt gate evaluated (the experiment's falsifier)
- [x] At step ≥100, compute windowed dt from pct_done deltas. Decision table:
  - dt ≤ 21.5ms → hypothesis live, let it run
  - 21.5 < dt ≤ 23.5ms → speedup insufficient for the bar but not pathological: let it run to completion anyway (cheap, gives a clean converged record); expected no-improvement
  - dt > 23.5ms → max-autotune pathology (slower than baseline): kill, retry ONCE with `mode="reduce-overhead"`, record in § Experimental Adjustments
- [x] Record the measured dt and the branch taken in exp-log-021.md

### Milestone 4: Run completed cleanly with consistent signatures
- [x] `TRAIN_EXIT rc=0`, summary block present, total_seconds ≤ 600
- [x] Contention sanity: ≤2 windows >30ms AND num_epochs consistent with measured dt (expected ≈ 139 × 22.4 / dt_measured, ±3 epochs). Violation → contaminated: rerun once after confirming GPU 0 free

### Milestone 5: Verification executed per protocol
- [x] First-failure-stop conditions checked in order and recorded in exp-log-021.md

## Code Changes
- **train.py** (`main()`, two arguments only):
  1. `model = torch.compile(model)` → `model = torch.compile(model, mode="max-autotune")` — enables Triton template autotuning for convs/matmuls plus CUDA graphs; attacks the ~18–19ms compiled compute and per-kernel launch overhead inside dt.
  2. `optimizer = optim.SGD([...], lr=0.0, momentum=MOMENTUM, nesterov=True)` → same call with `fused=True` — single fused CUDA update kernel replacing the multi-kernel foreach pass over 65 param tensors; `optimizer.step()` runs inside dt. PyTorch's fused SGD supports momentum + nesterov + per-group weight_decay; update rule mathematically identical.
- **Why this tests the hypothesis**: dt is the only quantity changed; hyperparameters, augmentation, schedule, and update math are byte-identical, so any accuracy delta is attributable to the extra epochs the smaller dt buys (EXP-006 mechanism).
- **Risks / edge cases**: (1) max-autotune autotuning runs during the existing 3-iteration pre-loop warmup → lands in startup, not dt; (2) CUDA-graphs constraints satisfied: static shapes every step (batch 512, drop_last=True), optimizer outside the compiled region, in-place weight updates at static addresses, eager `base_model` eval unaffected; inductor's cudagraph wrapper copies inputs into static buffers automatically; (3) fused SGD initializes lazily at the first real `step()` (warmup does no optimizer.step) — one-time ~ms cost in the first timed step, negligible; (4) VRAM grows with cudagraph static buffers — soft constraint, expect ~1.7–2.5GB; (5) per-step `g["lr"] = lr_now` assignment is read by the fused kernel each step — supported (lr is a kwarg-scalar, optimizer is eager).

## Configuration Changes
- None. All training constants (PEAK_LR 0.4, WARMUP_FRAC 0.15, MOMENTUM 0.9, WD 5e-4, LS 0.1, BATCH_SIZE 512), augmentation, and schedule byte-identical to baseline @ 1990397. The two arguments change execution speed, not optimization math.

## Execution Environment
- Method: local, GPU 0 only, standard composite launcher in ONE background Bash from the project root (cd inside the command; absolute paths):
  pre-check (abort if GPU-0 compute apps) → `rm -f run.log` → `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1 &` → inline 15s watchdog (windowed dt from pct_done deltas; SLOW counter; CONTENTION_KILL at 4 consecutive >30ms) → `wait` → `TRAIN_EXIT rc=` → summary grep
- Resources: 1× H20 (GPU 0), ~1.7–2.5GB VRAM, ≤10 min wall
- Estimated runtime: ~470–560s total (300s timed + 40–150s startup incl. max-autotune compile + ~110–125s eval overhead); hard cap 600s
- Log output: ALL output to `run.log` via redirection (no tee); deleted after the experiment concludes
- Tool skill: none (local)

## Abort Criteria
- No step lines within 150s of launch → kill; retry ONCE with `mode="reduce-overhead"` (compile-time fallback, recorded as Run 2)
- Windowed dt > 23.5ms at step ≥100 → kill; same single fallback retry
- Watchdog: 4 consecutive >30ms windows → auto-kill (contention); rerun after confirming GPU 0 free (does not consume the fallback retry)
- Loss NaN/inf → kill; research failure, no retry
- Total wall clock > 10 min → kill; failure per goal hard constraint
- run.log opens with dataset download → data/ cache lost; judge total_seconds with startup inflation in mind (infra-errors EXP-015)

## Verification Protocol

### Verification Procedure
First-failure stop; baseline from `exp-index.sh baseline` = **96.71** → bar = **96.81**.

**Pre-condition (contention sanity / signature consistency — analyzability gate)**:
```bash
cd /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-fable-5
tr '\r' '\n' < run.log | grep -E "^step [0-9]+" | sed -E 's/^step 0*([0-9]+) ep [0-9]+ \(([0-9.]+)%\).*/\1 \2/' | awk 'NR>1{ms=($2-p2)*3000/($1-p1); if(ms>30) c++; n++; s+=ms} {p1=$1; p2=$2} END{printf "windows>30ms: %d of %d | mean win %.1f ms\n", c, n, s/n}'
grep "^num_epochs:" run.log
```
Pass: windows>30ms ≤ 2 AND num_epochs within ±3 of (139 × 22.4 / mean_win_ms). Fail → contaminated; rerun once. Timeout: 30s.

1. **best_test_acc ≥ 96.81**: `grep "^best_test_acc:" run.log`; empty grep = crash → `tail -n 50 run.log`. Timeout: 30s.
2. **Run completed within budget without crashing**: `TRAIN_EXIT rc=0` in launcher output AND `grep "^total_seconds:" run.log` ≤ 600. Timeout: 30s.
3. **Validation at most once per epoch**: `tr '\r' '\n' < run.log | grep -c "eval ep"` equals num_epochs. Timeout: 30s.

Post-verification (either outcome): record in exp-log-021.md; run.log deleted during analyze housekeeping.

### Informational Metrics (Optional)
Collected only if all necessary conditions pass:
- peak_vram_mb: `grep "^peak_vram_mb:" run.log` (expect 1700–2500MB with cudagraph buffers; soft constraint)
- num_epochs: `grep "^num_epochs:" run.log` (hypothesis: ≥147)
- num_params: `grep "^num_params:" run.log` (must read 4,286,026 — unchanged-model attribution check)
- startup_seconds + windowed mean dt: from run.log + profile — the dt delta is the experiment's mediating variable; record regardless of verdict
