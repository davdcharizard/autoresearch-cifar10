# Plan EXP-019: Whitening init for conv1 (patch-eigenvector filters ± negations, learnable)
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-019.md

## Milestones

### Milestone 1: Code change implemented, syntax-checked, math pre-validated
- [x] Edit `train.py` `main()`: insert the whitening-init block immediately after `model = ResNet(...).to(device, memory_format=torch.channels_last)` and before `base_model = model` (exact code in § Code Changes)
- [x] Syntax check: `uv run python -c "import ast; ast.parse(open('train.py').read())"` (never import train.py — module level instantiates Eval())
- [x] **Numerical pre-validation (CPU-only, before launch)**: run a standalone snippet replicating the whitening math on the cached dataset and assert the whitened-output covariance ≈ I: max |diag − 1| < 0.05 and max |off-diag| < 0.05. If it fails, fix the math BEFORE any GPU run — this de-risks the main implementation-bug failure mode
- [x] Diff review: ~15 added lines in one block, NOTHING else changed (constants, architecture, schedule, aug, compile path byte-identical to baseline @ 1990397)

### Milestone 2: Run launched cleanly
- [x] Pre-launch: GPU 0 has zero compute apps (composite script aborts otherwise)
- [x] Launch via the standard composite background script (pre-check + `uv run train.py > run.log 2>&1` + inline contention watchdog, auto-kill on 4 consecutive >30ms windows). No separate-turn Monitor (infra-errors: it first-polls after the run ends)
- [x] `run.log` params line reads exactly `params: 4,286,026` (init-only change); any deviation → kill and fix

### Milestone 3: Early signals (post-hoc)
- [x] Throughput signature unchanged: windowed dt ≈ 22.4ms (identical compiled graph — any sustained shift is contention by elimination); startup may run ~2–5s longer than the 13.2s warm-cache baseline (one-time covariance + eigh, inside the wall margin)
- [x] Record the early eval trail (eps 1–10) — hypothesis says FASTER than EXP-017's 63.76@5 / 75.06@8 (and far above EXP-018's 35.26@5); informational, not a gate

### Milestone 4: Completion and verification
- [x] TRAIN_EXIT rc=0, summary block present in run.log
- [x] Contention sanity (pre-condition): num_epochs ≈ 139 ± 10% AND post-hoc windowed profile ~0 windows >30ms
- [x] Verification protocol executed in order (below), first-failure stop

## Code Changes
- **train.py** (only file; one ~15-line block in `main()`, inserted right after model creation, before `base_model = model`):
  ```python
  # Whitening init for the stem conv: filters = patch-covariance eigenvectors
  # scaled by 1/sqrt(eigenvalue), plus their negations so both signs survive
  # ReLU (airbench, arXiv 2404.00498). Data-aligned decorrelating stem at
  # init; bn1 absorbs the output scale. Learnable (not frozen).
  with torch.no_grad():
      imgs = torch.tensor(train_set.data[:5000], dtype=torch.float32)
      imgs = imgs.permute(0, 3, 1, 2) / 255.0
      imgs -= torch.tensor(mean).view(1, 3, 1, 1)
      patches = (
          imgs.unfold(2, 3, 3).unfold(3, 3, 3)
          .permute(0, 2, 3, 1, 4, 5).reshape(-1, 27)
      )
      cov = torch.cov(patches.T)
      eigvals, eigvecs = torch.linalg.eigh(cov)
      filt = (eigvecs / (eigvals + 1e-4).sqrt()).T.reshape(27, 3, 3, 3)
      w = model.conv1.weight
      w[:27].copy_(filt.to(w.device))
      w[27:54].copy_(-filt.to(w.device))
  ```
  Why this tests the hypothesis: the stem starts as the decorrelating, variance-equalized, data-aligned feature extractor that training otherwise spends early heat discovering — the only intervention class (information-adding init) exempted from the EXP-018 deferral closure. Mechanics: `train_set.data` is the raw uint8 numpy array (N,32,32,3) — available untransformed regardless of the augmentation pipeline; stride-3 unfold gives 100 disjoint patches/image × 5000 images = 500k patches (deterministic — no RNG consumed, so the training stream's seed state is untouched: explicitly NOT seed hacking); patch flattening order (C,kh,kw) matches conv weight layout; `eigh` ascending eigenvalues with ε=1e-4 bounds the largest filter gain; filters 54–63 keep their Kaiming init (the 27-dim patch space is fully covered by the ± pairs). `copy_` handles the channels_last layout. Risks: none to throughput (identical graph; init values don't affect tracing); numerical risk handled by Milestone 1's pre-validation; weights stay in the decay group (ndim>1) same as baseline — no optimizer-surface change.

## Configuration Changes
- None. Every hyperparameter at its certified-optimal value; the intervention is init-time only (params 4,286,026, FLOPs, dt, VRAM unchanged) — perfect attribution, same property as EXP-018.

## Execution Environment
- Method: local, single run via the standard composite background script (GPU-0 zero-compute-apps pre-check → `rm -f run.log` → `uv run train.py > run.log 2>&1 &` → inline watchdog, 15s windows, auto-kill on 4 consecutive >30ms → `wait` + TRAIN_EXIT + summary grep)
- Resources: GPU 0 only (wait if busy — never GPU 1); ~1613MB VRAM (identical to baseline; the 500k×27 covariance is CPU-side, ~54MB host RAM)
- Estimated runtime: ~485–520s total (300s timed + ~139 evals + startup ~15–18s including the one-time eigh)
- Log output: `run.log` in project root (full redirect, no tee/stream); deleted after the experiment concludes
- Tool skill: none (local execution)

## Abort Criteria
- **Contention** (infra): watchdog auto-kill on 4 consecutive >30ms windows; post-kill `nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader`. Quarantine contaminated runs, relaunch into a clean window, max 2 infra retries. Identical graph ⇒ any sustained slowdown is contention by elimination.
- **Crash in the whitening block** (shape/dtype error): traceback in run.log before any step line → fix and relaunch (code-error retry; Milestone 1's pre-validation makes this unlikely).
- **Degenerate start**: if eval ep 1 ≤ 15% (near-random — would suggest a broken stem, e.g., filter scale blow-up), let it run to ep 10; if still < 40%, kill as a research failure (do not retry) — mirrors the EXP-018 plan's criterion.
- **No output**: no step lines within 120s of launch (startup includes the new ~2–5s eigh) → inspect run.log tail.
- **Wall cap**: > 600s total = failure per hard constraints.

## Verification Protocol

### Verification Procedure

Pre-condition (contention sanity, before any condition): num_epochs within ~10% of ~139 AND post-hoc windowed profile ≈ 0 windows >30ms:
`tr '\r' '\n' < run.log | grep -E "^step [0-9]+" | sed -E 's/^step 0*([0-9]+) ep [0-9]+ \(([0-9.]+)%\).*/\1 \2/' | awk 'NR>1{ms=($2-p2)*3000/($1-p1); if(ms>30) c++; n++; s+=ms} {p1=$1; p2=$2} END{printf "windows>30ms: %d of %d | mean win %.1f ms\n", c, n, s/n}'`
If contaminated: quarantine and relaunch (infra path) — never verify a contaminated run.

Conditions in goal-file order, FIRST-FAILURE STOP:
1. **best_test_acc ≥ 96.81** (baseline 96.71 + 0.1, via `exp-index.sh baseline`): `grep "^best_test_acc:" run.log` — pass iff ≥ 96.81. Empty grep = crash (read `tail -n 50 run.log`). Timeout: 1 min.
2. **Run completes within budget**: `grep "^total_seconds:" run.log` ≤ 600 and TRAIN_EXIT rc=0. Timeout: 1 min.
3. **Validation at most once per epoch**: `tr '\r' '\n' < run.log | grep -c "eval ep"` equals `num_epochs`. Timeout: 1 min.

Cleanup after verdict: delete `run.log`; on no-improvement discard via `git checkout -- . && git clean -fd -e .autoresearch/ -e data/` (never bare `-fd`).

### Informational Metrics (Optional)
- peak_vram_mb / num_epochs / num_params: summary greps — must be ≈1613 / ≈139 / 4,286,026 (attribution checks)
- Early trail: `tr '\r' '\n' < run.log | grep "eval ep" | head -10` — vs EXP-017's 63.76@5, 75.06@8 (onset sign) and EXP-018's 35.26@5 (the inverted comparator)
- Plateau shape: final ~10 evals — flatness and level vs baseline's 96.4–96.7 band
