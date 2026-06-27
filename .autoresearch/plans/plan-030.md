# Plan EXP-030: Concat avg+max global pooling head
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-030.md

## Milestones

### Milestone 1: Head change implemented and sanity-checked
- [ ] On branch `autoresearch/exp-030` (cut from `autoresearch/dev` @ 1990397), edit `train.py`:
  1. `ResNet.__init__`: `self.fc = nn.Linear(2 * w3, num_classes)` (input 256→512).
  2. `ResNet.forward`: replace
     `out = F.adaptive_avg_pool2d(out, 1)`
     with
     `out = torch.cat([F.adaptive_avg_pool2d(out, 1), F.adaptive_max_pool2d(out, 1)], dim=1)`
     (the following `out.view(out.size(0), -1)` and `self.fc(out)` lines are unchanged).
  3. NOTHING else changes — optimizer groups pick up the wider fc automatically (ndim>1 → decay group); Kaiming init applies via the existing `_weights_init`.
- [ ] Sanity: `uv run python -c "import ast; ast.parse(open('train.py').read())"`; expected param count = 4,286,026 + 2,560 = **4,288,586** (will be confirmed by the startup print); `git diff` shows exactly the two-site change with the timed step body untouched.

### Milestone 2: Run 1 launched with gates
- [ ] GPU-0 zero-compute-apps pre-check passes inside the composite launcher.
- [ ] Composite: pre-check → `rm -f run.log` → launch → 15s watchdog: contention kill (4 consecutive >30ms), early-dt gate (3 consecutive >27.0ms within first 7 ticks — the extra pool + 2× fc must cost ≲0.3ms; a dt regression means an unexpected inductor pattern → kill, code error), STARTUP_KILL tick 10, NaN/inf guard → `wait` → summary.

### Milestone 3: Trajectory readout and completion
- [ ] Early readout: ep1–10 vs baseline family (ep1 ~38, ep5 ~64, ep10 ~78) — hypothesis predicts family-tracking early (no deferral toll beyond ~2 epochs; a large ep1 deficit like EXP-018/020 would signal the new head is learned-from-zero costlier than expected).
- [ ] Completion: rc=0, total ≤600s (expect ≈ baseline 493s), eval_lines = num_epochs ≈ 139, params 4,288,586, post-hoc profile clean.

## Code Changes
- **train.py** (only file): 2-site head change (forward pooling concat + fc width). The trunk, optimizer, schedule, loaders, compile path, eval path, and the timed step body are untouched.
- Why this tests the hypothesis: the ONLY change is what the classifier sees — avg-pooled features alone vs avg ⊕ max — so any plateau shift is attributable to the head's information content.
- Risks/edge cases: (a) eager eval path (base_model) uses the same forward — consistent train/eval; (b) channels_last: pooling outputs (N,512,1,1) — `.view` after cat is safe because a (N,C,1,1) tensor's flatten is layout-invariant... use `.reshape` ONLY if a stride error appears (it should not for 1×1 spatial); (c) compile: new graph compiles in the existing 3-iter startup warmup; (d) VRAM delta negligible (+2,560 params + one pooled tensor).

## Configuration Changes
- None. All training constants byte-identical to baseline. fc input width 256→512 is an architecture change, not a hyperparameter change.

## Execution Environment
- Method: local composite background Bash (pre-check + launch + inline watchdog + wait + summary), branch `autoresearch/exp-030`, GPU 0 (`CUDA_VISIBLE_DEVICES=0`).
- Resources: VRAM ~1615MB; 8 loader workers.
- Estimated runtime: ~490–500s total (baseline-like). Under the 600s cap.
- Log output: `run.log` via `uv run train.py > run.log 2>&1`; watchdog WIN lines; post-hoc awk profile authoritative.
- Tool skill: none (local).

## Abort Criteria
- **Startup gate**: no step lines by tick 10 (150s) → kill.
- **Early-dt gate**: 3 consecutive windows >27.0ms within first 7 ticks → kill (head must be ≈free; regression = code error, not contention).
- **Contention kill**: 4 consecutive windows >30ms → kill, contaminated, rerun once (eval-side immunity caveat per EXP-029 precedent: if the result is decisively sub-bar AND epochs match expectation exactly, the no-rerun judgment may be applied and documented).
- **Divergence**: NaN/inf loss, or any eval <15% after epoch 5 → kill, code error.
- **Wall cap**: >600s → kill, failure.
- **Crash** (rc≠0, shape errors — e.g., fc width mismatch): code-error fix + resubmit per execute-skill rules (max 2).

## Verification Protocol

### Verification Procedure
First-failure-stop, in order. Baseline from `exp-index.sh baseline`: **96.71** @ 1990397; bar = 96.81. σ context (EXP-027): mean ≈96.57, σ ≈0.16; noise band ±0.15 around the mean.

1. **best_test_acc ≥ 96.81**:
   - Command: `grep "^best_test_acc:" run.log`. Empty ⇒ crash classification.
   - Pre-condition: post-hoc profile — `tr '\r' '\n' < run.log | grep -E "^step [0-9]+" | sed -E 's/^step 0*([0-9]+) ep [0-9]+ \(([0-9.]+)%\).*/\1 \2/' | awk 'NR>1{ms=($2-p2)*3000/($1-p1); if(ms>27) c++; n++; s+=ms} {p1=$1; p2=$2} END{printf "windows>27ms: %d of %d | mean win %.1f ms | expected epochs %.1f\n", c, n, s/n, 139*22.4/(s/n)}'` — require ≤2 slow windows AND epochs within ±3 of expected; contaminated ⇒ rerun once (subject to the documented eval-side-immunity judgment).
   - Integrity sub-check on a bar-pass: plateau-LEVEL shift (final-7 median ≥ 96.6, `tr '\r' '\n' < run.log | grep "eval ep" | tail -7`), not a single-eval spike; `grep "^num_params:" run.log` = 4,288,586; training_seconds = 300.0.
2. **Completes within budget**: rc=0 AND `grep "^total_seconds:" run.log` ≤ 600.
3. **Validation ≤ once/epoch**: `tr '\r' '\n' < run.log | grep -c "eval ep"` ≤ num_epochs.

On first failure: stop, classify, proceed to analyze.

### Informational Metrics (Optional)
- peak_vram_mb (expect ~1615), num_epochs (expect ≈139 — head must be throughput-free), startup_seconds (~13)
- dt delta vs 22.4ms — the measured cost of the extra pool + 2× fc (head-axis pricing, recorded regardless of verdict)
- ep1/5/10 vs family — deferral check on the re-initialized wider fc
