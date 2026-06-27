# Plan EXP-048: Numerics-identical charged-step de-overheading — collate-side channels_last + side-stream H2D prefetch
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-048.md

Baseline at planning time (exp-index.sh baseline): **96.71** @ 1990397 → bar = **96.81**. σ context (EXP-027): mean ≈ 96.57, σ ≈ 0.16. Conversion law (EXP-006): ≈ +0.019/epoch ⇒ −1ms/step ≈ +6 epochs ≈ +0.12.

Projection: zero param change (**4,286,026**); expected dt **21.4–22.2ms** (saving 0.3–1.0ms from removing the in-step layout-permutation kernel and taking the 6.3MB H2D copy off the critical path), ~139–146 epochs. Arithmetic byte-identical: same fp32 values, same layout, same kernel sequence, same RNG streams, same update math.

**Charging-semantics note (integrity)**: the timer code and TIME_BUDGET are untouched. `torch.cuda.synchronize()` still ends every charged window and fences ALL streams — overlapped prefetch copies complete inside charged windows; nothing is hidden from the timer. The layout permutation moves to the DataLoader worker processes, joining the already-uncharged CPU aug pipeline (ToTensor/Normalize/TA) — the same category of work, in the same place that work has always lived. This is throughput engineering of the EXP-000/006 class, not timer manipulation.

## Milestones

### Milestone 1: Code changes implemented and passing CPU sanity — ALL DONE
- [x] **Collate**: module-level `def collate_channels_last(batch):` → `x, y = torch.utils.data.default_collate(batch); return x.contiguous(memory_format=torch.channels_last), y`; pass `collate_fn=collate_channels_last` to the DataLoader (module-level def for worker picklability)
- [ ] **Prefetcher**: module-level `class CUDAPrefetcher:` — holds `iter(loader)` + a dedicated `torch.cuda.Stream()`; `_preload()` issues `.to(device, non_blocking=True)` for the next (inputs, targets) inside `with torch.cuda.stream(self.stream)`; `__next__` does `torch.cuda.current_stream().wait_stream(self.stream)`, takes the ready pair, calls `inputs.record_stream(torch.cuda.current_stream())` (and for targets), issues `_preload()` for the following batch, returns the pair; raises StopIteration when the loader is exhausted. CPU fallback (`device.type != "cuda"`): plain passthrough iterator (no streams) so sanity can run on CPU.
- [ ] **Loop wiring**: `for inputs, targets in CUDAPrefetcher(train_loader, device):` (fresh prefetcher each epoch); DELETE the two in-step `.to(device...)` lines and the `.to(memory_format=torch.channels_last)` conversion — tensors arrive on-device, channels_last. Timed region otherwise byte-identical (t0, lr math, zero_grad, autocast fwd, backward, step, synchronize, dt).
- [ ] CPU sanity (CUDA_VISIBLE_DEVICES="", `sys.path.insert(0, <project root>)`):
  - params == **4,286,026** (model untouched)
  - collate value-identity: on a fake batch, `collate_channels_last` output `torch.equal` to `default_collate` output AND `is_contiguous(memory_format=channels_last)` True, shape/dtype identical
  - CPU-fallback prefetcher sequence-identity: wrapping a 7-batch synthetic loader yields exactly the same tensors in the same order, no drops/duplicates (assert per-batch `torch.equal`), including epoch re-entry (second pass over a fresh wrapper)
  - 2-epoch mini train smoke on CPU (tiny synthetic dataset through a real DataLoader with the new collate + fallback prefetcher): finite decreasing loss
- [ ] Static check: `uv run python -c "import ast; ast.parse(open('train.py').read())"` plus import smoke

### Milestone 2: Gated launch on GPU 0 confirmed running
- [ ] Reuse `/tmp/exp046_composite.sh` AS-IS (gate 26ms / contention floor 26 remain valid — the change can only LOWER dt; verify file exists, else rebuild from /tmp/exp045_composite.sh via sed 31→26)
- [ ] Confirm GATE_DECISION; **record D0 as the overhead-itemization datum** (saving = 22.4 − D0)

### Milestone 3: Run resolved — full completion OR pre-registered branch
- [ ] **Numerics tripwire**: first eval (ep1) must land in the family band 36–41%; if it is wildly off (< 30%) with clean dt, treat as implementation defect (prefetcher feeding wrong/stale batches) → kill, fix per retry rules (code error, max 2), record in Errors & Dead Ends
- [ ] If GATE_KILL (D0 > 26ms): implementation defect (the change cannot legitimately slow the step) → fix-or-fail per retry rules
- [ ] If CONTENTION_KILL/STARTUP_KILL: relaunch byte-identically once gates re-clear (max 2, infra)
- [ ] If completed: rc=0 + summary block present

### Milestone 4: Verification executed (first-failure-stop) and exp-log updated
- [ ] Integrity pre-condition, then necessary conditions in order; results in exp-log-048.md

## Code Changes
- **train.py** (only file; 3 additions, 3 deletions): (1) module-level collate function — moves the per-step layout permutation into uncharged worker processes; (2) module-level CUDAPrefetcher (~25 lines) — overlaps the next batch's H2D copy with current compute on a side stream; (3) loop consumes the prefetcher and drops the in-step `.to` calls. Why this tests the hypothesis: it removes exactly the identified non-kernel overhead from the charged window while leaving every arithmetic operation, value, layout, and ordering identical — isolating the throughput→epochs conversion. Risks: prefetcher lifecycle bugs (stale batch, dropped batch, epoch-boundary leak) — covered by sequence-identity sanity + the ep1 tripwire; `record_stream` omission causing rare allocator reuse corruption — included per the standard pattern; pinned-memory + channels_last interaction — stock PyTorch.

## Configuration Changes
- None. All training constants identical (certified recipe). DataLoader gains only `collate_fn=`; num_workers/pin_memory/persistent_workers untouched.

## Execution Environment
- Method: local, composite background script `/tmp/exp046_composite.sh` reused verbatim (validated twice this session): dual launch gates (zero GPU-0 compute apps AND load < 60, poll 30s×240) → `rm -f run.log` → `uv run train.py > run.log 2>&1 &` → watchdog 44×15s (GATE_KILL D0 > 26ms; CONTENTION_KILL 4 consecutive > max(D0×1.25, 26); STARTUP_KILL tick 12; NaN; divergence eval < 15% after ep5; WALL_CAP tick 44)
- Resources: GPU 0 only (H20); VRAM ~1.7GB (one extra in-flight batch ≈ +13MB); CIFAR-10 cached in `data/`
- Estimated runtime: ~470–500s clean (startup ~20–25s, 300s charged, ~139–146 uncharged evals); failure branches ≤ ~180s
- Log output: `<project root>/run.log` + composite stdout (background task output file)
- Tool skill: none (local)

## Abort Criteria
- GATE_KILL D0 > 26ms → implementation defect (change can only speed up): fix-or-fail per code-error retry rules (max 2)
- ep1 eval < 30% with clean dt → prefetcher correctness defect: kill, fix per code-error retry rules (the divergence guard auto-kills < 15% after ep5 regardless)
- CONTENTION_KILL / STARTUP_KILL → infra: relaunch byte-identically when gates clear (max 2, then Outcome failed)
- NaN loss → research/implementation failure, no blind retry (inspect first)
- Wall ≥ ~660s → kill, failure (goal cap 600s)

## Verification Protocol

### Verification Procedure
First-failure-stop. **Integrity pre-condition**: pristine profile — ≥200-step windows mean ≤ 23.5ms and none > 27 (off-rung); num_epochs within **136–152** (≥ baseline-family 138 expected; below 136 implies contention/defect — cross-check pct deltas); printed params == 4,286,026; training_seconds == 300.0; eval lines ≤ num_epochs; ep1 eval within 36–41 (numerics-identity tripwire passed). Contention-caused integrity failure → rerun (infra), not a verdict.

1. **best_test_acc ≥ 96.81** (baseline 96.71 + 0.1pp): `grep "^best_test_acc:" run.log` (timeout 10s; empty ⇒ crash ⇒ `tail -n 50 run.log`). Pass ≥ 96.81. **Replicate branch**: read ∈ [96.70, 96.80] with pristine profile → pre-registered replicate pair (two byte-identical gated runs; improvement only if mean ≥ 96.81). Fail < 96.70 (or replicate mean < 96.81).
2. **Within budget**: composite rc == 0 AND `grep "^total_seconds:" run.log` ≤ 600 (timeout 10s)
3. **Eval cadence**: `grep -c "eval ep" run.log` ≤ num_epochs value (timeout 10s)

Cleanup per goal Procedure: delete `run.log` at loop end (analyze housekeeping).

### Informational Metrics (Optional)
- **D0 and windowed dt** (composite stdout): the overhead-itemization datum — saving = 22.4 − D0; maps the result to branch (ii) saving < 0.3ms vs (iii) ≥ 0.5ms
- num_epochs: `grep "^num_epochs:" run.log` (baseline family ~139; the epochs-delivered datum)
- peak_vram_mb: `grep "^peak_vram_mb:" run.log` (expect ~1.6–1.7GB)
- num_params: `grep "^num_params:" run.log` (expect 4,286,026)
- final_test_loss: `grep "^final_test_loss:" run.log` (family ~0.185 — confirms unchanged training distribution)
