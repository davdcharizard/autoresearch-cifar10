# Report EXP-032: Multi-scale feature-aggregation classifier head

- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-032.md
- **Plan**: plans/plan-032.md
- **Log**: logs/exp-log-032.md

## Goal
Maximize CIFAR-10 `best_test_acc` (%, higher-is-better) within the fixed 300s training-compute budget on a single H20, editing only `train.py`. Current baseline = **96.22%** (EXP-012, commit 6c417a4); pass bar = baseline + 0.1 = **96.32%**.

## Idea & Hypothesis
Chosen idea: a **multi-scale feature-aggregation classifier head**. Every one of the prior 32 experiments pooled only the final `layer3` output (256ch @8×8) into the classifier. The feature-aggregation/head axis was the one structural lever never touched. The idea: feed the classifier BOTH mid-level (`layer2`, 128ch @16×16) and high-level (`layer3`, 256ch @8×8) global-avg-pooled features, concatenated → `fc(384→10)`. Hypothesis: multi-scale semantics give the linear head a richer, more discriminative representation that lifts top-1 above the 96.32 bar — a compute-neutral, integrity-clean inductive-bias change on the generalization side (no epoch-wall, no optimizer/averaging polish-vs-top1 trap), adding only +1280 params and a direct gradient path to layer2.

## Approach
Two localized edits to `ResNet` in `train.py`:
1. `__init__`: `self.fc = nn.Linear(w3, num_classes)` → `nn.Linear(w2 + w3, num_classes)` (Linear(384, 10), +1280 weights).
2. `forward`: keep both `out2 = layer2(out)` and `out3 = layer3(out2)`, global-avg-pool each to (B,128) and (B,256), `torch.cat` → (B,384), then `self.fc`.

No config changes — PEAK_LR 0.2, warmup 5%, batch 128, WD 1e-4, LS 0.1, Cutout 16, TrivialAugment, Nesterov, cosine-to-0, seed 42, `torch.compile(reduce-overhead)` all unchanged. Smoke test pre-run confirmed params 4,301,146, fc.weight (10,384), clean forward/backward. The change isolates the multi-scale variable cleanly (avg-pool aggregation per scale kept; only the head's INPUT widened).

## Execution
Single run, no retries, completed exit 0 in 403.3s wall (300.0s training). The dominant observation, recorded in the exp-log's Experimental Adjustments during the run: convergence was **markedly slower** than baseline/EXP-031 from the very start — eval ep1=19.3% (vs EXP-031's 55.4%), ep7=47.7% (vs ≈80%). The gap narrowed over training (ep16 ~14pp behind, ep30 ~6pp, ep53 ~90.7%, ep66 ~93.0%, ep79 ~94.5%) but never closed. Loss decreased normally with no NaN, so no abort criterion was met; the run was allowed to complete per plan for a clean final metric + throughput reading. No errors or dead ends.

## Results

- **Primary metric**: best_test_acc = **94.72%** (baseline: 96.22%, delta: **−1.50pp, −1.56%**)
- **Observations**: final_test_loss 0.2309 (worse than baseline 0.195 — the head hurts on loss too, not just top-1); num_epochs 87 (vs baseline ~91, a mild ~4-epoch throughput cost from the extra layer2 pool + wider fc); mean dt 8.12ms; num_params 4,301,146 (as intended); peak_vram 493.7MB; total 403.3s.
- **Analysis**: The hypothesis is **refuted**. Far from enriching the representation, routing layer2 features directly into the classifier head and adding a direct gradient path to layer2 **disrupts the tuned feature hierarchy** — the network must satisfy a linear classifier from mid-level 16×16 features before they have matured, which fights the standard coarse-to-fine ResNet learning dynamic and slows convergence sharply. With a fixed 300s/~87-epoch budget, the slower trajectory never recovers. The ~4 lost epochs are a minor compute confound; they cannot explain a 1.5pp drop (budget-adders near this plateau cost ≤~1 epoch-equivalent of accuracy per project-insights). The regression is a genuine inductive-bias effect, corroborated by the worse final loss (a top-1-only noise artifact would not move loss).
- **Key Learning**: Multi-scale aggregation (concatenating pooled layer2+layer3 into the head) is a net negative on this net — the direct mid-level→classifier path disrupts the tuned coarse-to-fine feature hierarchy and slows convergence, regressing −1.5pp on a fixed budget.

## Verification

- **Conditions**: Cond 1 (best_test_acc ≥ 96.32) **FAILED** — 94.72 < 96.32. Cond 2 (clean completion <600s, 0 Traceback) passed. Cond 3 (only train.py changed; num_params 4,301,146 as intended; eval-count 87 == epochs 87; core torch only; seed 42) passed — no constraint violated.
- **Review Notes**: Results confirmed trustworthy. dt ~8ms held, no NaN, eval count matches epochs, metric is consistent with the observed slow-convergence trajectory (no parsing/stale-output concern). The intended +1280 param change is the experiment's content, not a constraint violation, so the verdict is no-improvement, NOT invalid.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid, trustworthy result; primary verification condition (clear the bar) failed by a wide −1.6pp margin; no hard constraint violated.

## Unexplored Avenues
- **Auxiliary deep-supervision head instead of input-concatenation**: attach a separate small classifier to layer2 used ONLY as an auxiliary loss (weighted, decayed to 0), keeping the main classifier on layer3 alone. This adds the multi-scale gradient signal WITHOUT forcing the final linear head to depend on immature mid-level features — avoids the convergence disruption seen here. (Low-medium confidence; deep supervision usually needs careful weighting and the net is small.)
- **GeM or avg+max pooling on layer3 only** (deferred from this brainstorm): richer pooling of the FINAL features, leaving the hierarchy untouched. Cheaper and lower-risk than multi-scale, but a weak prior on an already-strong net.
- These are variations of the same "head/aggregation" idea; the input-concatenation form specifically is exhausted and harmful.

## Next Steps
- **Treat the feature-aggregation/head axis as closed for top-1** (confidence: high). Combined with the closed optimizer/gradient-dynamics axis (EXP-030/031), the k=4 / 300s configuration is firmly **generalization-bound at fixed capacity**, and structural head changes do not help.
- **If revisiting multi-scale at all, use auxiliary deep supervision (decayed aux loss), not input concatenation** (confidence: low) — only avenue that might extract the gradient benefit without the convergence penalty.
- **More promising direction**: pivot back to the generalization/data side within the budget — e.g. revisit augmentation strength/scheduling or a mild capacity reshape that keeps the coarse-to-fine hierarchy intact, rather than head surgery. (Confidence: medium that the plateau is real and only data/regularization or a genuinely different capacity allocation can move it.)

## Exit Action Results
<!-- No exit actions defined for this goal. -->
