# Report EXP-012: CutMix-complementary GPU Cutout
- **Created**: 2026-08-06

## Goal

Maximize CIFAR-10 `best_test_acc` under the frozen 300-second charged-training protocol through a genuine `train.py` training change. EXP-012 grew from parent/global-best EXP-011 at 95.61%; formal improvement required at least 95.71% with physical GPU 0, one validation per epoch, a complete bounded run, and intact parent semantics.

## Idea & Hypothesis

Apply reference-geometry size-16 Cutout to every early batch on which EXP-011's 0.5 CutMix gate is not selected. This additive schedule preserves every validated CutMix batch, hard labels, and the entire clean SAM/EMA tail while using a private GPU RNG stream. Direct WideResNet/CIFAR evidence made spatial erasure plausible, and the hypothesis was that complementary occlusion would raise the stable EMA tail by about 0.25 points without materially reducing optimizer exposure.

## Approach

`train.py` adds a setup-owned `ComplementaryCutout` with a deterministic 1,024-entry FP32 channels-last mask bank for uniformly sampled integer centers and clipped `[center-8,center+8)` geometry. A private seed-43 CUDA generator and preallocated center, index, selected-mask, area, histogram, and scalar buffers drive one in-place multiply only in the lexical `else` of the early CutMix decision. GPU-resident accounting defers host synchronization until the terminal audit. New diagnostics enforce exact complement equality, mean area, all-center support, bank geometry, and dose, while evaluation output adds charged progress for tail analysis. Model, data, optimizer, CutMix, SAM, EMA, budget, seed-42 streams, and evaluation cadence remain unchanged.

## Execution

The initial CPU harness needed the repository root added to `sys.path`. GPU smoke then exposed that allocation-stable `torch.sum(..., out=...)` requires an explicit dimension in this PyTorch build; the implementation was corrected to `dim=(0,)`. A later allocation check and helper tensor constructor each needed harness-only fixes. The corrected exhaustive CPU test, full-WRN GPU smoke, and 1,000-call helper benchmark passed. Claude performed pre-run adversarial reviews and cleared the final diff.

The single decisive five-round preflight passed without rerun: parent drift 0.016205, paired-ratio MAD/median 0.001261, median weighted latency ratio 1.007941, projected exposure 25,594.8 steps, and projected total 451.5 seconds. The sole metric run then exited 0 on physical GPU 0 after 300.0 charged seconds. Claude independently reviewed the raw result, initially blocking an over-strong causal-falsification phrase; after the record distinguished the observed tail bin from full-dose inference, Claude returned `PASS`.

The failed implementation remains inspectable at commit `3b5b48d` on `tree-autoresearch/maximize-cifar10-best-test-accuracy-exp-012`. After exact durable transcription, Claude review, report completion, and tree insertion, transient run and harness files were removed as required.

## Results

- **Primary metric**: 95.52% (parent: 95.61%, delta vs parent: -0.09 points, -0.09%; global best: 95.61%)
- **Observations**: The run completed 25,376 steps, 131 epochs/evaluations, 450.1 total seconds, and used 1,228.4 MiB peak VRAM. CutMix applied 10,151/20,461 times; Cutout filled the exact 10,310-batch complement over 2,639,360 images with mean masked area 196.058612 and all 1,024 centers. SAM applied 2,458/4,915 times from step 20,462. EMA made 158 updates split 79/79, routed 105 live plus 26 EMA evaluations, performed 26 exact restores, and recorded zero restoration, coverage, nonfinite, or RNG failures.
- **Analysis**: Accuracy missed the formal threshold by 0.19 points and regressed 0.09 versus the parent. The final 16 EMA values ranged 95.33-95.52 and averaged 95.418125 over progress 0.860187-1.000000, 0.075 below EXP-011's 95.493125 reference plateau; all 16 occupy the preregistered `<95.59` falsified observation bin. However, realized exposure was 25,376 steps, 124 below the conjunctive 25,500 mechanism-dose floor and 218.8 below the preflight projection. The result therefore rejects this exact package as a tree improvement but cannot cleanly establish that full-dose complementary Cutout is causally ineffective. The likely tradeoff is a combination of excess occlusion overlap with CutMix and unpredicted production timing loss; the evidence does not separate those factors.
- **Key Learning**: Full-probability complementary Cutout reached 95.52 with a lower 95.42 tail, but step under-dose prevents a clean full-dose causal conclusion.

## Verification

- **Conditions**: Primary accuracy failed: 95.52% is below the required 95.71%. Formal verification stopped at that first necessary-condition failure. Prior integrity classification confirmed exit 0, complete timing/summary/evaluation evidence, and clean mechanism audits; mechanism dose separately missed because 25,376 <25,500.
- **Review Notes**: Claude independently recomputed epoch and source counts, CutMix/Cutout complement, image and area arithmetic, SAM boundary, EMA cadence/parity/restoration, final-16 values, all deltas, and the projection miss. Its follow-up `PASS` requires wording the tail as an observed falsified bin without claiming full-dose causal falsification.
- **Verdict**: no-improvement
- **Verdict Basis**: All hard constraints and process-integrity requirements held, but the primary verification condition failed and the planned mechanism dose was not fully delivered.

## Unexplored Avenues

- **Lower conditional Cutout probability**: applying size-16 Cutout to only half of early non-CutMix batches would reduce both occlusion dose and helper cost, but it introduces another stochastic gate and lacks evidence that the accuracy loss was dose rather than timing.
- **Smaller masks**: size 8-12 could provide weaker spatial erasure while preserving complement structure; the direct literature setting was size 16, so this would need a new preregistered operating point.
- **Group-shared masks**: one mask per compact image group could reduce the 71.1-us helper cost and recover exposure, but correlated erasure may weaken per-example diversity.

## Next Steps

- **Return to EXP-011 and test calibrated soft-target Poly-1 (medium confidence)**: it adds negligible geometry/transport cost and targets the decision loss rather than overlapping CutMix's spatial erasure.
- **Test a preregistered EMA horizon or EMA/live interpolation on EXP-011 (medium confidence)**: this directly targets the 95.49 tail plateau with sparse state work and no extra forward.
- **Explore a fused classifier or representation calibration change (low-medium confidence)**: memory headroom remains large, but paired latency must protect the 25,500-step dose before accuracy execution.
