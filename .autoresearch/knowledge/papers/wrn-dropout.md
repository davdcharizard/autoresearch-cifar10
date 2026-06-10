# Wide Residual Networks — dropout in residual blocks (Zagoruyko & Komodakis 2016)

**Paper**: "Wide Residual Networks", BMVC 2016, arXiv:1605.07146
**Topic**: Architecture / regularization — the WideResNet our k=4 model descends from
**Why relevant**: this project's model is a WRN-style widened ResNet ({16k,32k,64k} stages). The paper's central
*regularization* recommendation is dropout INSIDE the residual block — a regularization locus no experiment on this
goal has touched (closed axes are input-aug, weight-decay, label, weight-averaging). First applied in EXP-022.

## Core idea
Widening ResNets (more channels per layer, fewer layers) outperforms making them very deep, at similar param counts,
and trains faster on GPUs (more parallelizable). But the extra width adds capacity/redundancy that can overfit, so
the paper adds **dropout** to regularize it.

## The dropout recipe (this is the key detail)
- Place a single dropout layer **between the two 3×3 convolutions** of each residual block — i.e. after the first
  conv's activation (ReLU), before the second conv. NOT in the classifier, NOT after the block.
- In the paper's pre-activation block the order is `BN→ReLU→conv1→(dropout)→BN→ReLU→conv2`. For our POST-activation
  block (`relu(bn1(conv1(x))) → bn2(conv2(out)) → +shortcut → relu`), the equivalent placement is right after the
  first `relu(bn1(conv1(x)))`, before `conv2`.
- Use plain elementwise `nn.Dropout` (the paper uses standard dropout, not channel-wise Dropout2d/SpatialDropout).
- Dropout probability p ≈ 0.3 for CIFAR in the paper — but tuned for their LONG (200-epoch) schedules. At a short
  budget, a milder p is safer (strong dropout slows convergence → under-fit risk).
- Apply in TRAIN mode only (eval must call `model.eval()` so dropout is identity — the frozen `Eval.evaluate()`
  harness already calls `model.eval()`, so eval is unaffected automatically).

## Empirical claims
Dropout in WRN consistently lowered CIFAR error in the paper (e.g. WRN-28-10 CIFAR-10 4.00→3.89%; larger gains on
CIFAR-100 and on SVHN), and stabilized training against the disturbance of the LR drops. Gains are modest on CIFAR-10.

## Caveats for this project
- Params unchanged (dropout has none) → fair same-budget test; compute near-neutral (one elementwise mask per block).
- Short budget (~92 epochs vs paper's 200) + an already heavily-regularized recipe (TA+Cutout+LS+WD) → real risk that
  dropout UNDER-fits and regresses (cf. EXP-018 CutMix). Probe a MILD p first (0.1), sweep up only if it gains.
- ResNet-20 is shallow (9 blocks); dropout's documented benefit is largest on deep/very-wide nets, so the effect here
  may be small. Judge on best_test_acc; watch final_test_loss for under-fit (loss↑ = under-regularized/under-trained).
