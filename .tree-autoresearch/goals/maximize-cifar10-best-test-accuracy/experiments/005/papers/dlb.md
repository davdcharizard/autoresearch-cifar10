# Self-Distillation from the Last Mini-Batch (DLB)

- Source: Yiqing Shen et al., CVPR 2022, https://openaccess.thecvf.com/content/CVPR2022/html/Shen_Self-Distillation_From_the_Last_Mini-Batch_for_Consistency_Regularization_CVPR_2022_paper.html
- Relevance: a one-forward consistency regularizer with direct CIFAR-10 and CutMix compatibility evidence.

## Mechanism

The data order is rearranged so the first half of each current batch repeats the second half of the previous batch, while the current second half becomes the repeated half at the next iteration. The model caches detached logits for that outgoing half. At the next step, it adds `alpha * tau^2 * KL(softmax(cached/tau) || softmax(current_first_half/tau))` to the ordinary supervised loss. The published settings use `tau=3` and `alpha=1`.

Each image is evaluated once per appearance and the method needs only one ordinary model forward/backward per optimizer step. The historical target is maximally fresh: it comes from the parameters immediately before the last update.

## Evidence

On CIFAR-10, DLB reduces error by 0.37 to 1.01 points across VGG, ResNet, WRN-20-8, and DenseNet backbones. The paper explicitly combines DLB with CutMix. On CIFAR-10, CutMix+DLB improves over CutMix alone by 0.09 to 1.48 points depending on backbone; WRN-20-8 improves from 4.89 to 4.29 error. Results are three-run averages. The original protocol trains with duplicated half-batches while halving iterations/epochs for a fair sample-exposure comparison.

## Experiment implications

- The current loader emits independent batches, so implement an in-loop stream: retain the prior batch's outgoing raw or augmented half and construct the next batch from that half plus half of a fresh loader batch.
- Preserve the charged 300-second timer and step-based schedule. Do not increase examples processed per forward.
- CutMix complicates identity correspondence. The repeated half must receive a fresh ordinary crop/flip if testing augmentation consistency, but a batch-level CutMix can destroy one-to-one teacher alignment. A conservative design applies DLB only on clean (non-CutMix) steps or excludes any repeated examples modified by mixing from KL.
- SAM second passes must not update the cached teacher; cache logits from the primary forward only after the optimizer step semantics are clearly defined.
- The method changes sample ordering, so counters should report DLB-active examples and cache resets at epoch boundaries.

## Verdict

Strongest evidence-backed finalist. The central risk is semantic interaction with the parent's CutMix and period-two SAM, not compute cost.
