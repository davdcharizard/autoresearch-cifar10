# Reconciling Modern Deep Learning with Traditional Optimization Analyses: The Intrinsic Learning Rate
- **Authors**: Zhiyuan Li, Kaifeng Lyu, Sanjeev Arora
- **Venue**: NeurIPS 2020
- **URL**: https://papers.nips.cc/paper/2020/hash/a7453a5f026fb6831d68bdc9cb0edcae-Abstract.html

## Key Contributions
- Frames BatchNorm SGD dynamics through an intrinsic learning rate controlled by learning rate times weight decay.
- Provides theory and experiments on how normalized-network trajectories equilibrate under schedule changes.
- Supplemental CIFAR experiments show averaging checkpoints from one local SGD trajectory can improve test accuracy even as parameter distance continues growing.

## Relevance
The accepted model uses BatchNorm, momentum SGD, fixed all-parameter decay, a long `lr=0.1` phase, and a low-LR cosine tail. The evidence supports preserving the high-LR exploration and averaging only within the coherent late trajectory, rather than perturbing the validated decay or restarting optimization.

## Key Techniques
- Preserve the high-LR exploration needed to reach a useful normalized-network regime.
- Average within one initialization/trajectory, not across unrelated solutions.
- Treat LR and weight decay jointly when reasoning about BatchNorm optimization.
