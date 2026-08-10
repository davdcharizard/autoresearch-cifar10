# Swapout: Learning an ensemble of deep architectures
- **Authors**: Saurabh Singh, Derek Hoiem, David Forsyth
- **Venue**: NeurIPS 2016
- **URL**: https://papers.nips.cc/paper_files/paper/2016/hash/c51ce410c124a10e0db5e4b97fc2af39-Abstract.html

## Key Contributions
- Samples stochastic combinations of identity and residual paths, generalizing dropout and stochastic depth.
- Trains an implicit ensemble of related residual architectures with shared parameters.
- Reports improved CIFAR-10/100 accuracy over structurally identical deterministic ResNets.

## Relevance
Swapout supports residual-path stochasticity as a generalization lever, but the accepted run is already under pressure to fit strong augmented views. A conservative adaptation would perturb only same-width blocks during the high-LR phase and restore the exact accepted network for refinement.

## Key Techniques
- Independently mask identity and residual contributions during training.
- Use a deterministic expectation or sampling-based approximation for inference.
- Exploit shared parameters across sampled residual architectures.
