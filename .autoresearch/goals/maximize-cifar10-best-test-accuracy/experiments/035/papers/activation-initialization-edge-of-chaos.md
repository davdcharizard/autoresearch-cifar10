# On the Selection of Initialization and Activation Function for Deep Neural Networks
- **Authors**: Soufiane Hayou, Arnaud Doucet, Judith Rousseau
- **Venue**: ICLR 2019
- **URL**: https://openreview.net/forum?id=H1lJws05K7

## Key Contributions

- Extends signal-propagation analysis beyond ReLU-like functions and quantifies trainability near the edge of chaos.
- Identifies a class of activations, including Swish, that propagates information more deeply than ReLU-like functions under compatible initialization.
- Connects initialization and activation rather than treating either choice as independently safe.

## Relevance

The accepted model still uses ReLU at every stem/block site and has not tested an activation-family change. SiLU is intrinsically smooth and bounded in derivative behavior, so it offers a representation lever without recruiting a new branch or changing parameter scale. Transfer is uncertain because this local network is shallow, BatchNorm-normalized, trained for only 300 seconds, and the paper does not establish a gain for this exact CIFAR recipe. Full text was unavailable through OpenReview during retrieval; this distillation uses the official abstract/search record.

## Key Techniques

- Replace ReLU with `SiLU`/Swish while preserving shapes and residual topology.
- Pair activation changes with explicit initialization and early trajectory checks.
- Measure throughput because sigmoid-backed activations may reduce fixed-budget exposure.
