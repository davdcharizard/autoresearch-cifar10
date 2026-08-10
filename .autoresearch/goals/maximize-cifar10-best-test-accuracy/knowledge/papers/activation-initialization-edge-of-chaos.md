# On the Selection of Initialization and Activation Function for Deep Neural Networks
- **Authors**: Soufiane Hayou, Arnaud Doucet, Judith Rousseau
- **Venue**: ICLR 2019
- **URL**: https://openreview.net/forum?id=H1lJws05K7

## Key Contributions

- Extends signal-propagation analysis beyond ReLU-like functions and quantifies trainability near the edge of chaos.
- Identifies a class of activations, including Swish, that propagates information more deeply than ReLU-like functions under compatible initialization.
- Connects initialization and activation rather than treating either choice as independently safe.

## Relevance

The accepted model still uses ReLU at every stem/block site and has not tested an activation-family change. SiLU is intrinsically smooth and adds no learned branch, so it offers a representation lever without changing parameter scale. Transfer is uncertain because the local network is shallow, BatchNorm-normalized, and trained for only 300 seconds. Full text was unavailable through OpenReview during retrieval; this distillation uses the official abstract/search record.

## Key Techniques

- Replace ReLU with fixed `SiLU`/Swish while preserving shapes and residual topology.
- Keep initialization isolated unless a compatible gain is independently justified.
- Measure full-step throughput because sigmoid-backed activations may reduce fixed-budget exposure.
