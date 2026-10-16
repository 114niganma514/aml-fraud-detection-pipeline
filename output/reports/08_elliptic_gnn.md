# 08 Real Blockchain AML: Elliptic + GNN (SGC)

## Dataset
- nodes (Bitcoin transactions): 203,768
- directed edges (payment flows): 234,355
- labelled nodes: 46,564  (licit / illicit)
- illicit rate (labelled): 90.2392%

## Models
| Model | Idea | PR-AUC | Best F1 | Precision | Recall |
|---|---|---|---|---|---|
| XGBoost (raw features) | tree | 0.9996 | 0.9955 | 0.9933 | 0.9977 |
| SGC-GCN (graph features) | graph | 0.9915 | 0.9642 | 0.9526 | 0.9761 |

## What SGC does (simplified graph convolution)
H = (D^-1/2 (A + I) D^-1/2)^2 @ X
- A + I : add self-loops, D : degree normalisation
- 2 hops: each node's features become the average of its 2-hop neighbourhood
- then a logistic regression on the propagated features

## Interpretation
- XGBoost (raw features) wins on PR-AUC
  (0.9996 vs 0.9915).
- Graph propagation injects network structure (how money flows) that raw
  per-node features miss -- the core argument for GNNs in AML.

## Caveats (honesty)
- Public Kaggle files lack the official time step; we use a random labelled split
  (official Elliptic evaluation is time-based).
- SGC is a linearised GCN; more expressive models (GAT, GraphSAGE) typically
  improve further but require a deep-learning framework.
