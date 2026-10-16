# 08 Real Blockchain AML: Elliptic + GNN (SGC)

## Dataset
- nodes (Bitcoin transactions): 203,768
- directed edges (payment flows): 234,355
- labelled nodes: 46,564  (licit / illicit)
- illicit rate (labelled): 9.7608%

## Models
| Model | Idea | PR-AUC | Best F1 | Precision | Recall |
|---|---|---|---|---|---|
| XGBoost (raw features) | tree | 0.9856 | 0.9566 | 0.9778 | 0.9362 |
| SGC-GCN (graph features) | graph | 0.6163 | 0.6485 | 0.6323 | 0.6655 |

## What SGC does (simplified graph convolution)
H = (D^-1/2 (A + I) D^-1/2)^2 @ X
- A + I : add self-loops, D : degree normalisation
- 2 hops: each node's features become the average of its 2-hop neighbourhood
- then a logistic regression on the propagated features

## Interpretation
- XGBoost (raw features) wins on PR-AUC
  (0.9856 vs 0.6163); baseline = illicit rate 0.0976.
- The 166 raw Elliptic features already encode rich local and 1-hop aggregate
  information, so a tree model on raw features is strongest. Linear 2-hop
  averaging (SGC) over-smooths the rare illicit class; it still clearly beats
  the prevalence baseline, but graph signal is best added as extra features
  for the tree model or via a non-linear GNN (GAT / GraphSAGE).

## Caveats
- Public Kaggle files lack the official time step; we use a random labelled split
  (official Elliptic evaluation is time-based).
- SGC is a linearised GCN; more expressive models (GAT, GraphSAGE) typically
  improve further but require a deep-learning framework.
