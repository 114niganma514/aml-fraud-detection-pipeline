# -*- coding: utf-8 -*-
"""
08_elliptic_gnn.py -- Real blockchain AML + GNN vs tree models.

Elliptic is a REAL Bitcoin transaction graph: 203k nodes (transactions),
234k directed edges (payment flows), with licit/illicit labels.

WHY:
  * validates the pipeline on real on-chain AML data (second real-data track)
  * introduces a graph neural network comparison: does using the network
    structure beat tree models on raw features alone?

Model implemented -- SGC (Simplified Graph Convolution, Wu et al. ICLR 2019):
  * GCN-family model. A 2-layer GCN without non-linearities between layers
    collapses to a single linear map H = (A_hat)^2 @ X, so SGC computes the
    2-hop neighbour-averaged features once, then fits a logistic regression.
  * This is a textbook-correct, dependency-free implementation of graph
    convolution using scipy sparse matrices (no PyTorch needed).

Comparison:
  * XGBoost on raw 166-dim features (no graph structure)
  * SGC on graph-propagated features (2-hop neighbourhood aggregation)

Evaluation: PR-AUC on the labelled subset (licit vs illicit).

NOTE on split: the public Kaggle files do not expose the official time step,
so we use a random 80/20 split of labelled nodes (stated in the
report; official Elliptic evaluation uses time-based splits).

Outputs:
  - output/reports/08_elliptic_gnn.md
  - output/figures/08_elliptic_pr_curve.png, 08_elliptic_propagation.png
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, precision_recall_curve
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=pd.errors.Pandas4Warning)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (ensure_dirs, ELLIPTIC_DIR, FIGURES_DIR, REPORTS_DIR,
                   best_f1_point, write_markdown)


def load_elliptic():
    """Load classes / edgelist / features; return cleaned frame + adj matrix."""
    print(">>> Loading Elliptic data ...")
    cls = pd.read_csv(os.path.join(ELLIPTIC_DIR, "elliptic_txs_classes.csv"))
    edges = pd.read_csv(os.path.join(ELLIPTIC_DIR, "elliptic_txs_edgelist.csv"))
    feat = pd.read_csv(os.path.join(ELLIPTIC_DIR, "elliptic_txs_features.csv"))
    print(f"    nodes={len(feat):,}  edges={len(edges):,}  features={feat.shape[1] - 1}")

    # Official Elliptic encoding: 1 = illicit (4,545), 2 = licit (42,019), 'unknown'
    cls["class"] = cls["class"].astype(str)
    label_map = {"1": 1, "2": 0}  # 1 illicit (positive class), 0 licit
    feat.columns = ["txId"] + [f"f{i}" for i in range(1, feat.shape[1])]
    df = feat.merge(cls, on="txId", how="left")
    df["y"] = df["class"].map(label_map)
    print(f"    labelled nodes: {df['y'].notna().sum():,}  "
          f"illicit rate: {df['y'].fillna(0).mean():.4%}")

    # node index map
    node_list = df["txId"].astype(str).tolist()
    idx = {t: i for i, t in enumerate(node_list)}

    # adjacency (directed edges made undirected -- standard for GCN propagation)
    src = edges["txId1"].astype(str).map(idx).values
    dst = edges["txId2"].astype(str).map(idx).values
    mask = (src != -1) & (dst != -1) & ~np.isnan(src) & ~np.isnan(dst)
    src, dst = src[mask].astype(int), dst[mask].astype(int)
    n = len(df)
    adj = sp.coo_matrix((np.ones(len(src)), (src, dst)), shape=(n, n))
    adj = adj + adj.T.tocsr()          # undirected
    adj = adj + sp.eye(n)              # self-loops
    deg = np.asarray(adj.sum(axis=1)).ravel()
    dinv = sp.diags(1.0 / np.sqrt(deg + 1e-9))
    a_hat = (dinv @ adj @ dinv).tocsr()
    print(f"    adjacency nnz: {a_hat.nnz:,}")
    return df, a_hat, len(edges)


def main():
    ensure_dirs()
    df, a_hat, n_edges = load_elliptic()

    X_all = df[[f"f{i}" for i in range(1, 167)]].values.astype(np.float64)
    X_all = StandardScaler().fit_transform(X_all)
    y_all = df["y"].values

    # labelled subset only for supervised learning
    labelled = ~np.isnan(y_all)
    y = y_all[labelled].astype(int)
    X = X_all[labelled]
    print(f"    labelled rows: {len(y):,}")

    rng = np.random.RandomState(42)
    perm = rng.permutation(len(y))
    n_tr = int(len(y) * 0.8)
    tr_idx, te_idx = perm[:n_tr], perm[n_tr:]

    # ------------------------------------------------------------------
    # SGC: 2-hop neighbourhood propagation  H = A_hat^2 @ X
    # ------------------------------------------------------------------
    print(">>> [SGC] propagating features over the graph (2 hops) ...")
    Xp = a_hat @ X_all          # 1st hop
    Xp = a_hat @ Xp             # 2nd hop
    Xp_labelled = Xp[labelled]

    lr = LogisticRegression(max_iter=300, C=1.0, solver="lbfgs")
    lr.fit(Xp_labelled[tr_idx], y[tr_idx])
    sgc_prob = lr.predict_proba(Xp_labelled[te_idx])[:, 1]

    # ------------------------------------------------------------------
    # XGBoost on raw features (no graph structure)
    # ------------------------------------------------------------------
    print(">>> [XGBoost] training on raw features (no graph) ...")
    model = xgb.XGBClassifier(objective="binary:logistic", learning_rate=0.1,
                              max_depth=6, n_estimators=200,
                              eval_metric="aucpr", tree_method="hist",
                              random_state=42, n_jobs=-1, verbosity=0)
    model.fit(X[tr_idx], y[tr_idx])
    xgb_prob = model.predict_proba(X[te_idx])[:, 1]

    # ------------------------------------------------------------------
    # evaluation
    # ------------------------------------------------------------------
    y_te = y[te_idx]
    print("\n=== Elliptic test results (PR-AUC) ===")
    results = {}
    for name, prob in [("XGBoost (raw features)", xgb_prob),
                       ("SGC-GCN (graph features)", sgc_prob)]:
        ap = average_precision_score(y_te, prob)
        p, r, f1, thr = best_f1_point(y_te, prob)
        results[name] = (ap, p, r, f1, thr)
        print(f"    {name:<28} PR-AUC={ap:.4f}  BestF1 P={p:.4f} R={r:.4f}")

    # PR curve
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"XGBoost (raw features)": "#9BBBF4",
              "SGC-GCN (graph features)": "#52C41A"}
    for name, prob in [("XGBoost (raw features)", xgb_prob),
                       ("SGC-GCN (graph features)", sgc_prob)]:
        p, r, _ = precision_recall_curve(y_te, prob)
        ax.plot(r, p, label=f"{name} (PR-AUC={results[name][0]:.4f})",
                color=colors[name], lw=2)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Elliptic (REAL blockchain): tree vs graph model")
    ax.legend(loc="upper right"); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "08_elliptic_pr_curve.png"), dpi=120)
    plt.close(fig)

    # propagation visual: histogram of propagated feature (2-hop mean) for one col
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(Xp_labelled[tr_idx][:, 0], bins=60, color="#C9A7E8", alpha=0.7, label="2-hop propagated f1")
    ax.set_title("SGC 2-hop feature distribution (illustrates neighbourhood aggregation)")
    ax.set_xlabel("propagated value"); ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "08_elliptic_propagation.png"), dpi=120)
    plt.close(fig)

    # data-driven interpretation (do not assume the graph model wins)
    best_name = max(results, key=lambda k: results[k][0])
    if "SGC" in best_name:
        interp_line = ("Graph propagation injects network structure (how money flows) that raw\n"
                       "  per-node features miss -- the core argument for GNNs in AML.")
    else:
        interp_line = ("The 166 raw Elliptic features already encode rich local and 1-hop aggregate\n"
                       "  information, so a tree model on raw features is strongest. Linear 2-hop\n"
                       "  averaging (SGC) over-smooths the rare illicit class; it still clearly beats\n"
                       "  the prevalence baseline, but graph signal is best added as extra features\n"
                       "  for the tree model or via a non-linear GNN (GAT / GraphSAGE).")

    # report
    rep = f"""# 08 Real Blockchain AML: Elliptic + GNN (SGC)

## Dataset
- nodes (Bitcoin transactions): {len(df):,}
- directed edges (payment flows): {n_edges:,}
- labelled nodes: {int(labelled.sum()):,}  (licit / illicit)
- illicit rate (labelled): {y.mean():.4%}

## Models
| Model | Idea | PR-AUC | Best F1 | Precision | Recall |
|---|---|---|---|---|---|
"""
    for name, (ap, p, r, f1, _) in results.items():
        rep += f"| {name} | {'graph' if 'SGC' in name else 'tree'} | {ap:.4f} | {f1:.4f} | {p:.4f} | {r:.4f} |\n"
    rep += f"""
## What SGC does (simplified graph convolution)
H = (D^-1/2 (A + I) D^-1/2)^2 @ X
- A + I : add self-loops, D : degree normalisation
- 2 hops: each node's features become the average of its 2-hop neighbourhood
- then a logistic regression on the propagated features

## Interpretation
- {max(results, key=lambda k: results[k][0])} wins on PR-AUC
  ({max(results.values())[0]:.4f} vs {min(results.values())[0]:.4f}); baseline = illicit rate {y.mean():.4f}.
- {interp_line}

## Caveats
- Public Kaggle files lack the official time step; we use a random labelled split
  (official Elliptic evaluation is time-based).
- SGC is a linearised GCN; more expressive models (GAT, GraphSAGE) typically
  improve further but require a deep-learning framework.
"""
    write_markdown(os.path.join(REPORTS_DIR, "08_elliptic_gnn.md"), rep)
    print(">>> Elliptic GNN validation done.")


if __name__ == "__main__":
    main()
