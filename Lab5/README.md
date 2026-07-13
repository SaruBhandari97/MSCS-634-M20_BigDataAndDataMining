# MSCS_634_Lab_5 — Hierarchical and DBSCAN Clustering

## Purpose

This lab explores two unsupervised clustering approaches — **Agglomerative Hierarchical Clustering** and **DBSCAN** — applied to the Wine dataset from `sklearn.datasets`. The goals are to:

- Practice preparing data for clustering (exploration + standardization).
- Apply and tune both algorithms across different parameter settings.
- Visualize clustering results and evaluate them using silhouette, homogeneity, and completeness scores.
- Compare the two algorithms' behavior, strengths, and weaknesses on the same dataset.

## Contents

- `MSCS_634_Lab_5.ipynb` — Jupyter notebook containing all code, visualizations, and analysis, organized into four steps: data preparation, hierarchical clustering, DBSCAN clustering, and a final comparison/analysis section.

## Key Insights

- After standardizing the 13 chemical-analysis features, **Agglomerative Clustering with `n_clusters=3`** most closely recovered the three known wine cultivars, matching both the PCA scatter plots and the dendrogram's clearest structural break.
- The **dendrogram** confirmed this: the largest jumps in merge distance occur around the 3-cluster cut, reinforcing that 3 is a natural number of groups for this data.
- **DBSCAN's** results were highly sensitive to `eps` and `min_samples`. Small `eps` values produced excessive noise and fragmented clusters; large `eps` values collapsed everything into a single cluster. A mid-range setting (`eps=2.5`, `min_samples=5`) gave the best balance of cluster count, noise level, and evaluation metrics.
- Overall, Hierarchical Clustering slightly outperformed DBSCAN on this particular dataset in terms of homogeneity/completeness against the true labels, since the three cultivars form roughly compact, similarly-dense groups. DBSCAN's main advantage was explicitly flagging ambiguous, low-density boundary points as noise instead of forcing every point into a cluster.

## Challenges and Decisions

- Because DBSCAN operates on distance and density rather than a fixed cluster count, a small grid search over `eps` and `min_samples` was used to find a workable configuration before selecting representative values for visualization.
- Silhouette Score is undefined when DBSCAN produces only one cluster (or all noise), so those configurations were excluded from the silhouette comparison while still reporting homogeneity/completeness.
- PCA (2 components) was used purely for visualization; all clustering itself was performed on the full standardized 13-feature space to avoid losing information before fitting the models.
