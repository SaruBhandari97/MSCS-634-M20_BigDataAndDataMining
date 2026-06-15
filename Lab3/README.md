# MSCS-634 Lab 3 — Clustering Analysis Using K-Means and K-Medoids

**Course:** MSCS-634-M20 — Advanced Big Data and Data Mining (2026 Summer, Full Term)
**Author:** Saru Bhandari

## Purpose

This lab explores unsupervised clustering on the sklearn **Wine Dataset** (178 samples, 13 chemical
features, 3 cultivars). The goal is to apply two partitioning algorithms — **K-Means** and **K-Medoids
(PAM)** — with `k = 3`, then judge how well each one recovers the underlying structure of the data using
two complementary metrics:

- **Silhouette Score** — an internal measure of how tight and well-separated the clusters are.
- **Adjusted Rand Index (ARI)** — an external measure of how closely the clusters match the true cultivar labels.

## What's in the repo

| File | Description |
|------|-------------|
| `MSCS_634_Lab_3.ipynb` | Full notebook: data loading, standardisation, both clustering algorithms, metrics, and side-by-side PCA visualisation with cluster centres marked. |
| `clustering_comparison.png` | Exported comparison plot (also embedded in the notebook). |
| `README.md` | This file. |

## How to run

```bash
pip install scikit-learn numpy pandas matplotlib
jupyter notebook MSCS_634_Lab_3.ipynb
```

Run the cells top to bottom. No external clustering library is required — the K-Medoids algorithm is
implemented directly in the notebook.

## Key results and insights

| Algorithm | Silhouette Score | Adjusted Rand Index |
|-----------|:---------------:|:-------------------:|
| K-Means | **0.285** | **0.897** |
| K-Medoids (PAM) | 0.268 | 0.741 |

- **Standardisation mattered a lot.** The raw features are on wildly different scales (`proline` runs into
  the hundreds, `hue` sits near 1). Without z-score normalisation the distance calculations are dominated
  by a couple of large-magnitude features and the clusters collapse. After scaling, both algorithms cleanly
  separate the three cultivars.
- **The two methods are nearly tied on Silhouette** but K-Means agrees with the true classes noticeably
  more (ARI 0.90 vs 0.74). The Wine clusters are roughly spherical and outlier-free, which is the ideal
  case for K-Means since it minimises squared distance to mean-valued centroids.
- **The centres behave differently.** K-Means centroids are computed averages and can land in empty space
  between points; K-Medoids medoids are real wine samples (indices 35, 106, 148), so each cluster centre is
  an actual, interpretable observation.
- **When to prefer each:** K-Means is the fast default for clean, compact, roughly spherical clusters.
  K-Medoids is the better choice when the data has outliers, when you need the cluster centre to be a real
  data point, or when you want a non-Euclidean distance metric — at a higher computational cost.

## Challenges and decisions

- **`scikit-learn-extra` is broken on NumPy 2.x.** The usual `from sklearn_extra.cluster import KMedoids`
  fails to import in current environments (a Cython/NumPy 2 incompatibility). Rather than pin old library
  versions, I implemented the classic **PAM (Partitioning Around Medoids)** algorithm directly — a seeded
  build step plus the standard swap step that exchanges medoids with non-medoids whenever it lowers the
  total clustering cost. This keeps the notebook fully reproducible on any modern setup and makes the
  algorithm's mechanics explicit.
- **Visualisation in 2-D.** The data is 13-dimensional, so clustering is done on all standardised features
  but the scatter plots use a 2-component **PCA** projection purely for display.
- **Reproducibility.** A fixed `random_state = 42` is used for K-Means, PCA, and the K-Medoids
  initialisation so results are stable across runs.
