# MSCS_634_Lab_2 – KNN and Radius Neighbors Classification on the Wine Dataset

**Course:** Advanced Big Data and Data Mining (MSCS-634-M20)  
**Student:** Saru Bhandari  

---

## Purpose

This lab explores two distance-based classification algorithms — **K-Nearest Neighbors (KNN)** and **Radius Neighbors (RNN)** — using the Wine Dataset from `sklearn`. The Wine Dataset contains 178 samples across three wine classes, described by 13 chemical features (e.g., alcohol content, flavanoids, proline). The goal is to understand how parameter choices (number of neighbors `k` for KNN; search radius for RNN) affect classification accuracy, and to determine which model performs better for this dataset.

---

## Repository Contents

| File | Description |
|------|-------------|
| `Lab2_KNN_RNN_Wine.ipynb` | Jupyter Notebook with full implementation, visualizations, and analysis |
| `README.md` | This file |

---

## Key Results and Observations

### KNN Accuracy by k Value

| k | Accuracy |
|---|----------|
| 1  | 94.44% |
| 5  | 94.44% |
| 11 | 94.44% |
| **15** | **97.22%** ← best |
| 21 | 94.44% |

KNN performed consistently well across all tested values of k. The highest accuracy of **97.22%** was achieved at **k = 15**. The results indicate that the three wine classes are well-separated in feature space, allowing KNN to classify them with high confidence. The slight improvement at k = 15 reflects the benefit of averaging over a larger local neighborhood to suppress noise without losing class-level specificity.

### RNN Accuracy by Radius

| Radius | Accuracy |
|--------|----------|
| **350** | **75.00%** ← best |
| 400    | 72.22% |
| 450    | 72.22% |
| 500    | 72.22% |
| 550    | 72.22% |
| 600    | 72.22% |

RNN accuracy was notably lower than KNN. At a radius of **350**, the classifier returned **75.00%**, and performance declined as the radius increased. Larger radii include more distant training points, many of which belong to different classes, diluting the local signal. Additionally, for some test samples, no training point fell within the specified radius, triggering the `outlier_label='most_frequent'` fallback (assigning the most frequent class), which contributed to misclassifications.

### Model Comparison

| Model | Best Parameter | Best Accuracy |
|-------|---------------|---------------|
| KNN   | k = 15        | **97.22%**    |
| RNN   | radius = 350  | 75.00%        |

**KNN clearly outperforms RNN** on this dataset by a margin of over 22 percentage points.

---

## When to Use KNN vs. RNN

- **KNN** is the better choice when data density is roughly uniform across classes and when you want guaranteed predictions for every sample. It is easier to tune via cross-validation and is more robust in practice.
- **RNN** is more appropriate when data density varies meaningfully — for instance, in spatial or sensor datasets where a "local neighborhood" has a natural physical meaning. However, it requires careful radius tuning and must handle the case of samples with no neighbors in range.

For the Wine Dataset, KNN is the preferred classifier due to its higher accuracy, consistent predictions, and simpler hyperparameter tuning.

---

## Challenges and Decisions

- **Radius selection for RNN:** The lab-prescribed radius values (350–600) are designed for the raw (unscaled) feature space. Applying `StandardScaler` before RNN collapses all features to a similar scale, making these radii meaninglessly large and causing all samples to fall into the same over-inclusive neighborhood. The decision was made to use unscaled features for RNN to keep the prescribed radius values meaningful.
- **Outlier label handling:** `RadiusNeighborsClassifier` requires an `outlier_label` parameter for test samples that fall outside the search radius. `'most_frequent'` was used so that such samples are still classified (using the majority class) rather than raising an error.
- **Feature scale and KNN:** KNN is sensitive to feature scale since the Wine Dataset includes features with very different ranges (e.g., *proline* spans ~278–1680 vs. *nonflavanoid_phenols* which spans ~0.13–0.66). Both scaled and unscaled versions were explored; the final notebook uses the raw features, consistent with the lab's prescribed RNN radii.

---

## How to Run

```bash
# Clone the repository
git clone https://github.com/SaruBhandari97/MSCS-634-M20_BigDataAndDataMining/tree/master/Lab2


# Install dependencies
pip install scikit-learn matplotlib pandas numpy jupyter

# Launch the notebook
jupyter notebook Lab2_KNN_RNN_Wine.ipynb
```

All dependencies are part of the standard scientific Python stack and require no special installation beyond the above.
