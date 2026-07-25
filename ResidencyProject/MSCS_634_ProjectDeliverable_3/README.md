# MSCS-634 Project Deliverable 3: Classification, Clustering, and Pattern Mining

**Author:** Saru Bhandari
**Course:** MSCS-634 – Advanced Big Data and Data Mining
**Deliverable:** 3 of 4 — Classification, Clustering, and Pattern Mining

## Repository Contents

- `MSCS_634_Deliverable3.ipynb` — full notebook covering classification, clustering, and association rule mining
- `retail_transactions_cleaned.csv` — the cleaned dataset carried over from Deliverables 1-2
- `README.md` — this file

---

## Dataset Summary

This deliverable builds on the same **cleaned retail transactions dataset**
(1,000 rows) used in Deliverables 1-2, applied to three distinct data mining
tasks: predicting a **binary outcome** (classification), discovering
**natural customer groupings** (clustering), and discovering **co-occurrence
patterns** among purchases (association rule mining).

## 1. Classification

**Target:** `HighValue` — whether a transaction's `TotalAmount` is above the
dataset median ($153.90), a binary target with direct business relevance
(e.g., flagging transactions for a loyalty tier). As in Deliverable 2,
`Quantity`/`UnitPrice` were excluded from the features to avoid leaking the
target.

**Models built (3, exceeding the "at least two" requirement):** Decision
Tree, k-Nearest Neighbors, and Naive Bayes.

**Hyperparameter tuning:** `GridSearchCV` (5-fold CV, scored on F1) was
applied to the Decision Tree over `max_depth`, `min_samples_split`, and
`min_samples_leaf`. The search selected `max_depth=10`,
`min_samples_leaf=4`, `min_samples_split=10` — a much more regularized tree
than the unconstrained default.

**Evaluation (confusion matrix, ROC curve, accuracy/F1):**

| Model | Accuracy | F1 Score | ROC AUC |
|---|---|---|---|
| Decision Tree (default) | 0.490 | 0.505 | 0.490 |
| Decision Tree (tuned) | 0.575 | 0.569 | 0.639 |
| k-NN | 0.595 | 0.609 | 0.634 |
| **Naive Bayes** | **0.595** | **0.640** | **0.678** |

**Best model: Naive Bayes**, with the tuned Decision Tree close behind.
Tuning had a large effect on the Decision Tree — accuracy rose from 0.490
(barely above chance) to 0.575, showing the default tree was overfitting
badly. All non-default models sit clearly above the random-guess line on
the ROC curve.

## 2. Clustering

**Approach:** K-Means clustering at the **customer level** (340 customers),
using aggregated features: average transaction amount, total spend,
transaction count, average quantity, age, and number of unique categories
purchased. The number of clusters (*k*) was chosen using the elbow method
and silhouette score together; **k=2** gave the best silhouette score.

**Segments identified:**

| Cluster | Customers | Avg Transaction | Total Spend | Transactions | Unique Categories |
|---|---|---|---|---|---|
| 0 — Casual / single-category shoppers | 184 | $169.5 | $341.3 | 2.1 | 1.8 |
| 1 — High-value / diverse-category shoppers | 142 | $272.5 | $1,122.3 | 4.4 | 3.4 |

Customer age is similar across both clusters (~45-46), so **spend and
engagement — not age — drive the segmentation**. Cluster 1 is a natural
target for a loyalty/VIP program; Cluster 0 is a natural target for
re-engagement campaigns.

## 3. Association Rule Mining

**Approach:** Each customer's full set of purchased product categories was
treated as one "basket," and the **Apriori algorithm** (via `mlxtend`) was
applied with `min_support=0.05` to find frequent category combinations,
then `association_rules` (`min_threshold=1.0` on lift) to generate rules.

**Result:** 54 association rules were found. The strongest rules by lift
involve `Home & Kitchen`, `Beauty`, `Books`, and `Clothing` — e.g., customers
who buy from both `Beauty` and `Books` are notably more likely than average
to also buy `Home & Kitchen` (lift ≈ 1.62).

**Real-world applications:**
- **Cross-category bundling/promotions** for category pairs with lift > 1
- **Targeted marketing** — recommend a rule's consequent category to
  customers who already bought its antecedent
- **Store/catalog layout** — place or cross-link frequently co-purchased categories
- **Inventory planning** — anticipate demand shifts in associated categories

## Key Insights Across All Three Techniques

- **`ProductCategory` is the recurring driver of signal** across regression
  (Deliverable 2), classification, and association rule mining — a
  consistent finding across every technique applied to this dataset.
- **Model performance is modest but genuine everywhere**, which is expected
  given the amount of built-in within-category randomness in the synthetic
  dataset — the goal of this deliverable was correct methodology and honest
  interpretation, not inflated metrics.
- **Clustering revealed a spend/engagement split, not a demographic one** —
  useful for the business recommendations to come in Deliverable 4.
- **Association rules point to concrete, actionable bundling opportunities**
  that connect directly to the classification and clustering results (e.g.,
  Cluster 1's diverse-category shoppers are the customers most likely to be
  affected by these cross-category rules).

## Challenges Encountered

- **Choosing a classification target.** The dataset doesn't have a natural
  pre-existing binary label, so a median-split `HighValue` target was
  engineered from `TotalAmount` — the same care taken in Deliverable 2 to
  exclude `Quantity`/`UnitPrice` from the features was needed here to avoid
  trivially leaking the label.
- **Interpreting a tie-breaking silhouette result.** The silhouette score
  favored a simple k=2 segmentation over more granular options; rather than
  forcing a larger, less-separated cluster count, the notebook accepts and
  interprets the 2-cluster solution the data actually supports.
- **Building baskets from transaction-level data.** Because each row is a
  single product-category purchase, baskets had to be constructed by
  grouping all of a customer's purchases together (rather than using the
  raw rows directly) before Apriori could find meaningful multi-item
  patterns.

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn mlxtend jupyter
jupyter notebook MSCS_634_Deliverable3.ipynb
```
