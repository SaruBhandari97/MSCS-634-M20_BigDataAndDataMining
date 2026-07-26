# MSCS-634 Project Deliverable 4: Final Insights, Recommendations, and Presentation

**Author:** Saru Bhandari  
**Course:** MSCS-634 – Advanced Big Data and Data Mining  
**Deliverable:** 4 of 4 — Final Insights, Recommendations, and Presentation

## Repository Contents

- `MSCS_634_Deliverable4.ipynb` — consolidated notebook with all code from Deliverables 1-3, plus a final synthesis section
- `MSCS634_Deliverable4_Report.docx` — comprehensive written report (title page, introduction, data prep, modeling, evaluation, insights, ethics, recommendations, references)
- `MSCS634_Deliverable4_Presentation.pptx` — presentation slides for the video walkthrough, with speaker notes embedded on every slide
- `PRESENTATION_SCRIPT.md` — standalone narration script (same content as the speaker notes, for easy reading while recording)
- `retail_transactions_cleaned.csv` — the cleaned dataset used throughout the project
- `README.md` — this file

## Dataset Summary

**Retail Transactions Dataset** — 1,038 raw / 1,000 cleaned retail purchase
transactions across 340 customers, with 14 attributes spanning customer
demographics (age, gender, state, city), product details (category, product
name), and purchase context (date, quantity, price, total, payment method).
The dataset was chosen for its size (exceeding the 500-record/8-10-attribute
minimum), its mix of numeric/categorical/datetime fields, and its
transactional structure, which supports market-basket analysis.

## Project Steps

1. **Deliverable 1 — Data Cleaning & EDA:** resolved missing values (column
   drop for `CustomerSatisfaction`, median/mode imputation elsewhere),
   removed 38 duplicate rows, standardized inconsistent state names, and
   capped outliers using the IQR method. EDA revealed a strong
   `ProductCategory` <-> price relationship and an age-to-category
   preference pattern.
2. **Deliverable 2 — Regression:** built Linear, Ridge, and Lasso Regression
   to predict `TotalAmount` (excluding `Quantity`/`UnitPrice` to avoid
   leakage). Lasso performed best (R2 = 0.138, RMSE = $169.01), evaluated
   with 5-fold cross-validation.
3. **Deliverable 3 — Classification, Clustering, Pattern Mining:** built
   Decision Tree, k-NN, and Naive Bayes to classify high- vs. low-value
   transactions (Naive Bayes best: F1 = 0.640, AUC = 0.678), with
   hyperparameter tuning via `GridSearchCV` on the Decision Tree. K-Means
   clustering (k=2, chosen via silhouette score) segmented customers into
   "casual" and "high-value" groups. Apriori association rule mining on
   customer purchase baskets found 54 rules.
4. **Deliverable 4 — Synthesis:** consolidated all code, wrote a
   comprehensive report, and built a presentation tying the findings
   together into practical recommendations and ethical considerations.

## Major Findings

- **`ProductCategory` is the consistent driver of signal** across EDA,
  regression, classification, and association rule mining — the single
  most important finding of the project.
- Model performance is **modest but genuine** throughout (R2 ~0.13-0.14,
  classification accuracy ~0.58-0.60), a deliberate result of excluding
  features that would otherwise leak the prediction targets.
- Customer segmentation is driven by **spend and engagement, not
  demographics** — the two K-Means clusters have nearly identical average
  age but very different spending patterns.
- The strongest association rules cluster around `Home & Kitchen`,
  `Beauty`, `Books`, and `Clothing`, pointing to concrete cross-category
  bundling opportunities.

## Practical Recommendations

1. Prioritize category-level pricing and merchandising strategy
2. Launch a two-tier loyalty program aligned with the customer segments found
3. Use the classification model to flag likely high-value transactions
4. Act on the strongest association rules with cross-category bundling
5. Treat all model outputs as directional, not deterministic

## Ethical Considerations

The dataset is fully synthetic, so this project carries no real-world
privacy risk. The written report (Section 7) discusses, in detail: data
privacy considerations for a real deployment, fairness/bias risks from
using demographic features, the disclosed limitation that the synthetic
data's age-category relationship was built in by design (not an organic
finding), and profiling risk from applying association rules to
individual customers. Concrete steps taken in this project — using
synthetic data, excluding leakage-prone features, and reporting honest
(modest) metrics — are also detailed there.


## How to Run the Notebook

```bash
pip install pandas numpy matplotlib seaborn scikit-learn mlxtend jupyter
jupyter notebook MSCS_634_Deliverable4.ipynb
```
