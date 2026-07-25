# MSCS-634 Project Deliverable 2: Regression Modeling and Performance Evaluation

**Author:** Saru Bhandari
**Course:** MSCS-634 – Advanced Big Data and Data Mining
**Deliverable:** 2 of 4 — Regression Modeling and Performance Evaluation

## Repository Contents

- `MSCS_634_Deliverable2.ipynb` — full regression analysis notebook (feature engineering, model building, evaluation, cross-validation)
- `retail_transactions_cleaned.csv` — the cleaned dataset carried over from Deliverable 1
- `README.md` — this file

---

## Dataset Summary

This deliverable builds directly on the **cleaned retail transactions
dataset** produced in Deliverable 1 (1,000 rows, 13 columns) — retail
purchase transactions with customer demographics, product details, and
purchase context. The goal here is to predict **`TotalAmount`** (the dollar
value of a transaction) from customer, product-category, and purchase-context
features.

## Feature Engineering

`Quantity` and `UnitPrice` were deliberately **excluded** from the feature
set, since `TotalAmount = Quantity x UnitPrice` — including them would let a
model trivially reconstruct the target instead of learning genuine signal
(the multicollinearity/leakage risk flagged in Deliverable 1). In their
place, the following features were engineered:

- **Date-derived features:** `Month`, `DayOfWeek`, `IsWeekend`, extracted from `TransactionDate`
- **Customer purchase frequency:** `CustomerTransactionCount`, the number of transactions on record for that customer
- **One-hot encoded categoricals:** `ProductCategory`, `Gender`, `State`, `PaymentMethod`
- **`CustomerAge`**, retained as a numeric feature

## Modeling Process

- **Train/test split:** 80/20, with all features standardized (`StandardScaler`)
- **Models built (3, exceeding the "at least two" requirement):**
  1. Linear Regression (baseline OLS)
  2. Ridge Regression (L2 regularization, alpha=1.0)
  3. Lasso Regression (L1 regularization, alpha=0.5)
- **Evaluation metrics:** R-squared, MSE, RMSE, MAE on the held-out test set
- **Cross-validation:** 5-fold cross-validation (R2) to check generalization beyond a single train/test split

## Evaluation Results

| Model | Test R2 | Test RMSE | CV R2 Mean | CV R2 Std |
|---|---|---|---|---|
| Linear Regression | 0.130 | $169.84 | 0.153 | 0.043 |
| Ridge Regression | 0.130 | $169.78 | 0.153 | 0.043 |
| Lasso Regression | **0.138** | **$169.01** | **0.156** | 0.042 |

**Best model: Lasso Regression** — it edges out the other two on every
metric (highest R2, lowest RMSE, best cross-validation R2), though the
margin over Linear/Ridge is modest. Ridge and Linear Regression perform
almost identically, which is expected given the feature set here isn't
large or collinear enough for L2 regularization to make a large difference.

## Key Insights

- **R2 is modest (~0.13-0.14) but genuine.** The models explain roughly
  13-14% of the variance in `TotalAmount` using only category, demographic,
  and context features — with `Quantity`/`UnitPrice` deliberately excluded.
  This is real signal, not the near-perfect fit that would appear if the
  models could "cheat" using the target's own components.
- **`ProductCategory` is the dominant predictor.** This matches the strong
  category-to-price relationship found in Deliverable 1's EDA.
- **The unexplained variance is by design.** Within each `ProductCategory`,
  price is still drawn from a range and `Quantity` varies independently, so
  a meaningful share of transaction value comes from within-category
  variation the engineered features can't see — realistic behavior for
  retail data, where category signals a pricing tier but not an exact spend.
- **Regularization mainly trims weak features here, rather than fixing
  overfitting.** Lasso's small edge comes from zeroing out a handful of
  low-value one-hot encoded `State`/`PaymentMethod` dummies.
- **Implication for Deliverable 3:** the strength of `ProductCategory` as a
  predictor (and its relationship to `CustomerAge` found in Deliverable 1)
  makes it a strong candidate target/feature for the classification and
  clustering work ahead.

## Challenges Encountered

- **Avoiding target leakage.** The most direct predictors of `TotalAmount`
  (`Quantity`, `UnitPrice`) are mathematically part of the target itself.
  Excluding them was necessary to build a meaningful model, but it also
  meant accepting a lower R2 than a "leaky" model would report — the
  challenge was interpreting model quality on those terms rather than
  chasing a higher score.
- **Encoding a moderate-cardinality categorical (`State`).** One-hot
  encoding `State` (10 categories) alongside `ProductCategory` (7),
  `PaymentMethod` (5), and `Gender` (2) expanded the feature space
  considerably; comparing Ridge and Lasso helped clarify which of those
  dummy variables were actually contributing versus adding noise.
- **Distinguishing a "weak but real" model from a "broken" one.** With R2
  around 0.13, it was important to verify (via the excluded-feature check
  and the category-level EDA from Deliverable 1) that the models were
  correctly picking up real category-based pricing signal, rather than
  concluding the modeling approach had failed.

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook MSCS_634_Deliverable2.ipynb
```
