# MSCS-634 Project Deliverable 1: Data Collection, Cleaning, and Exploration

**Author:** Saru Bhandari
**Course:** MSCS-634 – Advanced Big Data and Data Mining
**Deliverable:** 1 of 4 — Data Collection, Cleaning, and Exploration

## Repository Contents

- `retail_transactions_raw.csv` — the original, uncleaned dataset (1,038 rows, 14 columns)
- `retail_transactions_cleaned.csv` — the cleaned dataset produced by the notebook (1,000 rows, 13 columns)
- `MSCS_634_Deliverable1.ipynb` — full analysis notebook (loading, cleaning, EDA)
- `README.md` — this file

---

## 1. Dataset Selection and Justification

**Dataset:** Retail Transactions Dataset (`retail_transactions_raw.csv`)

**Description:** The dataset contains 1,038 individual retail purchase
transactions across 340 customers. Each row is one line-item transaction and
includes:
- **Customer demographics:** `CustomerID`, `CustomerAge`, `Gender`, `State`, `City`
- **Product details:** `ProductCategory`, `ProductName`
- **Purchase details:** `TransactionID`, `TransactionDate`, `Quantity`, `UnitPrice`, `TotalAmount`, `PaymentMethod`
- **Feedback (partially populated):** `CustomerSatisfaction`

**Why this dataset is appropriate for the project:**
- It has **14 attributes** and **1,038 records**, comfortably exceeding the
  assignment's 8-10 attribute / 500+ record minimum.
- It mixes **numeric** (age, quantity, price, total), **categorical**
  (gender, state, category, payment method), and **datetime** (transaction
  date) fields, which supports the regression, classification, and
  clustering work required in later deliverables.
- Because each transaction records a `CustomerID`, `ProductCategory`, and
  `ProductName`, the data can be grouped into **customer/date "baskets"** for
  market-basket / association rule mining (Apriori, FP-Growth) in
  Deliverable 3.
- It contains **realistic data-quality issues** — missing values, duplicate
  rows, inconsistently formatted state names, and a few extreme outliers —
  giving genuine cleaning work to perform and document, rather than starting
  from an already-clean dataset.

---

## 2. Loading the Dataset and Inspecting Its Structure

In the notebook, the dataset is loaded with `pandas.read_csv()` and its
structure is inspected using:
- `df.shape` — confirms 1,038 rows × 14 columns
- `df.info()` — data types and non-null counts per column
- `df.describe(include='all')` — summary statistics for numeric and
  categorical columns
- `df.isnull().sum()` / percentage — quantifies missingness per column
- `df.duplicated().sum()` — counts exact duplicate rows
- `df['State'].unique()` — surfaces the inconsistent state-name formatting

This inspection step is what determined the specific cleaning strategy
described below (which columns to impute vs. drop, which fields needed
standardizing).

---

## 3. Data Cleaning

### 3.1 Handling Missing Values

Strategy, chosen per column based on how much data is missing and what type
each column is:

| Column | % Missing | Strategy | Rationale |
|---|---|---|---|
| `CustomerSatisfaction` | ~68% | **Column dropped** | Imputing a rating for two-thirds of records would manufacture data that was never collected |
| `CustomerAge` | ~9% | **Imputed with median** | Numeric field; median is robust to skew/outliers |
| `Gender` | ~7% | **Imputed with mode** | Categorical field; filled with the most frequently occurring category |

### 3.2 Removing Duplicates or Correcting Inconsistent Data

- **Duplicates:** 38 exact duplicate rows (re-exported transactions) were
  identified with `df.duplicated().sum()` and removed with
  `df.drop_duplicates()`. Leaving them in would have over-weighted certain
  customers/products and biased any model trained on the data.
- **Inconsistent data:** The `State` column contained multiple
  representations of the same state (e.g., `NY`, `New York`, `ny`, `N.Y.`).
  These were standardized to two-letter USPS codes using an explicit mapping
  dictionary, so the column can be reliably used for grouping, filtering, or
  encoding downstream.

### 3.3 Identifying and Addressing Noisy Data

`Quantity` and `UnitPrice` contained a small number of extreme values
consistent with data-entry errors (e.g., an implausibly large quantity on a
single retail transaction, or a unit price far outside the normal range for
its product category). These were identified using the **IQR
(interquartile range) method**: values falling outside
`[Q1 − 1.5×IQR, Q3 + 1.5×IQR]` were flagged as outliers and **capped
(winsorized)** at those bounds rather than deleted, preserving the rest of
each transaction's data. `TotalAmount` was then recomputed from the
cleaned `Quantity × UnitPrice` so it stays internally consistent.

**Net effect:** 1,038 raw rows → 1,000 rows after cleaning (duplicates
removed); `CustomerSatisfaction` column dropped (14 → 13 columns); all
missing values resolved; state names standardized to 10 consistent codes;
outlier values capped.

---

## 4. Exploratory Data Analysis (EDA)

Conducted in the notebook using Matplotlib and Seaborn:

- **Distributions:** Histograms (with KDE) of `CustomerAge`, `TotalAmount`,
  and `Quantity` to examine shape, skew, and spread.
- **Outlier identification:** Boxplots of `UnitPrice` and `TotalAmount`
  (post-cleaning) to visually confirm the IQR capping worked as intended.
- **Categorical relationships:** Bar charts of transaction counts by
  `ProductCategory` and `PaymentMethod`; a boxplot of `TotalAmount` by
  `ProductCategory` to compare spending across categories.
- **Feature relationships:** A correlation heatmap across the numeric
  features (`CustomerAge`, `Quantity`, `UnitPrice`, `TotalAmount`), plus bar
  charts of transaction counts by state and average transaction amount by
  gender.

---

## 5. Insights from EDA and Implications for Future Modeling

- **Product category drives spend variation:** `TotalAmount` varies
  noticeably by `ProductCategory`, making category a strong candidate
  feature for regression modeling and a natural grouping variable for
  clustering customers by spend profile in later deliverables.
- **Age is a weak linear predictor:** `CustomerAge` shows little linear
  correlation with `Quantity`, `UnitPrice`, or `TotalAmount` in the
  correlation heatmap. This suggests age alone won't be a strong linear
  regression predictor, and that non-linear or interaction features
  (e.g., age × category) may be worth exploring in Deliverable 2.
- **Multicollinearity risk identified:** `Quantity` and `UnitPrice` jointly
  and mechanically determine `TotalAmount`. For regression modeling, we'll
  need to avoid using all three as independent features simultaneously to
  prevent multicollinearity / target leakage.
- **Class imbalance to account for:** `ProductCategory` and `PaymentMethod`
  distributions are uneven (some categories/methods are far more common than
  others). This is important context for classification model evaluation
  later — accuracy alone won't be a reliable metric if a target class is
  imbalanced, so metrics like F1 or a confusion matrix will matter more.
- **Transactional structure preserved for pattern mining:** Because
  `CustomerID`, `ProductCategory`, and `ProductName` remain intact after
  cleaning, the cleaned dataset can be grouped into per-customer/per-date
  "baskets" for association rule mining (Apriori/FP-Growth) in
  Deliverable 3.
- **Cleaning materially changed the data:** ~68% of rows lost their
  `CustomerSatisfaction` value (column dropped), ~9%/~7% of rows had
  `CustomerAge`/`Gender` imputed, 38 duplicate rows were removed, several
  inconsistent state spellings were consolidated into 10 standard codes, and
  a handful of extreme `Quantity`/`UnitPrice` outliers were capped — all of
  which reduces bias in any model trained on this data going forward.

---

## Challenges Encountered

- **Deciding a threshold for dropping vs. imputing a column.** With
  `CustomerSatisfaction` missing in two-thirds of rows, imputing values would
  have manufactured data rather than reflected real customer feedback, so
  the column was dropped rather than filled. A general rule of "drop columns
  missing above ~50%, impute below that" was applied consistently.
- **Choosing between removing vs. capping outliers.** Deleting the outlier
  rows would have also discarded legitimate transaction/customer data in
  other columns, so outlier values were capped (winsorized) at the IQR
  fences instead of removing the rows outright.
- **Recomputing a dependent column after cleaning.** Because `TotalAmount`
  is derived from `Quantity × UnitPrice`, capping those two columns required
  recomputing `TotalAmount` afterward to avoid leaving stale, inconsistent
  totals in the cleaned dataset.

---

## How to Run

```bash
pip install pandas numpy matplotlib seaborn jupyter
jupyter notebook MSCS_634_Deliverable1.ipynb
```
