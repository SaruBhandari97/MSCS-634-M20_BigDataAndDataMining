# MSCS_634_Lab_6 — Association Rule Mining with Apriori and FP-Growth

## Purpose

This lab applies **association rule mining** to a real-world transactional dataset in order to:

- Discover frequent itemsets using both the **Apriori** and **FP-Growth** algorithms.
- Generate association rules (support, confidence, lift) and interpret which product combinations are meaningfully associated.
- Compare the two algorithms' output and runtime efficiency on the same data and thresholds.
- Use Seaborn visualizations (bar plots, a co-occurrence heatmap, and a confidence-vs-lift scatter plot) to interpret the mined patterns.

## Dataset

The **Groceries Market Basket dataset** — 9,835 real supermarket transactions covering 169 unique items (average basket size ≈ 4.4 items). This is a widely used, publicly available transactional dataset (originally distributed with the R `arules` package) in the same spirit as the Instacart Market Basket dataset suggested in the assignment, and it satisfies the lab's core requirement of item-level transaction records suitable for Apriori/FP-Growth.

## Contents

- `MSCS_634_Lab_6.ipynb` — Jupyter notebook containing all code, visualizations, and analysis, organized into five steps: data preparation, Apriori mining, FP-Growth mining, association rule generation/analysis, and a final comparative analysis.
- `groceries.csv` — the raw transactional dataset used in the notebook (one basket per line, comma-separated items).
- `README.md` — this file.

## Approach

The raw file has no header and a variable number of items per line, so it's parsed into a list-of-lists transaction format and one-hot encoded with `mlxtend`'s `TransactionEncoder`. Both Apriori and FP-Growth are run at the same `min_support = 0.01` threshold so their outputs and runtimes are directly comparable, and association rules are then generated from the frequent itemsets at `min_confidence = 0.3`.

## Key Insights

- At `min_support = 0.01`, Apriori and FP-Growth discovered **the exact same 333 frequent itemsets** with identical support values — as expected, since both algorithms solve the same problem, just via different search strategies. Most itemsets were single items or pairs; few 3-item combinations cleared the 1% threshold.
- **FP-Growth ran roughly 2.5x faster than Apriori** (0.27s vs. 0.68s), consistent with FP-Growth's compressed FP-tree approach avoiding the repeated candidate generation and database scans that Apriori requires.
- Generated **125 association rules** at `min_confidence = 0.3`. The strongest rules by lift centered on a "produce + dairy" shopping pattern — e.g., `other vegetables, tropical fruit → root vegetables` and `citrus fruit, other vegetables → root vegetables` had the highest lift values (~3.1–3.3), meaning customers buying these items together do so well above what random chance would predict.
- Rules with `whole milk` as the consequent tended to have high confidence but only moderate lift, since `whole milk` is purchased so frequently on its own — a reminder that confidence alone can overstate the strength of an association for very popular items, and lift should be considered alongside it.

## Challenges and Decisions

- The originally suggested dataset sources (UCI, Kaggle) weren't directly reachable from the working environment's network, so the well-known Groceries Market Basket dataset was used as an equivalent, real, publicly available transactional dataset.
- The raw file is a ragged CSV (variable items per line, no header), so it required custom line-by-line parsing before one-hot encoding, rather than a direct `pd.read_csv`.
- An initial support threshold of 0.05 surfaced very few itemsets beyond single popular items, so it was lowered to 0.01 to capture a richer set of 2-item (and a few 3-item) associations while keeping the itemset count manageable.
- A confidence threshold of 0.3 was chosen to balance rule quantity against rule quality, filtering out very weak associations.
