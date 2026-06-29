# MSCS-634 Lab 4: Regression Analysis and Regularization Techniques

**Name:** Saru Bhandari
**Course:** MSCS-634 — Advanced Big Data and Data Mining
**Lab:** Lab 4 — Regression Analysis and Regularization Techniques

## Purpose

This lab works through a progression of regression techniques on the scikit-learn **Diabetes
dataset** (442 samples, 10 health-measurement features, target = a quantitative measure of
disease progression one year after baseline). The goal is to understand how model complexity and
regularization affect performance, and to evaluate every model on a common footing using MAE, MSE,
RMSE, and R².

The notebook covers, in order:

1. **Data preparation** — loading the dataset, exploring feature/target distributions, checking
   for missing values, and reviewing the correlation structure.
2. **Simple linear regression** — a single-feature baseline using BMI, the strongest individual
   predictor.
3. **Multiple regression** — using all ten features together.
4. **Polynomial regression** — expanding into polynomial features and sweeping the degree to
   demonstrate underfitting vs. overfitting.
5. **Regularization** — Ridge (L2) and Lasso (L1) regression applied to the polynomial feature
   space, with an alpha sweep to show how penalty strength changes behavior.
6. **Comparison and analysis** — a consolidated metrics table, comparison charts, and discussion.

## Files

| File | Description |
|------|-------------|
| `MSCS_634_Lab_4.ipynb` | Fully executed Jupyter Notebook with code, outputs, and visualizations. |
| `README.md` | This file. |

## How to Run

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
jupyter notebook MSCS_634_Lab_4.ipynb
```

Run the cells top to bottom. A fixed `random_state = 42` is used throughout, so every split,
model, and metric is reproducible.

## Key Insights

- **More features beat a single strong one.** BMI alone explained only about 23% of the variance
  (R² ≈ 0.23). Using all ten features in a multiple regression roughly doubled that to R² ≈ 0.45
  and was the single biggest performance gain in the lab — disease progression genuinely depends
  on several factors acting together.
- **Polynomial expansion overfits fast on a small dataset.** A degree-2 model stayed competitive,
  but from degree 3 onward training R² climbed toward 1.0 while test R² collapsed and went sharply
  negative (e.g., test R² ≈ −14 at degree 3, worsening from there). With only 442 samples, the
  expanded feature space let the model memorize noise — a textbook overfitting curve.
- **Regularization fixes the overfitting.** Applied to the same degree-2 features, Ridge and Lasso
  both restored solid positive test performance. Lasso (α = 0.1) reached the best fixed-alpha
  result (R² ≈ 0.48) while driving 11 of the 65 polynomial coefficients to exactly zero —
  automatic feature selection. After tuning alpha, both Ridge and Lasso pushed test R² above 0.51.
- **Ridge vs. Lasso behave differently.** Ridge shrinks all coefficients smoothly and keeps every
  feature; Lasso can zero coefficients out entirely, producing a sparser, more interpretable model.
- **The dataset has a real ceiling.** Even the best models top out around R² ≈ 0.45–0.51, a
  reminder that a meaningful share of the variation in disease progression simply isn't captured
  by these ten measurements.

## Challenges and Decisions

- **Feature choice for simple regression.** I used the correlation matrix to pick BMI (highest
  absolute correlation with the target) as the single predictor, so the baseline was as strong as
  a one-feature model can reasonably be.
- **Scaling inside pipelines.** Polynomial and regularized models are sensitive to feature scale
  once squared and interaction terms are introduced. I wrapped `StandardScaler`,
  `PolynomialFeatures`, and the estimator in a single `Pipeline` so scaling is fit only on training
  data and applied consistently, avoiding data leakage.
- **Demonstrating overfitting deliberately.** Rather than just stating that high-degree polynomials
  overfit, I swept degrees 1–6 and plotted train vs. test R² so the divergence is visible in the
  output.
- **Lasso convergence.** Lasso needed a higher `max_iter` (10,000) to converge cleanly on the
  larger polynomial feature set; the default raised convergence warnings.
- **Consistent evaluation.** A single `evaluate()` helper computes all four metrics for every model,
  keeping the comparison table fair and reducing copy-paste errors.
